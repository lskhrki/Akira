import os
import csv
from typing import List, Dict, Any

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

CSV_PATH = os.environ.get("CATALOG_CSV_PATH", "sample_products.csv")
PERSIST_DIR = os.environ.get("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = "thrift_catalog"

_embeddings = None
_vectorstore = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return _embeddings


def _load_catalog_rows(csv_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_vectorstore(csv_path: str = CSV_PATH, persist_dir: str = PERSIST_DIR) -> Chroma:
    """
    Reads the product catalog CSV, embeds each listing, and stores it
    in a persisted Chroma vector store. Run this once (or whenever the
    catalog changes) -- app.py calls it automatically on first startup
    if no store exists yet.
    """
    rows = _load_catalog_rows(csv_path)

    documents = []
    for row in rows:
        text = f"{row.get('title', '')}. {row.get('description', '')}. Brand: {row.get('brand', '')}. Category: {row.get('category', '')}."
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "title": row.get("title", ""),
                    "url": row.get("url", ""),
                    "price": row.get("price", ""),
                    "category": row.get("category", ""),
                    "brand": row.get("brand", ""),
                },
            )
        )

    store = Chroma.from_documents(
        documents=documents,
        embedding=_get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    return store


def load_vectorstore(persist_dir: str = PERSIST_DIR) -> Chroma:
    """Loads an existing persisted Chroma store without re-embedding."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_get_embeddings(),
        persist_directory=persist_dir,
    )


def get_vectorstore() -> Chroma:
    """
    Returns a ready-to-query vector store, building it from the CSV
    the first time this is called if it doesn't exist on disk yet.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        _vectorstore = load_vectorstore()
    else:
        _vectorstore = build_vectorstore()

    return _vectorstore


def retrieve_from_catalog(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Semantic similarity search over the product catalog.
    Returns results in the same shape as the Tavily web results
    so they can be merged directly in the LangGraph state.
    """
    store = get_vectorstore()
    docs_and_scores = store.similarity_search_with_relevance_scores(query, k=k)

    results = []
    for doc, score in docs_and_scores:
        results.append(
            {
                "title": doc.metadata.get("title", ""),
                "url": doc.metadata.get("url", ""),
                "description": doc.page_content,
                "score": score,
                "source": "catalog",
            }
        )
    return results


if __name__ == "__main__":
    # Run this file directly to (re)build the vector store from the CSV:
    #   python rag.py
    print(f"Building vector store from {CSV_PATH} into {PERSIST_DIR} ...")
    build_vectorstore()
    print("Done.")
