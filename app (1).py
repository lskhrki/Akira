import os
import json
from typing import TypedDict, List, Dict, Any

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


# --------------------------------------------------
# 1. STATE
# --------------------------------------------------

class ThriftState(TypedDict, total=False):
    user_query: str
    parsed_query: Dict[str, Any]
    search_results: List[Dict[str, Any]]
    filtered_results: List[Dict[str, Any]]
    ranked_results: List[Dict[str, Any]]
    final_results: str


# --------------------------------------------------
# 2. MODELS
# --------------------------------------------------
# NOTE: set OPENAI_API_KEY and TAVILY_API_KEY as environment
# variables on Render (see README for steps). Double check the
# model name below is one your OpenAI account actually has access
# to -- swap it for "gpt-4o" or similar if "gpt-5.4" isn't valid
# for your account.

model = ChatOpenAI(
    model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
    temperature=0,
)


# --------------------------------------------------
# 3. SEARCH TOOL
# --------------------------------------------------

tavily = TavilySearch(
    max_results=8,
    topic="general",
    search_depth="basic",
)


# --------------------------------------------------
# 4. QUERY UNDERSTANDING NODE
# --------------------------------------------------

def understand_query(state: ThriftState):
    query = state["user_query"]

    prompt = f"""
You are a fashion search assistant.

Analyze this user's thrift-fashion search:

"{query}"

Extract the following information:

- category
- brand
- style
- color
- size
- gender
- material
- maximum_price
- minimum_price
- keywords

Return ONLY valid JSON.

If something is not specified, use null.

Example:

{{
    "category": "shoes",
    "brand": "Nike",
    "style": "Dunk",
    "color": null,
    "size": null,
    "gender": null,
    "material": null,
    "maximum_price": null,
    "minimum_price": null,
    "keywords": ["Nike Dunks"]
}}
"""

    response = model.invoke(prompt)

    try:
        parsed = json.loads(response.content)
    except Exception:
        parsed = {
            "category": None,
            "brand": None,
            "style": None,
            "color": None,
            "size": None,
            "gender": None,
            "material": None,
            "maximum_price": None,
            "minimum_price": None,
            "keywords": [query],
        }

    return {"parsed_query": parsed}


# --------------------------------------------------
# 5. PRODUCT SEARCH NODE
# --------------------------------------------------

def search_products(state: ThriftState):
    parsed = state["parsed_query"]
    keywords = parsed.get("keywords", [])
    search_query = " ".join(keywords)

    if parsed.get("brand"):
        search_query += f" {parsed['brand']}"

    if parsed.get("style"):
        search_query += f" {parsed['style']}"

    search_query += " thrift second hand vintage fashion"

    results = tavily.invoke({"query": search_query})

    products = []

    if isinstance(results, dict):
        raw_results = results.get("results", [])
    else:
        raw_results = results

    for item in raw_results:
        if isinstance(item, dict):
            products.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("content", ""),
                    "score": item.get("score", 0),
                }
            )

    return {"search_results": products}


# --------------------------------------------------
# 6. FILTER NODE
# --------------------------------------------------

def filter_products(state: ThriftState):
    products = state.get("search_results", [])
    parsed = state.get("parsed_query", {})

    filtered = []

    brand = parsed.get("brand")
    category = parsed.get("category")

    for product in products:
        text = (product.get("title", "") + " " + product.get("description", "")).lower()
        relevant = True

        if brand:
            if brand.lower() not in text:
                relevant = False

        if category:
            category_keywords = {
                "shoes": ["shoe", "sneaker", "dunk", "footwear"],
                "shirt": ["shirt", "oxford", "tee", "t-shirt"],
                "jeans": ["jeans", "denim"],
                "jacket": ["jacket", "coat"],
            }

            keywords = category_keywords.get(category.lower(), [])

            if keywords:
                if not any(keyword in text for keyword in keywords):
                    relevant = False

        if relevant:
            filtered.append(product)

    return {"filtered_results": filtered}


# --------------------------------------------------
# 7. RANKING NODE
# --------------------------------------------------

def rank_products(state: ThriftState):
    products = state.get("filtered_results", [])
    ranked = sorted(products, key=lambda x: x.get("score", 0), reverse=True)
    return {"ranked_results": ranked[:6]}


# --------------------------------------------------
# 8. RESULT GENERATION NODE
# --------------------------------------------------

def generate_results(state: ThriftState):
    query = state["user_query"]
    products = state.get("ranked_results", [])

    if not products:
        return {
            "final_results": (
                "I couldn't find relevant thrift items "
                f"for '{query}'. Try using different keywords."
            )
        }

    product_text = ""
    for index, product in enumerate(products, start=1):
        product_text += f"""
Product {index}

Title:
{product.get('title')}

Description:
{product.get('description')}

URL:
{product.get('url')}

"""

    prompt = f"""
You are the final recommendation assistant for
a thrift-fashion discovery website.

User searched for:

"{query}"

Here are the products found:

{product_text}

Create a concise recommendation list.

For every product include:

1. Product name
2. Short description
3. Why it matches the user's search
4. Link

Do not invent prices, sizes, brands, or product information
that is not present in the search results.
"""

    response = model.invoke(prompt)
    return {"final_results": response.content}


# --------------------------------------------------
# 9. BUILD LANGGRAPH
# --------------------------------------------------

builder = StateGraph(ThriftState)

builder.add_node("understand_query", understand_query)
builder.add_node("search_products", search_products)
builder.add_node("filter_products", filter_products)
builder.add_node("rank_products", rank_products)
builder.add_node("generate_results", generate_results)

builder.add_edge(START, "understand_query")
builder.add_edge("understand_query", "search_products")
builder.add_edge("search_products", "filter_products")
builder.add_edge("filter_products", "rank_products")
builder.add_edge("rank_products", "generate_results")
builder.add_edge("generate_results", END)

graph = builder.compile()


# --------------------------------------------------
# 10. WEB APP (FastAPI) -- this is what Render runs
# --------------------------------------------------

app = FastAPI(title="Thrift Grail Finder")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Thrift Grail Finder</title>
    <style>
        body {{
            font-family: system-ui, sans-serif;
            max-width: 640px;
            margin: 60px auto;
            padding: 0 16px;
            color: #222;
        }}
        h1 {{ font-size: 1.6rem; }}
        form {{ display: flex; gap: 8px; margin: 24px 0; }}
        input[type=text] {{
            flex: 1;
            padding: 10px 12px;
            font-size: 1rem;
            border: 1px solid #ccc;
            border-radius: 6px;
        }}
        button {{
            padding: 10px 18px;
            font-size: 1rem;
            border: none;
            border-radius: 6px;
            background: #222;
            color: white;
            cursor: pointer;
        }}
        pre {{
            white-space: pre-wrap;
            background: #f7f7f7;
            padding: 16px;
            border-radius: 8px;
            line-height: 1.5;
        }}
    </style>
</head>
<body>
    <h1>🧥 Thrift Grail Finder</h1>
    <form action="/search" method="post">
        <input type="text" name="query" placeholder="Example: nike dunks, oversized oxford shirt" value="{query}" required />
        <button type="submit">Search</button>
    </form>
    {results}
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE_TEMPLATE.format(query="", results="")


@app.post("/search", response_class=HTMLResponse)
def search(query: str = Form(...)):
    result = graph.invoke({"user_query": query})
    final_text = result.get("final_results", "No results.")
    results_html = f"<h2>Results</h2><pre>{final_text}</pre>"
    return PAGE_TEMPLATE.format(query=query, results=results_html)


@app.post("/api/search")
def api_search(query: str = Form(...)):
    """JSON endpoint, useful if you build a separate JS frontend later."""
    result = graph.invoke({"user_query": query})
    return {"query": query, "final_results": result.get("final_results", "")}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
