"""
The Thrift & Subculture Grail Finder.

Three files, no Docker: app.py, requirements.txt, index.html (must sit next
to app.py — it's served at "/" by reading it straight off disk).

Pipeline: search Indian fashion/retail platforms (Playwright) -> score
candidates with Google Gemini vision -> alert on Telegram with an offer
button. The web frontend hits the faster /compare endpoint (scrape only,
no Gemini) for a plain price/availability comparison across platforms.

Platforms: Myntra, Ajio, Meesho, Amazon.in, Flipkart. All prices are INR.

NOTE ON RELIABILITY: Amazon.in and Flipkart run aggressive bot-detection
against datacenter IPs (which is what a Render server is) — expect those
two to intermittently return 0 results or get CAPTCHA'd even when the
scraper code itself is correct. Myntra/Ajio/Meesho are generally more
permissive but their DOM selectors will drift over time and need upkeep —
check scrape_debug in the API response when something returns 0 results.

Run modes:
    # One-off CLI search
    python app.py --prompt "Oversized Y2K oxford shirt" --tags y2k --max-price 1500

    # Telegram offer-button listener only (no HTTP server)
    python app.py --bot

    # Web service mode (what Render's "Web Service" needs — binds to $PORT,
    # serves index.html at "/", exposes /compare and /search, and runs the
    # Telegram bot polling loop in a background thread)
    python app.py --web

Env vars (put these in a .env file, see bottom of this file for the list):
    GOOGLE_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    MATCH_SCORE_THRESHOLD, MAX_PRICE_INR, ENABLE_AUTO_OFFERS, HEADLESS, PORT,
    PLAYWRIGHT_BROWSERS_PATH (set to "0" on Render — see deploy notes below)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, TypedDict

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field, HttpUrl
from langgraph.graph import END, StateGraph
from playwright.async_api import async_playwright
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CallbackQueryHandler, ContextTypes
import google.generativeai as genai
from flask import Flask, jsonify, request

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("grail_finder")


# ============================== CONFIG ==================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
MATCH_SCORE_THRESHOLD = int(os.getenv("MATCH_SCORE_THRESHOLD", "85"))
MAX_PRICE_INR = float(os.getenv("MAX_PRICE_INR", "3000"))
ENABLE_AUTO_OFFERS = os.getenv("ENABLE_AUTO_OFFERS", "false").lower() in {"1", "true", "yes"}
HEADLESS = os.getenv("HEADLESS", "true").lower() in {"1", "true", "yes"}
REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "2.0"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


# ============================== MODELS ===================================

class Platform(str, Enum):
    MYNTRA = "myntra"
    AJIO = "ajio"
    MEESHO = "meesho"
    AMAZON = "amazon"
    FLIPKART = "flipkart"


class SearchQuery(BaseModel):
    prompt: str
    style_tags: list[str] = Field(default_factory=list)
    max_price: Optional[float] = Field(default=None, description="INR")
    min_price: Optional[float] = Field(default=None, description="INR")
    sizes: list[str] = Field(default_factory=list)
    platforms: list[Platform] = Field(default_factory=lambda: list(Platform))

    def search_terms(self) -> str:
        parts = [self.prompt] + self.style_tags
        return " ".join(dict.fromkeys(p.strip() for p in parts if p.strip()))


class Listing(BaseModel):
    platform: Platform
    listing_id: str
    url: HttpUrl
    title: str
    description: str = ""
    price: float = Field(..., description="Price in INR")
    currency: str = "INR"
    size: Optional[str] = None
    brand: Optional[str] = None
    photo_urls: list[HttpUrl] = Field(default_factory=list)
    scraped_at: datetime = Field(default_factory=datetime.utcnow)

    def unique_key(self) -> str:
        return f"{self.platform.value}:{self.listing_id}"


class EvaluationResult(BaseModel):
    listing_key: str
    match_score: int = Field(..., ge=0, le=100)
    authenticity_confidence: int = Field(..., ge=0, le=100)
    condition_score: int = Field(..., ge=0, le=100)
    flagged_defects: list[str] = Field(default_factory=list)
    counterfeit_signals: list[str] = Field(default_factory=list)
    summary: str
    recommended_offer: Optional[float] = Field(default=None, description="INR")

    def is_grail(self, threshold: int) -> bool:
        return self.match_score >= threshold and self.authenticity_confidence >= 60


class MatchAlert(BaseModel):
    listing: Listing
    evaluation: EvaluationResult

    def render_message(self) -> str:
        d, e = self.listing, self.evaluation
        lines = [
            f"🧥 *Grail Alert* — {e.match_score}% match",
            f"*{d.title}*",
            f"💵 ₹{d.price:.0f} | 📏 {d.size or 'n/a'} | 🏷 {d.brand or 'n/a'}",
            f"🏪 {d.platform.value} | 🔍 authenticity {e.authenticity_confidence}%",
        ]
        if e.flagged_defects:
            lines.append("⚠️ " + "; ".join(e.flagged_defects))
        lines.append(f"_{e.summary}_")
        lines.append(str(d.url))
        return "\n".join(lines)


class GraphState(TypedDict, total=False):
    query: SearchQuery
    listings: list[Listing]
    evaluations: list[EvaluationResult]
    alerts: list[MatchAlert]
    scrape_debug: dict
    warnings: list[str]


# ============================== SCRAPERS =================================
# NOTE: DOM selectors are best-effort as of this writing and WILL need
# upkeep — these sites change their frontends often, and several (notably
# Amazon/Flipkart) actively try to detect and block headless browsers.
# Check scrape_debug in the API response when a platform returns 0 results.

def _parse_price(text: str) -> Optional[float]:
    """Parses Indian price formats like '₹1,299', 'Rs. 999', '₹ 2,50,000'."""
    cleaned = text.replace("₹", "").replace("Rs.", "").replace("Rs", "").replace(",", "").strip()
    match = re.search(r"\d+\.?\d*", cleaned)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


_SIZE_PATTERNS = [
    re.compile(r"\bsize\s*[:\-]?\s*([a-zA-Z0-9\.]{1,4})\b", re.IGNORECASE),
    re.compile(r"\b(xxs|xs|s|m|l|xl|xxl|xxxl)\b", re.IGNORECASE),
    re.compile(r"\b(uk|eu|us)\s?(\d{1,2}(?:\.\d)?)\b", re.IGNORECASE),
]


def _guess_size_from_text(text: str) -> Optional[str]:
    """Best-effort size extraction from a listing title — most of these
    search-result cards don't expose structured size data."""
    for pattern in _SIZE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(match.lastindex).upper()
    return None


async def _new_page(playwright):
    browser = await playwright.chromium.launch(headless=HEADLESS)
    context = await browser.new_context(
        user_agent=USER_AGENT,
        viewport={"width": 1366, "height": 900},
        locale="en-IN",
    )
    page = await context.new_page()
    page.set_default_timeout(15000)  # fail fast instead of hanging on a blocked/slow site
    return browser, context, page


async def scrape_myntra(query: SearchQuery, max_results: int = 15) -> tuple[list[Listing], Optional[str]]:
    listings: list[Listing] = []
    try:
        async with async_playwright() as pw:
            browser, context, page = await _new_page(pw)
            search_slug = query.search_terms().strip().replace(" ", "-")
            url = f"https://www.myntra.com/{search_slug}"
            await page.goto(url, wait_until="domcontentloaded")
            await asyncio.sleep(REQUEST_DELAY_SECONDS)
            cards = await page.locator("li.product-base").all()
            for card in cards[:max_results]:
                try:
                    brand = (await card.locator(".product-brand").inner_text()).strip()
                    name = (await card.locator(".product-product").inner_text()).strip()
                    price_text = (await card.locator(".product-discountedPrice, .product-price").first.inner_text()).strip()
                    price = _parse_price(price_text)
                    href = await card.locator("a").first.get_attribute("href")
                    img = await card.locator("img").first.get_attribute("src")
                    if price is None or not href:
                        continue
                    full_url = href if href.startswith("http") else f"https://www.myntra.com/{href.lstrip('/')}"
                    listings.append(Listing(
                        platform=Platform.MYNTRA,
                        listing_id=full_url.rstrip("/").split("/")[-1],
                        url=full_url, title=f"{brand} {name}".strip(), brand=brand or None,
                        price=price, photo_urls=[img] if img else [],
                    ))
                except Exception:
                    continue
            await context.close()
            await browser.close()
    except Exception as exc:
        logger.exception("Myntra scrape failed")
        return listings, f"{exc.__class__.__name__}: {exc}"
    return listings, None


async def scrape_ajio(query: SearchQuery, max_results: int = 15) -> tuple[list[Listing], Optional[str]]:
    listings: list[Listing] = []
    try:
        async with async_playwright() as pw:
            browser, context, page = await _new_page(pw)
            url = f"https://www.ajio.com/search/?text={query.search_terms().replace(' ', '%20')}"
            await page.goto(url, wait_until="domcontentloaded")
            await asyncio.sleep(REQUEST_DELAY_SECONDS)
            cards = await page.locator(".item, .rilrtl-products-list__item").all()
            for card in cards[:max_results]:
                try:
                    brand = await _safe_text(card, ".brand")
                    name = await _safe_text(card, ".nameCls")
                    price_text = await _safe_text(card, ".price strong, .price")
                    price = _parse_price(price_text) if price_text else None
                    href = await card.locator("a").first.get_attribute("href")
                    img = await card.locator("img").first.get_attribute("src")
                    if price is None or not href:
                        continue
                    full_url = href if href.startswith("http") else f"https://www.ajio.com{href}"
                    listings.append(Listing(
                        platform=Platform.AJIO,
                        listing_id=full_url.rstrip("/").split("/")[-1],
                        url=full_url, title=f"{brand or ''} {name or query.search_terms()}".strip(),
                        brand=brand, price=price, photo_urls=[img] if img else [],
                    ))
                except Exception:
                    continue
            await context.close()
            await browser.close()
    except Exception as exc:
        logger.exception("Ajio scrape failed")
        return listings, f"{exc.__class__.__name__}: {exc}"
    return listings, None


async def scrape_meesho(query: SearchQuery, max_results: int = 15) -> tuple[list[Listing], Optional[str]]:
    listings: list[Listing] = []
    try:
        async with async_playwright() as pw:
            browser, context, page = await _new_page(pw)
            url = f"https://www.meesho.com/search?q={query.search_terms().replace(' ', '%20')}"
            await page.goto(url, wait_until="domcontentloaded")
            await asyncio.sleep(REQUEST_DELAY_SECONDS + 1)  # Meesho is a heavier client-side app
            cards = await page.locator("[class*='ProductList'] a, a[href*='/product/']").all()
            for card in cards[:max_results]:
                try:
                    href = await card.get_attribute("href")
                    if not href:
                        continue
                    full_url = href if href.startswith("http") else f"https://www.meesho.com{href}"
                    text_blob = (await card.inner_text()).strip()
                    price = _parse_price(text_blob)
                    img = await card.locator("img").first.get_attribute("src")
                    if price is None:
                        continue
                    title_line = text_blob.split("\n")[0][:120] if text_blob else query.search_terms()
                    listings.append(Listing(
                        platform=Platform.MEESHO,
                        listing_id=full_url.rstrip("/").split("/")[-1],
                        url=full_url, title=title_line, price=price,
                        photo_urls=[img] if img else [],
                    ))
                except Exception:
                    continue
            await context.close()
            await browser.close()
    except Exception as exc:
        logger.exception("Meesho scrape failed")
        return listings, f"{exc.__class__.__name__}: {exc}"
    return listings, None


async def scrape_amazon(query: SearchQuery, max_results: int = 15) -> tuple[list[Listing], Optional[str]]:
    listings: list[Listing] = []
    try:
        async with async_playwright() as pw:
            browser, context, page = await _new_page(pw)
            url = f"https://www.amazon.in/s?k={query.search_terms().replace(' ', '+')}"
            await page.goto(url, wait_until="domcontentloaded")
            await asyncio.sleep(REQUEST_DELAY_SECONDS)
            cards = await page.locator("div[data-component-type='s-search-result']").all()
            for card in cards[:max_results]:
                try:
                    title = await _safe_text(card, "h2 a span, h2 span")
                    price_whole = await _safe_text(card, ".a-price-whole")
                    price = _parse_price(price_whole) if price_whole else None
                    href = await card.locator("h2 a").first.get_attribute("href")
                    img = await card.locator("img.s-image").first.get_attribute("src")
                    if price is None or not href or not title:
                        continue
                    full_url = href if href.startswith("http") else f"https://www.amazon.in{href}"
                    asin_match = re.search(r"/dp/([A-Z0-9]{10})", full_url)
                    listing_id = asin_match.group(1) if asin_match else full_url.rstrip("/").split("/")[-1]
                    listings.append(Listing(
                        platform=Platform.AMAZON,
                        listing_id=listing_id,
                        url=full_url, title=title, price=price,
                        photo_urls=[img] if img else [],
                    ))
                except Exception:
                    continue
            await context.close()
            await browser.close()
    except Exception as exc:
        logger.exception("Amazon scrape failed (likely bot-blocked — Amazon is aggressive against datacenter IPs)")
        return listings, f"{exc.__class__.__name__}: {exc}"
    return listings, None


async def scrape_flipkart(query: SearchQuery, max_results: int = 15) -> tuple[list[Listing], Optional[str]]:
    listings: list[Listing] = []
    try:
        async with async_playwright() as pw:
            browser, context, page = await _new_page(pw)
            url = f"https://www.flipkart.com/search?q={query.search_terms().replace(' ', '%20')}"
            await page.goto(url, wait_until="domcontentloaded")
            await asyncio.sleep(REQUEST_DELAY_SECONDS)
            # Flipkart shows a login modal on most page loads; dismiss it if present.
            try:
                await page.locator("button:has-text('✕')").first.click(timeout=2000)
            except Exception:
                pass
            cards = await page.locator("div._1AtVbE, div._4rR01T, a.s1Q9rs").all()
            for card in cards[:max_results]:
                try:
                    title = await _safe_text(card, "._4rR01T, .s1Q9rs, a") or (await card.get_attribute("title"))
                    price_text = await _safe_text(card, "._30jeq3")
                    price = _parse_price(price_text) if price_text else None
                    href = await card.locator("a").first.get_attribute("href") if await card.locator("a").count() else await card.get_attribute("href")
                    img = await card.locator("img").first.get_attribute("src") if await card.locator("img").count() else None
                    if price is None or not href or not title:
                        continue
                    full_url = href if href.startswith("http") else f"https://www.flipkart.com{href}"
                    listings.append(Listing(
                        platform=Platform.FLIPKART,
                        listing_id=full_url.split("pid=")[-1].split("&")[0] if "pid=" in full_url else full_url.rstrip("/").split("/")[-1],
                        url=full_url, title=title.strip(), price=price,
                        photo_urls=[img] if img else [],
                    ))
                except Exception:
                    continue
            await context.close()
            await browser.close()
    except Exception as exc:
        logger.exception("Flipkart scrape failed")
        return listings, f"{exc.__class__.__name__}: {exc}"
    return listings, None


async def _safe_text(locator, selector: str) -> Optional[str]:
    try:
        el = locator.locator(selector).first
        if await el.count() == 0:
            return None
        return (await el.inner_text()).strip()
    except Exception:
        return None


SCRAPERS = {
    Platform.MYNTRA: scrape_myntra,
    Platform.AJIO: scrape_ajio,
    Platform.MEESHO: scrape_meesho,
    Platform.AMAZON: scrape_amazon,
    Platform.FLIPKART: scrape_flipkart,
}


def _apply_size_guess(listing: Listing) -> Listing:
    if not listing.size:
        listing.size = _guess_size_from_text(listing.title)
    return listing


async def compare_platforms(query: SearchQuery) -> tuple[dict[str, list[Listing]], dict]:
    """Fast path: scrape only, no vision evaluation. Returns listings grouped
    by platform (for a price/availability comparison view) plus per-platform
    scrape/error diagnostics."""
    platforms = query.platforms or list(SCRAPERS.keys())
    results = await asyncio.gather(*(SCRAPERS[p](query) for p in platforms))

    grouped: dict[str, list[Listing]] = {}
    debug_info: dict = {}
    for platform, (group, error) in zip(platforms, results):
        filtered = []
        for listing in group:
            if query.max_price and listing.price > query.max_price:
                continue
            if query.min_price and listing.price < query.min_price:
                continue
            filtered.append(_apply_size_guess(listing))
        grouped[platform.value] = filtered
        debug_info[platform.value] = {"scraped": len(group), "kept_after_filters": len(filtered), "error": error}

    return grouped, debug_info


async def gather_candidates(query: SearchQuery) -> tuple[list[Listing], dict]:
    """Returns (merged_listings, debug_info) — same scraping as
    compare_platforms but flattened + deduped for the vision-evaluation
    pipeline in /search."""
    platforms = query.platforms or list(SCRAPERS.keys())
    results = await asyncio.gather(*(SCRAPERS[p](query) for p in platforms))

    debug_info: dict = {}
    seen, merged = set(), []
    for platform, (group, error) in zip(platforms, results):
        raw_count = len(group)
        kept_count = 0
        for listing in group:
            key = listing.unique_key()
            if key in seen:
                continue
            if query.max_price and listing.price > query.max_price:
                continue
            if query.min_price and listing.price < query.min_price:
                continue
            seen.add(key)
            merged.append(_apply_size_guess(listing))
            kept_count += 1
        debug_info[platform.value] = {"scraped": raw_count, "kept_after_filters": kept_count, "error": error}

    logger.info("Gathered %d unique candidates: %s", len(merged), debug_info)
    return merged, debug_info


# ============================== EVALUATOR (Gemini) =========================

_SYSTEM_PROMPT = """You are an expert fashion buyer evaluating listings from \
Indian retail platforms (Myntra, Ajio, Meesho, Amazon.in, Flipkart) for a \
buyer. All prices are in Indian Rupees (INR). Given a buyer's request and a \
candidate listing (title, description, price, photos), evaluate:
1. match_score (0-100): fit vs. the buyer's request (style, era, silhouette, price).
2. authenticity_confidence (0-100): likelihood this is genuine vs. a replica \
   or mislabeled product (this matters especially on marketplace platforms \
   like Meesho/Amazon/Flipkart where third-party sellers list items).
3. condition_score (0-100): physical condition visible in photos, if a used/preloved item.
4. flagged_defects: short list of visible issues, if any.
5. counterfeit_signals: short list of reasons to doubt authenticity, if any.
6. summary: one or two sentence verdict.
7. recommended_offer: a fair lower price in INR if the platform supports \
   negotiation, or null if the listed price is already fair/fixed.

Respond with ONLY a JSON object with exactly these keys: match_score, \
authenticity_confidence, condition_score, flagged_defects, \
counterfeit_signals, summary, recommended_offer. No markdown, no \
commentary, no code fences."""


async def _download_image_bytes(client: httpx.AsyncClient, url: str) -> Optional[dict]:
    try:
        resp = await client.get(url, timeout=10)
        resp.raise_for_status()
        return {"mime_type": resp.headers.get("content-type", "image/jpeg"), "data": resp.content}
    except Exception:
        return None


async def evaluate_listing(query: SearchQuery, listing: Listing) -> Optional[EvaluationResult]:
    if not GOOGLE_API_KEY:
        logger.warning("GOOGLE_API_KEY not set — skipping evaluation for %s", listing.unique_key())
        return None

    text_prompt = (
        f"Buyer request: {query.prompt}\nStyle tags: {', '.join(query.style_tags) or 'none'}\n"
        f"Max price: ₹{query.max_price or 'no limit'}\n\n"
        f"Listing title: {listing.title}\nDescription: {listing.description or 'n/a'}\n"
        f"Price: ₹{listing.price:.0f}\nPlatform: {listing.platform.value}\n"
        f"Brand: {listing.brand or 'n/a'}\nSize: {listing.size or 'n/a'}"
    )

    async with httpx.AsyncClient() as client:
        image_parts = []
        for photo_url in listing.photo_urls[:6]:
            part = await _download_image_bytes(client, str(photo_url))
            if part:
                image_parts.append(part)

    def _sync_call():
        model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=_SYSTEM_PROMPT)
        result = model.generate_content(
            [text_prompt, *image_parts],
            generation_config={"response_mime_type": "application/json", "temperature": 0.2},
        )
        return json.loads(result.text)

    try:
        raw = await asyncio.to_thread(_sync_call)
        return EvaluationResult(listing_key=listing.unique_key(), **raw)
    except Exception:
        logger.exception("Evaluation failed for %s", listing.unique_key())
        return None


async def evaluate_listings(query: SearchQuery, listings: list[Listing], concurrency: int = 4) -> list[EvaluationResult]:
    if not listings:
        return []
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(listing: Listing):
        async with sem:
            return await evaluate_listing(query, listing)

    results = await asyncio.gather(*(_bounded(l) for l in listings))
    return [r for r in results if r is not None]


# ============================== ALERTS & OFFERS ============================

_ALERT_REGISTRY: dict[str, MatchAlert] = {}  # in-memory; swap for Redis/DB for persistence
_OFFER_PREFIX = "offer:"


def build_alerts(listings: list[Listing], evaluations: list[EvaluationResult], threshold: int = MATCH_SCORE_THRESHOLD) -> list[MatchAlert]:
    by_key = {l.unique_key(): l for l in listings}
    alerts = []
    for ev in evaluations:
        if not ev.is_grail(threshold):
            continue
        listing = by_key.get(ev.listing_key)
        if listing:
            alerts.append(MatchAlert(listing=listing, evaluation=ev))
    alerts.sort(key=lambda a: a.evaluation.match_score, reverse=True)
    return alerts


async def send_alert(alert: MatchAlert) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured — skipping alert for %s", alert.listing.unique_key())
        return
    key = alert.listing.unique_key()
    _ALERT_REGISTRY[key] = alert
    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("💸 Make Lowball Offer", callback_data=f"{_OFFER_PREFIX}{key}")]])

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    async with app:
        await app.bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=alert.render_message(),
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=keyboard,
        )


async def submit_offer(alert: MatchAlert) -> str:
    listing = alert.listing
    suggested = alert.evaluation.recommended_offer or round(listing.price * 0.85, 2)
    offer_amount = min(suggested, MAX_PRICE_INR) if MAX_PRICE_INR else suggested

    if not ENABLE_AUTO_OFFERS:
        return f"🧪 DRY RUN (set ENABLE_AUTO_OFFERS=true to go live): would offer ₹{offer_amount:.0f} on {listing.title} ({listing.url})"

    # Most Indian retail platforms (Myntra/Ajio/Amazon/Flipkart) are fixed-price
    # and don't support buyer-initiated offers at all — Meesho sellers
    # sometimes negotiate via chat, but there's no public offer API/UI flow
    # to automate reliably. This is left as a manual step by design.
    return f"⚠️ These platforms don't support automated offers — go negotiate/purchase manually: {listing.url}"


async def _handle_offer_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    key = (query.data or "").removeprefix(_OFFER_PREFIX)
    alert = _ALERT_REGISTRY.get(key)
    if alert is None:
        await query.edit_message_text("This listing expired from memory — search again to refresh it.")
        return
    await query.message.reply_text("⏳ Checking offer options...")
    result = await submit_offer(alert)
    await query.message.reply_text(result)


def run_telegram_bot_polling() -> None:
    """Standalone process that listens for offer-button taps: `python app.py --bot`"""
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CallbackQueryHandler(_handle_offer_callback, pattern=f"^{_OFFER_PREFIX}"))
    logger.info("Telegram bot polling for offer button callbacks...")
    app.run_polling()


# ============================== LANGGRAPH PIPELINE =========================

async def search_node(state: GraphState) -> GraphState:
    query = state["query"]
    logger.info("🔎 Searching platforms for: %s", query.search_terms())
    listings, scrape_debug = await gather_candidates(query)
    return {**state, "listings": listings, "scrape_debug": scrape_debug}


async def evaluate_node(state: GraphState) -> GraphState:
    listings = state.get("listings", [])
    warnings = list(state.get("warnings", []))
    if not GOOGLE_API_KEY:
        warnings.append("GOOGLE_API_KEY is not set on the server — evaluation was skipped, so nothing can pass the match threshold.")
    if not listings:
        return {**state, "evaluations": [], "warnings": warnings}
    logger.info("🧠 Evaluating %d candidate(s)...", len(listings))
    return {**state, "evaluations": await evaluate_listings(state["query"], listings), "warnings": warnings}


async def alert_node(state: GraphState) -> GraphState:
    alerts = build_alerts(state.get("listings", []), state.get("evaluations", []))
    logger.info("🚨 %d listing(s) cleared the %d%% threshold", len(alerts), MATCH_SCORE_THRESHOLD)
    for alert in alerts:
        await send_alert(alert)
    return {**state, "alerts": alerts}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("search", search_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("alert", alert_node)
    graph.set_entry_point("search")
    graph.add_conditional_edges("search", lambda s: "evaluate" if s.get("listings") else "end", {"evaluate": "evaluate", "end": END})
    graph.add_edge("evaluate", "alert")
    graph.add_edge("alert", END)
    return graph.compile()


async def run_search(query: SearchQuery) -> GraphState:
    return await build_graph().ainvoke({"query": query})


# ============================== WEB SERVER (Render Web Service) ============
#
# Render's "Web Service" type requires something bound to $PORT that answers
# HTTP requests, or the deploy fails. This gives it that: index.html served
# at "/", plus a health check + /compare + /search, while the Telegram
# offer-button listener runs continuously in a background thread alongside it.

flask_app = Flask(__name__)

# Frontend lives in index.html, next to this file — read fresh each request
# so you can edit index.html without restarting the server.
_INDEX_HTML_PATH = Path(__file__).parent / "index.html"


def _load_index_html() -> str:
    try:
        return _INDEX_HTML_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "<h1>index.html not found</h1><p>Make sure index.html sits next to app.py.</p>"


@flask_app.get("/")
def index():
    return _load_index_html()


@flask_app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "thrift-grail-finder"}), 200


@flask_app.post("/compare")
def http_compare():
    """
    POST /compare
    Body: {"prompt": "...", "sizes": [...], "max_price": 3000, "min_price": 500}
    (max_price/min_price are in INR)

    Fast path: scrapes Myntra/Ajio/Meesho/Amazon.in/Flipkart and returns raw
    listings grouped by platform (title, price in INR, best-effort size,
    url) — no Gemini vision scoring, so this is much quicker than /search
    and doesn't need GOOGLE_API_KEY at all. This is what the frontend uses.
    """
    data = request.get_json(force=True, silent=True) or {}
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    query = SearchQuery(
        prompt=prompt,
        style_tags=data.get("tags", []),
        sizes=data.get("sizes", []),
        max_price=data.get("max_price"),
        min_price=data.get("min_price"),
    )

    grouped, scrape_debug = asyncio.run(compare_platforms(query))

    platforms_out = {}
    for platform_name, listings in grouped.items():
        prices = [l.price for l in listings]
        platforms_out[platform_name] = {
            "count": len(listings),
            "min_price": min(prices) if prices else None,
            "max_price": max(prices) if prices else None,
            "items": [
                {
                    "title": l.title,
                    "price": l.price,
                    "currency": l.currency,
                    "size": l.size,
                    "brand": l.brand,
                    "url": str(l.url),
                }
                for l in listings
            ],
        }

    warnings = []
    if all(v["scraped"] == 0 and v["error"] is None for v in scrape_debug.values()):
        warnings.append(
            "All platforms returned 0 listings with no errors — the scraper selectors are likely "
            "stale, or the sites are detecting/blocking headless Chromium on this host."
        )
    blocked = [p for p, v in scrape_debug.items() if v["error"] and p in ("amazon", "flipkart")]
    if blocked:
        warnings.append(
            f"{', '.join(blocked)} failed — Amazon/Flipkart frequently block datacenter IPs like Render's; "
            "this can happen even when the scraper code is correct."
        )

    return jsonify({
        "query": prompt,
        "platforms": platforms_out,
        "scrape_debug": scrape_debug,
        "warnings": warnings,
    })


@flask_app.post("/search")
def http_search():
    """
    POST /search
    Body: {"prompt": "...", "tags": [...], "sizes": [...], "max_price": 3000, "min_price": 500}
    (prices in INR)

    Slower path: scrapes AND runs Gemini vision evaluation, only returning
    items that score above MATCH_SCORE_THRESHOLD (the "grail alert" mode —
    this is also what triggers Telegram alerts). Use /compare instead if
    you just want a fast price/availability comparison.
    """
    data = request.get_json(force=True, silent=True) or {}
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    query = SearchQuery(
        prompt=prompt,
        style_tags=data.get("tags", []),
        sizes=data.get("sizes", []),
        max_price=data.get("max_price"),
        min_price=data.get("min_price"),
    )

    final_state = asyncio.run(run_search(query))
    alerts = final_state.get("alerts", [])
    listings = final_state.get("listings", [])
    evaluations = final_state.get("evaluations", [])

    return jsonify({
        "alerts_found": len(alerts),
        "alerts": [
            {
                "title": a.listing.title,
                "price": a.listing.price,
                "url": str(a.listing.url),
                "match_score": a.evaluation.match_score,
                "authenticity_confidence": a.evaluation.authenticity_confidence,
            }
            for a in alerts
        ],
        "listings_found": len(listings),
        "evaluations_count": len(evaluations),
        "scrape_debug": final_state.get("scrape_debug", {}),
        "warnings": final_state.get("warnings", []),
    })


def _start_bot_polling_in_background() -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set — skipping Telegram bot thread.")
        return

    def _worker():
        asyncio.set_event_loop(asyncio.new_event_loop())
        try:
            run_telegram_bot_polling()
        except Exception:
            logger.exception("Telegram bot polling thread crashed")

    thread = threading.Thread(target=_worker, daemon=True, name="telegram-bot-polling")
    thread.start()
    logger.info("Telegram bot polling started in background thread.")


def run_web_service() -> None:
    port = int(os.environ.get("PORT", "10000"))  # Render sets $PORT for you
    if not GOOGLE_API_KEY:
        logger.warning("GOOGLE_API_KEY not set — /search evaluation step will fail (but /compare still works).")
    _start_bot_polling_in_background()
    logger.info("Starting web service on 0.0.0.0:%d", port)
    flask_app.run(host="0.0.0.0", port=port)


# ============================== CLI =========================================

def _parse_args():
    parser = argparse.ArgumentParser(description="The Thrift & Subculture Grail Finder")
    parser.add_argument("--prompt", help="Free-text description of the item/vibe you want.")
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--sizes", nargs="*", default=[])
    parser.add_argument("--max-price", type=float, default=None, help="INR")
    parser.add_argument("--min-price", type=float, default=None, help="INR")
    parser.add_argument("--bot", action="store_true", help="Run only the Telegram offer-button listener (no HTTP server).")
    parser.add_argument("--web", action="store_true", help="Run as a web service: index.html + /compare + /search, with the Telegram bot polling in the background. Use this on Render.")
    return parser.parse_args()


async def _run_cli_search(args) -> None:
    if not GOOGLE_API_KEY:
        logger.warning("GOOGLE_API_KEY not set — evaluation step will fail.")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured — alerts will be skipped.")

    query = SearchQuery(
        prompt=args.prompt, style_tags=args.tags, sizes=args.sizes,
        max_price=args.max_price, min_price=args.min_price,
    )
    final_state = await run_search(query)

    alerts = final_state.get("alerts", [])
    if not alerts:
        logger.info("No grails found above the %d%% threshold this run.", MATCH_SCORE_THRESHOLD)
    for alert in alerts:
        logger.info("MATCH %d%% — %s — ₹%.0f — %s", alert.evaluation.match_score, alert.listing.title, alert.listing.price, alert.listing.url)


def main() -> None:
    args = _parse_args()

    if args.bot:
        run_telegram_bot_polling()
        return

    if args.web:
        run_web_service()
        return

    if not args.prompt:
        raise SystemExit("Provide --prompt (for a CLI search), --bot (Telegram listener only), or --web (Render web service).")

    asyncio.run(_run_cli_search(args))


if __name__ == "__main__":
    main()


# ============================== .env TEMPLATE ===============================
# Save the block below as a separate ".env" file in the same folder:
#
# GOOGLE_API_KEY=AIza...
# GEMINI_MODEL=gemini-1.5-pro
# TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
# TELEGRAM_CHAT_ID=123456789
# MATCH_SCORE_THRESHOLD=85
# MAX_PRICE_INR=3000
# ENABLE_AUTO_OFFERS=false
# HEADLESS=true
# REQUEST_DELAY_SECONDS=2.0
# PORT=10000
# (Render sets PORT for you automatically — you don't need to set it manually there.)
#
# On Render specifically, also set:
# PLAYWRIGHT_BROWSERS_PATH=0
# (forces the Chromium install to live inside site-packages instead of
# ~/.cache, which is what reliably persists from Render's build step into
# the runtime container — see the Build Command note in the project README.)
