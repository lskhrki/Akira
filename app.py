"""
The Thrift & Subculture Grail Finder — single-file version.

Pipeline: search resale platforms (Playwright) -> score candidates with
Google Gemini vision -> alert on Telegram with an offer button.

Run modes:
    # One-off CLI search
    python app.py --prompt "Oversized 90s leather racing jacket, size M" \
                   --tags y2k grunge --sizes M --max-price 70

    # Telegram offer-button listener only (no HTTP server)
    python app.py --bot

    # Web service mode (what Render's "Web Service" needs — binds to $PORT,
    # exposes a health check + /search endpoint, and runs the Telegram bot
    # polling loop in a background thread)
    python app.py --web

Env vars (put these in a .env file, see bottom of this file for the list):
    GOOGLE_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    MATCH_SCORE_THRESHOLD, MAX_PRICE_USD, ENABLE_AUTO_OFFERS, HEADLESS, PORT
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
MAX_PRICE_USD = float(os.getenv("MAX_PRICE_USD", "70"))
ENABLE_AUTO_OFFERS = os.getenv("ENABLE_AUTO_OFFERS", "false").lower() in {"1", "true", "yes"}
HEADLESS = os.getenv("HEADLESS", "true").lower() in {"1", "true", "yes"}
REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "2.0"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


# ============================== MODELS ===================================

class Platform(str, Enum):
    EBAY = "ebay"
    DEPOP = "depop"
    GRAILED = "grailed"


class SearchQuery(BaseModel):
    prompt: str
    style_tags: list[str] = Field(default_factory=list)
    max_price_usd: Optional[float] = None
    min_price_usd: Optional[float] = None
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
    price_usd: float
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
    recommended_offer_usd: Optional[float] = None

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
            f"💵 ${d.price_usd:.2f} | 📏 {d.size or 'n/a'} | 🏷 {d.brand or 'n/a'}",
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
# NOTE: DOM selectors are best-effort and will need upkeep as these sites
# change their frontends. eBay scraping here uses the public search page;
# swap in the official Browse API for production use.

def _parse_price(text: str) -> Optional[float]:
    match = re.search(r"[\d,]+\.?\d*", text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


async def _new_page(playwright):
    browser = await playwright.chromium.launch(headless=HEADLESS)
    context = await browser.new_context(user_agent=USER_AGENT, viewport={"width": 1366, "height": 900})
    page = await context.new_page()
    page.set_default_timeout(15000)  # fail fast (15s) instead of hanging on a blocked/slow site
    return browser, context, page


_SIZE_PATTERNS = [
    re.compile(r"\bsize\s*[:\-]?\s*([a-zA-Z0-9\.]{1,4})\b", re.IGNORECASE),
    re.compile(r"\bus\s?(\d{1,2}(?:\.\d)?)\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,2}(?:\.\d)?)\s?(?:us|uk|eu)\b", re.IGNORECASE),
    re.compile(r"\b(xxs|xs|s|m|l|xl|xxl|xxxl)\b", re.IGNORECASE),
]


def _guess_size_from_text(text: str) -> Optional[str]:
    """Best-effort size extraction from a listing title/description.
    eBay and Depop search cards don't expose structured size data — only
    Grailed does — so this is a fallback, not a guarantee."""
    for pattern in _SIZE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    return None


async def scrape_ebay(query: SearchQuery, max_results: int = 15) -> tuple[list[Listing], Optional[str]]:
    """Returns (listings, error_message). error_message is None on success,
    even if zero listings were found (that's a selector/blocking issue, not
    a hard failure) — check both fields to tell the two cases apart."""
    listings: list[Listing] = []
    try:
        async with async_playwright() as pw:
            browser, context, page = await _new_page(pw)
            url = f"https://www.ebay.com/sch/i.html?_nkw={query.search_terms().replace(' ', '+')}"
            await page.goto(url, wait_until="domcontentloaded")
            await asyncio.sleep(REQUEST_DELAY_SECONDS)
            cards = await page.locator("li.s-item").all()
            for card in cards[:max_results]:
                try:
                    title = (await card.locator(".s-item__title").inner_text()).strip()
                    link = await card.locator("a.s-item__link").get_attribute("href")
                    price = _parse_price((await card.locator(".s-item__price").inner_text()).strip())
                    img = await card.locator("img.s-item__image-img").get_attribute("src")
                    if not link or "ebay.com/itm" not in link or price is None:
                        continue
                    listings.append(Listing(
                        platform=Platform.EBAY,
                        listing_id=link.split("/itm/")[-1].split("?")[0],
                        url=link, title=title, price_usd=price,
                        photo_urls=[img] if img else [],
                    ))
                except Exception:
                    continue
            await context.close()
            await browser.close()
    except Exception as exc:
        logger.exception("eBay scrape failed")
        return listings, f"{exc.__class__.__name__}: {exc}"
    return listings, None


async def scrape_depop(query: SearchQuery, max_results: int = 15) -> tuple[list[Listing], Optional[str]]:
    listings: list[Listing] = []
    try:
        async with async_playwright() as pw:
            browser, context, page = await _new_page(pw)
            url = f"https://www.depop.com/search/?q={query.search_terms().replace(' ', '%20')}"
            await page.goto(url, wait_until="domcontentloaded")
            await asyncio.sleep(REQUEST_DELAY_SECONDS)
            for _ in range(3):
                await page.mouse.wheel(0, 2000)
                await page.wait_for_timeout(600)
            cards = await page.locator("[data-testid='product-card'], a[href^='/products/']").all()
            for card in cards[:max_results]:
                try:
                    href = await card.get_attribute("href")
                    if not href:
                        continue
                    full_url = href if href.startswith("http") else f"https://www.depop.com{href}"
                    price_el = card.locator("[data-testid='product-price'], [aria-label*='price']").first
                    price = _parse_price(await price_el.inner_text()) if await price_el.count() else None
                    img = await card.locator("img").first.get_attribute("src")
                    if price is None:
                        continue
                    listings.append(Listing(
                        platform=Platform.DEPOP,
                        listing_id=full_url.rstrip("/").split("/")[-1],
                        url=full_url, title=query.search_terms(), price_usd=price,
                        photo_urls=[img] if img else [],
                    ))
                except Exception:
                    continue
            await context.close()
            await browser.close()
    except Exception as exc:
        logger.exception("Depop scrape failed")
        return listings, f"{exc.__class__.__name__}: {exc}"
    return listings, None


async def scrape_grailed(query: SearchQuery, max_results: int = 15) -> tuple[list[Listing], Optional[str]]:
    listings: list[Listing] = []
    try:
        async with async_playwright() as pw:
            browser, context, page = await _new_page(pw)
            url = f"https://www.grailed.com/shop?query={query.search_terms().replace(' ', '+')}"
            await page.goto(url, wait_until="domcontentloaded")
            await asyncio.sleep(REQUEST_DELAY_SECONDS)
            cards = await page.locator("div.feed-item, a.listing-item-link").all()
            for card in cards[:max_results]:
                try:
                    href = await card.get_attribute("href") or await card.locator("a").first.get_attribute("href")
                    if not href:
                        continue
                    full_url = href if href.startswith("http") else f"https://www.grailed.com{href}"
                    price_el = card.locator(".sub-title, .ListingMetadata-module__price").first
                    price = _parse_price(await price_el.inner_text()) if await price_el.count() else None
                    img = await card.locator("img").first.get_attribute("src")
                    if price is None:
                        continue
                    listings.append(Listing(
                        platform=Platform.GRAILED,
                        listing_id=full_url.rstrip("/").split("/")[-1],
                        url=full_url, title=query.search_terms(), price_usd=price,
                        photo_urls=[img] if img else [],
                    ))
                except Exception:
                    continue
            await context.close()
            await browser.close()
    except Exception as exc:
        logger.exception("Grailed scrape failed")
        return listings, f"{exc.__class__.__name__}: {exc}"
    return listings, None


SCRAPERS = {Platform.EBAY: scrape_ebay, Platform.DEPOP: scrape_depop, Platform.GRAILED: scrape_grailed}


def _apply_size_guess(listing: Listing) -> Listing:
    if not listing.size:
        listing.size = _guess_size_from_text(listing.title)
    return listing


async def compare_platforms(query: SearchQuery) -> tuple[dict[str, list[Listing]], dict]:
    """Fast path: scrape only, no vision evaluation. Returns listings grouped
    by platform (for a price/availability comparison view) plus the same
    scrape_debug diagnostics as gather_candidates."""
    platforms = query.platforms or list(SCRAPERS.keys())
    results = await asyncio.gather(*(SCRAPERS[p](query) for p in platforms))

    grouped: dict[str, list[Listing]] = {}
    debug_info: dict = {}
    for platform, (group, error) in zip(platforms, results):
        filtered = []
        for listing in group:
            if query.max_price_usd and listing.price_usd > query.max_price_usd:
                continue
            if query.min_price_usd and listing.price_usd < query.min_price_usd:
                continue
            filtered.append(_apply_size_guess(listing))
        grouped[platform.value] = filtered
        debug_info[platform.value] = {"scraped": len(group), "kept_after_filters": len(filtered), "error": error}

    return grouped, debug_info


async def gather_candidates(query: SearchQuery) -> tuple[list[Listing], dict]:
    """Returns (merged_listings, debug_info). debug_info has one entry per
    platform: {"found": N, "error": str|None} so failures are visible
    instead of silently producing zero results."""
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
            if query.max_price_usd and listing.price_usd > query.max_price_usd:
                continue
            if query.min_price_usd and listing.price_usd < query.min_price_usd:
                continue
            seen.add(key)
            merged.append(listing)
            kept_count += 1
        debug_info[platform.value] = {"scraped": raw_count, "kept_after_filters": kept_count, "error": error}

    logger.info("Gathered %d unique candidates: %s", len(merged), debug_info)
    return merged, debug_info


# ============================== EVALUATOR (Gemini) =========================

_SYSTEM_PROMPT = """You are an expert vintage/streetwear authenticator and \
resale buyer for Gen Z secondhand fashion. Given a buyer's request and a \
candidate listing (title, description, price, photos), evaluate:
1. match_score (0-100): fit vs. the buyer's request (style, era, silhouette, price).
2. authenticity_confidence (0-100): likelihood this is genuine vs. a replica.
3. condition_score (0-100): physical condition visible in photos.
4. flagged_defects: short list of visible issues, if any.
5. counterfeit_signals: short list of reasons to doubt authenticity, if any.
6. summary: one or two sentence verdict.
7. recommended_offer_usd: a fair lowball offer below asking, or null if already a steal.

Respond with ONLY a JSON object with exactly these keys: match_score, \
authenticity_confidence, condition_score, flagged_defects, \
counterfeit_signals, summary, recommended_offer_usd. No markdown, no \
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
        f"Max price: {query.max_price_usd or 'no limit'}\n\n"
        f"Listing title: {listing.title}\nDescription: {listing.description or 'n/a'}\n"
        f"Price: ${listing.price_usd:.2f}\nBrand: {listing.brand or 'n/a'}\nSize: {listing.size or 'n/a'}"
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
    suggested = alert.evaluation.recommended_offer_usd or round(listing.price_usd * 0.85, 2)
    offer_amount = min(suggested, MAX_PRICE_USD) if MAX_PRICE_USD else suggested

    if not ENABLE_AUTO_OFFERS:
        return f"🧪 DRY RUN (set ENABLE_AUTO_OFFERS=true to go live): would offer ${offer_amount:.2f} on {listing.title} ({listing.url})"

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=HEADLESS)
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(str(listing.url), wait_until="domcontentloaded")

            if listing.platform == Platform.DEPOP:
                await page.get_by_role("button", name="Make offer").click(timeout=5000)
                await page.get_by_role("textbox").fill(str(int(offer_amount)))
                await page.get_by_role("button", name="Send offer").click(timeout=5000)
            elif listing.platform == Platform.GRAILED:
                await page.get_by_role("button", name="Offer").click(timeout=5000)
                await page.get_by_role("textbox").fill(str(int(offer_amount)))
                await page.get_by_role("button", name="Submit").click(timeout=5000)
            else:
                await browser.close()
                return f"⚠️ Auto-offer not implemented for {listing.platform.value}: {listing.url}"

            await context.close()
            await browser.close()
            return f"✅ Offer of ${offer_amount:.2f} submitted on {listing.platform.value}."
    except Exception as exc:
        logger.exception("Offer submission failed")
        return f"❌ Offer automation failed ({exc.__class__.__name__}) — go offer manually: {listing.url}"


async def _handle_offer_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    key = (query.data or "").removeprefix(_OFFER_PREFIX)
    alert = _ALERT_REGISTRY.get(key)
    if alert is None:
        await query.edit_message_text("This listing expired from memory — search again to refresh it.")
        return
    await query.message.reply_text("⏳ Submitting offer...")
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
# HTTP requests, or the deploy fails. This gives it that: a tiny Flask app
# with a health check + a /search endpoint, while the Telegram offer-button
# listener runs continuously in a background thread alongside it.

flask_app = Flask(__name__)

# Your frontend. Served directly at "/" so Render's URL opens straight into it.
_INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Thrift Grail Finder</title>
    <style>
        body{
            font-family:Arial,sans-serif;
            background:#f4f4f4;
            max-width:800px;
            margin:auto;
            padding:40px;
        }
        h1{
            text-align:center;
        }
        input,button{
            width:100%;
            padding:12px;
            margin:10px 0;
            font-size:16px;
            box-sizing:border-box;
        }
        button{
            background:#000;
            color:white;
            border:none;
            cursor:pointer;
        }
        #results{
            margin-top:30px;
        }
        .platform-block{
            background:#fff;
            border:1px solid #ddd;
            border-radius:10px;
            padding:15px 20px;
            margin:15px 0;
        }
        .platform-header{
            display:flex;
            justify-content:space-between;
            align-items:baseline;
        }
        .platform-name{
            text-transform:capitalize;
            font-size:20px;
            font-weight:bold;
        }
        .platform-stats{
            color:#555;
            font-size:14px;
        }
        table{
            width:100%;
            border-collapse:collapse;
            margin-top:10px;
            font-size:14px;
        }
        th,td{
            text-align:left;
            padding:6px 4px;
            border-bottom:1px solid #eee;
        }
        .debug{
            margin-top:20px;
            font-size:13px;
            color:#666;
            background:#fff;
            border:1px solid #ddd;
            border-radius:8px;
            padding:12px;
        }
        .warning{
            color:#b30000;
            font-weight:bold;
        }
    </style>
</head>
<body>
<h1>🧥 Thrift Grail Finder</h1>
<input
id="prompt"
placeholder="Example: nike dunks"
/>
<button onclick="searchItem()">
Search
</button>
<div id="results"></div>
<div id="debug" class="debug" style="display:none;"></div>
<script>
async function searchItem(){
const prompt=document.getElementById("prompt").value;
const results = document.getElementById("results");
const debugBox = document.getElementById("debug");
results.innerHTML = "<p>Searching eBay, Depop, and Grailed... usually 5-20s.</p>";
debugBox.style.display = "none";

const response=await fetch("/compare",{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify({
prompt:prompt
})
});
const data=await response.json();

if (data.warnings && data.warnings.length > 0) {
    debugBox.style.display = "block";
    debugBox.innerHTML = "<div class='warning'>" + data.warnings.join("<br>") + "</div>";
}
if (data.scrape_debug) {
    debugBox.style.display = "block";
    debugBox.innerHTML += "<pre>" + JSON.stringify(data.scrape_debug, null, 2) + "</pre>";
}

const platforms = data.platforms || {};
const platformNames = Object.keys(platforms);
const totalFound = platformNames.reduce((sum, p) => sum + platforms[p].items.length, 0);

if (totalFound === 0) {
    results.innerHTML = `
        <h2>No listings found 😔</h2>
        <p>Try a different or shorter search term.</p>
    `;
} else {
    let html = "";
    platformNames.forEach(p => {
        const info = platforms[p];
        if (info.items.length === 0) return;
        html += `<div class="platform-block">
            <div class="platform-header">
                <span class="platform-name">${p}</span>
                <span class="platform-stats">${info.items.length} listings &middot; $${info.min_price} - $${info.max_price}</span>
            </div>
            <table>
                <tr><th>Item</th><th>Size</th><th>Price</th><th></th></tr>`;
        info.items.forEach(item => {
            html += `<tr>
                <td>${item.title}</td>
                <td>${item.size || "n/a"}</td>
                <td>$${item.price_usd}</td>
                <td><a href="${item.url}" target="_blank">View</a></td>
            </tr>`;
        });
        html += `</table></div>`;
    });
    results.innerHTML = html;
}
}
</script>
</body>
</html>"""


@flask_app.get("/")
def index():
    return _INDEX_HTML


@flask_app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "thrift-grail-finder"}), 200


@flask_app.post("/compare")
def http_compare():
    """
    POST /compare
    Body: {"prompt": "...", "sizes": [...], "max_price": 70, "min_price": 10}

    Fast path: scrapes eBay/Depop/Grailed and returns raw listings grouped
    by platform (title, price, best-effort size, url) — no Gemini vision
    scoring, so this is much quicker than /search and doesn't need
    GOOGLE_API_KEY at all. This is what the built-in frontend uses.
    """
    data = request.get_json(force=True, silent=True) or {}
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    query = SearchQuery(
        prompt=prompt,
        style_tags=data.get("tags", []),
        sizes=data.get("sizes", []),
        max_price_usd=data.get("max_price"),
        min_price_usd=data.get("min_price"),
    )

    grouped, scrape_debug = asyncio.run(compare_platforms(query))

    platforms_out = {}
    for platform_name, listings in grouped.items():
        prices = [l.price_usd for l in listings]
        platforms_out[platform_name] = {
            "count": len(listings),
            "min_price": min(prices) if prices else None,
            "max_price": max(prices) if prices else None,
            "items": [
                {
                    "title": l.title,
                    "price_usd": l.price_usd,
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
    Body: {"prompt": "...", "tags": [...], "sizes": [...], "max_price": 70, "min_price": 10}

    Slower path: scrapes AND runs Gemini vision evaluation, only returning
    items that score above MATCH_SCORE_THRESHOLD (the "grail alert" mode —
    this is also what triggers Telegram alerts). Use /compare instead if
    you just want a fast price/availability comparison.

    Response includes scrape_debug (per-platform scraped/kept counts + any
    scraper error) and warnings (e.g. missing GOOGLE_API_KEY) so a silent
    "0 results" isn't a black box.
    """
    data = request.get_json(force=True, silent=True) or {}
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    query = SearchQuery(
        prompt=prompt,
        style_tags=data.get("tags", []),
        sizes=data.get("sizes", []),
        max_price_usd=data.get("max_price"),
        min_price_usd=data.get("min_price"),
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
                "price_usd": a.listing.price_usd,
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
        logger.warning("GOOGLE_API_KEY not set — evaluation step will fail on /search.")
    _start_bot_polling_in_background()
    logger.info("Starting web service on 0.0.0.0:%d", port)
    flask_app.run(host="0.0.0.0", port=port)


# ============================== CLI =========================================

def _parse_args():
    parser = argparse.ArgumentParser(description="The Thrift & Subculture Grail Finder")
    parser.add_argument("--prompt", help="Free-text description of the item/vibe you want.")
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--sizes", nargs="*", default=[])
    parser.add_argument("--max-price", type=float, default=None)
    parser.add_argument("--min-price", type=float, default=None)
    parser.add_argument("--bot", action="store_true", help="Run only the Telegram offer-button listener (no HTTP server).")
    parser.add_argument("--web", action="store_true", help="Run as a web service: health check + /search endpoint, with the Telegram bot polling in the background. Use this on Render.")
    return parser.parse_args()


async def _run_cli_search(args) -> None:
    if not GOOGLE_API_KEY:
        logger.warning("GOOGLE_API_KEY not set — evaluation step will fail.")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured — alerts will be skipped.")

    query = SearchQuery(
        prompt=args.prompt, style_tags=args.tags, sizes=args.sizes,
        max_price_usd=args.max_price, min_price_usd=args.min_price,
    )
    final_state = await run_search(query)

    alerts = final_state.get("alerts", [])
    if not alerts:
        logger.info("No grails found above the %d%% threshold this run.", MATCH_SCORE_THRESHOLD)
    for alert in alerts:
        logger.info("MATCH %d%% — %s — $%.2f — %s", alert.evaluation.match_score, alert.listing.title, alert.listing.price_usd, alert.listing.url)


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
# MAX_PRICE_USD=70
# ENABLE_AUTO_OFFERS=false
# HEADLESS=true
# REQUEST_DELAY_SECONDS=2.0
# PORT=10000
# (Render sets PORT for you automatically — you don't need to set it manually there.)
