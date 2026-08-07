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
from flask import Flask, jsonify, request, render_template

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
    return browser, context, page


async def scrape_ebay(query: SearchQuery, max_results: int = 15) -> list[Listing]:
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
    except Exception:
        logger.exception("eBay scrape failed")
    return listings


async def scrape_depop(query: SearchQuery, max_results: int = 15) -> list[Listing]:
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
    except Exception:
        logger.exception("Depop scrape failed")
    return listings


async def scrape_grailed(query: SearchQuery, max_results: int = 15) -> list[Listing]:
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
    except Exception:
        logger.exception("Grailed scrape failed")
    return listings


SCRAPERS = {Platform.EBAY: scrape_ebay, Platform.DEPOP: scrape_depop, Platform.GRAILED: scrape_grailed}


async def gather_candidates(query: SearchQuery) -> list[Listing]:
    platforms = query.platforms or list(SCRAPERS.keys())
    results = await asyncio.gather(*(SCRAPERS[p](query) for p in platforms))
    seen, merged = set(), []
    for group in results:
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
    logger.info("Gathered %d unique candidates", len(merged))
    return merged


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
    return {**state, "listings": await gather_candidates(query)}


async def evaluate_node(state: GraphState) -> GraphState:
    listings = state.get("listings", [])
    if not listings:
        return {**state, "evaluations": []}
    logger.info("🧠 Evaluating %d candidate(s)...", len(listings))
    return {**state, "evaluations": await evaluate_listings(state["query"], listings)}


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


@flask_app.get("/")
def home():
    return render_template("index.html")

@flask_app.get("/health")
def health():
    return jsonify({
        "status":"ok",
        "service":"thrift-grail-finder"
    })


@flask_app.post("/search")
def http_search():
    """
    POST /search
    Body: {"prompt": "...", "tags": [...], "sizes": [...], "max_price": 70, "min_price": 10}
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
