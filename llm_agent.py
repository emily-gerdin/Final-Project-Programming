"""
llm_agent.py
============

**Phase 3b - LLM Integration (the "NYC Travel Concierge")**

What this module does
---------------------
1. Compresses the listings currently on the map into a compact, factual
   **payload**: top-reviewed listing names, neighbourhoods, prices, review
   counts and aggregated sentiment.
2. Sends that payload to a Large Language Model and receives a natural-language
   travel guide recommending where to stay.
3. Supports **OpenAI** and **Google Gemini** behind one interface, with a
   **deterministic offline fallback** so a demo never breaks without a key.

Design decisions worth defending in the report
----------------------------------------------
* The model never sees raw review text or individual rows -- only aggregates and
  a short list of top listings. This keeps the prompt small and cheap, avoids
  sending user-level data, and means every number in the answer is one *we*
  computed and can verify.
* Every provider call has a timeout and is wrapped in ``try/except``; any failure
  degrades to the offline report, clearly labelled.
* The system prompt forbids inventing statistics.

Usage
-----
>>> import llm_agent as llm
>>> payload = llm.build_context_payload(filtered_listings, keywords)
>>> report = llm.generate_travel_report(payload, provider="auto")
>>> print(report.text)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

import config as cfg

# Persona and guardrails. Instructions are explicit because the model must not
# fabricate numbers and must write for a traveller, not an analyst.
SYSTEM_PROMPT = (
    "You are 'NYC Travel Concierge', an expert local guide for New York City "
    "short-term rentals. You receive ONLY aggregated statistics and a short list "
    "of top-reviewed listings computed from a real Airbnb dataset. Rules:\n"
    "1. Never invent numbers. Use only the figures provided in the JSON.\n"
    "2. Name specific neighbourhoods and listings from the data.\n"
    "3. Write for a traveller choosing where to stay: pros, cons, a clear pick.\n"
    "4. Use short markdown sections and bullet points.\n"
    "5. If the data is thin or missing, say so plainly instead of guessing."
)

# Keeps latency and token spend predictable.
MAX_OUTPUT_TOKENS = 1000
REQUEST_TIMEOUT_SECONDS = 45


@dataclass
class LLMReport:
    """
    Container for one generated travel guide.

    Attributes
    ----------
    text:
        Markdown guide to render in Streamlit.
    provider:
        ``"openai"``, ``"gemini"`` or ``"offline"``.
    model:
        Model identifier actually used.
    latency_seconds:
        Wall-clock generation time.
    tokens:
        Token usage when the provider reports it, else ``None``.
    warning:
        Human-readable note explaining any fallback.
    context:
        The payload that was sent (useful for debugging and screenshots).
    """

    text: str
    provider: str
    model: str
    latency_seconds: float = 0.0
    tokens: int | None = None
    warning: str | None = None
    context: dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# SECTION 1 - PROVIDER HELPERS
# ===========================================================================
def _round(value: Any, digits: int = 2) -> Any:
    """Round a number defensively; pass non-numbers through untouched."""
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def active_provider() -> str | None:
    """
    Return the provider that has an API key configured.

    Preference follows ``config.LLM_PROVIDER`` when both keys exist.

    Returns
    -------
    str | None
        ``"openai"``, ``"gemini"`` or ``None`` (offline mode).
    """
    if cfg.LLM_PROVIDER == "gemini" and cfg.GEMINI_API_KEY:
        return "gemini"
    if cfg.OPENAI_API_KEY:
        return "openai"
    if cfg.GEMINI_API_KEY:
        return "gemini"
    return None


def provider_status() -> dict[str, Any]:
    """
    Describe LLM availability for the sidebar (no secrets exposed).

    Returns
    -------
    dict[str, Any]
        ``openai_configured``, ``gemini_configured``, ``active`` and a readable
        ``message``.
    """
    active = active_provider()
    if active == "openai":
        message = f"OpenAI configured (model: {cfg.LLM_MODEL})."
    elif active == "gemini":
        message = f"Gemini configured (model: {cfg.GEMINI_MODEL})."
    else:
        message = ("No LLM key found - the concierge will use its built-in "
                   "offline guide generator (still fully functional).")
    return {
        "openai_configured": bool(cfg.OPENAI_API_KEY),
        "gemini_configured": bool(cfg.GEMINI_API_KEY),
        "active": active,
        "message": message,
    }


# ===========================================================================
# SECTION 2 - BUILDING THE PROMPT PAYLOAD
# ===========================================================================
def build_context_payload(
    filtered_listings: pd.DataFrame,
    keywords: dict[str, list[tuple[str, int]]] | None = None,
    top_listings: int = 10,
    top_neighbourhoods: int = 5,
) -> dict[str, Any]:
    """
    Compress the currently-filtered listings into a small JSON-serialisable dict.

    Parameters
    ----------
    filtered_listings:
        Rows currently visible on the map (cleaned, sentiment attached).
    keywords:
        Output of ``nlp_sentiment.extract_keywords`` (optional).
    top_listings:
        How many individual top-reviewed listing names to include.
    top_neighbourhoods:
        How many neighbourhoods per borough to include.

    Returns
    -------
    dict[str, Any]
        Nested dict with ``summary``, ``by_borough``, ``top_listings``,
        ``top_neighbourhoods``, ``sentiment`` and ``review_keywords``.
    """
    if filtered_listings is None or filtered_listings.empty:
        return {
            "summary": {"n_listings": 0},
            "note": "No listings match the current filters.",
        }

    df = filtered_listings
    has_sentiment = (
        "avg_sentiment" in df.columns and df["avg_sentiment"].notna().any()
    )

    payload: dict[str, Any] = {}

    # --- 1. headline statistics ------------------------------------------
    payload["summary"] = {
        "n_listings": int(len(df)),
        "median_price": _round(df["price"].median()),
        "p25_price": _round(df["price"].quantile(0.25)),
        "p75_price": _round(df["price"].quantile(0.75)),
        "median_reviews": _round(df["number_of_reviews"].median(), 0),
        "max_reviews": int(df["number_of_reviews"].max()),
        "pct_entire_home": _round((df["room_type"] == "Entire home/apt").mean() * 100, 1),
        "median_min_nights": _round(df["minimum_nights"].median(), 0),
    }

    # --- 2. per-borough table --------------------------------------------
    borough_aggs: dict[str, tuple[str, str]] = {
        "listings": ("id", "size"),
        "median_price": ("price", "median"),
        "median_reviews": ("number_of_reviews", "median"),
    }
    if has_sentiment:
        borough_aggs["avg_sentiment"] = ("avg_sentiment", "mean")

    by_borough = (
        df.groupby("neighbourhood_group")
        .agg(**borough_aggs)
        .round(2)
        .sort_values("listings", ascending=False)
        .reset_index()
    )
    payload["by_borough"] = by_borough.to_dict(orient="records")

    # --- 3. named top-reviewed listings ----------------------------------
    # Names + key facts only; the model needs concrete places to recommend.
    top = df.sort_values("number_of_reviews", ascending=False).head(top_listings)
    listing_fields = [c for c in (
        "name", "neighbourhood", "neighbourhood_group", "room_type",
        "price", "number_of_reviews", "avg_sentiment",
    ) if c in top.columns]
    payload["top_listings"] = [
        {key: (str(row[key])[:70] if key == "name" else row[key])
         for key in listing_fields}
        for row in top.to_dict(orient="records")
    ]

    # --- 4. top neighbourhoods -------------------------------------------
    hood_aggs: dict[str, tuple[str, str]] = {
        "listings": ("id", "size"),
        "median_price": ("price", "median"),
        "median_reviews": ("number_of_reviews", "median"),
    }
    if has_sentiment:
        hood_aggs["avg_sentiment"] = ("avg_sentiment", "mean")

    neighbourhood_table = (
        df.groupby(["neighbourhood", "neighbourhood_group"])
        .agg(**hood_aggs)
        .round(2)
        .reset_index()
        .sort_values("listings", ascending=False)
    )
    payload["top_neighbourhoods"] = (
        neighbourhood_table.head(top_neighbourhoods * 3).to_dict(orient="records")
    )
    payload["top_by_borough"] = {
        str(borough): (
            neighbourhood_table[neighbourhood_table["neighbourhood_group"] == borough]
            .head(top_neighbourhoods).to_dict(orient="records")
        )
        for borough in by_borough["neighbourhood_group"].head(3)
    }

    # --- 5. sentiment block ----------------------------------------------
    if has_sentiment:
        scored = df.dropna(subset=["avg_sentiment"])
        payload["sentiment"] = {
            "listings_with_reviews": int(len(scored)),
            "avg_sentiment": _round(scored["avg_sentiment"].mean(), 3),
            "pct_positive": _round((scored["avg_sentiment"] >= 0.05).mean() * 100, 1),
            "pct_negative": _round((scored["avg_sentiment"] <= -0.05).mean() * 100, 1),
            "best_rated_neighbourhoods": (
                scored.groupby("neighbourhood")["avg_sentiment"]
                .mean().sort_values(ascending=False).head(5).round(3).to_dict()
            ),
        }
    else:
        payload["sentiment"] = {"note": "Sentiment scores unavailable for this slice."}

    if keywords:
        payload["review_keywords"] = {
            band: [word for word, _ in words[:10]]
            for band, words in keywords.items()
        }

    return payload



def build_user_prompt(payload: dict[str, Any]) -> str:
    """
    Wrap the payload in the traveller-facing instruction sent to the model.

    Parameters
    ----------
    payload:
        Output of ``build_context_payload``.

    Returns
    -------
    str
        The user message: JSON data plus the required output structure.
    """
    return (
        "Using ONLY the statistics below, write a travel guide with exactly these "
        "markdown sections:\n\n"
        "## What your filters selected\n"
        "## Where to stay: the best-reviewed areas\n"
        "## Standout listings\n"
        "## Price reality check\n"
        "## What guests say\n"
        "## Final recommendation\n\n"
        f"DATA:\n```json\n{json.dumps(payload, indent=2, default=str)}\n```"
    )


# ===========================================================================
# SECTION 3 - PROVIDER IMPLEMENTATIONS
# ===========================================================================
def _call_openai(user_prompt: str) -> tuple[str, int | None]:
    """
    Send one chat-completion request to the OpenAI API.

    Parameters
    ----------
    user_prompt:
        Message built by ``build_user_prompt``.

    Returns
    -------
    tuple[str, int | None]
        ``(guide_markdown, total_tokens)``.

    Raises
    ------
    RuntimeError
        When the key is missing, the package is absent, or the call fails.
    """
    if not cfg.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    try:
        from openai import OpenAI
    except ImportError as exc:                        # pragma: no cover
        raise RuntimeError("the openai package is not installed.") from exc

    client = OpenAI(api_key=cfg.OPENAI_API_KEY, timeout=REQUEST_TIMEOUT_SECONDS)
    response = client.chat.completions.create(
        model=cfg.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,          # low temperature keeps the output factual
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    text = (response.choices[0].message.content or "").strip()
    tokens = response.usage.total_tokens if response.usage else None
    return text, tokens


def _call_gemini(user_prompt: str) -> tuple[str, int | None]:
    """
    Send one generation request to the Google Gemini API.

    Parameters
    ----------
    user_prompt:
        Message built by ``build_user_prompt``.

    Returns
    -------
    tuple[str, int | None]
        ``(guide_markdown, total_tokens)``.

    Raises
    ------
    RuntimeError
        When the key is missing, the package is absent, or the call fails.
    """
    if not cfg.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    try:
        import google.generativeai as genai
    except ImportError as exc:                        # pragma: no cover
        raise RuntimeError("google-generativeai is not installed.") from exc

    genai.configure(api_key=cfg.GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=cfg.GEMINI_MODEL, system_instruction=SYSTEM_PROMPT
    )
    response = model.generate_content(
        user_prompt,
        generation_config={"temperature": 0.4, "max_output_tokens": MAX_OUTPUT_TOKENS},
        request_options={"timeout": REQUEST_TIMEOUT_SECONDS},
    )
    text = (getattr(response, "text", "") or "").strip()
    tokens = None
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        tokens = int(getattr(usage, "total_token_count", 0) or 0) or None
    return text, tokens



# ===========================================================================
# SECTION 4 - OFFLINE / DETERMINISTIC GUIDE (graceful degradation)
# ===========================================================================
def build_offline_report(payload: dict[str, Any]) -> str:
    """
    Generate a factual markdown guide **without** any LLM call.

    This safety net makes the app demo-proof: with no key, no network or a
    provider outage the user still gets a readable, number-driven guide built
    from exactly the payload the model would have received.

    Parameters
    ----------
    payload:
        Output of ``build_context_payload``.

    Returns
    -------
    str
        Markdown guide.
    """
    summary = payload.get("summary", {}) or {}
    n_listings = int(summary.get("n_listings", 0) or 0)

    lines: list[str] = ["## What your filters selected", ""]
    if n_listings == 0:
        lines.append("No listings match the current filters. Try widening the "
                     "price range or lowering the minimum review count.")
        return "\n".join(lines)

    lines.append(
        f"**{n_listings:,} top-reviewed listings** match your filters, with a "
        f"median price of **${summary.get('median_price')}** per night "
        f"(middle 50%: ${summary.get('p25_price')}-${summary.get('p75_price')}) "
        f"and a median of **{summary.get('median_reviews')} reviews** each.\n"
    )

    # --- boroughs ---------------------------------------------------------
    lines += ["## Where to stay: the best-reviewed areas", ""]
    for row in payload.get("by_borough", [])[:4]:
        detail = (
            f"- **{row.get('neighbourhood_group')}** - "
            f"{int(row.get('listings', 0)):,} listings, "
            f"median ${row.get('median_price')}, "
            f"median {row.get('median_reviews')} reviews"
        )
        sentiment = row.get("avg_sentiment")
        if sentiment is not None:
            detail += f", avg sentiment {sentiment:+.2f}"
        lines.append(detail)

        for hood in payload.get("top_by_borough", {}).get(
            str(row.get("neighbourhood_group")), []
        )[:3]:
            lines.append(
                f"  - {hood.get('neighbourhood')}: "
                f"{int(hood.get('listings', 0)):,} listings, "
                f"median ${hood.get('median_price')}"
            )

    # --- standout listings ------------------------------------------------
    lines += ["", "## Standout listings", ""]
    for listing in payload.get("top_listings", [])[:5]:
        line = (
            f"- **{listing.get('name')}** "
            f"({listing.get('neighbourhood')}, {listing.get('neighbourhood_group')}) "
            f"- ${listing.get('price')}/night, "
            f"{listing.get('number_of_reviews')} reviews"
        )
        sentiment = listing.get("avg_sentiment")
        if sentiment is not None:
            line += f", sentiment {sentiment:+.2f}"
        lines.append(line)

    # --- price ------------------------------------------------------------
    lines += [
        "",
        "## Price reality check",
        "",
        f"- Median nightly price: **${summary.get('median_price')}**",
        f"- Share of entire-home listings: **{summary.get('pct_entire_home')}%**",
        f"- Median minimum stay: **{summary.get('median_min_nights')} nights**",
        f"- Most-reviewed listing has **{summary.get('max_reviews'):,} reviews**",
        "",
        "## What guests say",
        "",
    ]

    sentiment_block = payload.get("sentiment", {}) or {}
    if sentiment_block.get("avg_sentiment") is not None:
        lines.append(
            f"- Average review sentiment: **{sentiment_block.get('avg_sentiment')}** "
            "on a -1 (negative) to +1 (positive) scale."
        )
        lines.append(
            f"- **{sentiment_block.get('pct_positive')}%** of listings are clearly "
            f"positive, **{sentiment_block.get('pct_negative')}%** clearly negative."
        )
        best = sentiment_block.get("best_rated_neighbourhoods", {}) or {}
        if best:
            lines.append(
                "- Best-reviewed neighbourhoods: "
                + ", ".join(f"{name} ({score})" for name, score in list(best.items())[:3])
            )
    else:
        lines.append("- Sentiment scores are not available for this selection.")

    keywords = payload.get("review_keywords", {}) or {}
    if keywords.get("positive"):
        lines.append(
            "- Guests repeatedly praise: "
            + ", ".join(f"*{w}*" for w in keywords["positive"][:6])
        )
    if keywords.get("negative"):
        lines.append(
            "- Common complaints mention: "
            + ", ".join(f"*{w}*" for w in keywords["negative"][:6])
        )

    # --- recommendation ---------------------------------------------------
    lines += ["", "## Final recommendation", ""]
    borough_rows = payload.get("by_borough", []) or []
    if borough_rows:
        top_borough = borough_rows[0].get("neighbourhood_group")
        best_hoods = payload.get("top_by_borough", {}).get(str(top_borough), [])
        if best_hoods:
            names = ", ".join(str(h.get("neighbourhood")) for h in best_hoods[:2])
            lines.append(
                f"Start with **{names} ({top_borough})** - it has the deepest supply "
                "at these settings, so you get the most choice. Compare against the "
                "next borough for better value per night, and use the sentiment score "
                "to avoid cheap-but-poorly-rated stays."
            )
        else:
            lines.append(
                f"Start with **{top_borough}**, which has the most listings at your "
                "settings, then narrow by review count and sentiment."
            )
    return "\n".join(lines)



# ===========================================================================
# SECTION 5 - PUBLIC ENTRY POINTS
# ===========================================================================
def generate_travel_report(
    payload: dict[str, Any],
    provider: str = "auto",
    allow_offline: bool = True,
) -> LLMReport:
    """
    Generate the "NYC Travel Concierge" guide from an aggregated payload.

    Parameters
    ----------
    payload:
        Output of ``build_context_payload``.
    provider:
        ``"auto"`` (use whichever key exists), ``"openai"``, ``"gemini"`` or
        ``"offline"`` (skip the API entirely).
    allow_offline:
        When ``True`` (default) any API failure falls back to
        ``build_offline_report``. When ``False`` the exception is re-raised,
        which is handy in tests.

    Returns
    -------
    LLMReport
        The guide text plus metadata describing how it was produced.
    """
    chosen = active_provider() if provider == "auto" else provider

    # --- offline path -----------------------------------------------------
    if chosen == "offline" or chosen is None:
        started = time.perf_counter()
        return LLMReport(
            text=build_offline_report(payload),
            provider="offline",
            model="offline-template-v1",
            latency_seconds=round(time.perf_counter() - started, 3),
            warning=("Offline mode: no LLM API key configured, so this guide was "
                     "generated from the same statistics with a deterministic "
                     "template."),
            context=payload,
        )

    # --- live provider ----------------------------------------------------
    user_prompt = build_user_prompt(payload)
    started = time.perf_counter()
    try:
        if chosen == "openai":
            text, tokens = _call_openai(user_prompt)
            model = cfg.LLM_MODEL
        elif chosen == "gemini":
            text, tokens = _call_gemini(user_prompt)
            model = cfg.GEMINI_MODEL
        else:
            raise RuntimeError(f"Unknown provider '{chosen}'.")

        if not text:
            raise RuntimeError("The model returned an empty response.")

        return LLMReport(
            text=text, provider=chosen, model=model,
            latency_seconds=round(time.perf_counter() - started, 3),
            tokens=tokens, context=payload,
        )
    except Exception as exc:
        if not allow_offline:
            raise
        return LLMReport(
            text=build_offline_report(payload),
            provider="offline",
            model="offline-template-v1",
            latency_seconds=round(time.perf_counter() - started, 3),
            warning=f"{chosen} call failed ({exc}). Showing the offline guide instead.",
            context=payload,
        )



def _answer_offline(question: str, payload: dict[str, Any]) -> str:
    """
    Small keyword-driven Q&A used when no LLM is available.

    Parameters
    ----------
    question:
        The user's question.
    payload:
        Aggregated data payload.

    Returns
    -------
    str
        Markdown answer.
    """
    text = question.lower()
    summary = payload.get("summary", {}) or {}
    by_borough = payload.get("by_borough", []) or []

    if any(word in text for word in ("cheap", "budget", "afford", "value", "price")):
        if not by_borough:
            return ("No listings match the current filters, so there is no price "
                    "data to compare.")
        cheapest = min(by_borough, key=lambda row: row.get("median_price") or float("inf"))
        return (
            f"The most affordable option in your selection is "
            f"**{cheapest.get('neighbourhood_group')}** with a median of "
            f"**${cheapest.get('median_price')}** per night. Across all "
            f"{summary.get('n_listings', 0):,} listings the median is "
            f"**${summary.get('median_price')}**."
        )

    if any(word in text for word in ("best", "review", "sentiment", "rated", "good")):
        best = (payload.get("sentiment", {}) or {}).get("best_rated_neighbourhoods", {})
        if best:
            top = list(best.items())[:3]
            return "Best-reviewed neighbourhoods in your selection: " + ", ".join(
                f"**{name}** ({score})" for name, score in top
            ) + "."
        return "Sentiment data is not available for this selection yet."

    if any(word in text for word in ("family", "kids", "quiet")):
        return (
            "For families, favour listings with a high sentiment score and a median "
            "minimum stay of several nights, away from the busiest tourist core. Use "
            "the borough filter to compare Brooklyn, Queens and Staten Island."
        )

    return (
        "Offline mode can answer price, value and review questions. Add an "
        "OPENAI_API_KEY or GEMINI_API_KEY to .env to unlock free-form answers. "
        f"Current selection: {summary.get('n_listings', 0):,} listings, "
        f"median ${summary.get('median_price')}."
    )


def answer_followup(
    question: str,
    payload: dict[str, Any],
    chat_history: list[dict[str, str]] | None = None,
    provider: str = "auto",
) -> LLMReport:
    """
    Answer a free-text follow-up question about the current data slice.

    Powers the question box under the travel guide. Falls back to keyword
    matching when no provider is available.

    Parameters
    ----------
    question:
        What the user typed.
    payload:
        Same aggregated payload used for the main guide.
    chat_history:
        Optional prior ``{"role": ..., "content": ...}`` turns (OpenAI format).
    provider:
        ``"auto"``, ``"openai"``, ``"gemini"`` or ``"offline"``.

    Returns
    -------
    LLMReport
        The answer plus metadata (``provider="offline"`` when degraded).
    """
    if not question or not question.strip():
        return LLMReport(text="Please type a question first.", provider="offline",
                         model="offline-template-v1", context=payload)

    chosen = active_provider() if provider == "auto" else provider
    if chosen not in ("openai", "gemini"):
        return LLMReport(text=_answer_offline(question, payload), provider="offline",
                         model="offline-template-v1",
                         warning="Offline mode: answered with keyword matching.",
                         context=payload)

    context_json = json.dumps(payload, indent=2, default=str)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"DATA:\n```json\n{context_json}\n```"},
        *(chat_history or []),
        {"role": "user", "content": question},
    ]

    started = time.perf_counter()
    try:
        if chosen == "openai":
            from openai import OpenAI

            client = OpenAI(api_key=cfg.OPENAI_API_KEY, timeout=REQUEST_TIMEOUT_SECONDS)
            response = client.chat.completions.create(
                model=cfg.LLM_MODEL, messages=messages,
                temperature=0.3, max_tokens=400,
            )
            text = (response.choices[0].message.content or "").strip()
            tokens = response.usage.total_tokens if response.usage else None
            model = cfg.LLM_MODEL
        else:
            import google.generativeai as genai

            genai.configure(api_key=cfg.GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel(
                model_name=cfg.GEMINI_MODEL, system_instruction=SYSTEM_PROMPT
            )
            response = gemini_model.generate_content(
                f"DATA:\n```json\n{context_json}\n```\n\nQuestion: {question}",
                generation_config={"temperature": 0.3, "max_output_tokens": 400},
            )
            text = (getattr(response, "text", "") or "").strip()
            tokens = None
            model = cfg.GEMINI_MODEL

        if not text:
            raise RuntimeError("Empty response from provider.")
        return LLMReport(text=text, provider=chosen, model=model,
                         latency_seconds=round(time.perf_counter() - started, 3),
                         tokens=tokens, context=payload)
    except Exception as exc:
        return LLMReport(text=_answer_offline(question, payload), provider="offline",
                         model="offline-template-v1",
                         warning=f"{chosen} call failed ({exc}).", context=payload)

