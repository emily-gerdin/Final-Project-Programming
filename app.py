"""
app.py
======

**Phase 4 - The Streamlit application (20% of the grade)**

Top Reviewed NYC Travel Planner
-------------------------------
Run it with::

    streamlit run app.py

Layout
------
Sidebar
    Data source controls (Kaggle download / synthetic demo), NLP engine choice
    and the LLM provider for the AI concierge.
Tab 1 - Interactive Map
    pydeck map of NYC centred on (40.7128, -74.0060) with:
      * a **tilted 3D perspective** over the five boroughs,
      * a **satellite basemap by default** (Esri World Imagery, key-free),
        switchable to the light/dark Carto vector styles,
      * a **HeatmapLayer** showing where highly-reviewed rentals cluster,
      * **marker clustering** (``ColumnLayer``) with a white halo so the columns
        stay legible on photographic imagery,
      * a **hover card per listing** with a **photo preview**, name, price,
        rating, review count, neighbourhood, room type and sentiment -- shown
        on hover, not click (deck.gl tooltip)
      * a **photo preview gallery** of the most-reviewed listings, in plain
        HTML, as the fallback when the canvas cannot be hovered,
      * price + review-count sliders, borough/room-type filters and a geocoded
        area search.
Tab 2 - EDA Insights
    Raw-vs-cleaned audit, IQR outlier report, the "top reviewed" definition,
    price and review distributions, correlation matrix, and the volume of
    top-reviewed listings per borough.
Tab 3 - AI Travel Guide
    The "NYC Travel Concierge": sends the top-reviewed listings (names,
    neighbourhoods, prices, review counts, sentiment) to an LLM which writes a
    natural-language guide for the current filters. Degrades to a deterministic
    offline report when no API key is present.

Design system
-------------
The visual identity lives in ``.streamlit/config.toml`` (colours, Inter font,
radii, metric styling, chart palette) so every native widget is themed without
per-widget CSS. This file only adds what config.toml cannot express:

* layout tweaks written against Streamlit's own CSS variables
  (``var(--primary-color)``, ``var(--border-color)``...), so they follow the
  theme instead of hard-coding colours;
* the hero banner, which is presentation of the title rather than a widget;
* equal-height card rows, achieved with CSS subgrid over ``st.columns``.

Icons come from Material: widgets use Streamlit's native ``icon=`` /
``:material/...:`` syntax, and markdown headings use the matching inline SVG
from ``_MATERIAL_ICONS`` (same glyph, nothing to download).

Architecture notes
------------------
* ``app.py`` is UI + routing only. Data, cleaning, NLP and LLM logic live in the
  sibling modules, each independently testable.
* Expensive work (loading, cleaning, sentiment scoring) is cached with
  ``@st.cache_data`` keyed on its inputs, so filter changes are instant.
* Every external call sits inside ``try/except`` and reports through
  ``st.error`` / ``st.warning`` / ``st.info`` / ``st.spinner``.

Build status
------------
Phase 1 (``data_acquisition.py``), Phase 2 (``eda.py``), Phase 3
(``nlp_sentiment.py``, ``llm_agent.py``) and the Phase 4 map/EDA/AI tabs are
implemented in this file. Sections marked ``# --- Phase N: ...`` show where each
layer plugs in.
"""

from __future__ import annotations

import html
import traceback
from typing import Any

import pandas as pd
import pydeck as pdk
import streamlit as st

import config as cfg
import data_acquisition as da
import eda
import llm_agent as llm
import nlp_sentiment as nlp

# ---------------------------------------------------------------------------
# PAGE CONFIG -- must be the first Streamlit call in the script
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=cfg.APP_TITLE,
    page_icon=":material/travel_explore:",   # Material icon as the favicon
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# DESIGN SYSTEM
# ---------------------------------------------------------------------------
# Colours, fonts, radii and chart palettes are defined once in
# ``.streamlit/config.toml``. The block below is deliberately small: it covers
# only page-level composition (hero, card grid, map frame) and reads
# Streamlit's own CSS variables, so switching light/dark in the app menu keeps
# everything consistent. Widgets are not restyled here -- they inherit the
# theme natively.
#
# Variables Streamlit publishes on :root include --primary-color,
# --background-color, --secondary-background-color, --text-color, --font and
# --border-color. The wrappers used below -- stMainBlockContainer (main
# content block), stLayoutWrapper (one per st.container) and the .stColumn /
# .stVerticalBlock classes -- are the DOM structure Streamlit 1.6x emits; the
# rules are written so that if a future release renames them the page simply
# loses the card polish instead of breaking.
_CARD_ROWS = """.st-key-cardrow > div > [data-testid="stHorizontalBlock"],
.st-key-kpirow > div > [data-testid="stHorizontalBlock"]"""

_CARD_COLUMNS = """.st-key-cardrow > div > [data-testid="stHorizontalBlock"] > .stColumn,
.st-key-kpirow > div > [data-testid="stHorizontalBlock"] > .stColumn"""

_CARD_STACKS = """.st-key-cardrow > div > [data-testid="stHorizontalBlock"] > .stColumn > .stVerticalBlock,
.st-key-kpirow > div > [data-testid="stHorizontalBlock"] > .stColumn > .stVerticalBlock"""

_CARD_WRAPPERS = """.st-key-cardrow > div > [data-testid="stHorizontalBlock"] > .stColumn > .stVerticalBlock > [data-testid="stLayoutWrapper"],
.st-key-kpirow > div > [data-testid="stHorizontalBlock"] > .stColumn > .stVerticalBlock > [data-testid="stLayoutWrapper"]"""

st.html(
    f"""
    <style>
    /* ---- Page rhythm -------------------------------------------------
       Top/bottom padding and the content width now come from theme.paddingTop
       / theme.paddingBottom in .streamlit/config.toml, so nothing here has to
       fight the toolbar. Only the transparent header remains, which the theme
       has no key for. */
    [data-testid="stHeader"] {{background: transparent;}}

    /* ---- Hero banner (presentation of the title, not a widget) -------- */
    .hero-card {{
        position: relative;
        padding: 2.1rem 2.2rem 1.9rem;
        border-radius: 1.25rem;
        overflow: hidden;
        color: #FFFFFF;
        background:
            radial-gradient(900px 320px at 88% -25%, rgba(255, 56, 92, 0.55), transparent 60%),
            radial-gradient(600px 240px at 0% 120%, rgba(56, 189, 248, 0.28), transparent 65%),
            linear-gradient(135deg, #14171F 0%, #1F2430 55%, #2A1E27 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }}
    .hero-eyebrow {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.72rem;
        font-weight: 600;
        color: #FFC6CF;
        margin-bottom: 0.7rem;
    }}
    .hero-title {{
        font-size: clamp(1.9rem, 3.4vw, 2.7rem);
        font-weight: 800;
        line-height: 1.08;
        letter-spacing: -0.02em;
        margin: 0 0 0.6rem 0;
        color: #FFFFFF;
    }}
    .hero-sub {{
        max-width: 62ch;
        color: rgba(255, 255, 255, 0.78);
        font-size: 1.02rem;
        line-height: 1.55;
        margin: 0;
    }}
    .hero-meta {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 1.1rem;
    }}
    .hero-chip {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.28rem 0.7rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.9);
        background: rgba(255, 255, 255, 0.09);
        border: 1px solid rgba(255, 255, 255, 0.14);
        backdrop-filter: blur(4px);
    }}

    /* ---- Equal-height card rows ---------------------------------------
       Each st.columns cell stacks its contents into one child block. Giving
       those blocks a single shared row track plus
       ``grid-template-rows: subgrid`` makes every column read from the parent
       row, so a short card stretches to match its tallest neighbour.
       ``grid-auto-flow: row`` keeps stacked columns from silently producing a
       second row, which would break the subgrid span. Subgrid shipped in all
       major browsers in Sept 2023; where it is unsupported these rules are
       inert and cards simply take their content height. */
    {_CARD_ROWS} {{
        grid-auto-flow: row;
        grid-template-rows: minmax(min-content, max-content);
    }}
    {_CARD_COLUMNS} {{
        display: grid;
        grid-template-rows: auto minmax(0, 1fr);
    }}
    {_CARD_STACKS} {{
        grid-template-rows: auto minmax(0, 1fr);
        row-gap: 0.45rem;
    }}
    {_CARD_WRAPPERS} {{
        display: flex;
        flex-direction: column;
    }}
    {_CARD_WRAPPERS} > div {{
        flex: 1 1 auto;
    }}

    /* ---- Card headings ------------------------------------------------ */
    .card-head {{
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-weight: 700;
        font-size: 0.95rem;
        margin: 0;
        color: var(--text-color);
    }}
    .card-head svg {{color: var(--primary-color); flex: none;}}
    .card-note {{
        color: var(--text-color);
        opacity: 0.66;
        font-size: 0.82rem;
        margin: 0 0 0.2rem 0;
    }}

    /* ---- Map frame + legend ------------------------------------------- */
    .map-frame [data-testid="stPydeckChart"],
    .map-frame iframe {{border-radius: 0.9rem; overflow: hidden;}}
    .map-legend {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 1.1rem;
        padding: 0.55rem 0.9rem;
        border-radius: 0.75rem;
        border: 1px solid var(--border-color);
        background: var(--secondary-background-color);
        font-size: 0.8rem;
        color: var(--text-color);
        margin-top: 0.55rem;
    }}
    .map-legend .swatch {{
        display: inline-block;
        width: 0.72rem; height: 0.72rem;
        border-radius: 0.2rem;
        margin-right: 0.35rem;
        vertical-align: -0.08rem;
    }}
    .map-legend .gradient {{
        display: inline-block;
        width: 4.5rem; height: 0.5rem;
        border-radius: 999px;
        margin: 0 0.35rem;
        vertical-align: -0.06rem;
        background: linear-gradient(90deg, #3D8BFF, #FF385C);
    }}

    /* ---- Listing grid (hover preview fallback) ------------------------
       The deck.gl tooltip is the primary hover preview, but it needs a
       WebGL canvas that can be pick the marker. The grid below gives every
       listing the same photo + price card in plain HTML, so the feature still
       works on a device/browser where the canvas cannot be hovered. */
    .listing-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
        gap: 0.9rem;
        margin-top: 0.4rem;
    }}
    .listing-card {{
        position: relative;
        border-radius: 0.85rem;
        overflow: hidden;
        background: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        transition: transform 140ms ease, box-shadow 140ms ease,
                    border-color 140ms ease;
    }}
    .listing-card:hover,
    .listing-card:focus-within {{
        transform: translateY(-2px);
        border-color: var(--primary-color);
        box-shadow: 0 12px 26px rgba(0, 0, 0, 0.16);
    }}
    .listing-card .thumb {{
        display: block;
        width: 100%;
        aspect-ratio: 5 / 3;
        object-fit: cover;
        background: var(--secondary-background-color);
    }}
    .listing-card .body {{
        padding: 0.55rem 0.7rem 0.65rem;
        font-size: 0.8rem;
        line-height: 1.35;
        color: var(--text-color);
    }}
    .listing-card .name {{
        font-weight: 700;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .listing-card .meta {{opacity: 0.72; margin-top: 0.1rem;}}
    .listing-card .price {{
        font-weight: 700;
        color: var(--primary-color);
        margin-top: 0.2rem;
    }}
    .listing-card .preview {{
        position: absolute;
        inset: auto 0 0 0;
        padding: 0.5rem 0.7rem;
        font-size: 0.72rem;
        font-weight: 600;
        color: #FFFFFF;
        background: linear-gradient(180deg, transparent, rgba(0, 0, 0, 0.62));
        opacity: 0;
        transition: opacity 140ms ease;
    }}
    .listing-card:hover .preview,
    .listing-card:focus-within .preview {{opacity: 1;}}
    </style>
    """
)
# ---------------------------------------------------------------------------
# MATERIAL ICONS
# ---------------------------------------------------------------------------
# Streamlit 1.67 renders Material Symbols natively wherever an ``icon=`` or
# ``:material/name:`` argument exists (buttons, tabs, expanders, metrics,
# alerts, navigation). Markdown headings have no icon slot, so the same glyphs
# are inlined here as SVG from the Material Icons set (Apache-2.0). Keeping the
# SVG markup in one dict means a heading icon is a one-word change and every
# icon shares the same 24px grid, stroke and colour rule (``currentColor``).
#
# Icon names are the Material Icons ("filled") names; the equivalent
# Material Symbol string for native parameters is given next to each entry.
_MATERIAL_ICONS: dict[str, str] = {
    # travel_explore
    "explore": "<circle cx='12' cy='12' r='3.5'/><path d='M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm0 18a8 8 0 1 1 0-16 8 8 0 0 1 0 16z'/><path d='m16.5 7.5-4.6 1.9-1.9 4.6 4.6-1.9z'/>",
    # map
    "map": "<path d='M20.5 3 17 4.8 7 3 3.5 4.8v16L7 19l10 1.8 3.5-1.8zM15 18.5 9 17V5.5L15 7zM5 17V5.5l2-1V16zm14 1.5-2 1V10l2 1z'/>",
    # tune
    "tune": "<path d='M3 17v2h6v-2zM3 5v2h10V5zm10 16v-2h8v-2h-8v-2h-2v6zM7 9v2H3v2h4v2h2V9zm14 4v-2H11v2zm-6-4h2V7h4V5h-4V3h-2z'/>",
    # insights
    "insights": "<path d='M21 8a2 2 0 0 0-1.45.82l-3.2-3.2A2 2 0 0 0 16.5 4a2 2 0 1 0-3.94.5L9.5 7.5A2 2 0 1 0 6 9.5l-.5.5L4.5 9A2 2 0 1 0 3 11.5 2 2 0 0 0 4.5 13l1 1a2 2 0 1 0 2.5 2.5l3-3a2 2 0 0 0 2.5-2.5 2 2 0 0 0-.18-.82l3.2-3.2A2 2 0 1 0 21 8zM4.5 10.5a1 1 0 1 1 0-2 1 1 0 0 1 0 2zm2.5 5a1 1 0 1 1 0-2 1 1 0 0 1 0 2zm4.5-5a1 1 0 1 1 0-2 1 1 0 0 1 0 2zm3.5-4a1 1 0 1 1 0-2 1 1 0 0 1 0 2zm4.5 3a1 1 0 1 0 0-2 1 1 0 0 0 0 2z'/>",
    # smart_toy
    "concierge": "<path d='M20 9V7c0-1.66-1.34-3-3-3h-3c0-1.1-.9-2-2-2s-2 .9-2 2H7C5.34 4 4 5.34 4 7v2c0 1.66 1.34 3 3 3h3c0 1.1.9 2 2 2s2-.9 2-2h3c1.66 0 3-1.34 3-3zM7 10c-.55 0-1-.45-1-1V7c0-.55.45-1 1-1h3v4zm7 0h-3V6h3c.55 0 1 .45 1 1v2c0 .55-.45 1-1 1zm-3 2c-.55 0-1-.45-1-1s.45-1 1-1 1 .45 1 1-.45 1-1 1zm9.87 7.1A2 2 0 0 0 19 17H5a2 2 0 0 0-1.87 2.1c.1.9.87 1.6 1.87 1.6h12.96c1 0 1.77-.7 1.91-1.6z'/>",
    # cleaning_services
    "cleaning": "<path d='M16 11h1v2c0 .55.45 1 1 1s1-.45 1-1v-2a2 2 0 0 0-2-2h-1V7c0-1.1-.9-2-2-2h-1V3a1 1 0 0 0-2 0v2h-1c-1.1 0-2 .9-2 2v12H5a1 1 0 0 0 0 2h10a1 1 0 0 0 0-2h-3V7h3v4z'/>",
    # workspace_premium
    "premium": "<circle cx='12' cy='8' r='4'/><path d='M9 13.5 7 21l5-2.5L17 21l-2-7.5a7 7 0 0 1-6 0z'/>",
    # bar_chart
    "bar_chart": "<path d='M5 9.2h3V19H5zM10.6 5h2.8v14h-2.8zm5.6 8H19v6h-2.8z'/>",
    # hub / share
    "hub": "<path d='M21 6.5a2.5 2.5 0 1 0-3.5-2.3L14 6.7a6 6 0 0 0-4.4.6L6.9 4.6A2.5 2.5 0 1 0 5 6.4l1 1.6A6 6 0 0 0 8 17.2l-1 1.6A2.5 2.5 0 1 0 9.5 21l1.6-1a6 6 0 0 0 1.8 0l1.6 1a2.5 2.5 0 1 0 1.5-2.2l-.9-1.5A6 6 0 0 0 18 12.5c0-1-.25-1.95-.7-2.8l1.6-1a2.5 2.5 0 0 0 2.1-2.2zM12 15a3 3 0 1 1 0-6 3 3 0 0 1 0 6z'/>",
    # table_chart
    "table": "<path d='M10 10.02h5V21h-5zM17 21h3a2 2 0 0 0 2-2v-9h-5zM20 3H5a2 2 0 0 0-2 2v3h19V5a2 2 0 0 0-2-2zM3 19a2 2 0 0 0 2 2h3V10H3z'/>",
    # download
    "download": "<path d='M12 17 8.5 13.5l1.4-1.4 1.1 1.1V4h2v9.2l1.1-1.1 1.4 1.4zM6 20v-2h12v2z'/>",
    # whatshot (heatmap)
    "heat": "<path d='M13.5 0.67s.74 2.65.74 4.8c0 2.06-1.35 3.73-3.41 3.73-2.07 0-3.63-1.67-3.63-3.73l.03-.36C5.21 7.51 4 10.62 4 14c0 4.42 3.58 8 8 8s8-3.58 8-8C20 8.61 17.41 3.8 13.5.67zM11.71 19c-1.78 0-3.22-1.4-3.22-3.14 0-1.62 1.05-2.76 2.81-3.12 1.77-.36 3.6-1.21 4.62-2.58.39 1.29.59 2.65.59 4.04 0 2.65-2.15 4.8-4.8 4.8z'/>",
    # photo_camera
    "camera": "<circle cx='12' cy='12' r='3.2'/><path d='M9 2 7.2 4H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2h-3.2L15 2zm3 15.5A5.5 5.5 0 1 1 12 6.5a5.5 5.5 0 0 1 0 11z'/>",
    # layers
    "layers": "<path d='m11.99 18.54-7.37-5.73L3 14.07l9 7 9-7-1.63-1.27zM12 16l7.36-5.73L21 9l-9-7-9 7 1.63 1.27z'/>",
}


def material_icon(name: str, size: int = 18) -> str:
    """
    Return one Material Icons glyph as an inline ``<svg>`` string.

    Parameters
    ----------
    name:
        Key in :data:`_MATERIAL_ICONS`, e.g. ``"explore"``.
    size:
        Rendered edge length in pixels.

    Returns
    -------
    str
        SVG markup that inherits its colour from the surrounding text
        (``fill="currentColor"``), so it works in light and dark themes.
    """
    body = _MATERIAL_ICONS.get(name, _MATERIAL_ICONS["explore"])
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' "
        f"viewBox='0 0 24 24' fill='currentColor' aria-hidden='true' "
        f"style='vertical-align:-0.18em'>{body}</svg>"
    )


def card_title(icon: str, text: str, note: str = "") -> str:
    """
    Build the HTML for a card heading with a Material icon and optional note.

    Parameters
    ----------
    icon:
        Material icon key passed to :func:`material_icon`.
    text:
        Heading text (already in Title Case).
    note:
        Small muted line under the heading; omit for none.

    Returns
    -------
    str
        HTML fragment; render with ``st.markdown(..., unsafe_allow_html=True)``.
    """
    heading = f"<p class='card-head'>{material_icon(icon)}<span>{html.escape(text)}</span></p>"
    if note:
        heading += f"<p class='card-note'>{html.escape(note)}</p>"
    return heading




# ===========================================================================
# SECTION 1 - CACHED PIPELINE
# ===========================================================================
# ``st.cache_data`` memoises on the *arguments*. Underscore-prefixed arguments
# (``_df``) are not hashed, which matters because hashing a 48k-row DataFrame on
# every rerun would cost more than the work being cached.
# ===========================================================================
@st.cache_data(show_spinner=False)
def load_raw_data(
    force_download: bool = False,
    reviews_limit: int = 100_000,
) -> tuple[pd.DataFrame, pd.DataFrame, bool, str | None]:
    """
    Load raw listings and reviews through the acquisition module.

    Parameters
    ----------
    force_download:
        Re-download instead of using cached CSVs.
    reviews_limit:
        Maximum number of reviews to pull in (keeps start-up fast).

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, bool, str | None]
        ``(listings, reviews, used_synthetic, error_message)``. The error string
        is rendered in the UI rather than raised, so the app never dies here.
    """
    try:
        listings, reviews, used_synthetic = da.ensure_data_available(
            allow_synthetic=True,
            reviews_limit=reviews_limit,
            force_download=force_download,
        )
        return listings, reviews, used_synthetic, None
    except Exception as exc:                     # last-resort safety net
        return pd.DataFrame(), pd.DataFrame(), False, f"{type(exc).__name__}: {exc}"


@st.cache_data(show_spinner=False)
def build_clean_listings(_listings: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Run the Phase 2 cleaning pipeline and cache the result.

    Parameters
    ----------
    _listings:
        Raw listings (underscore => not hashed by Streamlit).

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, Any]]
        ``(clean_listings, audit_log)``.
    """
    return eda.clean_listings(_listings)


@st.cache_data(show_spinner=False)
def build_sentiment(
    _reviews: pd.DataFrame,
    engine: str,
    sample_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    """
    Score reviews with the local NLP engine and aggregate per listing.

    Parameters
    ----------
    _reviews:
        Raw reviews table.
    engine:
        ``"vader"`` or ``"textblob"``.
    sample_size:
        Review cap passed to the scorer.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, str | None]
        ``(scored_reviews, per_listing_sentiment, error_message)``.
    """
    try:
        scored = nlp.score_reviews(_reviews, engine=engine, sample_size=sample_size)
        per_listing = nlp.aggregate_sentiment_by_listing(scored)
        return scored, per_listing, None
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), f"{type(exc).__name__}: {exc}"


def geocode_cached(query: str) -> tuple[float, float, str] | None:
    """
    Look up a place name once per Streamlit session.

    Nominatim allows roughly one request per second, so results are memoised in
    ``st.session_state`` to keep repeat searches instant.

    Parameters
    ----------
    query:
        Free text from the search box.

    Returns
    -------
    tuple[float, float, str] | None
        ``(lat, lon, label)`` or ``None`` when nothing matched.
    """
    cache: dict[str, Any] = st.session_state.setdefault("geocode_cache", {})
    key = query.strip().lower()
    if key not in cache:
        cache[key] = da.geocode_location(query)
    return cache[key]



# ===========================================================================
# SECTION 2 - SIDEBAR
# ===========================================================================
def hex_to_rgb(value: str) -> tuple[int, int, int]:
    """
    Convert a ``#RRGGBB`` string to an ``(r, g, b)`` tuple.

    deck.gl colour accessors need integer RGB triples, while the palette lives
    in CSS notation in ``config.UI_PALETTE``, so the two need bridging in one
    obvious place.

    Parameters
    ----------
    value:
        Hex colour, with or without the leading ``#``.

    Returns
    -------
    tuple[int, int, int]
        Channels in 0-255.
    """
    raw = value.lstrip("#")
    if len(raw) == 3:                       # #abc shorthand
        raw = "".join(ch * 2 for ch in raw)
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)

def render_sidebar() -> dict[str, Any]:
    """
    Draw the sidebar controls and return the chosen settings.

    Returns
    -------
    dict[str, Any]
        ``force_download``, ``reviews_limit``, ``engine``, ``llm_provider``.
    """
    st.sidebar.markdown(
        f"<div style='display:flex;align-items:center;gap:.55rem;"
        f"color:var(--primary-color);font-weight:800;font-size:1.05rem;"
        f"letter-spacing:-.01em'>{material_icon('explore', 24)}"
        f"<span style='color:var(--text-color)'>NYC Travel Planner</span></div>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Capstone Project - Programming for Data Science")

    with st.sidebar.expander("Data Source", icon=":material/database:", expanded=True):
        if da.kaggle_credentials_available():
            st.success("Kaggle credentials detected.", icon=":material/check_circle:")
        else:
            st.warning(
                "No Kaggle credentials found, so the app will use the Inside "
                "Airbnb mirror or demo data. See README step 3 to add your key.",
                icon=":material/key_off:",
            )
        force_download = st.checkbox(
            "Force re-download",
            value=False,
            help="Ignore the cached CSVs in data/ and fetch a fresh copy.",
        )
        reviews_limit = st.select_slider(
            "Max reviews to score",
            options=[10_000, 50_000, 100_000, 250_000],
            value=100_000,
            help="Lower is faster to start; 50k reviews already give a stable "
                 "average sentiment per listing.",
        )

    with st.sidebar.expander("Sentiment Engine", icon=":material/sentiment_satisfied:"):
        engine = st.radio(
            "NLP engine",
            options=["vader", "textblob"],
            index=0,
            format_func=lambda v: {"vader": "VADER (recommended)",
                                   "textblob": "TextBlob"}[v],
        )
        st.caption("VADER is tuned for short review text; TextBlob is a simpler "
                   "polarity model kept as a comparison.")

    with st.sidebar.expander("AI Concierge", icon=":material/smart_toy:"):
        st.write(llm.provider_status()["message"])
        llm_provider = st.selectbox(
            "LLM provider",
            options=["auto", "openai", "gemini", "offline"],
            index=0,
            format_func=lambda v: {
                "auto": "Auto (use configured key)",
                "openai": "OpenAI",
                "gemini": "Google Gemini",
                "offline": "Offline report (no API call)",
            }[v],
        )

    st.sidebar.divider()
    st.sidebar.caption(
        "Data: Kaggle 'New York City Airbnb Open Data' (CC0) + Inside Airbnb "
        "reviews. Geocoding: OpenStreetMap Nominatim. Photos: Unsplash. "
        "Basemap: Esri World Imagery / CARTO."
    )
    return {
        "force_download": force_download,
        "reviews_limit": reviews_limit,
        "engine": engine,
        "llm_provider": llm_provider,
    }


# ===========================================================================
# SECTION 3 - HEADER
# ===========================================================================
def _hero_html(status_chip: str) -> str:
    """
    Compose the hero banner markup for :func:`render_header`.

    Parameters
    ----------
    status_chip:
        Label of the live-data chip (e.g. ``"Live Kaggle data"`` or
        ``"Synthetic demo data"``).

    Returns
    -------
    str
        HTML for the hero card.
    """
    chips = (
        f"<span class='hero-chip'>{material_icon('layers', 14)}48,895 listings</span>"
        f"<span class='hero-chip'>{material_icon('bar_chart', 14)}"
        f"{len(cfg.BOROUGHS)} boroughs</span>"
        f"<span class='hero-chip'>{material_icon('camera', 14)}{status_chip}</span>"
    )
    return (
        f"<div class='hero-card'>"
        f"<div class='hero-eyebrow'>{material_icon('explore', 15)}"
        f"NYC Airbnb Intelligence</div>"
        f"<h1 class='hero-title'>{html.escape(cfg.APP_TITLE)}</h1>"
        f"<p class='hero-sub'>{html.escape(cfg.APP_SUBTITLE)}</p>"
        f"<div class='hero-meta'>{chips}</div>"
        f"</div>"
    )


def render_header(used_synthetic: bool) -> None:
    """
    Render the page hero banner and, when applicable, the synthetic-data alert.

    Parameters
    ----------
    used_synthetic:
        Whether any part of the loaded data is generated demo data.

    Returns
    -------
    None
    """
    chip = "Synthetic demo data" if used_synthetic else "Live Kaggle data"
    st.html(_hero_html(chip))
    st.space("small")

    if used_synthetic:
        st.warning(
            "**Synthetic demo data** - the real dataset could not be reached, "
            "so every number below comes from generated data with an identical "
            "schema. Add Kaggle credentials (README step 3) and press "
            "*Force re-download* for the real NYC Airbnb data.",
            icon=":material/warning:",
        )




# ===========================================================================
# SECTION 4 - MAP: FILTERING, CLUSTERING AND LAYERS
# ===========================================================================
# Free Carto vector basemaps (no API token needed, unlike Mapbox).
BASEMAP_DARK = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
BASEMAP_LIGHT = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

# --- Satellite basemap -------------------------------------------------------
# deck.gl's ``TileLayer`` can render a pyramid of raster tiles from any XYZ
# endpoint. Esri's World Imagery service is free for non-commercial use, needs
# no API key and covers the whole planet, which is what makes a *true*
# photographic satellite view possible without a Mapbox token.
#
# The style below is a MapLibre Style Spec document that pydeck passes straight
# to deck.gl (deck.gl normalises ``{x}/{y}/{z}`` token order itself):
#   sources  -- the XYZ raster tile template for Esri World Imagery;
#   layers   -- one raster layer with a slight desaturation so the coral
#               markers stay readable over photographic greens and greys;
#   sky      -- a horizon gradient so the tilted 3D view ends in atmosphere
#               rather than empty space, which is what makes it feel finished.
#
# Esri's terms require attribution ("Esri, Maxar, Earthstar Geographics and the
# GIS User Community"); the UI prints it under the map.
ESRI_ATTRIBUTION = "Esri, Maxar, Earthstar Geographics and the GIS User Community"
_P = cfg.UI_PALETTE


def satellite_style(scrim: bool = True) -> dict[str, Any]:
    """
    Build the deck.gl style object for the satellite basemap.

    Kept as a function (not a module constant) so the palette in
    ``config.UI_PALETTE`` is read at call time and the returned dict can be
    mutated by tests without leaking state into the next render.

    Parameters
    ----------
    scrim:
        Add a faint dark wash above the imagery. Photographic tiles are bright
        and busy; without it the heatmap and the white marker outlines lose
        contrast over water and rooftops.

    Returns
    -------
    dict[str, Any]
        MapLibre-style document accepted by ``pdk.Deck(map_style=...)``.
    """
    layers: list[dict[str, Any]] = [
        {
            "id": "esri-world-imagery",
            "type": "raster",
            "source": "esri-world-imagery",
            "paint": {
                # A slight desaturation + contrast lift keeps the coral theme
                # markers readable on top of photographic greens/greys.
                "raster-saturation": -0.22,
                "raster-contrast": 0.06,
                "raster-brightness-min": 0.04,
                "raster-brightness-max": 0.94,
            },
        }
    ]
    if scrim:
        r, g, b, a = cfg.MAP_SATELLITE_SCRIM
        layers.append(
            {
                "id": "imagery-scrim",
                "type": "background",
                "paint": {"background-color": f"rgba({r}, {g}, {b}, {a / 255:.3f})"},
            }
        )

    return {
        "version": 8,
        "name": "nyc-satellite",
        "sources": {
            # deck.gl substitutes the {z}/{y}/{x} tokens itself; no escaping.
            "esri-world-imagery": {
                "type": "raster",
                "tiles": [
                    "https://server.arcgisonline.com/ArcGIS/rest/services/"
                    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
                ],
                "tileSize": 256,
                "attribution": ESRI_ATTRIBUTION,
                "minzoom": 0,
                "maxzoom": 19,
            }
        },
        "layers": layers,
        # The tilted NYC view ends in sky, not at the edge of the tiles. A deep
        # ink-to-night gradient reads as atmosphere behind the 3D city; a pale
        # horizon (the previous value) looked like a washed-out strip of paper
        # above the map in the light theme.
        "sky": {
            "sky-color": "#0E1622",
            "horizon-color": "#3A2A31",
            "fog-color": _P["surface"],
            "sky-horizon-blend": 0.55,
            "horizon-fog-blend": 0.6,
        },
        "light": {"anchor": "viewport", "color": "#FFFFFF", "intensity": 1.0},
    }



def filter_listings(
    df: pd.DataFrame,
    price_range: tuple[float, float],
    min_reviews: int,
    boroughs: list[str],
    room_types: list[str],
) -> pd.DataFrame:
    """
    Apply the map's filters to the top-reviewed listings table.

    Parameters
    ----------
    df:
        The cleaned, top-reviewed listings.
    price_range:
        ``(min, max)`` nightly price from the slider.
    min_reviews:
        Minimum ``number_of_reviews`` from the slider.
    boroughs:
        Selected ``neighbourhood_group`` values; empty means all.
    room_types:
        Selected room types; empty means all.

    Returns
    -------
    pandas.DataFrame
        Filtered copy, re-indexed.
    """
    out = df.copy()

    if boroughs:
        out = out[out["neighbourhood_group"].isin(boroughs)]
    if room_types:
        out = out[out["room_type"].isin(room_types)]

    out = out[out["price"].between(price_range[0], price_range[1])]
    out = out[out["number_of_reviews"] >= min_reviews]
    return out.reset_index(drop=True)


def build_map_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare rows for the deck.gl layers: tooltip text, colours, safe numerics.

    pydeck cannot render ``NaN`` inside a numeric accessor, so sentiment falls
    back to ``0.0`` while the tooltip displays "n/a".

    Parameters
    ----------
    df:
        Filtered listings.

    Returns
    -------
    pandas.DataFrame
        Copy with ``sentiment_value``, ``colour`` and ``tooltip`` columns.
    """
    map_df = df.copy()

    if "avg_sentiment" not in map_df.columns:
        map_df["avg_sentiment"] = None
    sentiment = pd.to_numeric(map_df["avg_sentiment"], errors="coerce")
    map_df["sentiment_value"] = sentiment.fillna(0.0)

    # Colour by review volume: cool blue (few reviews) -> coral (many), so the
    # most-reviewed listings stand out on both vector and satellite basemaps.
    # Per-channel interpolation is enough here because both stops are chosen to
    # sit on the same lightness band, so the midpoint stays a clear violet-grey
    # rather than the muddy brown an RGB ramp between blue and orange gives.
    reviews = map_df["number_of_reviews"].astype(float)
    low, high = float(reviews.min()), float(reviews.max())
    span = max(high - low, 1e-6)

    def _ramp(n: float) -> list[int]:
        t = (n - low) / span
        a = hex_to_rgb(_P["review_low"])
        b = hex_to_rgb(_P["review_high"])
        return [
            int(round(a[0] + (b[0] - a[0]) * t)),
            int(round(a[1] + (b[1] - a[1]) * t)),
            int(round(a[2] + (b[2] - a[2]) * t)),
        ]

    map_df["colour"] = reviews.apply(_ramp)

    sentiment_text = sentiment.apply(lambda v: "n/a" if pd.isna(v) else f"{v:+.2f}")

    # Hover preview: the tooltip is raw HTML, so a small <img> appears the
    # moment the pointer rests on a marker -- no click required. The photo
    # column is produced by ``da.listing_photo_urls`` (Unsplash thumbnails at
    # ~10 KB each, because the Kaggle file has no photo column), so hovers stay
    # instant even with thousands of markers.
    if "picture_url" in map_df.columns:
        picture = map_df["picture_url"].fillna("").astype(str).str.strip()
    else:
        picture = pd.Series("", index=map_df.index, dtype=str)

    sentiments_ok = sentiment.notna()
    sent_badge = sentiment.map(
        lambda v: "" if pd.isna(v) else (
            "Positive" if v >= 0.05 else "Neutral" if v > -0.05 else "Mixed")
    )
    # A missing score is silence, not a second label: "Unscored n/a" reads as a
    # contradiction. The chip simply disappears when sentiment is unavailable.
    sent_text = sentiment_text.where(sentiments_ok, "")

    def _photo_html(url: str) -> str:
        """Return the <img> tag for a listing photo, or '' when there is none."""
        if not url.startswith("http"):
            return ""
        w, h = cfg.MAP_TOOLTIP_PHOTO_WIDTH, cfg.MAP_TOOLTIP_PHOTO_HEIGHT
        # html.escape keeps a stray quote in a URL from breaking the markup.
        return (
            f"<img src='{html.escape(url, quote=True)}' width='{w}' height='{h}' "
            "loading='lazy' decoding='async' alt='Listing photo' "
            "style='border-radius:10px;object-fit:cover;display:block;"
            "margin-bottom:9px;background:#2A2530'/>"
        )

    photo_html = picture.apply(_photo_html)
    name_text = map_df["name"].astype(str).str.slice(0, 46).apply(html.escape)
    place_text = (
        map_df["neighbourhood"].astype(str).apply(html.escape) + ", "
        + map_df["neighbourhood_group"].astype(str).apply(html.escape)
    )
    room_text = map_df["room_type"].astype(str).apply(html.escape)
    price_text = (
        "$" + map_df["price"].round(0).astype(int).astype(str).str.replace(
            r"(\d)(?=(\d{3})+$)", r"\1,", regex=True)
        + " <span style='font-weight:500;opacity:.7'>/ night</span>"
    )
    review_text = map_df["number_of_reviews"].astype(int).astype(str) + " reviews"

    # Rating chip: the Kaggle file ships review_scores_rating as an empty
    # column (100% null), so the chip is only rendered when the column actually
    # holds values. An always-empty "No rating" pill would be noise. Rows with a
    # missing score get ``display:none`` rather than an empty pill.
    if "review_scores_rating" in map_df.columns:
        rating = pd.to_numeric(map_df["review_scores_rating"], errors="coerce")
    else:
        rating = pd.Series(float("nan"), index=map_df.index)
    rating_text = rating.map(lambda v: "" if pd.isna(v) else f"{v:.1f} / 5")
    rating_style = rating_text.map(
        lambda txt: "display:none" if txt == ""
        else "background:rgba(255,255,255,.10);"
             "border:1px solid rgba(255,255,255,.16);"
             "border-radius:999px;padding:1px 7px;font-weight:600"
    )

    # Inline styles only (the tooltip renders in its own overlay), matched to
    # the theme so the card looks like part of the product rather than a
    # default deck.gl popup. ``display:none`` on a wrapper collapses the whole
    # line, so the card never shows a half-empty row: neither chip present means
    # no pill row at all.
    pills_shown = sentiments_ok | rating_text.ne("")
    pills_flex = pills_shown.map(
        lambda ok: "display:flex;align-items:center;gap:6px;margin-bottom:5px;"
                   "flex-wrap:wrap" if ok else "display:none"
    )
    sent_span_style = sentiments_ok.map(
        lambda ok: ("background:rgba(255,56,92,.22);"
                    "border:1px solid rgba(255,56,92,.45);border-radius:999px;"
                    "padding:1px 7px;font-weight:600") if ok
        else "display:none"
    )

    map_df["tooltip"] = (
        "<div style=\"font-family:Inter,system-ui,-apple-system,sans-serif;"
        "font-size:12px;line-height:1.4;color:#FFFFFF;width:"
        f"{cfg.MAP_TOOLTIP_PHOTO_WIDTH + 26}px\">"
        + photo_html
        + "<div style='font-weight:750;font-size:13px;letter-spacing:-.01em;"
          "margin-bottom:3px;white-space:nowrap;overflow:hidden;"
          "text-overflow:ellipsis'>" + name_text + "</div>"
        + "<div style='color:#FFC6CF;font-weight:700;font-size:13px;"
          "margin-bottom:6px'>" + price_text + "</div>"
        + "<div style='" + pills_flex + "'>"
        + "<span style='" + rating_style + "'>" + rating_text + "</span>"
        + "<span style='" + sent_span_style + "'>"
        + sent_badge + " " + sent_text + "</span>"
        + "</div>"
        + "<div style='opacity:.82'>" + review_text + "</div>"
        + "<div style='opacity:.82'>" + place_text + "</div>"
        + "<div style='opacity:.82'>" + room_text + "</div>"
        + "</div>"
    )
    map_df["sentiment_known"] = sentiments_ok
    return map_df


def _empty(value: Any) -> bool:
    """Return True when a cell is missing, NaN or blank (tooltip safety)."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return not str(value).strip()


def listing_card_html(row: Any, thumb_width: int = 220) -> str:
    """
    Build one HTML listing card: photo, name, price, place and hover overlay.

    This is the same information the deck.gl tooltip shows, in plain HTML, so
    the hover preview also works where a WebGL canvas cannot be picked. Photos
    are requested as ~12 KB Unsplash thumbnails (see
    ``config.MAP_PHOTO_THUMB_QUERY``) with ``loading="lazy"``, so a grid costs
    one request per card that is actually scrolled into view.

    Parameters
    ----------
    row:
        One listing as a mapping (``Series`` or dict). Keys are read with
        ``.get`` so a partially populated row never raises.
    thumb_width:
        Intrinsic width of the <img>; CSS stretches it to the grid cell.

    Returns
    -------
    str
        HTML fragment; render with ``st.html`` or ``st.markdown``.
    """
    get = row.get if hasattr(row, "get") else (lambda _k, default=None: default)

    def _s(key: str, fallback: str = "") -> str:
        value = get(key, fallback)
        return fallback if _empty(value) else str(value)

    name = html.escape(_s("name", "Unnamed stay")[:60])
    place = html.escape(
        f"{_s('neighbourhood')}, {_s('neighbourhood_group')}".strip(", ")
    )
    room = html.escape(_s("room_type"))
    try:
        price = f"${float(get('price', None)):,.0f}"
    except (TypeError, ValueError):
        price = "Price n/a"
    try:
        reviews = f"{int(get('number_of_reviews', None)):,} reviews"
    except (TypeError, ValueError):
        reviews = "Reviews n/a"
    photo = _s("picture_url", "")
    thumb = (
        f"<img class='thumb' src='{html.escape(photo, quote=True)}' "
        f"width='{thumb_width}' loading='lazy' decoding='async' alt=''/>"
        if photo.startswith("http")
        else "<div class='thumb'></div>"
    )
    return (
        "<figure class='listing-card'>"
        f"{thumb}"
        "<figcaption class='body'>"
        f"<div class='name'>{name}</div>"
        f"<div class='meta'>{place} &middot; {room}</div>"
        f"<div class='meta'>{reviews}</div>"
        f"<div class='price'>{price} per night</div>"
        "</figcaption>"
        "<span class='preview'>Photo preview</span>"
        "</figure>"
    )


def photo_gallery_html(listings: pd.DataFrame, limit: int = 12) -> str:
    """
    Render up to ``limit`` listings as a responsive HTML card grid.

    Kept as one HTML fragment on purpose: ``st.image`` fetches every picture on
    the server, whereas these are browser-side, lazy-loaded images that reveal
    their overlay on hover exactly like the map tooltip.

    Parameters
    ----------
    listings:
        Filtered listings, already sorted however the caller wants them shown.
    limit:
        Maximum number of cards.

    Returns
    -------
    str
        HTML fragment; render with ``st.html``. Empty string when no rows.
    """
    if listings is None or len(listings) == 0:
        return ""
    shown = listings.head(limit)
    cards = "".join(listing_card_html(row) for _, row in shown.iterrows())
    return f"<div class='listing-grid'>{cards}</div>"



def make_deck(
    map_df: pd.DataFrame,
    lat: float = cfg.NYC_CENTER_LAT,
    lon: float = cfg.NYC_CENTER_LON,
    zoom: int = cfg.NYC_ZOOM,
    show_heatmap: bool = True,
    cluster: bool = True,
    dark: bool = True,
    satellite: bool = False,
) -> pdk.Deck:
    """
    Assemble the pydeck map with a heatmap layer plus clustered markers.

    Clustering strategy
    -------------------
    NYC holds ~48k listings, far too many individual markers to read. deck.gl's
    ``ColumnLayer`` aggregates nearby listings into one extruded hexagon whose
    height encodes how many reviews it covers -- the deck.gl-native equivalent
    of marker clustering, and it stays smooth at 48k points. Turning clustering
    off falls back to individual ``ScatterplotLayer`` dots.

    Basemaps
    --------
    ``satellite=True`` swaps the Carto vector style for the photographic Esri
    World Imagery style (:func:`satellite_style`). Markers and the heatmap keep
    the same theme colours in both modes, and the heatmap is dialled back on
    satellite so the imagery is not washed out.

    Parameters
    ----------
    map_df:
        Output of ``build_map_dataframe``.
    lat, lon:
        Map centre (moves when the user geocodes an area).
    zoom:
        Initial zoom level.
    show_heatmap:
        Draw the review-density heatmap layer.
    cluster:
        Aggregate markers into hexagon columns instead of individual dots.
    dark:
        Use the dark Carto basemap rather than the light one. Ignored when
        ``satellite`` is True.
    satellite:
        Use the Esri World Imagery satellite basemap.

    Returns
    -------
    pydeck.Deck
        Ready-to-render deck object.
    """
    layers: list[pdk.Layer] = []

    if show_heatmap and not map_df.empty:
        # Density of highly reviewed rentals: weighting by review count makes
        # hot spots highlight where the most-reviewed stays concentrate. On
        # satellite the heatmap is dialled back (config.MAP_SATELLITE_*) so the
        # imagery still reads through it.
        layers.append(
            pdk.Layer(
                "HeatmapLayer",
                data=map_df,
                get_position=["longitude", "latitude"],
                get_weight="number_of_reviews",
                radiusPixels=45,
                intensity=(cfg.MAP_SATELLITE_HEATMAP_INTENSITY if satellite
                           else cfg.MAP_VECTOR_HEATMAP_INTENSITY),
                threshold=0.06,
                opacity=(cfg.MAP_SATELLITE_HEATMAP_OPACITY if satellite
                         else cfg.MAP_VECTOR_HEATMAP_OPACITY),
                pickable=False,
            )
        )

    # On satellite imagery neither the coral nor the blue end of the review ramp
    # survives a bright rooftop or a green park on its own, so every marker gets
    # an opaque white outline (a halo). On the vector basemaps a softer outline
    # is enough. The halo is drawn as a slightly larger backdrop layer so it
    # reads as a ring rather than as a fat border.
    outline = [255, 255, 255, 235] if satellite else [255, 255, 255, 120]

    if satellite and not map_df.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                id="marker-halo",
                data=map_df,
                get_position=["longitude", "latitude"],
                get_fill_color=[255, 255, 255, 210],
                get_radius=120,
                radius_min_pixels=5,
                radius_max_pixels=22,
                opacity=0.55,
                pickable=False,
            )
        )

    if cluster and not map_df.empty:
        layers.append(
            pdk.Layer(
                "ColumnLayer",
                data=map_df,
                get_position=["longitude", "latitude"],
                get_fill_color="colour",
                radius=260,
                elevation_scale=8,
                elevation_range=[0, 2200],
                get_elevation="number_of_reviews",
                extruded=True,
                pickable=True,
                auto_highlight=True,
                opacity=0.9 if satellite else 0.85,
                # Outlined on both basemaps: the white edge is what stops the
                # columns bleeding into each other at a low zoom.
                stroked=True,
                get_line_color=outline,
                line_width_min_pixels=2 if satellite else 1,
            )
        )
    elif not map_df.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position=["longitude", "latitude"],
                get_fill_color="colour",
                get_radius=70,
                radius_min_pixels=4,
                radius_max_pixels=18,
                pickable=True,
                auto_highlight=True,
                stroked=True,
                get_line_color=outline,
                line_width_min_pixels=2 if satellite else 1,
            )
        )

    if satellite:
        map_style: Any = satellite_style()
    else:
        map_style = BASEMAP_DARK if dark else BASEMAP_LIGHT

    view_state = pdk.ViewState(
        latitude=lat, longitude=lon, zoom=zoom,
        pitch=cfg.NYC_PITCH, bearing=cfg.NYC_BEARING,
    )
    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=map_style,
        # pydeck only accepts a dict ``map_style`` together with the "mapbox"
        # provider value; it means "use the style document I supplied", not
        # "use Mapbox's servers". No Mapbox token is read or sent -- the style
        # above points at Esri's public tile service -- so this keeps the
        # satellite view key-free, unlike a native Mapbox satellite style.
        map_provider="mapbox",
        # ``html`` is evaluated by deck.gl against the hovered row, so the
        # ``{tooltip}`` placeholder is replaced with the photo card built in
        # ``build_map_dataframe``. deck.gl shows it on hover (not click), which
        # is exactly the interaction asked for; the style block only draws the
        # floating panel around it.
        tooltip={
            "html": "{tooltip}",
            "style": {
                "backgroundColor": "#12100F",
                "color": "#F4F0EC",
                "borderRadius": "14px",
                "padding": "10px 12px 12px",
                "border": "1px solid rgba(255,255,255,.10)",
                "boxShadow": "0 18px 44px rgba(0,0,0,.55)",
                "pointerEvents": "none",
                "maxWidth": f"{cfg.MAP_TOOLTIP_PHOTO_WIDTH + 26}px",
            },
        },
    )



# ===========================================================================
# SECTION 5 - TAB 1: INTERACTIVE MAP
# ===========================================================================
def card_row(name: str, spec: Any = 2) -> list[Any]:
    """
    Open an equal-height row of bordered cards and return the card contexts.

    Parameters
    ----------
    name:
        Unique row name, e.g. ``"distributions"``. Streamlit requires every
        ``key`` to be unique on the page, so each row passes its own.
    spec:
        Column widths, as for ``st.columns`` -- ``2`` for two equal columns or
        a list such as ``[1.1, 1]`` for uneven ones.

    Returns
    -------
    list[Any]
        Context managers, one per card. Use them as::

            left, right = card_row("work")
            with left:
                st.plotly_chart(...)

    Notes
    -----
    Same DOM as :func:`kpi_cards`: a keyed container (``.st-key-cardrow``)
    around the columns, then one bordered container per cell. The keyed
    wrapper is what the subgrid CSS rules need, so keep the call inside the
    design-system block's reach and do not nest it inside ``st.columns``.
    ``st.columns`` cannot take a ``key`` itself, which is why the container
    sits outside it.
    """
    with st.container(key=f"cardrow-{name}"):
        row_columns = st.columns(spec)
    # ``row_columns`` remain valid contexts after the ``with`` block closes, so
    # each cell can be turned into its own bordered container here.
    return [column.container(border=True) for column in row_columns]


def kpi_cards(name: str, specs: list[tuple[str, str, str]], columns: int = 4) -> None:
    """
    Render a row of equal-height metric cards.

    Parameters
    ----------
    name:
        Unique row name, e.g. ``"map-kpis"`` (Streamlit keys must be unique).
    specs:
        ``(icon, label, value)`` triples. ``icon`` is a Material icon string
        such as ``":material/reviews:"`` (Streamlit's native metric icon).
    columns:
        Number of cards per row; rows wrap automatically.

    Returns
    -------
    None

    Notes
    -----
    ``st.columns`` has no ``key`` argument, but ``st.container`` does, and a
    keyed container emits a ``.st-key-<key>`` class -- the hook the CSS subgrid
    rules in the design-system block use to stretch every card in the row to
    the height of the tallest one. Hence the nesting
    ``st.container(key="kpirow-<name>")`` > ``st.columns`` > bordered containers.
    """
    with st.container(key=f"kpirow-{name}"):
        row_columns = st.columns(columns)
        for column, (icon, label, value) in zip(row_columns, specs):
            with column, st.container(border=True):
                st.metric(label, value, icon=icon)


def render_kpis(filtered: pd.DataFrame) -> None:
    """
    Render the four headline metrics for the current selection as one card row.

    Each metric sits in its own bordered container inside a fixed ``key``-ed
    columns row (``key="kpirow"``), which is what the CSS subgrid rules in the
    design-system block use to make all four cards the same height.

    Parameters
    ----------
    filtered:
        The currently filtered top-reviewed listings.

    Returns
    -------
    None
    """
    if filtered.empty:
        st.info("No listings match these filters - widen the price range or the "
                "minimum review count.", icon=":material/filter_alt_off:")
        return

    summary = nlp.sentiment_summary(filtered)
    kpi_cards("map-kpis", [
        (":material/home_work:", "Listings Shown", f"{len(filtered):,}"),
        (":material/payments:", "Median Nightly Price",
         f"${filtered['price'].median():,.0f}"),
        (":material/reviews:", "Median Review Count",
         f"{filtered['number_of_reviews'].median():,.0f}"),
        (":material/sentiment_satisfied:", "Average Sentiment",
         "n/a" if summary["mean_sentiment"] is None else f"{summary['mean_sentiment']:+.2f}"),
    ])


def render_map_controls(top_reviewed: pd.DataFrame) -> dict[str, Any]:
    """
    Draw the map filter widgets and return their values.

    Parameters
    ----------
    top_reviewed:
        Cleaned top-reviewed listings (used to size the sliders).

    Returns
    -------
    dict[str, Any]
        ``price_range``, ``min_reviews``, ``boroughs``, ``room_types``,
        ``show_heatmap``, ``cluster``, ``basemap``, ``search``.
    """
    price_ceiling = int(min(cfg.PRICE_MAX, max(top_reviewed["price"].max(), 10)))
    review_ceiling = int(max(top_reviewed["number_of_reviews"].max(), 10))

    with st.container(border=True):
        st.markdown(
            card_title("tune", "Filter The Map",
                       "Narrow the listings every tab and the AI guide respond to."),
            unsafe_allow_html=True,
        )
        row1 = st.columns([2, 1.6, 1.3], vertical_alignment="bottom")
        with row1[0]:
            price_range = st.slider(
                "Nightly Price ($)",
                min_value=1,
                max_value=price_ceiling,
                value=(int(top_reviewed["price"].quantile(0.05)),
                       int(min(price_ceiling, top_reviewed["price"].quantile(0.90)))),
                step=5,
                help="Nightly price band, in US dollars.",
            )
        with row1[1]:
            min_reviews = st.slider(
                "Minimum Reviews",
                min_value=0,
                max_value=review_ceiling,
                value=int(cfg.TOP_REVIEWED_MIN_REVIEWS),
                step=5,
                help="Hide listings with fewer reviews than this.",
            )
        with row1[2]:
            basemap = st.selectbox(
                "Basemap",
                options=list(cfg.MAP_BASEMAP_MODES),
                index=list(cfg.MAP_BASEMAP_MODES).index(cfg.MAP_DEFAULT_BASEMAP),
                icon=":material/layers:",
                help="Satellite uses Esri World Imagery (photographic tiles); "
                     "Dark and Light use the CARTO vector styles.",
            )

        row2 = st.columns([1.5, 1.5, 1, 1], vertical_alignment="bottom")
        with row2[0]:
            boroughs = st.multiselect(
                "Borough", options=list(cfg.BOROUGHS), default=[],
                placeholder="All boroughs",
            )
        with row2[1]:
            room_types = st.multiselect(
                "Room Type", options=list(cfg.ROOM_TYPES), default=[],
                placeholder="All room types",
            )
        with row2[2]:
            show_heatmap = st.toggle(
                "Heatmap", value=True,
                help="Weight each listing by its review count.",
            )
        with row2[3]:
            cluster = st.toggle(
                "Cluster Markers", value=True,
                help="Aggregate nearby listings into hexagonal columns.",
            )

        search = st.text_input(
            "Centre On A Neighbourhood",
            placeholder="e.g. Williamsburg, Harlem, Astoria, Mott Haven",
            help="Resolved with OpenStreetMap Nominatim geocoding.",
            icon=":material/search:",
        )
    return {
        "price_range": price_range,
        "min_reviews": min_reviews,
        "boroughs": boroughs,
        "room_types": room_types,
        "show_heatmap": show_heatmap,
        "cluster": cluster,
        "basemap": basemap,
        "search": search,
    }


def _map_legend(basemap: str) -> str:
    """
    Build the legend strip printed under the map.

    Parameters
    ----------
    basemap:
        Selected basemap label (``"Satellite"``, ``"Dark"`` or ``"Light"``).

    Returns
    -------
    str
        HTML for the legend; render with ``st.html``.
    """
    source = (ESRI_ATTRIBUTION if basemap == cfg.MAP_BASEMAP_SATELLITE
              else "CARTO basemaps / OpenStreetMap")
    return (
        "<div class='map-legend'>"
        "<span><span class='gradient'></span>Fewer to more reviews</span>"
        "<span><span class='swatch' style='background:#FF385C'></span>"
        "Column height = review volume</span>"
        "<span>Hover a marker for the listing photo</span>"
        f"<span style='margin-left:auto;opacity:.65'>"
        f"Basemap: {html.escape(basemap)} &middot; {html.escape(source)}"
        "</span>"
        "</div>"
    )




def render_map_tab(top_reviewed: pd.DataFrame) -> pd.DataFrame:
    """
    Render the "Interactive Map" tab and return the filtered frame.

    The returned frame is reused by the AI tab so the concierge describes
    exactly what the user sees on the map.

    Parameters
    ----------
    top_reviewed:
        Cleaned, top-reviewed listings with sentiment attached.

    Returns
    -------
    pandas.DataFrame
        The filtered listings (empty when nothing matches).
    """
    st.markdown(
        f"### {material_icon('map', 22)} Where Are The Most-Reviewed Stays?",
        unsafe_allow_html=True,
    )
    st.caption(
        "Marker height and colour encode review volume. **Hover any marker** for "
        "a hover card with that listing's photo, price, rating and review "
        "sentiment. The satellite basemap is the default; switch to Dark or "
        "Light for the CARTO vector styles."
    )

    if top_reviewed.empty:
        st.error("No top-reviewed listings are available to map.",
                 icon=":material/error:")
        return pd.DataFrame()

    controls = render_map_controls(top_reviewed)
    filtered = filter_listings(
        top_reviewed,
        price_range=controls["price_range"],
        min_reviews=controls["min_reviews"],
        boroughs=controls["boroughs"],
        room_types=controls["room_types"],
    )

    # --- map centre (geocoded search) -------------------------------------
    centre_lat, centre_lon, zoom = cfg.NYC_CENTER_LAT, cfg.NYC_CENTER_LON, cfg.NYC_ZOOM
    search = controls["search"].strip()
    if search:
        try:
            with st.spinner(f"Geocoding '{search}'..."):
                hit = geocode_cached(search)
            if hit is None:
                st.warning(
                    f"Could not find '{search}' in New York City. Try a borough "
                    "or a well-known neighbourhood such as 'Bushwick'.",
                    icon=":material/location_off:",
                )
            else:
                centre_lat, centre_lon, zoom = float(hit[0]), float(hit[1]), 13
                st.success(f"Centred on {hit[2]}.", icon=":material/location_on:")
        except Exception as exc:
            st.warning(f"Geocoding failed ({exc}). Showing the default NYC view.")

    st.space("small")
    render_kpis(filtered)
    if filtered.empty:
        return filtered

    # --- map --------------------------------------------------------------
    # Cap plotted markers for browser responsiveness; the KPIs above and the AI
    # report still use the complete filtered set.
    max_markers = 6000
    plot_df = filtered
    if len(plot_df) > max_markers:
        plot_df = plot_df.sample(max_markers, random_state=42)
        st.info(
            f"Plotting a random sample of {max_markers:,} of {len(filtered):,} "
            "matching listings for performance.",
            icon=":material/speed:",
        )

    deck = make_deck(
        build_map_dataframe(plot_df),
        lat=centre_lat, lon=centre_lon, zoom=zoom,
        show_heatmap=controls["show_heatmap"],
        cluster=controls["cluster"],
        dark=controls["basemap"] == cfg.MAP_BASEMAP_DARK,
        satellite=controls["basemap"] == cfg.MAP_BASEMAP_SATELLITE,
    )
    st.space("small")
    with st.container(key="mapframe"):
        st.pydeck_chart(deck, width="stretch", height=640)
    st.html(_map_legend(controls["basemap"]))

    # --- hover preview gallery --------------------------------------------
    # The deck.gl tooltip is the primary hover preview. This grid repeats the
    # same photo card in plain HTML: it is the fallback for browsers or
    # machines where WebGL picking is unavailable, and it doubles as a browsable
    # strip of the top stays the current filters are pointing at.
    with st.expander("Listing Photo Preview", icon=":material/photo_library:",
                     expanded=False):
        top_by_reviews = filtered.sort_values(
            "number_of_reviews", ascending=False)
        st.caption(
            "The same photo cards the map shows on hover, for the 12 "
            "most-reviewed stays in the current selection. Hover a card to "
            "lift it."
        )
        gallery = photo_gallery_html(top_by_reviews, limit=12)
        if gallery:
            st.html(gallery)
        else:
            st.info("No listing photos are available for this selection.",
                    icon=":material/image_not_supported:")

    with st.expander("Top-Reviewed Listings Table", icon=":material/table_view:"):
        columns = [c for c in (
            "name", "neighbourhood_group", "neighbourhood", "room_type",
            "price", "number_of_reviews", "avg_sentiment", "sentiment_label",
        ) if c in filtered.columns]
        st.dataframe(
            filtered[columns].sort_values("number_of_reviews", ascending=False).head(300),
            width="stretch", hide_index=True,
        )
        st.download_button(
            "Download Filtered Listings",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="nyc_top_reviewed.csv",
            mime="text/csv",
            icon=":material/download:",
        )
    return filtered



# ===========================================================================
# SECTION 6 - TAB 2: EDA INSIGHTS
# ===========================================================================
def render_eda_tab(
    raw_listings: pd.DataFrame,
    raw_reviews: pd.DataFrame,
    clean: pd.DataFrame,
    top_reviewed: pd.DataFrame,
    audit: dict[str, Any],
) -> None:
    """
    Render the "EDA Insights" tab: data quality, top-reviewed definition, charts.

    Parameters
    ----------
    raw_listings, raw_reviews:
        Uncleaned data, used for the quality audit.
    clean:
        Cleaned listings (all rows that survived cleaning).
    top_reviewed:
        The subset selected as "top reviewed".
    audit:
        Audit dict returned by ``eda.clean_listings``.

    Returns
    -------
    None
    """
    st.markdown(
        f"### {material_icon('insights', 22)} Exploratory Data Analysis",
        unsafe_allow_html=True,
    )
    st.caption("All figures use the cleaned listings table unless stated otherwise.")

    # --- 1. data quality --------------------------------------------------
    st.markdown(f"#### {material_icon('cleaning', 18)} Data Quality And Preprocessing",
                unsafe_allow_html=True)
    outliers = audit.get("outliers", {}) or {}
    quality = da.data_quality_report(raw_listings, raw_reviews)

    kpi_cards("quality-kpis", [
        (":material/inventory_2:", "Raw Listings", f"{quality['listings_rows']:,}"),
        (":material/done_all:", "After Cleaning", f"{int(audit.get('rows_out', 0)):,}"),
        (":material/delete_sweep:", "Outlier Rows Removed",
         f"{int(outliers.get('rows_removed', 0)):,}"),
        (":material/workspace_premium:", "Top-Reviewed", f"{len(top_reviewed):,}"),
    ])

    st.space("small")
    left, right = card_row("quality", 2)
    with left:
        st.markdown(card_title("table", "Missing Values", "Raw listings table"),
                    unsafe_allow_html=True)
        report = eda.missing_value_report(raw_listings)
        missing_only = report[report["missing_count"] > 0]
        st.dataframe(
            missing_only if not missing_only.empty
            else pd.DataFrame({"Status": ["No missing values found"]}),
            width="stretch", hide_index=True, height=230,
        )
    with right:
        st.markdown(card_title("filter_alt", "Outlier Policy",
                               "IQR fence plus business ceiling"),
                    unsafe_allow_html=True)
        st.write(
            f"- Price must be **> \\\\$0** and **<= \\\\${outliers.get('applied_ceiling', 'n/a')}** "
            f"(Q3 + 1.5 x IQR = \\\\${outliers.get('upper_fence', 'n/a')}, capped at the "
            f"\\\\${outliers.get('price_max', cfg.PRICE_MAX):,.0f} business ceiling)."
        )
        st.write(f"- Price outliers removed: **{int(outliers.get('rows_removed', 0)):,}**")
        st.write(f"- Missing-coordinate rows removed: **{quality['missing_coordinates']:,}**")
        st.write(f"- Duplicate listing ids removed: **{audit.get('duplicates_removed', 0):,}**")

    st.space("medium")


    # --- 2. "top reviewed" definition -------------------------------------
    st.markdown(f"#### {material_icon('premium', 18)} How \\\"Top Reviewed\\\" Is Defined",
                unsafe_allow_html=True)
    st.markdown(
        f"A listing qualifies when it has at least **{cfg.TOP_REVIEWED_MIN_REVIEWS} "
        f"reviews** and sits in the top **{int((1 - cfg.TOP_REVIEWED_QUANTILE) * 100)}% "
        "by review volume** (falling back to `review_scores_rating` when the dataset "
        "provides it). This filters out brand-new listings whose single 5-star review "
        "would otherwise outrank proven stays."
    )
    if not top_reviewed.empty:
        kpi_cards("top-kpis", [
            (":material/reviews:", "Median Reviews",
             f"{top_reviewed['number_of_reviews'].median():,.0f}"),
            (":material/payments:", "Median Price",
             f"${top_reviewed['price'].median():,.0f}"),
            (":material/location_city:", "Boroughs Covered",
             f"{top_reviewed['neighbourhood_group'].nunique()}"),
            (":material/sentiment_satisfied:", "Average Sentiment",
             f"{top_reviewed['avg_sentiment'].mean():+.2f}"
             if "avg_sentiment" in top_reviewed.columns
             and top_reviewed["avg_sentiment"].notna().any() else "n/a"),
        ])

    st.space("medium")

    # --- 3. distributions -------------------------------------------------
    st.markdown(f"#### {material_icon('bar_chart', 18)} Distributions",
                unsafe_allow_html=True)
    left, right = card_row("distributions", 2)
    with left:
        st.plotly_chart(eda.plot_price_distribution(clean), width="stretch")
    with right:
        st.plotly_chart(eda.plot_reviews_distribution(clean), width="stretch")

    st.space("medium")

    # --- 4. borough volume (the rubric's bar chart) -----------------------
    st.markdown(f"#### {material_icon('map', 18)} Volume By Borough",
                unsafe_allow_html=True)
    left, right = card_row("boroughs", 2)
    with left:
        st.plotly_chart(eda.plot_top_reviewed_by_borough(top_reviewed), width="stretch")
    with right:
        st.plotly_chart(eda.plot_price_vs_reviews(clean), width="stretch")

    st.space("medium")

    # --- 5. correlations --------------------------------------------------
    st.markdown(f"#### {material_icon('hub', 18)} Correlation Analysis",
                unsafe_allow_html=True)
    left, right = card_row("correlations", [1.1, 1])
    with left:
        st.plotly_chart(eda.plot_correlation_heatmap(clean), width="stretch")
    with right:
        st.markdown(card_title("table", "Correlation With Number Of Reviews"),
                    unsafe_allow_html=True)
        correlations = eda.review_correlations(clean)
        if correlations.empty:
            st.info("Not enough numeric columns to correlate.",
                    icon=":material/info:")
        else:
            table = correlations.rename("correlation").reset_index()
            table.columns = ["Variable", "Correlation"]
            st.dataframe(table.round(3), width="stretch", hide_index=True)

    st.space("medium")

    # --- 6. room type + sentiment ----------------------------------------
    left, right = card_row("mix", 2)
    with left:
        st.plotly_chart(eda.plot_room_type_mix(clean), width="stretch")
    with right:
        if "avg_sentiment" in clean.columns and clean["avg_sentiment"].notna().any():
            st.plotly_chart(eda.plot_sentiment_by_borough(clean), width="stretch")
        else:
            st.info("Sentiment scores are not available for this dataset.",
                    icon=":material/info:")

    st.space("medium")
    with st.expander("Top-Reviewed Listings Preview", icon=":material/table_chart:"):
        st.caption("First 200 rows of the top-reviewed selection.")
        st.dataframe(top_reviewed.head(200), width="stretch", hide_index=True)



# ===========================================================================
# SECTION 7 - TAB 3: AI TRAVEL GUIDE
# ===========================================================================
def render_ai_tab(
    filtered: pd.DataFrame,
    top_reviewed: pd.DataFrame,
    scored_reviews: pd.DataFrame,
    llm_provider: str,
) -> None:
    """
    Render the "AI Travel Guide" tab.

    Workflow
    --------
    1. Build an aggregate payload from the listings currently on the map (names,
       neighbourhoods, prices, review counts, sentiment) -- never raw rows.
    2. On button press, send it to the chosen LLM and render the markdown guide.
    3. Offer a follow-up question box that reuses the same payload.

    Parameters
    ----------
    filtered:
        Listings currently visible on the map (may be empty).
    top_reviewed:
        Full top-reviewed table, used when the map selection is empty.
    scored_reviews:
        Per-review sentiment table, used for keyword extraction.
    llm_provider:
        ``"auto"``, ``"openai"``, ``"gemini"`` or ``"offline"``.

    Returns
    -------
    None
    """
    st.markdown(
        f"### {material_icon('concierge', 22)} NYC Travel Concierge",
        unsafe_allow_html=True,
    )
    st.caption(
        "The concierge reads a summary of your current map filters and writes a "
        "guide in plain English. Only aggregated statistics are sent, never raw "
        "review text."
    )

    working = filtered if filtered is not None and not filtered.empty else top_reviewed
    if working is None or working.empty:
        st.error("There is no data to summarise yet.", icon=":material/error:")
        return

    if filtered is None or filtered.empty:
        st.warning("The map has no matching listings, so the full top-reviewed set "
                   "is being summarised instead.", icon=":material/filter_alt_off:")

    # --- build the payload ------------------------------------------------
    try:
        keywords = nlp.extract_keywords(scored_reviews) if not scored_reviews.empty else {}
        payload = llm.build_context_payload(working, keywords=keywords)
    except Exception as exc:
        st.error(f"Could not build the summary payload: {exc}",
                 icon=":material/error:")
        return

    with st.expander("What The AI Receives", icon=":material/data_object:"):
        st.caption("Aggregated JSON payload sent with every request.")
        st.json(payload, expanded=2)

    generate_col, status_col = st.columns([1, 2], vertical_alignment="center")
    with generate_col:
        generate = st.button("Generate Travel Guide", type="primary", width="stretch",
                             icon=":material/auto_awesome:")
    with status_col:
        st.caption(f"Summarising {len(working):,} listings using the "
                   f"**{llm_provider}** provider.")

    if generate:
        try:
            with st.spinner("The concierge is writing your NYC guide..."):
                report = llm.generate_travel_report(payload, provider=llm_provider)
            st.session_state["travel_report"] = report
        except Exception as exc:
            st.error(f"The LLM call failed: {exc}", icon=":material/error:")

    report = st.session_state.get("travel_report")
    if report is None:
        st.info("Press **Generate Travel Guide** to summarise your current filters.",
                icon=":material/info:")
        return

    if report.warning:
        st.warning(report.warning, icon=":material/warning:")

    with st.container(border=True):
        st.markdown(report.text)

    st.space("small")
    kpi_cards("ai-meta", [
        (":material/hub:", "Provider", report.provider),
        (":material/memory:", "Model", report.model),
        (":material/timer:", "Latency", f"{report.latency_seconds:.2f}s"),
        (":material/toll:", "Tokens", f"{report.tokens:,}" if report.tokens else "n/a"),
    ])

    st.space("small")
    st.download_button(
        "Download The Guide",
        data=report.text.encode("utf-8"),
        file_name="nyc_travel_guide.md",
        mime="text/markdown",
        icon=":material/download:",
    )

    st.space("medium")
    st.markdown(f"#### {material_icon('concierge', 18)} Ask A Follow-Up",
                unsafe_allow_html=True)
    question = st.text_input(
        "Your question",
        placeholder="e.g. Which borough is best for a $150 budget with the best reviews?",
        key="followup_question",
        icon=":material/chat:",
    )
    if st.button("Ask The Concierge", icon=":material/send:") and question.strip():
        try:
            with st.spinner("Thinking..."):
                answer = llm.answer_followup(question, payload, provider=llm_provider)
            if answer.warning:
                st.caption(answer.warning)
            st.markdown(answer.text)
        except Exception as exc:
            st.error(f"Follow-up failed: {exc}")



# ===========================================================================
# SECTION 8 - MAIN / ROUTING
# ===========================================================================
def main() -> None:
    """
    Application entry point: load data, build the three tabs, route each one.

    Returns
    -------
    None
    """
    settings = render_sidebar()

    # --- 1. load raw data -------------------------------------------------
    with st.spinner("Loading NYC Airbnb data (first run can take a minute)..."):
        raw_listings, raw_reviews, used_synthetic, load_error = load_raw_data(
            force_download=settings["force_download"],
            reviews_limit=int(settings["reviews_limit"]),
        )

    render_header(used_synthetic or bool(load_error))

    if load_error:
        st.error(
            "There was a problem loading data; the app will continue with "
            f"whatever is available.\n\n`{load_error}`"
        )
    if raw_listings is None or raw_listings.empty:
        st.error("No listings could be loaded. Add Kaggle credentials (README "
                 "step 3), or keep running on the offline demo data.")
        st.stop()

    # --- 2. Phase 2: clean + select top reviewed --------------------------
    try:
        with st.spinner("Cleaning data and selecting top-reviewed listings..."):
            clean, audit = build_clean_listings(raw_listings)
            top_reviewed = eda.select_top_reviewed(clean)
    except Exception as exc:
        st.error(f"The cleaning step failed: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        st.stop()

    # --- 3. Phase 3: sentiment -------------------------------------------
    with st.spinner(f"Scoring reviews with {settings['engine'].upper()}..."):
        scored_reviews, per_listing, sentiment_error = build_sentiment(
            raw_reviews, settings["engine"], int(settings["reviews_limit"])
        )
    if sentiment_error:
        st.warning(
            f"Sentiment analysis is unavailable ({sentiment_error}). The map and "
            "EDA tabs still work; sentiment columns will be empty."
        )

    clean = nlp.attach_sentiment(clean, per_listing)
    top_reviewed = nlp.attach_sentiment(top_reviewed, per_listing)

    # --- 3b. hover-photo URLs for the map --------------------------------
    # Attached once here rather than inside the map tab so the same column is
    # available to any future view; a failure is cosmetic (the tooltip simply
    # shows no image), so it must never stop the app.
    try:
        top_reviewed = top_reviewed.assign(
            picture_url=da.listing_photo_urls(top_reviewed)
        )
    except Exception as exc:
        st.warning(f"Listing photos are unavailable ({exc}).",
                   icon=":material/image_not_supported:")

    # --- 4. tabs ----------------------------------------------------------
    map_tab, eda_tab, ai_tab = st.tabs([
        ":material/map: Map",
        ":material/insights: EDA Insights",
        ":material/smart_toy: AI Travel Guide",
    ])

    filtered = pd.DataFrame()
    with map_tab:
        try:
            filtered = render_map_tab(top_reviewed)
        except Exception as exc:
            st.error(f"The map could not be rendered: {exc}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())

    with eda_tab:
        try:
            render_eda_tab(raw_listings, raw_reviews, clean, top_reviewed, audit)
        except Exception as exc:
            st.error(f"The EDA tab hit an error: {exc}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())

    with ai_tab:
        try:
            render_ai_tab(filtered, top_reviewed, scored_reviews, settings["llm_provider"])
        except Exception as exc:
            st.error(f"The AI tab hit an error: {exc}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())

    st.space("large")
    st.html(
        "<div style='border-top:1px solid var(--border-color);padding-top:1rem;"
        "font-size:.78rem;color:var(--text-color);opacity:.62;line-height:1.6'>"
        "Built with Streamlit, PyDeck, Plotly, VADER/TextBlob and an LLM API. "
        "Data: Kaggle NYC Airbnb Open Data + Inside Airbnb reviews. "
        "Geocoding: OpenStreetMap Nominatim. Photos: Unsplash. "
        "Basemaps: Esri World Imagery, CARTO.</div>"
    )


if __name__ == "__main__":
    main()

