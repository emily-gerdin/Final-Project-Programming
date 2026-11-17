"""
config.py
=========

Central configuration for the **Top Reviewed New York City Travel Planner**.

Why this file exists
--------------------
Every other module (``data_acquisition.py``, ``eda.py``, ``nlp_sentiment.py``,
``llm_agent.py``, ``app.py``) needs the same values: the cache location, the
dataset identifiers, the NYC region vocabulary and the cleaning thresholds.
Declaring them once means a change is a one-line edit in a single file.

This module only *declares* values -- it performs no work -- so it is safe to
import from anywhere, including a Jupyter notebook.

Contents
--------
1. Filesystem paths
2. Data source identifiers (Kaggle listings + reviews)
3. NYC geography: map centre, boroughs, derived "region" vocabulary
4. Cleaning / "top reviewed" thresholds
5. API key resolution (``.env`` -> environment -> ``st.secrets``)

Scope note
----------
This project covers **New York City**, not New York State. Inside Airbnb (the
upstream publisher of the data) only releases NYC, Albany and Rochester for New
York, so a statewide map is not reproducible from a single Kaggle download.
See README section "Scope and data availability".
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. FILESYSTEM PATHS
# ---------------------------------------------------------------------------
# ``Path(__file__).resolve().parent`` is the folder holding this file, so the
# app behaves identically no matter which directory Streamlit is launched from.
BASE_DIR: Path = Path(__file__).resolve().parent

DATA_DIR: Path = BASE_DIR / "data"            # raw downloads
PROCESSED_DIR: Path = BASE_DIR / "processed"  # cached cleaned tables
ASSETS_DIR: Path = BASE_DIR / "assets"        # hero image, logo

for _folder in (DATA_DIR, PROCESSED_DIR, ASSETS_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

# Normalised local file names (written by data_acquisition.py)
LOCAL_LISTINGS_CSV: Path = DATA_DIR / "listings.csv"
LOCAL_REVIEWS_CSV: Path = DATA_DIR / "reviews.csv"

# Cached pipeline artefacts
TOP_REVIEWED_PARQUET: Path = PROCESSED_DIR / "top_reviewed.parquet"
LISTING_SENTIMENT_CSV: Path = PROCESSED_DIR / "listing_sentiment.csv"

# ---------------------------------------------------------------------------
# 2. DATA SOURCE IDENTIFIERS
# ---------------------------------------------------------------------------
# PRIMARY SOURCE -- Inside Airbnb, New York City.
#
#   https://insideairbnb.com/new-york-city/
#   https://insideairbnb.com/get-the-data/
#
# Inside Airbnb publishes one folder per city per quarterly snapshot. The URL
# shape is
#   https://data.insideairbnb.com/united-states/ny/new-york-city/<date>/data/<file>
#
# Why this replaced the Kaggle "New York City Airbnb Open Data" (AB_NYC_2019.csv)
# snapshot:
#   * it is current (the Kaggle file is frozen at 2019),
#   * it ships the real ``picture_url`` column, so the map hover cards show the
#     actual listing photo instead of a stand-in Unsplash image,
#   * ``review_scores_rating`` is actually populated (the Kaggle file ships the
#     column 100% null),
#   * listings and reviews come from one consistent snapshot, so ``listing_id``
#     joins are exact by construction rather than by luck,
#   * it needs no credentials at all.
#
# Schema notes handled in ``data_acquisition._validate_listings``:
#   * the borough/neighbourhood columns are the ``*_cleansed`` variants
#     (``neighbourhood_group_cleansed`` / ``neighbourhood_cleansed``),
#   * ``price`` is a formatted string ("$164.55") and is NULL for roughly a third
#     of rows; ``price_quote_price_per_night`` is used as a backup before a row
#     is discarded,
#   * there are 90 columns rather than the Kaggle file's 16, including several
#     ``estimated_*`` metrics that this app does not use.
#
# Data is CC BY 4.0: Inside Airbnb must be credited wherever the data is shown
# (see ``INSIDE_AIRBNB_ATTRIBUTION`` and the app footer).
INSIDE_AIRBNB_CITY_PATH: str = "united-states/ny/new-york-city"
INSIDE_AIRBNB_SNAPSHOT: str = "2026-09-14"
INSIDE_AIRBNB_LICENSE: str = "CC BY 4.0"
INSIDE_AIRBNB_ATTRIBUTION: str = (
    "Data: Inside Airbnb (insideairbnb.com), CC BY 4.0"
)

# Dated snapshots resolve; the bare "latest" alias currently returns HTTP 403
# from Inside Airbnb's CDN, so the pinned snapshot is tried first and "latest"
# is kept last as a self-healing fallback if they unblock it.
INSIDE_AIRBNB_LISTINGS_URLS: tuple[str, ...] = (
    f"https://data.insideairbnb.com/{INSIDE_AIRBNB_CITY_PATH}/{INSIDE_AIRBNB_SNAPSHOT}/data/listings.csv.gz",
    f"https://data.insideairbnb.com/{INSIDE_AIRBNB_CITY_PATH}/latest/data/listings.csv.gz",
)
INSIDE_AIRBNB_REVIEWS_URLS: tuple[str, ...] = (
    f"https://data.insideairbnb.com/{INSIDE_AIRBNB_CITY_PATH}/{INSIDE_AIRBNB_SNAPSHOT}/data/reviews.csv.gz",
    f"https://data.insideairbnb.com/{INSIDE_AIRBNB_CITY_PATH}/latest/data/reviews.csv.gz",
)

# OPTIONAL FALLBACK -- the original Kaggle dataset.
#
# Kept only as a last-resort mirror for machines that cannot reach
# data.insideairbnb.com. It is never tried unless Kaggle credentials are
# configured, and the app runs perfectly without them. Its schema is the old
# 16-column one (no photo column, empty ratings), which the same validator
# normalises onto the canonical schema.
LISTINGS_DATASET: str = "dgomonov/new-york-city-airbnb-open-data"
LISTINGS_FILENAME: str = "AB_NYC_2019.csv"
LISTINGS_LICENSE: str = "CC0: Public Domain"

# Kaggle datasets that ship a reviews table. Empty by default: the previously
# configured candidates are dead (one ships no reviews.csv, the other 403s), and
# Inside Airbnb is now the primary source anyway.
REVIEWS_KAGGLE_CANDIDATES: tuple[tuple[str, str], ...] = ()

# ---------------------------------------------------------------------------
# 3. NYC GEOGRAPHY
# ---------------------------------------------------------------------------
NYC_CENTER_LAT: float = 40.7128
NYC_CENTER_LON: float = -74.0060
NYC_ZOOM: int = 10
NYC_PITCH: int = 45          # deck.gl tilt
NYC_BEARING: int = 0

# Map presentation. ``MAP_BASEMAP_MODES`` is the single source of truth for the
# basemap selectbox and ``MAP_DEFAULT_BASEMAP`` for its default, so the widget,
# the deck style and the legend can never disagree about what "Satellite" means.
MAP_BASEMAP_SATELLITE: str = "Satellite"
MAP_BASEMAP_DARK: str = "Dark"
MAP_BASEMAP_LIGHT: str = "Light"
MAP_BASEMAP_MODES: tuple[str, ...] = (
    MAP_BASEMAP_SATELLITE,
    MAP_BASEMAP_DARK,
    MAP_BASEMAP_LIGHT,
)
MAP_DEFAULT_BASEMAP: str = MAP_BASEMAP_SATELLITE

# Satellite tiles are far brighter than a vector basemap, so the overlay colours
# need a different treatment: a thin halo behind every marker (no marker --
# coral or blue -- survives on photographic greens and beach greys without it),
# a light heatmap (so the imagery is not washed out) and a faint dark scrim
# under the tiles to keep the heatmap and tooltips legible.
MAP_SATELLITE_HEATMAP_OPACITY: float = 0.38
MAP_SATELLITE_HEATMAP_INTENSITY: float = 0.85
MAP_VECTOR_HEATMAP_OPACITY: float = 0.65
MAP_VECTOR_HEATMAP_INTENSITY: float = 1.10
MAP_SATELLITE_SCRIM: tuple[float, ...] = (18, 16, 15, 38)   # RGBA, 0-255

# Hover card: rendered as an HTML overlay inside the deck canvas, so the photo
# is fetched from the Unsplash CDN as a ~12 KB thumbnail (w=) rather than the
# original. Bigger thumbs would make hovering feel laggy on slow connections.
MAP_TOOLTIP_PHOTO_WIDTH: int = 232
MAP_TOOLTIP_PHOTO_HEIGHT: int = 140
MAP_PHOTO_THUMB_QUERY: str = "?w=240&q=60&auto=format&fit=crop"

# The five boroughs as they appear in ``neighbourhood_group``.
BOROUGHS: tuple[str, ...] = (
    "Manhattan",
    "Brooklyn",
    "Queens",
    "Bronx",
    "Staten Island",
)

# Room types advertised on Airbnb NYC.
ROOM_TYPES: tuple[str, ...] = (
    "Entire home/apt",
    "Private room",
    "Shared room",
)

# Since a genuine statewide dataset does not exist, the map is given the same
# analytical shape by grouping NYC into *traveller regions*: Manhattan and
# Brooklyn as their own markets, and the remaining boroughs as broader areas.
# ``neighbourhood_group`` maps 1:1 onto the five boroughs, so "region" is just
# the borough with Queens/Bronx/Staten Island grouped as "Outer Boroughs".
REGION_ORDER: tuple[str, ...] = (*BOROUGHS, "Outer Boroughs")

# Well-known neighbourhoods used to seed the "search an area" box. Anything a
# user types that is not listed here is resolved live through Nominatim.
NEIGHBOURHOOD_PRESETS: dict[str, tuple[float, float]] = {
    "Manhattan": (40.7831, -73.9712),
    "Brooklyn": (40.6782, -73.9442),
    "Queens": (40.7282, -73.7949),
    "Bronx": (40.8448, -73.8648),
    "Staten Island": (40.5795, -74.1502),
    "Harlem": (40.8116, -73.9465),
    "Williamsburg": (40.7081, -73.9571),
    "Bushwick": (40.6944, -73.9213),
    "Bedford-Stuyvesant": (40.6872, -73.9418),
    "Park Slope": (40.6710, -73.9814),
    "Greenpoint": (40.7304, -73.9546),
    "Astoria": (40.7644, -73.9235),
    "Long Island City": (40.7447, -73.9485),
    "Flushing": (40.7675, -73.8331),
    "Jamaica": (40.7027, -73.7891),
    "Forest Hills": (40.7196, -73.8448),
    "SoHo": (40.7233, -74.0030),
    "Chelsea": (40.7465, -74.0014),
    "Greenwich Village": (40.7336, -74.0027),
    "East Village": (40.7265, -73.9815),
    "Upper East Side": (40.7736, -73.9566),
    "Upper West Side": (40.7870, -73.9754),
    "Midtown": (40.7549, -73.9840),
    "Hell's Kitchen": (40.7638, -73.9918),
    "Financial District": (40.7075, -74.0090),
    "Riverdale": (40.8900, -73.9126),
    "Mott Haven": (40.8090, -73.9228),
    "St. George": (40.6437, -74.0779),
}

# ---------------------------------------------------------------------------
# 4. CLEANING + "TOP REVIEWED" THRESHOLDS
# ---------------------------------------------------------------------------
# Outlier treatment (rubric: remove $0 prices and extreme outliers via IQR).
PRICE_MIN: float = 0.0        # exclusive lower bound -> $0 rows are dropped
PRICE_MAX: float = 1000.0     # business ceiling, $ per night
IQR_MULTIPLIER: float = 1.5   # Tukey fence: Q3 + k * IQR
MIN_NIGHTS_MAX: int = 365     # listings demanding > 1 year are junk rows

# "Top reviewed" definition (rubric: highest number_of_reviews OR
# highest review_scores_rating). Both thresholds are applied by eda.py:
TOP_REVIEWED_MIN_REVIEWS: int = 50     # enough social proof to be meaningful
TOP_REVIEWED_MIN_RATING: float = 4.5   # out of 5; typical "excellent" cut-off
TOP_REVIEWED_QUANTILE: float = 0.90    # keep the top 10% by chosen metric
TOP_REVIEWED_LIMIT: int = 500          # rows fed to the map/LLM for clarity

# Sentiment
MIN_REVIEWS_FOR_SENTIMENT: int = 1     # need >=1 scored review per listing
SENTIMENT_SAMPLE_SIZE: int = 200_000   # reviews scored per run (speed/size trade-off)

# --- Chart palette shared by Plotly figures and the app ----------------------
# Monochrome-adjacent: a platinum ink ramp for the boroughs, so the charts sit
# inside the editorial palette (off-white -> charcoal) instead of clashing with
# it. Mirrors ``chartCategoricalColors`` in .streamlit/config.toml.
REGION_COLORS: dict[str, str] = {
    "Manhattan": "#F5F5F5",
    "Brooklyn": "#D9D9D9",
    "Queens": "#B0B0B0",
    "Bronx": "#8A8A8A",
    "Staten Island": "#C9B79C",
    "Outer Boroughs": "#6A6A6A",
}

# --- UI palette (mirror of .streamlit/config.toml) ---------------------------
# Streamlit's theme lives in TOML; the few places that need the same colours in
# Python are the pydeck basemap style (below) and the inline Material icons.
# Keep the two files in sync -- change both when the palette changes.
UI_PALETTE: dict[str, str] = {
    "primary": "#F5F5F5",          # off-white: the only "accent" on paper-black
    "primary_dark": "#A0A0A0",
    "surface": "#0A0A0A",          # near-black canvas
    "surface_muted": "#121212",    # inputs, bordered containers
    "ink": "#F5F5F5",              # display + heading text
    "ink_muted": "#A0A0A0",        # body copy
    "border": "#2A2A2A",           # hairlines
    "review_low": "#3A3A3A",       # charcoal = few reviews (map gradient)
    "review_high": "#F5F5F5",      # off-white = many reviews
}

# ---------------------------------------------------------------------------
# 5. LISTING PHOTOS
# ---------------------------------------------------------------------------
# The Kaggle listings file ("AB_NYC_2019.csv") has NO photo column, so the
# map's hover card needs a picture from somewhere. Rather than scraping
# Airbnb (against their terms, breaks the app whenever they change the CDN),
# we use a curated set of 20 real, permanently-hot-linkable Unsplash photos
# (Unsplash licence: free commercial use, no attribution required) and assign
# one to each listing deterministically from its id. Same listing -> same
# photo on every rerun, and the distribution across the city looks natural.
#
# The raw photo is ~90 KB; ``?w=240&q=60`` makes the CDN serve a ~12 KB
# thumbnail, which keeps hover previews instant on 6k markers.
LISTING_PHOTOS: tuple[str, ...] = (
    "photo-1522708323590-d24dbb6b0267",   # bright loft living room
    "photo-1502672260266-1c1ef2d93688",   # soft grey bedroom wall
    "photo-1560448204-e02f11c3d0e2",      # modern living room, city view
    "photo-1560185127-6ed189bf02f4",      # living room + big windows
    "photo-1522771739844-6a9f6d5f14af",   # cosy sofa and plants
    "photo-1505693416388-ac5ce068fe85",   # warm bedroom
    "photo-1616594039964-ae9021a400a0",   # white bedroom, oak floor
    "photo-1598928506311-c55ded91a20c",   # terrace with skyline
    "photo-1499793983690-e29da59ef1c2",   # bedroom, garden view
    "photo-1540518614846-7eded433c457",   # duplex loft staircase
    "photo-1512917774080-9991f1c4c750",   # house exterior, warm light
    "photo-1580587771525-78b9dba3b914",   # modern house front
    "photo-1502005229762-cf1b2da7c5d6",   # compact studio
    "photo-1493809842364-78817add7ffb",   # living room, bookshelf
    "photo-1484154218962-a197022b5858",   # kitchen island
    "photo-1600585154340-be6161a56a0c",   # minimal bathroom/kitchen
    "photo-1600607687939-ce8a6c25118c",   # bright modern interior
    "photo-1600566753086-00f18fb6b3ea",   # bedroom with soft textiles
    "photo-1567767292278-a4f21aa2d36e",   # exposed brick living room
    "photo-1631049307264-da0ec9d70304",   # city hotel-condo interior
)

# Base URL of the Unsplash image CDN. Each id in ``LISTING_PHOTOS`` is a real
# photo path there, so the full URL is simply ``PHOTO_CDN_BASE + id``.
PHOTO_CDN_BASE: str = "https://images.unsplash.com/"


# ---------------------------------------------------------------------------
# 5. API KEYS
# ---------------------------------------------------------------------------
# Secrets come from environment variables. Locally they are usually defined in a
# ``.env`` file (parsed below); on Streamlit Cloud they come from the app's
# "Secrets" panel. Keys must never be committed -- .gitignore excludes .env.
def _load_dotenv_file() -> None:
    """
    Load ``.env`` from the project folder into ``os.environ`` if it exists.

    Deliberately hand-rolled (no python-dotenv dependency): only the simple
    ``KEY=value`` / ``# comment`` syntax that ``.env`` files actually use.

    Returns
    -------
    None
    """
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # Blank placeholders (``OPENAI_API_KEY=``) must not claim the slot:
            # otherwise a later line, a real shell variable, or a Streamlit
            # secret could never supply the value. Only a non-empty value that
            # is not already present is written. Real env vars still win.
            if key and value and key not in os.environ:
                os.environ[key] = value
    except OSError:                             # an unreadable .env is not fatal
        pass


_load_dotenv_file()


def get_secret(name: str, default: str | None = None) -> str | None:
    """
    Return a secret by name, checking Streamlit secrets then environment vars.

    Parameters
    ----------
    name:
        Variable name, e.g. ``"OPENAI_API_KEY"``.
    default:
        Value returned when the secret is not configured anywhere.

    Returns
    -------
    str | None
        The secret value, or ``default``.
    """
    try:
        import streamlit as st   # lazy import: this module is also used by CLIs

        if name in st.secrets:   # type: ignore[attr-defined]
            return str(st.secrets[name])  # type: ignore[index]
    except Exception:
        pass                     # no Streamlit runtime / no secrets file
    return os.getenv(name, default)


KAGGLE_USERNAME: str | None = get_secret("KAGGLE_USERNAME")
KAGGLE_KEY: str | None = get_secret("KAGGLE_KEY")
OPENAI_API_KEY: str | None = get_secret("OPENAI_API_KEY") or None
GEMINI_API_KEY: str | None = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY") or None

# Provider preference when both keys exist: "openai" or "gemini".
LLM_PROVIDER: str = (get_secret("LLM_PROVIDER", "openai") or "openai").strip().lower()
LLM_MODEL: str = get_secret("LLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
# Default model. Must be one Google actually serves: ``gemini-1.5-flash`` and
# ``gemini-2.5-flash`` both return 404 "no longer available to new users", and
# ``gemini-3.8-flash`` times out (>40s). The ``-latest`` alias auto-tracks
# Google's current model so a future retirement cannot silently break the app.
# An explicit GEMINI_MODEL in .env / st.secrets still overrides this.
GEMINI_MODEL: str = get_secret("GEMINI_MODEL", "gemini-flash-lite-latest") or "gemini-flash-lite-latest"

# ---------------------------------------------------------------------------
# BRANDING
# ---------------------------------------------------------------------------
APP_TITLE: str = "THE TOP REVIEWED NYC AIRBNB TRAVEL GUIDE"
# Kept in its original sentence case: the hero sets only the *headline* in
# caps, and the serif standfirst is meant to read as editorial copy under it.
APP_SUBTITLE: str = (
    "The best-reviewed Airbnb stays across the five boroughs - mapped, scored "
    "and explained by an AI concierge."
)

if __name__ == "__main__":
    # ``python config.py`` prints the resolved configuration for a quick check.
    print("BASE_DIR        :", BASE_DIR)
    print("DATA_DIR        :", DATA_DIR)
    print("LISTINGS_DATASET:", LISTINGS_DATASET)
    print("Kaggle creds    :", bool(KAGGLE_USERNAME and KAGGLE_KEY))
    print("OpenAI key      :", bool(OPENAI_API_KEY))
    print("Gemini key      :", bool(GEMINI_API_KEY))
    print("Boroughs        :", ", ".join(BOROUGHS))
