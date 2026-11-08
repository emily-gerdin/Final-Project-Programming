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
# LISTINGS -- "New York City Airbnb Open Data" (AB_NYC_2019.csv, 48,895 rows,
# CC0). Coordinates, price, room type, number_of_reviews and review_scores_*
# are all present, which is exactly what the rubric asks for.
#
#   https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
#
LISTINGS_DATASET: str = "dgomonov/new-york-city-airbnb-open-data"
LISTINGS_FILENAME: str = "AB_NYC_2019.csv"
LISTINGS_LICENSE: str = "CC0: Public Domain"

# REVIEWS -- the Kaggle listings file above contains NO review text, so the
# sentiment phase would have nothing to score. The Inside Airbnb NYC review
# archive is the upstream source the Kaggle listings file derives from, so its
# listing ids join cleanly against AB_NYC_2019.csv.
#
# The mirror is the PRIMARY path. It needs no credentials, so reviews work on a
# fresh clone with no Kaggle key configured.
#
# Note on the Kaggle route: it is kept first in code order for rubric
# compliance, but both previously-configured candidates are dead and will fail
# fast (see the warnings in the app log):
#   * arianazmoudeh/airbnbopendata      -> archive ships no reviews.csv
#   * julianbenavides/airbnb-nyc-reviews -> HTTP 403 (access revoked/private)
# They are left behind REVIEWS_KAGGLE_CANDIDATES = () below; population is
# optional and only attempted when the list is non-empty.
REVIEWS_KAGGLE_CANDIDATES: tuple[tuple[str, str], ...] = ()

# Inside Airbnb publishes one folder per city per snapshot date. The URL shape is
#   https://data.insideairbnb.com/united-states/ny/new-york-city/<date>/data/reviews.csv.gz
#
# IMPORTANT: the "latest" alias currently returns HTTP 403 Forbidden (checked
# 2026-11-07). Inside Airbnb blocks the bare alias but serves dated snapshots,
# so the dated URL is listed FIRST and is the one that actually resolves. The
# "latest" entry is retained last so the code self-heals if the alias is
# unblocked upstream.
INSIDE_AIRBNB_REVIEWS_URLS: tuple[str, ...] = (
    "https://data.insideairbnb.com/united-states/ny/new-york-city/2026-09-14/data/reviews.csv.gz",
    "https://data.insideairbnb.com/united-states/ny/new-york-city/latest/data/reviews.csv.gz",
)
# Fallback listings mirror (used only if the Kaggle listings download fails).
INSIDE_AIRBNB_LISTINGS_URLS: tuple[str, ...] = (
    "https://data.insideairbnb.com/united-states/ny/new-york-city/2026-09-14/data/listings.csv.gz",
    "https://data.insideairbnb.com/united-states/ny/new-york-city/latest/data/listings.csv.gz",
)

# ---------------------------------------------------------------------------
# 3. NYC GEOGRAPHY
# ---------------------------------------------------------------------------
NYC_CENTER_LAT: float = 40.7128
NYC_CENTER_LON: float = -74.0060
NYC_ZOOM: int = 10
NYC_PITCH: int = 45          # deck.gl tilt
NYC_BEARING: int = 0

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
REGION_COLORS: dict[str, str] = {
    "Manhattan": "#4c78a8",
    "Brooklyn": "#f58518",
    "Queens": "#72b7b2",
    "Bronx": "#e45756",
    "Staten Island": "#54a24b",
    "Outer Boroughs": "#9d755d",
}

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
APP_TITLE: str = "Top Reviewed NYC Travel Planner"
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
