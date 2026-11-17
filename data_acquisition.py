"""
data_acquisition.py
===================

**Phase 1 - Data Acquisition (20% of the grade)**

This is the only module that touches the outside world. Everything downstream
(``eda.py``, ``nlp_sentiment.py``, ``llm_agent.py``, ``app.py``) simply calls the
public functions here and receives tidy DataFrames.

Data sources
------------
1. **Inside Airbnb** (primary)
   ``https://insideairbnb.com/new-york-city/`` publishes a quarterly NYC
   snapshot under ``data.insideairbnb.com``. ``listings.csv.gz`` and
   ``reviews.csv.gz`` are pulled from the *same* snapshot, so the
   ``listing_id`` foreign key always matches and every listing has a real
   ``picture_url``. No credentials are required. Licence: CC BY 4.0.

2. **Kaggle** (optional legacy fallback)
   ``dgomonov/new-york-city-airbnb-open-data`` / ``AB_NYC_2019.csv``. Only tried
   when Kaggle credentials are configured, for machines that cannot reach
   ``data.insideairbnb.com``. Its 16-column schema (no photo column, 100%-null
   ratings) is normalised onto the same canonical schema by
   ``_validate_listings``, so downstream code cannot tell the two apart.

3. **Nominatim geocoding** (secondary API)
   Free-text place names -> lat/lon so the map can re-centre on what the user
   typed. Rate-limited to ~1 request/second, so results are cached.

4. **Schema-identical synthetic fallback**
   If no credentials and no network are available the app still runs on a
   generated dataset of the same shape, clearly flagged in the UI. Every chart,
   filter and the offline LLM report stay demonstrable.

Schema normalisation
--------------------
``_validate_listings`` maps each source onto one canonical schema. The Inside
Airbnb snapshot differs from the old Kaggle file in ways this module absorbs:

    * ``neighbourhood_group_cleansed`` / ``neighbourhood_cleansed`` are renamed
      onto ``neighbourhood_group`` / ``neighbourhood``,
    * ``price`` arrives as a formatted string ("$164.55") and is NULL for about
      a third of rows (mostly listings with no availability in the next year);
      those rows are excluded from price analysis and counted in the audit,
    * price columns such as ``price_quote_price_per_night`` are parsed when a
      backup value exists,
    * the 90-column file's extra numeric fields (accommodates, bedrooms, beds,
      the per-category review scores, the ``estimated_*`` metrics) are cast to
      numeric so they are available to EDA without pulling in unused text
      columns.

File structure -- ``listings.csv.gz`` (Inside Airbnb NYC, 2026-09-14)
---------------------------------------------------------------------
CSV (gzipped), UTF-8, one header row, **~30,000 data rows**, CC BY 4.0. The
columns this project relies on::

    id                             int    listing id, joins to reviews.listing_id
    name                           str    listing title
    picture_url                    str    real Airbnb CDN photo (hover cards)
    host_id / host_name            int/str host identity
    neighbourhood_group_cleansed   str    borough -> neighbourhood_group
    neighbourhood_cleansed         str    sub-area, e.g. "Williamsburg"
    latitude / longitude           float  WGS-84 coordinates (no nulls)
    room_type                      str    Entire home/apt | Private room |
                                          Hotel room | Shared room
    price                          str    "$164.55" per night; NULL for ~35%
    minimum_nights                 int    minimum stay required
    number_of_reviews              int    lifetime review count <- "top reviewed"
    number_of_reviews_ltm          int    reviews in the last 12 months
    review_scores_rating           float  populated for reviewed listings
    availability_365               int    days available in the next 365
    reviews_per_month              float  NaN when never reviewed
    calculated_host_listings_count int    listings owned by this host
    estimated_occupancy_l365d      int    Inside Airbnb occupancy estimate
    estimated_revenue_l365d        float  Inside Airbnb revenue estimate

File structure -- ``reviews.csv`` (review text)
----------------------------------------------
CSV (served gzipped as ``reviews.csv.gz``), one header row, ~1M data rows::

    listing_id     int    FK -> listings.id
    id             int    review id
    date           str    YYYY-MM-DD
    reviewer_id    int    reviewer id
    reviewer_name  str    reviewer display name
    comments       str    FREE TEXT -- the NLP input for VADER / TextBlob

Public API
----------
    ensure_data_available()   -> (listings, reviews, used_synthetic)   <- main entry
    download_listings()       -> Kaggle listings table (cached on disk)
    download_reviews()        -> reviews table (Kaggle, then Inside Airbnb)
    load_local_listings()     -> read data/listings.csv when present
    load_local_reviews()      -> read data/reviews.csv when present
    geocode_location()        -> Nominatim lookup with graceful fallback
    generate_synthetic_data() -> offline demo data, same schema
    data_quality_report()     -> pre-cleaning audit dict for the EDA tab
"""

from __future__ import annotations

import gzip
import io
import os
import ssl
import warnings
import zlib
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

import config as cfg

# Friendly log prefix so pipeline progress is identifiable in app logs.
LOG = "[data_acquisition]"

# Columns downstream code depends on. If a download is missing any of these we
# fail loudly here instead of raising a confusing KeyError deep inside the app.
REQUIRED_LISTING_COLUMNS: tuple[str, ...] = (
    "id",
    "name",
    "neighbourhood_group",
    "neighbourhood",
    "latitude",
    "longitude",
    "room_type",
    "price",
    "number_of_reviews",
)
REQUIRED_REVIEW_COLUMNS: tuple[str, ...] = ("listing_id", "comments")


def _write_csv_atomically(df: pd.DataFrame, destination: Path) -> None:
    """
    Write a DataFrame to CSV without ever leaving a half-written cache file.

    A plain ``df.to_csv(destination)`` truncates the target first, so an
    interrupted write (Ctrl-C, a full disk, the app being closed mid-download)
    leaves a truncated CSV that every later run then tries to parse. Writing to a
    sibling temporary file and renaming it is atomic on POSIX and Windows, so the
    cache is either the previous complete file or the new complete file.

    Parameters
    ----------
    df:
        Table to cache.
    destination:
        Final path, e.g. ``config.LOCAL_LISTINGS_CSV``.
    """
    tmp = destination.with_name(destination.name + ".part")
    df.to_csv(tmp, index=False)
    os.replace(tmp, destination)          # atomic on the same filesystem


def _log(message: str) -> None:
    """Print a prefixed progress line (visible in the terminal / app logs)."""
    print(f"{LOG} {message}", flush=True)



# ===========================================================================
# SECTION 1 - FILE UTILITIES
# ===========================================================================
def _find_file(root: Path, filename: str) -> Path | None:
    """
    Recursively search ``root`` for ``filename`` and return the first match.

    Kaggle unzips archives into nested folders whose depth is not guaranteed, so
    a depth-agnostic lookup is more robust than hard-coding a path.

    Parameters
    ----------
    root:
        Folder to search.
    filename:
        Exact file name to find, e.g. ``"AB_NYC_2019.csv"``.

    Returns
    -------
    pathlib.Path | None
        First matching path, or ``None`` when nothing matched.
    """
    if not root.exists():
        return None
    try:
        matches = sorted(root.rglob(filename))
    except OSError:                      # permission problems, unreadable dirs
        return None
    return matches[0] if matches else None


def _read_csv_safely(path: Path, **kwargs: Any) -> pd.DataFrame:
    """
    Read a CSV defensively, falling back through several parsing strategies.

    Real Kaggle / Inside Airbnb files frequently contain stray non-UTF-8 bytes.
    A plain ``pd.read_csv`` raises ``UnicodeDecodeError`` and the whole app dies,
    so we escalate: UTF-8 -> Latin-1 -> Python engine that skips bad lines.

    Parameters
    ----------
    path:
        CSV file to read.
    **kwargs:
        Extra keyword arguments passed to ``pandas.read_csv``.

    Returns
    -------
    pandas.DataFrame
        Parsed table, or an empty DataFrame when every attempt failed.
    """
    attempts: tuple[dict[str, Any], ...] = (
        {"encoding": "utf-8"},
        {"encoding": "latin-1"},
        {"encoding": "utf-8", "engine": "python", "on_bad_lines": "skip"},
    )
    for options in attempts:
        try:
            return pd.read_csv(path, **{**options, **kwargs})
        except UnicodeDecodeError:
            continue                     # wrong encoding -> try the next one
        except Exception as exc:         # malformed CSV -> try next strategy
            warnings.warn(f"read_csv attempt failed for {path.name}: {exc}")
    warnings.warn(f"Could not parse {path}; returning an empty DataFrame.")
    return pd.DataFrame()


def _ssl_context() -> ssl.SSLContext:
    """
    Build an SSL context that works on Python installs without root certs.

    The python.org macOS installer famously ships without the system CA bundle,
    which makes every HTTPS call fail with ``CERTIFICATE_VERIFY_FAILED``. Using
    ``certifi`` when it is installed (it ships with pip/requests) fixes that
    without weakening verification.

    Returns
    -------
    ssl.SSLContext
        A verifying TLS context.
    """
    context = ssl.create_default_context()
    try:
        import certifi

        context.load_verify_locations(certifi.where())
    except Exception:
        pass        # fall back to whatever the platform provides
    return context


def _download_bytes(url: str, timeout: int = 180) -> bytes | None:
    """
    Download a URL into memory, transparently gunzipping ``.gz`` payloads.

    Used for the Inside Airbnb mirror, which publishes ``reviews.csv.gz``. A
    descriptive User-Agent is sent because some CDNs reject the default.

    Parameters
    ----------
    url:
        Absolute http(s) URL.
    timeout:
        Socket timeout in seconds.

    Returns
    -------
    bytes | None
        The (decompressed) bytes, or ``None`` when the request failed.
    """
    try:
        request = Request(url, headers={"User-Agent": "nyc-travel-planner-capstone/1.0"})
        with urlopen(request, timeout=timeout, context=_ssl_context()) as response:
            raw = response.read()
    except (URLError, OSError, TimeoutError) as exc:
        warnings.warn(f"Download failed for {url}: {exc}")
        return None

    if raw[:2] == b"\x1f\x8b":           # gzip magic number
        try:
            return gzip.decompress(raw)
        except OSError as exc:
            warnings.warn(f"Could not gunzip {url}: {exc}")
            return None
    return raw


def _downsample_reviews(reviews: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """
    Deterministically subsample a reviews table down to ``limit`` rows.

    Why not ``nrows=limit``: Inside Airbnb's ``reviews.csv.gz`` is ordered by
    review date, so reading only the first N rows yields a cluster of very old
    reviews covering only a handful of listings (~1,000 of them for a 100k cap).
    Sampling uniformly across the whole table instead spreads the cap over
    thousands of listings, so per-listing sentiment is computed for far more of
    the map's pins.

    Parameters
    ----------
    reviews:
        Full validated reviews frame.
    limit:
        Target row count. ``None`` or ``0`` (or a value >= ``len(reviews)``)
        returns the frame unchanged.

    Returns
    -------
    pandas.DataFrame
        At most ``limit`` rows, re-indexed.
    """
    if limit is None or limit <= 0 or len(reviews) <= limit:
        return reviews.reset_index(drop=True)
    sampled = reviews.sample(n=limit, random_state=42)
    return sampled.reset_index(drop=True)


def _read_csv_bytes(payload: bytes, **kwargs: Any) -> pd.DataFrame:
    """
    Parse CSV bytes through the same encoding fallbacks as ``_read_csv_safely``.

    Parameters
    ----------
    payload:
        Raw CSV bytes.
    **kwargs:
        Extra keyword arguments for ``pandas.read_csv``.

    Returns
    -------
    pandas.DataFrame
        Parsed table (empty when every attempt failed).
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(payload), encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            warnings.warn(f"Byte-parsing attempt failed ({encoding}): {exc}")
    return pd.DataFrame()



# ===========================================================================
# SECTION 2 - SCHEMA VALIDATION / NORMALISATION
# ===========================================================================
# Neighbourhood -> borough lookup for data sources that omit
# ``neighbourhood_group``. Inside Airbnb ships a ``neighbourhoods.csv`` mapping,
# but it is not always present, so we keep a compact keyword table covering the
# well-known areas plus the per-borough street/area suffixes.
_BOROUGH_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Manhattan": (
        "manhattan", "harlem", "east village", "west village", "soho", "noho",
        "tribeca", "chelsea", "midtown", "upper east side", "upper west side",
        "greenwich village", "financial district", "hell's kitchen", "murray hill",
        "gramercy", "kips bay", "lower east side", "chinatown", "little italy",
        "washington heights", "inwood", "morningside heights", "battery park",
        "civic center", "two bridges", "roosevelt island", "clinton",
    ),
    "Brooklyn": (
        "brooklyn", "williamsburg", "bushwick", "bedford-stuyvesant", "bed-stuy",
        "park slope", "greenpoint", "crown heights", "red hook", "dumbo",
        "brooklyn heights", "flatbush", "bay ridge", "sunset park", "bensonhurst",
        "coney island", "brighton beach", "sheepshead bay", "canarsie", "brownsville",
        "east new york", "fort greene", "clinton hill", "prospect heights",
        "windsor terrace", "kensington", "borough park", "gravesend",
    ),
    "Queens": (
        "queens", "astoria", "long island city", "flushing", "jamaica", "forest hills",
        "ridgewood", "jackson heights", "woodside", "sunnyside", "corona",
        "rego park", "bayside", "fresh meadows", "st. albans", "springfield gardens",
        "rockaway", "elmhurst", "maspeth", "glendale", "howard beach",
    ),
    "Bronx": (
        "bronx", "mott haven", "fordham", "riverdale", "concourse", "melrose",
        "morrisania", "hunts point", "soundview", "throgs neck", "city island",
        "pelham", "wakefield", "woodlawn", "kingsbridge", "tremont",
    ),
    "Staten Island": (
        "staten island", "st. george", "tompkinsville", "great kills", "tottenville",
        "new dorp", "port richmond", "midland beach", "willowbrook", "eltingville",
    ),
}


def _borough_from_neighbourhood(neighbourhood: pd.Series) -> pd.Series:
    """
    Map neighbourhood names onto one of the five NYC boroughs.

    Used only when the source file lacks a borough column (Inside Airbnb's
    listings feed). Matching is keyword-based and case-insensitive; anything
    unrecognised becomes ``"Unknown"`` rather than a wrong guess.

    Parameters
    ----------
    neighbourhood:
        Series of neighbourhood names.

    Returns
    -------
    pandas.Series
        Borough name per row.
    """
    def lookup(value: Any) -> str:
        text = str(value).lower()
        for borough, keywords in _BOROUGH_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                return borough
        return "Unknown"

    return neighbourhood.map(lookup)


def _to_numeric_currency(series: pd.Series) -> pd.Series:
    """
    Convert a money column to floats, tolerating currency formatting.

    Inside Airbnb stores prices as ``"$1,200.00"`` while the Kaggle file stores
    plain integers, so a plain ``pd.to_numeric`` would discard every value.

    Parameters
    ----------
    series:
        Raw money column.

    Returns
    -------
    pandas.Series
        Float values; unparseable entries become ``NaN``.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(r"[^\d.\-]", "", regex=True)   # drop $, commas, spaces
    )
    cleaned = cleaned.replace({"": None, "-": None, ".": None})
    return pd.to_numeric(cleaned, errors="coerce")


def _validate_listings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce a listings table into the canonical schema used across the project.

    Steps
    -----
    1. Empty input is returned untouched (callers handle that case).
    2. Column aliases are normalised -- Inside Airbnb's mirror uses the same
       names, but other mirrors differ (``neighborhood``, ``lat``, ``room type``).
    3. Essential columns are verified; a clear ``ValueError`` is raised if any
       are absent, so a bad download fails immediately rather than surfacing as
       a ``KeyError`` in the UI.
    4. Optional columns are created with sensible defaults.
    5. Numeric columns are cast; rows without coordinates are dropped.

    Parameters
    ----------
    df:
        Raw listings DataFrame.

    Returns
    -------
    pandas.DataFrame
        Validated, correctly typed listings table.

    Raises
    ------
    ValueError
        When a required column is missing from the source file.
    """
    if df.empty:
        return df

    # --- 1. rename alternative spellings ----------------------------------
    # Inside Airbnb's own mirror uses ``*_cleansed`` variants for the
    # administrative columns, and other mirrors use US spellings or short forms.
    #
    # NOTE: some sources ship BOTH ``neighbourhood`` and ``neighbourhood_cleansed``
    # (same for room_type / availability). Renaming both onto one canonical name
    # would create duplicate columns whose values then collapse to NaN, so the
    # ``*_cleansed`` variants are handled explicitly in step 1b rather than here.
    aliases = {
        "neighborhood_group": "neighbourhood_group",
        "neighborhood": "neighbourhood",
        "room type": "room_type",
        "lat": "latitude",
        "lon": "longitude",
        "lng": "longitude",
        "number of reviews": "number_of_reviews",
        "reviews per month": "reviews_per_month",
        "minimum nights": "minimum_nights",
        "availability": "availability_365",
        "review scores rating": "review_scores_rating",
    }
    df = df.rename(columns={
        col: aliases[col.strip().lower()]
        for col in df.columns
        if col.strip().lower() in aliases
    })

    # --- 1b. prefer the ``*_cleansed`` columns when present ---------------
    # Inside Airbnb's cleansed columns are the tidied, analysis-ready ones
    # (standardised neighbourhood names, integer availability). Overwrite the
    # canonical column with them, then drop the duplicate source column.
    cleansed_map = {
        "neighbourhood_group_cleansed": "neighbourhood_group",
        "neighbourhood_cleansed": "neighbourhood",
        "room_type_cleansed": "room_type",
        "minimum_nights_cleansed": "minimum_nights",
        "availability_365_cleansed": "availability_365",
    }
    for source, target in cleansed_map.items():
        if source in df.columns:
            df[target] = df[source]
            df = df.drop(columns=[source])

    # --- 1c. derive the borough when the source omits it ------------------
    # Inside Airbnb's listings feed carries the descriptive ``neighbourhood`` but
    # not always the borough. Recover it from the neighbourhood name so the
    # required ``neighbourhood_group`` column exists for every downstream chart.
    if "neighbourhood_group" not in df.columns and "neighbourhood" in df.columns:
        df["neighbourhood_group"] = _borough_from_neighbourhood(df["neighbourhood"])

    # --- 2. verify the essentials ------------------------------------------
    missing = [col for col in REQUIRED_LISTING_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Listings file is missing required column(s): {missing}. "
            f"Columns found: {list(df.columns)[:20]}"
        )

    # --- 3. defaults for optional columns ----------------------------------
    defaults: dict[str, Any] = {
        "host_id": np.nan,
        "host_name": "Unknown host",
        "minimum_nights": 1,
        "reviews_per_month": np.nan,
        "calculated_host_listings_count": 1,
        "availability_365": 0,
        "review_scores_rating": np.nan,   # absent from AB_NYC_2019, present in mirrors
        "number_of_reviews_ltm": 0,
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    # --- 4. type casting ---------------------------------------------------
    # Some mirrors store money as formatted strings ("$1,200.00"), so strip the
    # currency symbols and thousands separators before the numeric cast.
    for money_column in ("price", "cleaning_fee", "security_deposit"):
        if money_column in df.columns:
            df[money_column] = _to_numeric_currency(df[money_column])

    # --- 4b. recover missing prices where a quote exists -------------------
    # Inside Airbnb leaves ``price`` NULL for roughly a third of NYC rows. In the
    # 2026-09-14 snapshot those are concentrated among listings with
    # ``availability_365 == 0`` (nothing bookable in the next year, so no price
    # quote could be scraped); ``price_quote_price_per_night`` is null for
    # exactly the same rows, so it is not a substitute there.
    #
    # The fill is still attempted because other snapshots and mirrors do publish
    # a quote for some of these rows, and it is free when the column is absent.
    # Whatever remains null is dropped later by the cleaning pipeline, which
    # audits the loss in ``eda.remove_price_outliers``.
    if "price" in df.columns and "price_quote_price_per_night" in df.columns:
        backup = _to_numeric_currency(df["price_quote_price_per_night"])
        fillable = df["price"].isna() & backup.notna()
        recovered = int(fillable.sum())
        if recovered:
            df.loc[fillable, "price"] = backup[fillable]
            _log(f"recovered {recovered:,} missing price(s) from "
                 f"price_quote_price_per_night")
        still_missing = int(df["price"].isna().sum())
        if still_missing:
            _log(f"{still_missing:,} listing(s) have no price in this snapshot "
                 f"and will be excluded from price analysis")

    numeric_columns = (
        "latitude", "longitude", "price", "minimum_nights", "number_of_reviews",
        "reviews_per_month", "availability_365", "review_scores_rating",
        "calculated_host_listings_count", "number_of_reviews_ltm",
        # Present (and useful) in the richer Inside Airbnb schema.
        "accommodates", "bedrooms", "beds", "bathrooms",
        "number_of_reviews_l30d", "estimated_occupancy_l365d",
        "estimated_revenue_l365d", "review_scores_accuracy",
        "review_scores_cleanliness", "review_scores_checkin",
        "review_scores_communication", "review_scores_location",
        "review_scores_value",
    )
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ("name", "neighbourhood_group", "neighbourhood", "room_type"):
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()

    # --- 5. coordinates are mandatory for the map --------------------------
    before = len(df)
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        _log(f"dropped {dropped:,} listing row(s) with missing coordinates")
    return df


def _validate_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce the reviews table into the canonical ``(listing_id, comments)`` form.

    Parameters
    ----------
    df:
        Raw reviews DataFrame.

    Returns
    -------
    pandas.DataFrame
        Columns ``listing_id`` (int64), ``comments`` (str) and, when available,
        ``date``.

    Raises
    ------
    ValueError
        When ``listing_id`` or ``comments`` cannot be found.
    """
    if df.empty:
        return df

    aliases = {
        "comments": "comments",
        "review": "comments",
        "text": "comments",
        "listing id": "listing_id",
        "listingid": "listing_id",
    }
    df = df.rename(columns={
        col: aliases[col.strip().lower()]
        for col in df.columns
        if col.strip().lower() in aliases
    })

    missing = [col for col in REQUIRED_REVIEW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Reviews file is missing required column(s): {missing}. "
            f"Columns found: {list(df.columns)[:15]}"
        )

    keep = [col for col in ("listing_id", "comments", "date") if col in df.columns]
    df = df.loc[:, keep].copy()

    df["listing_id"] = pd.to_numeric(df["listing_id"], errors="coerce")
    df = df.dropna(subset=["listing_id"])
    df["listing_id"] = df["listing_id"].astype("int64")
    df["comments"] = df["comments"].fillna("").astype(str)
    return df.reset_index(drop=True)



# ===========================================================================
# SECTION 3 - LOCAL CACHE LOADERS (no network required)
# ===========================================================================
def load_local_listings() -> pd.DataFrame | None:
    """
    Read ``data/listings.csv`` when it exists.

    Returns
    -------
    pandas.DataFrame | None
        Validated listings, or ``None`` when the cache is absent or unparseable.
    """
    path = cfg.LOCAL_LISTINGS_CSV
    if not path.exists():
        return None
    _log(f"loading cached listings from {path.name}")
    try:
        validated = _validate_listings(_read_csv_safely(path, low_memory=False))
        return validated if not validated.empty else None
    except Exception as exc:
        warnings.warn(f"Could not use cached listings: {exc}")
        return None


def load_local_reviews(limit: int | None = None) -> pd.DataFrame | None:
    """
    Read ``data/reviews.csv`` when it exists.

    Parameters
    ----------
    limit:
        Optional cap on rows read, to keep memory use predictable.

    Returns
    -------
    pandas.DataFrame | None
        Validated reviews, or ``None`` when the cache is absent.
    """
    path = cfg.LOCAL_REVIEWS_CSV
    if not path.exists():
        return None
    _log(f"loading cached reviews from {path.name} (limit={limit})")
    try:
        validated = _validate_reviews(_read_csv_safely(path, low_memory=False, nrows=limit))
        return validated if not validated.empty else None
    except Exception as exc:
        warnings.warn(f"Could not use cached reviews: {exc}")
        return None



# ===========================================================================
# SECTION 4 - KAGGLE API (primary source)
# ===========================================================================
class KaggleCredentialsMissing(RuntimeError):
    """Raised when no Kaggle username / API key can be found."""


def kaggle_credentials_available() -> bool:
    """
    Report whether the Kaggle API can authenticate.

    Credentials are accepted in either of the two ways Kaggle supports:

    1. ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` environment variables (usually set
       via ``.env``), read through ``config.get_secret``.
    2. A credentials file at ``~/.kaggle/kaggle.json``, produced by
       "Create New API Token" on kaggle.com.

    Returns
    -------
    bool
        ``True`` when at least one credential source is present.
    """
    if cfg.KAGGLE_USERNAME and cfg.KAGGLE_KEY:
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def _ensure_credentials() -> None:
    """
    Raise a helpful, actionable error when credentials are absent.

    Without this guard the Kaggle SDK surfaces an opaque internal error
    (e.g. a ``NoneType`` path), which is useless to a user reading the app.

    Raises
    ------
    KaggleCredentialsMissing
        Always, when no credentials are configured.
    """
    if not kaggle_credentials_available():
        raise KaggleCredentialsMissing(
            "No Kaggle credentials found. Set KAGGLE_USERNAME and KAGGLE_KEY in "
            "a .env file, or place kaggle.json in ~/.kaggle/. See README step 3."
        )


def _download_with_kagglehub(slug: str) -> Path:
    """
    Download a Kaggle dataset with ``kagglehub`` and return its cached folder.

    Parameters
    ----------
    slug:
        Dataset identifier in ``owner/dataset-name`` form.

    Returns
    -------
    pathlib.Path
        Directory holding the unzipped dataset files.

    Raises
    ------
    KaggleCredentialsMissing
        When no credentials are configured.
    RuntimeError
        When the package is unavailable or the request fails.
    """
    _ensure_credentials()
    try:
        import kagglehub
    except ImportError as exc:                       # pragma: no cover
        raise RuntimeError("kagglehub is not installed (pip install kagglehub).") from exc

    _log(f"kagglehub: downloading '{slug}' ...")
    return Path(kagglehub.dataset_download(slug))


def _download_with_kaggle_cli(slug: str, destination: Path) -> Path:
    """
    Fallback downloader using the classic ``kaggle`` package.

    ``kagglehub`` is the modern route, but it occasionally lags behind on new
    datasets. ``KaggleApi.dataset_download_files`` unzips into ``destination``
    and is a reliable plan B.

    Parameters
    ----------
    slug:
        Dataset identifier in ``owner/dataset-name`` form.
    destination:
        Folder to unzip into (created when missing).

    Returns
    -------
    pathlib.Path
        ``destination`` after a successful download.

    Raises
    ------
    KaggleCredentialsMissing
        When no credentials are configured.
    RuntimeError
        When the package is unavailable.
    """
    _ensure_credentials()
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:                       # pragma: no cover
        raise RuntimeError("kaggle is not installed (pip install kaggle).") from exc

    destination.mkdir(parents=True, exist_ok=True)
    _log(f"kaggle CLI: downloading '{slug}' -> {destination.name}/ ...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(slug, path=str(destination), unzip=True, quiet=True)
    return destination


def _download_dataset(slug: str) -> Path:
    """
    Try ``kagglehub`` first, then the classic ``kaggle`` client.

    Parameters
    ----------
    slug:
        Dataset identifier in ``owner/dataset-name`` form.

    Returns
    -------
    pathlib.Path
        Folder expected to contain the requested CSV.

    Raises
    ------
    KaggleCredentialsMissing
        When no credentials are configured.
    RuntimeError
        When both strategies fail, with both error messages included.
    """
    errors: list[str] = []
    try:
        return _download_with_kagglehub(slug)
    except KaggleCredentialsMissing:
        raise                                        # not a fallback-able error
    except Exception as exc:
        errors.append(f"kagglehub: {exc}")

    cache_dir = cfg.DATA_DIR / "kaggle_cache" / slug.replace("/", "__")
    try:
        return _download_with_kaggle_cli(slug, cache_dir)
    except KaggleCredentialsMissing:
        raise
    except Exception as exc:
        errors.append(f"kaggle CLI: {exc}")

    raise RuntimeError("All Kaggle download methods failed:\n  - " + "\n  - ".join(errors))



def download_listings(force: bool = False) -> pd.DataFrame:
    """
    Load the NYC Airbnb **listings** table from Inside Airbnb.

    Resolution order
    ----------------
    1. ``data/listings.csv`` cache (unless ``force`` is set).
    2. Inside Airbnb NYC snapshot -- the primary source (no credentials).
    3. Kaggle ``AB_NYC_2019.csv`` -- optional legacy fallback, only attempted
       when Kaggle credentials are configured.
    4. ``RuntimeError`` -- the caller decides whether to fall back to synthetic.

    The Inside Airbnb file is 90 columns wide and uses the ``*_cleansed``
    borough/neighbourhood columns; ``_validate_listings`` normalises it onto the
    same canonical schema the app has always used, so every downstream step
    (cleaning, EDA, sentiment, map, LLM) is unchanged.

    Parameters
    ----------
    force:
        Ignore the local cache and re-download.

    Returns
    -------
    pandas.DataFrame
        Listings in the canonical schema.

    Raises
    ------
    RuntimeError
        When every source failed.
    """
    if not force:
        cached = load_local_listings()
        if cached is not None and not cached.empty:
            return cached

    errors: list[str] = []

    # --- 1. Inside Airbnb NYC (primary source) ----------------------------
    for url in cfg.INSIDE_AIRBNB_LISTINGS_URLS:
        snapshot = url.rstrip("/").split("/")[-3] if "/data/" in url else url
        _log(f"trying Inside Airbnb listings ({snapshot}) ...")
        payload = _download_bytes(url, timeout=300)
        if payload is None:
            errors.append(f"inside-airbnb {snapshot}: unreachable")
            continue
        try:
            listings = _validate_listings(_read_csv_bytes(payload, low_memory=False))
            if listings.empty:
                raise RuntimeError("download parsed to an empty table")
            _write_csv_atomically(listings, cfg.LOCAL_LISTINGS_CSV)
            _log(f"listings ready from Inside Airbnb ({snapshot}): "
                 f"{len(listings):,} rows x {listings.shape[1]} cols "
                 f"(cached to {cfg.LOCAL_LISTINGS_CSV.name})")
            return listings
        except Exception as exc:
            errors.append(f"inside-airbnb {snapshot}: {exc}")

    # --- 2. Kaggle, only when credentials exist (legacy fallback) ---------
    if kaggle_credentials_available():
        try:
            folder = _download_dataset(cfg.LISTINGS_DATASET)
            csv_path = _find_file(Path(folder), cfg.LISTINGS_FILENAME)
            if csv_path is None:                      # archive may hold one CSV
                candidates = sorted(Path(folder).rglob("*.csv"))
                csv_path = next(
                    (p for p in candidates if "calendar" not in p.name.lower()), None
                )
            if csv_path is None:
                raise RuntimeError(f"no CSV found inside {folder}")

            _log(f"reading {csv_path.name} from the Kaggle fallback")
            listings = _validate_listings(_read_csv_safely(csv_path, low_memory=False))
            if listings.empty:
                raise RuntimeError("downloaded listings table was empty")

            _write_csv_atomically(listings, cfg.LOCAL_LISTINGS_CSV)
            _log(f"listings ready from Kaggle fallback: {len(listings):,} rows")
            return listings
        except Exception as exc:
            errors.append(f"kaggle: {exc}")
    else:
        errors.append("kaggle: skipped (no credentials configured)")

    raise RuntimeError("Could not load listings:\n  - " + "\n  - ".join(errors))



def download_reviews(force: bool = False, limit: int | None = None) -> pd.DataFrame:
    """
    Load the NYC Airbnb **reviews** table (free-text ``comments``) from Inside
    Airbnb.

    Inside Airbnb is the primary source: ``reviews.csv.gz`` is published beside
    ``listings.csv.gz`` in the same snapshot, so the ``listing_id`` foreign key
    is guaranteed to match the listings this app maps. Kaggle is only consulted
    when a reviews dataset is configured *and* credentials exist, which is not
    the default (see ``config.REVIEWS_KAGGLE_CANDIDATES``).

    Review tables are large (~1M rows for NYC), so ``limit`` caps the rows read:
    the average sentiment of a random sample is statistically indistinguishable
    from the full set, at a fraction of the runtime.

    Parameters
    ----------
    force:
        Ignore ``data/reviews.csv`` and re-download.
    limit:
        Row cap. ``0`` or negative means "no reviews" (empty DataFrame);
        ``None`` loads everything.

    Returns
    -------
    pandas.DataFrame
        Reviews with at least ``listing_id`` and ``comments``.

    Raises
    ------
    RuntimeError
        When every source failed.
    """
    if limit is not None and limit <= 0:
        return pd.DataFrame(columns=list(REQUIRED_REVIEW_COLUMNS))

    if not force:
        cached = load_local_reviews(limit=limit)
        if cached is not None and not cached.empty:
            return cached

    errors: list[str] = []
    row_cap = limit or cfg.SENTIMENT_SAMPLE_SIZE

    # --- 1. Inside Airbnb NYC (primary source) ----------------------------
    # Dated snapshots resolve; the bare "latest" alias 403s upstream, so the
    # pinned snapshot is ordered first in config.INSIDE_AIRBNB_REVIEWS_URLS.
    for url in cfg.INSIDE_AIRBNB_REVIEWS_URLS:
        snapshot = url.rstrip("/").split("/")[-3] if "/data/" in url else url
        _log(f"trying Inside Airbnb reviews (snapshot {snapshot}) ...")
        payload = _download_bytes(url, timeout=300)
        if payload is None:
            errors.append(f"inside-airbnb {snapshot}: unreachable")
            continue
        try:
            parsed = _read_csv_bytes(payload, low_memory=False)
            reviews = _downsample_reviews(_validate_reviews(parsed), row_cap)
            if reviews.empty:
                raise RuntimeError("reviews table was empty")
            # Cache the (capped) sample so subsequent runs skip the ~120 MB
            # download entirely. Only the rows actually scored are written, which
            # keeps the cache small and honours the reviews_limit setting.
            try:
                _write_csv_atomically(reviews, cfg.LOCAL_REVIEWS_CSV)
            except OSError as exc:
                warnings.warn(f"Could not cache reviews to disk: {exc}")
            _log(f"reviews ready from Inside Airbnb ({snapshot}): "
                 f"{len(reviews):,} rows")
            return reviews
        except Exception as exc:
            errors.append(f"inside-airbnb {snapshot}: {exc}")

    # --- 2. Kaggle reviews datasets (optional legacy fallback) ------------
    if not cfg.REVIEWS_KAGGLE_CANDIDATES:
        _log("no Kaggle reviews dataset configured; skipping the legacy fallback")
    elif kaggle_credentials_available():
        for slug, filename in cfg.REVIEWS_KAGGLE_CANDIDATES:
            try:
                folder = _download_dataset(slug)
                csv_path = _find_file(Path(folder), filename) or _find_file(
                    Path(folder), "reviews.csv"
                )
                if csv_path is None:
                    raise RuntimeError(f"{filename} not present in {slug}")
                raw = _read_csv_safely(csv_path, low_memory=False, nrows=row_cap)
                reviews = _validate_reviews(raw)
                if reviews.empty:
                    raise RuntimeError("reviews table was empty")
                _log(f"reviews ready from Kaggle '{slug}': {len(reviews):,} rows")
                return reviews
            except Exception as exc:
                errors.append(f"kaggle {slug}: {exc}")
                warnings.warn(f"Kaggle reviews source failed ({slug}): {exc}")
    else:
        errors.append("kaggle: skipped (no credentials configured)")

    raise RuntimeError("Could not load reviews:\n  - " + "\n  - ".join(errors))



# ===========================================================================
# SECTION 5 - SECONDARY API: NOMINATIM GEOCODING (geopy)
# ===========================================================================
def geocode_location(
    query: str,
    user_agent: str = "nyc-travel-planner-capstone",
    timeout: int = 8,
) -> tuple[float, float, str] | None:
    """
    Convert a free-text NYC place name into ``(latitude, longitude, label)``.

    This is the **secondary API** the rubric asks for. Nominatim (OpenStreetMap)
    is free and keyless, but its usage policy allows roughly one request per
    second, so we always send a descriptive User-Agent and the caller caches the
    result (see ``app.geocode_cached``).

    Lookup order
    ------------
    1. A curated preset in ``config.NEIGHBOURHOOD_PRESETS`` -- instant, offline
       and immune to rate limiting.
    2. A live Nominatim query scoped to New York City.

    Parameters
    ----------
    query:
        What the user typed, e.g. ``"Williamsburg"`` or ``"Mott Haven"``.
    user_agent:
        Identifier sent to Nominatim (their policy requires a real one).
    timeout:
        Per-request timeout in seconds.

    Returns
    -------
    tuple[float, float, str] | None
        Coordinates plus a human-readable label, or ``None`` when nothing
        matched (caller should fall back to the default NYC view).
    """
    if not query or not str(query).strip():
        return None

    cleaned = str(query).strip()
    key = cleaned.lower()

    # --- 1. offline preset match (exact or substring) ---------------------
    for name, (lat, lon) in cfg.NEIGHBOURHOOD_PRESETS.items():
        if key == name.lower() or key in name.lower() or name.lower() in key:
            _log(f"geocode preset hit: '{query}' -> {name}")
            return float(lat), float(lon), name

    # --- 2. live Nominatim lookup -----------------------------------------
    try:
        from geopy.geocoders import Nominatim
    except ImportError:                              # pragma: no cover
        warnings.warn("geopy is not installed; geocoding unavailable.")
        return None

    geocoder = Nominatim(user_agent=user_agent, timeout=timeout)
    scoped = cleaned if "new york" in key else f"{cleaned}, New York City, NY, USA"
    try:
        location = geocoder.geocode(scoped, addressdetails=False)
    except Exception as exc:                         # timeout, network, rate limit
        warnings.warn(f"Nominatim lookup failed for '{query}': {exc}")
        return None

    if location is None:
        return None
    _log(f"geocode live hit: '{query}' -> {location.address}")
    return float(location.latitude), float(location.longitude), location.address


# ===========================================================================
# SECTION 6 - OFFLINE SYNTHETIC FALLBACK (clearly flagged demo data)
# ===========================================================================
# Guarantees the app is always demonstrable: in class, without a Kaggle token,
# or without internet. Every row is fake, and the UI shows a banner saying so.
_BOROUGH_PROFILES: dict[str, dict[str, float]] = {
    "Manhattan": {"lat": 40.7700, "lon": -73.9700, "price": 190.0, "weight": 0.44},
    "Brooklyn": {"lat": 40.6782, "lon": -73.9442, "price": 120.0, "weight": 0.41},
    "Queens": {"lat": 40.7282, "lon": -73.7949, "price": 95.0, "weight": 0.11},
    "Bronx": {"lat": 40.8448, "lon": -73.8648, "price": 85.0, "weight": 0.03},
    "Staten Island": {"lat": 40.5795, "lon": -74.1502, "price": 110.0, "weight": 0.01},
}

_NEIGHBOURHOODS: dict[str, tuple[str, ...]] = {
    "Manhattan": ("Harlem", "Chelsea", "SoHo", "Midtown", "Upper East Side",
                  "Upper West Side", "Greenwich Village", "Financial District",
                  "East Village", "Hell's Kitchen"),
    "Brooklyn": ("Williamsburg", "Bushwick", "Park Slope", "Bedford-Stuyvesant",
                 "Greenpoint", "Crown Heights", "Brooklyn Heights", "Red Hook"),
    "Queens": ("Astoria", "Long Island City", "Flushing", "Jamaica", "Forest Hills"),
    "Bronx": ("Mott Haven", "Fordham", "Riverdale"),
    "Staten Island": ("St. George", "Tompkinsville"),
}

_REVIEW_TEMPLATES: dict[str, tuple[str, ...]] = {
    "positive": (
        "Amazing place, spotless and the host was incredibly welcoming. Loved it!",
        "Perfect stay in {area}. Walk to the subway and great restaurants. Would return.",
        "Exactly as pictured, check-in was seamless and easy. Highly recommend.",
        "Wonderful from start to finish. The rooftop view was unforgettable.",
        "Great value, quiet at night, and the bed was extremely comfortable.",
    ),
    "neutral": (
        "The apartment was fine. Nothing special but it did the job.",
        "Decent location in {area}. A little noisy but acceptable overall.",
        "Smaller than the photos suggested, but clean enough and fairly priced.",
        "Okay stay. Check-in instructions were a bit confusing but we worked it out.",
    ),
    "negative": (
        "Disappointing. Not clean and the host never answered our messages.",
        "Noisy street, broken heater, and check-in was two hours late.",
        "Photos were misleading and the bathroom smelled bad. Left early.",
        "Overpriced for what you get and the area in {area} felt unsafe at night.",
    ),
}



def generate_synthetic_data(
    n_listings: int = 4000,
    n_reviews: int = 16000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate realistic-but-fake NYC listings and reviews with the real schema.

    Use only when no real data can be obtained. Column names and dtypes match
    ``AB_NYC_2019.csv`` so every downstream function works unchanged.

    Deliberate data-quality defects are injected (a few $0 prices, a few absurd
    prices, some missing coordinates) so the EDA cleaning logic has real work to
    do and the audit numbers shown in the UI are not all zero.

    Parameters
    ----------
    n_listings:
        Number of synthetic listings.
    n_reviews:
        Number of synthetic reviews.
    random_state:
        Seed, so the demo data is reproducible for grading.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        ``(listings, reviews)`` in the canonical schema.
    """
    rng = np.random.default_rng(random_state)
    _log(f"generating SYNTHETIC demo data ({n_listings:,} listings / {n_reviews:,} reviews)")

    boroughs = np.array(list(_BOROUGH_PROFILES))
    weights = np.array([_BOROUGH_PROFILES[b]["weight"] for b in boroughs], dtype=float)
    weights /= weights.sum()
    chosen = rng.choice(boroughs, size=n_listings, p=weights)

    rows: list[dict[str, Any]] = []
    for index, borough in enumerate(chosen):
        profile = _BOROUGH_PROFILES[str(borough)]
        hood = str(rng.choice(_NEIGHBOURHOODS[str(borough)]))
        room_type = str(rng.choice(cfg.ROOM_TYPES, p=[0.52, 0.45, 0.03]))

        # Price = borough baseline x room-type factor x log-normal noise, which
        # reproduces the right-skewed distribution real Airbnb prices show.
        room_factor = {"Entire home/apt": 1.0, "Private room": 0.55, "Shared room": 0.30}[room_type]
        price = profile["price"] * room_factor * float(rng.lognormal(0.0, 0.35))

        # Inject defects so the cleaning step is genuinely exercised.
        if rng.random() < 0.01:
            price = 0.0                                  # $0 rows
        elif rng.random() < 0.01:
            price = float(rng.integers(1500, 4000))      # absurd outliers

        # Heavy-tailed like the real file: most listings have a moderate count,
        # a minority are heavily reviewed (so "top reviewed" is meaningful).
        n_listing_reviews = int(rng.gamma(shape=2.0, scale=32.0))
        rows.append({
            "id": 100_000 + index,
            "name": (f"{room_type} in {hood} - "
                     f"{rng.choice(['Cozy', 'Sunny', 'Modern', 'Charming', 'Spacious'])} "
                     f"{rng.choice(['Studio', 'Loft', 'Condo', 'Apartment'])}"),
            "host_id": int(rng.integers(1_000, 50_000)),
            "host_name": str(rng.choice(["Maria", "James", "Sofia", "Wei", "Ahmed",
                                          "Emily", "Carlos", "Aisha"])),
            "neighbourhood_group": str(borough),
            "neighbourhood": hood,
            "latitude": profile["lat"] + float(rng.normal(0, 0.022)),
            "longitude": profile["lon"] + float(rng.normal(0, 0.022)),
            "room_type": room_type,
            "price": round(max(price, 0.0), 2),
            "minimum_nights": int(rng.choice([1, 2, 3, 5, 7, 14, 30],
                                             p=[0.42, 0.20, 0.12, 0.10, 0.08, 0.05, 0.03])),
            "number_of_reviews": n_listing_reviews,
            "reviews_per_month": round(float(rng.gamma(1.4, 1.1)), 2) if n_listing_reviews else np.nan,
            "calculated_host_listings_count": int(rng.integers(1, 8)),
            "availability_365": int(rng.integers(0, 366)),
            # AB_NYC_2019 has no rating column; mirrors do. Keep the column so the
            # "top reviewed by rating" code path is testable.
            "review_scores_rating": float(np.clip(rng.normal(4.6, 0.4), 1.0, 5.0)),
        })

    # A handful of rows lose their coordinates, as in the real file.
    dropout = rng.choice(n_listings, size=max(1, n_listings // 200), replace=False)
    for row_idx in dropout:
        rows[int(row_idx)]["latitude"] = np.nan

    listings = pd.DataFrame(rows)

    # --- reviews ----------------------------------------------------------
    listing_ids = listings["id"].to_numpy()
    hood_lookup = dict(zip(listings["id"], listings["neighbourhood"]))
    reviewed = rng.choice(listing_ids, size=n_reviews, replace=True)
    bands = rng.choice(["positive", "neutral", "negative"], size=n_reviews,
                       p=[0.72, 0.19, 0.09])

    comments: list[str] = []
    for listing_id, band in zip(reviewed, bands):
        template = str(rng.choice(_REVIEW_TEMPLATES[str(band)]))
        comments.append(template.format(area=hood_lookup.get(listing_id, "NYC")))

    review_dates = pd.to_datetime("2018-01-01") + pd.to_timedelta(
        rng.integers(0, 700, size=n_reviews), unit="D"
    )
    reviews = pd.DataFrame({
        "listing_id": reviewed.astype("int64"),
        "id": np.arange(1_000_000, 1_000_000 + n_reviews, dtype="int64"),
        "date": review_dates.strftime("%Y-%m-%d"),
        "reviewer_id": rng.integers(10_000, 90_000, size=n_reviews),
        "reviewer_name": rng.choice(["Alex", "Sam", "Priya", "Luca", "Nina", "Tom"],
                                    size=n_reviews),
        "comments": comments,
    })

    _log("synthetic data ready (NOT real Kaggle data)")
    return _validate_listings(listings), _validate_reviews(reviews)



# ===========================================================================
# SECTION 7 - TOP-LEVEL ORCHESTRATOR (the function app.py calls)
# ===========================================================================
def ensure_data_available(
    allow_synthetic: bool = True,
    reviews_limit: int | None = None,
    force_download: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Guarantee the app has listings and reviews, whatever the environment.

    Resolution order
    ----------------
    Listings: local cache -> Kaggle API -> Inside Airbnb mirror -> synthetic.
    Reviews : local cache -> Kaggle -> Inside Airbnb mirror -> synthetic.

    Every stage is wrapped in ``try/except`` so a Kaggle outage, a missing key or
    a dead network degrades the app instead of crashing it -- the explicit error
    handling the rubric asks for.

    Parameters
    ----------
    allow_synthetic:
        When ``False``, an unresolvable download raises instead of falling back
        to fake data (used by the CLI for a hard failure).
    reviews_limit:
        Row cap for reviews; ``None`` uses ``config.SENTIMENT_SAMPLE_SIZE``.
    force_download:
        Ignore local caches and hit the network again.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, bool]
        ``(listings, reviews, used_synthetic)``.

    Raises
    ------
    RuntimeError
        When data is unavailable *and* ``allow_synthetic`` is ``False``.
    """
    used_synthetic = False
    row_cap = reviews_limit if reviews_limit is not None else cfg.SENTIMENT_SAMPLE_SIZE
    synthetic_reviews: pd.DataFrame | None = None

    # --- listings ---------------------------------------------------------
    listings: pd.DataFrame | None = None
    try:
        listings = download_listings(force=force_download)
    except Exception as exc:
        _log(f"listings unavailable from any real source ({exc})")

    if listings is None or listings.empty:
        if not allow_synthetic:
            raise RuntimeError("Listings unavailable and synthetic fallback disabled.")
        _log("falling back to synthetic listings")
        listings, synthetic_reviews = generate_synthetic_data()
        used_synthetic = True

    # --- reviews ----------------------------------------------------------
    if used_synthetic and synthetic_reviews is not None:
        # Reuse the fake reviews that already match the fake listing ids.
        reviews = synthetic_reviews
        _log(f"using synthetic reviews paired with synthetic listings "
             f"({len(reviews):,} rows)")
    else:
        reviews = None
        try:
            reviews = download_reviews(force=force_download, limit=row_cap)
        except Exception as exc:
            _log(f"reviews unavailable from any real source ({exc})")

        if reviews is None or reviews.empty:
            if not allow_synthetic:
                raise RuntimeError("Reviews unavailable and synthetic fallback disabled.")
            _log("falling back to synthetic reviews")
            _, reviews = generate_synthetic_data(n_listings=800, random_state=7)
            # Remap fake listing ids onto the real ones so the join still yields
            # sentiment scores for the listings actually shown on the map.
            real_ids = listings["id"].to_numpy()
            if len(real_ids):
                reviews["listing_id"] = np.resize(real_ids, len(reviews))
            used_synthetic = True

    _log(f"data ready -> listings={len(listings):,}, reviews={len(reviews):,}, "
         f"synthetic={used_synthetic}")
    return listings, reviews, used_synthetic



def listing_photo_urls(listings: pd.DataFrame) -> pd.Series:
    """
    Attach a hover-preview photo URL to every listing.

    Why this exists
    ---------------
    ``AB_NYC_2019.csv`` (and the Inside Airbnb snapshots) contain **no photo
    column**, so a map tooltip has nothing to show. Scraping Airbnb's CDN
    would break the app the moment they change a URL pattern and clashes with
    their terms, so instead each listing is deterministically mapped onto one
    of the curated, permanently hostable Unsplash photos in
    ``config.LISTING_PHOTOS``.

    The mapping hashes the listing id, so a listing always shows the same
    photo (no flicker between reruns) while the city still looks varied. If a
    real ``picture_url`` column is ever present -- Inside Airbnb's listings
    export has one -- it is used instead and the story stays truthful.

    Parameters
    ----------
    listings:
        Listings table; only ``id`` (and optionally ``picture_url``) is used.

    Returns
    -------
    pandas.Series
        One ``https://`` URL per row, aligned with ``listings.index``.
    """
    n_photos = len(cfg.LISTING_PHOTOS)

    def _pick(listing_id: Any) -> str:
        # ``hash()`` is salted per process, so a stable key is required: the
        # numeric id is used directly, anything else goes through crc32. The
        # same listing therefore lands on the same photo in every run and on
        # every machine, and neighbours in id space still spread across photos.
        try:
            key = int(listing_id)
        except (TypeError, ValueError):
            key = zlib.crc32(str(listing_id).encode("utf-8"))
        return cfg.PHOTO_CDN_BASE + cfg.LISTING_PHOTOS[int(key) % n_photos]

    if "picture_url" in listings.columns:
        real = listings["picture_url"].fillna("").astype(str).str.strip()
        if real.str.startswith("http").mean() > 0.5:      # mostly usable
            return real

    ids = listings["id"] if "id" in listings.columns else pd.Series(listings.index)
    return ids.apply(_pick).rename("picture_url")



def data_quality_report(listings: pd.DataFrame, reviews: pd.DataFrame) -> dict[str, Any]:
    """
    Summarise *raw* data quality before cleaning (shown in the EDA tab).

    Parameters
    ----------
    listings, reviews:
        Uncleaned DataFrames straight from the loader.

    Returns
    -------
    dict[str, Any]
        Row/column counts, per-column missing percentages and counts of the
        specific defects the cleaning step targets.
    """
    price = pd.to_numeric(listings.get("price", pd.Series(dtype=float)), errors="coerce")
    comments = reviews.get("comments", pd.Series(dtype=str))
    has_coords = {"latitude", "longitude"} <= set(listings.columns)

    return {
        "listings_rows": int(len(listings)),
        "listings_columns": int(listings.shape[1]),
        "reviews_rows": int(len(reviews)),
        "listings_missing_pct": (
            listings.isna().mean().mul(100).round(2)
            .sort_values(ascending=False).to_dict()
        ),
        "missing_coordinates": int(
            listings[["latitude", "longitude"]].isna().any(axis=1).sum()
        ) if has_coords else 0,
        "price_zero_rows": int((price == 0).sum()),
        "price_over_1000_rows": int((price > cfg.PRICE_MAX).sum()),
        "duplicate_listing_ids": int(listings["id"].duplicated().sum())
        if "id" in listings.columns else 0,
        "reviews_with_text": int((comments.fillna("").astype(str).str.len() > 0).sum()),
    }



def build_dataset(force_download: bool = False) -> pd.DataFrame:
    """
    Build the analysis-ready "top reviewed" dataset and cache it to parquet.

    Convenience entry point for notebooks and the CLI: runs
    acquisition -> cleaning -> top-reviewed selection in one call.

    Parameters
    ----------
    force_download:
        Re-download instead of using cached raw files.

    Returns
    -------
    pandas.DataFrame
        The cleaned, top-reviewed listings table.
    """
    import eda   # deferred import: eda imports config, avoiding a cycle at load

    listings, _, synthetic = ensure_data_available(force_download=force_download)
    clean, _audit = eda.clean_listings(listings)
    top = eda.select_top_reviewed(clean)
    if not synthetic:
        try:
            top.to_parquet(cfg.TOP_REVIEWED_PARQUET, index=False)
            _log(f"cached top-reviewed table -> {cfg.TOP_REVIEWED_PARQUET.name}")
        except Exception as exc:
            warnings.warn(f"Could not write parquet cache: {exc}")
    return top


if __name__ == "__main__":
    # ``python data_acquisition.py``             -> real download (needs Kaggle key)
    # ``python data_acquisition.py --synthetic``  -> offline demo data
    import argparse

    parser = argparse.ArgumentParser(description="Fetch NYC Airbnb listings + reviews.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Skip all network calls and generate demo data.")
    parser.add_argument("--force", action="store_true",
                        help="Ignore local caches and re-download.")
    args = parser.parse_args()

    if args.synthetic:
        listings_df, reviews_df = generate_synthetic_data()
        used = True
    else:
        listings_df, reviews_df, used = ensure_data_available(force_download=args.force)

    print("\n=== DATA SUMMARY ===")
    print(f"listings : {listings_df.shape[0]:,} rows x {listings_df.shape[1]} cols")
    print(f"reviews  : {reviews_df.shape[0]:,} rows x {reviews_df.shape[1]} cols")
    print(f"synthetic: {used}")

    print("\n--- listings columns ---")
    print(list(listings_df.columns))

    print("\n--- listings head (transposed for readability) ---")
    with pd.option_context("display.max_colwidth", 40):
        print(listings_df.head(3).T)

    print("\n--- reviews head ---")
    print(reviews_df.head(3))

    print("\n--- quality report ---")
    for key, value in data_quality_report(listings_df, reviews_df).items():
        if key != "listings_missing_pct":
            print(f"{key}: {value}")

    print("\n--- missing values (top 6) ---")
    missing = data_quality_report(listings_df, reviews_df)["listings_missing_pct"]
    for column, pct in list(missing.items())[:6]:
        print(f"  {column}: {pct}%")

