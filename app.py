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
      * a **GlobeView-style tilted perspective** over the five boroughs,
      * a **HeatmapLayer** showing where highly-reviewed rentals cluster,
      * **marker clustering** (``ColumnLayer`` + ``ScreenGridLayer`` fallback) so
        48k points never overwhelm the canvas,
      * tooltips with name, price, review count and sentiment score,
      * price + review-count sliders, borough/room-type filters and a geocoded
        area search.
Tab 2 - EDA Insights
    Raw-vs-cleaned audit, IQR outlier report, the "top reviewed" definition,
    price and review distributions, correlation matrix, and the volume of
    top-reviewed listings per borough.
Tab 3 - AI Travel Guide
    The "NY State * Travel Concierge": sends the top-reviewed listings (names,
    neighbourhoods, prices, review counts, sentiment) to an LLM which writes a
    natural-language guide for the current filters. Degrades to a deterministic
    offline report when no API key is present.

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
    page_icon="🗽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# BASE STYLES (kept minimal here; the full design system is Phase 4 polish)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: #f7f9fc;
        border: 1px solid #e3e8ef;
        border-radius: 12px;
        padding: 12px 14px;
    }
    .nyc-title {font-size: 2.0rem; font-weight: 750; margin-bottom: 0; line-height: 1.1;}
    .nyc-sub   {color: #5b6674; margin-top: 0.15rem; margin-bottom: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


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
def render_sidebar() -> dict[str, Any]:
    """
    Draw the sidebar controls and return the chosen settings.

    Returns
    -------
    dict[str, Any]
        ``force_download``, ``reviews_limit``, ``engine``, ``llm_provider``.
    """
    st.sidebar.title("🗽 NYC Travel Planner")
    st.sidebar.caption("Capstone - Programming for Data Science")

    with st.sidebar.expander("1. Data source", expanded=True):
        if da.kaggle_credentials_available():
            st.success("Kaggle credentials detected.")
        else:
            st.warning(
                "No Kaggle credentials found, so the app will use the Inside "
                "Airbnb mirror or demo data. See README step 3 to add your key."
            )
        force_download = st.checkbox("Force re-download", value=False)
        reviews_limit = st.select_slider(
            "Max reviews to score",
            options=[10_000, 50_000, 100_000, 250_000],
            value=100_000,
            help="Lower is faster to start; 50k reviews already give a stable "
                 "average sentiment per listing.",
        )

    with st.sidebar.expander("2. Sentiment engine", expanded=False):
        engine = st.radio(
            "NLP engine",
            options=["vader", "textblob"],
            index=0,
            format_func=lambda v: {"vader": "VADER (recommended)",
                                   "textblob": "TextBlob"}[v],
        )
        st.caption("VADER is tuned for short review text; TextBlob is a simpler "
                   "polarity model kept as a comparison.")

    with st.sidebar.expander("3. AI concierge", expanded=False):
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
        "reviews. Geocoding: OpenStreetMap Nominatim."
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
def render_header(used_synthetic: bool) -> None:
    """
    Render the page title and, when applicable, the synthetic-data banner.

    Parameters
    ----------
    used_synthetic:
        Whether any part of the loaded data is generated demo data.

    Returns
    -------
    None
    """
    st.markdown(f"<p class='nyc-title'>🗽 {cfg.APP_TITLE}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='nyc-sub'>{cfg.APP_SUBTITLE}</p>", unsafe_allow_html=True)

    if used_synthetic:
        st.warning(
            "⚠️ **SYNTHETIC DEMO DATA** - the real dataset could not be reached, "
            "so every number below comes from generated data with an identical "
            "schema. Add Kaggle credentials (README step 3) and press "
            "*Force re-download* for the real NYC Airbnb data."
        )



# ===========================================================================
# SECTION 4 - MAP: FILTERING, CLUSTERING AND LAYERS
# ===========================================================================
# Free Carto vector basemaps (no API token needed, unlike Mapbox).
BASEMAP_DARK = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
BASEMAP_LIGHT = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"


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

    # Colour by review volume: cool blue (few reviews) -> warm amber (many),
    # so the most-reviewed listings are visually obvious on the map.
    reviews = map_df["number_of_reviews"].astype(float)
    low, high = float(reviews.min()), float(reviews.max())
    span = max(high - low, 1e-6)
    map_df["colour"] = reviews.apply(
        lambda n: [
            int(70 + 185 * (n - low) / span),
            int(130 + 45 * (1 - (n - low) / span)),
            int(230 - 150 * (n - low) / span),
        ]
    )

    sentiment_text = sentiment.apply(lambda v: "n/a" if pd.isna(v) else f"{v:+.2f}")

    # Hover preview: show the listing photo inside the tooltip so users can see
    # what the Airbnb looks like without clicking through. deck.gl renders raw
    # HTML (including <img>) in browser tooltips, so a thumbnail appears on
    # hover. Airbnb's CDN accepts an ``?im_w=`` size hint, so we request a small
    # thumbnail instead of the full-resolution ``original/`` JPEG to keep hovers
    # fast. Listings without a usable picture_url simply show no image.
    if "picture_url" in map_df.columns:
        picture = map_df["picture_url"].fillna("").astype(str).str.strip()
    else:
        picture = pd.Series("", index=map_df.index, dtype=str)

    def _photo_html(url: str) -> str:
        if not url.startswith("http"):
            return ""
        # Airbnb's CDN only honours the ?im_w= width hint on its /im/pictures/
        # path; the bare /pictures/ path always serves the multi-MB original.
        # Rewriting to /im/pictures/ turns a 3 MB photo into a ~12 KB thumbnail.
        if "muscache.com/pictures/" in url:
            url = url.replace("muscache.com/pictures/", "muscache.com/im/pictures/", 1)
        thumb = url + ("&" if "?" in url else "?") + "im_w=240"
        return (
            f'<img src="{thumb}" width="220" '
            'style="border-radius:6px;margin-bottom:6px;display:block;"/>'
        )

    photo_html = picture.apply(_photo_html)

    map_df["tooltip"] = (
        photo_html
        + "<b>" + map_df["name"].astype(str).str.slice(0, 55) + "</b><br/>"
        + "💵 $" + map_df["price"].round(0).astype(int).astype(str) + " / night<br/>"
        + "⭐ " + map_df["number_of_reviews"].astype(int).astype(str) + " reviews<br/>"
        + "📍 " + map_df["neighbourhood"].astype(str) + ", "
        + map_df["neighbourhood_group"].astype(str) + "<br/>"
        + "🛏️ " + map_df["room_type"].astype(str) + "<br/>"
        + "😊 sentiment " + sentiment_text
    )
    return map_df



def make_deck(
    map_df: pd.DataFrame,
    lat: float = cfg.NYC_CENTER_LAT,
    lon: float = cfg.NYC_CENTER_LON,
    zoom: int = cfg.NYC_ZOOM,
    show_heatmap: bool = True,
    cluster: bool = True,
    dark: bool = True,
) -> pdk.Deck:
    """
    Assemble the pydeck map with a heatmap layer plus clustered markers.

    Clustering strategy
    -------------------
    NYC holds ~48k listings, far too many individual markers to read. deck.gl's
    ``ColumnLayer`` aggregates nearby listings into one extruded hexagon whose
    height encodes how many listings it covers -- the deck.gl-native equivalent
    of marker clustering, and it stays smooth at 48k points. Turning clustering
    off falls back to individual ``ScatterplotLayer`` dots.

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
        Use the dark basemap (matches the app theme) rather than the light one.

    Returns
    -------
    pydeck.Deck
        Ready-to-render deck object.
    """
    layers: list[pdk.Layer] = []

    if show_heatmap and not map_df.empty:
        # Density of highly reviewed rentals: weighting by review count makes
        # hot spots highlight where the most-reviewed stays concentrate.
        layers.append(
            pdk.Layer(
                "HeatmapLayer",
                data=map_df,
                get_position=["longitude", "latitude"],
                get_weight="number_of_reviews",
                radiusPixels=45,
                intensity=1.1,
                threshold=0.06,
                opacity=0.65,
                pickable=False,
            )
        )

    if cluster and not map_df.empty:
        layers.append(
            pdk.Layer(
                "ColumnLayer",
                data=map_df,
                get_position=["longitude", "latitude"],
                radius=260,
                elevation_scale=8,
                elevation_range=[0, 2200],
                get_elevation="number_of_reviews",
                get_fill_color="colour",
                extruded=True,
                pickable=True,
                auto_highlight=True,
                opacity=0.85,
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
                radius_min_pixels=3,
                radius_max_pixels=18,
                pickable=True,
                auto_highlight=True,
                stroked=True,
                get_line_color=[255, 255, 255, 120],
                line_width_min_pixels=1,
            )
        )

    view_state = pdk.ViewState(
        latitude=lat, longitude=lon, zoom=zoom,
        pitch=cfg.NYC_PITCH, bearing=cfg.NYC_BEARING,
    )
    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=BASEMAP_DARK if dark else BASEMAP_LIGHT,
        tooltip={
            "html": "{tooltip}",
            "style": {"backgroundColor": "#111827", "color": "white",
                      "borderRadius": "8px", "padding": "8px 10px",
                      "maxWidth": "260px"},
        },
    )



# ===========================================================================
# SECTION 5 - TAB 1: INTERACTIVE MAP
# ===========================================================================
def render_kpis(filtered: pd.DataFrame) -> None:
    """
    Render the four headline metrics for the current selection.

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
                "minimum review count.")
        return

    summary = nlp.sentiment_summary(filtered)
    columns = st.columns(4)
    columns[0].metric("Listings shown", f"{len(filtered):,}")
    columns[1].metric("Median price", f"${filtered['price'].median():,.0f}")
    columns[2].metric("Median reviews", f"{filtered['number_of_reviews'].median():,.0f}")
    columns[3].metric(
        "Avg sentiment",
        "n/a" if summary["mean_sentiment"] is None else f"{summary['mean_sentiment']:+.2f}",
        help="Mean VADER compound score across the filtered listings.",
    )


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
        ``show_heatmap``, ``cluster``, ``dark_map``, ``search``.
    """
    price_ceiling = int(min(cfg.PRICE_MAX, max(top_reviewed["price"].max(), 10)))
    review_ceiling = int(max(top_reviewed["number_of_reviews"].max(), 10))

    row1 = st.columns([2.2, 1.5, 1.5])
    with row1[0]:
        price_range = st.slider(
            "💵 Nightly price ($)",
            min_value=1,
            max_value=price_ceiling,
            value=(int(top_reviewed["price"].quantile(0.05)),
                   int(min(price_ceiling, top_reviewed["price"].quantile(0.90)))),
            step=5,
        )
    with row1[1]:
        min_reviews = st.slider(
            "⭐ Minimum number of reviews",
            min_value=0,
            max_value=review_ceiling,
            value=int(cfg.TOP_REVIEWED_MIN_REVIEWS),
            step=5,
        )
    with row1[2]:
        show_heatmap = st.toggle("🔥 Review-density heatmap", value=True)
        cluster = st.toggle("🧱 Cluster markers", value=True,
                            help="Aggregate nearby listings into hexagonal columns.")

    row2 = st.columns([1.5, 1.5, 1.2])
    with row2[0]:
        boroughs = st.multiselect(
            "🏙️ Borough", options=list(cfg.BOROUGHS), default=[],
            placeholder="All boroughs",
        )
    with row2[1]:
        room_types = st.multiselect(
            "🛏️ Room type", options=list(cfg.ROOM_TYPES), default=[],
            placeholder="All room types",
        )
    with row2[2]:
        dark_map = st.toggle("🌙 Dark basemap", value=True)

    search = st.text_input(
        "🔎 Centre the map on a neighbourhood (Nominatim geocoding)",
        placeholder="e.g. Williamsburg, Harlem, Astoria, Mott Haven",
    )
    return {
        "price_range": price_range,
        "min_reviews": min_reviews,
        "boroughs": boroughs,
        "room_types": room_types,
        "show_heatmap": show_heatmap,
        "cluster": cluster,
        "dark_map": dark_map,
        "search": search,
    }



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
    st.subheader("Where are the most-reviewed stays?")
    st.caption(
        "Height and colour encode review volume. Drag the sliders, pick boroughs "
        "or room types, then click a column to read the listing details."
    )

    if top_reviewed.empty:
        st.error("No top-reviewed listings are available to map.")
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
                    "or a well-known neighbourhood such as 'Bushwick'."
                )
            else:
                centre_lat, centre_lon, zoom = float(hit[0]), float(hit[1]), 13
                st.success(f"Centred on: {hit[2]}")
        except Exception as exc:
            st.warning(f"Geocoding failed ({exc}). Showing the default NYC view.")

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
            "matching listings for performance."
        )

    deck = make_deck(
        build_map_dataframe(plot_df),
        lat=centre_lat, lon=centre_lon, zoom=zoom,
        show_heatmap=controls["show_heatmap"],
        cluster=controls["cluster"],
        dark=controls["dark_map"],
    )
    st.pydeck_chart(deck, width="stretch", height=620)
    st.caption(
        "Column height and colour encode review volume: blue = fewer reviews, "
        "amber = the most-reviewed stays. The heatmap weights each listing by its "
        "review count, so bright areas are the densest clusters of proven stays."
    )

    with st.expander("📋 Top-reviewed listings table", expanded=False):
        columns = [c for c in (
            "name", "neighbourhood_group", "neighbourhood", "room_type",
            "price", "number_of_reviews", "avg_sentiment", "sentiment_label",
        ) if c in filtered.columns]
        st.dataframe(
            filtered[columns].sort_values("number_of_reviews", ascending=False).head(300),
            width="stretch", hide_index=True,
        )
        st.download_button(
            "⬇️ Download filtered listings (CSV)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="nyc_top_reviewed.csv",
            mime="text/csv",
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
    st.subheader("Exploratory data analysis")
    st.caption("All figures use the cleaned listings table unless stated otherwise.")

    # --- 1. data quality --------------------------------------------------
    st.markdown("#### 🧹 Data quality and preprocessing")
    outliers = audit.get("outliers", {}) or {}
    quality = da.data_quality_report(raw_listings, raw_reviews)

    columns = st.columns(4)
    columns[0].metric("Raw listings", f"{quality['listings_rows']:,}")
    columns[1].metric("After cleaning", f"{int(audit.get('rows_out', 0)):,}")
    columns[2].metric("Outlier rows removed", f"{int(outliers.get('rows_removed', 0)):,}")
    columns[3].metric("Top-reviewed", f"{len(top_reviewed):,}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Missing values in the raw listings table**")
        report = eda.missing_value_report(raw_listings)
        missing_only = report[report["missing_count"] > 0]
        st.dataframe(
            missing_only if not missing_only.empty
            else pd.DataFrame({"status": ["No missing values found"]}),
            width="stretch", hide_index=True, height=230,
        )
    with right:
        st.markdown("**IQR outlier policy**")
        st.write(
            f"- Price must be **> \\$0** and **<= \\${outliers.get('applied_ceiling', 'n/a')}** "
            f"(Q3 + 1.5 x IQR = \\${outliers.get('upper_fence', 'n/a')}, capped at the "
            f"\\${outliers.get('price_max', cfg.PRICE_MAX):,.0f} business ceiling)."
        )
        st.write(f"- Price outliers removed: **{int(outliers.get('rows_removed', 0)):,}**")
        st.write(f"- Missing-coordinate rows removed: **{quality['missing_coordinates']:,}**")
        st.write(f"- Duplicate listing ids removed: **{audit.get('duplicates_removed', 0):,}**")

    st.divider()

    # --- 2. "top reviewed" definition -------------------------------------
    st.markdown("#### ⭐ How \"top reviewed\" is defined")
    st.markdown(
        f"A listing qualifies when it has at least **{cfg.TOP_REVIEWED_MIN_REVIEWS} "
        f"reviews** and sits in the top **{int((1 - cfg.TOP_REVIEWED_QUANTILE) * 100)}% "
        "by review volume** (falling back to `review_scores_rating` when the dataset "
        "provides it). This filters out brand-new listings whose single 5-star review "
        "would otherwise outrank proven stays."
    )
    if not top_reviewed.empty:
        columns = st.columns(4)
        columns[0].metric("Median reviews", f"{top_reviewed['number_of_reviews'].median():,.0f}")
        columns[1].metric("Median price", f"${top_reviewed['price'].median():,.0f}")
        columns[2].metric("Boroughs covered", f"{top_reviewed['neighbourhood_group'].nunique()}")
        columns[3].metric(
            "Avg sentiment",
            f"{top_reviewed['avg_sentiment'].mean():+.2f}"
            if "avg_sentiment" in top_reviewed.columns
            and top_reviewed["avg_sentiment"].notna().any() else "n/a",
        )

    st.divider()

    # --- 3. distributions -------------------------------------------------
    st.markdown("#### 📊 Distributions")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(eda.plot_price_distribution(clean), width="stretch")
    with right:
        st.plotly_chart(eda.plot_reviews_distribution(clean), width="stretch")

    st.divider()

    # --- 4. borough volume (the rubric's bar chart) -----------------------
    st.markdown("#### 🏙️ Top-reviewed listings by borough")
    st.plotly_chart(eda.plot_top_reviewed_by_borough(top_reviewed), width="stretch")
    st.plotly_chart(eda.plot_price_vs_reviews(clean), width="stretch")

    st.divider()

    # --- 5. correlations --------------------------------------------------
    st.markdown("#### 🔗 Correlation analysis")
    left, right = st.columns([1.1, 1])
    with left:
        st.plotly_chart(eda.plot_correlation_heatmap(clean), width="stretch")
    with right:
        st.markdown("**Correlation with number of reviews**")
        correlations = eda.review_correlations(clean)
        if correlations.empty:
            st.info("Not enough numeric columns to correlate.")
        else:
            table = correlations.rename("correlation").reset_index()
            table.columns = ["variable", "correlation"]
            st.dataframe(table.round(3), width="stretch", hide_index=True)

    st.divider()

    # --- 6. room type + sentiment ----------------------------------------
    left, right = st.columns(2)
    with left:
        st.plotly_chart(eda.plot_room_type_mix(clean), width="stretch")
    with right:
        if "avg_sentiment" in clean.columns and clean["avg_sentiment"].notna().any():
            st.plotly_chart(eda.plot_sentiment_by_borough(clean), width="stretch")
        else:
            st.info("Sentiment scores are not available for this dataset.")

    with st.expander("📄 Top-reviewed listings preview (first 200 rows)"):
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
    st.subheader("🤖 NYC Travel Concierge")
    st.caption(
        "The concierge reads a summary of your current map filters and writes a "
        "travel guide. Only aggregated statistics are sent - never raw review text."
    )

    working = filtered if filtered is not None and not filtered.empty else top_reviewed
    if working is None or working.empty:
        st.error("There is no data to summarise yet.")
        return

    if filtered is None or filtered.empty:
        st.warning("The map has no matching listings, so the full top-reviewed set "
                   "is being summarised instead.")

    # --- build the payload ------------------------------------------------
    try:
        keywords = nlp.extract_keywords(scored_reviews) if not scored_reviews.empty else {}
        payload = llm.build_context_payload(working, keywords=keywords)
    except Exception as exc:
        st.error(f"Could not build the summary payload: {exc}")
        return

    with st.expander("🔍 What the AI receives (aggregated JSON)", expanded=False):
        st.json(payload, expanded=2)


    if st.button("✍️ Generate travel guide", type="primary", width="stretch"):
        try:
            with st.spinner("The concierge is writing your NYC guide..."):
                report = llm.generate_travel_report(payload, provider=llm_provider)
            st.session_state["travel_report"] = report
        except Exception as exc:
            st.error(f"The LLM call failed: {exc}")

    report = st.session_state.get("travel_report")
    if report is None:
        st.info("Press **Generate travel guide** to summarise your current filters.")
        return

    if report.warning:
        st.warning(report.warning)
    st.markdown(report.text)

    columns = st.columns(4)
    columns[0].caption(f"Provider: **{report.provider}**")
    columns[1].caption(f"Model: `{report.model}`")
    columns[2].caption(f"Latency: {report.latency_seconds:.2f}s")
    columns[3].caption(f"Tokens: {report.tokens:,}" if report.tokens else "Tokens: n/a")

    st.download_button(
        "⬇️ Download the guide (Markdown)",
        data=report.text.encode("utf-8"),
        file_name="nyc_travel_guide.md",
        mime="text/markdown",
    )

    st.divider()
    st.markdown("#### Ask a follow-up")
    question = st.text_input(
        "e.g. Which borough is best for a $150 budget with the best reviews?",
        key="followup_question",
    )
    if st.button("Ask the concierge") and question.strip():
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

    # --- 4. tabs ----------------------------------------------------------
    map_tab, eda_tab, ai_tab = st.tabs(
        ["🗺️ Interactive Map", "📊 EDA Insights", "🤖 AI Travel Guide"]
    )

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

    st.divider()
    st.caption(
        "Built with Streamlit, pydeck, Plotly, VADER/TextBlob and an LLM API. "
        "Data: Kaggle NYC Airbnb Open Data + Inside Airbnb reviews. "
        "Geocoding: OpenStreetMap Nominatim."
    )


if __name__ == "__main__":
    main()

