"""
eda.py
======

**Phase 2 - Exploratory Data Analysis & Preprocessing (20% of the grade)**

Responsibilities
----------------
1. **Cleaning** -- duplicates, missing values, missing coordinates, $0 prices and
   IQR-based extreme outlier removal.
2. **"Top reviewed" selection** -- the rubric's core concept: rank listings by
   ``number_of_reviews`` (or ``review_scores_rating``) and keep the best.
3. **Statistics** -- descriptive stats, correlation matrices, regional volume.
4. **Visualisations** -- Plotly figures the Streamlit EDA tab renders directly.

Every function takes a DataFrame and returns a DataFrame, dict or Plotly figure.
No Streamlit calls live here, so the module is fully testable on its own::

    python eda.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import config as cfg


# ===========================================================================
# SECTION 1 - DATA CLEANING
# ===========================================================================
def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a tidy per-column missing-value table.

    Parameters
    ----------
    df:
        Any DataFrame (usually the raw listings table).

    Returns
    -------
    pandas.DataFrame
        ``column``, ``missing_count``, ``missing_pct``, ``dtype`` sorted by
        missing percentage descending.
    """
    report = pd.DataFrame({
        "column": list(df.columns),
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        "dtype": [str(df[c].dtype) for c in df.columns],
    })
    report["missing_pct"] = (
        report["missing_count"] / max(len(df), 1) * 100
    ).round(2)
    return report.sort_values("missing_pct", ascending=False).reset_index(drop=True)


def handle_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Impute or drop missing values using a documented, defensible strategy.

    Strategy
    --------
    * ``name`` -> ``"Unnamed listing"`` (the row is still mappable)
    * ``host_name`` -> ``"Unknown host"``
    * ``neighbourhood*`` / ``room_type`` -> ``"Unknown"``
    * ``reviews_per_month`` / ``number_of_reviews`` -> ``0`` (no reviews *is* zero)
    * ``minimum_nights`` / ``availability_365`` -> column median (robust to skew)
    * ``review_scores_rating`` -> column median: a missing score usually means
      "too few reviews to score" rather than "bad"
    * rows with no coordinates -> dropped (cannot appear on the map)
    * rows with no ``price`` -> dropped (price is the analysis target)

    Parameters
    ----------
    df:
        Listings DataFrame.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, str]]
        The cleaned copy plus an ``{column: action}`` audit log so the UI can
        explain exactly what changed.
    """
    clean = df.copy()
    actions: dict[str, str] = {}

    text_defaults = {
        "name": "Unnamed listing",
        "host_name": "Unknown host",
        "neighbourhood_group": "Unknown",
        "neighbourhood": "Unknown",
        "room_type": "Unknown",
    }
    for column, default in text_defaults.items():
        if column in clean.columns and clean[column].isna().any():
            clean[column] = clean[column].fillna(default)
            actions[column] = f"filled with '{default}'"

    for column in ("number_of_reviews", "reviews_per_month"):
        if column in clean.columns and clean[column].isna().any():
            clean[column] = clean[column].fillna(0)
            actions[column] = "filled with 0"

    for column in ("minimum_nights", "availability_365", "review_scores_rating"):
        if column in clean.columns and clean[column].isna().any():
            median = float(clean[column].median())
            clean[column] = clean[column].fillna(median)
            actions[column] = f"filled with median ({median:.2f})"

    # Coordinates are mandatory: a listing we cannot place is useless on a map.
    if {"latitude", "longitude"} <= set(clean.columns):
        before = len(clean)
        clean = clean.dropna(subset=["latitude", "longitude"])
        if len(clean) < before:
            actions["latitude/longitude"] = (
                f"dropped {before - len(clean)} row(s) with missing coordinates"
            )

    before = len(clean)
    clean = clean.dropna(subset=["price"]).reset_index(drop=True)
    if len(clean) < before:
        actions["price"] = f"dropped {before - len(clean)} row(s) with no price"

    return clean, actions


def remove_price_outliers(
    df: pd.DataFrame,
    iqr_multiplier: float = cfg.IQR_MULTIPLIER,
    price_max: float = cfg.PRICE_MAX,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Remove $0 prices and extreme price outliers using the IQR rule.

    Method
    ------
    1. Drop ``price <= 0`` -- a free stay is a data error, not a listing.
    2. Compute ``Q1``/``Q3`` and the upper fence ``Q3 + k * IQR``.
    3. Keep ``0 < price <= min(upper_fence, price_max)`` so the statistical fence
       and the rubric's business ceiling are both satisfied.

    Parameters
    ----------
    df:
        Listings with a numeric ``price`` column.
    iqr_multiplier:
        Tukey fence multiplier (1.5 is standard).
    price_max:
        Hard business cap, $ per night.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, float]]
        Filtered frame plus the thresholds used, for display in the UI.
    """
    if df.empty or "price" not in df.columns:
        return df.copy(), {"applied_ceiling": np.nan, "rows_removed": 0.0}

    working = df.copy()
    positive = working.loc[working["price"] > 0, "price"]
    if positive.empty:
        return working, {"applied_ceiling": float(price_max), "rows_removed": 0.0}

    q1 = float(positive.quantile(0.25))
    q3 = float(positive.quantile(0.75))
    upper_fence = q3 + iqr_multiplier * (q3 - q1)
    ceiling = float(min(upper_fence, price_max))

    filtered = working.loc[(working["price"] > 0) & (working["price"] <= ceiling)]
    return filtered.reset_index(drop=True), {
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(q3 - q1, 2),
        "upper_fence": round(upper_fence, 2),
        "price_max": float(price_max),
        "applied_ceiling": round(ceiling, 2),
        "rows_removed": float(len(working) - len(filtered)),
    }



def clean_listings(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Run the full cleaning pipeline and return an analysis-ready table.

    Steps
    -----
    1. Drop duplicate listing ids.
    2. ``handle_missing_values``
    3. ``remove_price_outliers`` (IQR fence + business ceiling)
    4. Drop listings demanding more than a year's minimum stay.
    5. Add derived columns the app relies on: ``region``, ``region_group``,
       ``price_band`` and ``popularity``.

    Parameters
    ----------
    df:
        Raw listings DataFrame.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, object]]
        ``(clean_df, audit_log)`` where the audit log records every row count so
        the UI can show exactly what was removed.
    """
    audit: dict[str, object] = {"rows_in": int(len(df))}
    if df.empty:
        return df.copy(), audit

    clean = df.drop_duplicates(subset=["id"]).copy()
    audit["duplicates_removed"] = audit["rows_in"] - len(clean)

    clean, missing_actions = handle_missing_values(clean)
    audit["missing_actions"] = missing_actions

    clean, outlier_info = remove_price_outliers(clean)
    audit["outliers"] = outlier_info

    if "minimum_nights" in clean.columns:
        before = len(clean)
        clean = clean.loc[clean["minimum_nights"].between(1, cfg.MIN_NIGHTS_MAX)]
        audit["min_nights_removed"] = before - len(clean)

    clean = clean.copy()

    # ``region`` is the traveller-facing geography used in every chart. NYC's five
    # boroughs are already meaningful markets, so region == borough; the three
    # smallest are additionally rolled into "Outer Boroughs" so the volume
    # comparison reads cleanly.
    clean["region"] = clean["neighbourhood_group"]
    clean["region_group"] = clean["neighbourhood_group"].where(
        clean["neighbourhood_group"].isin(["Manhattan", "Brooklyn"]),
        "Outer Boroughs",
    )

    clean["price_band"] = pd.cut(
        clean["price"],
        bins=[0, 75, 150, 250, 400, np.inf],
        labels=["Budget (<$75)", "Mid ($75-150)", "Premium ($150-250)",
                "Luxury ($250-400)", "Ultra (>$400)"],
        include_lowest=True,
    ).astype(str)

    # 0-1 blend of lifetime review volume and review velocity.
    clean["popularity"] = (
        clean["number_of_reviews"].fillna(0).clip(upper=300) / 300 * 0.6
        + clean["reviews_per_month"].fillna(0).clip(upper=10) / 10 * 0.4
    ).round(4)

    clean = clean.reset_index(drop=True)
    audit["rows_out"] = int(len(clean))
    return clean, audit



# ===========================================================================
# SECTION 2 - "TOP REVIEWED" SELECTION
# ===========================================================================
def select_top_reviewed(
    df: pd.DataFrame,
    metric: str = "number_of_reviews",
    min_reviews: int = cfg.TOP_REVIEWED_MIN_REVIEWS,
    min_rating: float = cfg.TOP_REVIEWED_MIN_RATING,
    quantile: float = cfg.TOP_REVIEWED_QUANTILE,
    limit: int | None = cfg.TOP_REVIEWED_LIMIT,
) -> pd.DataFrame:
    """
    Rank listings and keep the "top reviewed" ones.

    Definition (rubric: highest ``number_of_reviews`` OR
    ``review_scores_rating``)
    ------------------------------------------------------------------
    * ``metric="number_of_reviews"`` (default): keep listings with at least
      ``min_reviews`` reviews that also sit in the top ``1 - quantile`` of the
      review distribution. Requiring both avoids the classic trap where a brand
      new listing with two glowing reviews outranks a proven stay with 400.
    * ``metric="review_scores_rating"``: keep listings rated at least
      ``min_rating`` out of 5, still subject to the ``min_reviews`` floor so the
      score is meaningful. Used when the dataset provides scores -- AB_NYC_2019
      does not, Inside Airbnb mirrors do.

    A data-quality guard guarantees a non-empty result: if the thresholds are too
    strict for a small dataset, the filter relaxes to ``min_reviews`` and finally
    to the single highest-reviewed listing.

    Parameters
    ----------
    df:
        Cleaned listings.
    metric:
        ``"number_of_reviews"`` or ``"review_scores_rating"``.
    min_reviews:
        Minimum lifetime reviews.
    min_rating:
        Minimum rating out of 5 (rating metric only).
    quantile:
        Quantile cut for the review-volume metric (0.90 => top 10%).
    limit:
        Optional cap on returned rows, best first. ``None`` keeps all.

    Returns
    -------
    pandas.DataFrame
        The top-reviewed subset sorted best-first, with ``review_rank`` and
        ``top_reviewed_by`` columns.
    """
    if df.empty:
        return df.copy()

    working = df.copy()
    has_rating = (
        "review_scores_rating" in working.columns
        and working["review_scores_rating"].notna().any()
    )

    if metric == "review_scores_rating" and has_rating:
        mask = working["review_scores_rating"] >= min_rating
        if min_reviews > 0:
            mask &= working["number_of_reviews"] >= min_reviews
        selected = working.loc[mask]
        sort_columns = ["review_scores_rating", "number_of_reviews"]
        label = "review_scores_rating"
    else:
        # Review-volume mode: explicit threshold AND a quantile cut.
        quantile_cut = float(working["number_of_reviews"].quantile(quantile))
        threshold = max(float(min_reviews), quantile_cut)
        selected = working.loc[working["number_of_reviews"] >= threshold]
        sort_columns = ["number_of_reviews", "reviews_per_month"]
        label = "number_of_reviews"

    # --- data-quality guard: never return an empty "top" set ---------------
    if selected.empty:
        selected = working.loc[working["number_of_reviews"] >= min_reviews]
    if selected.empty:
        selected = working.sort_values("number_of_reviews", ascending=False).head(1)

    selected = selected.sort_values(
        by=[c for c in sort_columns if c in selected.columns],
        ascending=False,
    ).reset_index(drop=True)

    selected["review_rank"] = np.arange(1, len(selected) + 1)
    selected["top_reviewed_by"] = label
    if limit is not None and limit > 0:
        selected = selected.head(limit)
    return selected.reset_index(drop=True)



# ===========================================================================
# SECTION 3 - DESCRIPTIVE STATISTICS
# ===========================================================================
# Numeric columns correlated against review volume and price.
CORRELATION_COLUMNS: tuple[str, ...] = (
    "price",
    "number_of_reviews",
    "reviews_per_month",
    "minimum_nights",
    "availability_365",
    "calculated_host_listings_count",
)


def get_summary_stats(df: pd.DataFrame) -> dict[str, object]:
    """
    Compute the headline KPIs and the per-borough aggregation table.

    Parameters
    ----------
    df:
        Cleaned listings.

    Returns
    -------
    dict[str, object]
        Scalars plus ``borough_stats`` (one row per borough).
    """
    if df.empty:
        return {
            "n_listings": 0, "median_price": np.nan, "mean_price": np.nan,
            "median_reviews": np.nan, "n_neighbourhoods": 0, "n_hosts": 0,
            "pct_entire_home": np.nan, "borough_stats": pd.DataFrame(),
        }

    borough_stats = (
        df.groupby("neighbourhood_group")
        .agg(
            listings=("id", "size"),
            median_price=("price", "median"),
            median_reviews=("number_of_reviews", "median"),
            total_reviews=("number_of_reviews", "sum"),
        )
        .round(1)
        .sort_values("listings", ascending=False)
        .reset_index()
    )
    borough_stats["share_pct"] = (
        borough_stats["listings"] / borough_stats["listings"].sum() * 100
    ).round(1)

    return {
        "n_listings": int(len(df)),
        "median_price": float(df["price"].median()),
        "mean_price": float(df["price"].mean()),
        "median_reviews": float(df["number_of_reviews"].median()),
        "n_neighbourhoods": int(df["neighbourhood"].nunique()),
        "n_hosts": int(df["host_id"].nunique()),
        "pct_entire_home": float((df["room_type"] == "Entire home/apt").mean() * 100),
        "borough_stats": borough_stats,
    }


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation matrix over the key numeric variables.

    Parameters
    ----------
    df:
        Cleaned listings.

    Returns
    -------
    pandas.DataFrame
        Square matrix, empty when fewer than two columns are present.
    """
    columns = [c for c in CORRELATION_COLUMNS if c in df.columns]
    if len(columns) < 2:
        return pd.DataFrame()
    return df[columns].apply(pd.to_numeric, errors="coerce").corr().round(3)


def review_correlations(df: pd.DataFrame) -> pd.Series:
    """
    Correlation of every numeric feature with ``number_of_reviews``.

    Parameters
    ----------
    df:
        Cleaned listings.

    Returns
    -------
    pandas.Series
        Sorted descending, with ``number_of_reviews`` itself excluded.
    """
    matrix = get_correlation_matrix(df)
    if matrix.empty or "number_of_reviews" not in matrix.columns:
        return pd.Series(dtype=float)
    return matrix["number_of_reviews"].drop(
        labels=["number_of_reviews"]
    ).sort_values(ascending=False)


def top_neighbourhoods(df: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """
    Rank neighbourhoods by listing count, with median price and reviews.

    Parameters
    ----------
    df:
        Cleaned listings.
    n:
        Number of neighbourhoods to return.

    Returns
    -------
    pandas.DataFrame
        ``neighbourhood``, ``neighbourhood_group``, ``listings``,
        ``median_price``, ``median_reviews``.
    """
    if df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby("neighbourhood")
        .agg(
            neighbourhood_group=("neighbourhood_group", "first"),
            listings=("id", "size"),
            median_price=("price", "median"),
            median_reviews=("number_of_reviews", "median"),
        )
        .round(1)
        .reset_index()
        .sort_values("listings", ascending=False)
        .head(n)
    )
    return grouped.reset_index(drop=True)



# ===========================================================================
# SECTION 4 - VISUALISATIONS (Plotly figures, Streamlit-ready)
# ===========================================================================
# A transparent, theme-matched template so charts sit on the app background
# instead of appearing as white boxes.
PLOTLY_TEMPLATE = "nyc_planner"

_NYC_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, -apple-system, Segoe UI, Arial, sans-serif",
              "color": "#0f172a", "size": 12},
        title={"x": 0.01, "xanchor": "left", "font": {"size": 15}},
        colorway=["#4c78a8", "#f58518", "#72b7b2", "#e45756", "#54a24b", "#9d755d"],
        xaxis={"gridcolor": "rgba(15,23,42,0.08)", "linecolor": "rgba(15,23,42,0.15)",
               "zerolinecolor": "rgba(15,23,42,0.15)"},
        yaxis={"gridcolor": "rgba(15,23,42,0.08)", "linecolor": "rgba(15,23,42,0.15)",
               "zerolinecolor": "rgba(15,23,42,0.15)"},
        margin={"l": 55, "r": 20, "t": 48, "b": 45},
        legend={"bgcolor": "rgba(0,0,0,0)"},
    )
)
pio.templates[PLOTLY_TEMPLATE] = _NYC_TEMPLATE
pio.templates.default = PLOTLY_TEMPLATE


def _empty_figure(message: str) -> go.Figure:
    """
    Build a friendly placeholder so the UI never shows an empty box.

    Parameters
    ----------
    message:
        Text shown centred in the figure.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font={"size": 14, "color": "#64748b"})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=320, template=PLOTLY_TEMPLATE,
                      margin={"l": 20, "r": 20, "t": 30, "b": 20})
    return fig


def plot_price_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Histogram of nightly prices after outlier removal.

    Parameters
    ----------
    df:
        Cleaned listings with ``price``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df.empty:
        return _empty_figure("No listings to plot")

    fig = px.histogram(
        df, x="price", nbins=50, marginal="box",
        color_discrete_sequence=["#4c78a8"],
        title="Nightly price distribution (after IQR cleaning)",
        template=PLOTLY_TEMPLATE,
    )
    median = float(df["price"].median())
    fig.update_layout(xaxis_title="Price per night ($)",
                      yaxis_title="Listings", bargap=0.04, height=420)
    fig.add_vline(x=median, line_dash="dash", line_color="#e45756",
                  annotation_text=f"median ${median:.0f}")
    return fig


def plot_reviews_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Histogram of ``number_of_reviews``, showing how skewed review volume is.

    Parameters
    ----------
    df:
        Cleaned listings.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df.empty or "number_of_reviews" not in df.columns:
        return _empty_figure("No review data to plot")

    fig = px.histogram(
        df, x="number_of_reviews", nbins=50,
        color_discrete_sequence=["#f58518"],
        title="Distribution of review counts per listing",
        template=PLOTLY_TEMPLATE,
    )
    median = float(df["number_of_reviews"].median())
    fig.update_layout(xaxis_title="Number of reviews", yaxis_title="Listings",
                      bargap=0.04, height=420)
    fig.add_vline(x=median, line_dash="dash", line_color="#4c78a8",
                  annotation_text=f"median {median:.0f}")
    return fig



def plot_top_reviewed_by_borough(top_reviewed: pd.DataFrame) -> go.Figure:
    """
    Bar chart of top-reviewed listing volume per borough.

    This is the rubric's "volume of top-reviewed listings by region" chart.

    Parameters
    ----------
    top_reviewed:
        Output of ``select_top_reviewed``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if top_reviewed.empty:
        return _empty_figure("No top-reviewed listings to compare")

    counts = (
        top_reviewed["neighbourhood_group"].value_counts()
        .rename_axis("borough").reset_index(name="listings")
        .sort_values("listings")
    )
    fig = px.bar(
        counts, x="listings", y="borough", orientation="h",
        color="borough", color_discrete_map=cfg.REGION_COLORS, text="listings",
        title="Top-reviewed listings by borough", template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=390,
                      xaxis_title="Top-reviewed listings", yaxis_title="")
    return fig


def plot_price_by_borough(df: pd.DataFrame) -> go.Figure:
    """
    Box plot of price per borough.

    Parameters
    ----------
    df:
        Cleaned listings.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df.empty:
        return _empty_figure("No listings to plot")

    fig = px.box(
        df, x="neighbourhood_group", y="price",
        color="neighbourhood_group", color_discrete_map=cfg.REGION_COLORS,
        points=False, title="Nightly price by borough", template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(showlegend=False, height=400,
                      xaxis_title="Borough", yaxis_title="Price per night ($)")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Correlation matrix of price vs review-related variables.

    Parameters
    ----------
    df:
        Cleaned listings.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    matrix = get_correlation_matrix(df)
    if matrix.empty:
        return _empty_figure("Not enough numeric columns for a correlation matrix")

    fig = px.imshow(
        matrix, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Correlation matrix: reviews, price and availability",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=450, coloraxis_colorbar_title="r")
    return fig



def plot_price_vs_reviews(df: pd.DataFrame, max_points: int = 4000) -> go.Figure:
    """
    Scatter of nightly price against review count, coloured by borough.

    Plotly slows down with tens of thousands of markers, so the frame is sampled.

    Parameters
    ----------
    df:
        Cleaned listings.
    max_points:
        Maximum number of markers drawn.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df.empty:
        return _empty_figure("No listings to plot")

    sample = df.sample(min(len(df), max_points), random_state=42)
    try:
        import statsmodels.api  # noqa: F401  (trendline needs statsmodels)

        trendline: str | None = "ols" if len(sample) > 30 else None
    except Exception:
        trendline = None

    fig = px.scatter(
        sample, x="number_of_reviews", y="price",
        color="neighbourhood_group", color_discrete_map=cfg.REGION_COLORS,
        opacity=0.55, trendline=trendline,
        title="Price vs. number of reviews", template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=430, xaxis_title="Number of reviews",
                      yaxis_title="Price per night ($)")
    return fig


def plot_room_type_mix(df: pd.DataFrame) -> go.Figure:
    """
    Donut chart of the room-type mix.

    Parameters
    ----------
    df:
        Cleaned listings.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df.empty:
        return _empty_figure("No listings to plot")

    counts = (
        df["room_type"].value_counts()
        .rename_axis("room_type").reset_index(name="listings")
    )
    fig = px.pie(
        counts, names="room_type", values="listings", hole=0.45,
        title="Room-type mix", template=PLOTLY_TEMPLATE,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=380)
    return fig


def plot_sentiment_by_borough(df: pd.DataFrame) -> go.Figure:
    """
    Mean review sentiment per borough (requires an ``avg_sentiment`` column).

    Parameters
    ----------
    df:
        Listings already joined with sentiment scores.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if "avg_sentiment" not in df.columns or df["avg_sentiment"].dropna().empty:
        return _empty_figure(
            "Attach sentiment with nlp_sentiment.attach_sentiment() first"
        )

    grouped = (
        df.dropna(subset=["avg_sentiment"])
        .groupby("neighbourhood_group")["avg_sentiment"]
        .mean().round(3).reset_index().sort_values("avg_sentiment")
    )
    fig = px.bar(
        grouped, x="avg_sentiment", y="neighbourhood_group", orientation="h",
        color="neighbourhood_group", color_discrete_map=cfg.REGION_COLORS,
        text="avg_sentiment",
        title="Average review sentiment by borough (-1 negative, +1 positive)",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=380,
                      xaxis_title="Mean VADER compound score", yaxis_title="")
    fig.add_vline(x=0, line_dash="dot", line_color="grey")
    return fig


if __name__ == "__main__":
    # ``python eda.py`` runs the whole pipeline on demo data and prints results.
    import data_acquisition as da

    listings_df, _reviews_df, synthetic = da.ensure_data_available()
    print("synthetic:", synthetic, "| listings:", len(listings_df))

    clean_df, audit = clean_listings(listings_df)
    print("\n--- cleaning audit ---")
    for key, value in audit.items():
        print(f"{key}: {value}")

    top_df = select_top_reviewed(clean_df)
    print(f"\n--- top reviewed: {len(top_df):,} listings ---")
    print(top_df[["name", "neighbourhood_group", "price",
                  "number_of_reviews"]].head(8).to_string(index=False))

    print("\n--- summary stats ---")
    for key, value in get_summary_stats(clean_df).items():
        if not isinstance(value, pd.DataFrame):
            print(f"{key}: {value}")

    print("\n--- correlation with reviews ---")
    print(review_correlations(clean_df))

