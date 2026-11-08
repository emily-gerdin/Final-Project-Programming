"""
nlp_sentiment.py
================

**Phase 3a - Sentiment Analysis (local NLP, no API key)**

What this module does
---------------------
1. Scores every review's free-text ``comments`` with **VADER** (default) or
   **TextBlob**, both of which run locally and cost nothing.
2. Aggregates the per-review scores **to the listing level**, which is what the
   rubric asks for, producing ``avg_sentiment`` per ``listing_id``.
3. Attaches that back onto the listings table and adds a short review snippet for
   the map tooltips.
4. Extracts the most frequent positive/negative keywords so the LLM and the
   written report have concrete vocabulary instead of invented detail.

Why VADER
---------
VADER is tuned for short social/free-form text such as reviews. It returns a
``compound`` score in ``[-1, +1]`` and needs no model download, so it installs in
seconds and runs on a CPU-only Streamlit Cloud instance. TextBlob is kept as a
switchable comparison (``polarity`` in ``[-1, +1]``, ``subjectivity`` in ``[0, 1]``).

Usage
-----
>>> import data_acquisition as da, nlp_sentiment as nlp
>>> listings, reviews, _ = da.ensure_data_available()
>>> scored = nlp.score_reviews(reviews, engine="vader")
>>> per_listing = nlp.aggregate_sentiment_by_listing(scored)
>>> listings = nlp.attach_sentiment(listings, per_listing)
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from typing import Literal

import pandas as pd

import config as cfg

# The two engines exposed in the sidebar.
Engine = Literal["vader", "textblob"]

# Words that appear in almost every review and carry no signal.
_STOPWORDS: frozenset[str] = frozenset(
    """
    the and was were for with this that from have has had not but you your they
    their them there here are our out get got very just also can will would
    about into over after before when what who how all any more most some such
    than then too only own same so now one two place stay stayed apartment room
    rooms nyc new york city great good nice
    """.split()
)


# ===========================================================================
# SECTION 1 - ENGINES
# ===========================================================================
def _vader_analyzer():
    """
    Return a cached VADER ``SentimentIntensityAnalyzer``.

    Constructing the analyzer parses ~7,500 lexicon entries, so it is built once
    per process and stored on the function object.

    Returns
    -------
    vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer | None
        The analyzer, or ``None`` when the package is missing.
    """
    if getattr(_vader_analyzer, "_instance", None) is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            _vader_analyzer._instance = SentimentIntensityAnalyzer()  # type: ignore[attr-defined]
        except Exception as exc:                      # pragma: no cover
            warnings.warn(f"VADER unavailable ({exc}); falling back to TextBlob.")
            _vader_analyzer._instance = None          # type: ignore[attr-defined]
    return _vader_analyzer._instance                  # type: ignore[attr-defined]


def score_text_vader(text: str) -> dict[str, float]:
    """
    Score one piece of text with VADER.

    Parameters
    ----------
    text:
        Free-form review text.

    Returns
    -------
    dict[str, float]
        ``neg``, ``neu``, ``pos`` (each 0-1) and ``compound`` (-1..+1).
    """
    analyzer = _vader_analyzer()
    if analyzer is None or not isinstance(text, str) or not text.strip():
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    scores = analyzer.polarity_scores(text)
    return {key: float(scores[key]) for key in ("neg", "neu", "pos", "compound")}


def score_text_textblob(text: str) -> dict[str, float]:
    """
    Score one piece of text with TextBlob.

    TextBlob's ``polarity`` maps onto the same ``compound`` key as VADER, so the
    rest of the pipeline is engine-agnostic.

    Parameters
    ----------
    text:
        Free-form review text.

    Returns
    -------
    dict[str, float]
        ``compound`` (-1..+1) plus ``subjectivity`` and the pos/neg/neu split.
    """
    try:
        from textblob import TextBlob

        polarity = float(TextBlob(text if isinstance(text, str) else "").sentiment.polarity)
        return {
            "compound": polarity,
            "subjectivity": abs(polarity),
            "pos": max(polarity, 0.0),
            "neg": max(-polarity, 0.0),
            "neu": 1.0 - abs(polarity),
        }
    except Exception as exc:                          # pragma: no cover
        warnings.warn(f"TextBlob unavailable ({exc}); using neutral scores.")
        return {"compound": 0.0, "subjectivity": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}


def sentiment_label(compound: float) -> str:
    """
    Convert a compound score into a readable band.

    Standard VADER cut-offs: ``>= 0.05`` positive, ``<= -0.05`` negative.

    Parameters
    ----------
    compound:
        Score in ``[-1, 1]`` (NaN allowed).

    Returns
    -------
    str
        ``"Positive"``, ``"Neutral"``, ``"Negative"`` or ``"Unknown"``.
    """
    if compound is None or pd.isna(compound):
        return "Unknown"
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


# ===========================================================================
# SECTION 2 - BATCH SCORING
# ===========================================================================
def score_reviews(
    reviews: pd.DataFrame,
    engine: Engine = "vader",
    sample_size: int | None = cfg.SENTIMENT_SAMPLE_SIZE,
    random_state: int = 42,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Score every review in the reviews table.

    Parameters
    ----------
    reviews:
        DataFrame with ``listing_id`` and ``comments``.
    engine:
        ``"vader"`` (default) or ``"textblob"``.
    sample_size:
        Cap on reviews scored. VADER handles thousands of reviews per second, so
        a sample this large is statistically indistinguishable from scoring
        everything, at a fraction of the runtime. ``None`` scores all rows.
    random_state:
        Seed for the sample, so results are reproducible.
    progress_callback:
        Optional ``callable(done, total)`` used by the app to drive a progress
        bar.

    Returns
    -------
    pandas.DataFrame
        One row per scored review: ``listing_id``, ``comments``, ``compound`` and
        ``sentiment_label``. Empty input returns an empty frame with those
        columns so callers never crash.
    """
    output_columns = ["listing_id", "comments", "compound", "sentiment_label"]
    if reviews is None or reviews.empty or "comments" not in reviews.columns:
        return pd.DataFrame(columns=output_columns)

    working = reviews.copy()
    working["comments"] = working["comments"].fillna("").astype(str)
    working = working.loc[working["comments"].str.len() > 0]

    if sample_size is not None and len(working) > sample_size:
        working = working.sample(sample_size, random_state=random_state)

    working = working.reset_index(drop=True)
    scorer = score_text_vader if engine == "vader" else score_text_textblob

    compounds: list[float] = []
    total = len(working)
    for position, text in enumerate(working["comments"].tolist(), start=1):
        compounds.append(scorer(text)["compound"])
        if progress_callback is not None and total and position % 2000 == 0:
            progress_callback(position, total)

    working["compound"] = compounds
    working["sentiment_label"] = working["compound"].apply(sentiment_label)

    keep = [c for c in ("listing_id", "comments", "date", "compound",
                        "sentiment_label") if c in working.columns]
    return working[keep]


def aggregate_sentiment_by_listing(scored_reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse per-review scores into one row per listing.

    Produces exactly the "average sentiment score per listing" the assignment
    asks for, plus context for the map tooltips.

    Parameters
    ----------
    scored_reviews:
        Output of ``score_reviews``.

    Returns
    -------
    pandas.DataFrame
        ``listing_id``, ``avg_sentiment``, ``n_reviews_scored``,
        ``pct_positive``, ``pct_negative``, ``sentiment_label``,
        ``best_review``, ``worst_review``.
    """
    columns = [
        "listing_id", "avg_sentiment", "n_reviews_scored", "pct_positive",
        "pct_negative", "sentiment_label", "best_review", "worst_review",
    ]
    if scored_reviews is None or scored_reviews.empty:
        return pd.DataFrame(columns=columns)

    grouped = scored_reviews.groupby("listing_id")
    aggregated = grouped.agg(
        avg_sentiment=("compound", "mean"),
        n_reviews_scored=("compound", "size"),
        pct_positive=("compound", lambda values: float((values >= 0.05).mean() * 100)),
        pct_negative=("compound", lambda values: float((values <= -0.05).mean() * 100)),
    ).round(3)

    # Best/worst snippet per listing (first occurrence wins on ties).
    best = scored_reviews.loc[grouped["compound"].idxmax()].set_index("listing_id")["comments"]
    worst = scored_reviews.loc[grouped["compound"].idxmin()].set_index("listing_id")["comments"]
    aggregated["best_review"] = best.reindex(aggregated.index)
    aggregated["worst_review"] = worst.reindex(aggregated.index)
    aggregated["sentiment_label"] = aggregated["avg_sentiment"].apply(sentiment_label)

    return aggregated.reset_index()[columns].astype(
        {"listing_id": "int64"}, errors="ignore"
    )


def attach_sentiment(listings: pd.DataFrame, per_listing: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join aggregated sentiment onto the listings table.

    Listings with no reviews receive ``NaN`` so the UI can show "no review data"
    instead of pretending they are neutral.

    Parameters
    ----------
    listings:
        Listings DataFrame containing ``id``.
    per_listing:
        Output of ``aggregate_sentiment_by_listing``.

    Returns
    -------
    pandas.DataFrame
        Listings plus ``avg_sentiment``, ``n_reviews_scored``, ``pct_positive``,
        ``sentiment_label``, ``best_review`` and ``review_snippet``.
    """
    if listings is None or listings.empty:
        return listings.copy() if listings is not None else pd.DataFrame()

    merged = listings.copy()
    if per_listing is None or per_listing.empty:
        for column in ("avg_sentiment", "n_reviews_scored", "pct_positive",
                       "sentiment_label", "best_review"):
            merged[column] = None
        merged["review_snippet"] = ""
        return merged

    merged = merged.merge(per_listing, left_on="id", right_on="listing_id", how="left")
    if "listing_id" in merged.columns:
        merged = merged.drop(columns=["listing_id"])

    # ~140-character snippet for map tooltips.
    merged["review_snippet"] = (
        merged["best_review"].fillna("").astype(str).str.slice(0, 140)
    )
    return merged



# ===========================================================================
# SECTION 3 - KEYWORDS AND SUMMARY (feeds the LLM and the report)
# ===========================================================================
def _tokenize(text: str) -> list[str]:
    """
    Lower-case a review and return its meaningful word tokens.

    Parameters
    ----------
    text:
        Raw review text.

    Returns
    -------
    list[str]
        Alphabetic tokens of 3+ characters that are not stop-words.
    """
    words = re.findall(r"[a-z']+", str(text).lower())
    return [w for w in words if len(w) >= 3 and w not in _STOPWORDS]


def extract_keywords(
    scored_reviews: pd.DataFrame,
    top_n: int = 15,
    sample_size: int = 20_000,
) -> dict[str, list[tuple[str, int]]]:
    """
    Find the words that appear most often in positive vs negative reviews.

    Gives the LLM (and the written report) concrete vocabulary instead of
    invented detail -- e.g. "guests repeatedly praise *subway, clean, host*".

    Parameters
    ----------
    scored_reviews:
        Output of ``score_reviews`` (needs ``comments`` and ``compound``).
    top_n:
        Number of words returned per band.
    sample_size:
        Cap on reviews scanned, keeping this under a second.

    Returns
    -------
    dict[str, list[tuple[str, int]]]
        ``{"positive": [(word, count), ...], "negative": [...]}``.
    """
    if scored_reviews is None or scored_reviews.empty:
        return {"positive": [], "negative": []}

    sample = scored_reviews
    if len(sample) > sample_size:
        sample = sample.sample(sample_size, random_state=42)

    positive_counter: Counter[str] = Counter()
    negative_counter: Counter[str] = Counter()
    for comments, compound in zip(sample["comments"], sample["compound"]):
        tokens = _tokenize(comments)
        if compound >= 0.05:
            positive_counter.update(tokens)
        elif compound <= -0.05:
            negative_counter.update(tokens)

    return {
        "positive": positive_counter.most_common(top_n),
        "negative": negative_counter.most_common(top_n),
    }


def sentiment_summary(listings_with_sentiment: pd.DataFrame) -> dict[str, object]:
    """
    Aggregate sentiment statistics for the dashboard KPI cards.

    Parameters
    ----------
    listings_with_sentiment:
        Listings that already carry ``avg_sentiment``.

    Returns
    -------
    dict[str, object]
        ``n_scored``, ``mean_sentiment``, ``pct_positive``, ``pct_negative`` and
        the best/worst borough by sentiment.
    """
    empty = {"n_scored": 0, "mean_sentiment": None, "pct_positive": None,
             "pct_negative": None, "best_borough": None, "worst_borough": None}
    if listings_with_sentiment is None or "avg_sentiment" not in listings_with_sentiment.columns:
        return empty

    scored = listings_with_sentiment.dropna(subset=["avg_sentiment"])
    if scored.empty:
        return empty

    by_borough = scored.groupby("neighbourhood_group")["avg_sentiment"].mean()
    return {
        "n_scored": int(len(scored)),
        "mean_sentiment": round(float(scored["avg_sentiment"].mean()), 3),
        "pct_positive": round(float((scored["avg_sentiment"] >= 0.05).mean() * 100), 1),
        "pct_negative": round(float((scored["avg_sentiment"] <= -0.05).mean() * 100), 1),
        "best_borough": str(by_borough.idxmax()) if not by_borough.empty else None,
        "worst_borough": str(by_borough.idxmin()) if not by_borough.empty else None,
    }



def emoji_for_sentiment(compound: float | None) -> str:
    """
    Map a compound score to a single emoji (used in tooltips and tables).

    Parameters
    ----------
    compound:
        Score in ``[-1, 1]``, or ``None``/NaN when a listing has no reviews.

    Returns
    -------
    str
        One of 😀 🙂 😐 🙁 😞 ❔
    """
    if compound is None or pd.isna(compound):
        return "❔"
    if compound >= 0.6:
        return "😀"
    if compound >= 0.05:
        return "🙂"
    if compound >= -0.05:
        return "😐"
    if compound >= -0.5:
        return "🙁"
    return "😞"


def happiest_listings(
    listings_with_sentiment: pd.DataFrame,
    n: int = 10,
    min_reviewed: int = 25,
) -> pd.DataFrame:
    """
    Rank listings by average sentiment among sufficiently reviewed stays.

    This answers the rubric's "find the happiest top-reviewed listings".

    Parameters
    ----------
    listings_with_sentiment:
        Listings carrying ``avg_sentiment`` and ``n_reviews_scored``.
    n:
        Number of listings to return.
    min_reviewed:
        Minimum reviews scored, so one glowing review cannot top the chart.

    Returns
    -------
    pandas.DataFrame
        Top ``n`` listings sorted by average sentiment (empty when unavailable).
    """
    if listings_with_sentiment is None or "avg_sentiment" not in listings_with_sentiment.columns:
        return pd.DataFrame()

    scored = listings_with_sentiment.dropna(subset=["avg_sentiment"])
    if "n_reviews_scored" in scored.columns:
        scored = scored[scored["n_reviews_scored"] >= min_reviewed]
    if scored.empty:
        return pd.DataFrame()

    columns = [c for c in ("name", "neighbourhood_group", "neighbourhood",
                           "price", "number_of_reviews", "avg_sentiment",
                           "n_reviews_scored", "sentiment_label")
               if c in scored.columns]
    return (
        scored.sort_values("avg_sentiment", ascending=False)
        .head(n)[columns]
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    # ``python nlp_sentiment.py`` -> score real/demo reviews and print a summary.
    import data_acquisition as da

    _, reviews_df, synthetic = da.ensure_data_available()
    print("synthetic:", synthetic, "| reviews:", len(reviews_df))

    scored_df = score_reviews(reviews_df, sample_size=25_000)
    if scored_df.empty:
        print("No reviews available to score.")
        raise SystemExit(0)

    print("\n--- sample scores ---")
    print(scored_df[["listing_id", "compound", "sentiment_label"]]
          .head(6).to_string(index=False))

    per_listing_df = aggregate_sentiment_by_listing(scored_df)
    print(f"\nper-listing rows  : {len(per_listing_df):,}")
    print(f"mean avg_sentiment: {per_listing_df['avg_sentiment'].mean():+.3f}")

    print("\n--- keywords ---")
    for band, words in extract_keywords(scored_df, top_n=8).items():
        print(f"{band}: {[w for w, _ in words]}")

