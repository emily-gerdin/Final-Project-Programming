# Project Report — Top Reviewed NYC Travel Planner

**Course:** Programming for Data Science — Final Year-End Project
**Author:** Emily Gerdin · **Date:** November 2026
**Repository:** *(GitHub URL)* · **App:** *(Streamlit Cloud URL, or "run locally")*

---

## 1. Objectives

Build a Streamlit app satisfying the four course requirements: acquire data from
an external API, perform EDA on it, integrate an LLM via API, and present it all
through an interactive interface.

The application — the **Top Reviewed NYC Travel Planner** — answers a practical
question: *which New York City Airbnbs are actually the best-reviewed, where are
they, and what do their reviews say?* It maps the most-reviewed listings across
the five boroughs, scores review text by sentiment, and lets an LLM write a
travel guide for whatever subset the user filters to.

The project covers **New York City, not New York State** — Inside Airbnb only
publishes NYC, Albany and Rochester for New York.

---

## 2. Methodology

Four phases, each in its own module so it can be tested independently.

**Phase 1 — Data acquisition** (`data_acquisition.py`). Two datasets are needed
because the listings file contains no review text.

| Dataset | Source | Size |
|---|---|---|
| `AB_NYC_2019.csv` | **Kaggle API** (`dgomonov/new-york-city-airbnb-open-data`, CC0) | 48,895 × 18 |
| Review comments | Inside Airbnb NYC archive | 200,000 |

Review text comes from the Inside Airbnb archive — the same upstream source the
Kaggle file derives from — so listing IDs join cleanly (92.6% of sampled reviews
matched). A second, keyless API, **OpenStreetMap Nominatim**, converts place
names into coordinates for the map's search box. Downloads are cached in `data/`.
If no credentials or network are available, a synthetic fallback generates
schema-identical data and the UI displays a warning banner, so fake data is never
presented as real.

**Phase 2 — Cleaning and EDA** (`eda.py`). Rows without coordinates are dropped
(the map cannot plot them) and missing `name`/`host_name` are filled. Price is
filtered with the IQR rule: Q1 = $69, Q3 = $175, IQR = $106, upper fence = **$334**.
This removes **2,983 listings (6.1%)**, including 11 priced at $0 and several
absurdities. Result: **45,899 clean rows**. The 500 listings with the highest
review counts (212–629, median 254) form the "top reviewed" subset used by the
map and the LLM.

**Phase 3 — Sentiment and LLM** (`nlp_sentiment.py`, `llm_agent.py`). Each review
is scored by **VADER** (a lexicon model tuned for short informal text) on a −1 to
+1 scale, then aggregated per listing. The AI tab serialises the filtered
listings into a compact context payload — medians, per-borough breakdowns, top
neighbourhoods, sentiment — and sends it to **Google Gemini** with a concierge
system prompt. Because the LLM receives aggregates rather than raw rows, it
cannot invent listings. With no key it falls back to a deterministic offline
generator from the same payload, so the tab never hard-fails.

**Phase 4 — Streamlit app** (`app.py`). Three tabs: an **Interactive Map**
(pydeck heatmap, clustering, price/review/borough/room filters, geocoded search),
**EDA Insights** (raw-vs-clean audit, outlier report, distributions, correlation
matrix), and the **AI Travel Guide** (LLM output plus provider, model, latency
and token metadata). Work is cached with `@st.cache_data`; every external call is
wrapped in `try`/`except` and reports through `st.error`/`st.warning`.

---

## 3. Results

| Metric | Raw | Clean |
|---|---|---|
| Rows | 48,895 | 45,899 |
| Median price | — | **$99 / night** |
| Median review count | — | 254 |
| Removed as price outliers | — | 2,983 (6.1%) |

**Top-reviewed listings by borough.** Brooklyn 211 (42.2%), Manhattan 187
(37.4%), Queens 89 (17.8%), Bronx 10 (2.0%), Staten Island 3 (0.6%). Brooklyn
and Manhattan hold **79.6%** of the most-reviewed listings — review turnover is
concentrated in those two boroughs.

**Room type.** Private rooms 274 (54.8%), entire homes 219 (43.8%), shared rooms
7 (1.4%). Notably, **private rooms outnumber entire homes in the top-reviewed
tier**, though entire homes dominate the full dataset — cheaper, longer-tenured
listings accumulate reviews faster.

**Sentiment.** Average compound ≈ **+0.8**, strongly positive across the corpus.
Because Airbnb reviews are overwhelmingly positive, sentiment ranks listings
*within* the good tier rather than separating good from bad.

**LLM.** The live Gemini guide generated in **3.3 s using 4,879 tokens**, citing
correct borough medians and neighbourhoods from the payload rather than generic
advice.

---

## 4. Screenshots

> Capture from the running app, save into `assets/`, and replace each line below
> with `![](assets/filename.png)`.

- **§4.1 Interactive Map** — heatmap, clustering, sidebar filters → `screenshot_map.png`
- **§4.2 EDA Insights** — raw-vs-clean audit, IQR report, distributions → `screenshot_eda.png`
- **§4.3 AI Travel Guide** — live guide with provider/model/latency → `screenshot_llm.png`
- **§4.4 Correlation matrix** — price, reviews, availability, sentiment → `screenshot_correlation.png`

---

## 5. Challenges and lessons learned

**The data didn't arrive in one piece.** The Kaggle listings file has no review
text, so sentiment — a core requirement — had nothing to score. Resolving it
meant locating the matching Inside Airbnb archive and verifying the join. An
early version joined only 14 of 800 reviews because a stale cache from a
different scrape year was in use; the lesson was to **validate joins explicitly**
rather than assume matching column names mean matching data.

**Silent failure is worse than loud failure.** Graceful degradation keeps the app
demoable without keys or network, but it hides errors: a 404 from the LLM falls
back to the offline generator and everything looks fine. Surfacing provider,
model and latency in the UI was how I made the live path observable.

**Model names expire.** Three Gemini models in a row were retired (404) or too
slow to answer within the request timeout. Setting the default to the `-latest`
alias in `config.py` — not in a local secrets file — means the project still
works when someone else clones it.

**Secrets hygiene from day one.** `.env`, `kaggle.json` and the data caches are
gitignored, and I confirmed by grepping the full commit history that no key was
ever staged. Committing ignore rules *before* the first `git add` is far easier
than purging a leaked key later.

---

## 6. Limitations and future work

- **Snapshot mismatch.** Listings are a 2019 Kaggle snapshot; reviews are a more
  recent scrape, so ~7% of sampled reviews have no matching listing and are
  discarded. A single common snapshot would recover them.
- **Weak sentiment signal.** Airbnb reviews are near-universally positive; a
  rating-prediction model would add more discrimination than lexicon scoring.
- **The LLM summarises, it doesn't query.** The concierge explains the current
  filter state but cannot run new aggregations. Tool-calling would make it
  genuinely conversational over the data.
