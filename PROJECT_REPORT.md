# Project Report — Top Reviewed NYC Travel Planner

**Course:** Programming for Data Science — Final Year-End Project
**Author:** Emily Gerdin
**Date:** November 2026
**Repository:** *(add your GitHub URL)*
**Live app:** *(add Streamlit Cloud URL, or state "run locally")*

---

## 1. Objectives

The goal of this project was to build a Streamlit application that satisfies the
four core requirements of the course: acquire real data from an external API,
perform exploratory data analysis on it, integrate a Large Language Model through
an API call, and present everything through an interactive web interface.

The application itself, the **Top Reviewed NYC Travel Planner**, answers a
practical question: *if you want to book an Airbnb in New York City, which stays
are actually the best-reviewed, where are they, and what do the reviews say about
them?* The app maps the most-reviewed listings across the five boroughs, scores
review text with sentiment analysis, and lets an LLM write a travel guide for
whatever subset the user has filtered to.

The project covers **New York City, not New York State**. Inside Airbnb only
publishes NYC, Albany and Rochester for New York, so a statewide map cannot be
reproduced from a single data download.

---

## 2. Methodology

The pipeline runs in four phases, each in its own module so it can be tested
independently.

### Phase 1 — Data acquisition (`data_acquisition.py`)

Two datasets are required, because the listings file contains no review text.

| Dataset | Source | Size | Used for |
|---|---|---|---|
| `AB_NYC_2019.csv` | **Kaggle API** — `dgomonov/new-york-city-airbnb-open-data` (CC0) | 48,895 rows × 18 cols | Location, price, room type, review counts |
| `reviews.csv` | Inside Airbnb NYC archive (dated snapshot) | 200,000 rows sampled | Free-text review comments for sentiment |

Listings are fetched with the **Kaggle API** via `kagglehub`, which authenticates
against the user's Kaggle credentials. Because the Kaggle listings file contains
no review text, review comments are pulled from the matching Inside Airbnb NYC
archive — the same upstream source the Kaggle file derives from — so listing IDs
join cleanly between the two (verified: 92.6% of sampled reviews matched a
listing).

A second, keyless API — **OpenStreetMap Nominatim** — powers the map's area
search box, converting free-text place names into coordinates.

Downloads are cached in `data/` so the app only hits the network once. If no
credentials and no network are available, a **synthetic fallback** generates a
schema-identical dataset and the UI displays a clear warning banner, so the app
is always demonstrable without ever presenting fake data as real.

### Phase 2 — Cleaning and EDA (`eda.py`)

Cleaning is deliberate and auditable rather than automatic:

| Step | Effect |
|---|---|
| Remove rows without coordinates | Required — the map cannot plot them |
| Fill missing `name` / `host_name` | `"Unnamed listing"` / `"Unknown host"` |
| Fill missing `reviews_per_month` | `0` (correct: never reviewed) |
| **IQR outlier filter on price** | Q1 = $69, Q3 = $175, IQR = $106, upper fence = **$334** |
| Minimum-nights sanity filter | Applied to implausible values |

The price filter removes **2,983 listings** above $334 (6.1% of the dataset),
including absurd "$10,000 per night" entries and 11 listings priced at $0. Final
clean dataset: **45,899 rows**.

A "top reviewed" subset is then selected — the 500 listings with the highest
lifetime review counts (range 212–629 reviews, median 254) — which is what the
map and the LLM both operate on.

### Phase 3 — Sentiment analysis and LLM (`nlp_sentiment.py`, `llm_agent.py`)

**Sentiment.** Each review comment is scored with **VADER**, a lexicon-based
model tuned for short informal text, producing a compound score from −1 to +1.
Scores are aggregated per listing into an average sentiment and a
positive/neutral/negative label, then joined back onto the listings table.
TextBlob is available as an alternative engine.

**LLM.** The AI Travel Guide tab serialises the currently-filtered listings into
a compact context payload — counts, medians, per-borough breakdowns, top
neighbourhoods, sentiment, and representative listing names — and sends it to
**Google Gemini** through its API with a system prompt instructing it to act as a
travel concierge. The returned Markdown is rendered directly in the app, with a
follow-up question box for conversational queries.

Crucially, the LLM gets **structured aggregates, not raw rows**: it cannot invent
a listing that isn't in the data, because the payload is a summary rather than a
dump. If no key is configured, or the API call fails, the module falls back to a
deterministic offline generator that produces the same report from the same
payload — so the tab never hard-fails.

### Phase 4 — Streamlit application (`app.py`)

Three tabs over a shared, cached data layer:

1. **Interactive Map** — pydeck map with a heatmap layer, marker clustering, and
   filters for price, review count, borough and room type, plus geocoded area
   search.
2. **EDA Insights** — raw-vs-cleaned audit, outlier report, price and review
   distributions, correlation matrix, sentiment by borough.
3. **AI Travel Guide** — the LLM report, provider/model/latency/token metadata,
   token usage, and a downloadable copy of the generated guide.

Expensive work is cached with `@st.cache_data` keyed on its inputs, so filter
changes are instant. Every external call sits inside `try`/`except` and reports
through `st.error`, `st.warning` or `st.info` rather than crashing.

---

## 3. Results

### Descriptive statistics

| Metric | Raw listings | Clean listings |
|---|---|---|
| Rows | 48,895 | 45,899 |
| Median price | — | **$99.00 / night** |
| Median review count | — | 254 |
| Rows removed as price outliers | — | 2,983 (6.1%) |

### Top-reviewed listings by borough

| Borough | Listings | Share |
|---|---|---|
| Brooklyn | 211 | 42.2% |
| Manhattan | 187 | 37.4% |
| Queens | 89 | 17.8% |
| Bronx | 10 | 2.0% |
| Staten Island | 3 | 0.6% |

Brooklyn and Manhattan together account for **79.6%** of the most-reviewed
listings — supply and turnover are concentrated in those two boroughs, while the
Bronx and Staten Island barely register even at 500-listing depth.

### Room type mix (top-reviewed subset)

| Room type | Listings | Share |
|---|---|---|
| Private room | 274 | 54.8% |
| Entire home/apt | 219 | 43.8% |
| Shared room | 7 | 1.4% |

A finding worth noting: in the *top-reviewed* tier, **private rooms outnumber
entire homes**, even though entire homes dominate the full dataset. This suggests
longer-tenured budget listings accumulate reviews faster than premium ones.

### Sentiment

Across the scored reviews, average sentiment is strongly positive (~+0.8
compound), consistent with Airbnb's review culture. Because nearly all reviews
are positive, sentiment is more useful for **ranking within** good listings than
for separating good from bad — an honest limitation of the metric.

### LLM output

The live LLM guide generated in **3.3 seconds using 4,879 tokens**, producing
correctly-cited statistics from the payload (borough medians, top
neighbourhoods, sentiment) rather than generic travel advice.

---

## 4. Screenshots

> **Capture these from the running app at http://localhost:8501 and insert below.**
> Save images into `assets/` and reference them, e.g. `![](assets/map_tab.png)`.

### 4.1 Interactive Map tab

*Insert screenshot: `assets/screenshot_map.png`*

Shows the heatmap over the five boroughs, the clustering layer, and the sidebar
filters (price range, minimum reviews, borough, room type).

### 4.2 EDA Insights tab

*Insert screenshot: `assets/screenshot_eda.png`*

Shows the raw-vs-cleaned audit, the IQR outlier report ($69–$175 quartiles,
$334 fence, 2,983 rows removed), and the price/review distributions.

### 4.3 AI Travel Guide tab

*Insert screenshot: `assets/screenshot_llm.png`*

Shows a live Gemini-generated guide with the provider, model name, latency and
token count displayed beneath it.

### 4.4 Correlation matrix

*Insert screenshot: `assets/screenshot_correlation.png`*

Shows relationships between price, review counts, availability and sentiment.

---

## 5. Challenges and lessons learned

**The data didn't come in one piece.** The single biggest obstacle was that the
Kaggle listings dataset has no review text, so sentiment analysis — a core
requirement — had nothing to score. Resolving it meant finding the matching
Inside Airbnb archive and verifying that its listing IDs actually joined to the
Kaggle file. An early version silently joined only 14 of 800 reviews because a
stale cached listings file from a different scrape year was being used; tracking
that down taught me to **validate joins explicitly** rather than trusting that
matching column names imply matching data.

**Silent failure is worse than loud failure.** This project leans heavily on
graceful degradation — no Kaggle key, no network, no LLM key all still produce a
working app. That's good for demoing, but it means failures can be invisible: an
LLM call that 404s falls back to the offline generator and the app looks fine
while quietly not using the LLM at all. I ended up surfacing provider, model and
latency in the UI precisely so I could tell at a glance whether the live path ran.

**Model names expire.** Three consecutive Gemini models I tried were either
retired for new accounts (404) or too slow to respond inside the request timeout.
The fix was to use the `-latest` alias and set it as the default in config rather
than in a local secrets file, so the project still works when someone else clones
it without my `.env`.

**Secrets hygiene from day one.** `.env`, `kaggle.json` and the data caches are
gitignored, and I verified by grepping the full commit history that no key
material was ever staged. Getting that ordering right — ignore rules committed
*before* the first `git add` — is much easier than trying to purge a leaked key
afterwards.

---

## 6. Limitations and future work

- **Listing IDs are from a 2019 Kaggle snapshot** while reviews are from a more
  recent Inside Airbnb scrape. Roughly 7% of sampled reviews have no matching
  listing and are discarded. Pulling both from the same snapshot would recover
  them.
- **Sentiment is weakly discriminating** on Airbnb data because reviews are
  overwhelmingly positive. A rating-prediction model would add more signal than
  lexicon sentiment alone.
- **The LLM summarises; it does not query.** The concierge explains the current
  filter state but cannot run new aggregations. Tool-calling would make it
  genuinely conversational over the data.

