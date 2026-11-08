# 🗽 Top Reviewed NYC Travel Planner

An interactive Streamlit app that maps, scores and explains the **best-reviewed
Airbnb stays across the five boroughs of New York City**.

Capstone project — *Programming for Data Science*.

---

## What it does

| Tab | What you get |
|---|---|
| 🗺️ **Interactive Map** | pydeck map of 48k listings over the five boroughs with a heatmap layer, clustering, price/review sliders, borough and room-type filters, and geocoded area search |
| 📊 **EDA Insights** | Raw-vs-cleaned data audit, IQR outlier report, price and review distributions, correlation matrix, sentiment by borough |
| 🤖 **AI Travel Guide** | Sends the currently-filtered top-reviewed listings to an LLM (Gemini or OpenAI) which writes a natural-language travel guide. Falls back to a deterministic offline report when no key is configured |

---

## Quick start

```bash
# 1. Install dependencies (Python 3.10+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

The app opens at <http://localhost:8501>.

> **No credentials? No problem.** With no keys and no network the app runs on a
> generated dataset with the same schema. Every chart and filter still works and
> the UI shows a clear **SYNTHETIC DEMO DATA** banner so fake numbers are never
> mistaken for real ones.

---

## Step 3 — Credentials (optional but recommended)

The app is designed to degrade gracefully, so **this step is optional**. Add keys
to unlock real data and the live AI concierge.

### Kaggle API — real listings data

The primary listings source is the
[New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)
dataset (48,895 rows, CC0). Two ways to authenticate:

**Option A — credentials file (easiest)**

1. Sign in at <https://www.kaggle.com>
2. Click your avatar → **Settings** → **Create New API Token**
3. This downloads `kaggle.json` — move it to `~/.kaggle/kaggle.json`

```bash
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

**Option B — `.env` file**

```ini
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_32_char_api_key
```

### LLM key — live AI travel guide

Without a key the AI tab uses a built-in **offline guide generator**, which is
fully functional and produces a complete written guide from your data. To use a
real LLM, add **one** of these to `.env`:

```ini
# Google Gemini (free tier available at https://aistudio.google.com/apikey)
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIza...

# — or — OpenAI (requires funded billing at https://platform.openai.com/api-keys)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-...
```

Optionally override the model with `GEMINI_MODEL` / `LLM_MODEL`. The default is
`gemini-flash-lite-latest`.

> ⚠️ **Never commit `.env`.** It is gitignored, along with `kaggle.json` and
> `.streamlit/secrets.toml`. On Streamlit Cloud, paste the same names into the
> app's **Secrets** panel instead.

---

## Scope and data availability

**This project covers New York City, not New York State.** Inside Airbnb — the
upstream publisher of the data — only releases NYC, Albany and Rochester for New
York, so a statewide map is not reproducible from a single Kaggle download.

**Why reviews come from Inside Airbnb, not Kaggle.** The Kaggle listings file
contains no review text, so sentiment analysis would have nothing to score. The
app therefore fetches the matching Inside Airbnb NYC review archive, which is the
same upstream source the Kaggle file derives from, so listing ids join cleanly.
A dated snapshot is used rather than the `latest` alias, which currently returns
HTTP 403.

**Data is not committed.** `data/` and `processed/` are gitignored — the raw
files are ~74 MB. On a fresh clone they are re-downloaded on first run, so **the
first launch needs either Kaggle credentials or a network connection.** If both
are unavailable the synthetic fallback takes over.

---

## Project structure

```
config.py             Central settings: paths, dataset ids, NYC geography,
                      cleaning thresholds, secret resolution
data_acquisition.py   Phase 1 — Kaggle listings, Inside Airbnb reviews,
                      Nominatim geocoding, synthetic fallback
eda.py                Phase 2 — cleaning, IQR outlier removal,
                      top-reviewed selection, Plotly charts
nlp_sentiment.py      Phase 3 — VADER / TextBlob scoring, per-listing
                      aggregation, keyword extraction
llm_agent.py          Phase 3 — Gemini / OpenAI concierge, offline fallback
app.py                Phase 4 — Streamlit UI: map, EDA and AI tabs
requirements.txt      Pinned dependencies
```

Each module is independently runnable for debugging:

```bash
python data_acquisition.py --synthetic   # offline demo data
python data_acquisition.py               # real download
python eda.py                            # cleaning + charts on demo data
```

---

## Data sources

| Source | Used for | Licence |
|---|---|---|
| [Kaggle — NYC Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) | Listings, coordinates, price, review counts | CC0 |
| [Inside Airbnb](https://insideairbnb.com/get-the-data/) | Review text for sentiment | Public |
| [OpenStreetMap Nominatim](https://nominatim.org/) | Area search geocoding | ODbL |
| [CARTO basemaps](https://carto.com/basemaps/) | Map tiles (no token required) | Free tier |

---

## Troubleshooting

**App shows "SYNTHETIC DEMO DATA"** — no listings could be loaded. Add Kaggle
credentials (step 3) and tick **Force re-download** in the sidebar.

**AI tab returns the offline guide** — no LLM key configured, or the configured
model is unavailable. Check the `.env` variable names and confirm the model name
is one your account can access.

**`UnicodeDecodeError` reading a CSV** — the loader already retries with Latin-1
and a lenient parser; a persistent failure usually means a truncated download.
Delete the file in `data/` and re-run with **Force re-download**.
