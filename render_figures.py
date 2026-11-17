"""
render_figures.py
=================

Renders the seven Plotly figures from ``eda.py`` to PNG, so the presentation
slides can show the same charts the Streamlit app draws instead of describing
them in words.

Why this exists: kaleido is not installed in this environment, so ``fig.write_image``
is unavailable. Instead each figure is written as a standalone HTML page, and
headless Chrome is then used to screenshot that page. The result is
pixel-identical to the browser rendering in the app.

Run it with::

    .venv/bin/python render_figures.py

It writes ``assets/slides_src/plot_*.png``, which are the images
``make_slides.py`` embeds. The intermediate HTML pages go to
``assets/slides_src/_html`` and can be deleted at any time.

Requirements: Chrome (or Chromium) and Pillow. If Chrome is not found the script
stops with a message naming the path it looked in, rather than leaving a
half-built deck behind.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

import config as cfg
import eda
import nlp_sentiment as nlp

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "assets" / "slides_src"      # the PNGs the deck uses
HTML_DIR = BASE_DIR / "assets" / "slides_src" / "_html"  # intermediate pages, gitignored
DARK, LIGHT = "#0A0A0A", "#F5F5F5"

# Each entry: output stem, the function that builds the figure, whether it wants
# the top-reviewed subset (only the borough volume chart does).
FIGURES: tuple[tuple[str, str], ...] = (
    ("plot_price_dist", "plot_price_distribution"),
    ("plot_review_dist", "plot_reviews_distribution"),
    ("plot_borough", "plot_top_reviewed_by_borough"),
    ("plot_price_scatter", "plot_price_vs_reviews"),
    ("plot_correlation", "plot_correlation_heatmap"),
    ("plot_room_mix", "plot_room_type_mix"),
    ("plot_sentiment", "plot_sentiment_by_borough"),
)


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return the cleaned listings and the top-reviewed subset the app charts.

    Sentiment is attached the same way the app does it, because two of the seven
    figures (room type, sentiment by borough) read the ``avg_sentiment`` column
    that the NLP step adds.
    """
    raw = pd.read_csv(cfg.LOCAL_LISTINGS_CSV, low_memory=False)
    clean, _audit = eda.clean_listings(raw)

    reviews = pd.read_csv(cfg.LOCAL_REVIEWS_CSV, low_memory=False)
    scored = nlp.score_reviews(reviews, engine="vader", sample_size=100_000)
    per_listing = nlp.aggregate_sentiment_by_listing(scored)
    clean = nlp.attach_sentiment(clean, per_listing)

    top = eda.select_top_reviewed(clean)
    return clean, top


def _find_chrome() -> str | None:
    """
    Locate a Chrome/Chromium binary for the screenshot step.

    Checks the usual macOS application path, then PATH, so the script works on a
    machine where Chrome is installed in the standard location or exposed as a
    command. Returns ``None`` when nothing is found; the caller reports it.
    """
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chrome"),
    ]
    return next((c for c in candidates if c and Path(c).exists()), None)


def _screenshot(chrome: str, page: Path, png: Path, height: int) -> bool:
    """
    Render one HTML chart page to a PNG and trim the surrounding page margin.

    ``--enable-unsafe-swiftshader`` is required because the price-scatter figure
    draws its trend line with WebGL, which is unavailable to headless Chrome
    otherwise; without it that chart renders as "WebGL is not supported".
    Returns ``True`` when a usable PNG was written.
    """
    from PIL import Image, ImageChops

    subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--disable-extensions",
         "--hide-scrollbars", "--enable-unsafe-swiftshader",
         f"--window-size=1240,{height}", "--virtual-time-budget=20000",
         f"--screenshot={png}", page.resolve().as_uri()],
        capture_output=True, check=False,
    )
    if not png.exists():
        return False
    # The page frame is pure black while the figure paper is #0A0A0A, so trimming
    # pure black removes the border without eating the chart.
    with Image.open(png) as shot:
        image = shot.convert("RGB")
    inbox = ImageChops.difference(image, Image.new("RGB", image.size, (0, 0, 0))).getbbox()
    if inbox:
        image.crop(inbox).save(png)
    return True


def main() -> None:
    """Build each figure, screenshot it, and leave the PNGs the deck embeds."""
    chrome = _find_chrome()
    if chrome is None:
        sys.exit("Chrome or Chromium not found; install it, or screenshot "
                 "assets/slides_src/_html/*.html by hand.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    clean, top = _load()
    print(f"cleaned rows: {len(clean):,} | top reviewed: {len(top):,}")

    for stem, func_name in FIGURES:
        func = getattr(eda, func_name)
        fig = func(top) if func_name == "plot_top_reviewed_by_borough" else func(clean)

        # Render on the app's own dark paper rather than Plotly's white default,
        # and give the category labels room: at the app's width they fit, but a
        # standalone page is narrower, so long borough names were being clipped.
        fig.update_layout(
            paper_bgcolor=DARK,
            plot_bgcolor=DARK,
            font={"color": LIGHT, "size": 13},
            margin={"l": 232, "r": 40, "t": 64, "b": 70},
        )
        fig.update_xaxes(gridcolor="#2A2A2A", zerolinecolor="#2A2A2A", linecolor="#3A3A3A")
        fig.update_yaxes(gridcolor="#2A2A2A", zerolinecolor="#2A2A2A", linecolor="#3A3A3A")

        html = fig.to_html(include_plotlyjs="cdn", full_html=True,
                           config={"staticPlot": True, "displayModeBar": False})
        page = HTML_DIR / f"{stem}.html"
        page.write_text(html, encoding="utf-8")

        png = OUT_DIR / f"{stem}.png"
        height = int(fig.layout.height or 400) + 13
        ok = _screenshot(chrome, page, png, height)
        status = "png" if ok else "PNG FAILED"
        print(f"{stem:22} {status}  ({fig.layout.title.text})")


if __name__ == "__main__":
    main()
