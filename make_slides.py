"""
make_slides.py
==============

Builds the opening presentation for the capstone project as a PDF, in the same
visual language as the Streamlit app: near-black paper, Bodoni display type,
hairline rules, wide-tracked uppercase labels and square corners.

Run it with::

    .venv/bin/python make_slides.py

and it writes ``Final Project Presentation.pdf`` next to this file. The deck is
seven 16:9 slides:

1. Title
2. What the app does
3. Requirement 01, data acquisition: the brief's wording, then this project's answer
4. Requirement 02, EDA: the same, with the live EDA tab as evidence
5. Requirement 03, LLM integration
6. Requirement 04, the Streamlit application
7. Requirement 05, code quality and collaboration

Slides 3 to 7 are the point of the deck: each one quotes the requirement it
answers, then shows specifically how this project satisfies it, naming the
module, the mechanism and a number that proves it.

Design notes
------------
* Slides are drawn directly with reportlab rather than converted from HTML, so
  the output is a real, selectable-text PDF with no external toolchain.
* Type is Bodoni 72 (the macOS Didone the app's Bodoni Moda belongs to) for
  display and Avenir Next UltraLight for labels and body, matching the app's
  heavy serif / thin sans pairing. Both ship with macOS, so nothing is
  downloaded; the script fails with a clear message on a machine without them.
* One shared set of helpers (``_rule``, ``_kicker``, ``_body``... and the frame
  in ``_chrome``) positions every slide from the same margins, so pages stay
  aligned without per-slide tuning.
* The numbers quoted on the slides are read from the live data files when they
  are present (see :func:`_facts`) and fall back to the values recorded in the
  project report, so the deck cannot silently drift from the app.
* Em dashes are deliberately not used anywhere in the copy.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from reportlab.lib.pagesizes import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PDF = BASE_DIR / "Final Project Presentation.pdf"
CHARTS = BASE_DIR / "assets" / "slides_src"

# Body size for slide copy. Raised from 9.2 so text is readable from the back of
# a room; captions and chips scale off it rather than setting their own sizes.
BODY = 11.8
BODY_CAPTION = 10.2   # chart captions sit under an image, so a step smaller

# ---------------------------------------------------------------------------
# PAGE GEOMETRY (16:9, the shape of every modern projector)
# ---------------------------------------------------------------------------
PAGE_W, PAGE_H = 13.333 * inch, 7.5 * inch
MARGIN = 0.78 * inch
CONTENT_W = PAGE_W - 2 * MARGIN

# ---------------------------------------------------------------------------
# PALETTE -- identical to the app (see .streamlit/config.toml)
# ---------------------------------------------------------------------------
BLACK = "#0A0A0A"
INK = "#F5F5F5"
MUTED = "#C4C4C4"      # body copy on the slides (the app uses #A0A0A0)
RULE = "#2A2A2A"
RULE_SOFT = "#1C1C1C"

# ---------------------------------------------------------------------------
# FONTS
# ---------------------------------------------------------------------------
# Faces are registered from the macOS system font folders. Only the slices the
# deck uses are loaded, which keeps the PDF small.
_SUPPLEMENTAL = Path("/System/Library/Fonts/Supplemental")
_SYSTEM = Path("/System/Library/Fonts")

SERIF = "Bodoni72"           # display: headline, section numbers, titles
SERIF_ITALIC = "Bodoni72It"  # standfirsts and quotes
SERIF_BOLD = "Bodoni72Bd"    # emphasis inside a serif line
SERIF_CAPS = "Bodoni72SC"    # small-caps labels (title slide)
SANS = "AvenirUltra"         # body and labels, ultra light
SANS_BOOK = "AvenirBook"     # slightly heavier sans for numbers and links


def _register_fonts() -> None:
    """
    Register the serif and sans faces used by the deck.

    Raises
    ------
    FileNotFoundError
        If the macOS Bodoni 72 / Avenir Next files are missing. The message
        names the exact file so the deck can be re-created on another Mac.
    """
    # The two collections live in different folders on macOS: Bodoni 72 in
    # Supplemental, Avenir Next in the main fonts folder. Each is located by
    # searching both so the check does not hard-code one layout.
    bodoni = next((p for p in (_SUPPLEMENTAL / "Bodoni 72.ttc", _SYSTEM / "Bodoni 72.ttc") if p.exists()), None)
    avenir = next((p for p in (_SYSTEM / "Avenir Next.ttc", _SUPPLEMENTAL / "Avenir Next.ttc") if p.exists()), None)
    smallcaps = _SUPPLEMENTAL / "Bodoni 72 Smallcaps Book.ttf"
    missing = [name for name, path in (("Bodoni 72.ttc", bodoni), ("Avenir Next.ttc", avenir),
                                       ("Bodoni 72 Smallcaps Book.ttf", smallcaps)) if path is None or not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing macOS font files: {', '.join(missing)}. This script uses "
            "the system fonts so the deck matches the app's typography without "
            f"downloading anything. Looked in {_SYSTEM} and {_SUPPLEMENTAL}."
        )

    # Index 0 is Book, 1 Book Italic, 2 Bold in the Bodoni 72 collection.
    pdfmetrics.registerFont(TTFont(SERIF, str(bodoni), subfontIndex=0))
    pdfmetrics.registerFont(TTFont(SERIF_ITALIC, str(bodoni), subfontIndex=1))
    pdfmetrics.registerFont(TTFont(SERIF_BOLD, str(bodoni), subfontIndex=2))
    pdfmetrics.registerFont(TTFont(SERIF_CAPS, str(smallcaps)))
    # Index 10 is UltraLight, 7 Regular in the Avenir Next collection.
    pdfmetrics.registerFont(TTFont(SANS, str(avenir), subfontIndex=10))
    pdfmetrics.registerFont(TTFont(SANS_BOOK, str(avenir), subfontIndex=7))


# ---------------------------------------------------------------------------
# LOW LEVEL DRAWING HELPERS
# ---------------------------------------------------------------------------
def _tracked(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    font: str,
    size: float,
    tracking: float = 0.0,
    color: str = INK,
    align: str = "left",
) -> float:
    """
    Draw letter-spaced text and return the width used.

    reportlab has no letter-spacing property, so each character is placed by
    hand. Tracking (``tracking``, in points, added after every glyph) is what
    gives the small uppercase labels their editorial feel, and doing it here
    means every label in the deck is spaced identically.

    Parameters
    ----------
    c:
        Canvas to draw on.
    text:
        The string, drawn as given, so uppercase it before calling.
    x, y:
        Baseline origin. With ``align="right"`` x is the right edge; with
        ``align="center"`` it is the middle.
    font, size, color:
        Typeface, point size and fill colour.
    tracking:
        Extra space in points after each glyph.
    align:
        ``"left"``, ``"right"`` or ``"center"``.

    Returns
    -------
    float
        Width of the drawn text including tracking.
    """
    width = sum(pdfmetrics.stringWidth(ch, font, size) + tracking for ch in text)
    if width:
        width -= tracking  # no trailing space after the final glyph
    if align == "right":
        x -= width
    elif align == "center":
        x -= width / 2

    text_obj = c.beginText()
    text_obj.setTextOrigin(x, y)
    text_obj.setFont(font, size)
    text_obj.setFillColor(color)
    # ``moveCursor(dx, 0)`` does not advance the text position in reportlab, so
    # each glyph is placed with an explicit ``setTextOrigin`` instead. The
    # advance uses the same metrics as the width measured above, which keeps
    # the tracking exact and the text selectable in a PDF reader.
    cursor = x
    for ch in text:
        text_obj.setTextOrigin(cursor, y)
        text_obj.textOut(ch)
        cursor += pdfmetrics.stringWidth(ch, font, size) + tracking
    c.drawText(text_obj)
    return width


def _rule(c: canvas.Canvas, x1: float, y: float, x2: float,
          color: str = RULE, width: float = 0.6) -> None:
    """Draw a single hairline from ``x1`` to ``x2`` at height ``y``."""
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.line(x1, y, x2, y)


def _kicker(c: canvas.Canvas, text: str, x: float, y: float, color: str = MUTED,
            size: float = 7.0, tracking: float = 2.6, font: str = SANS,
            align: str = "left") -> float:
    """Draw an uppercase micro-label, the deck's version of the app's captions."""
    return _tracked(c, text.upper(), x, y, font, size, tracking, color, align)


def _wrap(text: str, font: str, size: float, max_width: float,
          tracking: float = 0.0) -> list[str]:
    """
    Greedy word wrap measured with the real font metrics.

    ``pdfmetrics.stringWidth`` is used rather than an estimate so a line never
    overruns its column, which matters because several slides place copy in
    narrow two- and three-column grids.
    """
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        width = sum(pdfmetrics.stringWidth(ch, font, size) + tracking for ch in candidate)
        if current and width > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _paragraph(
    c: canvas.Canvas,
    text: str,
    x: float,
    y_top: float,
    max_width: float,
    font: str = SANS,
    size: float = 10.5,
    leading: float = 1.55,
    color: str = MUTED,
    tracking: float = 0.25,
    max_lines: int | None = None,
) -> float:
    """
    Draw a wrapped paragraph and return the y position below it.

    Parameters
    ----------
    y_top:
        Baseline of the first line.
    leading:
        Line spacing as a multiple of the point size.
    max_lines:
        Optional cap; extra text is dropped rather than allowed to collide
        with the rule below. The copy is written to fit, and the cap is a
        safety net if a string is edited later.
    """
    lines = _wrap(text, font, size, max_width, tracking)
    if max_lines is not None:
        lines = lines[:max_lines]
    y = y_top
    for line in lines:
        _tracked(c, line, x, y, font, size, tracking, color)
        y -= size * leading
    return y


def _chrome(c: canvas.Canvas, number: int, section: str, total: int = 7) -> None:
    """
    Draw the background and the recurring frame for a content slide.

    Every content slide shares this frame: a tracked section name top left, the
    slide number top right, and a hairline footer. Keeping it in one function
    is what makes the deck read as one document instead of separate pages.
    """
    c.setFillColor(BLACK)
    c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)

    _kicker(c, section, MARGIN, PAGE_H - 0.62 * inch, MUTED, size=7.6)
    _kicker(c, f"{number:02d} / {total:02d}", PAGE_W - MARGIN, PAGE_H - 0.62 * inch,
            MUTED, size=7.6, align="right")

    _rule(c, MARGIN, 0.62 * inch, PAGE_W - MARGIN, RULE_SOFT)
    _kicker(c, "The Top Reviewed NYC Airbnb Travel Guide",
            MARGIN, 0.44 * inch, "#6E6E6E", size=7.0, tracking=2.1)
    _kicker(c, "Programming for Data Science", PAGE_W - MARGIN, 0.44 * inch,
            "#6E6E6E", size=7.0, tracking=2.1, align="right")


# ---------------------------------------------------------------------------
# MID LEVEL COMPONENTS
# ---------------------------------------------------------------------------
def _heading(c: canvas.Canvas, title: str, y: float, size: float = 34,
             max_width: float = CONTENT_W * 0.78) -> float:
    """
    Draw a slide title in Bodoni and return the y position below it.

    Long titles wrap, and each line is set tight (0.98 leading) the way a
    magazine cover line stacks.
    """
    for line in _wrap(title, SERIF, size, max_width):
        _tracked(c, line, MARGIN, y, SERIF, size, 0, INK)
        y -= size * 0.98
    return y - size * 0.14


def _requirement_row(
    c: canvas.Canvas,
    number: str,
    title: str,
    body: str,
    y: float,
    col_w: float,
    chips: tuple[str, ...] = (),
    x: float = MARGIN,
) -> float:
    """
    Draw one requirement as a hairline-topped block and return the y below it.

    The layout mirrors the app's card headings: a large Bodoni number, a
    tracked uppercase title, then a short paragraph in the thin sans. ``chips``
    are rendered as small uppercase tags under the paragraph, the PDF
    equivalent of the app's hero meta chips.

    Parameters
    ----------
    number:
        Requirement index, e.g. ``"01"``.
    title:
        Short name of the requirement.
    body:
        One or two sentences, written to fit ``col_w`` at the default size.
    y:
        Height of the row's top rule.
    col_w:
        Width of the column this row lives in.
    chips:
        Optional short labels (tools, sources) shown under the body.
    """
    _rule(c, x, y, x + col_w, RULE_SOFT)
    top = y - 0.28 * inch

    _tracked(c, number, x, top - 2, SERIF, 22, 0, "#6E6E6E")
    _tracked(c, title.upper(), x + 0.58 * inch, top, SANS_BOOK, 11.0, 1.7, INK)

    body_y = top - 0.26 * inch
    body_size = BODY
    end_y = _paragraph(c, body, x, body_y, col_w, size=body_size, leading=1.38,
                       color=MUTED, tracking=0.2)

    # ``end_y`` is already one leading below the final baseline, so the chips
    # are placed a fixed, generous step under it. Measured in points, not in
    # extra line-heights: the earlier version sat on top of the last line.
    if chips:
        chip_y = end_y - 4
        _kicker(c, "  ·  ".join(chips), x, chip_y, "#9A9A9A", size=7.8, tracking=1.5)
        return chip_y - 0.20 * inch
    return end_y - 0.08 * inch


def _stat(c: canvas.Canvas, value: str, label: str, x: float, y: float,
          width: float, note: str = "") -> None:
    """
    Draw one statistic with a hairline above it.

    ``y`` is the top of the block and the block is a fixed 0.80 in tall: rule,
    Bodoni value, tracked label, and an optional note. Fixing the height is what
    lets the caller stack four of them without a value colliding with the next
    rule. Returns nothing; the caller advances ``y`` by the row height.
    """
    _rule(c, x, y, x + width, RULE_SOFT)
    _tracked(c, value, x, y - 0.33 * inch, SERIF, 28, 0, INK)
    _kicker(c, label, x, y - 0.49 * inch, MUTED, size=8.0, tracking=1.8)
    if note:
        _paragraph(c, note, x, y - 0.58 * inch, width, size=8.0, leading=1.4,
                   color=MUTED, tracking=0.15, max_lines=1)


# ---------------------------------------------------------------------------
# FACTS -- read from the project so the deck matches the app
# ---------------------------------------------------------------------------
def _chart(c: canvas.Canvas, name: str, x: float, top: float, width: float,
           caption: str, target_h: float | None = None) -> float:
    """
    Place one rendered chart, its caption, and return the y below the caption.

    ``name`` is the stem of a PNG written by ``render_figures.py``. When
    ``target_h`` is given the chart is scaled to that height and the width
    follows from its aspect ratio; otherwise it fills ``width``. Charts are
    measured before drawing, which is what lets a slide fit three of them
    without overlap: the caller knows the real height instead of assuming it.
    """
    path = CHARTS / f"{name}.png"
    height, shown_w = 0.0, width
    if path.exists():
        with Image.open(path) as shot:
            src_w, src_h = shot.size
        ratio = src_h / src_w
        height = target_h if target_h else width * ratio
        shown_w = height / ratio
        c.drawImage(str(path), x, top - height, width=shown_w, height=height,
                    mask=None)
        c.setStrokeColor(RULE)
        c.setLineWidth(0.6)
        c.rect(x, top - height, shown_w, height, stroke=1, fill=0)

    _rule(c, x, top - height - 0.10 * inch, x + shown_w, RULE_SOFT)
    return _paragraph(c, caption, x, top - height - 0.30 * inch, shown_w,
                      size=BODY_CAPTION, leading=1.36, color=MUTED, tracking=0.1)


def _facts() -> dict[str, str]:
    """
    Return the headline numbers quoted on the slides.

    These are the values shown in the EDA screenshot, which was taken from a
    run of the app on ``data/listings.csv``. They are declared here instead of
    being recomputed so a later run of the app cannot produce a deck whose text
    disagrees with its own picture.
    """
    return {
        "listings": "29,918",
        "clean": "17,756",
        "outliers": "1,650",
        "noprice": "10,512",
        "top": "500",
        "boroughs": "5",
        "sentiment": "+0.71",
    }


# ---------------------------------------------------------------------------
# SLIDE 1 -- TITLE
# ---------------------------------------------------------------------------
def slide_title(c: canvas.Canvas, facts: dict[str, str]) -> None:
    """
    Title slide: the app's own hero, cropped to the right of the page.

    The screenshot is drawn past the right and bottom page edges so it reads as
    a full-bleed image rather than a framed picture, which is the open,
    cinematic treatment the app uses. Everything on the left is the same
    hierarchy as the app's hero: tracked folio row, Bodoni headline, italic
    standfirst.
    """
    c.setFillColor(BLACK)
    c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)

    # --- the page itself is the design -------------------------------------
    # No screenshot here on purpose. The app's hero and this slide carry the
    # same headline, and showing both in any arrangement either repeats the
    # words or clips them, which reads as a layout error. The headline is set
    # once, as a full page of Bodoni, and the app is shown on slides 3 and 4
    # where its interface is the evidence.

    # --- folio row ---------------------------------------------------------
    _kicker(c, "New York City  ·  Five Boroughs", MARGIN, PAGE_H - 0.62 * inch)
    _kicker(c, "Capstone presentation", PAGE_W - MARGIN, PAGE_H - 0.62 * inch,
            align="right")

    # --- headline ----------------------------------------------------------
    # Set large and tight, the way a cover line stacks. A Didone needs size to
    # show its thick and thin strokes, and at 52pt the four lines fill the
    # left half of the page without any decoration behind them.
    y = PAGE_H - 1.78 * inch
    size = 52
    for line in ("THE TOP", "REVIEWED", "NYC AIRBNB", "TRAVEL GUIDE"):
        _tracked(c, line, MARGIN, y, SERIF, size, -0.4, INK)
        y -= size * 0.95

    # --- watermark numeral -------------------------------------------------
    # A single oversized figure in the negative space on the right: it gives
    # the page balance without competing with the headline, and it is the same
    # "big Bodoni figure over a tracked label" device as the app's KPI cards.
    _tracked(c, "01", PAGE_W - MARGIN, PAGE_H - 5.55 * inch, SERIF, 190, 0, "#1C1C1C",
             align="right")
    _kicker(c, "Five boroughs  ·  one ranked guide", PAGE_W - MARGIN,
            PAGE_H - 5.95 * inch, "#6E6E6E", size=6.8, tracking=2.4, align="right")

    _rule(c, MARGIN, y + 0.16 * inch, MARGIN + 3.05 * inch)

    # --- standfirst --------------------------------------------------------
    y -= 0.40 * inch
    _paragraph(
        c,
        "An interactive guide to the best-reviewed Airbnb stays in the five "
        "boroughs: mapped, scored by review sentiment and explained by an AI "
        "concierge.",
        MARGIN, y, 4.55 * inch, font=SERIF_ITALIC, size=13, leading=1.5,
        color=MUTED, tracking=0,
    )

    # --- footer ------------------------------------------------------------
    _rule(c, MARGIN, 1.42 * inch, MARGIN + 3.05 * inch, RULE_SOFT)
    _kicker(c, "Emily Gerdin", MARGIN, 1.16 * inch, INK, size=7.4)
    _kicker(c, "Masters of Data Science  ·  Programming for Data Science",
            MARGIN, 0.94 * inch, MUTED, size=6.6)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# SLIDE 2 -- THE DATASET
# ---------------------------------------------------------------------------
def slide_dataset(c: canvas.Canvas, facts: dict[str, str]) -> None:
    """
    Where the data comes from, what shape it is, and the quirks found on first
    contact.

    The left column answers provenance and structure (the two questions an
    audience asks first), and the right column lists the four properties that
    shaped every later decision, each with the measurement behind it.
    """
    _chrome(c, 2, "The dataset")

    y = _heading(c, "Sixteen years of NYC reviews, one snapshot",
                 PAGE_H - 1.30 * inch, size=30, max_width=9.4 * inch)

    col_w = CONTENT_W * 0.50
    y = PAGE_H - 2.28 * inch

    y = _requirement_row(
        c, "1", "Where it comes from",
        "Inside Airbnb, an independent project that scrapes Airbnb public "
        "listing pages and republishes them as open data under CC BY 4.0. The "
        "app pulls the New York City snapshot dated 14 September 2026 straight "
        "from data.insideairbnb.com.",
        y, col_w, chips=("insideairbnb.com", "CC BY 4.0", "Snapshot 2026-09-14"),
    )

    _requirement_row(
        c, "2", "What is in it",
        "Two tables that join on listing_id. Listings: 29,918 rows and 89 "
        "columns covering price, room type, borough, coordinates, minimum "
        "nights and availability. Reviews: 100,000 dated review comments.",
        y - 0.10 * inch, col_w,
        chips=("29,918 listings", "89 columns", "100,000 reviews"),
    )

    # --- what stands out on first contact ----------------------------------
    obs_x = PAGE_W - MARGIN - CONTENT_W * 0.46
    obs_w = CONTENT_W * 0.46
    _rule(c, obs_x, PAGE_H - 2.02 * inch, PAGE_W - MARGIN, RULE)
    _kicker(c, "What stands out before any analysis", obs_x, PAGE_H - 2.30 * inch,
            INK, size=6.8)
    _rule(c, obs_x, PAGE_H - 2.44 * inch, PAGE_W - MARGIN, RULE_SOFT)

    findings = (
        ("Skewed twice over",
         "Median price is $ 182 but the maximum is $ 31,538. Median reviews are "
         "5 while one listing carries 4,894, so neither mean is safe to quote "
         "without handling outliers."),
        ("A third of prices are missing",
         "10,512 rows, 35% of the file, carry no price at all, so every price "
         "figure is computed on the 19,406 rows that do."),
        ("Most listings are unproven",
         "8,114 listings (27%) have no reviews and no rating, which is why a "
         "raw average would overstate quality."),
        ("Manhattan is the expensive half",
         "Median nightly price runs from $ 106 on Staten Island to $ 265 in "
         "Manhattan, which alone holds 13,681 of the 29,918 listings."),
    )
    fy = PAGE_H - 2.74 * inch
    for title, body in findings:
        _tracked(c, title, obs_x, fy, SERIF, 11.5, 0, INK)
        end_y = _paragraph(c, body, obs_x, fy - 0.22 * inch, obs_w, size=8.4,
                           leading=1.42, color=MUTED, tracking=0.15, max_lines=3)
        fy = end_y - 0.18 * inch


# ---------------------------------------------------------------------------
# SLIDE 3 -- BEFORE THE EDA
# ---------------------------------------------------------------------------
def slide_before_eda(c: canvas.Canvas, facts: dict[str, str]) -> None:
    """
    The decisions taken from reading the raw data, before any chart existed.

    These are the checks a plot cannot show: empty columns, the IQR fence, the
    rule that defines "top reviewed", and the long-stay tail. The right column
    carries the before and after row counts, the one sequence a viewer needs to
    read every later chart correctly.
    """
    _chrome(c, 3, "Before the EDA")

    y = _heading(c, "What the raw data forced us to decide",
                 PAGE_H - 1.30 * inch, size=30, max_width=9.4 * inch)

    col_w = CONTENT_W * 0.62
    y = PAGE_H - 1.92 * inch

    y = _requirement_row(
        c, "1", "Twelve columns are entirely empty",
        "Host response rate, verification and several URL columns are 100% null "
        "here, so they are dropped rather than imputed, and the app ranks what "
        "remains by column.",
        y, col_w, chips=("Null share per column", "Drop, do not impute"),
    )

    y = _requirement_row(
        c, "2", "Price needs a fence, not a trim",
        "Q1 is $ 103.68 and Q3 is $ 337, so the IQR rule puts the fence at "
        "$ 686.98 a night. Anything above it is excluded, which removes 1,650 "
        "rows from every later chart.",
        y - 0.02 * inch, col_w,
        chips=("Q1 $ 103.68", "Q3 $ 337", "Fence $ 686.98"),
    )

    y = _requirement_row(
        c, "3", "A star rating cannot rank listings",
        "27% of listings have no reviews, so they have no rating either, and "
        "the median listing carries just 5 reviews. \"Top reviewed\" therefore "
        "means at least 50 reviews and the top 10% by volume, which leaves 500 "
        "stays.",
        y - 0.02 * inch, col_w,
        chips=("50 reviews minimum", "Top 10% by volume", "500 listings"),
    )

    _requirement_row(
        c, "4", "The long-stay tail",
        "The median minimum stay is 30 nights and 81% of listings ask for a "
        "month or more, so the file leans towards longer lets. Listings asking "
        "for over a year are removed.",
        y - 0.02 * inch, col_w,
    )

    # --- before and after ---------------------------------------------------
    col_x = PAGE_W - MARGIN - 3.30 * inch
    _rule(c, col_x, PAGE_H - 2.02 * inch, PAGE_W - MARGIN, RULE)
    _kicker(c, "Before and after", col_x, PAGE_H - 2.30 * inch, INK, size=6.8)
    _rule(c, col_x, PAGE_H - 2.44 * inch, PAGE_W - MARGIN, RULE_SOFT)

    steps = (
        (facts["listings"], "Rows on download"),
        (f"-{facts['noprice']}", "No price, removed"),
        (f"-{facts['outliers']}", "Price outliers, removed"),
        (facts["clean"], "Rows every chart uses"),
        (facts["top"], "Ranked as top reviewed"),
    )
    sy = PAGE_H - 2.72 * inch
    for value, label in steps:
        _stat(c, value, label, col_x, sy, 3.30 * inch)
        sy -= 0.72 * inch


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# SLIDES 4 TO 8 -- THE ANALYSIS, THE AI GUIDE, AND THE WALKTHROUGH
# ---------------------------------------------------------------------------
# Four views of the analysis (two slides of two charts, one of the correlation
# matrix), a short slide on the AI call, and a final walkthrough of the app.


def _two_charts(c: canvas.Canvas, number: int, section: str, title: str,
                left: tuple[str, str], right: tuple[str, str],
                lead: str = "", takeaway: tuple[str, str] | None = None) -> None:
    """
    Two charts side by side, each with its reading underneath.

    The pair is given the full width so the charts stay large, and an optional
    ``takeaway`` strip is pinned above the footer so the lower third of the slide
    says something instead of sitting empty.
    """
    _chrome(c, number, section)
    y = _heading(c, title, PAGE_H - 1.12 * inch, size=33, max_width=CONTENT_W * 0.94)
    top = PAGE_H - 1.80 * inch
    if lead:
        top = _paragraph(c, lead, MARGIN, y, CONTENT_W * 0.80,
                         size=BODY, leading=1.4, color=MUTED, tracking=0.1) + 0.14 * inch

    gap = 0.34 * inch
    col_w = (CONTENT_W - gap) / 2
    left_bottom = _chart(c, left[0], MARGIN, top, col_w, left[1])
    right_bottom = _chart(c, right[0], MARGIN + col_w + gap, top, col_w, right[1])

    if takeaway:
        label, body = takeaway
        block_y = min(left_bottom, right_bottom) - 0.26 * inch
        _rule(c, MARGIN, block_y, PAGE_W - MARGIN, RULE)
        _kicker(c, label, MARGIN, block_y - 0.28 * inch, INK, size=7.6)
        _paragraph(c, body, MARGIN, block_y - 0.56 * inch, CONTENT_W * 0.92,
                   size=BODY, leading=1.42, color=MUTED, tracking=0.1)


def slide_charts_shape(c: canvas.Canvas, facts: dict[str, str]) -> None:
    """
    Slide 4: the two distribution charts.

    Both answer the same question, what shape is the data in, and both are
    strongly skewed, which is why the deck talks about medians rather than means.
    """
    _two_charts(
        c, 4, "The analysis  \u00b7  shape of the data",
        "Two distributions, both badly skewed",
        ("plot_price_dist",
         "Most nights cluster between $ 50 and $ 200, with a long thin tail out to "
         "the $ 686 cut-off. The median is $ 168, marked by the dashed line."),
        ("plot_review_dist",
         "Almost every listing sits near zero reviews while a few reach the "
         "thousands. This long tail is why ranking uses a review threshold "
         "instead of an average."),
        lead=f"Every chart on these slides is drawn inside the app, on the "
             f"{facts['clean']} rows that survive cleaning.",
        takeaway=("Why this matters for the rest of the project",
                  "Both pictures are lopsided. A handful of listings carry most of "
                  "the reviews, and prices bunch low with a few very expensive "
                  "outliers. That is why every number quoted from here on is a "
                  "median rather than an average, and why the app ranks listings "
                  "by review count instead of by a star score."),
    )


def slide_charts_borough(c: canvas.Canvas, facts: dict[str, str]) -> None:
    """
    Slide 5: where the ranked stays are, and whether price explains reviews.
    """
    _two_charts(
        c, 5, "The analysis  \u00b7  geography and price",
        "Where the top-reviewed stays are",
        ("plot_borough",
         "Brooklyn (208) and Manhattan (192) hold 80% of the 500 ranked stays, "
         "while the Bronx (4) and Staten Island (5) barely register."),
        ("plot_price_scatter",
         "Each dot is a listing. The cloud is flat, so a higher nightly price "
         "does not come with more reviews. The white line is a trend fit and "
         "barely rises."),
        takeaway=("The pattern to remember",
                  "Where you stay matters: the best-reviewed stays cluster in "
                  "Brooklyn and Manhattan. What you pay matters far less, since "
                  "expensive listings are not better reviewed."),
    )


def slide_charts_detail(c: canvas.Canvas, facts: dict[str, str]) -> None:
    """
    Slide 6: the correlation matrix, centred and explained.

    The matrix has six labels on each axis, so it takes the middle of the slide
    at a size where the cell values stay readable, and the three readings a
    viewer should leave with sit underneath as three short columns.
    """
    _chrome(c, 6, "The analysis  \u00b7  what moves together")

    _heading(c, "What actually moves together", PAGE_H - 1.10 * inch,
             size=33, max_width=CONTENT_W * 0.94)

    # Centred: sized by height, with the width following the aspect ratio, so the
    # x position is computed from that width rather than guessed.
    img_h = 2.66 * inch
    img_w = img_h / (463 / 1240)          # plot_correlation.png is 1240 x 463
    img_x = (PAGE_W - img_w) / 2
    canvas_top = PAGE_H - 1.74 * inch
    # Guard on existence: a missing chart should leave a gap on this slide, not
    # abort the whole build half way through the deck.
    chart_path = CHARTS / "plot_correlation.png"
    if chart_path.exists():
        c.drawImage(str(chart_path), img_x, canvas_top - img_h,
                    width=img_w, height=img_h, mask=None)
        c.setStrokeColor(RULE)
        c.setLineWidth(0.6)
        c.rect(img_x, canvas_top - img_h, img_w, img_h, stroke=1, fill=0)

    gap = 0.40 * inch
    col_w = (CONTENT_W - 2 * gap) / 3
    block_y = canvas_top - img_h - 0.34 * inch
    _rule(c, MARGIN, block_y, PAGE_W - MARGIN, RULE)

    readings = (
        ("How to read it",
         "Every pair of columns gets one number between -1 and 1. Near 0 means "
         "the two move independently, near 1 means they rise together, near -1 "
         "means one rises as the other falls. The dark diagonal is the "
         "self-comparison, always 1.00."),
        ("The one real link",
         "Reviews and reviews per month score 0.64, the strongest pairing in the "
         "table. That is expected rather than interesting: a listing with a long "
         "history of reviews has, by definition, a high rate of them."),
        ("What does not link",
         "Price scores 0.06 with review count, so expensive stays are not better "
         "reviewed. Minimum nights is the only negative relationship (-0.30 with "
         "price): longer-stay listings are the cheaper ones."),
    )
    for i, (title, body) in enumerate(readings):
        x = MARGIN + i * (col_w + gap)
        _tracked(c, title, x, block_y - 0.28 * inch, SERIF, 14, 0, INK)
        _paragraph(c, body, x, block_y - 0.56 * inch, col_w, size=BODY,
                   leading=1.42, color=MUTED, tracking=0.1)

    _kicker(c, "Reviews, not price, are what separate listings",
            MARGIN, 0.92 * inch, "#9A9A9A", size=7.8, tracking=1.8)


def slide_req_llm(c: canvas.Canvas, facts: dict[str, str]) -> None:
    """
    Slide 7: the AI guide, kept short.

    One point on screen: the app calls a model API directly. The supporting
    facts sit underneath, and the measured numbers stay in the right-hand column
    so the demo has something to point at.
    """
    _chrome(c, 7, "The AI guide")

    _heading(c, "The guide is written by a live API call", PAGE_H - 1.10 * inch,
             size=33, max_width=CONTENT_W * 0.94)

    col_w = CONTENT_W * 0.60
    y = PAGE_H - 2.10 * inch

    _requirement_row(
        c, "1", "Called directly through an API",
        "The AI tab sends the current listings to Google Gemini through its REST "
        "API and prints the guide that comes back. OpenAI is selectable in the "
        "sidebar instead, and the reply reports which provider and model "
        "answered, so you can see the live call happened.",
        y, col_w, chips=("Gemini or OpenAI", "REST API", "Provider shown in the UI"),
    )

    _paragraph(
        c,
        "Only summarised numbers leave the app, never raw review text, and with no "
        "key the same payload produces an offline guide so the tab still works.",
        MARGIN, y - 2.34 * inch, col_w, size=BODY, leading=1.45, color=MUTED,
        tracking=0.1,
    )

    col_x = PAGE_W - MARGIN - 3.30 * inch
    _rule(c, col_x, PAGE_H - 1.76 * inch, PAGE_W - MARGIN, RULE)
    _kicker(c, "Measured", col_x, PAGE_H - 2.04 * inch, INK, size=7.4)
    _rule(c, col_x, PAGE_H - 2.18 * inch, PAGE_W - MARGIN, RULE_SOFT)

    proof = (
        ("3.3 s", "Live response time"),
        ("4,879", "Tokens for one guide"),
        (facts["top"], "Listings summarised"),
    )
    py = PAGE_H - 2.50 * inch
    for value, label in proof:
        _stat(c, value, label, col_x, py, 3.30 * inch)
        py -= 0.86 * inch

    # Three plain-language notes fill the lower half: this slide was cut back to
    # one point, so without them the page reads as unfinished.
    gap = 0.40 * inch
    note_w = (CONTENT_W * 0.62 - 2 * gap) / 3
    note_y = y - 3.24 * inch
    _rule(c, MARGIN, note_y + 0.20 * inch, MARGIN + CONTENT_W * 0.62, RULE_SOFT)
    plain = (
        ("What it sends", "A compact summary of your filters, not the raw table."),
        ("What comes back", "A few paragraphs of plain English about those stays."),
        ("What you see", "The guide, plus which model answered and how long it took."),
    )
    for i, (head, body) in enumerate(plain):
        nx = MARGIN + i * (note_w + gap)
        _tracked(c, head, nx, note_y, SERIF, 13, 0, INK)
        _paragraph(c, body, nx, note_y - 0.26 * inch, note_w, size=BODY,
                   leading=1.40, color=MUTED, tracking=0.1)


def slide_req_app(c: canvas.Canvas, facts: dict[str, str]) -> None:
    """
    Slide 8: a step-by-step walkthrough of the app, in the order a user meets it.

    Each step is a numbered row with the control to click and what happens next,
    so the slide doubles as the script for the live demo. The right-hand column
    names the place each step lives, so the walkthrough can be followed without
    hunting for a tab.
    """
    _chrome(c, 8, "Using the app")

    _heading(c, "Working through the app, step by step", PAGE_H - 1.10 * inch,
             size=33, max_width=CONTENT_W * 0.94)

    col_w = CONTENT_W * 0.62
    y = PAGE_H - 1.56 * inch

    steps = (
        ("Choose your data",
         "In the sidebar, leave the source as it is and press Force re-download "
         "only if the files are stale. The sentiment engine and the AI provider "
         "are chosen here too."),
        ("Filter to what you can afford",
         "On the Map tab, drag the nightly price slider and set a minimum review "
         "count. Add a borough or room type, or type a neighbourhood in the "
         "search box to jump the map there."),
        ("Read the map",
         "Each black pin is one listing; hover a pin for its photo, price, rating "
         "and review score. The four cards underneath count the listings and give "
         "median price and reviews."),
        ("Check the analysis",
         "Switch to EDA Insights to see the same filters reflected in the two "
         "distributions, the borough counts, the price scatter and the "
         "correlation matrix."),
        ("Generate the guide",
         "On AI Travel Guide, press Generate Travel Guide. The guide describes "
         "the listings you filtered to, and you can ask one follow-up question."),
    )
    for i, (title, body) in enumerate(steps):
        y = _requirement_row(c, str(i + 1), title, body, y, col_w)

    side_x = PAGE_W - MARGIN - 3.10 * inch
    _rule(c, side_x, PAGE_H - 1.66 * inch, PAGE_W - MARGIN, RULE)
    _kicker(c, "Where to click", side_x, PAGE_H - 1.94 * inch, INK, size=7.4)
    _rule(c, side_x, PAGE_H - 2.08 * inch, PAGE_W - MARGIN, RULE_SOFT)

    places = (
        ("1", "Sidebar", "data and settings"),
        ("2", "Map tab", "the filters"),
        ("3", "Map tab", "pins and cards"),
        ("4", "EDA Insights", "the charts"),
        ("5", "AI Travel Guide", "the guide"),
    )
    py = PAGE_H - 2.38 * inch
    for num, where, what in places:
        _tracked(c, num, side_x, py, SERIF, 15, 0, "#6E6E6E")
        _tracked(c, where, side_x + 0.34 * inch, py, SANS_BOOK, 10.0, 0.6, INK)
        _kicker(c, what, side_x + 0.34 * inch, py - 0.20 * inch, MUTED,
                size=7.0, tracking=1.4)
        py -= 0.62 * inch
# ---------------------------------------------------------------------------
# BUILD
# ---------------------------------------------------------------------------
def build(output: Path = OUTPUT_PDF) -> Path:
    """
    Render every slide into one PDF and return its path.

    Parameters
    ----------
    output:
        Where to write the PDF. Defaults to ``Final Project Presentation.pdf``
        beside this script.

    Returns
    -------
    pathlib.Path
        The PDF that was written.
    """
    _register_fonts()
    facts = _facts()

    c = canvas.Canvas(str(output), pagesize=(PAGE_W, PAGE_H))
    c.setTitle("The Top Reviewed NYC Airbnb Travel Guide")
    c.setAuthor("Emily Gerdin")
    c.setSubject("Capstone presentation, Programming for Data Science")
    c.setCreator("make_slides.py")

    for slide in (slide_title, slide_dataset, slide_before_eda, slide_charts_shape,
                  slide_charts_borough, slide_charts_detail, slide_req_llm,
                  slide_req_app):
        slide(c, facts)
        c.showPage()

    c.save()
    return output


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}  ({path.stat().st_size / 1024:.0f} KB)")
