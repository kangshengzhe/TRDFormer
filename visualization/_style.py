"""
Single source of truth for the manuscript's figure design language.

WHY THIS MODULE EXISTS
----------------------
Two problems motivated it.

1. **Figures were designed far too wide.** The target journal
   (elsarticle, ``[preprint,review,12pt]``) has ``\\textwidth = 390pt =
   5.42in``. Legacy scripts built canvases 10-13.5in wide, so
   ``\\includegraphics[width=0.95\\linewidth]`` scaled them by ~0.38 and a
   label authored at 9.5pt reached the page at **3.6pt** - unreadable, and
   inconsistent from figure to figure because each script picked its own
   canvas width. Figures here are authored at their *final printed size*
   (scale factor ~1.0), so the point sizes below are the point sizes the
   reviewer actually sees.

2. **Two competing visual languages.** Some scripts used SciencePlots,
   others a hand-rolled Times New Roman rcParams dict, with base font
   sizes of 7.5/8.5/9/9.5/11 across the set. One import now fixes all of
   it.

The palette is deliberately identical to the TikZ architecture figure
(``manuscript/figures/method/fig02_architecture_tikz.tex``) so that a
colour learned in the architecture diagram carries its meaning into every
data figure: gold = trend branch, blue = DWT, purple = endogenous
encoder, green = exogenous encoder, red = fusion / the proposed model.

DPI POLICY
----------
:func:`save_figure` writes the print asset at 640 dpi and, next to it, a
downscaled ``_preview/`` copy capped at 1500px on the long edge. Always
inspect the preview, never the asset: full-page art at 640 dpi is
~3840x4352px and blows past the size limit of some image consumers.

640 dpi is set by the EPSR artwork rules, not by taste. The guide requires
line/halftone combinations -- which every plot here is, being line art plus
filled colour and text -- to be at least 500 dpi, AND a full-page-width
image to be at least 3740 px wide. At the FULL_W = 6.0 in design width the
pixel rule binds: 6.0 x 500 = 3000 px would satisfy the dpi rule yet fail
the pixel rule, whereas 6.0 x 640 = 3840 px clears both. The earlier
400 dpi (2400 px) met neither.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Geometry.  All widths in inches, matching elsarticle [preprint,review,12pt].
# ---------------------------------------------------------------------------
TEXT_W = 5.42          # \textwidth = 390pt
TEXT_H = 7.62          # \textheight = 548.5pt

#: Single-column figure authored at \linewidth.
COL_W = TEXT_W

#: Full-page art. Rendered through
#: ``\makebox[\textwidth][c]{\includegraphics[width=1.10\textwidth]{...}}``
#: which yields 5.96in, so authoring at 6.0in keeps the scale at ~0.99.
#: Height leaves room for a multi-line caption on a ``[p]`` float page.
FULL_W = 6.00
FULL_H = 6.80

# ---------------------------------------------------------------------------
# Palette - mirrors \definecolor in fig02_architecture_tikz.tex
# ---------------------------------------------------------------------------
HERO = "#C41E3A"           # TRDFormer / the proposed model, everywhere
HERO_FILL = "#F2B8B5"

#: Innovation identity. Reused for ablation bars, group boxes, sub-band
#: strips - anything that "belongs to" one of the paper's contributions.
INNOV = {
    "A": "#B7950B",        # trend-residual decomposition (gold)
    "B": "#2166AC",        # partition-isolated DWT (blue)
    "C_endo": "#762A83",   # endogenous iTransformer branch (purple)
    "C_exo": "#1B7837",    # exogenous LSTM branch (green)
    "D": "#C41E3A",        # adaptive gated fusion + KAN head (red)
}
INNOV_FILL = {
    "A": "#F7E3A0",
    "B": "#BBD9EE",
    "C_endo": "#DCC9EA",
    "C_exo": "#C9E6D3",
    "D": "#F2B8B5",
}
INNOV_LABEL = {
    "A": "Trend-residual",
    "B": "DWT sub-bands",
    "C_endo": "Endogenous branch",
    "C_exo": "Exogenous branch",
    "D": "Gated fusion / head",
}

#: One colour per forecast horizon, used for row bands and horizon series.
HORIZON_COLOR = {
    1: "#2166AC",
    6: "#1B7837",
    12: "#C41E3A",
    24: "#6A3D9A",
}
HORIZON_LABEL = {
    1: "$h$=1  (10 min)",
    6: "$h$=6  (1 h)",
    12: "$h$=12  (2 h)",
    24: "$h$=24  (4 h)",
}

ACTUAL_COLOR = "#1A1A1A"   # ground truth is always near-black
GRID_COLOR = "#B0B0B0"
MUTED = "#6E6E6E"
NEUTRAL_BAR = "#9AA5B1"    # ablation variants with no innovation identity

# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------


def rc(base: float = 7.5) -> dict:
    """Return an rcParams dict for a figure authored at print size.

    Parameters
    ----------
    base:
        Body font size in points **as printed**. 7.5 suits dense full-page
        composites; 8.5 suits single-column figures with few panels.
        Elsevier asks for nothing below ~7pt, so do not go under 7.
    """
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman No9 L",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base + 0.5,
        "xtick.labelsize": base - 0.7,
        "ytick.labelsize": base - 0.7,
        "legend.fontsize": base - 0.7,
        "figure.titlesize": base + 2.0,

        "axes.linewidth": 0.6,
        "axes.edgecolor": "#333333",
        "axes.labelpad": 2.0,
        "axes.titlepad": 3.0,
        "axes.axisbelow": True,

        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.35,
        "grid.alpha": 0.35,

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "xtick.minor.width": 0.35,
        "ytick.minor.width": 0.35,
        "xtick.minor.size": 1.2,
        "ytick.minor.size": 1.2,
        "xtick.major.pad": 1.8,
        "ytick.major.pad": 1.8,

        "lines.linewidth": 0.9,
        "lines.markersize": 2.6,
        "lines.solid_capstyle": "round",

        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#CCCCCC",
        "legend.borderpad": 0.3,
        "legend.labelspacing": 0.25,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.45,
        "legend.columnspacing": 1.0,

        "savefig.dpi": 640,     # EPSR artwork floor; see DPI POLICY above
        "savefig.bbox": None,     # explicit layout; bbox_inches would
                                  # silently crop the figure-level boxes
        "figure.dpi": 120,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


# ---------------------------------------------------------------------------
# Panel-level helpers
# ---------------------------------------------------------------------------


def panel_tag(ax, letter: str, *, loc: str = "upper left",
              dx: float = 0.0, dy: float = 0.0, size: float | None = None,
              color: str = "black", weight: str = "bold",
              boxed: bool = True) -> None:
    """Stamp a consistent ``(a)``-style badge on an axes.

    Legacy scripts variously put the letter in the title, in an
    ``annotate`` call, or nowhere. A badge in axes coordinates keeps every
    panel label at the same visual weight and position regardless of how
    the axes is later resized.
    """
    pos = {
        "upper left": (0.020, 0.975, "left", "top"),
        "upper right": (0.980, 0.975, "right", "top"),
        "lower left": (0.020, 0.030, "left", "bottom"),
        "lower right": (0.980, 0.030, "right", "bottom"),
        "outside left": (-0.055, 1.045, "left", "top"),
    }[loc]
    x, y, ha, va = pos
    bbox = dict(boxstyle="round,pad=0.16", facecolor="white",
                edgecolor="#B8B8B8", linewidth=0.4, alpha=0.94) if boxed else None
    ax.text(x + dx, y + dy, f"({letter})", transform=ax.transAxes,
            ha=ha, va=va, fontsize=size or (mpl.rcParams["font.size"] + 0.5),
            fontweight=weight, color=color, bbox=bbox, zorder=2000)


def hero_frame(ax, color: str = HERO, *, lw: float = 1.3,
               tint: str | None = None, tint_alpha: float = 0.045) -> None:
    """Outline (and optionally tint) an axes that carries the proposed model.

    Gives the reader's eye a fixed anchor: in every figure, the red-framed
    panel is TRDFormer.
    """
    for side in ax.spines.values():
        side.set_color(color)
        side.set_linewidth(lw)
    if tint is not None:
        ax.set_facecolor(tint)
        ax.patch.set_alpha(tint_alpha)


def shade_error(ax, x, y_true, y_pred, *, color: str = HERO,
                alpha: float = 0.20, zorder: float = 1.0):
    """Fill the gap between truth and forecast.

    A shaded residual reads as "how wrong, and where" far faster than two
    overlaid lines, and it survives being printed small.
    """
    return ax.fill_between(x, y_true, y_pred, color=color, alpha=alpha,
                           linewidth=0, zorder=zorder)


def zoom_inset(ax, bounds: Sequence[float], xlim, ylim, *,
               edge: str = "#444444", lw: float = 0.6,
               connector_alpha: float = 0.55):
    """Create a magnified inset and mark the region it magnifies.

    ``bounds`` is ``[x0, y0, w, h]`` in axes coordinates. Returns the inset
    axes; the caller replots into it.
    """
    axins = ax.inset_axes(bounds)
    axins.set_xlim(*xlim)
    axins.set_ylim(*ylim)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.tick_params(length=1.0, width=0.3)
    for s in axins.spines.values():
        s.set_linewidth(lw)
        s.set_color(edge)
    ax.indicate_inset_zoom(axins, edgecolor=edge, linewidth=lw,
                           alpha=connector_alpha)
    return axins


def callout(ax, text: str, xy, xytext, *, color: str = "#333333",
            fill: str = "#FFFDF0", size: float | None = None,
            rad: float = -0.22, ha: str = "left", va: str = "center",
            lw: float = 0.6, arrow: bool = True, zorder: float = 1500):
    """Annotate the one thing in a panel the reader must not miss.

    This is the disciplined version of the "improvement" arrows seen in
    published composite figures: a short claim, tied by a thin curved
    leader to the evidence for it.
    """
    kw = dict(
        xy=xy, xytext=xytext, textcoords="axes fraction",
        ha=ha, va=va, zorder=zorder,
        fontsize=size or (mpl.rcParams["font.size"] - 0.8),
        color=color,
        bbox=dict(boxstyle="round,pad=0.28", facecolor=fill,
                  edgecolor=color, linewidth=lw, alpha=0.95),
    )
    if arrow:
        kw["arrowprops"] = dict(arrowstyle="-|>", color=color, linewidth=lw,
                                shrinkA=1.0, shrinkB=2.0,
                                connectionstyle=f"arc3,rad={rad}",
                                mutation_scale=7)
    return ax.annotate(text, **kw)


# ---------------------------------------------------------------------------
# Significance annotation
# ---------------------------------------------------------------------------


def sig_stars(p: float | None) -> str:
    """APA-style significance code. ``ns`` when a difference is not real."""
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def sig_bracket(ax, x1: float, x2: float, y: float, label: str, *,
                h: float = 0.02, color: str = "#333333",
                size: float | None = None, lw: float = 0.6,
                in_axes_frac: bool = True) -> None:
    """Draw a ``|__|`` bracket with a significance label above it.

    The paired t-tests behind the paper's tables were never surfaced in
    the figures; this puts them where the comparison is actually made.
    """
    trans = ax.get_xaxis_transform() if in_axes_frac else ax.transData
    dh = h
    ax.plot([x1, x1, x2, x2], [y, y + dh, y + dh, y],
            transform=trans, color=color, linewidth=lw,
            clip_on=False, zorder=1200)
    ax.text((x1 + x2) / 2, y + dh * 1.15, label, transform=trans,
            ha="center", va="bottom", color=color,
            fontsize=size or (mpl.rcParams["font.size"] - 1.0),
            clip_on=False, zorder=1200)


# ---------------------------------------------------------------------------
# Figure-level grouping devices
# ---------------------------------------------------------------------------


def group_box(fig, x0: float, y0: float, x1: float, y1: float, *,
              label: str = "", color: str = MUTED,
              fill: str | None = None, fill_alpha: float = 0.035,
              lw: float = 0.9, dashed: bool = True,
              label_side: str = "top left", pad: float = 0.008,
              size: float | None = None, zorder: float = 0.2):
    """Enclose a block of panels and name it.

    The single highest-impact device for making a multi-panel figure read
    as *one argument* rather than a grid of unrelated plots: the reader is
    told explicitly which panels form a stage of the story.

    Coordinates are figure fractions. Draw these *before* the axes are
    populated (low zorder) so they sit behind the data.
    """
    rect = FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad={pad},rounding_size=0.012",
        transform=fig.transFigure,
        facecolor=fill if fill else "none",
        alpha=1.0 if fill is None else fill_alpha,
        edgecolor=color, linewidth=lw,
        linestyle=(0, (4.5, 2.5)) if dashed else "-",
        zorder=zorder, clip_on=False,
    )
    fig.patches.append(rect)

    if label:
        fs = size or (mpl.rcParams["font.size"] - 0.3)
        if label_side.startswith("top"):
            ty = y1 + pad * 0.55
            va = "center"
        else:
            ty = y0 - pad * 0.55
            va = "center"
        tx = x0 + 0.018 if label_side.endswith("left") else x1 - 0.018
        ha = "left" if label_side.endswith("left") else "right"
        fig.text(tx, ty, f" {label} ", ha=ha, va=va, fontsize=fs,
                 fontweight="bold", color="white", zorder=zorder + 0.3,
                 bbox=dict(boxstyle="round,pad=0.30", facecolor=color,
                           edgecolor="none"))
    return rect


def band_label(fig, x: float, y: float, text: str, *, color: str = MUTED,
               rotation: float = 0.0, size: float | None = None,
               fill: str = "white", ha: str = "center", va: str = "center",
               weight: str = "bold", pad: float = 0.30, alpha: float = 1.0):
    """A row/column header chip in figure coordinates.

    Used for the horizon rows and model columns of the prediction matrix so
    that row/column identity is a first-class graphic element instead of
    being buried in 16 separate axes titles.
    """
    return fig.text(
        x, y, text, ha=ha, va=va, rotation=rotation,
        fontsize=size or (mpl.rcParams["font.size"] - 0.2),
        fontweight=weight, color=color, zorder=900,
        bbox=dict(boxstyle=f"round,pad={pad}", facecolor=fill,
                  edgecolor=color, linewidth=0.6, alpha=alpha),
    )


def flow_arrow(fig, p0, p1, *, label: str = "", color: str = MUTED,
               lw: float = 1.6, rad: float = 0.0, size: float | None = None,
               label_offset: float = 0.012, mutation: float = 11.0,
               zorder: float = 800, label_rotation: float | None = None):
    """A directed connector between two points in figure coordinates.

    Reserve an empty GridSpec row/column as a gutter and route the arrow
    through it. Earlier attempts drew arrows across populated axes and hit
    the tick labels, which is why they were removed; the fix is layout, not
    abandoning the device.
    """
    arr = FancyArrowPatch(
        p0, p1, transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=mutation,
        linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder, clip_on=False,
    )
    fig.patches.append(arr)
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        rot = label_rotation
        if rot is None:
            rot = 0.0 if abs(p1[0] - p0[0]) >= abs(p1[1] - p0[1]) else 90.0
        fig.text(mx, my + label_offset, label, ha="center", va="bottom",
                 rotation=rot, color=color, zorder=zorder + 1,
                 fontsize=size or (mpl.rcParams["font.size"] - 1.2),
                 fontweight="bold")
    return arr


def innovation_key(fig, x: float, y: float, keys: Iterable[str] = ("A", "B",
                                                                   "C_endo",
                                                                   "C_exo",
                                                                   "D"),
                   *, size: float | None = None, dx: float = 0.105,
                   swatch: float = 0.011):
    """Horizontal legend mapping the innovation colours to their meaning.

    Placed once per figure that uses the innovation palette so the reader
    only has to learn the code once.
    """
    fs = size or (mpl.rcParams["font.size"] - 1.2)
    for i, k in enumerate(keys):
        cx = x + i * dx
        fig.patches.append(FancyBboxPatch(
            (cx, y - swatch / 2), swatch, swatch,
            boxstyle="round,pad=0.001,rounding_size=0.002",
            transform=fig.transFigure, facecolor=INNOV_FILL[k],
            edgecolor=INNOV[k], linewidth=0.7, zorder=901, clip_on=False))
        fig.text(cx + swatch * 1.5, y, INNOV_LABEL[k], ha="left",
                 va="center", fontsize=fs, color="#333333", zorder=901)


# ---------------------------------------------------------------------------
# Axis cosmetics
# ---------------------------------------------------------------------------


def tidy(ax, *, grid: str = "y", minor: bool = True,
         spines: Sequence[str] = ("top", "right")) -> None:
    """Apply the house rules for grid and spines to one axes."""
    if grid in ("y", "both"):
        ax.grid(True, axis="y", zorder=0)
    if grid in ("x", "both"):
        ax.grid(True, axis="x", zorder=0)
    if grid == "none":
        ax.grid(False)
    for s in spines:
        ax.spines[s].set_visible(False)
    if minor:
        ax.minorticks_on()
        ax.tick_params(which="minor", top=False, right=False)


def kw_label(ax, which: str = "y", unit: str = "kW", text: str = "Power"):
    """Consistent ``Quantity (unit)`` axis labelling."""
    lab = f"{text} ({unit})"
    (ax.set_ylabel if which == "y" else ax.set_xlabel)(lab)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_figure(fig, out_path: str | Path, *, dpi: int = 640,
                preview_px: int = 1500, preview_dir: str = "_preview",
                also_pdf: bool = False) -> Path:
    """Write the print asset plus a size-capped preview.

    Returns the path of the print asset. The preview exists purely so the
    figure can be reviewed without handling a >2000px image.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    if also_pdf:
        fig.savefig(out_path.with_suffix(".pdf"))

    try:
        from PIL import Image

        prev_dir = out_path.parent / preview_dir
        prev_dir.mkdir(parents=True, exist_ok=True)
        prev = prev_dir / out_path.name
        with Image.open(out_path) as im:
            scale = min(1.0, preview_px / max(im.size))
            if scale < 1.0:
                new = (max(1, int(im.width * scale)),
                       max(1, int(im.height * scale)))
                im = im.convert("RGB").resize(new, Image.LANCZOS)
            else:
                im = im.convert("RGB")
            im.save(prev, format="PNG", optimize=True)
        logger.info("wrote %s (preview %s at %s)", out_path, prev, im.size)
    except Exception as exc:            # preview is a convenience, not a dep
        logger.warning("preview generation failed for %s: %s", out_path, exc)
    return out_path
