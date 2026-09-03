"""Plots for anagram eval results (matplotlib).

    plot_css(summaries)              sorted CSS bars per model, CI whiskers, human + chance lines
    plot_acc_vs_css(summaries)       single-image accuracy vs CSS, with the css = acc**2 reference
    plot_confusion(results)          9x9 true x predicted
    plot_margins(results)            decision-margin and foil-margin distributions
    show_pairs(results, ...)         image grid of anagram pairs with predictions

`summaries` is a DataFrame with one row per model: pd.DataFrame([r.summary for r in ...]) plus a
`model_name` column, or one of the reference csvs in reference/.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import DEFAULT_CONFIG, load_anagrams
from .mapping import CLASSES
from .scoring import CHANCE_ACC, CHANCE_CSS, AnagramResults

# Human baseline from Doshi et al. (n = 4, pairs-72, 750 ms + mask)
HUMAN_CSS = 0.896
HUMAN_ACC = 0.948

COLORS = {
    "bar": "#c3c2b7",        # neutral, for context models
    "highlight": "#2a78d6",  # the model(s) you care about
    "series2": "#eb6834",
    "correct": "#008300",
    "wrong": "#e34948",
    "text": "#52514e",
    "grid": "#e6e5e1",
}


def _tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def _reference_lines(ax, human, chance, x_text):
    if human is not None:
        ax.axhline(human, color=COLORS["text"], linestyle="--", linewidth=1)
        ax.text(x_text, human + 0.01, f"human {human:.2f}", color=COLORS["text"], fontsize=8, ha="right", va="bottom")
    if chance is not None:
        ax.axhline(chance, color=COLORS["text"], linestyle=":", linewidth=1)
        ax.text(x_text, chance + 0.01, f"chance {chance:.3f}", color=COLORS["text"], fontsize=8, ha="right",
                va="bottom")


def plot_css(
    summaries: pd.DataFrame,
    label: str = "model_name",
    value: str = "css",
    highlight: Iterable[str] | None = None,
    human: float | None = HUMAN_CSS,
    chance: float | None = CHANCE_CSS,
    sort: bool = True,
    ax=None,
    title: str | None = None,
):
    """Bar per model, sorted by CSS. Whiskers from css_ci_low/high (bounds) or css_err_low/high (lengths)
    when present. Models in `highlight` are colored and value-labeled; the rest are neutral."""
    df = summaries.copy()
    if sort:
        df = df.sort_values(value).reset_index(drop=True)
    highlight = set(highlight or [])
    x = np.arange(len(df))
    colors = [COLORS["highlight"] if m in highlight else COLORS["bar"] for m in df[label]]

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 0.28 * len(df) + 2), 4))
    ax.bar(x, df[value], color=colors, width=0.72)

    if {"css_ci_low", "css_ci_high"} <= set(df.columns):
        yerr = np.vstack([df[value] - df["css_ci_low"], df["css_ci_high"] - df[value]])
    elif {"css_err_low", "css_err_high"} <= set(df.columns):
        yerr = np.vstack([df["css_err_low"], df["css_err_high"]])
    else:
        yerr = None
    if yerr is not None:
        ax.errorbar(x, df[value], yerr=np.clip(yerr, 0, None), fmt="none", ecolor=COLORS["text"], elinewidth=1)

    tops = df[value].to_numpy() + (yerr[1] if yerr is not None else 0)  # label above the whisker
    for xi, (m, v, top) in enumerate(zip(df[label], df[value], tops)):
        if m in highlight:
            ax.text(xi, top + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=COLORS["text"])

    _reference_lines(ax, human, chance, x_text=len(df) - 0.5)
    ax.set_xticks(x)
    many = len(df) > 12
    ax.set_xticklabels(df[label], rotation=90 if many else 45, ha="center" if many else "right", fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Configural Shape Score")
    if title:
        ax.set_title(title)
    _tidy(ax)
    return ax


def plot_acc_vs_css(
    summaries: pd.DataFrame,
    label: str = "model_name",
    highlight: Iterable[str] | None = None,
    human: tuple[float, float] | None = (HUMAN_ACC, HUMAN_CSS),
    ax=None,
):
    """Single-image accuracy (x) vs CSS (y). Dashed curve: css = acc**2, what you'd get if the two
    images of a pair were independent draws at the model's accuracy. Points above it recognize pairs
    *more* consistently than their accuracy predicts."""
    highlight = set(highlight or [])
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    colors = [COLORS["highlight"] if m in highlight else COLORS["bar"] for m in summaries[label]]
    ax.scatter(summaries["acc"], summaries["css"], s=28, c=colors, edgecolors="white", linewidths=0.5, zorder=3)
    for m, a, c in zip(summaries[label], summaries["acc"], summaries["css"]):
        if m in highlight:
            ax.annotate(m, (a, c), xytext=(4, 4), textcoords="offset points", fontsize=8, color=COLORS["text"])
    grid = np.linspace(0, 1, 100)
    ax.plot(grid, grid**2, linestyle="--", color=COLORS["text"], linewidth=1, label="css = acc²")
    if human is not None:
        ax.scatter([human[0]], [human[1]], marker="*", s=120, c=COLORS["text"], zorder=4, label="human")
    ax.axvline(CHANCE_ACC, color=COLORS["text"], linestyle=":", linewidth=0.8)
    ax.axhline(CHANCE_CSS, color=COLORS["text"], linestyle=":", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("single-image accuracy (9-way)")
    ax.set_ylabel("Configural Shape Score")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _tidy(ax)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    return ax


def plot_confusion(results: AnagramResults, normalize: bool = True, ax=None):
    """9x9 confusion matrix (rows = true). normalize=True shows row proportions."""
    cm = results.confusion.to_numpy().astype(float)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            v = cm[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}", ha="center", va="center", fontsize=7,
                        color="white" if v > (0.5 if normalize else cm.max() / 2) else COLORS["text"])
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    name = results.summary.get("model_name")
    ax.set_title(f"{name}  acc {results.summary['acc']:.2f}  css {results.summary['css']:.2f}" if name else "")
    return ax


def plot_margins(results: AnagramResults, bins: int = 30, ax=None):
    """Distributions of decision margin (label vs best other) and foil margin (label vs the pair's
    other animal). Mass left of zero = errors (decision margin) / confusions with the foil."""
    p = results.predictions
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 3.5))
    both = pd.concat([p["decision_margin"], p["foil_margin"]])
    lo, hi = both.min(), both.max()
    edges = np.linspace(lo, hi, bins + 1)
    ax.hist(p["decision_margin"], bins=edges, color=COLORS["highlight"], alpha=0.75, label="decision margin")
    ax.hist(p["foil_margin"], bins=edges, color=COLORS["series2"], alpha=0.6, label="foil margin")
    ax.axvline(0, color=COLORS["text"], linewidth=1)
    ax.set_xlabel("margin  (S_label − S_other) / √2")
    ax.set_ylabel("images")
    ax.legend(frameon=False, fontsize=8)
    _tidy(ax)
    return ax


def show_pairs(
    results: AnagramResults,
    ds=None,
    select: str = "all",
    n: int = 8,
    anagram_ids: Iterable[str] | None = None,
    ncols: int = 4,
    seed: int = 0,
):
    """Grid of anagram pairs with the model's predictions.

    select: "all" | "both_correct" | "one_wrong" | "both_wrong" (ignored if anagram_ids given).
    ds: the HF dataset for results' config (loaded if None).
    """
    pairs = results.pairs
    if anagram_ids is None:
        mask = {
            "all": np.ones(len(pairs), bool),
            "both_correct": pairs["both_correct"].to_numpy(),
            "one_wrong": (pairs["correct0"] ^ pairs["correct1"]).to_numpy(),
            "both_wrong": (~pairs["correct0"] & ~pairs["correct1"]).to_numpy(),
        }[select]
        candidates = pairs.loc[mask, "anagram_id"].tolist()
        rng = np.random.default_rng(seed)
        anagram_ids = sorted(rng.choice(candidates, size=min(n, len(candidates)), replace=False)) if candidates else []
    anagram_ids = list(anagram_ids)
    if not anagram_ids:
        raise ValueError(f"no pairs match select={select!r}")

    if ds is None:
        ds = load_anagrams(results.summary.get("config", DEFAULT_CONFIG))
    index_of = {f: i for i, f in enumerate(ds["filename"])}
    preds = results.predictions.set_index("filename")

    nrows = -(-len(anagram_ids) // ncols)
    fig, axes = plt.subplots(nrows, 2 * ncols, figsize=(2.1 * 2 * ncols, 2.5 * nrows), squeeze=False)
    for k, aid in enumerate(anagram_ids):
        r, c = divmod(k, ncols)
        for pos in (0, 1):
            row = preds[(preds["anagram_id"] == aid) & (preds["position"] == pos)].iloc[0]
            ax = axes[r, 2 * c + pos]
            ax.imshow(ds[index_of[row.name]]["image"])
            ok = bool(row["correct"])
            ax.set_title(f"{row['label']} → {row['pred']}  (dm {row['decision_margin']:+.2f})", fontsize=8,
                         color=COLORS["correct"] if ok else COLORS["wrong"])
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"{results.summary.get('model_name', '')}  pairs: {select}", fontsize=10, color=COLORS["text"])
    fig.tight_layout()
    return fig
