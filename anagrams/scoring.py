"""From per-image scores to predictions, decision margins, pair outcomes, and summary metrics.

Everything downstream of the model forward pass lives here and is pure numpy/pandas, so
metrics can be recomputed from a saved predictions table without a model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .mapping import CLASSES
from .version import __version__

SQRT2 = math.sqrt(2.0)
CHANCE_ACC = 1 / len(CLASSES)
CHANCE_CSS = CHANCE_ACC**2

META_COLS = ["filename", "anagram_id", "pair_id", "variant", "position", "label", "foil", "object0", "object1"]
SCORE_COLS = [f"score_{c}" for c in CLASSES]


# ---------------------------------------------------------------------------------------------
# per-image predictions
# ---------------------------------------------------------------------------------------------
def build_predictions(meta: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """One row per image: metadata + 9 scores + prediction + decision margins.

    Args:
        meta: DataFrame with META_COLS; label/foil/object0/object1 as category *names*.
        scores: (N, 9) array aligned with meta rows, columns in CLASSES order.

    Decision margin (Alvarez & Konkle 2024): (S_label - max_{j != label} S_j) / sqrt(2).
    Foil margin (anagram-specific): (S_label - S_foil) / sqrt(2).
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(meta), len(CLASSES)):
        raise ValueError(f"scores must be ({len(meta)}, {len(CLASSES)}), got {scores.shape}")

    df = meta[META_COLS].reset_index(drop=True).copy()
    rows = np.arange(len(df))
    label_idx = df["label"].map(CLASSES.index).to_numpy()
    foil_idx = df["foil"].map(CLASSES.index).to_numpy()
    pred_idx = scores.argmax(axis=1)

    target_score = scores[rows, label_idx]
    others = scores.copy()
    others[rows, label_idx] = -np.inf
    max_nontarget_score = others.max(axis=1)

    df["pred"] = [CLASSES[i] for i in pred_idx]
    df["correct"] = pred_idx == label_idx
    for j, name in enumerate(CLASSES):
        df[f"score_{name}"] = scores[:, j]
    df["target_score"] = target_score
    df["max_nontarget_score"] = max_nontarget_score
    df["decision_margin"] = (target_score - max_nontarget_score) / SQRT2
    df["foil_margin"] = (target_score - scores[rows, foil_idx]) / SQRT2
    return df


# ---------------------------------------------------------------------------------------------
# per-pair outcomes
# ---------------------------------------------------------------------------------------------
def build_pairs(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per anagram pair. both_correct is the CSS event; pair_margin = min(dm0, dm1)."""
    p0 = predictions[predictions["position"] == 0].set_index("anagram_id").sort_index()
    p1 = predictions[predictions["position"] == 1].set_index("anagram_id").sort_index()
    if not p0.index.equals(p1.index):
        raise ValueError("every anagram_id must have exactly one position-0 and one position-1 image")

    pairs = pd.DataFrame(
        {
            "anagram_id": p0.index.to_numpy(),
            "pair_id": p0["pair_id"].to_numpy(),
            "variant": p0["variant"].to_numpy(),
            "object0": p0["object0"].to_numpy(),
            "object1": p0["object1"].to_numpy(),
            "correct0": p0["correct"].to_numpy(),
            "correct1": p1["correct"].to_numpy(),
            "dm0": p0["decision_margin"].to_numpy(),
            "dm1": p1["decision_margin"].to_numpy(),
        }
    )
    pairs["both_correct"] = pairs["correct0"] & pairs["correct1"]
    pairs["pair_margin"] = np.minimum(pairs["dm0"], pairs["dm1"])
    return pairs


# ---------------------------------------------------------------------------------------------
# summary metrics
# ---------------------------------------------------------------------------------------------
def bootstrap_ci(values, n_boot: int = 1000, ci: float = 95.0, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (resampling items with replacement). Seeded."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(lo), float(hi)


def confusion_matrix(predictions: pd.DataFrame) -> pd.DataFrame:
    """9x9 counts, rows = true label, columns = predicted label."""
    true_idx = predictions["label"].map(CLASSES.index).to_numpy()
    pred_idx = predictions["pred"].map(CLASSES.index).to_numpy()
    counts = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(counts, (true_idx, pred_idx), 1)
    cm = pd.DataFrame(counts, index=CLASSES, columns=CLASSES)
    cm.index.name, cm.columns.name = "true", "pred"
    return cm


def summarize(predictions: pd.DataFrame, pairs: pd.DataFrame, seed: int = 0) -> dict:
    """Headline numbers. `css` is the Configural Shape Score (fraction of pairs with both correct)."""
    css_lo, css_hi = bootstrap_ci(pairs["both_correct"].to_numpy(), seed=seed)
    dm = predictions["decision_margin"]
    return {
        "css": float(pairs["both_correct"].mean()),
        "css_ci_low": css_lo,
        "css_ci_high": css_hi,
        "acc": float(predictions["correct"].mean()),
        "foil_rate": float((predictions["pred"] == predictions["foil"]).mean()),
        "dm_mean": float(dm.mean()),
        "dm_median": float(dm.median()),
        "dm_min": float(dm.min()),
        "dm_max": float(dm.max()),
        "dm_kurtosis": float(dm.kurt()),  # pandas: Fisher, bias-corrected
        "foil_margin_mean": float(predictions["foil_margin"].mean()),
        "pair_margin_mean": float(pairs["pair_margin"].mean()),
        "chance_acc": CHANCE_ACC,
        "chance_css": CHANCE_CSS,
        "n_images": int(len(predictions)),
        "n_pairs": int(len(pairs)),
    }


# ---------------------------------------------------------------------------------------------
# results container
# ---------------------------------------------------------------------------------------------
@dataclass
class AnagramResults:
    """Everything the eval produces. `predictions` is the source of truth; the rest derives from it."""

    summary: dict
    predictions: pd.DataFrame
    pairs: pd.DataFrame
    confusion: pd.DataFrame

    def save(self, directory: str | Path) -> Path:
        """Write summary.json + predictions.csv (pairs/confusion are re-derived on load)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "summary.json").write_text(json.dumps(self.summary, indent=2) + "\n")
        self.predictions.to_csv(directory / "predictions.csv", index=False)
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "AnagramResults":
        directory = Path(directory)
        summary = json.loads((directory / "summary.json").read_text())
        predictions = pd.read_csv(directory / "predictions.csv", dtype={"anagram_id": str},
                                  float_precision="round_trip")
        meta = {k: v for k, v in summary.items() if k in _META_KEYS(summary)}
        return score_predictions(predictions, **meta)

    def __repr__(self) -> str:
        s = self.summary
        return (
            f"AnagramResults(css={s['css']:.3f} [{s['css_ci_low']:.3f}, {s['css_ci_high']:.3f}], "
            f"acc={s['acc']:.3f}, foil_rate={s['foil_rate']:.3f}, n_pairs={s['n_pairs']})"
        )


def _META_KEYS(summary: dict) -> set:
    """Keys in a saved summary that are bookkeeping (config, model, ...) rather than metrics."""
    metric_keys = {
        "css", "css_ci_low", "css_ci_high", "acc", "foil_rate", "dm_mean", "dm_median", "dm_min",
        "dm_max", "dm_kurtosis", "foil_margin_mean", "pair_margin_mean", "chance_acc", "chance_css",
        "n_images", "n_pairs",
    }
    return set(summary) - metric_keys


def score_predictions(predictions: pd.DataFrame, seed: int = 0, **meta) -> AnagramResults:
    """Build the full results object from a predictions table. Extra kwargs (config, model, ...)
    are recorded in the summary as bookkeeping."""
    pairs = build_pairs(predictions)
    summary = {"eval_version": __version__, **meta, **summarize(predictions, pairs, seed=seed)}
    confusion = confusion_matrix(predictions)
    return AnagramResults(summary=summary, predictions=predictions, pairs=pairs, confusion=confusion)
