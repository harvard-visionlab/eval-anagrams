"""visionlab.evals.anagrams — Object-Anagram / Configural Shape Score (CSS) eval.

Doshi, Fel, Konkle & Alvarez (NeurIPS 2025), arXiv:2507.00493.

    from visionlab.evals.anagrams import anagram_eval
    results = anagram_eval(model, transform, config="pairs-72")
    results.summary["css"]        # fraction of anagram pairs with both images correct
    results.predictions           # per-image table incl. decision margins
"""

from .data import CONFIGS, DEFAULT_CONFIG, REPO_ID, AnagramDataset, load_anagrams, metadata_frame
from .eval import anagram_eval, predict_scores
from .legacy import legacy_metrics
from .mapping import CLASSES, IMAGENET_CLASS_MAP, default_to_anagram_scores, imagenet_to_anagram_scores
from .plot import HUMAN_ACC, HUMAN_CSS, plot_acc_vs_css, plot_confusion, plot_css, plot_margins, show_pairs
from .scoring import (
    CHANCE_ACC,
    CHANCE_CSS,
    AnagramResults,
    bootstrap_ci,
    build_pairs,
    build_predictions,
    confusion_matrix,
    score_predictions,
    summarize,
)
from .version import __version__
from .zero_shot import ZeroShotClassifier, imagenet_class_names

__all__ = [
    "__version__",
    "anagram_eval",
    "predict_scores",
    "AnagramResults",
    "score_predictions",
    "build_predictions",
    "build_pairs",
    "summarize",
    "confusion_matrix",
    "bootstrap_ci",
    "legacy_metrics",
    "load_anagrams",
    "metadata_frame",
    "AnagramDataset",
    "CLASSES",
    "IMAGENET_CLASS_MAP",
    "imagenet_to_anagram_scores",
    "default_to_anagram_scores",
    "plot_css",
    "plot_acc_vs_css",
    "plot_confusion",
    "plot_margins",
    "show_pairs",
    "HUMAN_CSS",
    "HUMAN_ACC",
    "ZeroShotClassifier",
    "imagenet_class_names",
    "CHANCE_ACC",
    "CHANCE_CSS",
    "CONFIGS",
    "DEFAULT_CONFIG",
    "REPO_ID",
]
