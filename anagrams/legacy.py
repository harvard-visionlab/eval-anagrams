"""Metrics from the original NeurIPS implementation, kept only for parity checks.

The published code reported, besides single-image accuracy and CSS ("global_pair_acc"):
  target_or_foil_acc   P(pred in {object0, object1})
  target_over_foil_acc P(pred == label | pred in {object0, object1})
  target_foil_bias     sqrt(target_or_foil_acc * target_over_foil_acc)
Our summary replaces the trio with `foil_rate` = P(pred == foil); use this module to reproduce
the old numbers from a predictions table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .scoring import build_pairs


def legacy_metrics(predictions: pd.DataFrame) -> dict:
    pred, label, foil = predictions["pred"], predictions["label"], predictions["foil"]
    in_pair = (pred == label) | (pred == foil)
    target_or_foil_acc = float(in_pair.mean())
    target_over_foil_acc = float((pred[in_pair] == label[in_pair]).mean()) if in_pair.any() else float("nan")
    return {
        "acc": float((pred == label).mean()),
        "target_or_foil_acc": target_or_foil_acc,
        "target_over_foil_acc": target_over_foil_acc,
        "target_foil_bias": float(np.sqrt(target_or_foil_acc * target_over_foil_acc)),
        "global_pair_acc": float(build_pairs(predictions)["both_correct"].mean()),
    }
