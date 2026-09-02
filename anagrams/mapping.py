"""Category names and the 1000-way ImageNet -> 9-way anagram-category mapping."""

from __future__ import annotations

import torch

CLASSES = ["bear", "bunny", "cat", "elephant", "frog", "lizard", "tiger", "turtle", "wolf"]

# Paper Table 2 (Baker & Elder 2022 mapping): ImageNet-1k class indices per anagram category.
IMAGENET_CLASS_MAP = {
    "bear": [294, 295, 296, 297],
    "bunny": [330, 331, 332],
    "cat": [281, 282, 283, 284, 285],
    "elephant": [101, 385, 386],
    "frog": [30, 31, 32],
    "lizard": [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
    "tiger": [286, 287, 288, 289, 290, 291, 292, 293],
    "turtle": [33, 34, 35, 36, 37],
    "wolf": [269, 270, 271, 272, 273, 274, 275],
}


def imagenet_to_anagram_scores(outputs: torch.Tensor, reduction: str = "max") -> torch.Tensor:
    """Map (B, 1000) ImageNet scores to (B, 9) anagram-category scores.

    For each category, take the max (paper default) or mean over its ImageNet classes.
    Works on logits or probabilities; argmax is unchanged by monotone transforms.
    """
    if outputs.ndim != 2 or outputs.shape[1] != 1000:
        raise ValueError(f"expected (B, 1000) ImageNet outputs, got {tuple(outputs.shape)}")
    if reduction not in ("max", "mean"):
        raise ValueError(f"reduction must be 'max' or 'mean', got {reduction!r}")
    columns = []
    for name in CLASSES:
        sub = outputs[:, IMAGENET_CLASS_MAP[name]]
        columns.append(sub.max(dim=1).values if reduction == "max" else sub.mean(dim=1))
    return torch.stack(columns, dim=1)


def default_to_anagram_scores(outputs) -> torch.Tensor:
    """Default hook: 1000-d outputs -> ImageNet mapping; 9-d outputs -> unchanged."""
    if not isinstance(outputs, torch.Tensor):
        raise TypeError(
            f"model returned {type(outputs).__name__}, not a Tensor; pass "
            "to_anagram_scores=<callable returning (B, 9) scores> to anagram_eval"
        )
    if outputs.ndim != 2:
        raise ValueError(f"expected 2-d (B, C) outputs, got shape {tuple(outputs.shape)}")
    n_classes = outputs.shape[1]
    if n_classes == len(CLASSES):
        return outputs
    if n_classes == 1000:
        return imagenet_to_anagram_scores(outputs)
    raise ValueError(
        f"model returned {n_classes}-d outputs; anagram_eval knows how to handle 1000-d (ImageNet) "
        "and 9-d outputs. Pass to_anagram_scores=<callable mapping outputs to (B, 9) scores>."
    )
