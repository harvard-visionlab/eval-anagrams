"""anagram_eval: run a model over the anagram images and score it."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import DEFAULT_CONFIG, REPO_ID, AnagramDataset, load_anagrams, metadata_frame
from .mapping import CLASSES, default_to_anagram_scores
from .scoring import AnagramResults, build_predictions, score_predictions


def _resolve_device(model, device) -> torch.device:
    if device is not None:
        return torch.device(device)
    params = list(model.parameters()) if isinstance(model, torch.nn.Module) else []
    if params:
        return params[0].device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def predict_scores(
    model,
    transform: Callable,
    ds,
    to_anagram_scores: Callable | None = None,
    batch_size: int = 64,
    device=None,
    num_workers: int = 4,
    progress: bool = True,
) -> np.ndarray:
    """Forward every image once; return (N, 9) anagram-category scores in dataset order."""
    to_scores = to_anagram_scores or default_to_anagram_scores
    device = _resolve_device(model, device)
    if isinstance(model, torch.nn.Module):
        model = model.to(device).eval()

    loader = DataLoader(AnagramDataset(ds, transform), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    scores = np.full((len(ds), len(CLASSES)), np.nan, dtype=np.float64)
    for images, index in tqdm(loader, disable=not progress, desc="anagram_eval"):
        outputs = model(images.to(device, non_blocking=True))
        batch_scores = to_scores(outputs)
        if batch_scores.shape != (len(index), len(CLASSES)):
            raise ValueError(f"to_anagram_scores must return (B, 9), got {tuple(batch_scores.shape)}")
        scores[index.numpy()] = batch_scores.detach().float().cpu().numpy()
    assert not np.isnan(scores).any(), "some images were never scored"
    return scores


def anagram_eval(
    model,
    transform: Callable,
    config: str = DEFAULT_CONFIG,
    to_anagram_scores: Callable | None = None,
    batch_size: int = 64,
    device=None,
    num_workers: int = 4,
    progress: bool = True,
    revision: str | None = None,
    **meta,
) -> AnagramResults:
    """Configural Shape Score eval (Doshi et al., NeurIPS 2025).

    Args:
        model: callable mapping a (B, 3, H, W) batch to (B, 1000) ImageNet scores or (B, 9)
            anagram-category scores. Anything else needs `to_anagram_scores`.
        transform: PIL image -> tensor, the model's own eval preprocessing. Images are 256x256,
            so Resize(224)+CenterCrop(224) (visionlab `transforms.test`) matches the paper.
        config: "pairs-72" (paper main set) or "pairs-1440" (expanded set).
        to_anagram_scores: optional hook, model outputs -> (B, 9) scores in CLASSES order.
        **meta: bookkeeping recorded in the summary (e.g. model_name="pytorch/alexnet:DEFAULT").

    Returns:
        AnagramResults with .summary (css, acc, foil_rate, decision-margin stats, ...),
        .predictions (per image), .pairs (per anagram pair), .confusion (9x9).
    """
    ds = load_anagrams(config, revision=revision)
    scores = predict_scores(
        model, transform, ds, to_anagram_scores=to_anagram_scores, batch_size=batch_size,
        device=device, num_workers=num_workers, progress=progress,
    )
    predictions = build_predictions(metadata_frame(ds), scores)
    return score_predictions(predictions, dataset=REPO_ID, config=config, **meta)
