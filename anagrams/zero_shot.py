"""Zero-shot (CLIP / SigLIP-style) models as ordinary classifiers, so they run through
`anagram_eval` unchanged: forward(images) -> (B, n_classes) logits over text prompts.

With the default 1000 ImageNet class names the standard 1000 -> 9 mapping applies; with the
9 anagram category names the outputs are used directly.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mapping import CLASSES


def imagenet_class_names(pretty: bool = False) -> list[str]:
    """The 1000 ImageNet class names (Keras `imagenet_class_index.json`, as used by Doshi et al.).
    pretty=True replaces underscores with spaces."""
    with resources.files(__package__).joinpath("imagenet_class_index.json").open() as f:
        index = json.load(f)
    names = [index[str(i)][1] for i in range(1000)]
    return [n.replace("_", " ") for n in names] if pretty else names


class ZeroShotClassifier(nn.Module):
    """Wrap an image encoder + fixed text embeddings as a classifier.

    logits = normalize(encode_image(x)) @ text_features.T * logit_scale + logit_bias
    """

    def __init__(self, model: nn.Module, text_features: torch.Tensor, logit_scale: float = 1.0,
                 logit_bias: float = 0.0, encode_fn: str = "encode_image"):
        super().__init__()
        self.model = model
        self.encode_fn = encode_fn
        self.register_buffer("text_features", F.normalize(text_features.float(), dim=-1))
        self.logit_scale = float(logit_scale)
        self.logit_bias = float(logit_bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        image_features = getattr(self.model, self.encode_fn)(images)
        image_features = F.normalize(image_features.float(), dim=-1)
        return image_features @ self.text_features.T * self.logit_scale + self.logit_bias

    @classmethod
    def from_open_clip(
        cls,
        model_name: str,
        class_names: Sequence[str] | None = None,
        templates: Sequence[str] = ("{}",),
        device: str | torch.device = "cpu",
    ):
        """Build from an open_clip model, e.g. 'hf-hub:timm/ViT-L-16-SigLIP2-256'.

        Returns (classifier, preprocess). class_names default to the 1000 ImageNet names with the
        bare '{}' template, matching the SigLIP setup in Doshi et al. Pass class_names=CLASSES for
        a direct 9-way zero-shot classifier.
        """
        import open_clip  # optional dependency: uv sync --extra validation

        model, preprocess = open_clip.create_model_from_pretrained(model_name)
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(device).eval()
        class_names = list(class_names) if class_names is not None else imagenet_class_names()

        with torch.inference_mode():
            text_features = []
            for name in class_names:
                tokens = tokenizer([t.format(name) for t in templates]).to(device)
                emb = F.normalize(model.encode_text(tokens).float(), dim=-1).mean(dim=0)
                text_features.append(emb)
            text_features = torch.stack(text_features)
            logit_scale = model.logit_scale.exp().item()
            logit_bias = model.logit_bias.item() if getattr(model, "logit_bias", None) is not None else 0.0

        return cls(model, text_features, logit_scale, logit_bias).to(device), preprocess


__all__ = ["ZeroShotClassifier", "imagenet_class_names", "CLASSES"]
