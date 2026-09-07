"""Evaluation-specification identity (identity contract v2, eval side).

An *evaluation spec* is everything that determines a result besides the model configuration:
which data (pinned revision), how images are preprocessed, how model outputs become the 9 category
scores, the scoring protocol, and execution settings that change numerics. Its sha256 is the
`eval_spec_id`; two runs with the same `config_id` + `eval_spec_id` are repeats of one experiment.

    spec = EvalSpec(config_id=..., dataset=DatasetRef(...), preprocessing=transform_signature(tfm), ...)
    spec.eval_spec_id           # sha256 hex
    spec.reusable               # False when any component is not exactly known (never a cache hit)
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .data import REPO_ID
from .mapping import CLASSES, IMAGENET_CLASS_MAP
from .version import __version__

SPEC_SCHEMA = "visionlab-evals/anagrams/spec@1"
SCORING_PROTOCOL = "css@1"  # both-images-correct per pair; bootstrap CI over pairs; margins /sqrt(2)


def canonical_json(obj) -> str:
    # same canonicalization as visionlab.models config_id: sorted keys, compact separators, ascii, no NaN
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False, default=_json_default
    )


def _json_default(o):
    if isinstance(o, (set, tuple)):
        return sorted(o) if isinstance(o, set) else list(o)
    if hasattr(o, "item"):  # numpy scalars
        return o.item()
    return str(o)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ------------------------------------------------------------------------------------ preprocessing
def transform_signature(transform) -> dict:
    """Structured description of a preprocessing pipeline.

    Recognizes torchvision Compose steps (Resize, CenterCrop, Normalize, ToTensor, visionlab OpenImage);
    anything else is recorded by repr and marks the signature `exact: False` (stored, never reused).
    """
    steps, exact = [], True
    for t in getattr(transform, "transforms", [transform]):
        name = type(t).__name__
        if name == "Resize":
            size = t.size if isinstance(t.size, int) else list(t.size)
            interp = getattr(t.interpolation, "value", str(t.interpolation))
            steps.append(
                {
                    "op": "Resize",
                    "size": size,
                    "interpolation": interp,
                    "antialias": bool(getattr(t, "antialias", True)),
                    "max_size": t.max_size,
                }
            )
        elif name == "CenterCrop":
            steps.append({"op": "CenterCrop", "size": list(t.size)})
        elif name == "Normalize":
            steps.append(
                {
                    "op": "Normalize",
                    "mean": [round(float(m), 6) for m in t.mean],
                    "std": [round(float(s), 6) for s in t.std],
                }
            )
        elif name in ("ToTensor", "MaybeToTensor", "PILToTensor", "OpenImage", "MaybeConvertMode"):
            steps.append({"op": name})
        else:
            steps.append({"op": "repr", "value": repr(t)})
            exact = False
    return {"kind": "torchvision", "steps": steps, "exact": exact}


def scorer_signature(to_anagram_scores) -> dict:
    """Identity of the outputs -> 9-way scores hook. Default hook and known reductions are exact."""
    if to_anagram_scores is None:
        return {"name": "default", "rule": "1000->imagenet_class_map:max | 9->identity", "exact": True}
    module = getattr(to_anagram_scores, "__module__", "?")
    qualname = getattr(to_anagram_scores, "__qualname__", type(to_anagram_scores).__name__)
    name = f"{module}.{qualname}"
    return {"name": f"custom:{name}", "exact": False}


OUTPUT_MAP = {"version": "doshi2025-table2@1", "classes": CLASSES, "imagenet_class_map": IMAGENET_CLASS_MAP}


@dataclass(frozen=True)
class DatasetRef:
    repo: str = REPO_ID
    config: str = "pairs-72"
    revision: str | None = None  # HF commit sha pinned at load; None = unpinned (not reusable)
    split: str = "test"


@dataclass(frozen=True)
class Execution:
    seed: int | None = None  # model seed (random init); None for pretrained weights
    precision: str = "fp32-strict"  # or "tf32" (Ampere+ default); device type is provenance, not spec


@dataclass(frozen=True)
class EvalSpec:
    """The hashed evaluation specification. `config_id` comes from the models manifest (A1); until models
    ships it, callers pass a fallback and set `config_id_source`."""

    config_id: str
    dataset: DatasetRef
    preprocessing: dict
    scorer: dict = field(default_factory=lambda: scorer_signature(None))
    execution: Execution = field(default_factory=Execution)
    output_map_version: str = OUTPUT_MAP["version"]
    scoring: str = SCORING_PROTOCOL
    eval_name: str = "anagrams"
    eval_version: str = __version__
    config_id_source: str = "models-manifest"  # or "eval-fallback" (not reusable)

    def hashed(self) -> dict:
        """The block whose canonical JSON is hashed. Everything in it changes the experiment."""
        return {
            "schema": SPEC_SCHEMA,
            "eval_name": self.eval_name,
            "eval_version": self.eval_version,
            "config_id": self.config_id,
            "dataset": asdict(self.dataset),
            "preprocessing": {k: v for k, v in self.preprocessing.items() if k != "exact"},
            "output_map_version": self.output_map_version,
            "scorer": {k: v for k, v in self.scorer.items() if k != "exact"},
            "scoring": self.scoring,
            "execution": asdict(self.execution),
        }

    @property
    def eval_spec_id(self) -> str:
        return sha256_hex(canonical_json(self.hashed()))

    @property
    def reusable(self) -> bool:
        """Can a stored run with this id stand in for a new run? Only if every component is exactly known."""
        return (
            self.dataset.revision is not None
            and self.preprocessing.get("exact", False)
            and self.scorer.get("exact", False)
            and self.config_id_source == "models-manifest"
        )

    def to_dict(self) -> dict:
        return {
            **self.hashed(),
            "eval_spec_id": self.eval_spec_id,
            "reusable": self.reusable,
            "preprocessing_exact": self.preprocessing.get("exact", False),
            "scorer_exact": self.scorer.get("exact", False),
            "config_id_source": self.config_id_source,
        }


def new_run_id(eval_spec_id: str) -> str:
    """'<UTC timestamp>-<8 hex>': sortable, unique per execution, never derived from results."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    nonce = sha256_hex(f"{eval_spec_id}|{os.uname().nodename}|{os.getpid()}|{secrets.token_hex(8)}")[:8]
    return f"{stamp}-{nonce}"
