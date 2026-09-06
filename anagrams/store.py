"""Results store: Hive-partitioned results on S3 with an identical local mirror.

Layout (same tree in the bucket and in the local cache):

    eval=anagrams/version=0.1.0/dataset=pairs-72/model=pytorch__alexnet__7be5be79/readout=head/results.parquet
    eval=anagrams/version=0.1.0/dataset=pairs-72/model=pytorch__alexnet__7be5be79/readout=head/summary.json

`readout=` names how class scores were read out of the backbone: `head` (native classifier),
`probe__<layer>__<hash8>`, `prototypes__<layer>__<hash8>`, `zeroshot__<hash8>`. Every readout_id is
the sha256[:8] of the head/prototype file (same rule as weights; contract with harvard-visionlab/models).

- `results.parquet`: one row per image (predictions, scores, margins) plus identity columns.
- `summary.json`: metrics + identity + run provenance. Both files are self-describing.
- Path segments are Hive-style so DuckDB / polars / Spark / Athena expose eval, version, dataset,
  model as columns with `hive_partitioning=true`. Dashboards filter on the `model_id` column, never
  the path. Objects are private; read through lab AWS credentials (boto3 default chain).

Model identity: `model_id = source/arch:weights_id` with `weights_id` = sha256[:8] of the weights
file (the harvard-visionlab/models convention). Aliases like `DEFAULT` are resolved before storing
so a path always names one specific set of weights.

    store = ResultsStore()
    store.run("pytorch/alexnet:DEFAULT", dataset="pairs-72")    # load, eval, save (or return cached)
    store.query(dataset="pairs-72")                              # one summary row per stored model
    store.load("pairs-72", "pytorch/alexnet:7be5be79")           # full AnagramResults (readout="head")
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .data import DEFAULT_CONFIG, REPO_ID
from .scoring import AnagramResults
from .version import __version__

EVAL_NAME = "anagrams"
DEFAULT_BUCKET = os.environ.get("VISIONLAB_EVALS_BUCKET", "visionlab-evals")
DEFAULT_CACHE = Path(os.environ.get("VISIONLAB_EVALS_CACHE", Path.home() / ".cache" / "visionlab" / "evals"))
IDENTITY_COLS = ["eval_name", "eval_version", "dataset", "model_id", "model_spec", "model_source", "model_arch",
                 "weights_id", "readout_type", "readout_layer", "readout_id", "readout_spec", "readout_n_classes",
                 "readout_train_data", "readout_primary"]
READOUT_TYPES = ("head", "probe", "prototypes", "zeroshot")
FILES = ("summary.json", "results.parquet")

_SLUG_OK = re.compile(r"^[A-Za-z0-9._-]+$")
_WEIGHTS_ID_OK = re.compile(r"^([0-9a-f]{8}|NONE)$")


def model_slug(model_id: str) -> str:
    """'pytorch/alexnet:7be5be79' -> 'pytorch__alexnet__7be5be79'. Only [A-Za-z0-9._-] survive."""
    slug = model_id.replace("/", "__").replace(":", "__")
    if not _SLUG_OK.match(slug):
        raise ValueError(f"model_id {model_id!r} -> slug {slug!r} has characters outside [A-Za-z0-9._-]")
    return slug


@dataclass(frozen=True)
class ReadoutIdentity:
    """How class scores were read out of a backbone (mirrors visionlab.models ReadoutIdentity).
    `readout_id` is sha256[:8] of the head / prototype file."""

    readout_type: str
    readout_id: str
    readout_layer: str | None = None
    readout_spec: str | None = None
    readout_n_classes: int | None = None
    readout_train_data: str | None = None
    readout_primary: bool = True

    def __post_init__(self):
        if self.readout_type not in READOUT_TYPES:
            raise ValueError(f"readout_type must be one of {READOUT_TYPES}, got {self.readout_type!r}")
        if not _WEIGHTS_ID_OK.match(self.readout_id):
            raise ValueError(f"readout_id must be sha256[:8], got {self.readout_id!r}")

    @property
    def slug(self) -> str:
        if self.readout_type == "head":
            return "head"
        layer = _layer_slug(self.readout_layer) if self.readout_layer else None
        parts = [self.readout_type] + ([layer] if layer else []) + [self.readout_id]
        return "__".join(parts)

    def as_dict(self) -> dict:
        return {
            "readout_type": self.readout_type,
            "readout_layer": self.readout_layer,
            "readout_id": self.readout_id,
            "readout_spec": self.readout_spec or self.slug,
            "readout_n_classes": self.readout_n_classes,
            "readout_train_data": self.readout_train_data,
            "readout_primary": self.readout_primary,
        }


def _layer_slug(layer: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", layer)


@dataclass(frozen=True)
class ModelIdentity:
    """Who produced a result: backbone (`weights_id` = sha256[:8] of the weights file, 'NONE' = random
    init) plus the readout used to get class scores (None = native head)."""

    model_source: str
    model_arch: str
    weights_id: str
    model_spec: str | None = None  # the spec as typed, e.g. 'pytorch/alexnet:DEFAULT'
    readout: ReadoutIdentity | None = None
    collection_names: dict = field(default_factory=dict)  # e.g. {"Doshi2025": "resnet50_in1k"} (summary only)

    def __post_init__(self):
        if not _WEIGHTS_ID_OK.match(self.weights_id):
            raise ValueError(
                f"weights_id must be sha256[:8] (8 lowercase hex chars) or 'NONE', got {self.weights_id!r}. "
                "Resolve aliases like 'DEFAULT' to the hashid before storing results."
            )
        for part in (self.model_source, self.model_arch):
            if not _SLUG_OK.match(part):
                raise ValueError(f"{part!r} has characters outside [A-Za-z0-9._-]")

    @property
    def model_id(self) -> str:
        return f"{self.model_source}/{self.model_arch}:{self.weights_id}"

    @property
    def readout_slug(self) -> str:
        return self.readout.slug if self.readout is not None else "head"

    def as_dict(self) -> dict:
        readout = self.readout or ReadoutIdentity("head", self.weights_id, readout_spec="head")
        return {
            "model_id": self.model_id,
            "model_spec": self.model_spec or self.model_id,
            "model_source": self.model_source,
            "model_arch": self.model_arch,
            "weights_id": self.weights_id,
            **readout.as_dict(),
        }

    @classmethod
    def from_visionlab(cls, identity, spec: str | None = None) -> "ModelIdentity":
        """Adapt a `visionlab.models.ModelIdentity` (fields source, name, hashid, alias, optional readout)."""
        readout = None
        ro = getattr(identity, "readout", None)
        if ro is not None and getattr(ro, "type", "head") != "head":
            readout = ReadoutIdentity(
                readout_type=ro.type, readout_id=ro.hashid, readout_layer=getattr(ro, "layer", None),
                readout_spec=getattr(ro, "tag", None), readout_n_classes=getattr(ro, "n_classes", None),
                readout_train_data=getattr(ro, "train_data", None), readout_primary=bool(getattr(ro, "primary", True)),
            )
        typed = spec or getattr(identity, "spec", None) or getattr(identity, "alias", None)
        names = getattr(identity, "collection_names", None) or {}
        return cls(identity.source, identity.name, identity.hashid, model_spec=typed, readout=readout,
                   collection_names=dict(names))

    @classmethod
    def from_spec(cls, spec: str) -> "ModelIdentity":
        """Resolve a visionlab.models spec ('source/name:weights[@readout]') without loading the model."""
        from visionlab.models import resolve  # optional dependency: uv sync --extra models

        return cls.from_visionlab(resolve(spec), spec=spec)

    @classmethod
    def from_torchvision(cls, weights) -> "ModelIdentity":
        """From a torchvision WeightsEnum member, e.g. AlexNet_Weights.IMAGENET1K_V1 -> pytorch/alexnet:7be5be79.
        torchvision names weight files '<arch>-<sha256[:8]>.pth', so the hashid comes from the URL."""
        arch = type(weights).__name__.removesuffix("_Weights").lower()
        stem = Path(weights.url).name.split(".")[0]
        return cls("pytorch", arch, stem.rsplit("-", 1)[-1], model_spec=f"pytorch/{arch}:{weights.name}")


class ResultsStore:
    """Save / load / query anagram results in the Hive tree, locally mirrored and on S3."""

    def __init__(
        self,
        bucket: str | None = DEFAULT_BUCKET,
        cache_dir: str | Path = DEFAULT_CACHE,
        eval_name: str = EVAL_NAME,
        eval_version: str = __version__,
    ):
        self.bucket = bucket or None  # None -> local only
        self.cache_dir = Path(cache_dir)
        self.eval_name = eval_name
        self.eval_version = eval_version
        self._client = None

    # ----------------------------------------------------------------------------------- paths
    def prefix(self, dataset: str | None = None, model_id: str | None = None, readout: str = "head") -> str:
        parts = [f"eval={self.eval_name}", f"version={self.eval_version}"]
        if dataset is not None:
            parts.append(f"dataset={dataset}")
        if model_id is not None:
            if dataset is None:
                raise ValueError("model_id requires dataset")
            parts += [f"model={model_slug(model_id)}", f"readout={readout}"]
        return "/".join(parts)

    def local_dir(self, dataset: str, model_id: str, readout: str = "head") -> Path:
        return self.cache_dir / self.prefix(dataset, model_id, readout)

    def s3_uri(self, dataset: str | None = None, model_id: str | None = None, readout: str = "head") -> str:
        return f"s3://{self.bucket}/{self.prefix(dataset, model_id, readout)}"

    def duckdb_glob(self, dataset: str | None = None) -> str:
        """Glob for `read_parquet(<glob>, hive_partitioning=true)` over the S3 tree (or the local mirror)."""
        root = f"s3://{self.bucket}" if self.bucket else str(self.cache_dir)
        depth = "*/*/" if dataset is not None else "*/*/*/"  # [dataset/]model/readout
        return f"{root}/{self.prefix(dataset)}/{depth}results.parquet"

    # ------------------------------------------------------------------------------------- s3
    @property
    def s3(self):
        if self._client is None and self.bucket:
            import boto3

            self._client = boto3.client("s3")
        return self._client

    def _remote_exists(self, key: str) -> bool:
        if not self.bucket:
            return False
        from botocore.exceptions import ClientError

        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def _list_summaries(self, dataset: str | None = None) -> list[str]:
        """Keys of every summary.json under the prefix (S3 if configured, else the local mirror)."""
        prefix = self.prefix(dataset) + "/"
        if self.bucket:
            keys = []
            for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
                keys += [o["Key"] for o in page.get("Contents", []) if o["Key"].endswith("/summary.json")]
            return sorted(keys)
        root = self.cache_dir / prefix
        return sorted(str(p.relative_to(self.cache_dir)) for p in root.glob("**/summary.json"))

    # -------------------------------------------------------------------------------- read
    def exists(self, dataset: str, model_id: str, readout: str = "head") -> bool:
        local = self.local_dir(dataset, model_id, readout)
        if all((local / f).exists() for f in FILES):
            return True
        return self._remote_exists(f"{self.prefix(dataset, model_id, readout)}/summary.json")

    def load(self, dataset: str, model_id: str, readout: str = "head") -> AnagramResults:
        """Full results (local mirror first, else fetched from S3 into the mirror). `readout` is the slug."""
        local = self.local_dir(dataset, model_id, readout)
        if not all((local / f).exists() for f in FILES):
            if not self.bucket:
                raise FileNotFoundError(f"no results for {model_id}@{readout} on {dataset} in {local}")
            local.mkdir(parents=True, exist_ok=True)
            for f in FILES:
                key = f"{self.prefix(dataset, model_id, readout)}/{f}"
                self.s3.download_file(self.bucket, key, str(local / f))
        return AnagramResults.load(local)

    def predictions(self, dataset: str, model_id: str, readout: str = "head") -> pd.DataFrame:
        return self.load(dataset, model_id, readout).predictions

    def query(self, dataset: str | None = None, models: list[str] | None = None,
              readout_type: str | None = None) -> pd.DataFrame:
        """One row per stored (dataset, model, readout): identity + metrics + provenance. summary.json files are
        small, so they are always re-fetched from S3 (the mirror copy is refreshed)."""
        rows = []
        for key in self._list_summaries(dataset):
            local = self.cache_dir / key
            if self.bucket:
                local.parent.mkdir(parents=True, exist_ok=True)
                self.s3.download_file(self.bucket, key, str(local))
            rows.append(json.loads(local.read_text()))
        df = pd.DataFrame(rows)
        if models is not None and len(df):
            df = df[df["model_id"].isin(models)]
        if readout_type is not None and len(df):
            df = df[df["readout_type"] == readout_type]
        if len(df):
            df = df.sort_values(["dataset", "css"], ascending=[True, False]).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------------------- write
    def save(self, results: AnagramResults, identity: ModelIdentity, dataset: str, force: bool = False) -> str:
        """Stamp identity into both files, write the local mirror, upload to S3. Returns the S3/local URI."""
        if results.summary.get("dataset") not in (None, dataset):
            raise ValueError(f"results were computed on {results.summary['dataset']!r}, not {dataset!r}")
        key = (dataset, identity.model_id, identity.readout_slug)
        if not force and self.exists(*key):
            raise FileExistsError(f"{self.prefix(*key)} exists; pass force=True to overwrite")

        ident = {"eval_name": self.eval_name, "eval_version": self.eval_version, "dataset": dataset,
                 **identity.as_dict()}
        extra = {"collection_names": json.dumps(identity.collection_names, sort_keys=True)}
        summary = {**ident, **extra, **{k: v for k, v in results.summary.items() if k not in ident},
                   **_dataset_revision()}
        predictions = results.predictions.copy()
        for k in IDENTITY_COLS:
            predictions[k] = ident[k]
        predictions = predictions[IDENTITY_COLS + [c for c in predictions.columns if c not in IDENTITY_COLS]]
        stamped = AnagramResults(summary, predictions, results.pairs, results.confusion)

        local = stamped.save(self.local_dir(*key))
        if self.bucket:
            for f in FILES:
                self.s3.upload_file(str(local / f), self.bucket, f"{self.prefix(*key)}/{f}")
            return self.s3_uri(*key)
        return str(local)

    def run(
        self,
        model_or_spec,
        transform=None,
        dataset: str = DEFAULT_CONFIG,
        identity: ModelIdentity | None = None,
        force: bool = False,
        seed: int | None = None,
        **eval_kwargs,
    ) -> AnagramResults:
        """Evaluate and store, or return the stored result.

        Pass a visionlab.models spec string ('source/name:weights[@readout]'; model, `transforms.test` and
        identity come from the model card), or a model + transform + explicit `identity`.
        Random-init baselines (weights 'NONE') are loaded with `seed` (default 0) and the seed is recorded
        as `run_seed`; the identity itself does not encode the seed.
        """
        from .eval import anagram_eval

        if isinstance(model_or_spec, str):
            identity = ModelIdentity.from_spec(model_or_spec)
        elif identity is None:
            raise ValueError("pass identity=ModelIdentity(...) when giving a model object")

        key = (dataset, identity.model_id, identity.readout_slug)
        if not force and self.exists(*key):
            return self.load(*key)

        if identity.weights_id == "NONE" and seed is None:
            seed = 0
        if seed is not None:
            eval_kwargs.setdefault("run_seed", seed)

        if isinstance(model_or_spec, str):
            from visionlab.models import load_model

            load_kwargs = {"seed": seed} if seed is not None else {}
            model, transforms, vl_identity = load_model(model_or_spec, **load_kwargs)
            identity = ModelIdentity.from_visionlab(vl_identity, spec=model_or_spec)
            transform = transforms.test if transform is None else transform
        else:
            model = model_or_spec
            if transform is None:
                raise ValueError("transform is required when giving a model object")

        results = anagram_eval(model, transform, config=dataset, **eval_kwargs)
        self.save(results, identity, dataset, force=force)
        return self.load(*key)


def _dataset_revision() -> dict:
    """Commit sha of the HF dataset at save time (best effort; empty when offline)."""
    try:
        from huggingface_hub import HfApi

        return {"dataset_revision": HfApi().dataset_info(REPO_ID).sha}
    except Exception:
        return {}
