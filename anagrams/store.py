"""Results store: Hive-partitioned results on S3 with an identical local mirror.

Layout (same tree in the bucket and in the local cache):

    eval=anagrams/version=0.1.0/dataset=pairs-72/model=pytorch__alexnet__7be5be79/results.parquet
    eval=anagrams/version=0.1.0/dataset=pairs-72/model=pytorch__alexnet__7be5be79/summary.json

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
    store.load("pairs-72", "pytorch/alexnet:7be5be79")           # full AnagramResults
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data import DEFAULT_CONFIG, REPO_ID
from .scoring import AnagramResults
from .version import __version__

EVAL_NAME = "anagrams"
DEFAULT_BUCKET = os.environ.get("VISIONLAB_EVALS_BUCKET", "visionlab-evals")
DEFAULT_CACHE = Path(os.environ.get("VISIONLAB_EVALS_CACHE", Path.home() / ".cache" / "visionlab" / "evals"))
IDENTITY_COLS = ["eval_name", "eval_version", "dataset", "model_id", "model_spec", "model_source", "model_arch",
                 "weights_id"]
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
class ModelIdentity:
    """Who produced a result. `weights_id` is sha256[:8] of the weights file ('NONE' = random init)."""

    model_source: str
    model_arch: str
    weights_id: str
    model_spec: str | None = None  # the spec as typed, e.g. 'pytorch/alexnet:DEFAULT'

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

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_spec": self.model_spec or self.model_id,
            "model_source": self.model_source,
            "model_arch": self.model_arch,
            "weights_id": self.weights_id,
        }

    @classmethod
    def from_spec(cls, spec: str) -> "ModelIdentity":
        """Resolve a visionlab.models spec ('source/name[:weights]') through its model card."""
        from visionlab.models import get_card, parse_spec  # optional dependency: uv sync --extra models

        parsed = parse_spec(spec)
        card = get_card(parsed)
        weights_id = "NONE" if parsed.weights == "NONE" else card.get_weights(parsed.weights).hashid
        return cls(card.source, card.name, weights_id, model_spec=spec)

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
    def prefix(self, dataset: str | None = None, model_id: str | None = None) -> str:
        parts = [f"eval={self.eval_name}", f"version={self.eval_version}"]
        if dataset is not None:
            parts.append(f"dataset={dataset}")
        if model_id is not None:
            if dataset is None:
                raise ValueError("model_id requires dataset")
            parts.append(f"model={model_slug(model_id)}")
        return "/".join(parts)

    def local_dir(self, dataset: str, model_id: str) -> Path:
        return self.cache_dir / self.prefix(dataset, model_id)

    def s3_uri(self, dataset: str | None = None, model_id: str | None = None) -> str:
        return f"s3://{self.bucket}/{self.prefix(dataset, model_id)}"

    def duckdb_glob(self, dataset: str | None = None) -> str:
        """Glob for `read_parquet(<glob>, hive_partitioning=true)` over the S3 tree (or the local mirror)."""
        root = f"s3://{self.bucket}" if self.bucket else str(self.cache_dir)
        return f"{root}/{self.prefix(dataset)}/{'*/' if dataset is not None else '*/*/'}results.parquet"

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
    def exists(self, dataset: str, model_id: str) -> bool:
        local = self.local_dir(dataset, model_id)
        if all((local / f).exists() for f in FILES):
            return True
        return self._remote_exists(f"{self.prefix(dataset, model_id)}/summary.json")

    def load(self, dataset: str, model_id: str) -> AnagramResults:
        """Full results (local mirror first, else fetched from S3 into the mirror)."""
        local = self.local_dir(dataset, model_id)
        if not all((local / f).exists() for f in FILES):
            if not self.bucket:
                raise FileNotFoundError(f"no results for {model_id} on {dataset} in {local}")
            local.mkdir(parents=True, exist_ok=True)
            for f in FILES:
                self.s3.download_file(self.bucket, f"{self.prefix(dataset, model_id)}/{f}", str(local / f))
        return AnagramResults.load(local)

    def predictions(self, dataset: str, model_id: str) -> pd.DataFrame:
        return self.load(dataset, model_id).predictions

    def query(self, dataset: str | None = None, models: list[str] | None = None) -> pd.DataFrame:
        """One row per stored (dataset, model): identity + metrics + provenance. summary.json files are
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
        if len(df):
            df = df.sort_values(["dataset", "css"], ascending=[True, False]).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------------------- write
    def save(self, results: AnagramResults, identity: ModelIdentity, dataset: str, force: bool = False) -> str:
        """Stamp identity into both files, write the local mirror, upload to S3. Returns the S3/local URI."""
        if results.summary.get("dataset") not in (None, dataset):
            raise ValueError(f"results were computed on {results.summary['dataset']!r}, not {dataset!r}")
        if not force and self.exists(dataset, identity.model_id):
            raise FileExistsError(f"{self.prefix(dataset, identity.model_id)} exists; pass force=True to overwrite")

        ident = {"eval_name": self.eval_name, "eval_version": self.eval_version, "dataset": dataset,
                 **identity.as_dict()}
        summary = {**ident, **{k: v for k, v in results.summary.items() if k not in ident}, **_dataset_revision()}
        predictions = results.predictions.copy()
        for k in IDENTITY_COLS:
            predictions[k] = ident[k]
        predictions = predictions[IDENTITY_COLS + [c for c in predictions.columns if c not in IDENTITY_COLS]]
        stamped = AnagramResults(summary, predictions, results.pairs, results.confusion)

        local = stamped.save(self.local_dir(dataset, identity.model_id))
        if self.bucket:
            for f in FILES:
                self.s3.upload_file(str(local / f), self.bucket, f"{self.prefix(dataset, identity.model_id)}/{f}")
            return self.s3_uri(dataset, identity.model_id)
        return str(local)

    def run(
        self,
        model_or_spec,
        transform=None,
        dataset: str = DEFAULT_CONFIG,
        identity: ModelIdentity | None = None,
        force: bool = False,
        **eval_kwargs,
    ) -> AnagramResults:
        """Evaluate and store, or return the stored result.

        Pass a visionlab.models spec string (model + `transforms.test` are loaded for you and the identity
        comes from the model card), or a model + transform + explicit `identity`.
        """
        from .eval import anagram_eval

        if isinstance(model_or_spec, str):
            identity = ModelIdentity.from_spec(model_or_spec)
        elif identity is None:
            raise ValueError("pass identity=ModelIdentity(...) when giving a model object")

        if not force and self.exists(dataset, identity.model_id):
            return self.load(dataset, identity.model_id)

        if isinstance(model_or_spec, str):
            from visionlab.models import load_model

            model, transforms = load_model(model_or_spec)
            transform = transforms.test if transform is None else transform
        else:
            model = model_or_spec
            if transform is None:
                raise ValueError("transform is required when giving a model object")

        results = anagram_eval(model, transform, config=dataset, **eval_kwargs)
        self.save(results, identity, dataset, force=force)
        return self.load(dataset, identity.model_id)


def _dataset_revision() -> dict:
    """Commit sha of the HF dataset at save time (best effort; empty when offline)."""
    try:
        from huggingface_hub import HfApi

        return {"dataset_revision": HfApi().dataset_info(REPO_ID).sha}
    except Exception:
        return {}
