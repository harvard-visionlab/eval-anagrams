"""Results store: Hive-partitioned, immutable runs on S3 with an identical local mirror.

Layout (same tree in the bucket and the local cache):

    eval=anagrams/version=0.1.0/dataset=pairs-72/model=pytorch__alexnet__7be5be79/readout=head/intervention=none/
        run=20260907T120000Z-9743c99d/results.parquet   one row per image (+ identity, config_id, eval_spec_id, run_id)
                                     summary.json      metrics + identity + provenance
                                     manifest.json     written LAST = completion marker; carries the eval spec,
                                                       the models manifest, artifact hashes and the summary

Four identity levers address a result (contract with harvard-visionlab/models):
    model=        backbone `source/arch:weights_id` (weights_id = sha256[:8] of the weights file; human alias)
    readout=      how class scores were read out: head | probe__<layer>__<hash8> | prototypes__… | zeroshot__<hash8>
                  | none (raw backbone, no class scores)
    intervention= declared parameter-free change to the computation: none | topk__k0.4 | lrm__passes1 | …
    run=          one execution. Runs are never overwritten; `force` adds a run and keeps the old one.

Identity contract v2: the *experiment* is identified by `config_id` (models: sha256 of the configuration
block) + `eval_spec_id` (ours: sha256 of dataset revision, preprocessing, output map, scoring, seed,
precision — see spec.py). `store.run()` reuses a stored run only if a *complete* run with the identical
eval_spec_id exists and every spec component is exactly known. A run directory without manifest.json is
incomplete and is never reused. Pre-v2 records (files directly under the intervention= level) are
`legacy` and never reused either.

    store = ResultsStore()
    store.run("pytorch/alexnet:DEFAULT", dataset="pairs-72")     # resolve, eval, save (or return the matching run)
    store.query(dataset="pairs-72")                              # one row per complete run (+ legacy rows flagged)
    store.load("pairs-72", "pytorch/alexnet:7be5be79")           # latest complete run at that address
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import DEFAULT_CONFIG, REPO_ID, resolve_dataset_revision
from .scoring import AnagramResults
from .spec import (
    DatasetRef,
    EvalSpec,
    Execution,
    canonical_json,
    new_run_id,
    scorer_signature,
    sha256_hex,
    transform_signature,
)
from .version import __version__

EVAL_NAME = "anagrams"
DEFAULT_BUCKET = os.environ.get("VISIONLAB_EVALS_BUCKET", "visionlab-evals")
DEFAULT_CACHE = Path(os.environ.get("VISIONLAB_EVALS_CACHE", Path.home() / ".cache" / "visionlab" / "evals"))
IDENTITY_COLS = [
    "eval_name",
    "eval_version",
    "dataset",
    "model_id",
    "model_spec",
    "model_source",
    "model_arch",
    "weights_id",
    "readout_type",
    "readout_layer",
    "readout_id",
    "readout_spec",
    "readout_n_classes",
    "readout_train_data",
    "readout_primary",
    "intervention_kind",
    "intervention_params",
    "intervention_id",
    "intervention_spec",
    "config_id",
    "eval_spec_id",
    "run_id",
]
READOUT_TYPES = ("head", "probe", "prototypes", "zeroshot", "none")  # none = raw backbone (features, no class scores)
NO_INTERVENTION = {
    "intervention_kind": "none",
    "intervention_params": "{}",
    "intervention_id": "none",
    "intervention_spec": "none",
}
DATA_FILES = ("results.parquet", "summary.json")
MANIFEST = "manifest.json"
MANIFEST_SCHEMA = "visionlab-evals/anagrams/run@1"

_SLUG_OK = re.compile(r"^[A-Za-z0-9._-]+$")
_WEIGHTS_ID_OK = re.compile(
    r"^([0-9a-f]{8}|NONE(-[sr][0-9a-z]+)?)$"
)  # sha256[:8] | NONE | NONE-s<seed> | NONE-r<digest8>
_RUN_DIR = re.compile(r"^run=([^/]+)$")


def model_slug(model_id: str) -> str:
    """'pytorch/alexnet:7be5be79' -> 'pytorch__alexnet__7be5be79'. Only [A-Za-z0-9._-] survive."""
    slug = model_id.replace("/", "__").replace(":", "__")
    if not _SLUG_OK.match(slug):
        raise ValueError(f"model_id {model_id!r} -> slug {slug!r} has characters outside [A-Za-z0-9._-]")
    return slug


def _layer_slug(layer: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", layer)


def _param_str(v) -> str:
    return str(v).lower() if isinstance(v, bool) else repr(v) if isinstance(v, float) else str(v)


# =============================================================================================
# identities
# =============================================================================================
@dataclass(frozen=True)
class ReadoutIdentity:
    """How class scores were read out of a backbone (mirrors visionlab.models ReadoutIdentity).
    `readout_id` is sha256[:8] of the head / prototype file; type 'none' = raw backbone."""

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
        if self.readout_type == "none":
            if self.readout_id != "none":
                raise ValueError("a raw-backbone readout ('none') has readout_id 'none'")
        elif not _WEIGHTS_ID_OK.match(self.readout_id):
            raise ValueError(f"readout_id must be sha256[:8], got {self.readout_id!r}")

    @property
    def slug(self) -> str:
        if self.readout_type in ("head", "none"):
            return self.readout_type
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


@dataclass(frozen=True)
class InterventionIdentity:
    """A declared, parameter-free change to the backbone's computation at inference.

    canonical = 'kind:k=v,...' with sorted params ('kind' alone if no params); intervention_id =
    sha256[:8] of canonical (human alias; the authoritative id is the models config_id, which also
    hashes card-fixed kwargs such as layer placement); slug e.g. 'topk__k0.4', 'lrm__passes1_steeringtrue'.
    """

    kind: str
    params: dict = field(default_factory=dict)
    spec: str | None = None  # as typed, e.g. 'topk:k=0.4'

    def __post_init__(self):
        if not _SLUG_OK.match(self.kind) or self.kind == "none":
            raise ValueError(f"intervention kind must match [A-Za-z0-9._-] and not be 'none', got {self.kind!r}")

    @property
    def canonical(self) -> str:
        if not self.params:
            return self.kind
        return self.kind + ":" + ",".join(f"{k}={_param_str(self.params[k])}" for k in sorted(self.params))

    @property
    def intervention_id(self) -> str:
        return hashlib.sha256(self.canonical.encode()).hexdigest()[:8]

    @property
    def slug(self) -> str:
        parts = [self.kind] + [f"{k}{_param_str(self.params[k])}" for k in sorted(self.params)]
        return _layer_slug("__".join([parts[0], "_".join(parts[1:])]) if self.params else parts[0])

    def as_dict(self) -> dict:
        return {
            "intervention_kind": self.kind,
            "intervention_params": json.dumps(self.params, sort_keys=True, default=str),
            "intervention_id": self.intervention_id,
            "intervention_spec": self.spec or self.canonical,
        }


@dataclass(frozen=True)
class ModelIdentity:
    """Who produced a result: backbone (`weights_id` = sha256[:8] of the weights file; 'NONE-s<seed>' /
    'NONE-r<digest>' / 'NONE' for random init), readout (None = native head) and intervention (None = intact).
    `config_id` and `manifest` are carried verbatim from visionlab.models when available."""

    model_source: str
    model_arch: str
    weights_id: str
    model_spec: str | None = None  # the spec as typed, e.g. 'pytorch/alexnet:DEFAULT'
    readout: ReadoutIdentity | None = None
    intervention: InterventionIdentity | None = None
    collection_names: dict = field(default_factory=dict)  # e.g. {"Doshi2025": "resnet50_in1k"} (summary only)
    config_id: str | None = None  # full sha256 from the models manifest; None -> eval fallback (not reusable)
    manifest: dict | None = None  # visionlab.models manifest(), stored verbatim in the run manifest

    def __post_init__(self):
        if not _WEIGHTS_ID_OK.match(self.weights_id):
            raise ValueError(
                f"weights_id must be sha256[:8] (8 lowercase hex chars) or NONE[-s<seed>|-r<digest>], got "
                f"{self.weights_id!r}. Resolve aliases like 'DEFAULT' to the hashid before storing results."
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

    @property
    def intervention_slug(self) -> str:
        return self.intervention.slug if self.intervention is not None else "none"

    @property
    def key(self) -> tuple[str, str, str]:
        """(model_id, readout_slug, intervention_slug) — with a dataset, the store address of a result."""
        return (self.model_id, self.readout_slug, self.intervention_slug)

    @property
    def is_random_init(self) -> bool:
        return self.weights_id.startswith("NONE")

    def as_dict(self) -> dict:
        # native head: readout_id == weights_id (the head lives in the weights file; 'NONE…' for random init)
        readout = self.readout or ReadoutIdentity("head", self.weights_id, readout_spec="head")
        intervention = self.intervention.as_dict() if self.intervention is not None else NO_INTERVENTION
        return {
            "model_id": self.model_id,
            "model_spec": self.model_spec or self.model_id,
            "model_source": self.model_source,
            "model_arch": self.model_arch,
            "weights_id": self.weights_id,
            **readout.as_dict(),
            **intervention,
        }

    def resolved_config_id(self) -> tuple[str, str]:
        """(config_id, source). Fallback = sha256 of our identity columns, flagged so it never yields a cache hit."""
        if self.config_id:
            return self.config_id, "models-manifest"
        return sha256_hex(canonical_json(self.as_dict())), "eval-fallback"

    @classmethod
    def from_visionlab(cls, identity, spec: str | None = None) -> "ModelIdentity":
        """Adapt a `visionlab.models.ModelIdentity` (source, name, hashid, alias/spec, readout, intervention,
        config_id, manifest())."""
        # models: readout None = raw backbone (`@none`, no class scores); type 'head' = native classifier
        ro = getattr(identity, "readout", None)
        if ro is None:
            readout = ReadoutIdentity("none", "none", readout_spec="none", readout_primary=False)
        elif getattr(ro, "type", "head") == "head":
            readout = None  # native head
        else:
            readout = ReadoutIdentity(
                readout_type=ro.type,
                readout_id=ro.hashid,
                readout_layer=getattr(ro, "layer", None),
                readout_spec=getattr(ro, "tag", None),
                readout_n_classes=getattr(ro, "n_classes", None),
                readout_train_data=getattr(ro, "train_data", None),
                readout_primary=bool(getattr(ro, "primary", True)),
            )
        intervention = None
        iv = getattr(identity, "intervention", None)
        if iv is not None:
            intervention = InterventionIdentity(
                kind=iv.kind,
                params=dict(getattr(iv, "params", {}) or {}),
                spec=getattr(iv, "spec", None) or getattr(iv, "canonical", None),
            )
        typed = spec or getattr(identity, "spec", None) or getattr(identity, "alias", None)
        names = getattr(identity, "collection_names", None) or {}
        manifest = getattr(identity, "manifest", None)
        manifest = manifest() if callable(manifest) else manifest
        return cls(
            identity.source,
            identity.name,
            identity.hashid,
            model_spec=typed,
            readout=readout,
            intervention=intervention,
            collection_names=dict(names),
            config_id=getattr(identity, "config_id", None),
            manifest=manifest,
        )

    @classmethod
    def from_spec(cls, spec: str) -> "ModelIdentity":
        """Resolve a visionlab.models spec ('source/name:weights[@readout][+intervention]') without loading."""
        from visionlab.models import resolve  # optional dependency: uv sync --extra models

        return cls.from_visionlab(resolve(spec), spec=spec)

    @classmethod
    def from_torchvision(cls, weights) -> "ModelIdentity":
        """From a torchvision WeightsEnum member, e.g. AlexNet_Weights.IMAGENET1K_V1 -> pytorch/alexnet:7be5be79.
        torchvision names weight files '<arch>-<sha256[:8]>.pth', so the hashid comes from the URL."""
        arch = type(weights).__name__.removesuffix("_Weights").lower()
        stem = Path(weights.url).name.split(".")[0]
        return cls("pytorch", arch, stem.rsplit("-", 1)[-1], model_spec=f"pytorch/{arch}:{weights.name}")


# =============================================================================================
# runs
# =============================================================================================
@dataclass(frozen=True)
class RunInfo:
    dataset: str
    model_id: str
    readout: str
    intervention: str
    run_id: str  # 'legacy' for pre-v2 records
    complete: bool  # manifest.json present
    legacy: bool = False
    eval_spec_id: str | None = None
    config_id: str | None = None
    manifest: dict | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.model_id, self.readout, self.intervention)


class ResultsStore:
    """Save / load / query anagram results as immutable runs in the Hive tree, mirrored locally and on S3."""

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
    def prefix(
        self,
        dataset: str | None = None,
        model_id: str | None = None,
        readout: str = "head",
        intervention: str = "none",
        run_id: str | None = None,
    ) -> str:
        parts = [f"eval={self.eval_name}", f"version={self.eval_version}"]
        if dataset is not None:
            parts.append(f"dataset={dataset}")
        if model_id is not None:
            if dataset is None:
                raise ValueError("model_id requires dataset")
            parts += [f"model={model_slug(model_id)}", f"readout={readout}", f"intervention={intervention}"]
            if run_id is not None:
                parts.append(f"run={run_id}")
        return "/".join(parts)

    def local_dir(
        self, dataset: str, model_id: str, readout: str = "head", intervention: str = "none", run_id: str | None = None
    ) -> Path:
        return self.cache_dir / self.prefix(dataset, model_id, readout, intervention, run_id)

    def s3_uri(
        self,
        dataset: str | None = None,
        model_id: str | None = None,
        readout: str = "head",
        intervention: str = "none",
        run_id: str | None = None,
    ) -> str:
        return f"s3://{self.bucket}/{self.prefix(dataset, model_id, readout, intervention, run_id)}"

    def duckdb_glob(self, dataset: str | None = None) -> str:
        """Glob for `read_parquet(<glob>, hive_partitioning=true)`: partitions eval, version, dataset, model,
        readout, intervention, run. Incomplete runs are rare and short-lived; join `query()` to be strict."""
        root = f"s3://{self.bucket}" if self.bucket else str(self.cache_dir)
        depth = "*/*/*/*/" if dataset is not None else "*/*/*/*/*/"  # [dataset/]model/readout/intervention/run
        return f"{root}/{self.prefix(dataset)}/{depth}results.parquet"

    # ------------------------------------------------------------------------------------- s3
    @property
    def s3(self):
        if self._client is None and self.bucket:
            import boto3

            self._client = boto3.client("s3")
        return self._client

    def _list_keys(self, prefix: str) -> list[str]:
        """Every object key (S3) or file path relative to cache_dir (local) under `prefix/`."""
        prefix = prefix.rstrip("/") + "/"
        if self.bucket:
            keys = []
            for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
                keys += [o["Key"] for o in page.get("Contents", [])]
            return sorted(keys)
        root = self.cache_dir / prefix
        return sorted(str(p.relative_to(self.cache_dir)) for p in root.glob("**/*") if p.is_file())

    def _fetch(self, key: str, refresh: bool = False) -> Path:
        """Local path of an object, downloading from S3 when missing (or when refresh=True)."""
        local = self.cache_dir / key
        if self.bucket and (refresh or not local.exists()):
            local.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(self.bucket, key, str(local))
        return local

    def _put(self, local: Path, key: str) -> None:
        if self.bucket:
            self.s3.upload_file(str(local), self.bucket, key)

    # ------------------------------------------------------------------------------------ runs
    def list_runs(
        self,
        dataset: str,
        model_id: str,
        readout: str = "head",
        intervention: str = "none",
        refresh_manifests: bool = True,
    ) -> list[RunInfo]:
        """All runs at an address, newest first. Complete = manifest.json present. Pre-v2 files directly
        under the address are one 'legacy' run."""
        address = self.prefix(dataset, model_id, readout, intervention)
        by_run: dict[str, set[str]] = {}
        legacy = set()
        for key in self._list_keys(address):
            rel = key[len(address) + 1 :]
            head, _, tail = rel.partition("/")
            m = _RUN_DIR.match(head)
            if m and tail:
                by_run.setdefault(m.group(1), set()).add(tail)
            elif not tail:
                legacy.add(head)
        runs = []
        for run_id, files in sorted(by_run.items(), reverse=True):
            complete = MANIFEST in files
            manifest = None
            if complete:
                manifest = json.loads(
                    self._fetch(f"{address}/run={run_id}/{MANIFEST}", refresh=refresh_manifests).read_text()
                )
            runs.append(
                RunInfo(
                    dataset,
                    model_id,
                    readout,
                    intervention,
                    run_id,
                    complete,
                    eval_spec_id=(manifest or {}).get("eval_spec_id"),
                    config_id=(manifest or {}).get("config_id"),
                    manifest=manifest,
                )
            )
        if legacy & set(DATA_FILES):
            runs.append(RunInfo(dataset, model_id, readout, intervention, "legacy", complete=False, legacy=True))
        return runs

    def find_run(self, dataset: str, key: tuple[str, str, str], eval_spec_id: str) -> RunInfo | None:
        """Newest complete run at `key` whose eval_spec_id matches exactly."""
        for run in self.list_runs(dataset, *key):
            if run.complete and run.eval_spec_id == eval_spec_id:
                return run
        return None

    def exists(
        self,
        dataset: str,
        model_id: str,
        readout: str = "head",
        intervention: str = "none",
        eval_spec_id: str | None = None,
    ) -> bool:
        """A complete run exists at the address (optionally with this exact eval_spec_id)."""
        runs = [
            r for r in self.list_runs(dataset, model_id, readout, intervention, refresh_manifests=False) if r.complete
        ]
        return any(eval_spec_id is None or r.eval_spec_id == eval_spec_id for r in runs)

    def load(
        self, dataset: str, model_id: str, readout: str = "head", intervention: str = "none", run_id: str | None = None
    ) -> AnagramResults:
        """Results of one run: `run_id` explicit, else the newest complete run. 'legacy' loads a pre-v2 record."""
        if run_id is None:
            complete = [
                r
                for r in self.list_runs(dataset, model_id, readout, intervention, refresh_manifests=False)
                if r.complete
            ]
            if not complete:
                raise FileNotFoundError(f"no complete run at {self.prefix(dataset, model_id, readout, intervention)}")
            run_id = complete[0].run_id
        base = self.prefix(dataset, model_id, readout, intervention, None if run_id == "legacy" else run_id)
        for f in DATA_FILES:
            self._fetch(f"{base}/{f}")
        return AnagramResults.load(self.cache_dir / base)

    def predictions(
        self, dataset: str, model_id: str, readout: str = "head", intervention: str = "none", run_id: str | None = None
    ) -> pd.DataFrame:
        return self.load(dataset, model_id, readout, intervention, run_id).predictions

    def query(
        self,
        dataset: str | None = None,
        models: list[str] | None = None,
        readout_type: str | None = None,
        intervention_kind: str | None = None,
        latest: bool = True,
        include_legacy: bool = True,
    ) -> pd.DataFrame:
        """One row per complete run (identity + metrics + provenance + run/spec ids), read from manifests.
        latest=True keeps only the newest run per (address, eval_spec_id). Legacy records are rows with
        legacy=True and no run/spec ids; incomplete runs are never returned."""
        rows = []
        for key in self._list_keys(self.prefix(dataset)):
            if key.endswith(f"/{MANIFEST}"):
                m = json.loads(self._fetch(key, refresh=True).read_text())
                rows.append(
                    {
                        **m.get("summary", {}),
                        "eval_spec_id": m.get("eval_spec_id"),
                        "config_id": m.get("config_id"),
                        "run_id": m.get("run_id"),
                        "reusable": m.get("eval_spec", {}).get("reusable"),
                        "legacy": False,
                        "completed_at": m.get("completed_at"),
                    }
                )
            elif include_legacy and key.endswith("/summary.json") and "/run=" not in key:
                rows.append(
                    {**json.loads(self._fetch(key, refresh=True).read_text()), "legacy": True, "run_id": "legacy"}
                )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        if models is not None:
            df = df[df["model_id"].isin(models)]
        if readout_type is not None:
            df = df[df["readout_type"] == readout_type]
        if intervention_kind is not None:
            df = df[df["intervention_kind"] == intervention_kind]
        if latest and len(df):
            df = df.sort_values("run_id", ascending=False)
            keys = ["dataset", "model_id", "readout_spec", "intervention_id", "eval_spec_id"]
            df = df.drop_duplicates(subset=[k for k in keys if k in df.columns], keep="first")
        return df.sort_values(["dataset", "css"], ascending=[True, False]).reset_index(drop=True)

    # -------------------------------------------------------------------------------- spec
    def make_spec(
        self,
        identity: ModelIdentity,
        dataset: str,
        transform,
        to_anagram_scores=None,
        seed: int | None = None,
        precision: str = "fp32-strict",
        dataset_revision: str | None = "pin",
    ) -> EvalSpec:
        """Build the evaluation spec *before* loading a model, so the cache can be checked first."""
        if dataset_revision == "pin":
            dataset_revision = resolve_dataset_revision()
        config_id, source = identity.resolved_config_id()
        return EvalSpec(
            config_id=config_id,
            config_id_source=source,
            dataset=DatasetRef(REPO_ID, dataset, dataset_revision),
            preprocessing=transform_signature(transform),
            scorer=scorer_signature(to_anagram_scores),
            execution=Execution(seed=seed, precision=precision),
            eval_name=self.eval_name,
            eval_version=self.eval_version,
        )

    # ------------------------------------------------------------------------------- write
    def save(self, results: AnagramResults, identity: ModelIdentity, spec: EvalSpec, dataset: str) -> RunInfo:
        """Write a new immutable run: results.parquet, summary.json, then manifest.json (the completion marker).
        Uploads happen in the same order, so an interrupted save never looks complete."""
        if results.summary.get("dataset") not in (None, dataset):
            raise ValueError(f"results were computed on {results.summary['dataset']!r}, not {dataset!r}")
        if spec.dataset.config != dataset:
            raise ValueError("spec.dataset.config does not match dataset")
        run_id = new_run_id(spec.eval_spec_id)
        ident = {
            "eval_name": self.eval_name,
            "eval_version": self.eval_version,
            "dataset": dataset,
            **identity.as_dict(),
            "config_id": spec.config_id,
            "eval_spec_id": spec.eval_spec_id,
            "run_id": run_id,
        }
        extra = {
            "collection_names": json.dumps(identity.collection_names, sort_keys=True),
            "config_id_source": spec.config_id_source,
            "dataset_revision": spec.dataset.revision,
        }
        summary = {**ident, **extra, **{k: v for k, v in results.summary.items() if k not in ident}}
        predictions = results.predictions.copy()
        for k in IDENTITY_COLS:
            predictions[k] = ident[k]
        predictions = predictions[IDENTITY_COLS + [c for c in predictions.columns if c not in IDENTITY_COLS]]

        base = self.prefix(dataset, *identity.key, run_id)
        local = AnagramResults(summary, predictions, results.pairs, results.confusion).save(self.cache_dir / base)
        for f in DATA_FILES:
            self._put(local / f, f"{base}/{f}")

        manifest = {
            "schema": MANIFEST_SCHEMA,
            "eval_name": self.eval_name,
            "eval_version": self.eval_version,
            "dataset": dataset,
            "run_id": run_id,
            "status": "complete",
            "config_id": spec.config_id,
            "eval_spec_id": spec.eval_spec_id,
            "eval_spec": spec.to_dict(),
            "identity": identity.as_dict(),
            "models_manifest": identity.manifest,
            "artifacts": {
                f: {"sha256": _file_sha256(local / f), "bytes": (local / f).stat().st_size} for f in DATA_FILES
            },
            "summary": summary,
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (local / MANIFEST).write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        self._put(local / MANIFEST, f"{base}/{MANIFEST}")
        return RunInfo(
            dataset,
            *identity.key,
            run_id,
            True,
            eval_spec_id=spec.eval_spec_id,
            config_id=spec.config_id,
            manifest=manifest,
        )

    def run(
        self,
        model_or_spec,
        transform=None,
        dataset: str = DEFAULT_CONFIG,
        identity: ModelIdentity | None = None,
        force: bool = False,
        seed: int | None = None,
        to_anagram_scores=None,
        **eval_kwargs,
    ) -> AnagramResults:
        """Evaluate and store a new run, or return the stored run with the identical evaluation spec.

        Pass a visionlab.models spec string ('source/name:weights[@readout][+intervention]'; model,
        `transforms.test` and identity come from the model card) or a model + transform + `identity`.
        Reuse requires a complete run whose eval_spec_id matches and a spec whose components are all exactly
        known; otherwise a new run is added. `force=True` always adds a run (old runs are kept).
        Random-init baselines are loaded with `seed` (default 0), recorded in the spec.
        """
        import torch

        from .eval import anagram_eval

        if isinstance(model_or_spec, str):
            identity = ModelIdentity.from_spec(model_or_spec)
            if transform is None:
                from visionlab.models import load_transforms

                transform = load_transforms(model_or_spec).test
        elif identity is None or transform is None:
            raise ValueError("pass identity=ModelIdentity(...) and transform when giving a model object")

        if identity.is_random_init and seed is None:
            seed = 0
        tf32 = (
            bool(torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32) and torch.cuda.is_available()
        )
        spec = self.make_spec(
            identity, dataset, transform, to_anagram_scores, seed=seed, precision="tf32" if tf32 else "fp32-strict"
        )

        if not force and spec.reusable:
            hit = self.find_run(dataset, identity.key, spec.eval_spec_id)
            if hit is not None:
                return self.load(dataset, *identity.key, run_id=hit.run_id)

        if isinstance(model_or_spec, str):
            from visionlab.models import load_model

            load_kwargs = {"seed": seed} if seed is not None else {}
            model, _, vl_identity = load_model(model_or_spec, **load_kwargs)
            identity = ModelIdentity.from_visionlab(vl_identity, spec=model_or_spec)
            # Unseeded random init (NONE-r<digest>) only has a config_id once the model is built: rebuild the
            # spec from the post-load identity so the run records the realization, and re-check the cache.
            if identity.resolved_config_id() != (spec.config_id, spec.config_id_source):
                spec = self.make_spec(
                    identity,
                    dataset,
                    transform,
                    to_anagram_scores,
                    seed=seed,
                    precision=spec.execution.precision,
                    dataset_revision=spec.dataset.revision,
                )
                if not force and spec.reusable:
                    hit = self.find_run(dataset, identity.key, spec.eval_spec_id)
                    if hit is not None:
                        return self.load(dataset, *identity.key, run_id=hit.run_id)
        else:
            model = model_or_spec

        if seed is not None:
            eval_kwargs.setdefault("run_seed", seed)
        results = anagram_eval(
            model,
            transform,
            config=dataset,
            to_anagram_scores=to_anagram_scores,
            revision=spec.dataset.revision,
            **eval_kwargs,
        )
        run = self.save(results, identity, spec, dataset)
        return self.load(dataset, *identity.key, run_id=run.run_id)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
