"""Anagrams results store: the core Hive store (visionlab.evals.core.store) bound to this eval.

    store = ResultsStore()
    store.run("pytorch/alexnet:DEFAULT", dataset="pairs-72")     # resolve, eval, save (or return the matching run)
    store.query(dataset="pairs-72")                              # one row per complete run (+ legacy rows flagged)
    store.load("pairs-72", "pytorch/alexnet:7be5be79")           # latest complete run at that address

Identity classes and the generic store live in core; they are re-exported here for existing callers.
"""

from __future__ import annotations

from pathlib import Path

from visionlab.evals.core.identity import (  # noqa: F401 (re-exported)
    NO_INTERVENTION,
    READOUT_TYPES,
    InterventionIdentity,
    ModelIdentity,
    ReadoutIdentity,
    model_slug,
)
from visionlab.evals.core.spec import current_precision
from visionlab.evals.core.store import (  # noqa: F401 (re-exported)
    DATA_FILES,
    DEFAULT_BUCKET,
    DEFAULT_CACHE,
    IDENTITY_COLS,
    MANIFEST,
    RunInfo,
)
from visionlab.evals.core.store import ResultsStore as _CoreStore

from .data import DEFAULT_CONFIG, REPO_ID, resolve_dataset_revision
from .scoring import AnagramResults
from .spec import DatasetRef, EvalSpec, Execution, scorer_signature, transform_signature
from .version import __version__

EVAL_NAME = "anagrams"
MANIFEST_SCHEMA = "visionlab-evals/anagrams/run@1"


class ResultsStore(_CoreStore):
    """Save / load / query anagram results as immutable runs in the Hive tree, mirrored locally and on S3."""

    def __init__(
        self,
        bucket: str | None = DEFAULT_BUCKET,
        cache_dir: str | Path = DEFAULT_CACHE,
        eval_name: str = EVAL_NAME,
        eval_version: str = __version__,
    ):
        super().__init__(eval_name, eval_version, AnagramResults, bucket, cache_dir, primary_metric="css")

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

        Pass a model spec string (any installed source: 'pytorch/alexnet:DEFAULT', a rearing hash, ...; see
        core.sources) or a model + transform + `identity`. Reuse requires a complete run whose eval_spec_id matches
        and a spec whose components are all exactly known; otherwise a new run is added. `force=True` always adds a
        run (old runs are kept). Random-init baselines are loaded with `seed` (default 0), recorded in the spec.
        """
        from .eval import anagram_eval

        if isinstance(model_or_spec, str):
            from visionlab.evals.core.evals import run_eval

            from .plugin import EVAL

            config = {}
            if to_anagram_scores is not None:  # hashed via the spec's scorer signature, applied by EVAL.run
                config["to_anagram_scores"] = eval_kwargs["to_anagram_scores"] = to_anagram_scores
            return run_eval(EVAL, model_or_spec, dataset, config, force=force, seed=seed, store=self, **eval_kwargs)
        if identity is None or transform is None:
            raise ValueError("pass identity=ModelIdentity(...) and transform when giving a model object")
        if identity.is_random_init and seed is None:
            seed = 0

        spec = self.make_spec(identity, dataset, transform, to_anagram_scores, seed=seed, precision=current_precision())
        if not force and spec.reusable and not identity.is_random_init:
            hit = self.find_run(dataset, identity.key, spec.eval_spec_id)
            if hit is not None:
                return self.load(dataset, *identity.key, run_id=hit.run_id)
        if seed is not None:
            eval_kwargs.setdefault("run_seed", seed)
        results = anagram_eval(
            model_or_spec,
            transform,
            config=dataset,
            to_anagram_scores=to_anagram_scores,
            revision=spec.dataset.revision,
            **eval_kwargs,
        )
        run = self.save(results, identity, spec, dataset)
        return self.load(dataset, *identity.key, run_id=run.run_id)
