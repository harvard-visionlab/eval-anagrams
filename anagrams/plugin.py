"""The anagrams eval as a visionlab.evals plugin (`model eval anagrams <spec>`; needs the lab extra).

Registered in pyproject.toml under the `visionlab.evals.evals` entry-point group. The model comes from any installed
source (visionlab.models, model-rearing, ...); its readout must give ImageNet-1k or 9-way anagram scores, unless
config["to_anagram_scores"] (Python API only) maps its outputs.
"""

from __future__ import annotations

from visionlab.evals.core.spec import current_precision

from .data import CONFIGS, DEFAULT_CONFIG
from .eval import anagram_eval
from .store import EVAL_NAME, ResultsStore
from .version import __version__

CONFIG_KEYS = {"to_anagram_scores"}


class AnagramsEval:
    name = EVAL_NAME
    version = __version__
    datasets = CONFIGS
    default_dataset = DEFAULT_CONFIG
    needs = "scores"

    def store(self, **kwargs) -> ResultsStore:
        return ResultsStore(**kwargs)

    def spec(self, identity, transforms, dataset, config, *, seed=None, previous=None):
        unknown = set(config) - CONFIG_KEYS
        if unknown:
            raise ValueError(f"anagrams: unknown config keys {sorted(unknown)}; allowed: {sorted(CONFIG_KEYS)}")
        return ResultsStore().make_spec(
            identity,
            dataset,
            transforms.test,
            config.get("to_anagram_scores"),
            seed=seed,
            precision=previous.execution.precision if previous else current_precision(),
            dataset_revision=previous.dataset.revision if previous else "pin",
        )

    def run(self, loaded, spec, **kwargs):
        if spec.execution.seed is not None:
            kwargs.setdefault("run_seed", spec.execution.seed)
        if loaded.identity.init_digest:
            kwargs.setdefault("init_digest", loaded.identity.init_digest)
        return anagram_eval(
            loaded.module,
            loaded.transforms.test,
            config=spec.dataset.config,
            to_anagram_scores=kwargs.pop("to_anagram_scores", None),
            revision=spec.dataset.revision,
            **kwargs,
        )


EVAL = AnagramsEval()
