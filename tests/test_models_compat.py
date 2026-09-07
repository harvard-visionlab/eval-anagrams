"""Cross-repo compatibility: run visionlab.models' golden identity fixtures through OUR adapter (B5).

Skipped when visionlab.models (>= identity contract v2) is not installed. Each fixture case carries the
spec, expected str(identity), model_id, result_key, config_id, the hashed configuration and its canonical
JSON; we assert the adapter and store preserve them exactly.
"""

import hashlib

import pytest

vm = pytest.importorskip("visionlab.models")
if not hasattr(vm, "load_fixtures"):
    pytest.skip("visionlab.models without identity fixtures (pre-v2)", allow_module_level=True)

from visionlab.evals.anagrams import ModelIdentity, ResultsStore  # noqa: E402
from visionlab.evals.anagrams.spec import canonical_json as our_canonical_json  # noqa: E402

from tests.test_store import TFM, _results  # noqa: E402

FIXTURES = vm.load_fixtures()
CASES = {c["spec"]: c for c in (FIXTURES["cases"] if isinstance(FIXTURES, dict) else FIXTURES)}
RESOLVABLE = [c for c in CASES.values() if c.get("resolvable", True)]


@pytest.mark.parametrize("case", RESOLVABLE, ids=[c["spec"] for c in RESOLVABLE])
def test_adapter_preserves_models_identity(case, tmp_path):
    ident = vm.resolve(case["spec"])
    assert str(ident) == case["str"]
    assert vm.resolve(str(ident)) == ident, "str(identity) must round-trip through resolve"

    ours = ModelIdentity.from_visionlab(ident, spec=case["spec"])
    assert ours.model_id == case["model_id"]
    assert list(ours.key) == list(case["result_key"])
    assert ours.config_id == case["config_id"]
    # config_id is reproducible from the configuration block with the shared canonicalization rule
    assert our_canonical_json(case["configuration"]) == case["canonical_json"]
    assert hashlib.sha256(case["canonical_json"].encode()).hexdigest() == case["config_id"]
    assert ours.manifest["configuration"] == case["configuration"]

    # the store carries config_id + manifest verbatim and the run is reusable (models-provided id)
    store = ResultsStore(bucket=None, cache_dir=tmp_path, eval_version="0.0.compat")
    spec = store.make_spec(ours, "pairs-72", TFM, dataset_revision="0" * 40)
    assert spec.config_id == case["config_id"] and spec.config_id_source == "models-manifest" and spec.reusable
    run = store.save(_results(), ours, spec, "pairs-72")
    assert run.manifest["config_id"] == case["config_id"]
    assert run.manifest["models_manifest"]["configuration"] == case["configuration"]
    assert run.manifest["summary"]["model_id"] == case["model_id"]
    # our row columns agree with models' row where they overlap
    row = case["row"]
    for ours_col, theirs_col in [("model_id", "model_id"), ("config_id", "config_id")]:
        if theirs_col in row:
            assert run.manifest["summary"][ours_col] == row[theirs_col]


def test_unseeded_random_init_is_preserved_not_reproduced():
    unseeded = [c for c in CASES.values() if not c.get("resolvable", True)]
    if not unseeded:
        pytest.skip("no unresolvable fixture case")
    case = unseeded[0]
    with pytest.raises(ValueError):
        vm.resolve(case["str"])  # a realization cannot be re-created from its spec
    # before realization there is no config_id: our fallback id is flagged and never reusable
    pre = vm.resolve(case["spec"])
    ours = ModelIdentity.from_visionlab(pre, spec=case["spec"])
    assert ours.config_id is None
    _, source = ours.resolved_config_id()
    assert source == "eval-fallback"


def test_unvalidated_dependency_runs_are_not_reusable():
    class FakeUnvalidated:
        source, name, hashid, alias, spec = "pytorch", "alexnet", "7be5be79", None, "pytorch/alexnet:7be5be79"
        readout = intervention = None
        readout = type(
            "R",
            (),
            {
                "type": "head",
                "hashid": "7be5be79",
                "tag": "head",
                "layer": None,
                "n_classes": 1000,
                "train_data": None,
                "primary": True,
            },
        )()
        collection_names = {}
        config_id = "f" * 64

        def manifest(self):
            return {"schema": "visionlab-models/config@1", "configuration": {}, "provenance": {"impl_validated": False}}

    ours = ModelIdentity.from_visionlab(FakeUnvalidated())
    assert ours.resolved_config_id() == ("f" * 64, "models-manifest-unvalidated")
    store = ResultsStore(bucket=None, cache_dir="/tmp/unused", eval_version="0.0.compat")
    assert not store.make_spec(ours, "pairs-72", TFM, dataset_revision="0" * 40).reusable
