"""Store tests run local-only (bucket=None) in a temp dir; no network, no credentials."""

import json

import numpy as np
import pandas as pd
import pytest
from torchvision import transforms as T
from torchvision.models import AlexNet_Weights, ResNet50_Weights, ViT_B_16_Weights
from visionlab.evals.anagrams import (
    CLASSES,
    InterventionIdentity,
    ModelIdentity,
    ReadoutIdentity,
    ResultsStore,
    build_predictions,
    model_slug,
    score_predictions,
)
from visionlab.evals.anagrams.store import DATA_FILES, MANIFEST

TFM = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
TFM_BICUBIC = T.Compose(
    [
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.5,) * 3, (0.5,) * 3),
    ]
)
REV = "519498ad5c1c43dd2ea0e628ad24f6bdf03cecdc"


def _results(dataset="pairs-72"):
    rows = []
    for pid, (o0, o1) in enumerate([("bear", "bunny"), ("cat", "frog")]):
        for pos, label in enumerate([o0, o1]):
            rows.append(
                dict(
                    filename=f"{pid:03d}_transform_{o0}_{o1}_object{pos}_{label}.png",
                    anagram_id=f"{pid:03d}",
                    pair_id=pid,
                    variant=0,
                    position=pos,
                    label=label,
                    foil=[o0, o1][1 - pos],
                    object0=o0,
                    object1=o1,
                )
            )
    scores = np.zeros((4, 9))
    for i, lab in enumerate(["bear", "bear", "cat", "frog"]):
        scores[i, CLASSES.index(lab)] = 1.0
    return score_predictions(build_predictions(pd.DataFrame(rows), scores), dataset=dataset, model_name="toy")


def _store(tmp_path):
    return ResultsStore(bucket=None, cache_dir=tmp_path, eval_version="0.0.test")


def _ident(**kw):
    base = dict(
        model_source="pytorch",
        model_arch="alexnet",
        weights_id="7be5be79",
        model_spec="pytorch/alexnet:DEFAULT",
        config_id="c" * 64,
    )
    base.update(kw)
    return ModelIdentity(**base)


def _spec(store, ident, transform=TFM, dataset="pairs-72", **kw):
    return store.make_spec(ident, dataset, transform, dataset_revision=kw.pop("dataset_revision", REV), **kw)


# ------------------------------------------------------------------------------ identities
def test_model_slug():
    assert model_slug("pytorch/alexnet:7be5be79") == "pytorch__alexnet__7be5be79"
    with pytest.raises(ValueError):
        model_slug("pytorch/alex net:7be5be79")


def test_identity_from_torchvision_and_validation():
    assert ModelIdentity.from_torchvision(AlexNet_Weights.IMAGENET1K_V1).model_id == "pytorch/alexnet:7be5be79"
    assert ModelIdentity.from_torchvision(ResNet50_Weights.IMAGENET1K_V1).model_id == "pytorch/resnet50:0676ba61"
    vit = ModelIdentity.from_torchvision(ViT_B_16_Weights.IMAGENET1K_V1)
    assert vit.model_id == "pytorch/vit_b_16:c867db91" and vit.model_spec == "pytorch/vit_b_16:IMAGENET1K_V1"
    for w in ("NONE", "NONE-s0", "NONE-rdeadbeef"):
        ident = ModelIdentity("pytorch", "alexnet", w)
        assert ident.is_random_init and ident.as_dict()["readout_id"] == w
    with pytest.raises(ValueError, match="DEFAULT"):
        ModelIdentity("pytorch", "alexnet", "DEFAULT")


def test_readout_and_intervention_slugs():
    probe = ReadoutIdentity("probe", "a1b2c3d4", readout_layer="features.10")
    assert probe.slug == "probe__features.10__a1b2c3d4"
    assert ReadoutIdentity("zeroshot", "5c4b3a29").slug == "zeroshot__5c4b3a29"
    assert ReadoutIdentity("none", "none").slug == "none"
    with pytest.raises(ValueError):
        ReadoutIdentity("none", "a1b2c3d4")
    with pytest.raises(ValueError):
        ReadoutIdentity("knn", "a1b2c3d4")
    topk = InterventionIdentity("topk", {"k": 0.4}, spec="topk:k=0.4")
    assert topk.canonical == "topk:k=0.4" and topk.slug == "topk__k0.4"
    assert InterventionIdentity("lrm", {"steering": True, "passes": 1}).slug == "lrm__passes1_steeringtrue"
    with pytest.raises(ValueError):
        InterventionIdentity("none")
    assert _ident(readout=probe, intervention=topk).key == (
        "pytorch/alexnet:7be5be79",
        "probe__features.10__a1b2c3d4",
        "topk__k0.4",
    )


# ------------------------------------------------------------------------------ eval spec
def test_eval_spec_id_changes_with_every_component(tmp_path):
    store, ident = _store(tmp_path), _ident()
    base = _spec(store, ident)
    assert base.reusable
    variants = {
        "config": _spec(store, _ident(config_id="d" * 64)),
        "preprocessing": _spec(store, ident, transform=TFM_BICUBIC),
        "dataset": _spec(store, ident, dataset="pairs-1440"),
        "revision": _spec(store, ident, dataset_revision="0" * 40),
        "seed": _spec(store, ident, seed=1),
        "precision": _spec(store, ident, precision="tf32"),
        "scorer": _spec(store, ident, to_anagram_scores=lambda x: x),
    }
    ids = {name: s.eval_spec_id for name, s in variants.items()}
    assert len(set(ids.values()) | {base.eval_spec_id}) == len(ids) + 1, "every component must change the id"
    assert _spec(store, ident).eval_spec_id == base.eval_spec_id, "deterministic"
    # not exactly known -> never reusable
    assert not variants["scorer"].reusable
    assert not _spec(store, ident, dataset_revision=None).reusable
    assert not _spec(store, _ident(config_id=None)).reusable  # eval-fallback config id
    assert _spec(store, _ident(config_id=None)).config_id_source == "eval-fallback"


# ------------------------------------------------------------------------------ runs
def test_run_roundtrip_immutable_runs_and_force(tmp_path):
    store, ident = _store(tmp_path), _ident()
    spec = _spec(store, ident)
    assert not store.exists("pairs-72", ident.model_id)

    run = store.save(_results(), ident, spec, "pairs-72")
    assert run.complete and run.eval_spec_id == spec.eval_spec_id
    run_dir = tmp_path / store.prefix("pairs-72", *ident.key, run.run_id)
    assert {p.name for p in run_dir.iterdir()} == set(DATA_FILES) | {MANIFEST}
    assert str(run_dir).endswith(f"model=pytorch__alexnet__7be5be79/readout=head/intervention=none/run={run.run_id}")

    assert store.exists("pairs-72", ident.model_id) and store.exists(
        "pairs-72", ident.model_id, eval_spec_id=spec.eval_spec_id
    )
    assert store.find_run("pairs-72", ident.key, spec.eval_spec_id).run_id == run.run_id
    assert store.find_run("pairs-72", ident.key, "0" * 64) is None

    back = store.load("pairs-72", ident.model_id)
    assert back.summary["css"] == 0.5 and back.summary["config_id"] == "c" * 64
    assert back.summary["eval_spec_id"] == spec.eval_spec_id and back.summary["run_id"] == run.run_id
    assert list(back.predictions.columns[:3]) == ["eval_name", "eval_version", "dataset"]
    assert {"config_id", "eval_spec_id", "run_id"} <= set(back.predictions.columns)

    manifest = json.loads((run_dir / MANIFEST).read_text())
    assert manifest["status"] == "complete" and manifest["eval_spec"]["eval_spec_id"] == spec.eval_spec_id
    assert manifest["identity"]["model_id"] == ident.model_id and set(manifest["artifacts"]) == set(DATA_FILES)
    assert len(manifest["artifacts"]["results.parquet"]["sha256"]) == 64

    # a second save never overwrites: two runs, both kept, newest first
    run2 = store.save(_results(), ident, spec, "pairs-72")
    assert run2.run_id != run.run_id
    runs = store.list_runs("pairs-72", *ident.key)
    assert [r.run_id for r in runs] == sorted([run.run_id, run2.run_id], reverse=True) and all(r.complete for r in runs)
    assert store.load("pairs-72", ident.model_id, run_id=run.run_id).summary["run_id"] == run.run_id

    with pytest.raises(ValueError, match="computed on"):
        store.save(_results(), ident, _spec(store, ident, dataset="pairs-1440"), "pairs-1440")


def test_interrupted_write_is_never_complete(tmp_path):
    store, ident = _store(tmp_path), _ident()
    spec = _spec(store, ident)
    run = store.save(_results(), ident, spec, "pairs-72")
    (
        tmp_path / store.prefix("pairs-72", *ident.key, run.run_id) / MANIFEST
    ).unlink()  # simulate a crash before the marker
    runs = store.list_runs("pairs-72", *ident.key)
    assert len(runs) == 1 and not runs[0].complete
    assert not store.exists("pairs-72", ident.model_id)
    assert store.find_run("pairs-72", ident.key, spec.eval_spec_id) is None
    with pytest.raises(FileNotFoundError):
        store.load("pairs-72", ident.model_id)
    assert store.query().empty


def test_legacy_records_are_flagged_and_not_reused(tmp_path):
    store, ident = _store(tmp_path), _ident()
    legacy_dir = tmp_path / store.prefix("pairs-72", *ident.key)
    _results().save(legacy_dir)  # pre-v2 layout: files directly under intervention=
    runs = store.list_runs("pairs-72", *ident.key)
    assert len(runs) == 1 and runs[0].legacy and not runs[0].complete
    assert not store.exists("pairs-72", ident.model_id)
    q = store.query()
    assert len(q) == 1 and bool(q.loc[0, "legacy"]) and q.loc[0, "run_id"] == "legacy"
    assert store.load("pairs-72", ident.model_id, run_id="legacy").summary["css"] == 0.5


def test_query_across_readouts_interventions_and_latest(tmp_path):
    store = _store(tmp_path)
    probe = ReadoutIdentity("probe", "a1b2c3d4", readout_layer="features.10", readout_primary=True)
    ids = [
        _ident(),
        _ident(readout=probe),
        _ident(intervention=InterventionIdentity("topk", {"k": 0.4}, spec="topk:k=0.4")),
    ]
    for ident in ids:
        store.save(_results(), ident, _spec(store, ident), "pairs-72")
    store.save(_results(), ids[0], _spec(store, ids[0]), "pairs-72")  # repeat of the first experiment

    q = store.query(dataset="pairs-72")
    assert (
        len(q) == 3 and set(q["readout_type"]) == {"head", "probe"} and set(q["intervention_kind"]) == {"none", "topk"}
    )
    assert len(store.query(dataset="pairs-72", latest=False)) == 4
    assert list(store.query(readout_type="probe")["readout_layer"]) == ["features.10"]
    assert store.query(intervention_kind="topk").iloc[0]["intervention_spec"] == "topk:k=0.4"
    assert store.duckdb_glob("pairs-72").endswith("dataset=pairs-72/*/*/*/*/results.parquet")


# ------------------------------------------------------------------------------ models adapter
class _FakeReadout:
    type, hashid, layer, tag = "probe", "a1b2c3d4", "features.10", "fc6_probe"
    n_classes, train_data, primary = 1000, "imagenet1k", True


class _FakeHeadReadout:
    type, hashid, layer, tag, n_classes, train_data, primary = "head", "3f9a1c2d", None, "head", 1000, None, True


class _FakeIntervention:
    kind, params, spec = "topk", {"k": 0.6}, "topk:k=0.6"


class _FakeVisionlabIdentity:
    source, name, hashid, alias, spec = "visionlab", "alexnet_ipcl", "3f9a1c2d", "visionlab/alexnet_ipcl:DEFAULT", None
    readout = _FakeReadout()
    intervention = None
    collection_names = {"Doshi2025": "alexnet_ipcl_probe"}
    config_id = "e" * 64

    def manifest(self):
        return {"schema": "visionlab-models/config@1", "configuration": {"arch": {"builder": "x"}}, "provenance": {}}


def test_identity_from_visionlab_adapter(tmp_path):
    ident = ModelIdentity.from_visionlab(_FakeVisionlabIdentity())
    assert ident.model_id == "visionlab/alexnet_ipcl:3f9a1c2d" and ident.model_spec == "visionlab/alexnet_ipcl:DEFAULT"
    assert ident.readout_slug == "probe__features.10__a1b2c3d4" and ident.config_id == "e" * 64
    assert ident.manifest["schema"] == "visionlab-models/config@1"
    d = ident.as_dict()
    assert d["readout_spec"] == "fc6_probe" and d["readout_n_classes"] == 1000 and d["readout_primary"] is True
    store = _store(tmp_path)
    run = store.save(_results(), ident, _spec(store, ident), "pairs-72")
    assert run.manifest["models_manifest"]["configuration"] == {"arch": {"builder": "x"}}
    assert run.manifest["config_id"] == "e" * 64


def test_adapter_maps_raw_backbone_to_none_and_head_to_head():
    class Raw(_FakeVisionlabIdentity):
        readout = None  # models returns None for '@none' (raw backbone)

    class Head(_FakeVisionlabIdentity):
        readout = _FakeHeadReadout()
        intervention = _FakeIntervention()

    raw = ModelIdentity.from_visionlab(Raw())
    assert (
        raw.readout_slug == "none" and raw.as_dict()["readout_type"] == "none" and raw.as_dict()["readout_id"] == "none"
    )
    head = ModelIdentity.from_visionlab(Head())
    assert head.readout_slug == "head" and head.intervention_slug == "topk__k0.6"
    assert head.as_dict()["intervention_spec"] == "topk:k=0.6"
