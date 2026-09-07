"""Store tests run local-only (bucket=None) in a temp dir; no network, no credentials."""

import numpy as np
import pandas as pd
import pytest
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


def _results():
    rows = []
    for pid, (o0, o1) in enumerate([("bear", "bunny"), ("cat", "frog")]):
        for pos, label in enumerate([o0, o1]):
            rows.append(dict(filename=f"{pid:03d}_transform_{o0}_{o1}_object{pos}_{label}.png", anagram_id=f"{pid:03d}",
                             pair_id=pid, variant=0, position=pos, label=label, foil=[o0, o1][1 - pos],
                             object0=o0, object1=o1))
    scores = np.zeros((4, 9))
    for i, lab in enumerate(["bear", "bear", "cat", "frog"]):
        scores[i, CLASSES.index(lab)] = 1.0
    return score_predictions(build_predictions(pd.DataFrame(rows), scores), dataset="pairs-72", model_name="toy")


def test_model_slug():
    assert model_slug("pytorch/alexnet:7be5be79") == "pytorch__alexnet__7be5be79"
    with pytest.raises(ValueError):
        model_slug("pytorch/alex net:7be5be79")


def test_identity_from_torchvision_and_validation():
    assert ModelIdentity.from_torchvision(AlexNet_Weights.IMAGENET1K_V1).model_id == "pytorch/alexnet:7be5be79"
    assert ModelIdentity.from_torchvision(ResNet50_Weights.IMAGENET1K_V1).model_id == "pytorch/resnet50:0676ba61"
    vit = ModelIdentity.from_torchvision(ViT_B_16_Weights.IMAGENET1K_V1)
    assert vit.model_id == "pytorch/vit_b_16:c867db91" and vit.model_spec == "pytorch/vit_b_16:IMAGENET1K_V1"
    assert ModelIdentity("pytorch", "alexnet", "NONE").model_id == "pytorch/alexnet:NONE"
    with pytest.raises(ValueError, match="DEFAULT"):
        ModelIdentity("pytorch", "alexnet", "DEFAULT")


def test_store_roundtrip_local(tmp_path):
    store = ResultsStore(bucket=None, cache_dir=tmp_path, eval_version="0.0.test")
    ident = ModelIdentity("pytorch", "alexnet", "7be5be79", model_spec="pytorch/alexnet:DEFAULT")
    assert not store.exists("pairs-72", ident.model_id)

    uri = store.save(_results(), ident, "pairs-72")
    assert uri.endswith("dataset=pairs-72/model=pytorch__alexnet__7be5be79/readout=head/intervention=none")
    assert store.exists("pairs-72", ident.model_id)
    assert (tmp_path / store.prefix("pairs-72", ident.model_id) / "results.parquet").exists()

    back = store.load("pairs-72", ident.model_id)
    assert back.summary["css"] == 0.5 and back.summary["model_id"] == ident.model_id
    assert back.summary["model_spec"] == "pytorch/alexnet:DEFAULT" and back.summary["eval_name"] == "anagrams"
    assert list(back.predictions.columns[:8]) == ["eval_name", "eval_version", "dataset", "model_id", "model_spec",
                                                  "model_source", "model_arch", "weights_id"]
    assert back.summary["readout_type"] == "head" and back.summary["readout_id"] == "7be5be79"
    assert back.summary["intervention_kind"] == "none" and back.summary["intervention_id"] == "none"

    with pytest.raises(FileExistsError):
        store.save(_results(), ident, "pairs-72")
    store.save(_results(), ident, "pairs-72", force=True)

    with pytest.raises(ValueError, match="computed on"):
        store.save(_results(), ident, "pairs-1440")

    q = store.query()
    assert len(q) == 1 and q.loc[0, "model_id"] == ident.model_id and q.loc[0, "dataset"] == "pairs-72"
    assert store.query(dataset="pairs-1440").empty
    assert store.duckdb_glob("pairs-72").endswith("dataset=pairs-72/*/*/*/results.parquet")
    assert store.duckdb_glob().endswith("version=0.0.test/*/*/*/*/results.parquet")


def test_readouts_partition_separately(tmp_path):
    store = ResultsStore(bucket=None, cache_dir=tmp_path, eval_version="0.0.test")
    probe = ReadoutIdentity("probe", "a1b2c3d4", readout_layer="features.10", readout_n_classes=1000,
                            readout_train_data="imagenet1k", readout_primary=True)
    proto = ReadoutIdentity("prototypes", "9f8e7d6c", readout_layer="blocks.11.norm", readout_primary=False)
    zs = ReadoutIdentity("zeroshot", "5c4b3a29")
    assert probe.slug == "probe__features.10__a1b2c3d4"
    assert proto.slug == "prototypes__blocks.11.norm__9f8e7d6c" and zs.slug == "zeroshot__5c4b3a29"
    with pytest.raises(ValueError):
        ReadoutIdentity("knn", "a1b2c3d4")

    backbone = dict(model_source="visionlab", model_arch="alexnet_ipcl", weights_id="3f9a1c2d")
    for ro in (probe, proto):
        ident = ModelIdentity(**backbone, readout=ro)
        assert ident.model_id == "visionlab/alexnet_ipcl:3f9a1c2d"
        store.save(_results(), ident, "pairs-72")
    assert store.exists("pairs-72", ident.model_id, probe.slug) and store.exists("pairs-72", ident.model_id, proto.slug)
    assert not store.exists("pairs-72", ident.model_id)  # no native head stored

    q = store.query(dataset="pairs-72")
    assert len(q) == 2 and set(q["readout_type"]) == {"probe", "prototypes"}
    assert set(q["model_id"]) == {"visionlab/alexnet_ipcl:3f9a1c2d"}
    assert list(store.query(readout_type="probe")["readout_layer"]) == ["features.10"]
    back = store.load("pairs-72", ident.model_id, probe.slug)
    assert back.summary["readout_primary"] is True and back.summary["readout_train_data"] == "imagenet1k"


class _FakeReadout:
    type, hashid, layer, tag = "probe", "a1b2c3d4", "features.10", "fc6_probe"
    n_classes, train_data, primary = 1000, "imagenet1k", True


class _FakeVisionlabIdentity:
    source, name, hashid, alias, spec = "visionlab", "alexnet_ipcl", "3f9a1c2d", "visionlab/alexnet_ipcl:DEFAULT", None
    readout = _FakeReadout()


def test_identity_from_visionlab_adapter():
    ident = ModelIdentity.from_visionlab(_FakeVisionlabIdentity())
    assert ident.model_id == "visionlab/alexnet_ipcl:3f9a1c2d" and ident.model_spec == "visionlab/alexnet_ipcl:DEFAULT"
    assert ident.readout_slug == "probe__features.10__a1b2c3d4"
    d = ident.as_dict()
    assert d["readout_spec"] == "fc6_probe" and d["readout_n_classes"] == 1000 and d["readout_primary"] is True


def test_interventions_partition_separately(tmp_path):
    topk = InterventionIdentity("topk", {"k": 0.4}, spec="topk:k=0.4")
    assert topk.canonical == "topk:k=0.4" and topk.slug == "topk__k0.4"
    assert topk.intervention_id == InterventionIdentity("topk", {"k": 0.4}).intervention_id  # deterministic
    lrm = InterventionIdentity("lrm", {"steering": True, "passes": 1})
    assert lrm.canonical == "lrm:passes=1,steering=true" and lrm.slug == "lrm__passes1_steeringtrue"
    assert InterventionIdentity("ablate").canonical == "ablate" and InterventionIdentity("ablate").slug == "ablate"
    with pytest.raises(ValueError):
        InterventionIdentity("none")

    store = ResultsStore(bucket=None, cache_dir=tmp_path, eval_version="0.0.test")
    backbone = dict(model_source="pytorch", model_arch="alexnet", weights_id="7be5be79")
    intact = ModelIdentity(**backbone)
    sparse = ModelIdentity(**backbone, intervention=topk)
    assert intact.key == ("pytorch/alexnet:7be5be79", "head", "none")
    assert sparse.key == ("pytorch/alexnet:7be5be79", "head", "topk__k0.4")
    store.save(_results(), intact, "pairs-72")
    store.save(_results(), sparse, "pairs-72")
    assert store.exists("pairs-72", intact.model_id) and store.exists("pairs-72", intact.model_id, "head", "topk__k0.4")

    q = store.query(dataset="pairs-72")
    assert len(q) == 2 and set(q["intervention_kind"]) == {"none", "topk"}
    row = store.query(intervention_kind="topk").iloc[0]
    assert row["intervention_params"] == '{"k": 0.4}' and row["intervention_spec"] == "topk:k=0.4"
    assert row["intervention_id"] == topk.intervention_id and row["model_id"] == intact.model_id
    back = store.load("pairs-72", intact.model_id, "head", topk.slug)
    assert back.predictions["intervention_kind"].eq("topk").all()


class _FakeIntervention:
    kind, params, spec = "topk", {"k": 0.6}, "topk:k=0.6"


class _FakeHeadReadout:
    type, hashid, layer, tag, n_classes, train_data, primary = "head", "3f9a1c2d", None, "head", 1000, None, True


class _FakeVisionlabIdentityWithIntervention(_FakeVisionlabIdentity):
    readout = _FakeHeadReadout()  # native classifier head (models: type 'head'); None would mean raw backbone
    intervention = _FakeIntervention()


def test_identity_from_visionlab_with_intervention():
    ident = ModelIdentity.from_visionlab(_FakeVisionlabIdentityWithIntervention())
    assert ident.readout_slug == "head" and ident.intervention_slug == "topk__k0.6"
    assert ident.as_dict()["intervention_spec"] == "topk:k=0.6"


class _FakeVisionlabRawBackbone(_FakeVisionlabIdentity):
    readout = None  # models returns None for '@none' (raw backbone)


def test_adapter_maps_raw_backbone_to_none_not_head():
    ident = ModelIdentity.from_visionlab(_FakeVisionlabRawBackbone())
    assert ident.readout_slug == "none" and ident.key[1] == "none"
    assert ident.as_dict()["readout_type"] == "none" and ident.as_dict()["readout_id"] == "none"
    assert ReadoutIdentity("none", "none").slug == "none"
    with pytest.raises(ValueError):
        ReadoutIdentity("none", "a1b2c3d4")
