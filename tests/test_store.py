"""Store tests run local-only (bucket=None) in a temp dir; no network, no credentials."""

import numpy as np
import pandas as pd
import pytest
from torchvision.models import AlexNet_Weights, ResNet50_Weights, ViT_B_16_Weights
from visionlab.evals.anagrams import (
    CLASSES,
    ModelIdentity,
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
    assert uri.endswith("eval=anagrams/version=0.0.test/dataset=pairs-72/model=pytorch__alexnet__7be5be79")
    assert store.exists("pairs-72", ident.model_id)
    assert (tmp_path / store.prefix("pairs-72", ident.model_id) / "results.parquet").exists()

    back = store.load("pairs-72", ident.model_id)
    assert back.summary["css"] == 0.5 and back.summary["model_id"] == ident.model_id
    assert back.summary["model_spec"] == "pytorch/alexnet:DEFAULT" and back.summary["eval_name"] == "anagrams"
    assert list(back.predictions.columns[:8]) == ["eval_name", "eval_version", "dataset", "model_id", "model_spec",
                                                  "model_source", "model_arch", "weights_id"]

    with pytest.raises(FileExistsError):
        store.save(_results(), ident, "pairs-72")
    store.save(_results(), ident, "pairs-72", force=True)

    with pytest.raises(ValueError, match="computed on"):
        store.save(_results(), ident, "pairs-1440")

    q = store.query()
    assert len(q) == 1 and q.loc[0, "model_id"] == ident.model_id and q.loc[0, "dataset"] == "pairs-72"
    assert store.query(dataset="pairs-1440").empty
    assert store.duckdb_glob("pairs-72").endswith("dataset=pairs-72/*/results.parquet")
