import math

import numpy as np
import pandas as pd
import pytest
from visionlab.evals.anagrams import (
    CHANCE_CSS,
    CLASSES,
    AnagramResults,
    bootstrap_ci,
    build_pairs,
    build_predictions,
    legacy_metrics,
    score_predictions,
)


def _meta():
    # two anagram pairs: (bear, bunny) and (cat, frog)
    rows = []
    for pid, (o0, o1) in enumerate([("bear", "bunny"), ("cat", "frog")]):
        for pos, label in enumerate([o0, o1]):
            rows.append(dict(filename=f"{pid:03d}_transform_{o0}_{o1}_object{pos}_{label}.png", anagram_id=f"{pid:03d}",
                             pair_id=pid, variant=0, position=pos, label=label, foil=[o0, o1][1 - pos],
                             object0=o0, object1=o1))
    return pd.DataFrame(rows)


def _scores():
    s = np.zeros((4, 9))
    c = CLASSES.index
    s[0, [c("bear"), c("bunny")]] = [3.0, 1.0]  # correct, dm = (3 - 1) / sqrt2
    s[1, [c("bunny"), c("bear")]] = [2.0, 2.5]  # wrong: picks the foil
    s[2, [c("cat"), c("wolf"), c("frog")]] = [4.0, 1.0, -1.0]  # correct
    s[3, [c("frog"), c("cat")]] = [1.0, 0.5]  # correct
    return s


def test_predictions_margins_and_pairs():
    pred = build_predictions(_meta(), _scores())
    assert list(pred["pred"]) == ["bear", "bear", "cat", "frog"]
    assert list(pred["correct"]) == [True, False, True, True]
    assert pred.loc[0, "decision_margin"] == pytest.approx((3 - 1) / math.sqrt(2))
    assert pred.loc[1, "decision_margin"] == pytest.approx((2 - 2.5) / math.sqrt(2))  # negative when wrong
    assert pred.loc[2, "decision_margin"] == pytest.approx((4 - 1) / math.sqrt(2))
    assert pred.loc[2, "foil_margin"] == pytest.approx((4 - (-1)) / math.sqrt(2))
    assert (pred["decision_margin"] > 0).equals(pred["correct"])

    pairs = build_pairs(pred)
    assert list(pairs["both_correct"]) == [False, True]
    assert pairs.loc[1, "pair_margin"] == pytest.approx(min(3 / math.sqrt(2), 0.5 / math.sqrt(2)))
    assert (pairs["pair_margin"] > 0).equals(pairs["both_correct"])


def test_summary_and_legacy_metrics():
    pred = build_predictions(_meta(), _scores())
    res = score_predictions(pred, model_name="toy", config="pairs-72")
    s = res.summary
    assert s["css"] == 0.5 and s["acc"] == 0.75 and s["foil_rate"] == 0.25
    assert s["n_images"] == 4 and s["n_pairs"] == 2 and s["chance_css"] == CHANCE_CSS
    assert s["model_name"] == "toy" and s["config"] == "pairs-72" and "eval_version" in s
    assert 0 <= s["css_ci_low"] <= s["css"] <= s["css_ci_high"] <= 1
    assert res.confusion.loc["bunny", "bear"] == 1 and res.confusion.to_numpy().sum() == 4

    leg = legacy_metrics(pred)
    assert leg["global_pair_acc"] == 0.5 and leg["target_or_foil_acc"] == 1.0
    assert leg["target_over_foil_acc"] == 0.75
    assert leg["target_foil_bias"] == pytest.approx(math.sqrt(0.75))


def test_bootstrap_is_seeded_and_bounded():
    v = np.array([1, 1, 0, 0, 1, 0, 1, 1])
    a, b = bootstrap_ci(v, seed=1), bootstrap_ci(v, seed=1)
    assert a == b
    assert 0 <= a[0] <= v.mean() <= a[1] <= 1


def test_save_load_roundtrip(tmp_path):
    res = score_predictions(build_predictions(_meta(), _scores()), model_name="toy", config="pairs-72")
    res.save(tmp_path)
    back = AnagramResults.load(tmp_path)
    assert back.summary == res.summary
    pd.testing.assert_frame_equal(back.pairs, res.pairs)
    assert "AnagramResults(css=0.500" in repr(back)
