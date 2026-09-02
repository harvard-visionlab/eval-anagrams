import pytest
import torch
from visionlab.evals.anagrams import CLASSES, IMAGENET_CLASS_MAP, default_to_anagram_scores, imagenet_to_anagram_scores


def test_class_map_covers_nine_disjoint_categories():
    assert list(IMAGENET_CLASS_MAP) == CLASSES
    all_idx = [i for idx in IMAGENET_CLASS_MAP.values() for i in idx]
    assert len(all_idx) == len(set(all_idx))
    assert all(0 <= i < 1000 for i in all_idx)


def test_imagenet_max_mapping_picks_category_max():
    out = torch.zeros(2, 1000)
    out[0, 295] = 5.0  # bear
    out[0, 43] = 3.0  # lizard
    out[1, 386] = 7.0  # elephant
    scores = imagenet_to_anagram_scores(out)
    assert scores.shape == (2, 9)
    assert scores[0, CLASSES.index("bear")] == 5.0
    assert scores[0, CLASSES.index("lizard")] == 3.0
    assert scores[0].argmax() == CLASSES.index("bear")
    assert scores[1].argmax() == CLASSES.index("elephant")
    mean_scores = imagenet_to_anagram_scores(out, reduction="mean")
    assert mean_scores[0, CLASSES.index("bear")] == pytest.approx(5.0 / 4)


def test_default_hook_passthrough_and_errors():
    nine = torch.randn(3, 9)
    assert torch.equal(default_to_anagram_scores(nine), nine)
    assert default_to_anagram_scores(torch.randn(3, 1000)).shape == (3, 9)
    with pytest.raises(ValueError, match="to_anagram_scores"):
        default_to_anagram_scores(torch.randn(3, 512))
    with pytest.raises(TypeError):
        default_to_anagram_scores((nine, nine))
