"""End-to-end on CPU with a tiny fake model (downloads the 18 MB pairs-72 config once)."""

import pytest
import torch
import torch.nn as nn
from torchvision import transforms as T
from visionlab.evals.anagrams import CLASSES, anagram_eval, imagenet_class_names

pytestmark = pytest.mark.network


class FakeImageNetHead(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.fc = nn.Linear(3 * 8 * 8, 1000)

    def forward(self, x):
        return self.fc(nn.functional.adaptive_avg_pool2d(x, 8).flatten(1))


def test_anagram_eval_end_to_end():
    tfm = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])
    res = anagram_eval(FakeImageNetHead(), tfm, config="pairs-72", batch_size=48, num_workers=0, progress=False,
                       model_name="fake")
    assert res.summary["n_images"] == 144 and res.summary["n_pairs"] == 72
    assert len(res.predictions) == 144 and set(res.predictions["label"]) == set(CLASSES)
    assert res.predictions["anagram_id"].str.len().eq(3).all()
    assert res.summary["model_name"] == "fake" and res.summary["config"] == "pairs-72"


def test_imagenet_class_names():
    names = imagenet_class_names()
    assert len(names) == 1000 and names[0] == "tench" and names[294] == "brown_bear"
    assert imagenet_class_names(pretty=True)[2] == "great white shark"
