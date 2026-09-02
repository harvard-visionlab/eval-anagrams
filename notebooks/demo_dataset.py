# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # visual-anagrams — dataset demo
#
# Object-Anagram stimuli from Doshi, Fel, Konkle & Alvarez (NeurIPS 2025).
# Each *anagram pair* is two images built from the **same 16 puzzle pieces** (identical
# local texture), rearranged to depict two different animals. Recognizing both images
# in a pair requires configural (global-arrangement) shape processing.
#
# Configs: `pairs-72` (144 images, paper main set) and `pairs-1440` (2880 images, expanded set).

# %%
from collections import Counter

import matplotlib.pyplot as plt
from datasets import load_dataset

# %%
ds = load_dataset("visionlab/visual-anagrams", "pairs-72", split="test")
ds

# %% [markdown]
# ## One row
#
# Labels are `ClassLabel` ints; `ds.features["label"].names` gives the 9 category names.

# %%
CLASSES = ds.features["label"].names
print(CLASSES)

row = ds[0]
row["image"].size, {k: v for k, v in row.items() if k != "image"}

# %%
row["image"]

# %% [markdown]
# ## Show anagram pairs
#
# `anagram_id` groups the two images of a pair; `position` 0/1 says which member.
# `label` is the depicted animal, `foil` is the other animal in the pair.

# %%
def show_pairs(ds, anagram_ids, ncols=4):
    classes = ds.features["label"].names
    by_id = {}
    for i, aid in enumerate(ds["anagram_id"]):
        by_id.setdefault(aid, []).append(i)
    nrows = -(-len(anagram_ids) // ncols)
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(2.2 * ncols * 2, 2.4 * nrows))
    axes = axes.reshape(nrows, ncols * 2)
    for k, aid in enumerate(anagram_ids):
        r, c = divmod(k, ncols)
        for idx in by_id[aid]:
            row = ds[idx]
            ax = axes[r, 2 * c + row["position"]]
            ax.imshow(row["image"])
            ax.set_title(f"{aid}: {classes[row['label']]}", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    return fig


ids = sorted(set(ds["anagram_id"]))
show_pairs(ds, ids[:8]);

# %% [markdown]
# ## Class balance and pair structure
#
# 72 pairs = every ordered pair of the 9 categories (9 × 8). Each image appears once,
# so each category appears 16 times.

# %%
Counter(CLASSES[l] for l in ds["label"])

# %%
pairs = sorted({(CLASSES[a], CLASSES[b]) for a, b in zip(ds["object0"], ds["object1"])})
len(pairs), pairs[:10]

# %% [markdown]
# ## Expanded set (1440 pairs)
#
# Same 72 category pairs × 20 variants (`variant` 0–19), generated with different seeds.
# Note: not a superset of `pairs-72` (those images are a separate generation run).

# %%
ds_big = load_dataset("visionlab/visual-anagrams", "pairs-1440", split="test")
ds_big

# %%
Counter(ds_big["variant"]).most_common(3), Counter(CLASSES[l] for l in ds_big["label"])["bear"]

# %%
# the 20 variants of one category pair (bear / bunny)
sub = ds_big.filter(lambda r: r["pair_id"] == 0)
show_pairs(sub, sorted(set(sub["anagram_id"]))[:8]);

# %% [markdown]
# ## Category → ImageNet class indices
#
# For ImageNet-1k classifiers, logits are mapped to the 9 categories via the max over
# these indices (paper Table 2). Shipped in the dataset repo as `imagenet_class_map.json`.

# %%
import json

from huggingface_hub import hf_hub_download

json.load(open(hf_hub_download("visionlab/visual-anagrams", "imagenet_class_map.json", repo_type="dataset")))

# %% [markdown]
# ## Animations
#
# 72 mp4s (one per pair in `pairs-72`) morphing between the two arrangements live under
# `animations/` in the dataset repo (h264 mp4, 1536², ~12 s each).

# %%
from IPython.display import Video

mp4 = hf_hub_download("visionlab/visual-anagrams", "animations/000_transform_bear_bunny.mp4", repo_type="dataset")
Video(mp4, embed=True, width=384)  # embed: file lives in the HF cache, outside the Jupyter root

# %%

# %%
