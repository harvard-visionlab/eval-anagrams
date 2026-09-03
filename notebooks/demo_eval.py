# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Configural Shape Score — eval demo
#
# Run `anagram_eval` on a few torchvision models, look at every level of the output
# (summary, per-image predictions with decision margins, per-pair outcomes, confusion), and
# place the models against the full Doshi et al. (2025) sweep.
#
# Everything here runs on CPU in a few minutes (144 images per model).

# %%
import pandas as pd
import torch
from torchvision import models, transforms as T

from visionlab.evals.anagrams import (
    AnagramResults,
    anagram_eval,
    legacy_metrics,
    load_anagrams,
    plot_acc_vs_css,
    plot_confusion,
    plot_css,
    plot_margins,
    show_pairs,
)

torch.set_num_threads(8)

# %% [markdown]
# ## Models and preprocessing
#
# `anagram_eval` takes a model and *its own* eval transform. The anagram images are 256×256, so
# `Resize((224, 224))` is identical to `Resize(224) + CenterCrop(224)` (visionlab `transforms.test`)
# and to the paper's preprocessing.

# %%
IMAGENET = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(**IMAGENET)])

MODELS = {
    "alexnet": lambda: models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1),
    "resnet50": lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
    "vit_b_16": lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1),
}
# names used for the same checkpoints in the paper's results table
DOSHI_NAMES = {"alexnet": "alexnet_in1k_v1_7be5be79", "resnet50": "resnet50_in1k", "vit_b_16": "vitb16_in1k"}

# %% [markdown]
# ## Run the eval

# %%
results = {}
for name, build in MODELS.items():
    results[name] = anagram_eval(build(), transform, config="pairs-72", num_workers=0, progress=False,
                                 model_name=name, doshi_name=DOSHI_NAMES[name])
    print(name, results[name])

# %% [markdown]
# ## Summary: one row per model
#
# `css` is the Configural Shape Score (both images of a pair correct; chance 1/81), `acc` is
# single-image 9-way accuracy, `foil_rate` is how often the model names the pair's *other* animal,
# and the `dm_*` columns describe the decision margin `(S_label − max other) / √2`.

# %%
summaries = pd.DataFrame([r.summary for r in results.values()])
summaries.set_index("model_name").T

# %% [markdown]
# Compare with the paper's numbers for the same checkpoints (should match exactly):

# %%
ref = pd.read_csv("../reference/doshi_css_pairs72.csv").set_index("model_name")
pd.DataFrame({
    "css": summaries.set_index("model_name")["css"].round(4).values,
    "doshi_css": ref.loc[summaries["doshi_name"], "css"].round(4).values,
    "acc": summaries.set_index("model_name")["acc"].round(4).values,
    "doshi_acc": ref.loc[summaries["doshi_name"], "acc"].round(4).values,
}, index=summaries["model_name"])

# %% [markdown]
# ## Per-image predictions
#
# The source of truth. Every summary metric can be recomputed from this table. Scores are the
# model's 9 category scores (max over each category's ImageNet classes for a 1000-way head).

# %%
res = results["resnet50"]
res.predictions.head(6)

# %% [markdown]
# `decision_margin` is positive iff the image is correct. `foil_margin` compares the label
# against the pair's other animal specifically: negative foil margins are "texture-twin" confusions.

# %%
res.predictions[["correct", "decision_margin", "foil_margin"]].describe().round(3)

# %% [markdown]
# ## Per-pair outcomes
#
# `both_correct` is the CSS event; `pair_margin = min(dm0, dm1)` is its continuous version.

# %%
res.pairs.sort_values("pair_margin", ascending=False).head(8)

# %%
res.pairs["both_correct"].mean(), (res.pairs["pair_margin"] > 0).mean()

# %% [markdown]
# ## Confusion and margins

# %%
plot_confusion(res);

# %%
plot_margins(res);

# %% [markdown]
# ## Look at the images
#
# Green titles are correct, red are wrong; `dm` is the decision margin.

# %%
ds = load_anagrams("pairs-72")
show_pairs(results["vit_b_16"], ds, select="both_correct", n=4);

# %%
show_pairs(results["vit_b_16"], ds, select="both_wrong", n=4);

# %% [markdown]
# ## Against the full Doshi et al. sweep
#
# `reference/doshi_css_pairs72.csv` has the paper's 91 models. Our three checkpoints are
# highlighted; the dashed line is the human estimate (n = 4), dotted is chance (1/81).

# %%
plot_css(ref.reset_index(), highlight=list(DOSHI_NAMES.values()), title="Configural Shape Score, 72 pairs (Doshi et al. 2025)");

# %%
plot_acc_vs_css(ref.reset_index(), highlight=list(DOSHI_NAMES.values()));

# %% [markdown]
# Our own runs, with bootstrap CIs over pairs:

# %%
plot_css(summaries, highlight=list(MODELS), title="this notebook");

# %% [markdown]
# ## Legacy metrics
#
# The paper's `target_or_foil_acc / target_over_foil_acc / target_foil_bias` are recoverable
# from the predictions table:

# %%
pd.DataFrame({name: legacy_metrics(r.predictions) for name, r in results.items()}).round(4)

# %% [markdown]
# ## Save and reload

# %%
path = res.save("../results/demo/resnet50_pairs72")
back = AnagramResults.load(path)
back.summary == res.summary, back

# %% [markdown]
# ## With `visionlab.models`
#
# Lab members can skip the torchvision boilerplate (`uv sync --extra models`):
#
# ```python
# from visionlab.models import load_model
# model, transforms = load_model("pytorch/alexnet:DEFAULT")
# results = anagram_eval(model, transforms.test, model_name="pytorch/alexnet:DEFAULT")
# ```
