# eval-anagrams

Lightweight implementation of the Object-Anagram / **Configural Shape Score (CSS)** eval from
Doshi, Fel, Konkle & Alvarez, *Visual Anagrams Reveal Hidden Differences in Holistic Shape
Processing Across Vision Models* (NeurIPS 2025, [arXiv:2507.00493](https://arxiv.org/abs/2507.00493)).

Each anagram pair is two images made from the same 16 puzzle pieces, rearranged to depict two
different animals. CSS = fraction of pairs for which a model classifies **both** images correctly
(9-way; chance 1/81). Local texture cannot solve it; global configuration can.

## Install (uv)

```bash
git clone https://github.com/harvard-visionlab/eval-anagrams.git
cd eval-anagrams
uv sync --dev                       # + --extra validation (timm, open_clip) for scripts/run_validation.py
                                    # + --extra models     (visionlab.models) for lab model loading
```

Torch resolves to the CUDA 12.6 build on Linux x86_64 and CPU builds elsewhere (see `pyproject.toml`).

## Usage

```python
from torchvision import transforms as T
from torchvision.models import alexnet, AlexNet_Weights
from visionlab.evals.anagrams import anagram_eval

model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
transform = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                       T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

results = anagram_eval(model, transform, config="pairs-72")   # or "pairs-1440"
results.summary["css"]      # 0.056 for AlexNet (paper: 0.056)
results.predictions         # one row per image: pred, correct, 9 scores, decision_margin, foil_margin
results.pairs               # one row per anagram pair: both_correct, pair_margin
results.confusion           # 9x9 true x pred
results.save("results/alexnet")   # summary.json + predictions.csv
```

With `visionlab.models`:

```python
from visionlab.models import load_model
model, transforms = load_model("pytorch/alexnet:DEFAULT")
results = anagram_eval(model, transforms.test, model_name="pytorch/alexnet:DEFAULT")
```

`anagram_eval(model, transform, ...)` accepts any callable returning (B, 1000) ImageNet scores
(mapped to the 9 categories via the paper's Table 2, max over indices) or (B, 9) category scores.
Anything else: pass `to_anagram_scores=<callable -> (B, 9)>`. CLIP/SigLIP-style models:
`ZeroShotClassifier.from_open_clip("hf-hub:timm/ViT-L-16-SigLIP2-256")` returns a classifier +
preprocess that plug straight in.

## Metrics (`results.summary`)

| key | meaning |
|---|---|
| `css`, `css_ci_low/high` | Configural Shape Score, seeded 1000× bootstrap CI over pairs |
| `acc` | single-image 9-way accuracy (chance 1/9) |
| `foil_rate` | P(pred == the pair's other animal): "saw the pieces, picked the partner" |
| `dm_*` | decision margin `(S_label − max other) / √2` stats (Alvarez & Konkle 2024) |
| `foil_margin_mean` | mean `(S_label − S_foil) / √2` |
| `pair_margin_mean` | mean `min(dm0, dm1)` per pair (pair correct ⇔ pair_margin > 0) |

The original paper's `target_or_foil / target_over_foil / target_foil_bias` are available via
`legacy_metrics(results.predictions)`.

## Dataset

[`visionlab/visual-anagrams`](https://huggingface.co/datasets/visionlab/visual-anagrams) on
HuggingFace: configs `pairs-72` (144 images, paper main set) and `pairs-1440` (2880 images,
expanded set), plus `animations/*.mp4` and `imagenet_class_map.json`. Built by
`scripts/build_hf_dataset.py`; card in `scripts/dataset_card.md`; walkthrough in
`notebooks/demo_dataset.ipynb`.

## Validation

`reference/` holds the paper's per-model numbers. `scripts/run_validation.py` runs a small sweep
spanning torchvision, timm heads, DINOv2-lc, and SigLIP2 zero-shot, and prints ours vs Doshi:

```bash
uv sync --dev --extra validation
uv run python scripts/run_validation.py --configs pairs-72 pairs-1440
```

## Results store (lab)

Results live in `s3://visionlab-evals` as a Hive-partitioned tree, mirrored at
`~/.cache/visionlab/evals` (`VISIONLAB_EVALS_CACHE`, `VISIONLAB_EVALS_BUCKET` to override):

```
eval=anagrams/version=0.1.0/dataset=pairs-72/model=pytorch__alexnet__7be5be79/readout=head/intervention=none/results.parquet  # per image
eval=anagrams/version=0.1.0/dataset=pairs-72/model=pytorch__alexnet__7be5be79/readout=head/intervention=none/summary.json     # metrics + provenance
```

Model identity is `source/arch:weights_id` with `weights_id` = sha256[:8] of the weights file
(the `visionlab.models` convention); aliases like `DEFAULT` are resolved before storing. `readout=`
says how class scores were obtained: `head` (native classifier), `probe__<layer>__<hash8>`,
`prototypes__<layer>__<hash8>`, `zeroshot__<hash8>` (self-supervised / CLIP backbones; readout ids are
file hashes provided by `visionlab.models`). `intervention=` is a declared, parameter-free change to
the computation at inference (top-k sparsification `topk__k0.4`, LRM pass count `lrm__passes1`,
attention masks, ablations), `none` when the model runs as-is; its id is sha256[:8] of the canonical
string `kind:k=v,...`. Both files carry the identity columns
(`eval_name, eval_version, dataset, model_id, model_spec, model_source, model_arch, weights_id,
readout_type, readout_layer, readout_id, readout_spec, readout_n_classes, readout_train_data,
readout_primary, intervention_kind, intervention_params, intervention_id, intervention_spec`), so
they stay self-describing when copied out of the tree. Objects are private (lab AWS credentials).

```python
from visionlab.evals.anagrams import ResultsStore, ModelIdentity

store = ResultsStore()
store.run("pytorch/alexnet:DEFAULT", dataset="pairs-72")        # via visionlab.models; skips if stored
store.run("visionlab/alexnet_ipcl:3f9a1c2d@fc6_probe", dataset="pairs-72")   # SSL backbone + readout
store.run("pytorch/alexnet:7be5be79+topk:k=0.4", dataset="pairs-72")          # backbone + intervention
store.run(model, transform, dataset="pairs-72",                  # any model, explicit identity
          identity=ModelIdentity.from_torchvision(AlexNet_Weights.IMAGENET1K_V1))
store.query(dataset="pairs-72")                                  # DataFrame: one row per (model, readout, intervention)
store.load("pairs-72", "pytorch/alexnet:7be5be79")               # full AnagramResults
```

Any Hive-aware tool reads the tree directly; e.g. DuckDB:

```sql
INSTALL httpfs; LOAD httpfs; CREATE SECRET (TYPE s3, PROVIDER credential_chain);
SELECT dataset, model_id, avg(correct::INT) acc, avg(decision_margin) dm
FROM read_parquet('s3://visionlab-evals/eval=anagrams/version=0.1.0/*/*/*/*/results.parquet', hive_partitioning=true)
GROUP BY ALL;
```

`scripts/run_doshi_sweep.py` runs the full Doshi2025 collection through the store (needs
`visionlab.models` with that collection tagged).

## Known differences from the paper

Validated against the authors' per-model csv (`reference/`): AlexNet, ResNet-50, ViT-B/16,
timm ViT-B/16-augreg, BEiTv2-B/16, DINOv2-B/14-lc match exactly on both sets when run in strict
fp32. Two things to expect when reproducing the full sweep:

- **Zero-shot (SigLIP/CLIP) models score slightly differently.** The paper's SigLIP wrapper
  resized 256→224 (bilinear), quantized to uint8, then open_clip upsampled to 256 (bicubic).
  We feed the native 256² image to the model's own preprocess. That blur costs ~0.6% of images
  (SigLIP2-L/16: CSS 0.806 vs 0.819 on 72 pairs, 0.861 vs 0.873 on 1440). Expect the same offset
  for every zero-shot model in the Doshi2025 suite. `scripts/run_validation.py --paper-pipeline`
  reproduces the paper's numbers exactly if parity is needed.
- **Resize interpolation.** The paper used torchvision's default *bilinear* `Resize((224,224))` for
  every model. We use each model's own eval interpolation from its card, so bicubic-native models
  (DINOv2, timm ViTs, BEiT, ConvNeXt, …) land 1–3 pairs from the paper (DINOv2-B/14: CSS 0.569 vs
  0.611 on 72 pairs; bilinear reproduces 0.611 exactly). `scripts/run_doshi_sweep.py --paper-pipeline`
  runs the paper's preprocessing as a replication check without touching the store.
- **Adversarially robust ResNet-50, ε = 0.25.** Doshi's builder pointed the ε=0.1 and ε=0.25 entries at
  the same Dropbox file, so the paper's ε=0.25 row was computed with ε=0.1 weights. The models repo
  mirrors Madry's canonical checkpoints for all ten ε values; expect the ε=0.25 result to differ from
  the paper (and the ε=0.1 result to match), pending the file-hash check.
- **TF32 can flip near-zero-margin images.** ResNet-50 has one image (cat/turtle, |dm| = 0.004)
  that flips between fp32 and TF32 kernels. Run with TF32 disabled for reproducible numbers
  (the validation script does this by default).

## Development

```bash
uv run pytest            # unit tests + a CPU end-to-end test (downloads the 18 MB pairs-72 config)
uv run ruff check anagrams tests scripts --line-length 120
```

Notebooks are jupytext-paired (`.ipynb` + `.py:percent`); configure output stripping once per clone:

```bash
git config filter.nbstripout.clean 'uv run nbstripout'
git config filter.nbstripout.smudge cat
git config filter.nbstripout.required true
git config diff.ipynb.textconv 'uv run nbstripout -t'
```
