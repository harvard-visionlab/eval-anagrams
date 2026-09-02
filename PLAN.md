# eval-anagrams — Plan

Lightweight, public implementation of the Object-Anagram / Configural Shape Score (CSS)
eval from Doshi, Fel, Konkle & Alvarez (NeurIPS 2025, arXiv 2507.00493).
Reference impl: `../anagram_holistic_shape_neurips` (research grade).

## Task 1 — Dataset → HuggingFace (`visionlab` org)

### Source audit (done 2026-09-01)
Source: `~/Documents/DataSets/Doshi2025-VisualAnagrms/neurips_datasets/`

| set | images | pairs | per class | size | notes |
|---|---|---|---|---|---|
| `anagram_stimulus_3_jigsaw` | 144 | 72 | 16 | 18 MB | paper main set; + 72 mp4 animations (190 MB) + 180 control pairs csv |
| `anagram_stimulus_3_expanded_post_rebuttal` | 2880 | 1440 | 320 | 368 MB | 72 pairs × 20 variants (Appendix A.10); NOT a superset of the 144 (different seeds) |

- All PNG, 256×256, RGB. 9 classes: bear bunny cat elephant frog lizard tiger turtle wolf.
- 72 pairs = all 9×8 ordered category pairs. Filename encodes everything:
  `{pair:03d}[_{variant:03d}]_transform_{obj0}_{obj1}_object{k}_{label}.png`.
- Labels verified: filename ↔ CSV `groundtruth/object0/object1` agree on all 144 + 2880 rows;
  `label == [obj0,obj1][k]` everywhere. Spot-checked images look correct.
- `*_behavior.csv` = same rows, different path prefix (ignore). `.ipynb_checkpoints/` contain
  stale/mislabeled copies (ignore). `.DS_Store` (ignore).
- `anagram_stimulus_3_jigsaw_twoimagedisplays.csv`: 180 image pairs, 60 each of
  `target_gdiff_lsame` (anagram pair), `control_gdiff_ldiff`, `control_gsame_ldiff` (Fig 4 RSA).
- Human data: 4 participants × 144 trials (jsPsych json in neurips repo
  `evals/anagram_perceptshift_benchmark/data/behavior/human/`), fields:
  image_name, ground_truth, response_value, response_time.
- HF: `HF_TOKEN` in env, user `grez72`, write token, admin of `visionlab` org
  (org already has 3 block-towers datasets). No `huggingface_hub` installed anywhere yet.

### Proposed HF layout — one repo, multiple configs
`visionlab/visual-anagrams`

| config | split | rows | contents |
|---|---|---|---|
| `pairs-72` (default) | `test` | 144 | main set (paper Fig 2) |
| `pairs-1440` | `test` | 2880 | expanded set (Appendix A.10) |
| `control-pairs` | `test` | 180 | two-image displays for RSA (images inline, from pairs-72) |
| `human-behavior` | `test` | 576 | 4 subjects × 144 trials |

Alternative (per SPEC): two repos `visual-anagrams-jigsaw-144` / `-2880`. One repo + configs
preferred: single card, single citation, `load_dataset("visionlab/visual-anagrams", "pairs-1440")`.

Row schema for `pairs-*` (labels as `ClassLabel` so `.features["label"].names` gives the 9 names):

| column | type | example |
|---|---|---|
| `image` | Image | 256×256 RGB PNG |
| `filename` | string | `000_003_transform_bear_bunny_object1_bunny.png` |
| `anagram_id` | string | `000_003` (pair_id[_variant]); groups the 2 images of a pair |
| `pair_id` | int | 0–71 (ordered category pair index) |
| `variant` | int | 0 for pairs-72; 0–19 for pairs-1440 |
| `position` | int | 0/1 = object0/object1 |
| `label` | ClassLabel(9) | `bunny` (ground truth for this image) |
| `foil` | ClassLabel(9) | `bear` (the other object in the pair) |
| `object0`, `object1` | ClassLabel(9) | category pair, in filename order |

Rows sorted by filename (source csv for 144 is shuffled; irrelevant).

`control-pairs`: `image1, image2 (Image), filename1, filename2, label1, label2 (ClassLabel), condition (ClassLabel: target_gdiff_lsame / control_gdiff_ldiff / control_gsame_ldiff)`.

`human-behavior`: `subject_id, trial_index, filename, anagram_id, label, response (ClassLabel), correct (bool), rt_ms`.

Extras (raw files, not parquet): `extras/animations/*.mp4` (190 MB) under the same repo.
Also ship `imagenet_class_map.json` (9-class → ImageNet indices, Table 2) in the repo root
and in the dataset card.

Dataset card (README.md YAML + prose): description, 9-class → ImageNet index table, how CSS
is computed, chance = 1/81, human baseline, bibtex, license (**decision needed** — suggest
CC-BY-4.0; images are DeepFloyd-IF generated). Tags: `image-classification`, `shape`, `vision`.

### Status (2026-09-01)
Decisions (George): one repo + configs; CC-BY-4.0; upload `pairs-72`, `pairs-1440`, animations
only for now (no control-pairs / human-behavior yet); start PRIVATE, flip public after Doshi review
(~week of 2026-09-08).

- [x] pyproject (uv, torch cu126/cpu indexes, jupyterlab dev group); import path `visionlab.evals.anagrams`
- [x] `scripts/build_hf_dataset.py` — parse filenames, validate, push
- [x] pushed `pairs-72` (144) + `pairs-1440` (2880) as `test` splits, PRIVATE
- [x] `animations/*.mp4` (72) + `imagenet_class_map.json` uploaded as raw files (not a config)
- [x] round-trip validated: all 3024 images pixel-identical to source; class counts 16/320
- [x] dataset card (`scripts/dataset_card.md` → repo README.md), default config = pairs-72
- [x] `notebooks/demo_dataset.{py,ipynb}` executes clean
- [ ] show Doshi → flip public: `HfApi().update_repo_settings("visionlab/visual-anagrams", repo_type="dataset", private=False)`
- [ ] maybe later: `control-pairs`, `human-behavior` configs

## Task 2 — Eval (`visionlab.evals.anagrams`)

### A. The published eval, as actually implemented
Reference: `anagram_holistic_shape_neurips/neurips_25/evals/anagram_perceptshift_benchmark/`
(`run_all_models.py`, `analyses/__init__.py`, `lib_custom_anagram/custom_model_utils.py`).

**Pipeline (per model)**
1. **Data**: `CSVImageDataset` over `anagram_stimulus_3_jigsaw.csv` (144 imgs); expanded run just
   swaps the folder. Each row: image, filename, groundtruth, object0, object1.
2. **Preprocess** (same for ALL models): `Resize((224,224))` + ImageNet mean/std. No crop
   (paper text says "centre-cropped 224" — equivalent on 256² squares). Models needing other
   preprocessing (SigLIP) *denormalize → PIL → re-preprocess* inside their forward.
3. **Forward → 1000-way ImageNet scores.** Sources of the 1000-way head:
   - torchvision/timm supervised: native classifier.
   - self-supervised (BEiT, MAE, Hiera, DINOv2): timm/hub fine-tuned linear heads (`*_lc`).
   - SigLIP/SigLIP2: zero-shot over the 1000 bare ImageNet class names (no prompt template),
     `sigmoid(img·txt * scale + bias)` → returns *probabilities* not logits.
4. **1000 → 9**: for each category, `max` over its ImageNet indices (Table 2) → 9 scores →
   softmax → argmax. (Softmax is a no-op for the argmax.)
5. **Metrics** (one csv row per model):
   - `acc` — 9-way single-image accuracy (chance 1/9).
   - `target_or_foil_acc` = P(pred ∈ {object0, object1}); `target_over_foil_acc` =
     P(pred == label | pred ∈ pair); `target_foil_bias` = sqrt(product) (Geirhos-style).
   - `global_pair_acc` = **CSS** = fraction of anagram pairs with both images correct
     (chance 1/81 ≈ 1.2%). Pair membership found by substring-matching the filename prefix.
   - 95% CI: 1000-sample bootstrap over pairs (unseeded `np.random`).
6. **Outputs**: summary csv per model under `results/anagram_perceptshift/summary/`, concatenated
   into `anagram_perceptshift_benchmark.csv`; `plot_results.ipynb` sorts models by CSS, grey bar
   chart, dashed human line, plus acc-vs-CSS scatter.
7. **Human baseline** (4 lab subjects, 750 ms + 500 ms mask, 9-AFC): acc 0.948, **CSS 0.896**
   (per-subject CSS .861/.903/.931/.889).
8. **Nulls / sanity checks in the notebook only** (not in the paper): random re-pairing of the
   144 images (10k sims) shows CSS is not just acc² — pair structure matters.

**Reference CSS** — 72 pairs: alexnet .056, resnet50 .167, vit-b/16-augreg .236, bagnet9 .028,
dinov2-b/14 .611, dinov2-g/14 .778, eva-clip-g .778, siglip2-L .819, untrained 0.
1440 pairs: alexnet .049, resnet50 .231, dinov2-b/14 .738 (r = .99 between sets).

**Quirks worth NOT reproducing**
- Fixed ImageNet normalization for every model (timm CLIP-finetuned heads expect CLIP stats).
- Metric computation keyed off filename substrings; anagram_id groupby is cleaner.
- `target_*` trio is three numbers for one idea ("does the model at least see the pair?").
- Unseeded bootstrap; `model.id/.name` monkey-patched onto modules; results only as csv rows,
  no per-image predictions saved (so nothing can be recomputed later).

### B. Plan — clean implementation in this repo  [revised 2026-09-02]

**Guiding choices**
- Readable > compact. Small modules, plain functions, one results dataclass.
- `anagram_eval(model, transform, ...)` takes ONLY a model and its transform. Model loading is
  `harvard-visionlab/models`' job (`load_model(spec)` → `model, transforms`; use `transforms.test`:
  on 256² inputs `Resize(224)+CenterCrop(224)` == paper preprocessing).
- Always return **per-image predictions** incl. **decision margin** (lab standard, Alvarez &
  Konkle 2024 UniReps; same formula as `eval-classification`). Every summary metric is a
  function of that table.
- Model-agnostic decision rule via one hook (`to_anagram_scores`), sensible default.
- All GPU validation runs on the remote workstation: I push code + a run script, George pulls & runs.

**Layout**
```
anagrams/
  __init__.py     # anagram_eval, load_anagrams, score_predictions, AnagramResults, CLASSES, IMAGENET_CLASS_MAP
  data.py         # load_anagrams(config) -> HF Dataset (cached); AnagramDataset(ds, transform) -> (tensor, index)
  mapping.py      # CLASSES, IMAGENET_CLASS_MAP, imagenet_to_anagram_scores(outputs, reduction="max"), auto_scorer()
  scoring.py      # decision margins; score_predictions(pred_df) -> AnagramResults; bootstrap_ci (seeded)
  eval.py         # anagram_eval(model, transform, config, to_anagram_scores, batch_size, device, num_workers)
  plot.py         # plot_css(summaries, human=0.896), show_pairs_with_predictions(results, ...)
  store.py        # (lab) ResultsStore: local cache + S3 mirror + query across model collections
  legacy.py       # legacy_metrics(pred_df): target_or_foil / target_over_foil / target_foil_bias — parity only
scripts/
  run_validation.py    # phase-1 sweep on the GPU box; writes results/ + comparison vs reference
  run_doshi_sweep.py   # phase-2 full replication via visionlab.models tags=["Doshi2025"] (later)
reference/
  doshi_css_pairs72.csv, doshi_css_pairs1440.csv   # published per-model numbers (from neurips repo)
notebooks/demo_eval.py
tests/   # mapping shapes; scoring on synthetic preds (css/acc/ci/margins known); tiny e2e on 2 pairs (CPU)
```

**Core API**
```python
from visionlab.evals.anagrams import anagram_eval

results = anagram_eval(model, transform,          # transform: PIL -> tensor, e.g. transforms.test
                       config="pairs-72",         # or "pairs-1440"
                       to_anagram_scores=None,    # None → auto: 1000-d → ImageNet max-map; 9-d → identity
                       batch_size=64, device=None, num_workers=4)

results.summary       # dict (see metrics)
results.predictions   # DataFrame, 1 row / image
results.pairs         # DataFrame, 1 row / anagram pair
results.confusion     # 9x9 DataFrame
results.save(dir) / AnagramResults.load(dir)     # summary.json + predictions.csv (pairs/confusion re-derived)
```
`to_anagram_scores(model_outputs) -> (B, 9)`: default handles ImageNet-1k heads (torchvision, timm,
hub `*_lc`); 9-way heads pass through; zero-shot CLIP/SigLIP pass a closure over 9 (or 1000) text
embeddings — we ship `zero_shot_scorer(...)` helper in phase 1 since SigLIP is a validation target.

**Per-image `predictions` columns**
`filename, anagram_id, pair_id, variant, position, label, foil, pred, correct,
score_bear … score_wolf (9), target_score, max_nontarget_score, decision_margin, foil_margin`
- `decision_margin = (S_label − max_{j≠label} S_j) / √2`  (UniReps eq. 4; identical to eval-classification)
- `foil_margin = (S_label − S_foil) / √2`  — anagram-specific: evidence for the depicted object
  over its texture-twin. New; cheap; proposed.
- Scores are whatever the hook returns (logits for ImageNet heads; zero-shot logits for CLIP-type).
  Margins are therefore comparable *within* a model across items (the UniReps use case), not in
  absolute units across models.

**Per-pair `pairs` columns**
`anagram_id, pair_id, variant, object0, object1, correct0, correct1, both_correct,
dm0, dm1, pair_margin = min(dm0, dm1)`  (pair correct ⇔ pair_margin > 0).

**Summary metrics** (decided: keep css/acc, add foil_rate, drop target_* trio to `legacy.py`)
| metric | definition |
|---|---|
| `css`, `css_ci_low/high` | fraction of pairs with both correct; seeded 1000× bootstrap over pairs |
| `acc` | single-image 9-way accuracy |
| `foil_rate` | P(pred == foil): "saw the pieces, picked the partner" |
| `dm_mean`, `dm_median`, `dm_min`, `dm_max`, `dm_kurtosis` | decision-margin stats (as in eval-classification) |
| `foil_margin_mean`, `pair_margin_mean` | |
| `chance_css` = 1/81, `n_images`, `n_pairs`, `config`, `eval_version` | bookkeeping |

**Lab convenience — results logging / caching / querying** (`store.py`; after core is solid)
Modeled on `eval-classification.personal_hub.ClassifierStore` but simpler:
- Identity of a result = `(eval_version, config, model_id, transform_sig)`; `model_id` = visionlab
  spec + weights hashid (`pytorch/alexnet:7be5be79`), `transform_sig` = `resize224_crop224_bilinear`
  parsed from the Compose (fallback: user-supplied string).
- Local cache `~/.cache/visionlab/results/eval-anagrams/<eval_version>/<config>/<model_id>_<sig>/{summary.json,predictions.csv}`
  (env override `VISIONLAB_RESULTS`), mirrored to S3/wasabi `visionlab-results/<user>/dnn-evals/eval-anagrams/...`
  via `s3_filestore` (same bucket/profile conventions as ClassifierStore). Both optional deps.
- API: `store.run(spec, config)` → load_model → anagram_eval → save (or return cached);
  `store.query(specs=..., tags=["Doshi2025"], config=...)` → one summary row per model (pulls from
  local, then S3); `store.predictions(spec, config)` for item-level analyses (DM consistency, etc.).
- Batch: `run_doshi_sweep.py` = `for spec in list_models(tags=["Doshi2025"]): store.run(spec, cfg)`.

**Self-supervised models without heads** (deferred; plan after full replication)
Two routes, both produce a `to_anagram_scores` hook from a chosen layer's features:
1. Linear probe on ImageNet-1k (matches paper's `*_lc` heads; expensive — cluster job;
   reuse `eval-classification.linear_probe` when ported).
2. Prototype classifier: class-mean (or few-shot) prototypes from ImageNet-1k features for the
   9 categories' ImageNet classes → cosine/nearest-prototype scores (≈1/10 compute, slightly lower acc).
Design the hook interface now so both drop in later; decide default + report both on a subset.

**Validation (all on the remote GPU workstation)**
- Phase 0 — repo plumbing: commit, create `harvard-visionlab/eval-anagrams` remote, push.
- Phase 1 — "all flavors work" (`scripts/run_validation.py`, small sweep, `[validation]` extra = timm, open_clip):
  - supervised torchvision: alexnet, resnet50, vit_b_16 → expect CSS .056 / .167 / .236 (pairs-72)
  - timm/hub with linear heads: `dinov2_vitb14_lc` (.611), a BEiT/BEiTv2 timm model (in reference csv)
  - SigLIP2 zero-shot: `ViT-L-16-SigLIP2-256` (.819) via `zero_shot_scorer`
  - also run pairs-1440 for alexnet/resnet50/dinov2-b (.049 / .231 / .738)
  - script prints table: ours vs Doshi, Δ pairs; saves `results/validation_<date>/`.
  Tolerance: exact or ±1 pair for identical preprocessing; SigLIP may differ slightly (their
  wrapper round-trips through PIL + open_clip preprocess).
- Phase 2 — full Doshi replication: needs (a) models repo with the Doshi2025 collection and
  (b) `store.py`. `run_doshi_sweep.py` over 86 models × {pairs-72, pairs-1440}; compare to
  `reference/*.csv`: per-model Δ, Pearson r (paper: r=.99 between sets), flag |Δ| > 2 pairs.
- Phase 3 — SSL extension (above).

**Status (2026-09-02)**
- [x] steps 1–3: mapping/scoring/legacy/data/eval/zero_shot, tests (9 pass), reference csvs, run_validation.py
- [x] CPU smoke: torchvision AlexNet pairs-72 → css .0556 / acc .3125 / legacy bias .559 == Doshi exactly
- [ ] step 4: George runs `scripts/run_validation.py --configs pairs-72 pairs-1440` on the GPU box
- [ ] steps 5–7

**Steps (coding order)**
1. `mapping.py`, `scoring.py` (+ margins), `legacy.py`, tests on synthetic predictions.
2. `data.py`, `eval.py`, `AnagramResults` save/load; CPU smoke test on 2 pairs.
3. `zero_shot_scorer` helper; `reference/` csvs; `scripts/run_validation.py`; `[validation]` extra.
4. Commit + remote + push → George runs phase 1 → fix diffs.
5. `plot.py`, `notebooks/demo_eval.py`, README (usage, metric definitions, reference table).
6. `store.py` (local → S3) + `run_doshi_sweep.py` → phase 2 when models repo is ready.
7. SSL hook implementations (probe / prototypes) → phase 3.
