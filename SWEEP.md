# Doshi2025 sweep — instructions for the GPU workstation session

Goal: evaluate the full Doshi2025 collection (89 loadable models; `hybrid_anime_alexnet` pending) on both
anagram datasets and write official results to `s3://visionlab-evals`. Do **not** run this on a CPU-only
machine. Everything below is idempotent: re-running reuses complete runs with identical evaluation specs.

## 1. Environment

```bash
git clone https://github.com/harvard-visionlab/eval-anagrams.git && cd eval-anagrams   # or git pull
uv sync --dev --extra validation --extra models          # torch cu126 on linux x86_64; cornet + clip included
uv run python -c "import torch, torchvision, timm; print(torch.cuda.is_available(), torchvision.__version__, timm.__version__)"
```

Pins that matter (all in `uv.lock`; do not loosen):
- `visionlab-models` @ git `2b88224` or later (identity contract v2; xformers pulled in on linux x86_64 for vendored DINOv2 — execution environment only, recorded in manifest provenance, config_ids unchanged)
- `torchvision >=0.28,<0.29`, `timm ==1.0.29`, `cornet 0.1.0` — the ranges `visionlab.models` has validated.
  Outside them `load_model` raises `UnvalidatedDependencyError`. Fix the env; do **not** set
  `VISIONLAB_MODELS_ALLOW_UNVALIDATED=1` (such runs are marked non-reusable and are not official results).

Credentials / caches:
- AWS credentials with write access to `s3://visionlab-evals` (boto3 default chain: env vars or `~/.aws`).
- `HF_TOKEN` with read access to `visionlab/visual-anagrams` (private until Doshi's review).
- Optional: `VISIONLAB_MODELS_CACHE=/fast/disk/visionlab/models` (weights, ~40 GB for the whole collection),
  `VISIONLAB_EVALS_CACHE=/fast/disk/visionlab/evals` (local mirror of the results tree).

## 2. Release gate (must pass before official writes)

```bash
uv run pytest -q                                   # 28 tests incl. tests/test_models_compat.py (must NOT skip)
uv run python scripts/run_validation.py --models alexnet resnet50 --configs pairs-72
```
Expected: compat test runs (not skipped) and passes; alexnet CSS 0.0556 / resnet50 0.1667 on pairs-72, exactly.
The store's own gate is enforced in code: a run is reused only if a complete run with the identical
`eval_spec_id` exists and the spec is fully known (models config_id, pinned dataset revision, recognized
transform, default scorer, strict fp32). Anything else adds a new run; nothing is ever overwritten.

## 3. The sweep

```bash
uv run python scripts/run_doshi_sweep.py                    # canonical: both datasets -> s3://visionlab-evals
uv run python scripts/run_doshi_sweep.py --paper-pipeline   # replication check -> results/doshi_replication/ (not stored)
```
- TF32 is disabled by the script (strict fp32). Batch size 64, 8 workers; adjust with `--batch-size/--num-workers`.
- Largest models ~4.5 GB fp32 (eva_giant, dinov2_vitg14, vit_huge); 24 GB is plenty.
- Failures are collected and printed at the end; the sweep continues past them. Re-run to fill gaps.
- Each run prints a comparison against the paper and writes `results/doshi_sweep_{canonical,paper_pipeline}_{dataset}.csv`.

Where results land (Hive tree, identical in the bucket and the local mirror):
```
s3://visionlab-evals/eval=anagrams/version=0.1.0/dataset=<pairs-72|pairs-1440>/model=<slug>/readout=<slug>/intervention=<slug>/
    run=<UTC timestamp>-<8hex>/results.parquet   summary.json   manifest.json (written last = complete)
```

## 4. Expected differences from the paper (not bugs; see README "Known differences")

- Zero-shot SigLIP/SigLIP2: ~0.6% of images below the paper (paper's wrapper double-resized).
- Bicubic-native models (DINOv2, timm ViT/BEiT/ConvNeXt/CLIP-ft): 1–3 pairs off in the canonical run;
  the `--paper-pipeline` run should match to the image.
- `robust_resnet50_l2_eps_0_25`: the paper's row was computed with ε=0.1 weights; ours uses true ε=0.25.
- Random-init baselines use `seed=0` (recorded as `run_seed`); the paper's were unseeded.

## 5. Hand back

Send `results/doshi_sweep_*_pairs-72.csv` and `..._pairs-1440.csv` (and the failure list, if any) to the
eval-anagrams session. `store.query()` / DuckDB on the bucket (see README) give the same numbers.
