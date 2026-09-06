"""Phase-2: evaluate the full Doshi2025 model collection through visionlab.models and store results.

    uv sync --dev --extra models
    uv run python scripts/run_doshi_sweep.py                           # all tagged models, both datasets
    uv run python scripts/run_doshi_sweep.py --datasets pairs-72 --models pytorch/alexnet:DEFAULT ...
    uv run python scripts/run_doshi_sweep.py --local                   # write only to the local mirror
    uv run python scripts/run_doshi_sweep.py --paper-pipeline          # replication check: NOT stored (see below)

Results land in s3://visionlab-evals/eval=anagrams/version=<v>/dataset=<d>/model=<slug>/readout=<slug>/ (skip if
present, --force to recompute). Comparison against the paper joins on the paper's model name, which the store
records from `identity.collection_names["Doshi2025"]` (models repo collection file);
reference/doshi_model_map.csv (columns: model_id, doshi_name) is a manual override.

Canonical results use each model's own preprocessing (`transforms.test`: native input size, interpolation,
stats). The paper resized every model's input with torchvision's default *bilinear* Resize((224,224)), so
bicubic-native models (DINOv2, timm ViTs/BEiT/ConvNeXt, ...) land 1-3 pairs away from the paper, and
zero-shot SigLIP models ~0.6% of images away (their wrapper's double resize). `--paper-pipeline` swaps in
Resize((224,224)) bilinear + the card's stats to check that models/weights are identical to the paper's;
those runs are written to results/doshi_replication/ only, never to the store. TF32 is off.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from visionlab.evals.anagrams import ResultsStore

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None,
                    help="visionlab specs; default: list_models(tags=['Doshi2025'])")
    ap.add_argument("--datasets", nargs="*", default=["pairs-72", "pairs-1440"], choices=["pairs-72", "pairs-1440"])
    ap.add_argument("--local", action="store_true", help="local mirror only, no S3")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--paper-pipeline", action="store_true",
                    help="bilinear Resize((224,224)) + card stats; written to results/doshi_replication/ only")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from visionlab.models import get_collection

    # collection ids are loadable specs: 'source/name:hashid[@readout_hashid]' (readout members carry their @hash)
    specs = args.models or list(get_collection("Doshi2025"))
    store = ResultsStore(bucket=None) if args.local else ResultsStore()
    print(f"{len(specs)} models x {args.datasets} -> {store.s3_uri() if store.bucket else store.cache_dir}")

    if args.paper_pipeline:
        store = ResultsStore(bucket=None, cache_dir=ROOT / "results" / "doshi_replication")
        print("paper-pipeline mode: bilinear Resize((224,224)) + card stats; writing to", store.cache_dir)

    failures = []
    for dataset in args.datasets:
        for spec in specs:
            try:
                transform = paper_transform(spec) if args.paper_pipeline else None
                res = store.run(spec, transform=transform, dataset=dataset, force=args.force,
                                batch_size=args.batch_size, num_workers=args.num_workers, progress=False)
                s = res.summary
                print(f"[{dataset}] {s['model_id']:<40} css {s['css']:.3f} acc {s['acc']:.3f}")
            except Exception as e:  # keep sweeping; report at the end
                print(f"[{dataset}] {spec}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
                failures.append((dataset, spec, repr(e)))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for dataset in args.datasets:
        compare(store, dataset, tag="paper_pipeline" if args.paper_pipeline else "canonical")
    if failures:
        print(f"\n{len(failures)} failures:")
        for f in failures:
            print("  ", *f)


def paper_transform(spec: str):
    """Doshi et al.'s preprocessing: torchvision Resize((224,224)) (bilinear) + the model's own mean/std."""
    from torchvision import transforms as T
    from visionlab.models import load_transforms

    stats = load_transforms(spec).stats
    return T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(stats.mean, stats.std)])


def paper_name_map(ours: pd.DataFrame) -> pd.DataFrame:
    """model_id -> doshi_name: from the stored summaries' collection_names (set by the models repo's
    collection file), overridden by reference/doshi_model_map.csv when present."""
    rows = {}
    for model_id, names in zip(ours["model_id"], ours.get("collection_names", pd.Series(dtype=str))):
        try:
            name = json.loads(names).get("Doshi2025") if isinstance(names, str) else None
        except json.JSONDecodeError:
            name = None
        if name:
            rows[model_id] = name
    map_path = ROOT / "reference" / "doshi_model_map.csv"
    if map_path.exists():
        rows.update(pd.read_csv(map_path).set_index("model_id")["doshi_name"].to_dict())
    return pd.DataFrame({"model_id": list(rows), "doshi_name": list(rows.values())})


def compare(store: ResultsStore, dataset: str, tag: str = "canonical"):
    """Ours vs the paper (primary readouts only)."""
    ours = store.query(dataset=dataset)
    if ours.empty:
        return
    ours = ours[ours["readout_primary"].fillna(True).astype(bool)]
    names = paper_name_map(ours)
    if names.empty:
        print(f"\n[{dataset}] no paper-name mapping available; skipping comparison")
        return
    ref = pd.read_csv(ROOT / "reference" / f"doshi_css_{dataset.replace('-', '')}.csv")
    m = names.merge(ours[["model_id", "css", "acc", "n_pairs"]], on="model_id")
    ref = ref.rename(columns={"model_name": "doshi_name", "css": "doshi_css", "acc": "doshi_acc"})
    m = m.merge(ref, on="doshi_name")
    m["delta_pairs"] = ((m["css"] - m["doshi_css"]) * m["n_pairs"]).round(1)
    r = m[["css", "doshi_css"]].corr().iloc[0, 1]
    print(f"\n[{dataset}] {len(m)} models matched to the paper; r = {r:.3f}; |delta| > 2 pairs:")
    flagged = m.loc[m["delta_pairs"].abs() > 2, ["model_id", "doshi_name", "css", "doshi_css", "delta_pairs"]]
    print(flagged.to_string(index=False))
    out = ROOT / "results" / f"doshi_sweep_{tag}_{dataset}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
