"""Phase-2: evaluate the full Doshi2025 model collection through visionlab.models and store results.

    uv sync --dev --extra models
    uv run python scripts/run_doshi_sweep.py                           # all tagged models, both datasets
    uv run python scripts/run_doshi_sweep.py --datasets pairs-72 --models pytorch/alexnet:DEFAULT ...
    uv run python scripts/run_doshi_sweep.py --local                   # write only to the local mirror

Results land in s3://visionlab-evals/eval=anagrams/version=<v>/dataset=<d>/model=<slug>/readout=<slug>/ (skip if
present, --force to recompute). Comparison against the paper joins on the paper's model name, taken from
`identity.collection_names["Doshi2025"]` (models repo collection file) with reference/doshi_model_map.csv
(columns: model_id, doshi_name) as a manual fallback/override.

Known, expected differences from the paper (see README "Known differences"): zero-shot SigLIP/CLIP models
sit ~0.6% of images lower (their double-resize preprocessing); ±1 image on convnets unless TF32 is off
(it is off here).
"""

from __future__ import annotations

import argparse
import gc
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
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from visionlab.models import list_models

    # str() of an identity is 'source/name:hashid[@readout]' — a loadable spec
    specs = args.models or [str(m) for m in list_models(tags="Doshi2025", include_untrained=True)]
    store = ResultsStore(bucket=None) if args.local else ResultsStore()
    print(f"{len(specs)} models x {args.datasets} -> {store.s3_uri() if store.bucket else store.cache_dir}")

    failures = []
    for dataset in args.datasets:
        for spec in specs:
            try:
                res = store.run(spec, dataset=dataset, force=args.force, batch_size=args.batch_size,
                                num_workers=args.num_workers, progress=False)
                s = res.summary
                print(f"[{dataset}] {s['model_id']:<40} css {s['css']:.3f} acc {s['acc']:.3f}")
            except Exception as e:  # keep sweeping; report at the end
                print(f"[{dataset}] {spec}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
                failures.append((dataset, spec, repr(e)))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for dataset in args.datasets:
        compare(store, dataset)
    if failures:
        print(f"\n{len(failures)} failures:")
        for f in failures:
            print("  ", *f)


def paper_name_map() -> pd.DataFrame:
    """model_id -> doshi_name from the models repo collection, overridden by reference/doshi_model_map.csv."""
    rows = {}
    try:
        from visionlab.models import list_models

        for ident in list_models(tags="Doshi2025", include_untrained=True):
            name = (getattr(ident, "collection_names", None) or {}).get("Doshi2025")
            if name:
                rows[ident.model_id] = name
    except Exception as e:  # models repo without collection names yet
        print(f"(collection names unavailable: {type(e).__name__}: {e})")
    map_path = ROOT / "reference" / "doshi_model_map.csv"
    if map_path.exists():
        rows.update(pd.read_csv(map_path).set_index("model_id")["doshi_name"].to_dict())
    return pd.DataFrame({"model_id": list(rows), "doshi_name": list(rows.values())})


def compare(store: ResultsStore, dataset: str):
    """Ours vs the paper (primary readouts only)."""
    names = paper_name_map()
    if names.empty:
        print(f"\n[{dataset}] no paper-name mapping available; skipping comparison")
        return
    ours = store.query(dataset=dataset)
    ours = ours[ours["readout_primary"].fillna(True).astype(bool)]
    ref = pd.read_csv(ROOT / "reference" / f"doshi_css_{dataset.replace('-', '')}.csv")
    m = names.merge(ours[["model_id", "css", "acc", "n_pairs"]], on="model_id")
    ref = ref.rename(columns={"model_name": "doshi_name", "css": "doshi_css", "acc": "doshi_acc"})
    m = m.merge(ref, on="doshi_name")
    m["delta_pairs"] = ((m["css"] - m["doshi_css"]) * m["n_pairs"]).round(1)
    r = m[["css", "doshi_css"]].corr().iloc[0, 1]
    print(f"\n[{dataset}] {len(m)} models matched to the paper; r = {r:.3f}; |delta| > 2 pairs:")
    flagged = m.loc[m["delta_pairs"].abs() > 2, ["model_id", "doshi_name", "css", "doshi_css", "delta_pairs"]]
    print(flagged.to_string(index=False))
    m.to_csv(ROOT / "results" / f"doshi_sweep_comparison_{dataset}.csv", index=False)


if __name__ == "__main__":
    main()
