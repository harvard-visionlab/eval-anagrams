"""Phase-1 validation: run anagram_eval on models spanning "all flavors" and compare to Doshi's numbers.

Run on the GPU workstation:
    uv sync --extra validation
    uv run python scripts/run_validation.py                                   # all models, pairs-72
    uv run python scripts/run_validation.py --configs pairs-72 pairs-1440     # both sets
    uv run python scripts/run_validation.py --models alexnet siglip2_l16_256  # subset

Outputs: results/validation/<config>/<model>/{summary.json,predictions.csv}
         results/validation/comparison.csv  (+ printed table: ours vs Doshi, delta in #pairs)

Preprocessing: Resize((224,224)) with each model's own mean/std — equivalent to Doshi's pipeline
(fixed Resize((224,224)); timm/SigLIP wrappers re-normalized with model stats internally).

SigLIP: by default the native 256px image goes straight into open_clip's preprocess. Doshi's wrapper
resized 256->224 (bilinear), quantized to uint8, then open_clip upsampled to 256 (bicubic); that blur
costs ~0.6% of images (-1 pair on pairs-72, -17 on pairs-1440). --paper-pipeline reproduces it exactly.

Numerics: TF32 is disabled by default so GPU results match fp32/CPU (and the paper) exactly; with
TF32 on, images with near-zero decision margin can flip (e.g. ResNet-50 cat/turtle 022, |dm| = 0.004).
Pass --tf32 to allow it.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torchvision import transforms as T
from visionlab.evals.anagrams import ZeroShotClassifier, anagram_eval

ROOT = Path(__file__).resolve().parents[1]
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def paper_transform(mean=IMAGENET_MEAN, std=IMAGENET_STD, size=224):
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(mean, std)])


# ---------------------------------------------------------------------------------------------
# model builders: name -> (doshi_model_name, builder) ; builder() -> (model, transform)
# ---------------------------------------------------------------------------------------------
def build_alexnet():
    from torchvision.models import AlexNet_Weights, alexnet
    return alexnet(weights=AlexNet_Weights.IMAGENET1K_V1), paper_transform()


def build_resnet50():
    from torchvision.models import ResNet50_Weights, resnet50
    return resnet50(weights=ResNet50_Weights.IMAGENET1K_V1), paper_transform()


def build_vit_b_16():
    from torchvision.models import ViT_B_16_Weights, vit_b_16
    return vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1), paper_transform()


def build_timm(name):
    def _build():
        import timm
        model = timm.create_model(name, pretrained=True)
        cfg = timm.data.resolve_data_config({}, model=model)
        return model, paper_transform(cfg["mean"], cfg["std"], size=cfg["input_size"][-1])
    return _build


def build_dinov2_lc(hub_name):
    def _build():
        model = torch.hub.load("facebookresearch/dinov2", hub_name)
        return model, paper_transform()
    return _build


PAPER_PIPELINE = False  # set by --paper-pipeline


def build_siglip(open_clip_name):
    def _build():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 1000 ImageNet names, bare prompt -> standard 1000->9 mapping (as in Doshi et al.)
        classifier, preprocess = ZeroShotClassifier.from_open_clip(open_clip_name, device=device)
        if PAPER_PIPELINE:
            # Doshi's SigLIP wrapper received Resize((224,224))-bilinear ImageNet-normalized tensors,
            # denormalized them to uint8 PIL, then re-ran open_clip's preprocess (256 bicubic).
            # This double resize reproduces the paper's SigLIP numbers exactly (css .8194 on pairs-72).
            preprocess = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.ToPILImage(), preprocess])
        return classifier, preprocess  # default: native 256 straight into open_clip's own preprocess
    return _build


MODELS = {
    # supervised torchvision
    "alexnet": ("alexnet_in1k_v1_7be5be79", build_alexnet),
    "resnet50": ("resnet50_in1k", build_resnet50),
    "vit_b_16": ("vitb16_in1k", build_vit_b_16),
    # timm with ImageNet heads (supervised + self-supervised fine-tuned)
    "timm_vit_b16_augreg": ("vit_base_patch16_224_augreg_in1k", build_timm("vit_base_patch16_224.augreg_in1k")),
    "timm_beitv2_b16": ("beitv2_base_patch16_224_in1k_ft_in1k", build_timm("beitv2_base_patch16_224.in1k_ft_in1k")),
    # hub model with linear-classifier head
    "dinov2_vitb14_lc": ("dinov2_vitb14_lc", build_dinov2_lc("dinov2_vitb14_lc")),
    # language-aligned zero-shot
    "siglip2_l16_256": ("siglip2_256_vitl16_zeroshot", build_siglip("hf-hub:timm/ViT-L-16-SigLIP2-256")),
}


def load_reference(config):
    path = ROOT / "reference" / f"doshi_css_{config.replace('-', '')}.csv"
    return pd.read_csv(path).set_index("model_name")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--configs", nargs="*", default=["pairs-72"], choices=["pairs-72", "pairs-1440"])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "validation")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="recompute even if results exist")
    ap.add_argument("--tf32", action="store_true",
                    help="allow TF32 matmul/conv on Ampere+ GPUs (default: strict fp32)")
    ap.add_argument("--paper-pipeline", action="store_true",
                    help="emulate Doshi's SigLIP double-resize preprocessing (parity check; not the recommended default)")
    args = ap.parse_args()

    global PAPER_PIPELINE
    PAPER_PIPELINE = args.paper_pipeline

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | torch {torch.__version__}")
    rows = []
    for config in args.configs:
        ref = load_reference(config)
        for name in args.models:
            doshi_name, builder = MODELS[name]
            out_dir = args.out / config / name
            t0 = time.time()
            if (out_dir / "summary.json").exists() and not args.force:
                from visionlab.evals.anagrams import AnagramResults
                res = AnagramResults.load(out_dir)
                status = "cached"
            else:
                try:
                    model, transform = builder()
                except Exception as e:  # keep going: report the failure in the table
                    print(f"[{config}] {name}: BUILD FAILED: {type(e).__name__}: {e}", file=sys.stderr)
                    rows.append(dict(config=config, model=name, doshi_name=doshi_name, status=f"build failed: {e}"))
                    continue
                res = anagram_eval(model, transform, config=config, batch_size=args.batch_size, device=device,
                                   num_workers=args.num_workers, model_name=name, doshi_name=doshi_name)
                res.save(out_dir)
                del model
                gc.collect()
                torch.cuda.empty_cache() if device == "cuda" else None
                status = f"{time.time() - t0:.0f}s"
            s = res.summary
            doshi_css = ref["css"].get(doshi_name, float("nan"))
            doshi_acc = ref["acc"].get(doshi_name, float("nan"))
            rows.append(dict(
                config=config, model=name, doshi_name=doshi_name, status=status,
                css=s["css"], doshi_css=doshi_css, delta_pairs=round((s["css"] - doshi_css) * s["n_pairs"], 1),
                acc=s["acc"], doshi_acc=doshi_acc, delta_images=round((s["acc"] - doshi_acc) * s["n_images"], 1),
                foil_rate=s["foil_rate"], dm_mean=s["dm_mean"],
            ))
            print(f"[{config}] {name:<22} css {s['css']:.3f} (doshi {doshi_css:.3f})  "
                  f"acc {s['acc']:.3f} (doshi {doshi_acc:.3f})  [{status}]")

    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "comparison.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nsaved {args.out / 'comparison.csv'}")


if __name__ == "__main__":
    main()
