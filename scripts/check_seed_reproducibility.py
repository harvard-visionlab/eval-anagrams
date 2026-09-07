"""Release-gate check: seeded random init is reproducible ON THIS MACHINE (George, 2026-09-07).

Builds each model twice with the same seed and asserts identical state digests and bit-identical outputs on a
fixed input, once more with a different seed and asserts they differ, and runs identity.verify_state(model).
Do not rely on the models repo's own tests for this: they prove it on one machine / torch version only.

    uv run python scripts/check_seed_reproducibility.py       # the 4 Doshi random-init baselines
    uv run python scripts/check_seed_reproducibility.py --specs pytorch/alexnet:NONE facebook/dinov2_vits14:NONE@none
"""

from __future__ import annotations

import argparse
import hashlib
import sys

import torch

DEFAULT_SPECS = ["pytorch/alexnet:NONE", "pytorch/vgg16:NONE", "pytorch/resnet50:NONE", "pytorch/resnet101:NONE"]


def state_digest(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--specs", nargs="*", default=DEFAULT_SPECS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from visionlab.models import load_model

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(1234)).to(device)
    failures = 0
    for spec in args.specs:
        m1, _, i1 = load_model(spec, seed=args.seed)
        m2, _, i2 = load_model(spec, seed=args.seed)
        m3, _, i3 = load_model(spec, seed=args.seed + 1)
        for ident, model in ((i1, m1), (i2, m2), (i3, m3)):
            if hasattr(ident, "verify_state"):
                ident.verify_state(model)  # raises on init-implementation drift
        same_state = state_digest(m1) == state_digest(m2)
        with torch.inference_mode():
            same_out = torch.equal(m1.to(device).eval()(x), m2.to(device).eval()(x))
            diff_out = not torch.equal(m1(x), m3.to(device).eval()(x))
        ok = same_state and same_out and diff_out and str(i1) == str(i2) and str(i1) != str(i3)
        failures += not ok
        print(
            f"{'OK  ' if ok else 'FAIL'} {spec:<40} {i1!s:<36} same_state={same_state} same_out={same_out} "
            f"diff_seed_differs={diff_out} device={device} torch={torch.__version__}"
        )
    if failures:
        print(f"\n{failures} spec(s) FAILED seeded reproducibility on this machine — do not run random-init baselines.")
        sys.exit(1)
    print("\nseeded random init is reproducible on this machine.")


if __name__ == "__main__":
    main()
