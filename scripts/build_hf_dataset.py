"""Build the visionlab/visual-anagrams HuggingFace dataset from Doshi's source folders.

Source layout (neurips_datasets/):
    anagram_stimulus_3_jigsaw/images/*.png                    -> config "pairs-72"   (144 images)
    anagram_stimulus_3_jigsaw/animations/*.mp4                -> raw files under animations/
    anagram_stimulus_3_expanded_post_rebuttal/images/*.png    -> config "pairs-1440" (2880 images)

Every filename encodes the full annotation:
    {pair_id:03d}[_{variant:03d}]_transform_{object0}_{object1}_object{position}_{label}.png

Usage:
    uv run python scripts/build_hf_dataset.py --source <path>/neurips_datasets --show
    uv run python scripts/build_hf_dataset.py --source ... --push            # private repo
    uv run python scripts/build_hf_dataset.py --source ... --push --animations
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import ClassLabel, Dataset, Features, Image, Value

REPO_ID = "visionlab/visual-anagrams"

CLASSES = ["bear", "bunny", "cat", "elephant", "frog", "lizard", "tiger", "turtle", "wolf"]

# 9 anagram categories -> ImageNet-1k class indices (paper Table 2; Baker & Elder 2022 mapping)
IMAGENET_CLASS_MAP = {
    "bear": [294, 295, 296, 297],
    "bunny": [330, 331, 332],
    "cat": [281, 282, 283, 284, 285],
    "elephant": [101, 385, 386],
    "frog": [30, 31, 32],
    "lizard": [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
    "tiger": [286, 287, 288, 289, 290, 291, 292, 293],
    "turtle": [33, 34, 35, 36, 37],
    "wolf": [269, 270, 271, 272, 273, 274, 275],
}

CONFIGS = {
    "pairs-72": "anagram_stimulus_3_jigsaw",
    "pairs-1440": "anagram_stimulus_3_expanded_post_rebuttal",
}

FILENAME_RE = re.compile(
    r"^(?P<pair_id>\d{3})(?:_(?P<variant>\d{3}))?_transform_"
    r"(?P<object0>[a-z]+)_(?P<object1>[a-z]+)_object(?P<position>[01])_(?P<label>[a-z]+)\.png$"
)

label_feature = ClassLabel(names=CLASSES)
FEATURES = Features(
    {
        "image": Image(),
        "filename": Value("string"),
        "anagram_id": Value("string"),
        "pair_id": Value("int32"),
        "variant": Value("int32"),
        "position": Value("int32"),
        "label": label_feature,
        "foil": label_feature,
        "object0": label_feature,
        "object1": label_feature,
    }
)


def parse_filename(name: str) -> dict:
    m = FILENAME_RE.match(name)
    if m is None:
        raise ValueError(f"unexpected filename: {name}")
    d = m.groupdict()
    pair_id, variant, position = int(d["pair_id"]), int(d["variant"] or 0), int(d["position"])
    objects = [d["object0"], d["object1"]]
    label = d["label"]
    assert label == objects[position], f"label/position mismatch: {name}"
    anagram_id = f"{pair_id:03d}" if d["variant"] is None else f"{pair_id:03d}_{variant:03d}"
    return dict(
        filename=name,
        anagram_id=anagram_id,
        pair_id=pair_id,
        variant=variant,
        position=position,
        label=label,
        foil=objects[1 - position],
        object0=objects[0],
        object1=objects[1],
    )


def build_split(image_dir: Path) -> Dataset:
    files = sorted(p for p in image_dir.iterdir() if p.suffix == ".png")
    rows = [dict(image=str(p), **parse_filename(p.name)) for p in files]
    ds = Dataset.from_list(rows).cast(FEATURES)
    validate(ds)
    return ds


def validate(ds: Dataset) -> None:
    df = ds.remove_columns("image").to_pandas()
    names = ds.features["label"].names
    counts = df.groupby("anagram_id").size()
    assert (counts == 2).all(), "every anagram_id must have exactly 2 images"
    assert set(df.groupby("anagram_id")["position"].sum()) == {1}, "positions must be {0,1} per pair"
    assert df["label"].nunique() == 9 and df["label"].value_counts().nunique() == 1, "unbalanced classes"
    for _, r in df.iterrows():
        objs = [r.object0, r.object1]
        assert r.label == objs[r.position] and r.foil == objs[1 - r.position]
    assert set(zip(df.object0, df.object1)) == {(a, b) for a in range(9) for b in range(9) if a != b}
    print(f"  ok: {len(df)} images, {len(counts)} pairs, {len(df) // 9} per class ({', '.join(names)})")


def show_sample(ds: Dataset, filename: str) -> None:
    idx = ds["filename"].index(filename)
    row = ds[idx]
    img = row.pop("image")
    print(f"\nsample row for {filename} (index {idx}):")
    print(f"  image: PIL {img.mode} {img.size}")
    for k, v in row.items():
        extra = f"  -> '{ds.features[k].int2str(v)}'" if isinstance(ds.features[k], ClassLabel) else ""
        print(f"  {k}: {v!r}{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True, help="path to neurips_datasets/")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--show", action="store_true", help="print a sample row")
    ap.add_argument("--push", action="store_true", help="push configs to the hub (private)")
    ap.add_argument("--animations", action="store_true", help="upload mp4 animations as raw files")
    ap.add_argument("--public", action="store_true", help="create as public (default: private)")
    args = ap.parse_args()

    for config in args.configs:
        image_dir = args.source / CONFIGS[config] / "images"
        print(f"[{config}] building from {image_dir}")
        ds = build_split(image_dir)
        if args.show:
            show_sample(ds, ds[0]["filename"])
        if args.push:
            print(f"[{config}] pushing to {REPO_ID} ...")
            ds.push_to_hub(
                REPO_ID,
                config_name=config,
                split="test",
                private=not args.public,
                commit_message=f"add {config} ({len(ds)} images)",
            )

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        map_path = Path(__file__).with_name("imagenet_class_map.json")
        map_path.write_text(json.dumps(IMAGENET_CLASS_MAP, indent=2) + "\n")
        api.upload_file(
            path_or_fileobj=str(map_path),
            path_in_repo="imagenet_class_map.json",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="add 9-class -> ImageNet index map",
        )
        if args.animations:
            anim_dir = args.source / CONFIGS["pairs-72"] / "animations"
            print(f"uploading {len(list(anim_dir.glob('*.mp4')))} animations ...")
            api.upload_folder(
                folder_path=str(anim_dir),
                path_in_repo="animations",
                repo_id=REPO_ID,
                repo_type="dataset",
                allow_patterns=["*.mp4"],
                commit_message="add 72 anagram animations (mp4)",
            )


if __name__ == "__main__":
    main()
