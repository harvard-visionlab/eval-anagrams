---
license: cc-by-4.0
pretty_name: Visual Anagrams (Object-Anagram Dataset)
task_categories:
- image-classification
tags:
- vision
- shape
- configural-shape
- holistic-shape
- visual-anagrams
- object-recognition
- neurips-2025
size_categories:
- 1K<n<10K
configs:
- config_name: pairs-72
  default: true
  data_files:
  - split: test
    path: pairs-72/test-*
- config_name: pairs-1440
  data_files:
  - split: test
    path: pairs-1440/test-*
dataset_info:
- config_name: pairs-1440
  features:
  - name: image
    dtype: image
  - name: filename
    dtype: string
  - name: anagram_id
    dtype: string
  - name: pair_id
    dtype: int32
  - name: variant
    dtype: int32
  - name: position
    dtype: int32
  - name: label
    dtype:
      class_label:
        names:
          '0': bear
          '1': bunny
          '2': cat
          '3': elephant
          '4': frog
          '5': lizard
          '6': tiger
          '7': turtle
          '8': wolf
  - name: foil
    dtype:
      class_label:
        names:
          '0': bear
          '1': bunny
          '2': cat
          '3': elephant
          '4': frog
          '5': lizard
          '6': tiger
          '7': turtle
          '8': wolf
  - name: object0
    dtype:
      class_label:
        names:
          '0': bear
          '1': bunny
          '2': cat
          '3': elephant
          '4': frog
          '5': lizard
          '6': tiger
          '7': turtle
          '8': wolf
  - name: object1
    dtype:
      class_label:
        names:
          '0': bear
          '1': bunny
          '2': cat
          '3': elephant
          '4': frog
          '5': lizard
          '6': tiger
          '7': turtle
          '8': wolf
  splits:
  - name: test
    num_bytes: 365948902
    num_examples: 2880
  download_size: 380330188
  dataset_size: 365948902
- config_name: pairs-72
  features:
  - name: image
    dtype: image
  - name: filename
    dtype: string
  - name: anagram_id
    dtype: string
  - name: pair_id
    dtype: int32
  - name: variant
    dtype: int32
  - name: position
    dtype: int32
  - name: label
    dtype:
      class_label:
        names:
          '0': bear
          '1': bunny
          '2': cat
          '3': elephant
          '4': frog
          '5': lizard
          '6': tiger
          '7': turtle
          '8': wolf
  - name: foil
    dtype:
      class_label:
        names:
          '0': bear
          '1': bunny
          '2': cat
          '3': elephant
          '4': frog
          '5': lizard
          '6': tiger
          '7': turtle
          '8': wolf
  - name: object0
    dtype:
      class_label:
        names:
          '0': bear
          '1': bunny
          '2': cat
          '3': elephant
          '4': frog
          '5': lizard
          '6': tiger
          '7': turtle
          '8': wolf
  - name: object1
    dtype:
      class_label:
        names:
          '0': bear
          '1': bunny
          '2': cat
          '3': elephant
          '4': frog
          '5': lizard
          '6': tiger
          '7': turtle
          '8': wolf
  splits:
  - name: test
    num_bytes: 18103502
    num_examples: 144
  download_size: 18105975
  dataset_size: 18103502
---

# Visual Anagrams — Object-Anagram Dataset

Stimuli for the **Configural Shape Score (CSS)** eval from

> Doshi, F. R., Fel, T., Konkle, T., & Alvarez, G. A. (2025). *Visual Anagrams Reveal Hidden
> Differences in Holistic Shape Processing Across Vision Models.* NeurIPS 2025.
> [arXiv:2507.00493](https://arxiv.org/abs/2507.00493) · [project page](https://www.fenildoshi.com/configural-shape/)

Each **anagram pair** is two 256×256 images (black-paint style on a grey background) built from the *same 16 square
puzzle pieces* (identical local texture, identical patch multiset), spatially rearranged to
depict two different animals. Local texture cues therefore cannot distinguish the two images;
recognizing both members of a pair requires sensitivity to the global arrangement of parts.

Nine categories: `bear, bunny, cat, elephant, frog, lizard, tiger, turtle, wolf`.
Images were synthesized with a visual-anagram diffusion pipeline (DeepFloyd-IF; Geng et al. 2024)
using the prompt *"high-quality painting of a well-drawn {animal} with simple black paint texture on a grey background"*.

## Configs

| config | images | anagram pairs | per class | notes |
|---|---|---|---|---|
| `pairs-72` (default) | 144 | 72 | 16 | main set (paper Fig. 2); every ordered pair of the 9 categories |
| `pairs-1440` | 2880 | 1440 | 320 | expanded set (paper Appendix A.10): 72 category pairs × 20 variants |

Both configs have a single `test` split. `pairs-1440` is a separate generation run and is **not** a superset of `pairs-72`.

```python
from datasets import load_dataset

ds = load_dataset("visionlab/visual-anagrams", "pairs-72", split="test")
# ds = load_dataset("visionlab/visual-anagrams", "pairs-1440", split="test")
row = ds[0]
row["image"]                  # PIL 256x256 RGB
ds.features["label"].names    # ['bear', 'bunny', ..., 'wolf']
```

## Columns

| column | type | description |
|---|---|---|
| `image` | Image | 256×256 RGB PNG |
| `filename` | string | original filename, e.g. `000_003_transform_bear_bunny_object1_bunny.png` |
| `anagram_id` | string | groups the two images of a pair (`"000"` in pairs-72, `"000_003"` in pairs-1440) |
| `pair_id` | int | 0–71, index of the ordered category pair `(object0, object1)` |
| `variant` | int | 0 in pairs-72; 0–19 in pairs-1440 |
| `position` | int | 0 or 1: which member of the pair this image is |
| `label` | ClassLabel(9) | ground-truth category of this image (`= [object0, object1][position]`) |
| `foil` | ClassLabel(9) | the *other* category in the pair |
| `object0`, `object1` | ClassLabel(9) | the pair's two categories, in filename order |

Every `anagram_id` has exactly two rows (`position` 0 and 1).

## Configural Shape Score (CSS)

For a classifier `f`, CSS is the fraction of anagram pairs for which **both** images are
classified correctly (9-way). Chance is 1/81 ≈ 1.2%. Single-image accuracy (chance 1/9) is
also informative but does not require configural processing.

For ImageNet-1k classifiers, 1000-way logits are mapped to the 9 categories by taking the
max logit over each category's ImageNet class indices (paper Table 2; `imagenet_class_map.json`
in this repo), followed by softmax/argmax:

| category | ImageNet indices |
|---|---|
| bear | 294, 295, 296, 297 |
| bunny | 330, 331, 332 |
| cat | 281, 282, 283, 284, 285 |
| elephant | 101, 385, 386 |
| frog | 30, 31, 32 |
| lizard | 38–48 |
| tiger | 286–293 |
| turtle | 33, 34, 35, 36, 37 |
| wolf | 269–275 |

Reference CSS on `pairs-72` (from the paper): AlexNet 0.06, ResNet-50 0.17, ViT-B/16 0.24,
DINOv2-B/14 0.61, DINOv2-G/14 0.78, EVA-CLIP-G 0.78, SigLIP2-L 0.82; humans ≈ 0.9.

A lightweight eval runner is provided at
[github.com/harvard-visionlab/eval-anagrams](https://github.com/harvard-visionlab/eval-anagrams).

## Extras

- `animations/*.mp4` — 72 videos (one per `pairs-72` pair) morphing between the two arrangements.
- `imagenet_class_map.json` — the 9-category → ImageNet index mapping above.

## Citation

```bibtex
@inproceedings{doshi2025visualanagrams,
  title     = {Visual Anagrams Reveal Hidden Differences in Holistic Shape Processing Across Vision Models},
  author    = {Doshi, Fenil R. and Fel, Thomas and Konkle, Talia and Alvarez, George A.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2507.00493}
}
```

## License

CC-BY-4.0. Images are synthetic (diffusion-generated).
