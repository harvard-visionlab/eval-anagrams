# Reference numbers from Doshi et al. (NeurIPS 2025)

Per-model results copied from the authors' repo (`anagram_holistic_shape_neurips`,
`evals/*/results/anagram_perceptshift/anagram_perceptshift_benchmark.csv`).

- `doshi_css_pairs72.csv` — 91 models on the 72-pair set (paper Fig. 2A / Fig. 11B)
- `doshi_css_pairs1440.csv` — same models on the 1440-pair expanded set (Fig. 11A)

Columns: `acc` = single-image 9-way accuracy; `css` = Configural Shape Score (`global_pair_acc`);
`css_err_low/high` = bootstrap error-bar *lengths* below/above `css` (not CI bounds);
`target_foil_bias` = legacy sqrt(target_or_foil_acc * target_over_foil_acc).

Human (n=4, 72 pairs): acc 0.948, css 0.896.
