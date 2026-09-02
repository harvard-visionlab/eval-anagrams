# Doshi et al. 2025 Visual Anagrams Holistic Shape Eval

# Goals

- add the visual anagrams datasets to huggingface (public dataset) under visionlab org
- we need to design the format / structure; e.g., where/what are the labels here?

```
from datasets import load_dataset
dataset = load_dataset("visionlab/visual-anagrams-jigsaw-144") # or ... visual-anagrams-jigsaw-2880
```

- create an easily installed eval script, visionlab namespaced, for running the eval, something like this:

```
from visionlab.evals import anagram_eval
from visionlab.models import load_model

model, transforms = load_model('pytorch/alexnet:DEFAULT')
results = anagram_eval(model, transforms['test_transform'])
```
