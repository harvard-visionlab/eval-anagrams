"""Private lab repos are pinned over ssh: an https pin needs a GitHub token, which the cluster and the workstation
containers do not have (model-rearing pinned model-evals over https until 2026-10-26 and machina could not install its venv).
Public lab repos may use https, so anyone can install them without credentials."""

import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
PUBLIC = {"slipstream", "datasets"}  # github.com/harvard-visionlab repos readable without credentials (checked 2026-10-26)
LAB_HTTPS = re.compile(r"https://github\.com/harvard-visionlab/(?P<repo>[A-Za-z0-9._-]+?)(?:\.git)?(?:[@/#]|$)")


def _urls(doc: dict) -> list[str]:
    proj = doc.get("project", {})
    deps = list(proj.get("dependencies", []))
    deps += [d for ds in proj.get("optional-dependencies", {}).values() for d in ds]
    deps += [d for ds in doc.get("dependency-groups", {}).values() for d in ds if isinstance(d, str)]
    srcs = doc.get("tool", {}).get("uv", {}).get("sources", {}).values()
    deps += [s["git"] for src in srcs for s in (src if isinstance(src, list) else [src]) if "git" in s]
    return deps


def https_private_pins(doc: dict) -> list[str]:
    return [u for u in _urls(doc) if (m := LAB_HTTPS.search(u)) and m["repo"] not in PUBLIC]


def test_private_lab_repos_are_pinned_over_ssh():
    bad = https_private_pins(tomllib.loads(PYPROJECT.read_text()))
    assert not bad, f"pin these private lab repos as ssh://git@github.com/harvard-visionlab/<repo>.git: {bad}"


def test_the_check_catches_https_private_pins_in_every_place():
    doc = {
        "project": {"dependencies": ["x @ git+https://github.com/harvard-visionlab/model-atoms.git@v0.1.0",
                                     "y @ git+https://github.com/harvard-visionlab/datasets.git@v0.19.1"],
                    "optional-dependencies": {"e": ["z @ git+https://github.com/harvard-visionlab/model-genesis@main"]}},
        "tool": {"uv": {"sources": {"a": {"git": "https://github.com/harvard-visionlab/model-evals.git", "tag": "v0.2.0"},
                                    "b": {"git": "ssh://git@github.com/harvard-visionlab/model-rearing.git"},
                                    "c": {"git": "https://github.com/harvard-visionlab/slipstream.git"}}}},
    }  # fmt: skip
    bad = https_private_pins(doc)
    assert [LAB_HTTPS.search(u)["repo"] for u in bad] == ["model-atoms", "model-genesis", "model-evals"]
