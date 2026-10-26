"""Lab repos are pinned over https (lab standard, George 2026-10-26): machines authenticate with the GitHub CLI
(`gh auth login` + `gh auth setup-git`); ssh pins fail for members without keys."""

import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
LAB_SSH = re.compile(r"(ssh://)?git@github\.com[:/]harvard-visionlab/")


def _urls(doc: dict) -> list[str]:
    proj = doc.get("project", {})
    deps = list(proj.get("dependencies", []))
    deps += [d for ds in proj.get("optional-dependencies", {}).values() for d in ds]
    deps += [d for ds in doc.get("dependency-groups", {}).values() for d in ds if isinstance(d, str)]
    srcs = doc.get("tool", {}).get("uv", {}).get("sources", {}).values()
    deps += [s["git"] for src in srcs for s in (src if isinstance(src, list) else [src]) if "git" in s]
    return deps


def ssh_lab_pins(doc: dict) -> list[str]:
    return [u for u in _urls(doc) if LAB_SSH.search(u)]


def test_lab_repos_are_pinned_over_https():
    bad = ssh_lab_pins(tomllib.loads(PYPROJECT.read_text()))
    assert not bad, f"pin these lab repos as https://github.com/harvard-visionlab/<repo>.git: {bad}"


def test_the_check_catches_ssh_pins_in_every_place():
    doc = {
        "project": {"dependencies": ["x @ git+ssh://git@github.com/harvard-visionlab/model-atoms.git@v0.1.0",
                                     "y @ git+https://github.com/harvard-visionlab/datasets.git@v0.19.1"],
                    "optional-dependencies": {"e": ["z @ git+ssh://git@github.com/harvard-visionlab/model-genesis@main"]}},
        "tool": {"uv": {"sources": {"a": {"git": "git@github.com:harvard-visionlab/model-evals.git", "tag": "v0.2.0"},
                                    "c": {"git": "https://github.com/harvard-visionlab/slipstream.git"}}}},
    }  # fmt: skip
    assert len(ssh_lab_pins(doc)) == 3
