"""Release metadata has to agree with itself and with the tag.

Zenodo builds its record from `.zenodo.json`, so a stale `version` there is
published under the new release's DOI with the old version number — silently,
and the DOI cannot be withdrawn afterwards. It happened once; this pins it.
"""
import json
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _zenodo():
    with open(os.path.join(ROOT, ".zenodo.json"), encoding="utf-8") as f:
        return json.load(f)


def _citation_field(name):
    with open(os.path.join(ROOT, "CITATION.cff"), encoding="utf-8") as f:
        m = re.search(r"^%s:\s*'?\"?([^'\"\n]+)'?\"?\s*$" % name, f.read(), re.M)
    assert m, f"{name} missing from CITATION.cff"
    return m.group(1).strip()


def test_zenodo_and_citation_agree_on_the_version():
    assert _zenodo()["version"] == _citation_field("version")


def test_version_looks_like_a_release():
    assert re.fullmatch(r"\d+\.\d+\.\d+", _zenodo()["version"])


def test_version_matches_the_most_recent_tag_when_one_is_reachable():
    """Skipped on a shallow CI checkout that carries no tags."""
    try:
        tag = subprocess.run(["git", "describe", "--tags", "--abbrev=0"],
                             cwd=ROOT, capture_output=True, timeout=30,
                             text=True)
    except (OSError, subprocess.SubprocessError):
        return
    if tag.returncode != 0 or not tag.stdout.strip():
        return                      # no tags fetched here; nothing to compare
    assert tag.stdout.strip().lstrip("v") == _zenodo()["version"]


def test_citation_still_points_at_the_concept_doi():
    """The concept DOI always resolves to the newest release, so it is the one
    a paper should cite; a version DOI here would freeze citations at 1.0.x."""
    assert _citation_field("doi") == "10.5281/zenodo.22307137"
