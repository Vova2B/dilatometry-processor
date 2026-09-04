"""cu_calibration_builder generalisation (cu_runs.json run spec):

* --help must print usage and exit 0 — the README documents that exact
  command, and the pre-argparse main() fell through to a full run and died
  with a FileNotFoundError.
* the spec loader rejects malformed specs with the offending field named;
* the auto-derived eq7/hysteresis pair defaults reproduce the authors'
  explicit pairs on the shipped 15-record registry;
* an end-to-end run on a synthetic Cu file writes a registry keyed by the
  user's chosen key, with spec-relative source_file paths.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)

import cu_calibration_builder as cu  # noqa: E402

BUILDER = os.path.join(SCRIPTS, "cu_calibration_builder.py")
EXAMPLE = os.path.join(SCRIPTS, "cu_runs.example.json")
REGISTRY = os.path.join(SCRIPTS, "calibrations.json")


def test_help_exits_zero_and_mentions_the_spec():
    p = subprocess.run([sys.executable, BUILDER, "--help"],
                       capture_output=True, text=True)
    assert p.returncode == 0
    assert "cu_runs.json" in p.stdout
    assert "--data" in p.stdout


def _write_spec(tmp_path, spec):
    path = tmp_path / "cu_runs.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def test_spec_missing_is_a_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="no Cu-run spec"):
        cu.load_runs_spec(None, str(tmp_path))
    with pytest.raises(SystemExit, match="file not found"):
        cu.load_runs_spec(str(tmp_path / "nope.json"), str(tmp_path))


def test_spec_field_errors_are_named(tmp_path):
    bad = {"runs": [{"key": "a", "path": "x.dat", "cell": "c"}]}  # no length
    with pytest.raises(SystemExit, match="cu_length_mm"):
        cu.load_runs_spec(str(_write_spec(tmp_path, bad)), str(tmp_path))
    dup = {"runs": [
        {"key": "a", "path": "x.dat", "cell": "c", "cu_length_mm": 1},
        {"key": "a", "path": "y.dat", "cell": "c", "cu_length_mm": 2}]}
    with pytest.raises(SystemExit, match="duplicate run key 'a'"):
        cu.load_runs_spec(str(_write_spec(tmp_path, dup)), str(tmp_path))
    empty = {"runs": []}
    with pytest.raises(SystemExit, match="empty"):
        cu.load_runs_spec(str(_write_spec(tmp_path, empty)), str(tmp_path))
    badpair = {"runs": [{"key": "a", "path": "x.dat", "cell": "c",
                         "cu_length_mm": 1}],
               "transfer_pairs": [["a/c1w", "b/c1w", "sideways"]]}
    with pytest.raises(SystemExit, match="same-length"):
        cu.load_runs_spec(str(_write_spec(tmp_path, badpair)), str(tmp_path))


def test_example_spec_parses_with_all_pair_lists():
    runs, pairs = cu.load_runs_spec(EXAMPLE, SCRIPTS)
    assert list(runs) == ["str_0.42mm", "str_1mm", "str_1mm_350K",
                          "mini_1mm", "mini_2mm", "field_1mm"]
    assert runs["field_1mm"]["kind"] == "field"
    assert len(pairs["transfer"]) == 11
    assert len(pairs["hysteresis"]) == 6


def test_derived_eq7_pairs_match_the_authors_explicit_pairs():
    """The auto-default must equal what the hardcoded EQ7_PAIRS table said,
    including picking str_1mm (run order) over the equal-length 350K file."""
    records = json.load(open(REGISTRY, encoding="utf-8"))["records"]
    derived = cu.derive_eq7_pairs(records)
    _, pairs = cu.load_runs_spec(EXAMPLE, SCRIPTS)
    explicit = {cell: {br: tuple(ids) for br, ids in brs.items()}
                for cell, brs in pairs["eq7"].items()}
    assert derived == explicit


def test_derived_hysteresis_pairs_cover_every_complete_cycle():
    records = json.load(open(REGISTRY, encoding="utf-8"))["records"]
    derived = set(cu.derive_hysteresis_pairs(records))
    _, pairs = cu.load_runs_spec(EXAMPLE, SCRIPTS)
    # authors' explicit choice is a subset; the derivation also picks up the
    # 350K c2 cycle the authors deliberately left out of their QC figure
    assert {tuple(p) for p in pairs["hysteresis"]} <= derived
    assert ("str_1mm_350K/c2c", "str_1mm_350K/c2w") in derived


def _synthetic_cu_dat(path, n=600):
    """One cool (300->10 K) + one warm (10->300 K) branch, smooth + tiny
    deterministic noise — enough to pass segment_branches' 300-point floor."""
    rng = np.random.default_rng(7)
    T = np.r_[np.linspace(300, 10, n), np.linspace(10, 300, n)]
    dl = 400 + 0.1 * T - 1e-4 * T ** 2 + rng.normal(0, 0.05, T.size)
    t = np.arange(T.size, dtype=float)          # 1 s cadence, no gaps
    C = np.full(T.size, 15.0)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("dilatometry PPMS. C_max = 100.000000 pF.\n")
        fh.write("T PPMS [K]\tdelta l [1E-6 cm]\tRel Time\tC [pF]\n")
        for row in zip(T, dl, t, C):
            fh.write("\t".join(f"{v:.6f}" for v in row) + "\n")


def test_end_to_end_synthetic_run_uses_the_users_key(tmp_path):
    _synthetic_cu_dat(tmp_path / "myCu.dat")
    _write_spec(tmp_path, {"runs": [
        {"key": "lab7_2mm", "path": "myCu.dat",
         "cell": "lab7_cell", "cu_length_mm": 2.0}]})
    out = tmp_path / "reg.json"
    cu.main(["--data", str(tmp_path), "--out", str(out),
             "--qc-dir", str(tmp_path / "qc")])
    reg = json.loads(out.read_text(encoding="utf-8"))
    ids = [r["id"] for r in reg["records"]]
    assert ids == ["lab7_2mm/c1c", "lab7_2mm/c1w"]
    assert all(r["source_file"] == "myCu.dat" for r in reg["records"])
    assert all(r["cell"] == "lab7_cell" for r in reg["records"])
    # no transfer_pairs declared and one Cu length -> honest absences
    assert "transfer" not in reg
    assert list(reg["eq7"]) == ["_doc"]
    assert "field_background" not in reg
    assert (tmp_path / "qc" / "seg_lab7_2mm.png").is_file()


def test_unknown_file_key_lists_available_keys(tmp_path):
    _synthetic_cu_dat(tmp_path / "myCu.dat")
    _write_spec(tmp_path, {"runs": [
        {"key": "lab7_2mm", "path": "myCu.dat",
         "cell": "lab7_cell", "cu_length_mm": 2.0}]})
    with pytest.raises(SystemExit, match="lab7_2mm"):
        cu.main(["--data", str(tmp_path), "--file", "typo"])
