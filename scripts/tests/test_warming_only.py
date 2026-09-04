"""Warming-only protocol regression (edge-case handoff #1, HIGH).

A run whose virgin segment has no cooling rows used to lose ALL virgin rows
silently: t_cool_start = min() over an empty selection is NaN and
`virgin[Rel Time >= NaN]` drops everything. The reducer must instead keep the
warm branch, flow it through alpha and the outputs, and report the
branch-dependent gates as N/A rather than FAIL.
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_COLS = ["Abs Time", "Rel Time", "T PPMS [K]", "T sample [K]", "B [T]",
            "C [pF]", "L [nS]", "delta l [1E-6 cm]"]


def _write_warming_only_archive(path, n=3000):
    """Minimal PPMS-shaped warming-only run: one 2->300 K warm ramp, B = 0."""
    t = np.arange(n, dtype=float) * 10.0            # s, one row / 10 s
    T = np.linspace(2.0, 300.0, n)
    dl = 0.5 * (T - 2.0) + 0.0004 * (T - 2.0) ** 2  # smooth, monotone, 1e-6 cm
    df = pd.DataFrame({
        "Abs Time": t + 1.7e9,
        "Rel Time": t,
        "T PPMS [K]": T,
        "T sample [K]": T,
        "B [T]": 0.0,
        "C [pF]": 7.6 - dl * 1e-4,
        "L [nS]": 0.0,
        "delta l [1E-6 cm]": dl,
    })
    df.to_csv(path, index=False, columns=RAW_COLS)


@pytest.fixture(scope="module")
def reduced(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    out = tmp_path_factory.mktemp("out")
    src = data / "synthetic_warm_only_all.csv"
    _write_warming_only_archive(str(src))
    proc = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS, "reduce_str_batch.py"),
         "--data", str(data), "--file", src.name, "--L0", "0.058",
         "--transition-type", "none", "--out", str(out)],
        capture_output=True, text=True, cwd=SCRIPTS)
    return proc, out


def test_exit_zero(reduced):
    proc, _ = reduced
    assert proc.returncode == 0, \
        f"reducer failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"


def test_virgin_rows_not_dropped(reduced):
    proc, _ = reduced
    m = [ln for ln in proc.stdout.splitlines() if "virgin" in ln
         and "rows" in ln]
    assert m, f"no virgin row-count line in output:\n{proc.stdout[-2000:]}"
    n = int(m[0].split(":")[1].split("rows")[0].strip().split()[-1])
    assert n > 2000, f"virgin rows lost: {m[0]}"


def test_alpha_output_has_warm_branch(reduced):
    proc, out = reduced
    alpha_csv = [p for p in out.rglob("*_alpha.csv")]
    assert alpha_csv, f"no alpha CSV written under {out}"
    df = pd.read_csv(alpha_csv[0], comment="#")
    warm = df[df["direction"] == "warm"]
    assert len(warm) > 20, f"warm alpha missing/short: {len(warm)} rows"


def test_branch_gates_na_not_fail(reduced):
    proc, _ = reduced
    text = proc.stdout
    assert "all gates PASS" in text or "-> N/A" in text, text[-2000:]
    # a warming-only run must never FAIL the cool/warm-dependent gates
    for ln in text.splitlines():
        if "cool-warm consistency" in ln or "paramagnetic alpha" in ln:
            assert "FAIL" not in ln, ln
