"""Regressions from the 2026-07-09 app-flow audit:

* output_stem must fully strip PPMS 'name.dat - -.dat' export junk — a single
  rstrip pass left 'name.dat -' stems (seen live on the Cu 0.42 mm run).
* Curve.apply_state_dict must not honor stale sidecar trims that exceed the
  current raw segment — they silently emptied the curve while the QC sliders
  showed clipped values.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reduce import Curve, output_stem  # noqa: E402


@pytest.mark.parametrize("name,want", [
    ("Cu_0.42mm_str_dil.dat - -.dat", "Cu_0.42mm_str_dil"),
    ("Cu_1mm_mini_dil - .dat", "Cu_1mm_mini_dil"),
    ("DataXX_d_0_58__str_dil.dat", "DataXX_d_0_58__str_dil"),
    ("sample_str_all.csv", "sample_str_all"),
    (" - -.dat", "run"),
])
def test_output_stem_strips_export_junk(name, want):
    assert output_stem(name) == want


def _curve(n=20):
    df = pd.DataFrame({
        "T_K": np.linspace(2, 300, n),
        "B_T": 0.0,
        "C [pF]": 7.6,
        "delta l [1E-6 cm]": 0.0,
        "(del_L/L_0)_Sam": np.linspace(0, 1e-3, n),
    })
    return Curve("T", 0.0, "cool", "1_1", "B=0T cool", "k", 0.0, df)


def test_stale_trims_beyond_raw_length_reset(capsys):
    c = _curve(20)
    c.apply_state_dict({"trim_start": 500, "trim_end": 100})
    assert (c.trim_start, c.trim_end) == (0, 0)
    assert len(c.cleaned()) == 20
    assert "stale trims" in capsys.readouterr().out


def test_valid_trims_still_apply():
    c = _curve(20)
    c.apply_state_dict({"trim_start": 3, "trim_end": 2})
    assert (c.trim_start, c.trim_end) == (3, 2)
    assert len(c.cleaned()) == 15
