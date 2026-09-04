import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import transition as tr


def test_active_and_labels():
    t = tr.Transition(177.0, "T_C")
    assert t.active is True
    assert t.split_active is True
    assert t.label == r"$T_C$"
    assert t.axis_text() == r"$T_C$"
    lo, hi = t.panel_titles()
    assert "ferromagnetic" in lo and "paramagnetic" in hi


def test_tn_and_cdw_generic_titles():
    tn = tr.Transition(50.0, "T_N")
    assert tn.label == r"$T_N$"
    lo, hi = tn.panel_titles()
    assert lo == r"$T < T_N$" and hi == r"$T \geq T_N$"
    cdw = tr.Transition(94.0, "T_CDW")
    assert cdw.label == r"$T_{CDW}$"


def test_none_is_inactive():
    n = tr.Transition(None, "none")
    assert n.active is False
    assert n.split_active is False
    assert n.label is None
    assert n.axis_text() is None


def test_value_without_kind_defaults_none_kind_rejected():
    # a value with kind "none" is inactive (no line) — explicit choice
    assert tr.Transition(120.0, "none").active is False


import numpy as np


def test_find_transition_flat_returns_none():
    T = np.linspace(2, 300, 4000)
    dLL = 1e-5 * (T - 2)                       # perfectly linear -> flat alpha
    assert tr.find_transition(T, dLL) == []


def test_rank_peaks_trims_edge_zone():
    # the boxcar baseline zero-pads at the sig ends, so peaks within half a
    # window (edge bins) of either extreme are biased and must be trimmed.
    tb = np.arange(3.5, 3.5 + 20 * 3.0, 3.0)          # 20 bins, 3 K apart
    sig = np.zeros(20)
    sig[1] = 5.0                                       # inside low-T edge zone
    sig[10] = 3.0                                      # genuine interior peak
    sig[18] = 4.0                                      # inside high-T edge zone
    out = tr._rank_peaks(sig, tb, min_prominence=0.15, max_candidates=3,
                         edge=2)
    assert [c["T"] for c in out] == [float(tb[10])], f"edge peaks kept: {out}"


def test_rank_peaks_all_in_edge_zone_returns_empty():
    tb = np.arange(3.5, 3.5 + 20 * 3.0, 3.0)
    sig = np.zeros(20)
    sig[1] = 5.0
    assert tr._rank_peaks(sig, tb, 0.15, 3, edge=2) == []


def test_find_transition_locates_lambda_anomaly():
    T = np.linspace(2, 300, 6000)
    # linear background + a lambda-like kink near 177 K (extra contraction)
    base = 1e-5 * (T - 2)
    anomaly = -6e-4 * np.exp(-((T - 177.0) ** 2) / (2 * 4.0 ** 2))
    dLL = base + anomaly
    cands = tr.find_transition(T, dLL)
    assert cands, "expected at least one candidate"
    assert abs(cands[0]["T"] - 177.0) < 6.0
