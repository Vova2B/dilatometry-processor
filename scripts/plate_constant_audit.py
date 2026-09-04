"""
plate_constant_audit.py — plate-constant audit for PPMS dilatometry runs.

Recover the effective plate constant K = eps0*pi*r^2 the PPMS actually used
when converting capacitance to delta_l, by inverting the Pott-Schefzyk gap
relation from each file's recorded (C, delta_l) pairs:

    d(C)     = (K / C) * (1 - C^2 / Cmax^2)          [gap]
    delta_l  = const - d(C)                           [PPMS output]

Method: robust regression on point-to-point DIFFERENCES,
    d(delta_l) = -K * d(g),   g(C) = (1 - C^2/Cmax^2) / C,
which is immune to re-zero offsets. Cmax is fixed at the header-confirmed
100 pF (at C = 8-19 pF the correction is <4%, and Cmax is unidentifiable
from data anyway at C << Cmax). MAD-clipped; re-zero jumps excluded.

The implied radius r = sqrt(K_SI / (eps0*pi)) identifies the PPMS config:
r ~ 7 mm = documented PPMS default plate radius, r ~ 5 mm = mini cell.
Use it to catch runs that were PPMS-converted with the wrong plate radius
(a sample/calibration K ratio of (7/5)^2 = 1.96 is the classic symptom;
fix = rescale that run's raw delta_l by the inverse ratio).

detect.py imports EPS0 / load_pairs / fit_K from here; main() is a
standalone audit — list your own files in FILES below and run it.
"""

import glob
import os
import sys

# Windows: pipes/files default to the ANSI codepage (cp1252/cp1251/cp932),
# which cannot encode the Greek/degree/box glyphs this tool prints and writes.
# Force UTF-8 so runs behave the same on Windows as on macOS/Linux.
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))      # scripts/
PROC = os.path.dirname(HERE)                            # Dilatometry processor/
PARENT = os.path.dirname(PROC)                          # USe root (raw Cu .dat files)
EPS0 = 8.8541878128e-12   # F/m
CMAX = 100.0              # pF (confirmed in every .dat header)

FILES = [
    # (label, path, loader) — loader is "dat" (raw PPMS export) or "csv"
    # (comma *_all.csv archive). EDIT: list your Cu calibration runs and
    # sample runs here, e.g.:
    # ("Cu mini 1mm", os.path.join(PARENT, "Cu_1mm_mini_dil.dat"), "dat"),
    # ("sample mini rot0", os.path.join(PROC, "Data", "rot0_all.csv"), "csv"),
]


def load_pairs(path, kind):
    if kind == "dat":
        df = pd.read_csv(path, delimiter="\t", skiprows=1, encoding="utf-8", encoding_errors="replace")
    else:
        df = pd.read_csv(path, usecols=["C [pF]", "delta l [1E-6 cm]"], encoding="utf-8", encoding_errors="replace")
    df = df[["C [pF]", "delta l [1E-6 cm]"]].apply(
        pd.to_numeric, errors="coerce").dropna()
    df = df[df["C [pF]"] > 0.5]
    # keep the working window around the median C — glitch rows (plate
    # touch, C collapse) would corrupt even the diff-based fit
    c_med = df["C [pF]"].median()
    df = df[np.abs(df["C [pF]"] - c_med) < 2.0]
    return df["C [pF]"].values, df["delta l [1E-6 cm]"].values


def fit_K(C, dl, jump_threshold=50.0, n_iter=4, clip=4.0):
    """K [1e-6 cm * pF] from diffs of delta_l vs diffs of g(C)."""
    g = (1.0 - C**2 / CMAX**2) / C          # gap / K, units 1/pF
    dgl = np.diff(dl)
    dg = np.diff(g)
    m = (np.abs(dgl) < jump_threshold) & (np.abs(dg) > 1e-9)
    keep = m.copy()
    K = np.nan
    for _ in range(n_iter):
        if keep.sum() < 100:
            return np.nan, 0
        K = -np.sum(dg[keep] * dgl[keep]) / np.sum(dg[keep] ** 2)
        r = dgl + K * dg
        s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = m & (np.abs(r - np.median(r[keep])) < clip * max(s, 1e-12))
    return K, int(keep.sum())


def main():
    print(f"{'file':22s} {'C range [pF]':>14s} {'K [1e-6cm*pF]':>14s} "
          f"{'r_impl [mm]':>12s} {'n_diff':>7s}")
    results = {}
    for label, path, kind in FILES:
        if not os.path.exists(path):
            print(f"{label:22s}  MISSING: {path}")
            continue
        C, dl = load_pairs(path, kind)
        if len(C) < 300:
            print(f"{label:22s}  skipped (only {len(C)} usable rows)")
            continue
        K, n = fit_K(C, dl)
        # K [1e-6 cm * pF] -> SI: 1e-8 m * 1e-12 F = 1e-20 F*m
        r_mm = np.sqrt(K * 1e-20 / (EPS0 * np.pi)) * 1e3
        results[label] = (K, r_mm)
        print(f"{label:22s} {C.min():6.2f}-{C.max():6.2f} "
              f"{K:14.0f} {r_mm:12.3f} {n:7d}")

    cu_mini = [v for k, v in results.items() if k.startswith("Cu mini")]
    sample_mini = [v for k, v in results.items() if k.startswith("sample mini")]
    if cu_mini and sample_mini:
        k_cu = np.mean([v[0] for v in cu_mini])
        k_sam = np.mean([v[0] for v in sample_mini])
        print(f"\nmini-cell K ratio  sample / Cu-calibration: "
              f"{k_sam / k_cu:.3f}   ((7mm/5mm)^2 = 1.96)")
    cu_str = [v for k, v in results.items() if k.startswith("Cu str")]
    sample_str = [v for k, v in results.items() if k.startswith("sample str")]
    if cu_str and sample_str:
        k_cu = np.mean([v[0] for v in cu_str])
        print(f"str-cell  K ratio  sample / Cu-calibration: "
              f"{sample_str[0][0] / k_cu:.3f}   (expected ~1.00)")


if __name__ == "__main__":
    main()
