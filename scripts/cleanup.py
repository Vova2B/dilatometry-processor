"""
cleanup.py — shared curve-cleanup primitives for the US dilatometry pipeline.

Extracted verbatim from reduce_str_batch.py (P1, 2026-07-03) so the batch
drivers, the interactive QC scripts, and the unified tool all share ONE
implementation. Pure numpy/pandas; no project imports, so nothing here can
introduce an import cycle.

Contents:
  glitch_mask      — rows sitting on a shifted delta_l zero level (re-zero)
  alpha_of         — MAD-clipped linear slope of y(T) over a window
  detect_steps     — indices where time-ordered dL/L0 jumps > threshold
  stitch_steps     — subtract persistent instrumental level shifts
  dwell_mask       — rows inside a stationary-temperature dwell
  smooth_outside   — rolling-median smoothing outside the transition window
  binned_alpha     — alpha(T) in bins, step-aware
  clean_t_curves   — full T-curve cleanup (settling/dwell/stitch/smooth/split)

clean_t_curves takes the Curve class as `curve_cls` so it stays decoupled
from reduce.py.
"""

import numpy as np
import pandas as pd


def glitch_mask(df, jump_threshold=200.0, level_tol=200.0):
    """Rows sitting on a shifted delta_l zero level (plate touch / re-zero).
    Level-tracking: cumulative sum of the large jumps."""
    dl = df["delta l [1E-6 cm]"].values
    d = np.diff(dl)
    level = np.zeros(len(dl))
    for i in np.where(np.abs(d) > jump_threshold)[0]:
        level[i + 1:] += d[i]
    level -= np.median(level)
    return np.abs(level) > level_tol, int((np.abs(d) > jump_threshold).sum())


def alpha_of(T, y, lo, hi, n_iter=3, clip=3.0):
    """MAD-clipped linear slope of y(T) over [lo, hi]."""
    m = (T >= lo) & (T <= hi)
    if m.sum() < 10:
        return np.nan
    Tm, ym = T[m], y[m]
    keep = np.ones(len(Tm), bool)
    for _ in range(n_iter):
        p = np.polyfit(Tm[keep], ym[keep], 1)
        r = ym - np.polyval(p, Tm)
        s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        new = np.abs(r - np.median(r[keep])) < clip * max(s, 1e-12)
        if new.sum() == keep.sum():
            break
        keep = new
    return np.polyfit(Tm[keep], ym[keep], 1)[0]


def detect_steps(rel_time, y, threshold=0.05e-3):
    """Indices where consecutive (time-ordered) dL/L0 points jump by more
    than `threshold` — small instrumental relaxation/re-zero steps (e.g. the
    -0.115e-3 step at 96 K on the virgin cool) that sit far below the
    200e-6 cm glitch gate but ruin derivative estimates."""
    o = np.argsort(rel_time)
    dy = np.abs(np.diff(y[o]))
    return o, np.where(dy > threshold)[0]


def stitch_steps(rel_time, y, jump_threshold=0.03e-3, max_dt=120.0,
                 T=None, protect=(155.0, 190.0)):
    """Remove instrumental level shifts: a jump |dy| > jump_threshold between
    consecutive points closer than max_dt seconds is treated as a re-zero /
    relaxation artifact and subtracted from all later points. Jumps across
    longer gaps (field loops, segment boundaries) are magnetic history and
    are NOT touched. Returns (y_corrected, corrections list)."""
    y = np.asarray(y, float)
    if len(y) < 2:                       # empty/single-point branch: nothing
        return y.copy(), []              # to stitch (e.g. a cooling-only cycle)
    o = np.argsort(rel_time)
    ys = y[o].astype(float).copy()
    dt = np.diff(rel_time[o])
    dy = np.diff(ys)
    corr = np.zeros_like(ys)
    fixes = []
    # never stitch inside the magnetic transition: its steep REAL slope
    # looks like persistent level shifts and would be "corrected" away
    prot = np.zeros(len(ys) - 1, bool)
    if T is not None and protect is not None:
        Ts = T[o]
        prot = (Ts[:-1] > protect[0]) & (Ts[:-1] < protect[1])
    for i in np.where((np.abs(dy) > jump_threshold) & (dt < max_dt)
                      & ~prot)[0]:
        # persistence check: only stitch if the LEVEL actually shifts
        # (median after vs before) — otherwise it is noise/a spike, and
        # stitching noise rectifies it into artificial drift
        pre = ys[max(0, i - 7):i + 1]
        post = ys[i + 1:i + 9]
        if len(pre) < 3 or len(post) < 3:
            continue
        shift = np.median(post) - np.median(pre)
        if abs(shift) < jump_threshold or np.sign(shift) != np.sign(dy[i]):
            continue
        corr[i + 1:] += shift
        fixes.append({"index": int(i),
                      "step_1e3": round(float(shift * 1e3), 3)})
    ys -= corr
    out = np.empty_like(ys)
    out[o] = ys
    return out, fixes


def dwell_mask(rel_time, T, win=15, span_k=0.15):
    """True for rows inside a stationary-temperature dwell (rolling T range
    over `win` rows below span_k) — plotting these draws vertical dL lines
    while T stands still (relaxation), an artifact on T-dep figures."""
    o = np.argsort(rel_time)
    Ts = pd.Series(T[o])
    rng = (Ts.rolling(win, center=True, min_periods=3).max()
           - Ts.rolling(win, center=True, min_periods=3).min())
    m = np.empty(len(T), bool)
    m[o] = (rng < span_k).values
    return m


def smooth_outside(T, y, lo=150.0, hi=190.0, win=21):
    """Light rolling-median smoothing applied ONLY outside [lo, hi] so the
    transition region keeps full sharpness (Origin-style cleanup)."""
    ys = pd.Series(y).rolling(win, center=True, min_periods=5).median().values
    out = y.astype(float).copy()
    m = (T < lo) | (T > hi)
    out[m] = ys[m]
    return out


def binned_alpha(rel_time, T, y, bin_k=4.0, min_pts=15, min_span=2.0,
                 step_threshold=0.05e-3):
    """alpha(T) in bin_k-wide bins, MAD-clipped, never straddling an
    instrumental step. Returns (T_centers, alpha_1e-6, steps_info)."""
    o, steps = detect_steps(rel_time, y, step_threshold)
    Ts, ys = T[o], y[o]
    seg_id = np.zeros(len(Ts), int)
    for s in steps:
        seg_id[s + 1:] += 1
    steps_info = [{"T_K": round(float(Ts[s]), 2),
                   "step_1e3": round(float((ys[s + 1] - ys[s]) * 1e3), 3)}
                  for s in steps]
    aT, aV = [], []
    for lo in np.arange(2, 298, bin_k):
        m = (Ts >= lo) & (Ts < lo + bin_k)
        if m.sum() < min_pts:
            continue
        # use only the dominant step-free segment inside the bin
        ids, counts = np.unique(seg_id[m], return_counts=True)
        mm = m & (seg_id == ids[np.argmax(counts)])
        if mm.sum() < min_pts or Ts[mm].max() - Ts[mm].min() < min_span:
            continue
        a = alpha_of(Ts[mm], ys[mm], lo, lo + bin_k)
        if np.isfinite(a):
            aT.append(lo + bin_k / 2)
            aV.append(a * 1e6)
    return np.array(aT), np.array(aV), steps_info


def clean_t_curves(t_curves, curve_cls, gap_s=600, min_rows=30,
                   smooth_lo=150.0, smooth_hi=190.0,
                   jump_thr=0.03e-3, dwell=True, smooth=True):
    """Shared T-curve cleanup: settling retraces out, stationary-dwell rows
    out (vertical dL-at-fixed-T artifact), persistent instrumental level
    shifts stitched, light smoothing outside the transition window, and
    curves SPLIT at acquisition gaps > gap_s so nothing draws a vertical
    connector across a field loop. Returns (new_curves, stats)."""
    out, n_settle, n_dwell, fixes = [], 0, 0, {}
    for c in t_curves:
        d = c.raw_df.sort_values("Rel Time")
        T_run = d["T PPMS [K]"]
        keep = (T_run <= T_run.cummin() + 0.7 if c.direction == "cool"
                else T_run >= T_run.cummax() - 0.7)
        n_settle += int((~keep).sum())
        d = d[keep].reset_index(drop=True)
        if len(d) < min_rows:
            continue
        if dwell:
            dm = dwell_mask(d["Rel Time"].values, d["T PPMS [K]"].values)
            n_dwell += int(dm.sum())
            d = d[~dm].reset_index(drop=True)
            if len(d) < min_rows:
                continue
        if jump_thr is not None:
            yv, fx = stitch_steps(d["Rel Time"].values,
                                  d["(del_L/L_0)_Sam"].values,
                                  jump_threshold=jump_thr,
                                  T=d["T PPMS [K]"].values,
                                  protect=(smooth_lo + 5, smooth_hi))
            if fx:
                fixes[c.label] = fx
        else:
            # loop-dense runs: "steps" are paired remanence jump+recovery —
            # stitching only the fast part biases the curve cumulatively
            yv = d["(del_L/L_0)_Sam"].values
        d["(del_L/L_0)_Sam"] = (smooth_outside(
            d["T PPMS [K]"].values, yv, lo=smooth_lo, hi=smooth_hi)
            if smooth else yv)
        # split at acquisition gaps (field loops / pauses)
        gaps = np.where(np.diff(d["Rel Time"].values) > gap_s)[0]
        bounds = [0] + [g + 1 for g in gaps] + [len(d)]
        first = True
        for a, b_ in zip(bounds[:-1], bounds[1:]):
            seg = d.iloc[a:b_].reset_index(drop=True)
            if len(seg) < min_rows:
                continue
            nc = curve_cls(kind=c.kind, param_value=c.param_value,
                           direction=c.direction, mode_index=c.mode_index,
                           label=(c.label if first
                                  else "_" + c.label),  # no legend dup
                           color=c.color, angle_deg=c.angle_deg,
                           raw_df=seg)
            nc.enabled = c.enabled
            out.append(nc)
            first = False
    return out, {"settling_rows": n_settle, "dwell_rows": n_dwell,
                 "stitched": fixes}


def auto_alpha_bin(T, requested, min_pts=15,
                   ladder=(0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)):
    """Resolution 'if the data allow': coarsen a requested alpha(T) bin width
    until the typical bin holds >= min_pts points (binned_alpha silently drops
    bins below min_pts, so a too-fine bin on sparse data yields an EMPTY alpha
    curve). Density = median count over non-empty bins at the requested width.
    Returns (bin_k, note); note is None when the requested width is kept."""
    T = np.asarray(T, float)
    T = T[np.isfinite(T)]
    if T.size < 2 * min_pts or T.max() - T.min() <= 0:
        return requested, None
    def _med(width):
        edges = np.arange(T.min(), T.max() + width, width)
        counts = np.histogram(T, bins=edges)[0]
        return float(np.median(counts[counts > 0])) if (counts > 0).any() \
            else 0.0

    med0 = _med(requested)
    if med0 >= min_pts:
        return requested, None
    for k in ladder:
        if k > requested and _med(k) >= min_pts:
            note = (f"alpha bin auto-coarsened {requested:g} -> {k:g} K "
                    f"(median {med0:.0f} pts per {requested:g} K bin "
                    f"< {min_pts})")
            return k, note
    return ladder[-1], (f"alpha bin auto-coarsened {requested:g} -> "
                        f"{ladder[-1]:g} K (very sparse data - alpha may "
                        "still be empty)")
