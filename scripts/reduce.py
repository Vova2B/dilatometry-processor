"""
reduce.py — shared reduction core for the US dilatometry pipeline.

Single home (P1, 2026-07-03) for the data-model + physics that the QC
interactive scripts and the headless batch drivers previously each carried
their own copy of. The QC modules now import these; the batch drivers import
build_b_loop_curves from here (re-exported by reduce_str_batch for back-compat).

Contents:
  COLORS, Curve            plotting palette + one QC-controllable sweep segment
  custom_round             0.5-K rounding used by separate_data
  assign_branch            per-row cool/warm calibration branch
  separate_data            split raw rows into field-dep (B) and temp-dep (T)
  estimate_C0, cmax_ratio  mini-cell true-Cmax correction primitives
  compute_del_l_l0         UNIFIED (ΔL/L₀)_Sam; cmax=None -> ratio 1.0 (str),
                           cmax=dict -> mini true-Cmax correction
  build_b_loop_curves      field loops -> up/down Curve pairs, loop-referenced

Depends only on cleanup.py (stitch_steps) + numpy/pandas/scipy — no import
cycle with the QC modules.
"""

import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from cleanup import stitch_steps


def output_stem(file_path):
    """Canonical output stem for a run file: basename minus extension, stray
    trailing dashes/whitespace (PPMS 'name - -.dat' exports) removed. Single
    source of truth — the batch reducers and the QC apps MUST derive stems
    identically, else one run writes two file families side by side."""
    stem = os.path.splitext(os.path.basename(file_path))[0]
    prev = None
    while stem != prev:                      # 'name.dat - -' needs repeated
        prev = stem                          # dash/space/.dat passes — a single
        stem = stem.strip().rstrip("-")      # rstrip leaves 'name.dat -'
        if stem.lower().endswith(".dat"):
            stem = stem[:-4]
    return stem or "run"

# B-loop fragment gates (a loop shorter/narrower than these is a ramp to a
# setpoint, not a sweep). Kept here with build_b_loop_curves; re-exported by
# reduce_str_batch for the batch drivers that reference them.
SWEEP_MIN_ROWS = 50
SWEEP_MIN_SPAN_T = 3.0


COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
    "#bcbd22", "#aec7e8",
]

# QC sidecar (_qc_state.json) schema version (review #7). Bump when a NEW
# required per-curve state field is added. The corrections[] list exists NOW
# so the P5 artifact-recognition contract is present before P5 fills it.
STATE_SCHEMA = 1

# Per-curve state keys this version knows how to interpret. Anything else in a
# sidecar (e.g. a field written by a NEWER tool) is kept in Curve._extra_state
# and re-emitted verbatim, so a round-trip never drops forward-compat data.
_KNOWN_STATE_KEYS = {
    "state_schema", "label", "enabled", "trim_start", "trim_end",
    "smooth_window", "neg_threshold", "corrections",
}


class Curve:
    """One QC-controllable sweep segment (cool or warm, one B or T value)."""

    def __init__(self, kind, param_value, direction, mode_index,
                 label, color, angle_deg, raw_df):
        # ── immutable identity ──────────────────────────────────────────────
        self.kind        = kind          # "T" or "B"
        self.param_value = param_value   # B value (T-sweep) or T value (B-sweep)
        self.direction   = direction     # "cool"/"warm" or "up"/"down"
        self.mode_index  = mode_index    # original Mode_Index label e.g. "1_3"
        self.label       = label         # display string e.g. "B=6T cool #2"
        self.color       = color
        self.angle_deg   = angle_deg
        self.raw_df      = raw_df.copy().reset_index(drop=True)

        # ── mutable QC state ────────────────────────────────────────────────
        self.enabled       = True
        self.trim_start    = 0           # rows to drop from head
        self.trim_end      = 0           # rows to drop from tail
        self.smooth_window = 0           # 0 = off; else odd int ≥ 5
        self.neg_threshold = None        # float or None; drops delta_l < threshold
        # P3 (review #7): auditable artifact corrections, passed through
        # verbatim. Each is a dict (P5 populates it via detect_corrections);
        # nothing here consumes it yet — it only persists in the sidecar.
        self.corrections  = []
        # forward-compat: unknown sidecar keys from a newer tool, re-emitted
        # unchanged so a load -> save round-trip never drops them.
        self._extra_state = {}

        # cache key for cleaned()
        self._cache_key  = None
        self._cache_data = None

    # ── computed ─────────────────────────────────────────────────────────────

    def cleaned(self):
        """Return cleaned DataFrame with x, y columns ready for plotting.

        Pipeline: trim → drop invalid rows → SG smooth.
        x = T_K (T-sweeps) or B_T (B-sweeps); y = (del_L/L_0)_Sam.
        """
        key = (self.trim_start, self.trim_end,
                self.smooth_window, self.neg_threshold)
        if key == self._cache_key and self._cache_data is not None:
            return self._cache_data

        n = len(self.raw_df)
        end = n - self.trim_end if self.trim_end > 0 else n
        df = self.raw_df.iloc[self.trim_start:end].copy()

        # drop bridge-dropout and unphysical rows
        df = df[df["C [pF]"] > 0]
        df = df[np.isfinite(df["(del_L/L_0)_Sam"])]
        if self.neg_threshold is not None:
            df = df[df["delta l [1E-6 cm]"] >= self.neg_threshold]

        # smoothing
        if self.smooth_window >= 5 and len(df) >= self.smooth_window:
            win = self.smooth_window
            if win % 2 == 0:
                win += 1   # must be odd
            df = df.copy()
            df["(del_L/L_0)_Sam"] = savgol_filter(
                df["(del_L/L_0)_Sam"].values, win, polyorder=3)

        self._cache_key  = key
        self._cache_data = df
        return df

    def reset(self):
        """Restore all QC state to defaults."""
        self.enabled       = True
        self.trim_start    = 0
        self.trim_end      = 0
        self.smooth_window = 0
        self.neg_threshold = None
        self.corrections   = []
        self._cache_key    = None
        self._cache_data   = None

    def to_state_dict(self):
        d = {
            "state_schema":  STATE_SCHEMA,
            "label":         self.label,
            "enabled":       self.enabled,
            "trim_start":    self.trim_start,
            "trim_end":      self.trim_end,
            "smooth_window": self.smooth_window,
            "neg_threshold": self.neg_threshold,
            "corrections":   list(self.corrections),
        }
        # re-emit any forward-compat keys we preserved on load
        for k, v in self._extra_state.items():
            d.setdefault(k, v)
        return d

    def apply_state_dict(self, d):
        self.enabled       = d.get("enabled",       self.enabled)
        self.trim_start    = d.get("trim_start",    self.trim_start)
        self.trim_end      = d.get("trim_end",       self.trim_end)
        # A stale sidecar (saved against a longer raw segment) can carry trims
        # that swallow the whole curve — cleaned() would silently go empty
        # while the sliders show clipped values. Reset such trims loudly.
        n = len(self.raw_df)
        ts, te = max(0, int(self.trim_start)), max(0, int(self.trim_end))
        if ts + te >= n:
            print(f"  QC state: stale trims ({ts}+{te} rows) exceed this "
                  f"curve's {n} rows ({self.label}) — trims reset to 0.")
            ts = te = 0
        self.trim_start, self.trim_end = ts, te
        self.smooth_window = d.get("smooth_window",  self.smooth_window)
        self.neg_threshold = d.get("neg_threshold",  self.neg_threshold)
        # corrections default to [] for OLD sidecars that predate the field
        self.corrections   = list(d.get("corrections", self.corrections))
        # keep unknown keys (forward-compat with a newer sidecar writer)
        self._extra_state  = {k: v for k, v in d.items()
                              if k not in _KNOWN_STATE_KEYS}
        self._cache_key    = None
        self._cache_data   = None


def custom_round(x, increment=0.5):
    return round(x / increment) * increment


def assign_branch(df: pd.DataFrame, field_idx) -> np.ndarray:
    """Per-row calibration branch. T-segments use their own direction
    (Mode 1 = cool, 10 = warm); B-sweep rows inherit the branch that brought
    the system to that temperature (last T-segment before the sweep) — the
    calibration term is ~constant within a fixed-T sweep, so the choice only
    fixes the offset bookkeeping."""
    br = pd.Series(np.where(df["Mode (1-cool,10-warm)"] == 1, "cool", "warm"),
                   index=df.index, dtype=object)
    br.loc[df.index.isin(field_idx)] = np.nan
    return br.ffill().bfill().values


# ════════════════════════════════════════════════════════════════════════════
# Cmax correction (mini cell; str passes cmax=None -> ratio identically 1.0)
# ════════════════════════════════════════════════════════════════════════════

def estimate_C0(df: pd.DataFrame, cmax: dict) -> float:
    """
    Estimate PPMS reference capacitance C₀ from the first low-field row.
    The PPMS computes: delta_l = K * (C − C₀)/(C*C₀) * (1 − C*C₀/Cmax²)
    with K ≈ 136 300 (r = 7 mm used internally for all cells) and
    sign convention: positive when C increases (gap decreases).
    Linearised for C ≈ C₀: C₀ ≈ C − dl * C² / (K * corr).
    Low-field row (|B| < 0.1 T) is preferred to avoid large field-induced
    delta_l values corrupting the estimate.

    K and Cmax_ppms come from the cell's cmax config (cells.py).
    """
    low_field = df[df["B [T]"].abs() < 0.1]
    row = low_field.iloc[0] if not low_field.empty else df.iloc[0]
    C1   = row["C [pF]"]
    dl1  = row["delta l [1E-6 cm]"]
    K    = cmax["K"]
    corr = 1.0 - C1**2 / cmax["cmax_ppms"]**2
    return C1 - dl1 * C1**2 / (K * corr)


def cmax_ratio(C_series: pd.Series, C0: float, cmax: dict) -> pd.Series:
    """
    Per-row Cmax correction factor. Multiply the PPMS-derived
    (delta_l − cell_fit)/L₀ by this factor to convert from Cmax=cmax_ppms to
    the cell's true Cmax. When cmax_true == cmax_ppms this is identically 1.0.
    """
    CC0 = C_series * C0
    return ((1.0 - CC0 / cmax["cmax_true"]**2)
            / (1.0 - CC0 / cmax["cmax_ppms"]**2))


def compute_del_l_l0(df: pd.DataFrame, L0: float, cal: dict,
                     C0: float = None, cmax: dict = None) -> pd.DataFrame:
    """
    Unified (ΔL/L₀)_Sam. Branch-aware: each row's cell calibration comes from
    the registry record matching its Branch column (assign_branch() must have
    run after separate_data()).

    Formula — P18 Eq. (6):
        ppms_term = (delta_l − cell_cal_branch(T)) * 1e-6 / L₀ * ratio
        raw       = ppms_term + cu_lit * 1e-6
        (ΔL/L₀)_Sam = raw − raw(T_min)

    cmax=None  -> str path: ratio is 1.0 exactly and NO Cmax_ratio column is
                  written (byte-identical to the legacy str compute).
    cmax=dict  -> mini path: ratio = cmax_ratio(C, C0, cmax); a Cmax_ratio
                  column is written. Requires C0.

    The registry calibrations were recorded through the same PPMS (same
    Cmax_ppms baked in), so the ratio applies to the full difference exactly.

    Sign note: PPMS delta_l is positive when C > C₀, which occurs when the sample
    EXPANDS (expansion pushes the movable part down, closing the plate gap; the
    positive delta_l seen on cooling is the cell background — frame contraction —
    not sample contraction; cf. P18 Fig. 6). delta_l and the calibration share
    this native sign and are subtracted directly, as in Küchler 2016 Eqs. (3)-(6).
    """
    T      = df["T PPMS [K]"].values
    dl     = df["delta l [1E-6 cm]"].values
    branch = df["Branch"].values
    cf = np.where(branch == "cool", cal["cool"](T), cal["warm"](T))
    cu = cu_lit_data_fit(T)

    if cmax is not None:
        if C0 is None:
            raise ValueError("compute_del_l_l0: cmax given but C0 is None")
        ratio = cmax_ratio(df["C [pF]"], C0, cmax).values
        ppms_term = (dl - cf) * 1e-6 / L0 * ratio
    else:
        ratio = None
        ppms_term = (dl - cf) * 1e-6 / L0

    total   = ppms_term + cu * 1e-6
    ref_idx = int(np.argmin(T))
    total  -= total[ref_idx]

    df["(del_L/L_0)_Sam"]  = total
    df["Cell_with_Cu_fit"] = cf
    df["Cu_lit_data_fit"]  = cu
    if ratio is not None:
        df["Cmax_ratio"] = ratio
    return df


def cu_lit_data_fit(T, r3=16549.7494, r4=-578.02392, r5=56.45939):
    """Cu literature thermal expansion [1e-6 cm]. Identical in both QC
    modules; the canonical copy lives here and they re-export it."""
    return r3 * np.exp(r4 / (T + r5))


# ════════════════════════════════════════════════════════════════════════════
# Data separation
# ════════════════════════════════════════════════════════════════════════════

def separate_data(df: pd.DataFrame,
                  field_change_threshold=100,
                  temp_stability_threshold=0.7,
                  min_points=25, lookahead=25):
    df = df.copy()
    df["Temp_Change"]  = df["T PPMS [K]"].diff().abs()
    df["Field_Change"] = df["B [T]"].diff().abs() * 10000
    df["Rounded_T_for_field"] = df["T PPMS [K]"].apply(
        lambda x: custom_round(x, 0.5))

    tdiff = df["T PPMS [K]"].diff()
    modes = tdiff.apply(lambda x: 1 if x < 0 else 10).tolist()
    for i in range(len(modes)):
        if i + lookahead < len(modes):
            nxt = modes[i+1:i+1+lookahead]
            if not all(v == modes[i] for v in nxt):
                modes[i] = modes[i-1] if i > 0 else modes[i]
        else:
            modes[i] = modes[i-1] if i > 0 else modes[i]
    if len(modes) > 1:
        modes[0] = modes[1]
    df["Mode (1-cool,10-warm)"] = modes

    midx, last = 1, df.iloc[0]["Mode (1-cool,10-warm)"]
    seg_labels = []
    for i in range(len(df)):
        cur = df.iloc[i]["Mode (1-cool,10-warm)"]
        la  = (df.iloc[i+1:i+1+lookahead]["Mode (1-cool,10-warm)"]
               .eq(last).all() if i+1+lookahead <= len(df) else True)
        if cur != last and not la:
            midx += 1; last = cur
        # int() guards against float upcast on mixed-dtype rows: a "1.0_2"
        # label would silently fail every `split("_")[0] == "1"` cool test.
        seg_labels.append(f"{int(cur)}_{midx}")
    df["Mode_Index"] = seg_labels

    fdr = df[(df["Field_Change"] > field_change_threshold) &
             (df["Temp_Change"]  < temp_stability_threshold)]
    if fdr.empty:
        return pd.DataFrame(), df

    first_idx = max(0, fdr.index[0] - 1)
    if first_idx not in fdr.index:
        fdr = pd.concat([df.loc[[first_idx]], fdr])

    fdr = fdr.copy()
    fdr["Rounded_Temp"] = fdr["T PPMS [K]"].round()
    valid_T = (fdr.groupby("Rounded_Temp")
               .filter(lambda g: len(g) >= min_points)["Rounded_Temp"].unique())
    fdr = fdr[fdr["Rounded_Temp"].isin(valid_T)]
    tdr = df[~df.index.isin(fdr.index)]
    return fdr, tdr


# ════════════════════════════════════════════════════════════════════════════
# Field-loop curve building (moved from reduce_str_batch.py, P1)
# ════════════════════════════════════════════════════════════════════════════

def build_b_loop_curves(field_dep, fb=None, L0_cm=None, stitch_thr=0.04e-3):
    """Field loops as Curve pairs (up/down legs), each loop re-referenced to
    its B~0 start; ramp-to-setpoint fragments dropped (returned for logging).

    stitch_thr: persistent instrumental level shifts above this are stitched
    out (str runs — validated visually, every fix in provenance). Pass None
    for the mini runs: their below-T_C staircase is domain-avalanche /
    remanence SIGNAL, and stitching rectifies it into fake drift (checked on
    the -45deg 150 K loop: raw +2.18e-3 at 13 T becomes -6.5e-3 stitched).

    Replaces build_curves' B path: that one sorts a whole 0->13->0 loop by
    B, interleaving the legs into a zigzag, and keeps absolute referencing.
    """
    curves, dropped, loop_steps = [], [], {}
    if field_dep.empty or "Rounded_Temp" not in field_dep.columns:
        return curves, dropped, loop_steps   # run with no field sweeps
    temps = sorted(field_dep["Rounded_Temp"].unique())
    color_map = {t: COLORS[i % len(COLORS)] for i, t in enumerate(temps)}
    for (t_round, mode_idx), grp in field_dep.groupby(
            ["Rounded_Temp", "Mode_Index"], sort=False):
        b = grp["B [T]"].values
        span = b.max() - b.min()
        if len(grp) < SWEEP_MIN_ROWS or span < SWEEP_MIN_SPAN_T:
            dropped.append((f"T={t_round:.0f}K {mode_idx}",
                            f"fragment: n={len(grp)}, B-span {span:.1f} T "
                            "(ramp to setpoint, not a sweep)"))
            continue
        ref = grp["(del_L/L_0)_Sam"].iloc[:5].median()
        g = grp.copy()
        g["(del_L/L_0)_Sam"] = g["(del_L/L_0)_Sam"] - ref
        # smooth cell field background (registry P2-scale quadratic); the
        # stick-slip snap component is not correctable, only enveloped
        if fb and fb.get("p2_scale_at_9T_1e6cm") and L0_cm:
            corr = (fb["p2_scale_at_9T_1e6cm"] * (g["B [T]"] / 9.0) ** 2
                    * 1e-6 / L0_cm)
            g["(del_L/L_0)_Sam"] = g["(del_L/L_0)_Sam"] - corr
        # unstitched lambda kept alongside: a sharp sub-T_C step could in
        # principle be a real domain avalanche — the stitch must be
        # auditable/reversible from the CSV (experimenter's call)
        g["lambda_raw"] = g["(del_L/L_0)_Sam"]
        if stitch_thr is not None:
            yv, fixes = stitch_steps(g["Rel Time"].values,
                                     g["(del_L/L_0)_Sam"].values,
                                     jump_threshold=stitch_thr)
            if fixes:
                for f_ in fixes:
                    f_["B_T"] = round(float(b[f_.pop("index")]), 2)
                loop_steps[f"T={t_round:.0f}K {mode_idx}"] = fixes
            g["(del_L/L_0)_Sam"] = yv
        peak = int(np.argmax(np.abs(b)))
        legs = [("up", g.iloc[:peak + 1])]
        if peak < len(g) - 10:
            legs.append(("down", g.iloc[peak:]))
        for direction, leg in legs:
            curves.append(Curve(
                kind="B", param_value=t_round, direction=direction,
                mode_index=mode_idx,
                label=f"T={t_round:.0f}K {direction}",
                color=color_map[t_round], angle_deg=0,
                raw_df=leg.sort_values("B [T]")))
    return curves, dropped, loop_steps
