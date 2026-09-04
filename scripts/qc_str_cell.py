"""
qc_str_cell.py
Standard dilatometer (str_dil) analysis script.

Cmax note: str_dil capacitance is ~7.5 pF — far from Cmax for any cell.
At this C value the Cmax=100 vs Cmax=50 error is only ~1.7%, which is
within typical experimental uncertainty. No correction is applied.

Cell: str_dil. Point it at your run with --data/--file.

Calibration: branch-aware records from Script/calibrations.json (built by
cu_calibration_builder.py) — cooling segments are reduced with a cooling
calibration, warming with warming (the str cell shows ~63-75e-6 cm
cooling/warming hysteresis, see fig_calibration_QC/hysteresis.png). With
USE_EQ7 the P18 Eq. (7) length decomposition builds a virtual Cu curve at the
sample thickness; otherwise (or when Eq. 7 is not resolved for the branch)
the closest-Cu-length record is used. The legacy warming-only 0_42mm
polynomial is kept below only for cross-checks.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.signal import savgol_filter
import json
from qc_window import (_style_axes, _thin_idx, _draw_curves_T,
                       _draw_curves_B, plot_temperature_dep,
                       plot_field_dep, QCWindow, save_qc_state,
                       load_qc_state)
import os as _os_pathfix
import sys as _sys_pathfix

# Windows: pipes/files default to the ANSI codepage (cp1252/cp1251/cp932),
# which cannot encode the Greek/degree/box glyphs this tool prints and writes.
# Force UTF-8 so runs behave the same on Windows as on macOS/Linux.
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

_sys_pathfix.path.insert(0, _os_pathfix.path.dirname(
    _os_pathfix.path.abspath(__file__)))   # ensure sibling modules importable

# ── Shared reduction core (P1 extraction, 2026-07-03) ────────────────────────
# Curve/COLORS/assign_branch/separate_data/compute_del_l_l0 now live in
# reduce.py; str path uses cmax=None (ratio identically 1.0).
import reduce as _reduce
import cells
from reduce import (COLORS, Curve, custom_round, assign_branch,
                    separate_data, cu_lit_data_fit, compute_del_l_l0)

# ── Global style ─────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.size": 14,
    "axes.linewidth": 1.5,
    "xtick.direction": "in",  "ytick.direction": "in",
    "xtick.top": True,        "ytick.right": True,
    "xtick.major.size": 6,    "ytick.major.size": 6,
    "xtick.minor.size": 3,    "ytick.minor.size": 3,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.width": 1.5, "ytick.major.width": 1.5,
    "lines.linewidth": 2.0,
    "legend.frameon": False,
    "legend.fontsize": 12,
    "axes.labelsize": 16,
    "figure.dpi": 150,
})

# COLORS and Curve are imported from reduce.py (P1 extraction, 2026-07-03).
# The canonical definitions now live there; this module re-exports them via
# the top-of-file `from reduce import ...`.


def build_curves(field_dep, temp_dep, angle_deg, min_points=20):
    """Detect individual sweep segments and wrap each in a Curve object.

    Returns (t_curves, b_curves).
    Uses Mode_Index from separate_data() — do not call before separate_data().
    """
    t_curves, b_curves = [], []

    # ── T-sweep curves ───────────────────────────────────────────────────────
    if not temp_dep.empty:
        td = temp_dep.copy()
        td["B_round"] = td["B [T]"].abs().round(1)
        color_map = {v: COLORS[i % len(COLORS)]
                     for i, v in enumerate(sorted(td["B_round"].unique()))}
        # segment counter per (B_round, direction) for unique labels
        seg_count = {}
        for (b_round, mode_idx), grp in td.groupby(
                ["B_round", "Mode_Index"], sort=False):
            if len(grp) < min_points:
                continue
            direction = "cool" if mode_idx.split("_")[0] == "1" else "warm"
            b_median = float(grp["B [T]"].median())
            if abs(b_median) < 0.05:
                b_median = 0.0
            key = (b_round, direction)
            seg_count[key] = seg_count.get(key, 0) + 1
            n = seg_count[key]
            # Signed one-decimal label (mirrors mini): a bare .0f collapsed
            # e.g. 0.0 T and 0.5 T sweeps to the same "B=0T" text, so the QC
            # radio (win.index(label)) edited the wrong curve. Kept in sync
            # with qc_mini_cell.build_curves per the replication rule.
            label = (f"B={b_median:+.1f}T {direction} #{n}"
                     if b_median != 0.0 else f"B=0T {direction} #{n}")
            grp_sorted = grp.sort_values("T PPMS [K]")
            t_curves.append(Curve(
                kind="T", param_value=b_median, direction=direction,
                mode_index=mode_idx, label=label,
                color=color_map[b_round], angle_deg=angle_deg,
                raw_df=grp_sorted))

    # ── B-sweep curves ───────────────────────────────────────────────────────
    if not field_dep.empty:
        fd = field_dep.copy()
        color_map = {v: COLORS[i % len(COLORS)]
                     for i, v in enumerate(sorted(fd["Rounded_Temp"].unique()))}
        seg_count = {}
        for (t_round, mode_idx), grp in fd.groupby(
                ["Rounded_Temp", "Mode_Index"], sort=False):
            if len(grp) < min_points:
                continue
            direction = "up" if grp["B [T]"].diff().median() > 0 else "down"
            key = (t_round, direction)
            seg_count[key] = seg_count.get(key, 0) + 1
            n = seg_count[key]
            label = f"T={int(t_round)}K {direction} #{n}"
            grp_sorted = grp.sort_values("B [T]")
            b_curves.append(Curve(
                kind="B", param_value=t_round, direction=direction,
                mode_index=mode_idx, label=label,
                color=color_map[t_round], angle_deg=angle_deg,
                raw_df=grp_sorted))

    return t_curves, b_curves


# ── Physical constants ────────────────────────────────────────────────────────
N_A = 6.022e23
k_B = 1.380649e-23


# ════════════════════════════════════════════════════════════════════════════
# Utilities
# ════════════════════════════════════════════════════════════════════════════

def extract_angle(path: str) -> int:
    """Return rotation angle (°) from filename, default 0."""
    name = os.path.basename(path).lower()
    if "plus90"  in name: return  90
    if "plus45"  in name: return  45
    if "minus90" in name: return -90
    if "minus45" in name: return -45
    return 0


# custom_round is imported from reduce.py (P1 extraction).


# ════════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════════

# Raw PPMS columns as recorded in a tab-delimited `.dat` and preserved in the
# comma-separated `_all.csv` archive (which also carries stale computed columns
# we drop). Same list the batch drivers select via usecols; duplicated here
# (P3.5) rather than shared — the plan bounds this phase's blast radius to the
# QC scripts + app and keeps new names out of reduce.py.
RAW_COLS = ["Abs Time", "Rel Time", "T PPMS [K]", "T sample [K]", "B [T]",
            "C [pF]", "L [nS]", "delta l [1E-6 cm]"]


def _is_csv_archive(file_path: str) -> bool:
    """Sniff the format from the first line: True for a comma-separated
    `_all.csv` archive (its first line IS the raw-column header), False for a
    tab-delimited PPMS `.dat` (a title line that skiprows=1 drops, followed by
    the header on line 2 — its first line comma-splits to a single field that
    never equals a raw column name)."""
    with open(file_path, encoding="utf-8") as fh:
        first = fh.readline()
    fields = first.rstrip("\n").split(",")
    return "T PPMS [K]" in fields or "delta l [1E-6 cm]" in fields


def load_data(file_path: str) -> pd.DataFrame:
    """Load one run's raw PPMS data, sniffing the format (P3.5):

      * tab-delimited `.dat` (delimiter='\\t', skiprows=1) — the legacy path;
      * comma-separated `_all.csv` archive (header row) — raw columns only,
        the stale computed columns dropped via usecols.

    Both paths apply identical cleanup and return an identical-shape frame, so
    Open QC works off the archives when no raw `.dat` is present on this Mac."""
    if not os.path.isfile(file_path):
        raise SystemExit(f"input file not found: {file_path}")
    if _is_csv_archive(file_path):
        df = pd.read_csv(file_path, usecols=lambda c: c in RAW_COLS, encoding="utf-8", encoding_errors="replace")
    else:
        try:
            df = pd.read_csv(file_path, delimiter="\t", skiprows=1, encoding="utf-8", encoding_errors="replace")
        except Exception:
            df = pd.DataFrame()
    need = ["T PPMS [K]", "C [pF]", "delta l [1E-6 cm]"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{os.path.basename(file_path)}: no recognizable PPMS dilatometry "
            f"columns (need {', '.join(need)}).\n"
            "Expected a raw PPMS .dat export (tab-delimited, title line first) "
            "or a comma *_all.csv archive — is this the right file?")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=need, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ════════════════════════════════════════════════════════════════════════════
# Calibration fits  (str_dil / 0_42mm coefficients)
# ════════════════════════════════════════════════════════════════════════════

# cu_lit_data_fit is imported from reduce.py (P1 extraction).


def cell_with_cu_fit(T,
        r6=413.53191,    r7=0.12944,      r8=-0.01064,
        r12=-3.13017e-4, r13=6.81607e-6,  r14=-6.1787e-8,
        r15=3.16305e-10, r16=-9.5141e-13, r17=1.56896e-15,
        r18=-1.0956e-18):
    """LEGACY str_dil cell + Cu calibration polynomial [1e-6 cm] (warming
    branch of Cu_0.42mm_str_dil). Superseded by calibrations.json; kept for
    cross-checks only."""
    return (r6 + r7*T + r8*T**2 + r12*T**3 + r13*T**4
            + r14*T**5 + r15*T**6 + r16*T**7 + r17*T**8 + r18*T**9)


# ════════════════════════════════════════════════════════════════════════════
# Calibration registry (calibrations.json — built by cu_calibration_builder.py)
# ════════════════════════════════════════════════════════════════════════════

CELL    = "str_dil"    # registry cell key for this script
USE_EQ7 = True         # P18 Eq. (7) length decomposition (auto-fallback when
                       # not resolved for a branch)
CALIBRATIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "calibrations.json")


def _poly_eval(coeffs):
    c = np.asarray(coeffs, float)
    return lambda T: sum(ci * np.asarray(T, float) ** i
                         for i, ci in enumerate(c))


_EXAMPLE_BANNER_SHOWN = False

def _example_registry_banner(reg, path):
    """One-shot warning when the SHIPPED EXAMPLE registry is in use. The
    marker key is metadata on the shipped file; cu_calibration_builder.py
    never writes it, so building your own registry clears it."""
    global _EXAMPLE_BANNER_SHOWN
    if reg.get("_example_registry") and not _EXAMPLE_BANNER_SHOWN:
        _EXAMPLE_BANNER_SHOWN = True
        print("=" * 72)
        print("NOTE: using the SHIPPED EXAMPLE calibration registry — the")
        print("      AUTHORS' dilatometer cells, not yours. The cell background")
        print(f"      subtracted from your data comes from {os.path.basename(path)}'s")
        print("      example records. Build your own registry from your Cu runs:")
        print("        python3 scripts/cu_calibration_builder.py --help")
        print("      then replace calibrations.json or point DILAT_CALIBRATIONS at yours.")
        print("=" * 72)


def load_calibration(cell=None, sample_thickness_cm=None, use_eq7=None,
                     t_max_needed=None, path=None):
    """Branch-aware cell calibration from the registry.

    Returns {"cool": f, "warm": f, "meta": ...} with f(T) -> delta_l_cell
    [1e-6 cm], to be subtracted from measured delta_l per P18 Eq. (6).

    Per branch:
      * use_eq7 and the branch's Eq. (7) record is resolved -> virtual Cu
        curve at the sample thickness: p_A(T) + D(T)*(L_s - L_A), the exact
        length interpolation of the two Cu-length calibrations.
      * otherwise -> the record whose Cu length is closest to the sample
        thickness (same cell and branch). Records that do not cover
        t_max_needed are deprioritized; ties prefer the primary (non-350K)
        file, then lower cycle number.
    """
    cell = CELL if cell is None else cell
    use_eq7 = USE_EQ7 if use_eq7 is None else use_eq7
    # Single-source override (review #5): DILAT_CALIBRATIONS points every code
    # path at one canonical calibrations.json; unset -> local copy (default).
    path = path or os.environ.get("DILAT_CALIBRATIONS") or CALIBRATIONS_PATH
    with open(path, encoding="utf-8") as fh:
        reg = json.load(fh)
    _example_registry_banner(reg, path)
    recs = {r["id"]: r for r in reg["records"]}
    cal = {"meta": {"registry": path, "cell": cell,
                    "use_eq7": use_eq7, "per_branch": {},
                    "example_registry": bool(reg.get("_example_registry")),
                    "field_background": reg.get("field_background"),
                    "transfer": reg.get("transfer", {}).get(cell)}}
    for branch in ("cool", "warm"):
        e = reg.get("eq7", {}).get(cell, {}).get(branch)
        # eq7 must also roughly COVER the run's T range; otherwise its
        # fixed-T_max polynomial would extrapolate far past its fit (e.g. a
        # 350 K run on 300 K records, ~0.8e-3 near 350 K). A small overhang
        # (<= 5 K) is tolerated: a correct-thickness eq7 curve extrapolated
        # a few K beats a wrong-thickness closest-length record (US mini
        # rot0: 300.02 K data on a 296.99 K cool eq7 fit).
        eq7_covers = bool(e and e["resolved"]
                          and (t_max_needed is None
                               or e["T_range"][1] >= t_max_needed - 5.0))
        if use_eq7 and eq7_covers and sample_thickness_cm is not None:
            ra = recs[e["records"][0]]
            La = ra["cu_length_mm"] / 10.0
            pa = np.asarray(ra["coefficients"], float)
            D  = np.asarray(e["dl2_poly_coefficients"], float)
            cal[branch] = _poly_eval(pa + D * (sample_thickness_cm - La))
            cal["meta"]["per_branch"][branch] = {
                "mode": "eq7_virtual", "records": e["records"],
                "T_range": e["T_range"]}
        else:
            cands = [r for r in reg["records"]
                     if r["cell"] == cell and r["branch"] == branch]
            if not cands:
                raise ValueError(f"no {cell}/{branch} records in {path}")
            L_s = sample_thickness_cm

            def rank(r):
                covers = (t_max_needed is None
                          or r["T_fit_max"] >= t_max_needed - 0.5)
                dist = (abs(r["cu_length_mm"] / 10.0 - L_s)
                        if L_s is not None else 0.0)
                return (not covers, dist, "350K" in r["source_file"],
                        r["cycle"], -(r["T_fit_max"] - r["T_fit_min"]))

            best = sorted(cands, key=rank)[0]
            cal[branch] = _poly_eval(best["coefficients"])
            note = (None if not use_eq7 else
                    None if e is None else
                    "eq7 not resolved for this branch -> closest-length"
                    if not e["resolved"] else
                    "eq7 T_range does not cover t_max_needed -> closest-length"
                    if not eq7_covers else None)
            cal["meta"]["per_branch"][branch] = {
                "mode": "closest_length", "records": [best["id"]],
                "T_range": [best["T_fit_min"], best["T_fit_max"]],
                **({"note": note} if note else {})}
    return cal


# assign_branch, compute_del_l_l0, separate_data are imported from reduce.py
# (P1 extraction, 2026-07-03). The str path calls compute_del_l_l0(df, L0, cal)
# -> reduce.compute_del_l_l0 with cmax=None, i.e. ratio identically 1.0 and no
# Cmax_ratio column, byte-identical to the former local implementation.


# ════════════════════════════════════════════════════════════════════════════
# Grüneisen / Debye model
# ════════════════════════════════════════════════════════════════════════════

def _debye_integrand(y):
    return 1.0 if y == 0 else y**4 * np.exp(y) / (np.exp(y) - 1)**2

def _debye_integral(x):
    if x > 700: return 0.0
    return quad(_debye_integrand, 0, x)[0]

def debye_specific_heat(T, theta_D):
    T  = np.asarray(T, dtype=float)
    Cv = np.zeros_like(T)
    for idx, t in np.ndenumerate(T):
        if t > 0:
            Cv[idx] = 9*N_A*k_B*(t/theta_D)**3 * _debye_integral(theta_D/t)
    return Cv

def thermal_expansion_model(T, B_mod, V, theta_D, alpha):
    Cv = debye_specific_heat(np.asarray(T), theta_D)
    return (alpha / (3 * B_mod * V)) * Cv

def fit_debye_model(T_arr, dL_arr):
    p0     = [130e9, 2.7e-9, 450, 1.0]
    bounds = ([1e9, 1e-11, 50, -100], [1e14, 1e-6, 2000, 100])
    try:
        popt, _ = curve_fit(
            lambda T, B, V, th, a: thermal_expansion_model(T, B, V, th, a),
            T_arr, dL_arr, p0=p0, bounds=bounds, maxfev=10000)
        return popt
    except Exception as e:
        print(f"  Debye fit failed: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# Publication-quality plots
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    HERE = os.path.dirname(os.path.abspath(__file__))
    PROC = os.path.dirname(HERE)
    ap = argparse.ArgumentParser(description="str-cell interactive analysis")
    ap.add_argument("--data", default=os.path.join(PROC, "Data"))
    ap.add_argument("--out", default=os.path.join(PROC, "Output", "str"))
    ap.add_argument("--file", default=None, required=False,
                    help="raw PPMS .dat (or *_all.csv archive) in --data")
    ap.add_argument("--thickness", type=float, default=0.05,
                    help="sample thickness L0 in cm")
    ap.add_argument("--precleanup", action="store_true",
                    help="save _Tdep_raw/_Bdep_raw (separated+recalculated, "
                         "no cleanup) and exit without opening QC windows")
    args = ap.parse_args()
    DATA_FOLDER, OUT_FOLDER = args.data, args.out
    os.makedirs(OUT_FOLDER, exist_ok=True)

    SAMPLE_THICKNESS = args.thickness   # cm (L0 along the measurement axis)

    # ── File to process ──────────────────────────────────────────────────────
    if not args.file:
        raise SystemExit(
            "no input given: pass --file <name>.dat (raw PPMS export) or a "
            "comma *_all.csv archive, with --data <folder> if needed.")
    file_path = os.path.join(DATA_FOLDER, args.file)

    angle      = extract_angle(file_path)
    stem       = _reduce.output_stem(file_path)
    out_prefix = os.path.join(OUT_FOLDER, stem)
    print(f"Processing: {stem}  (θ = {angle:+d}°)")

    # ── Load, segment, then reduce branch-aware ─────────────────────────────
    df = load_data(file_path)
    field_dep, temp_dep = separate_data(
        df, field_change_threshold=cells.STR_DIL["field_change_threshold_interactive"])
    full = pd.concat([temp_dep, field_dep]).sort_index()
    full["Branch"] = assign_branch(full, field_dep.index)

    cal = load_calibration(CELL, SAMPLE_THICKNESS,
                           t_max_needed=float(full["T PPMS [K]"].max()))
    for br, m in cal["meta"]["per_branch"].items():
        print(f"  calibration {br}: {m['mode']} {m['records']}"
              + (f"  ({m['note']})" if m.get("note") else ""))
    fb = cal["meta"]["field_background"]
    if fb and not fb.get("apply", False) and not field_dep.empty:
        print("  field background: NOT applied (unresolved, see registry) — "
              "B-sweep cell-background uncertainty envelope 0.1-2.1e-6 cm "
              "(T<=50 K), larger where drift-dominated.")

    full = compute_del_l_l0(full, SAMPLE_THICKNESS, cal)
    temp_dep = full.loc[temp_dep.index]
    field_dep = full.loc[field_dep.index] if not field_dep.empty else field_dep
    df = full

    # ── Save full processed dataset ──────────────────────────────────────────
    df.to_csv(f"{out_prefix}_all.csv", index=False, encoding="utf-8")
    print(f"  Full CSV: {out_prefix}_all.csv")

    # ── Build curve objects ──────────────────────────────────────────────────
    t_curves, b_curves = build_curves(field_dep, temp_dep, angle)
    print(f"  T-curves: {len(t_curves)}   B-curves: {len(b_curves)}")

    # ── Pre-cleanup plots (separated+recalculated, no cleanup) then exit ─────
    if args.precleanup:
        print("  [PRE-CLEANUP] plotting raw separated+recalculated curves.")
        if t_curves:
            plot_temperature_dep(t_curves, angle, out_prefix, use_raw=True)
        else:
            print("  No T-curves found.")
        if b_curves:
            plot_field_dep(b_curves, angle, out_prefix, use_raw=True)
        else:
            print("  No B-curves found.")
        plt.close("all")
        return

    # ── Auto-load previous QC state if sidecar exists ───────────────────────
    qc_state_path = f"{out_prefix}_qc_state.json"
    load_qc_state(t_curves, b_curves, qc_state_path)

    # ── T-sweep QC window ────────────────────────────────────────────────────
    if t_curves:
        win_t = QCWindow(t_curves, kind="T", angle_deg=angle,
                         out_prefix=out_prefix)
        win_t.show()
        if not win_t.exported:
            print("  [EXPORT] window closed without Export — saving now.")
            plot_temperature_dep(t_curves, angle, out_prefix)
    else:
        print("  No T-curves found.")

    # ── B-sweep QC window ────────────────────────────────────────────────────
    if b_curves:
        win_b = QCWindow(b_curves, kind="B", angle_deg=angle,
                         out_prefix=out_prefix)
        win_b.show()
        if not win_b.exported:
            print("  [EXPORT] window closed without Export — saving now.")
            plot_field_dep(b_curves, angle, out_prefix)
    else:
        print("  No B-curves found.")

    # ── Save QC state ────────────────────────────────────────────────────────
    save_qc_state(t_curves, b_curves, qc_state_path)
    plt.show()


if __name__ == "__main__":
    main()
