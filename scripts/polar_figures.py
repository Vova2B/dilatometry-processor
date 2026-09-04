"""
polar_figures.py
Polar plot of (ΔL/L₀) vs rotation angle from mini dilatometer data.

Workflow
────────
1. Finds all mini_dil angle files in DATA_FOLDER.
2. Processes each with Cmax=50 correction (same as qc_mini_cell).
3. Extracts (ΔL/L₀) at query temperatures T_QUERY by interpolation.
4. Plots on polar axes: azimuth = rotation angle, radius = (ΔL/L₀)×10³.
5. Saves polar_data.csv and polar.png.

Angle convention: angles in the filename are the rotation angle of the
mini dilatometer in degrees. The plot assumes 2-fold symmetry and mirrors
data to produce a full 360° polar diagram.
"""

from __future__ import annotations

# Windows: pipes/files default to the ANSI codepage (cp1252/cp1251/cp932),
# which cannot encode the Greek/degree/box glyphs this tool prints and writes.
# Force UTF-8 so runs behave the same on Windows as on macOS/Linux.
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d

# ── Style ─────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.size": 14,
    "axes.linewidth": 1.5,
    "lines.linewidth": 2.0,
    "legend.frameon": False,
    "legend.fontsize": 12,
    "axes.labelsize": 16,
    "figure.dpi": 150,
})

COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]

# ── Constants ─────────────────────────────────────────────────────────────────
CMAX_PPMS = 100.0
CMAX_TRUE =  50.0

# Query temperatures (K) to extract for polar plot
T_QUERY = [10, 50, 100, 200]


# ════════════════════════════════════════════════════════════════════════════
# Shared processing (mirrors qc_mini_cell)
# ════════════════════════════════════════════════════════════════════════════

def extract_angle(path: str) -> int:
    name = os.path.basename(path).lower()
    if "plus90"  in name: return  90
    if "plus45"  in name: return  45
    if "minus90" in name: return -90
    if "minus45" in name: return -45
    return 0


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter="\t", skiprows=1, encoding="utf-8", encoding_errors="replace")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["T PPMS [K]", "C [pF]", "delta l [1E-6 cm]"],
              inplace=True)
    df = df[df["C [pF]"] > 0]   # drop bridge-not-locked startup rows
    df.reset_index(drop=True, inplace=True)
    return df


def cu_lit_data_fit(T, r3=16549.7494, r4=-578.02392, r5=56.45939):
    return r3 * np.exp(r4 / (T + r5))


def cell_with_cu_fit(T,
        r6=376.02909,    r7=-0.18085,     r8=0.0143,
        r12=-6.69301e-4, r13=8.67563e-6,  r14=-5.89518e-8,
        r15=2.3577e-10,  r16=-5.62752e-13, r17=7.489e-16,
        r18=-4.31625e-19):
    """LEGACY mini-cell warming poly [1e-6 cm]; superseded by
    calibrations.json (kept for cross-checks only)."""
    return (r6 + r7*T + r8*T**2 + r12*T**3 + r13*T**4
            + r14*T**5 + r15*T**6 + r16*T**7 + r17*T**8 + r18*T**9)


# ── Calibration registry (calibrations.json, cu_calibration_builder.py) ─────
# This script keeps only B≈0 COOLING data, so it uses the cooling branch of
# the mini-cell registry calibration (Eq. 7 virtual curve at the sample
# thickness when resolved, else closest Cu length).

CELL    = "mini_dil"
USE_EQ7 = True
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


def load_cooling_calibration(sample_thickness_cm, path=None,
                             t_max_needed=None):
    """Cooling-branch registry calibration: f(T) -> delta_l_cell [1e-6 cm].

    Same selection rules as the QC load_calibration pair: eq7 is used only
    when resolved AND its T_range covers t_max_needed (no silent
    extrapolation); the closest-length fallback deprioritizes records that
    do not cover t_max_needed."""
    path = CALIBRATIONS_PATH if path is None else path
    with open(path, encoding="utf-8") as fh:
        reg = json.load(fh)
    _example_registry_banner(reg, path)
    e = reg.get("eq7", {}).get(CELL, {}).get("cool")
    # <= 5 K eq7 overhang tolerated — same rule as the QC pair: a
    # correct-thickness eq7 curve extrapolated a few K beats a
    # wrong-thickness closest-length record.
    eq7_covers = bool(e and e["resolved"]
                      and (t_max_needed is None
                           or e["T_range"][1] >= t_max_needed - 5.0))
    if USE_EQ7 and eq7_covers:
        recs = {r["id"]: r for r in reg["records"]}
        ra = recs[e["records"][0]]
        La = ra["cu_length_mm"] / 10.0
        pa = np.asarray(ra["coefficients"], float)
        D  = np.asarray(e["dl2_poly_coefficients"], float)
        print(f"  calibration cool: eq7_virtual {e['records']}")
        return _poly_eval(pa + D * (sample_thickness_cm - La))
    cands = [r for r in reg["records"]
             if r["cell"] == CELL and r["branch"] == "cool"]
    best = sorted(cands, key=lambda r: (
        not (t_max_needed is None or r["T_fit_max"] >= t_max_needed - 0.5),
        abs(r["cu_length_mm"] / 10.0 - sample_thickness_cm), r["cycle"]))[0]
    print(f"  calibration cool: closest_length ['{best['id']}']")
    return _poly_eval(best["coefficients"])


_CAL_COOL_CACHE = {}

def _get_cal_cool(L0, t_max_needed=None):
    key = (L0, t_max_needed)
    if key not in _CAL_COOL_CACHE:
        _CAL_COOL_CACHE[key] = load_cooling_calibration(
            L0, t_max_needed=t_max_needed)
    return _CAL_COOL_CACHE[key]


def estimate_C0(df: pd.DataFrame) -> float:
    low_field = df[df["B [T]"].abs() < 0.1]
    row  = low_field.iloc[0] if not low_field.empty else df.iloc[0]
    C1   = row["C [pF]"]
    dl1  = row["delta l [1E-6 cm]"]
    K    = 136_300.0
    corr = 1.0 - C1**2 / CMAX_PPMS**2
    return C1 - dl1 * C1**2 / (K * corr)


def cmax_ratio(C_series: pd.Series, C0: float) -> pd.Series:
    CC0 = C_series * C0
    return (1.0 - CC0 / CMAX_TRUE**2) / (1.0 - CC0 / CMAX_PPMS**2)


def compute_del_l_l0(df: pd.DataFrame, L0: float, C0: float,
                     cal_cool) -> pd.DataFrame:
    # cooling registry calibration for every row: only B≈0 cooling rows
    # survive process_angle_file(), so the warming rows discarded downstream
    # never reach the output.
    T     = df["T PPMS [K]"].values
    dl    = df["delta l [1E-6 cm]"].values
    cf    = cal_cool(T)
    cu    = cu_lit_data_fit(T)
    ratio = cmax_ratio(df["C [pF]"], C0).values

    # P18 Eq. (6): delta_l and cell_fit share the PPMS native sign (positive =
    # sample expansion; cooling background is frame contraction) — subtract directly.
    total   = (dl - cf) * 1e-6 / L0 * ratio + cu * 1e-6
    total  -= total[int(np.argmin(T))]
    df["(del_L/L_0)_Sam"] = total
    return df


def get_Bsweep_rows(df, field_threshold=3, temp_threshold=0.7):
    """Return row indices that belong to field-sweep segments."""
    tc = df["T PPMS [K]"].diff().abs()
    fc = df["B [T]"].diff().abs() * 10000
    return df[(fc > field_threshold) & (tc < temp_threshold)].index


# ════════════════════════════════════════════════════════════════════════════
# Process one angle file → (T, ΔL/L₀) for B ≈ 0 cooling sweep
# ════════════════════════════════════════════════════════════════════════════

def process_angle_file(path: str, L0: float) -> pd.DataFrame | None:
    """
    Load and process one angle file. Returns B≈0 temperature-sweep data
    (cooling segment) as a DataFrame with columns [T_K, dL_L0].
    Returns None if processing fails.
    """
    try:
        df   = load_data(path)
        C0   = estimate_C0(df)
        cal  = _get_cal_cool(L0, t_max_needed=float(df["T PPMS [K]"].max()))
        df   = compute_del_l_l0(df, L0, C0, cal)

        # Keep only B ≈ 0 rows
        df_b0 = df[df["B [T]"].abs() < 0.05].copy()
        if df_b0.empty:
            return None

        # Keep only cooling rows (T decreasing)
        tdiff = df_b0["T PPMS [K]"].diff()
        cool  = df_b0[tdiff.fillna(-1) < 0].copy()
        if cool.empty:
            cool = df_b0  # fall back to all B=0 data

        cool = cool.sort_values("T PPMS [K]")
        return cool[["T PPMS [K]", "(del_L/L_0)_Sam"]].rename(
            columns={"T PPMS [K]": "T_K", "(del_L/L_0)_Sam": "dL_L0"})
    except Exception as e:
        print(f"  Warning: could not process {os.path.basename(path)}: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# Polar plot
# ════════════════════════════════════════════════════════════════════════════

def fold_2fold(angles_deg, vals, label=""):
    """Canonicalize measured angles into [0, 360), add the θ+180° mirror
    (2-fold symmetry), and MERGE nodes landing on the same canonical angle
    by averaging (e.g. a measured −90° and the mirror of +90° both fall on
    270°). A disagreement between merged values is printed as a symmetry
    check, not silently dropped. Returns sorted (angles, values), open
    loop (caller closes it). Duplicated in reduce_mini_batch.polar_draw
    per the standalone-scripts convention — keep in sync."""
    a = np.asarray(angles_deg, float) % 360.0
    v = np.asarray(vals, float)
    a_full = np.concatenate([a, (a + 180.0) % 360.0])
    v_full = np.concatenate([v, v])
    ang_u = np.unique(a_full)
    merged = []
    for au in ang_u:
        vv = v_full[np.isclose(a_full, au)]
        spread = float(vv.max() - vv.min())
        if spread > 1e-9:
            print(f"  2-fold symmetry check{label}: {au:.0f} deg nodes "
                  f"differ by {spread:.3g} — averaged")
        merged.append(float(vv.mean()))
    return ang_u, np.asarray(merged)


def interpolate_at_T(df: pd.DataFrame, T_q: float) -> float | None:
    """
    Interpolate (ΔL/L₀) at query temperature T_q.
    Returns None if T_q is outside the data range.
    """
    T   = df["T_K"].values
    dL  = df["dL_L0"].values
    if T_q < T.min() or T_q > T.max():
        return None
    f = interp1d(T, dL, kind="linear", bounds_error=False,
                 fill_value="extrapolate")
    return float(f(T_q))


def build_polar_data(angle_files: list, L0: float) -> pd.DataFrame:
    """
    Process all angle files and collect (angle, T_query, ΔL/L₀) rows.
    """
    rows = []
    for path in angle_files:
        angle = extract_angle(path)
        print(f"  Processing angle {angle:+d}°: {os.path.basename(path)}")
        result = process_angle_file(path, L0)
        if result is None:
            print(f"    → skipped (no usable data)")
            continue
        for T_q in T_QUERY:
            val = interpolate_at_T(result, T_q)
            if val is not None:
                rows.append({"angle_deg": angle, "T_K": T_q, "dL_L0": val})
            else:
                print(f"    → T = {T_q} K outside range "
                      f"[{result.T_K.min():.0f}, {result.T_K.max():.0f}] K")
    return pd.DataFrame(rows)


def plot_polar(polar_df: pd.DataFrame, out_prefix: str) -> plt.Figure:
    """
    Polar plot of (ΔL/L₀) × 10³ vs rotation angle.

    Data are mirrored to produce a full 360° diagram (assumes 2-fold
    symmetry: ΔL/L₀(θ + 180°) = ΔL/L₀(θ), valid for orthorhombic
    and higher crystal symmetry).

    One curve per query temperature.
    Saves <out_prefix>_polar.{png,csv}.
    """
    if polar_df.empty:
        print("  No polar data to plot.")
        return None

    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")   # 0° at top
    ax.set_theta_direction(-1)        # clockwise (convention for rotation)

    csv_rows = []

    for i, T_q in enumerate(sorted(polar_df["T_K"].unique())):
        sub = polar_df[polar_df["T_K"] == T_q].sort_values("angle_deg")
        if sub.empty:
            continue

        angles_deg = sub["angle_deg"].values
        vals       = sub["dL_L0"].values * 1e3   # ×10³

        # Canonical [0,360) fold + 2-fold mirror; colliding nodes (e.g.
        # measured −90° vs the mirror of +90°, both at 270°) are AVERAGED
        # and their spread printed — the symmetry is checked, not assumed.
        ang_s, val_s = fold_2fold(angles_deg, vals, label=f" (T={int(T_q)} K)")

        # Shift so minimum = 0 (polar radius must be ≥ 0) — computed before
        # closing the loop so scatter points and curve share the same offset
        offset = val_s.min()

        # Close the loop for smooth interpolation
        ang_s = np.append(ang_s, ang_s[0] + 360)
        val_s = np.append(val_s, val_s[0])

        # Interpolate smooth curve
        if len(ang_s) > 3:
            ang_fine = np.linspace(ang_s[0], ang_s[-1], 360)
            f_interp = interp1d(ang_s, val_s, kind="linear")
            val_fine = f_interp(ang_fine)
        else:
            ang_fine, val_fine = ang_s, val_s

        ang_rad = np.deg2rad(ang_fine)
        color   = COLORS[i % len(COLORS)]

        ax.plot(ang_rad, val_fine - offset,
                color=color, lw=2, label=f"$T = {int(T_q)}$ K")
        ax.scatter(np.deg2rad(angles_deg % 360.0), vals - offset,
                   color=color, zorder=5, s=40)

        for a, v in zip(angles_deg, vals):
            csv_rows.append({"angle_deg": a, "T_K": T_q, "dL_L0_x1e3": v})

    ax.set_xlabel("")
    ax.set_title(r"$\Delta L / L_0 \times 10^{-3}$ vs rotation angle",
                 pad=20, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

    # Degree labels on angular axis
    ax.set_xticks(np.deg2rad(range(0, 360, 45)))
    ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°",
                        "225°", "270°", "315°"])

    pd.DataFrame(csv_rows).to_csv(f"{out_prefix}_polar.csv", index=False, encoding="utf-8")
    fig.savefig(f"{out_prefix}_polar.png", dpi=200, bbox_inches="tight")
    print(f"  Saved: {out_prefix}_polar.{{png,csv}}")
    return fig


def plot_polar_Tdep(angle_files: list, L0: float, out_prefix: str
                    ) -> plt.Figure:
    """
    Overlay (ΔL/L₀) vs T curves for all angles on a single Cartesian plot,
    colour-coded by angle. Saved as <out_prefix>_polar_Tdep.{png,csv}.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.subplots_adjust(left=0.15, right=0.95, top=0.93, bottom=0.13)

    csv_rows = []

    for i, path in enumerate(sorted(angle_files,
                                    key=lambda p: extract_angle(p))):
        angle  = extract_angle(path)
        result = process_angle_file(path, L0)
        if result is None:
            continue
        color     = COLORS[i % len(COLORS)]
        angle_str = f"{angle:+d}" if angle != 0 else "0"
        ax.plot(result["T_K"], result["dL_L0"] * 1e3,
                color=color, label=fr"$\theta = {angle_str}°$")
        for _, row in result.iterrows():
            csv_rows.append({
                "T_K":      row["T_K"],
                "dL_L0":    row["dL_L0"],
                "angle_deg": angle,
            })

    ax.set_xlabel(r"$T$ (K)")
    ax.set_ylabel(r"$\Delta L / L_0 \times 10^{-3}$")
    ax.legend(loc="best")
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)

    pd.DataFrame(csv_rows).to_csv(f"{out_prefix}_polar_Tdep.csv", index=False, encoding="utf-8")
    fig.savefig(f"{out_prefix}_polar_Tdep.png", dpi=200, bbox_inches="tight")
    print(f"  Saved: {out_prefix}_polar_Tdep.{{png,csv}}")
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    HERE_ = os.path.dirname(os.path.abspath(__file__))
    PROC_ = os.path.dirname(HERE_)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--data", default=os.path.join(PROC_, "Data"),
                    help="folder with raw PPMS .dat files")
    ap.add_argument("--out", default=os.path.join(PROC_, "Output", "mini"),
                    help="output folder")
    ap.add_argument("--glob", default="*mini_dil*.dat",
                    help="filename pattern of the angle files in --data")
    ap.add_argument("--stem", default="mini_dil",
                    help="output filename prefix")
    ap.add_argument("--thickness", type=float, default=0.02,
                    help="sample thickness L0 in cm")
    args = ap.parse_args()
    DATA_FOLDER = args.data
    OUT_FOLDER  = args.out
    os.makedirs(OUT_FOLDER, exist_ok=True)

    SAMPLE_THICKNESS = args.thickness   # cm

    # ── Find all angle files ─────────────────────────────────────────────────
    pattern = os.path.join(DATA_FOLDER, args.glob)
    angle_files = sorted(glob.glob(pattern))

    if not angle_files:
        print(f"No files matching: {pattern}")
        return

    print(f"Found {len(angle_files)} angle file(s):")
    for f in angle_files:
        print(f"  {os.path.basename(f)}  →  θ = {extract_angle(f):+d}°")

    out_prefix = os.path.join(OUT_FOLDER, args.stem)

    # ── Temperature-dependence overlay (all angles, Cartesian) ────────────────
    fig_Tdep = plot_polar_Tdep(angle_files, SAMPLE_THICKNESS, out_prefix)

    # ── Polar plot at query temperatures ─────────────────────────────────────
    polar_df = build_polar_data(angle_files, SAMPLE_THICKNESS)
    if not polar_df.empty:
        fig_polar = plot_polar(polar_df, out_prefix)
    else:
        print("  Polar data empty — check that T_QUERY values are in range.")

    plt.show()


if __name__ == "__main__":
    main()
