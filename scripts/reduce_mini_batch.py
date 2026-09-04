"""
reduce_mini_batch.py — headless reduction of a mini-cell rotation series
(the mini cell is rotatable 0-90 deg in the PPMS, P20).

The angle-run list comes from an angle_runs.json placed in the data folder
(or passed via --runs), e.g.:

  {"stem": "MYSAMPLE_mini",
   "L0_cm": 0.02,
   "transition_K": 100.0,
   "runs": [{"angle_deg": 0,  "tag": "rot0",   "glob": "*rot0*_all.csv"},
            {"angle_deg": 45, "tag": "plus45", "glob": "*45deg*.dat",
             "rescale": 1.0}]}

`rescale` is an optional per-run raw-delta_l factor (use it when a run was
PPMS-converted with the wrong plate radius). No mini-cell field-background
record is shipped — magnetostriction loops carry no field correction.

Mini reduction applies the Cmax correction (true mini Cmax vs the
PPMS-baked 100 pF) via qc_mini_cell.compute_del_l_l0.

Outputs (Output/mini/): per angle _Tdep_clean.{csv,png}, _Bdep_clean
.{csv,png} (when loops exist), _provenance.json; combined
{stem}_Tdep_all_angles.png (referenced at 200 K) and
{stem}_lambda_vs_angle.png (polar-style anisotropy).
"""

import glob
import json
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))      # scripts/
PROC = os.path.dirname(HERE)                            # Dilatometry processor/
sys.path.insert(0, HERE)
import qc_mini_cell as M
import cells
import transition as _tr
from reduce_str_batch import (glitch_mask, alpha_of, binned_alpha,
                              safe_relpath,
                              build_b_loop_curves, prepend_header,
                              cap_legend, clean_t_curves, RAW_COLS)
import qc_str_cell as S   # Curve/COLORS used by shared helpers

SRC_DIR = os.path.join(PROC, "Data")   # default; use --data
OUT_DIR = os.path.join(PROC, "Output", "mini")
# Sample parameters — defaults; override per dataset via angle_runs.json
# keys "L0_cm" and "transition_K" (see module docstring).
L0 = 0.02           # cm — sample thickness along the measurement axis
L0_SOURCE = "angle_runs.json / module default — record your own provenance"
TRANSITION = _tr.Transition(None, "none")  # optional transition drawn on figures

# Combined-output filename prefix; per-angle outputs are f"{STEM}_{tag}".
# Normally set by the "stem" key of angle_runs.json.
STEM = "sample_mini"
DL_SPACING = 0.0    # K; thin dL/L0 output (0 = every point). Set in main().
# cleanup toggles (set in main from the CLI; mini defaults: stitching OFF)
STITCH_T = False
STITCH_B = False
DWELL = True
SMOOTH = True


def load_runs_spec(runs_path, data_dir):
    """Resolve the angle-run list: --runs JSON > <data>/angle_runs.json.
    Returns (stem, [(angle, tag, glob, rescale)])."""
    path = runs_path or os.path.join(data_dir, "angle_runs.json")
    if not os.path.isfile(path):
        if runs_path:
            raise SystemExit(f"--runs {runs_path!r}: file not found")
        raise SystemExit(
            f"no angle-run spec: put an angle_runs.json in {data_dir!r} or "
            "pass --runs <spec.json>.\nSee the module docstring of "
            "reduce_mini_batch.py (or README.md) for the format.")
    with open(path, encoding="utf-8") as fh:
        spec = json.load(fh)
    try:
        runs = [(int(r["angle_deg"]), str(r["tag"]), str(r["glob"]),
                 float(r.get("rescale", 1.0))) for r in spec["runs"]]
    except (KeyError, TypeError, ValueError) as e:
        raise SystemExit(
            f"{path}: each entry in 'runs' needs angle_deg, tag, glob "
            f"(optional rescale). Problem: {e}")
    if not runs:
        raise SystemExit(f"{path}: 'runs' list is empty")
    stem = str(spec.get("stem", STEM))
    global L0, L0_SOURCE, TRANSITION
    if "L0_cm" in spec:
        L0 = float(spec["L0_cm"])
        L0_SOURCE = f"L0_cm from {os.path.basename(path)}"
    if "transition_K" in spec:
        _kind = spec.get("transition_type", "T_C")
        if _kind not in _tr.KINDS:
            raise SystemExit(
                f"{path}: transition_type must be one of {_tr.KINDS}, "
                f"got {_kind!r}")
        TRANSITION = _tr.Transition(float(spec["transition_K"]), _kind)
    print(f"angle runs: {path} ({len(runs)} runs, stem={stem!r}, "
          f"L0={L0} cm, transition={TRANSITION.value} {TRANSITION.kind})")
    return stem, runs


def load_raw(pattern):
    hits = sorted(glob.glob(os.path.join(SRC_DIR, pattern)))
    assert len(hits) == 1, f"pattern {pattern!r} matched {hits}"
    # S.load_data sniffs tab-delimited `.dat` vs comma `_all.csv` archive and
    # applies inf->NaN, T/C/delta_l dropna, reset_index; add the mini C>0
    # filter and magnet-off B fill.
    df = S.load_data(hits[0])
    df = df[df["C [pF]"] > 0]
    df["B [T]"] = df["B [T]"].fillna(0.0)
    return df.reset_index(drop=True), hits[0]


def thin_in_T(T, y, dT=1.9):
    """Indices spaced evenly in T (not in index) for symbol-only plotting:
    time-dense dwell regions otherwise pile markers into black bands. dT is
    fixed in axis units so short curves get the same marker density as
    full-range sweeps (~160 markers across a 0-305 K axis at 1.9 K)."""
    T = np.asarray(T, float)
    keep, last = [0], T[0]
    for i in range(1, len(T)):
        if np.isfinite(T[i]) and abs(T[i] - last) >= dT:
            keep.append(i)
            last = T[i]
    return np.asarray(keep)


def settle_filter(t_curves):
    n = 0
    for c in t_curves:
        d = c.raw_df.sort_values("Rel Time")
        T_run = d["T PPMS [K]"]
        if c.direction == "cool":
            keep = T_run <= T_run.cummin() + 0.7
        else:
            keep = T_run >= T_run.cummax() - 0.7
        n += int((~keep).sum())
        c.raw_df = d[keep].reset_index(drop=True)
        c._cache_key = None
    return n


def reduce_one(angle, tag, pattern, rescale=1.0):
    df, src = load_raw(pattern)
    if rescale != 1.0:
        df["delta l [1E-6 cm]"] *= rescale
    field_dep, temp_dep = M.separate_data(
        df, field_change_threshold=cells.MINI_DIL["field_change_threshold"])
    full = pd.concat([temp_dep, field_dep]).sort_index()
    full["Branch"] = M.assign_branch(full, field_dep.index)
    cal = M.load_calibration("mini_dil", L0,
                             t_max_needed=float(full["T PPMS [K]"].max()))
    C0 = M.estimate_C0(df)
    full = M.compute_del_l_l0(full, L0, C0, cal)

    gmask, n_jumps = glitch_mask(full)
    if gmask.any():
        full = full[~gmask]
    temp_dep = full.loc[full.index.intersection(temp_dep.index)]
    field_dep = (full.loc[full.index.intersection(field_dep.index)]
                 if not field_dep.empty else field_dep)

    # paramagnetic alpha check (185-215 K window exists for every angle)
    T = temp_dep["T PPMS [K]"].values
    y = temp_dep["(del_L/L_0)_Sam"].values
    br = temp_dep["Branch"].values
    b0 = np.abs(temp_dep["B [T]"].values) < 0.1
    alpha = {}
    for d in ("cool", "warm"):
        m = b0 & (br == d)
        alpha[d] = (alpha_of(T[m], y[m], 185, 215) * 1e6
                    if m.sum() > 100 else np.nan)
    ok_a = all(not np.isfinite(a) or a > -5 for a in alpha.values())

    out_prefix = os.path.join(OUT_DIR, f"{STEM}_{tag}")
    header = [
        f"reduce_mini_batch.py (2026-07-02) — mini cell, angle {angle:+d} deg",
        f"source: {safe_relpath(src, PROC)} (raw PPMS columns only)",
        f"sample stem: {STEM}; mini cell (P20), rotatable in PPMS",
        f"L0 = {L0} cm — {L0_SOURCE}",
        (f"delta_l RESCALED x{rescale:.4f} (per angle_runs.json — e.g. run "
         "PPMS-converted with the wrong plate radius)")
        if rescale != 1.0 else "delta_l as recorded (rescale = 1)",
        "calibration: calibrations.json, " + "; ".join(
            f"{b}: {m['mode']} {m['records']}"
            for b, m in cal["meta"]["per_branch"].items()),
        f"C0 = {C0:.3f} pF; Cmax correction applied (mini true Cmax vs "
        "PPMS 100 pF)",
        "field background: NONE for mini cell (no Cu field record)",
        "B-loops NOT step-stitched: below-transition staircase can be "
        "domain-avalanche/remanence signal — stitching would rectify it "
        "into fake drift",
        f"glitch gate: {n_jumps} re-zero jumps, "
        f"{int(gmask.sum())} rows excluded",
        f"paramagnetic alpha(185-215 K) cool/warm = "
        f"{alpha['cool']:+.1f}/{alpha['warm']:+.1f} x1e-6/K "
        f"[{'OK' if ok_a else 'SUSPECT'}; absolute value depends on the "
        "confirmed L0]",
        "read with: pandas.read_csv(path, comment='#')",
    ]

    t_curves, _ = M.build_curves(field_dep, temp_dep, angle)
    for c in t_curves:
        if 0 < c.param_value < 1:
            c.label = c.label.replace("B=0T", f"B={c.param_value:g}T")
    t_curves, cstats = clean_t_curves(
        t_curves, S.Curve,
        jump_thr=(0.03e-3 if STITCH_T else None),
        dwell=DWELL, smooth=SMOOTH)
    n_settle = cstats["settling_rows"] + cstats["dwell_rows"]
    color_by_sweep = {}
    for c in t_curves:
        base = c.label.lstrip("_")
        if base not in color_by_sweep:
            color_by_sweep[base] = S.COLORS[len(color_by_sweep)
                                            % len(S.COLORS)]
        c.color = color_by_sweep[base]
    fig = M.plot_temperature_dep(t_curves, angle, out_prefix,
                                 spacing_k=DL_SPACING)
    if fig is not None:
        cap_legend(fig)
        fig.axes[0].set_xlim(0, 305)
        fig.savefig(f"{out_prefix}_Tdep_clean.png", dpi=200,
                    bbox_inches="tight")
        prepend_header(f"{out_prefix}_Tdep_clean.csv", header)
    plt.close("all")

    # stitch_thr=None: mini loops are NEVER step-stitched — the below-T_C
    # staircase is domain-avalanche/remanence signal; stitching rectified
    # -45deg 150 K from +2.18e-3 to -6.5e-3 (caught 2026-07-02)
    b_curves, b_dropped, loop_steps = build_b_loop_curves(
        field_dep, stitch_thr=(0.04e-3 if STITCH_B else None))
    lam = {}    # T_round -> lambda at max B (up leg), for anisotropy plot
    if b_curves:
        from reduce_str_batch import marker_kw, MARKER_SHAPES
        b_temps = sorted({c.param_value for c in b_curves})
        b_shapes = {t: MARKER_SHAPES[i % len(MARKER_SHAPES)]
                    for i, t in enumerate(b_temps)}
        fig, ax = plt.subplots(figsize=(8, 6))
        for c in sorted(b_curves, key=lambda cc: (cc.param_value,
                                                  cc.mode_index)):
            d = c.cleaned()
            ax.plot(d["B [T]"], d["(del_L/L_0)_Sam"] * 1e3,
                    color=c.color,
                    label=(f"T={c.param_value:.0f} K"
                           if c.direction == "up" else None),
                    **marker_kw(c.color, len(d),
                                shape=b_shapes[c.param_value],
                                open_face=(c.direction != "up"),
                                n_markers=120))
            if c.direction == "up" and len(d) > 10:
                bmax = d["B [T]"].max()
                lam[float(c.param_value)] = {
                    "B_T": float(bmax),
                    "lambda_1e3": float(
                        d.loc[d["B [T]"] > bmax - 0.3,
                              "(del_L/L_0)_Sam"].median() * 1e3)}
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(r"$B$ (T)")
        ax.set_ylabel(r"$\lambda = \Delta L(B)/L_0 \;(\times 10^{-3})$")
        ax.set_xlim(0, None)
        ax.legend(fontsize=9)
        ax.set_title(f"mini cell, $\\theta={angle:+d}^\\circ$ "
                     "(filled up / open down; loop-referenced)",
                     fontsize=11)
        S._style_axes(ax)
        fig.savefig(f"{out_prefix}_Bdep_clean.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)

    prov = {"angle_deg": angle, "source": src, "L0_cm": L0,
            "L0_source": L0_SOURCE,
            "rescale": rescale if rescale != 1.0 else None,
            "C0_pF": round(float(C0), 4),
            "glitch": {"jumps": n_jumps, "rows_excluded": int(gmask.sum())},
            "settling_dwell_rows_dropped": n_settle,
            "t_steps_stitched": cstats["stitched"],
            "alpha_185_215K_1e6": {k: (round(v, 2) if np.isfinite(v)
                                       else None) for k, v in alpha.items()},
            "b_loop_steps": loop_steps,
            "b_fragments_dropped": b_dropped,
            "lambda_at_Bmax_by_T": lam,
            "calibration": cal["meta"]}
    with open(f"{out_prefix}_provenance.json", "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2)

    print(f"{tag:8s} ({angle:+3d} deg): rows={len(full)}  glitch_excl="
          f"{int(gmask.sum())}  settle={n_settle}  T-curves="
          f"{sum(c.enabled for c in t_curves)}  B-legs={len(b_curves)}  "
          f"alpha(185-215)={alpha['cool']:+.1f}/{alpha['warm']:+.1f} "
          f"{'OK' if ok_a else 'SUSPECT'}")
    return {"angle": angle, "tag": tag, "temp_dep": temp_dep,
            "t_curves": t_curves, "lam": lam, "alpha": alpha}


def main():
    global SRC_DIR, STEM, DL_SPACING, OUT_DIR
    global STITCH_T, STITCH_B, DWELL, SMOOTH
    import argparse
    ap = argparse.ArgumentParser(description="mini-cell batch reduction")
    ap.add_argument("--data", default=SRC_DIR,
                    help=f"folder holding the angle archives (default: {SRC_DIR})")
    ap.add_argument("--runs", default=None,
                    help="JSON angle-run spec (stem + runs[angle_deg,tag,glob,"
                         "rescale]); default: <data>/angle_runs.json")
    ap.add_argument("--file", default=None,
                    help="optional angle-filename substring; when given, only "
                         "runs whose tag/glob matches it are reduced (default: "
                         "all). Note: combined/polar outputs need >=2 angles.")
    ap.add_argument("--dl-spacing", type=float, default=0.2,
                    help="thin dL/L0 output to ~1 point per this many K in each "
                         "angle's _Tdep_clean.{csv,png}; 0 = every point "
                         "(default 0.2)")
    ap.add_argument("--alpha-bin", type=float, default=0.2,
                    help="accepted for GUI parity; the mini batch writes no "
                         "alpha CSV, so this has no effect here")
    # per-toggle cleanup wiring (GUI "cleanup:" checkboxes). Mini defaults:
    # stitching OFF — the below-T_C staircase is domain-avalanche/remanence
    # SIGNAL; stitching rectified -45deg 150 K from +2.18e-3 to -6.5e-3.
    ap.add_argument("--stitch-t-curves", action=argparse.BooleanOptionalAction,
                    default=False, help="stitch level shifts in T-curves "
                    "(default OFF for mini — loop-dense runs)")
    ap.add_argument("--stitch-b-loops", action=argparse.BooleanOptionalAction,
                    default=False, help="stitch level shifts in B-loops "
                    "(default OFF for mini — staircase is remanence signal)")
    ap.add_argument("--dwell-removal", action=argparse.BooleanOptionalAction,
                    default=True, help="drop stationary-dwell rows (default on)")
    ap.add_argument("--smooth-outside-protect",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="light smoothing outside the transition protect "
                    "window (default on)")
    ap.add_argument("--rescale-file", default=None,
                    help="basename of ONE run file whose delta_l should get "
                         "--rescale-factor applied (GUI mis-conversion rescale "
                         "offer); the run is matched by its tag/glob like "
                         "--file. Ignored — with a warning — when the "
                         "angle_runs.json spec already sets a rescale for "
                         "that run (the spec wins, no double-apply).")
    ap.add_argument("--rescale-factor", type=float, default=None,
                    help="delta_l factor for the --rescale-file run "
                         "(e.g. 0.5079 for a 7 mm plate-radius mis-conversion)")
    ap.add_argument("--out", default=None,
                    help=f"output folder (default: {OUT_DIR})")
    args, _ = ap.parse_known_args()
    if args.out:
        OUT_DIR = args.out
    DL_SPACING = float(args.dl_spacing)
    STITCH_T, STITCH_B = args.stitch_t_curves, args.stitch_b_loops
    DWELL, SMOOTH = args.dwell_removal, args.smooth_outside_protect
    if STITCH_B:
        print("WARNING: --stitch-b-loops is ON for a MINI run. On loop-dense "
              "runs the below-T_C staircase is remanence/domain-avalanche "
              "SIGNAL; stitching rectifies it into fake drift (verified on "
              "the -45deg 150 K loop). Only use on data you have inspected.")
    if not (STITCH_T is False and STITCH_B is False and DWELL and SMOOTH):
        print(f"cleanup toggles: stitch_t={STITCH_T} stitch_b={STITCH_B} "
              f"dwell={DWELL} smooth={SMOOTH}")
    SRC_DIR = args.data
    STEM, runs = load_runs_spec(args.runs, SRC_DIR)
    if args.file:
        key = os.path.basename(args.file).lower()
        runs = [r for r in runs
                if r[1].lower() in key or r[2].split("*")[0].lower() in key]
        if not runs:
            print(f"--file={args.file!r} matched no known angle run; nothing to do.")
            return
    if args.rescale_file and args.rescale_factor:
        # GUI rescale offer: apply the factor to exactly the picked run,
        # matched the same way --file matches (tag / glob prefix).
        key = os.path.basename(args.rescale_file).lower()
        hit = False
        for i, (a, t, p, rs) in enumerate(runs):
            if t.lower() in key or p.split("*")[0].lower() in key:
                if rs != 1.0:
                    print(f"WARNING: --rescale-file matches run '{t}' but the "
                          f"spec already sets rescale={rs} for it — the spec "
                          "wins; --rescale-factor NOT applied (no double-"
                          "apply).")
                else:
                    runs[i] = (a, t, p, float(args.rescale_factor))
                    print(f"rescale override: run '{t}' delta_l x "
                          f"{args.rescale_factor:.4f} (user-confirmed "
                          "mis-conversion rescale)")
                hit = True
                break
        if not hit:
            print(f"WARNING: --rescale-file={args.rescale_file!r} matched no "
                  "angle run — rescale NOT applied.")
    print(f"input dir: {SRC_DIR}   angles: {[r[1] for r in runs]}")

    os.makedirs(OUT_DIR, exist_ok=True)
    results = [reduce_one(a, t, p, rs) for a, t, p, rs in runs]

    if len(results) < 2:
        print(f"  {len(results)} angle(s) reduced; combined/polar outputs "
              "skipped (need >=2 angles).")
        return

    # ── combined T-dep, referenced at 200 K (window shared by all) ──────────
    fig, ax = plt.subplots(figsize=(8, 6))
    cool_trends = {}
    from reduce_str_batch import marker_kw, MARKER_SHAPES
    acolors = {r["tag"]: S.COLORS[i % len(S.COLORS)]
               for i, r in enumerate(results)}
    ashapes = {r["tag"]: MARKER_SHAPES[i % len(MARKER_SHAPES)]
               for i, r in enumerate(results)}
    for r in results:
        # longest enabled B~0 curve per direction = the main sweep of the
        # run, free of the field-loop sawtooth of the raw row cloud
        for d, ls in (("cool", "-"), ("warm", "--")):
            cands = [c for c in r["t_curves"]
                     if c.enabled and c.direction == d and c.param_value < 0.1]
            if not cands:
                continue
            c = max(cands, key=lambda cc: (cc.raw_df["T PPMS [K]"].max()
                                           - cc.raw_df["T PPMS [K]"].min()))
            dd = c.cleaned().sort_values("Rel Time")
            T = dd["T PPMS [K]"].values
            y = dd["(del_L/L_0)_Sam"].values.astype(float).copy()
            # overview trend: wide-median despike removes broad transients
            # (rot-0 ~240 K excursion) and post-loop remanence sawteeth,
            # which are real recorded dynamics — raw detail stays in the
            # per-angle _Tdep_clean figures
            wide = int(min(601, max(51, len(y) // 4)) // 2 * 2 + 1)
            med = pd.Series(y).rolling(wide, center=True,
                                       min_periods=20).median().values
            y[np.abs(y - med) > 0.4e-3] = np.nan
            y = pd.Series(y).rolling(31, center=True,
                                     min_periods=5).median().values.copy()
            near = np.abs(T - 200) < 3
            if np.isfinite(y[near]).sum() < 5:
                continue
            y_ref = (y - np.nanmedian(y[near])) * 1e3
            if d == "cool":
                cool_trends[r["tag"]] = (T, y_ref)
            k = thin_in_T(T, y_ref)
            ax.plot(T[k], y_ref[k], color=acolors[r["tag"]],
                    label=(f"$\\theta={r['angle']:+d}^\\circ$"
                           if d == "cool" else None),
                    **marker_kw(acolors[r["tag"]], len(k),
                                shape=ashapes[r["tag"]],
                                open_face=(d == "warm")))
    if TRANSITION.active:
        ax.axvline(TRANSITION.value, color="k", lw=0.8, ls=":")
        yl = ax.get_ylim()
        ax.text(TRANSITION.value + 4, yl[0] + 0.03 * (yl[1] - yl[0]),
                TRANSITION.axis_text(), fontsize=11)
    ax.set_xlim(0, 305)
    ax.set_xlabel(r"$T$ (K)")
    ax.set_ylabel(r"$\Delta L/L_0$ (ref.\ 200 K) $(\times 10^{-3})$")
    ax.legend(fontsize=10)
    ax.set_title("mini rotation series, B=0, rolling-median trend "
                 "(cool filled / warm open)",
                 fontsize=11)
    S._style_axes(ax)
    fig.savefig(os.path.join(OUT_DIR, f"{STEM}_Tdep_all_angles.png"),
                dpi=200, bbox_inches="tight")
    print(f"  Saved: {STEM}_Tdep_all_angles.png")

    # ── lambda(Bmax) vs angle ───────────────────────────────────────────────
    temps = sorted({t for r in results for t in r["lam"]})
    fig, ax = plt.subplots(figsize=(8, 6))
    tcolors = {t: S.COLORS[i % len(S.COLORS)] for i, t in enumerate(temps)}
    for t in temps:
        pts = [(r["angle"], r["lam"][t]["lambda_1e3"], r["lam"][t]["B_T"])
               for r in results if t in r["lam"]]
        if len(pts) < 2:
            continue
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                color=tcolors[t], ms=6, markeredgecolor="k",
                markeredgewidth=0.6,
                label=f"T={t:.0f} K (B={pts[0][2]:.0f} T)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$\theta$ (deg, w.r.t. $B$)")
    ax.set_ylabel(r"$\lambda(B_{\max}) \;(\times 10^{-3})$")
    ax.set_xticks([-90, -45, 0, 45, 90])
    ax.legend(fontsize=9)
    ax.set_title("magnetostriction anisotropy (mini rotation series)",
                 fontsize=11)
    S._style_axes(ax)
    fig.savefig(os.path.join(OUT_DIR, f"{STEM}_lambda_vs_angle.png"),
                dpi=200, bbox_inches="tight")
    print(f"  Saved: {STEM}_lambda_vs_angle.png")

    # ── selected best runs, transition region (raw data, documented pick) ───
    # criterion per angle/direction: highest (T-span within 130-220 K) x
    # (continuity = fraction of consecutive points < 60 s apart) — picks the
    # least loop-interrupted sweep; raw points, no smoothing
    fig, ax = plt.subplots(figsize=(8, 6))
    picked = {}
    for r in results:
        for d, ls in (("cool", "-"), ("warm", "--")):
            best, best_score = None, 0.0
            for c in r["t_curves"]:
                if not (c.enabled and c.direction == d
                        and c.param_value < 0.1):
                    continue
                dd = c.cleaned().sort_values("Rel Time")
                w = dd[(dd["T PPMS [K]"] > 130) & (dd["T PPMS [K]"] < 220)]
                if len(w) < 50:
                    continue
                span = w["T PPMS [K]"].max() - w["T PPMS [K]"].min()
                cont = float(np.mean(np.diff(w["Rel Time"].values) < 60))
                if span * cont > best_score:
                    best, best_score = c, span * cont
            if best is None:
                continue
            picked[f"{r['tag']}/{d}"] = best.label
            dd = best.cleaned().sort_values("Rel Time")
            w = dd[(dd["T PPMS [K]"] > 130) & (dd["T PPMS [K]"] < 220)]
            T = w["T PPMS [K]"].values
            y = w["(del_L/L_0)_Sam"].values
            near = np.abs(T - 200) < 3
            if near.sum() < 5:
                continue
            y_ref = (y - np.median(y[near])) * 1e3
            k = thin_in_T(T, y_ref, dT=0.6)   # 130-220 K axis
            ax.plot(T[k], y_ref[k],
                    color=acolors[r["tag"]],
                    label=(f"$\\theta={r['angle']:+d}^\\circ$"
                           if d == "cool" else None),
                    **marker_kw(acolors[r["tag"]], len(k),
                                shape=ashapes[r["tag"]],
                                open_face=(d == "warm")))
    if TRANSITION.active:
        ax.axvline(TRANSITION.value, color="k", lw=0.8, ls=":")
        yl = ax.get_ylim()
        ax.text(TRANSITION.value + 4, yl[0] + 0.03 * (yl[1] - yl[0]),
                TRANSITION.axis_text(), fontsize=11)
    ax.set_xlim(130, 220)
    ax.set_xlabel(r"$T$ (K)")
    ax.set_ylabel(r"$\Delta L/L_0$ (ref. 200 K) $(\times 10^{-3})$")
    ax.legend(fontsize=10)
    ax.set_title("selected best runs (raw), transition region "
                 "(cool filled / warm open)", fontsize=11)
    S._style_axes(ax)
    fig.savefig(os.path.join(OUT_DIR, f"{STEM}_Tdep_selected.png"),
                dpi=200, bbox_inches="tight")
    with open(os.path.join(OUT_DIR, f"{STEM}_selected_runs.json"), "w", encoding="utf-8") as fh:
        json.dump({"criterion": "max T-span(130-220 K) x continuity"
                   " (fraction of dt<60 s); raw data, no smoothing",
                   "picked": picked}, fh, indent=2)
    print(f"  Saved: {STEM}_Tdep_selected.png (+ selected_runs.json)")

    # ── polar figures (2-fold mirror, min-shifted radius as in polar_figures) ───
    def polar_axes():
        fg = plt.figure(figsize=(8, 8))
        axp = fg.add_subplot(111, projection="polar")
        axp.set_theta_zero_location("N")
        axp.set_theta_direction(-1)
        axp.set_xticks(np.deg2rad(range(0, 360, 45)))
        axp.set_xticklabels(["0°", "45°", "90°", "135°", "180°",
                             "225°", "270°", "315°"])
        return fg, axp

    def polar_draw(axp, series):
        """series = [(label, color, [(angle_deg, value_1e3)])]. One COMMON
        radial offset for the whole figure, true values on the radial axis,
        dashed circle at value = 0 — nothing artificially starts at the
        origin (a per-curve min-shift reads as 'value=0 at that angle')."""
        allv = np.concatenate([[p[1] for p in pts]
                               for _, _, pts in series])
        rng = max(allv.max() - allv.min(), 1e-6)
        r0 = allv.min() - 0.15 * rng
        for label, color, pts in series:
            a = np.array([p[0] for p in pts], float)
            v = np.array([p[1] for p in pts], float)
            # Canonical [0,360) fold + 2-fold mirror; colliding nodes
            # (measured -90 vs mirror of +90, both at 270) are AVERAGED and
            # the spread printed — symmetry checked, not first-wins-dropped.
            # Kept in sync with polar_figures.fold_2fold.
            a_can = a % 360.0
            a_full = np.concatenate([a_can, (a_can + 180.0) % 360.0])
            v_full = np.concatenate([v, v])
            a_s = np.unique(a_full)
            v_s = []
            for au in a_s:
                vv = v_full[np.isclose(a_full, au)]
                spread = float(vv.max() - vv.min())
                if spread > 1e-9:
                    print(f"  2-fold symmetry check ({label}): {au:.0f} deg "
                          f"nodes differ by {spread:.3g} — averaged")
                v_s.append(float(vv.mean()))
            v_s = np.asarray(v_s)
            a_c = np.append(a_s, a_s[0] + 360)
            v_c = np.append(v_s, v_s[0])
            axp.plot(np.deg2rad(a_c), v_c - r0, "-", color=color, lw=2,
                     label=label)
            axp.scatter(np.deg2rad(a_can), v - r0, color=color, s=40,
                        zorder=5)
        # radial ticks show TRUE values; dashed circle marks zero
        ticks = [t for t in np.linspace(allv.min(), allv.max(), 5)]
        axp.set_rgrids([t - r0 for t in ticks],
                       labels=[f"{t:+.2f}" for t in ticks], angle=22.5,
                       fontsize=9)
        if allv.min() < 0 < allv.max():
            th = np.linspace(0, 2 * np.pi, 181)
            axp.plot(th, np.full_like(th, -r0), "k--", lw=0.9)
            axp.text(np.deg2rad(200), -r0, "0", fontsize=9, va="bottom")
        axp.set_rlim(0, allv.max() - r0 + 0.1 * rng)

    # thermal expansion polar: dL/L0(T_q, ref 200 K) from best cool per angle
    T_QUERY = [60, 100, 150, 170, 190]
    fig, axp = polar_axes()
    series, polar_rows = [], []
    for i, tq in enumerate(T_QUERY):
        pts = []
        for r in results:
            # query the overlay's smoothed cool trend (same-curve 200 K
            # referencing, remanence teeth median-filtered) so overlay and
            # polar are consistent by construction
            if r["tag"] not in cool_trends:
                continue
            Tt, yt = cool_trends[r["tag"]]
            m = (np.abs(Tt - tq) < 2) & np.isfinite(yt)
            if m.sum() > 5:
                pts.append((r["angle"], float(np.median(yt[m]))))
                polar_rows.append({"T_q_K": tq, "angle_deg": r["angle"],
                                   "dL_L0_ref200K_1e3":
                                   round(float(np.median(yt[m])), 3)})
        if len(pts) >= 3:
            series.append((f"$T={tq}$ K", S.COLORS[i % len(S.COLORS)], pts))
    if not series:
        print("  polar Tdep figure skipped (needs >=3 angles sharing a "
              "query T)")
    else:
        polar_draw(axp, series)
    axp.set_title(r"$\Delta L/L_0$ (ref. 200 K) $\times10^{-3}$ vs angle"
                  "\n(2-fold mirror; radial axis = true values, dashed = 0)",
                  pad=25, fontsize=12)
    axp.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    if series:
        fig.savefig(os.path.join(OUT_DIR, f"{STEM}_polar_Tdep.png"),
                    dpi=200, bbox_inches="tight")
    pd.DataFrame(polar_rows).to_csv(
        os.path.join(OUT_DIR, f"{STEM}_polar_Tdep.csv"), index=False,
        encoding="utf-8")
    print("  Saved: " + STEM + "_polar_Tdep.csv"
          + (" + .png" if series else " (figure skipped)"))

    # magnetostriction polar: lambda(Bmax) vs angle per T
    fig, axp = polar_axes()
    series, ci = [], 0
    for t in temps:
        pts = [(r["angle"], r["lam"][t]["lambda_1e3"])
               for r in results if t in r["lam"]]
        if len(pts) < 3:
            continue
        series.append((f"$T={t:.0f}$ K", S.COLORS[ci % len(S.COLORS)], pts))
        ci += 1
    if not series:
        print("  polar lambda figure skipped (needs >=3 angles sharing a "
              "loop T)")
    else:
        polar_draw(axp, series)
    axp.set_title(r"$\lambda(B_{\max})\times10^{-3}$ vs angle"
                  "\n(2-fold mirror; radial axis = true values, dashed = 0)",
                  pad=25, fontsize=12)
    axp.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    if series:
        fig.savefig(os.path.join(OUT_DIR, f"{STEM}_polar_lambda.png"),
                    dpi=200, bbox_inches="tight")
        print(f"  Saved: {STEM}_polar_lambda.png")


if __name__ == "__main__":
    main()
