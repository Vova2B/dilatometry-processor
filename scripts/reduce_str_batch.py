"""
reduce_str_batch.py — headless (batch) reduction of a str-cell run.

Reduces a uniaxial-stress-cell (or standard-cell) run from the RAW PPMS
columns of a .dat export or a comma *_all.csv archive (archives' own
computed columns are ignored — recompute from raw).
Uses qc_str_cell as a library — same physics path as the
interactive script: separate_data -> assign_branch -> load_calibration
(branch-aware registry) -> compute_del_l_l0.

Referencing choices:
  * T-sweeps: single global reference at T_min (compute_del_l_l0 default).
    Segments recorded after field sweeps carry remanent-magnetostriction /
    cell-drift offsets below T_C — plotted per cycle, offsets tabulated in
    the provenance JSON, NOT hidden by re-referencing.
  * B-sweeps: each 0->13->0 T loop is re-referenced to its own B~0 starting
    value, so up/down legs share a zero and hysteresis is real.

Gates (enforced, non-zero exit on failure):
  1. glitch scan   — no delta_l re-zero jumps may enter the plots
  2. paramagnetic alpha (200-290 K) flat/positive, cool AND warm — VIRGIN
     cycle only (the only segments free of magnetic history)
  3. virgin cool-warm state-function consistency <= 0.20e-3
  4. span regression vs a verified reference value (skipped unless
     VERIFIED is set for your sample — see the constants below)

Outputs (Output/): _Tdep_clean.{csv,png}, _Bdep_clean.{csv,png},
_alpha.{csv,png}, _provenance.json. CSVs carry '#' provenance headers —
read them with pandas.read_csv(..., comment='#').
"""

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
import qc_str_cell as S
import cells
import transition as _tr

# Shared primitives now live in cleanup.py / reduce.py (P1 extraction,
# 2026-07-03). Re-exported here so external importers (reduce_mini_batch and
# any ad-hoc callers) keep finding them at reduce_str_batch.*.
from cleanup import (glitch_mask, alpha_of, detect_steps, stitch_steps,
                     dwell_mask, smooth_outside, binned_alpha, clean_t_curves)
from reduce import (build_b_loop_curves, SWEEP_MIN_ROWS, SWEEP_MIN_SPAN_T)

SOURCE = os.path.join(PROC, "Data", "your_run.dat")   # set via --data/--file
OUT_DIR = os.path.join(PROC, "Output", "str")
STEM = "str_run"    # overridden by the input filename (see main())
# Sample parameters — EDIT for your sample (or pass --L0 / --transition):
L0 = 0.05           # cm — sample thickness along the measurement axis
L0_SOURCE = "module default / --L0 — record your own provenance"
# Sample transition. US default is a ferromagnetic T_C = 177 K, set on the US
# run via --transition/--transition-type; module default is "no transition".
TRANSITION = _tr.Transition(None, "none")
RAW_COLS = ["Abs Time", "Rel Time", "T PPMS [K]", "T sample [K]", "B [T]",
            "C [pF]", "L [nS]", "delta l [1E-6 cm]"]
# Optional regression reference from a previously verified reduction of the
# SAME run, e.g. {"span_1e3": 3.08, "alpha_cool": 10.5, "alpha_warm": 9.1}.
# None -> gate 4 is skipped and gate 2 prints without a reference.
VERIFIED = None

DWELL_MIN_SPAN_K = 3.0      # T-curves spanning less are dwell/settling tails
PROBE_T = (50, 100, 130, 160, 200, 250)
LEGEND_MAX = 14


def safe_relpath(path, start):
    """os.path.relpath, but degrade to the absolute path when the two are on
    different drives — Windows raises ValueError across e.g. C: and F: (data on
    a USB stick, app on C:)."""
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return os.path.abspath(path)


MARKER_SHAPES = ["o", "s", "^", "D", "v", "<", ">", "p"]
BRANCH_MARKER = {"cool": "s", "warm": "D"}   # convention: sq/diamond


def marker_kw(color, n_points, shape="o", open_face=False, n_markers=160):
    """dL/L plotting convention (journal reference: UO2 kappa figure):
    symbol-only curves — no connecting line, dense black-edged markers with
    color-filled faces; open (white-faced, color-edged) symbols mark the
    reverse branch of a pair (e.g. field-down leg)."""
    if open_face:
        face, edge = "white", color
    else:
        face, edge = color, "k"
    return dict(ls="none", marker=shape, ms=4,
                markerfacecolor=face, markeredgecolor=edge,
                markeredgewidth=0.6,
                markevery=max(1, n_points // n_markers))


def load_raw():
    # S.load_data sniffs the format (P3.5): tab-delimited raw `.dat`
    # (skiprows=1) OR comma-separated `_all.csv` archive. It already applies
    # the inf->NaN replace, the T/C/delta_l dropna and reset_index; here we add
    # only the str-specific C>0 filter and magnet-off B fill.
    df = S.load_data(SOURCE)
    df = df[df["C [pF]"] > 0]
    n_bnan = int(df["B [T]"].isna().sum())
    df["B [T]"] = df["B [T]"].fillna(0.0)   # magnet off at run start; NaN
    df = df.reset_index(drop=True)          # keys vanish from groupby
    return df, n_bnan


# glitch_mask, alpha_of, detect_steps, stitch_steps, dwell_mask,
# smooth_outside, binned_alpha, clean_t_curves are imported from cleanup.py;
# build_b_loop_curves (+ SWEEP_MIN_ROWS/SWEEP_MIN_SPAN_T) from reduce.py.
# All re-exported above for backward compatibility (P1 extraction).


def prepend_header(csv_path, lines):
    with open(csv_path, encoding="utf-8") as fh:
        body = fh.read()
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("".join(f"# {ln}\n" for ln in lines) + body)


def cap_legend(fig):
    """Cap legend entries so the old 45-entry off-canvas legend can't recur."""
    if fig is None:
        return
    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > LEGEND_MAX:
        ax.legend(handles[:LEGEND_MAX],
                  labels[:LEGEND_MAX - 1] +
                  [f"... +{len(labels) - LEGEND_MAX + 1} more (see CSV)"],
                  loc="best", fontsize=8, framealpha=0.9)


def main():
    global SOURCE, STEM, L0, L0_SOURCE, OUT_DIR, TRANSITION
    import argparse
    default_source = SOURCE
    ap = argparse.ArgumentParser(description="str-cell batch reduction")
    ap.add_argument("--data", default=os.path.dirname(SOURCE),
                    help="folder holding the raw str archive/.dat "
                         f"(default: {os.path.dirname(SOURCE)})")
    ap.add_argument("--file", default=os.path.basename(SOURCE),
                    help="filename of the raw str archive/.dat to reduce "
                         f"(default: {os.path.basename(SOURCE)})")
    ap.add_argument("--L0", type=float, default=None,
                    help=f"sample thickness in cm (default: {L0})")
    ap.add_argument("--transition", type=float, default=None,
                    help="transition temperature in K used to split the "
                         "ferro/para panels (default: none)")
    ap.add_argument("--transition-type", choices=list(_tr.KINDS),
                    default=None,
                    help="transition kind: T_C / T_N / T_CDW / none "
                         "(default: T_C when --transition is given, else none)")
    ap.add_argument("--find-transition", action="store_true",
                    help="scan alpha(T) for transition candidates, print them, "
                         "and exit without reducing")
    ap.add_argument("--alpha-bin", type=float, default=0.2,
                    help="alpha(T) bin width in K — smaller = more but noisier "
                         "points in _alpha.{csv,png} (default 0.2; "
                         "auto-coarsened when the data cannot fill such bins)")
    ap.add_argument("--dl-spacing", type=float, default=0.2,
                    help="thin dL/L0 output to ~1 point per this many K in "
                         "_Tdep_clean.{csv,png}; 0 = every point (default 0.2)")
    ap.add_argument("--out", default=None,
                    help=f"output folder (default: {OUT_DIR})")
    # per-toggle cleanup wiring (GUI "cleanup:" checkboxes; str defaults all on)
    ap.add_argument("--stitch-t-curves", action=argparse.BooleanOptionalAction,
                    default=True, help="stitch instrumental level shifts in "
                    "T-curves and before alpha (default on)")
    ap.add_argument("--stitch-b-loops", action=argparse.BooleanOptionalAction,
                    default=True, help="stitch instrumental level shifts in "
                    "B-loops (default on for the str cell)")
    ap.add_argument("--dwell-removal", action=argparse.BooleanOptionalAction,
                    default=True, help="drop stationary-dwell rows from "
                    "T-curves (default on)")
    ap.add_argument("--smooth-outside-protect",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="light smoothing outside the transition protect "
                    "window (default on)")
    args, _ = ap.parse_known_args()
    if args.out:
        OUT_DIR = args.out
    alpha_bin = float(args.alpha_bin)
    dl_spacing = float(args.dl_spacing)
    if args.L0 is not None:
        L0 = float(args.L0)
        L0_SOURCE = "--L0 command-line argument"
    kind = args.transition_type
    if kind is None:
        kind = "T_C" if args.transition is not None else "none"
    val = args.transition if kind != "none" else None
    TRANSITION = _tr.Transition(val, kind)
    SOURCE = os.path.join(args.data, args.file)
    # Non-default input -> derive the output stem from its filename so runs
    # are self-labeled. The GUI discovers all runs by globbing
    # Output/str/*_provenance.json.
    if os.path.abspath(SOURCE) != os.path.abspath(default_source):
        from reduce import output_stem
        STEM = output_stem(SOURCE)
    if not os.path.isfile(SOURCE):
        raise SystemExit(
            f"input file not found: {SOURCE}\n"
            "Point the reducer at your raw run with:\n"
            "  python reduce_str_batch.py --data <folder> --file <name>.dat\n"
            "  (macOS/Linux: python3; Windows: python or py)\n"
            "(accepts a raw PPMS .dat or a comma *_all.csv archive)")
    print(f"input: {SOURCE}")
    print(f"output stem: {STEM}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_prefix = os.path.join(OUT_DIR, STEM)
    gates = {}

    df, n_bnan = load_raw()
    print(f"loaded {len(df)} rows from {os.path.basename(SOURCE)}"
          f" (B NaN filled with 0: {n_bnan})")

    field_dep, temp_dep = S.separate_data(
        df, field_change_threshold=cells.STR_DIL["field_change_threshold"])
    full = pd.concat([temp_dep, field_dep]).sort_index()
    full["Branch"] = S.assign_branch(full, field_dep.index)

    cal = S.load_calibration("str_dil", L0,
                             t_max_needed=float(full["T PPMS [K]"].max()))
    for br, m in cal["meta"]["per_branch"].items():
        print(f"calibration {br}: {m['mode']} {m['records']}"
              + (f"  ({m['note']})" if m.get("note") else ""))

    full = S.compute_del_l_l0(full, L0, cal)

    # ── gate 1: glitch scan ─────────────────────────────────────────────────
    gmask, n_jumps = glitch_mask(full)
    gates["glitch"] = {"jumps": n_jumps, "rows_excluded": int(gmask.sum()),
                       "pass": True}
    if gmask.any():
        full = full[~gmask]
        print(f"GATE glitch: {n_jumps} re-zero jumps, "
              f"{int(gmask.sum())} rows excluded from all outputs")
    else:
        print("GATE glitch: no re-zero jumps — clean run")

    temp_dep = full.loc[full.index.intersection(temp_dep.index)]
    field_dep = (full.loc[full.index.intersection(field_dep.index)]
                 if not field_dep.empty else field_dep)

    if args.find_transition:
        cands = _tr.find_transition(temp_dep["T PPMS [K]"].values,
                                    temp_dep["(del_L/L_0)_Sam"].values)
        if cands:
            print("transition candidates (strongest first):")
            for c in cands:
                print(f"  T = {c['T']:.1f} K  (prominence {c['prominence']}, "
                      f"alpha jump {c['alpha_jump']} x1e-6/K)")
        else:
            print("no clear transition found (flat alpha) -> use --transition-type none")
        return 0

    # ── virgin cycle: everything before the first field sweep ───────────────
    t_first_field = (field_dep["Rel Time"].min()
                     if not field_dep.empty else np.inf)
    virgin = temp_dep[temp_dep["Rel Time"] < t_first_field]
    # drop the settling dwell at run start (pre-cool rows are Mode-labeled
    # "warm" and would fake a step at the head of the warm series)
    t_cool_start = virgin.loc[virgin["Branch"] == "cool", "Rel Time"].min()
    if not np.isfinite(t_cool_start):
        # warming-only protocol: no cooling rows, so there is no pre-cool
        # settling dwell to drop — keep the whole warm branch (previously
        # `Rel Time >= NaN` silently dropped ALL virgin rows)
        t_cool_start = virgin["Rel Time"].min()
        print("virgin cycle: no cooling branch (warming-only run) — "
              "settling-dwell drop skipped")
    virgin = virgin[virgin["Rel Time"] >= t_cool_start]
    print(f"virgin (pre-field) cycle: {len(virgin)} rows, "
          f"T {virgin['T PPMS [K]'].min():.1f}-"
          f"{virgin['T PPMS [K]'].max():.1f} K")

    T = virgin["T PPMS [K]"].values
    y = virgin["(del_L/L_0)_Sam"].values
    br = virgin["Branch"].values

    # resolution "if the data allow": coarsen the alpha bin when the sparser
    # branch cannot fill bins of the requested width (both branches keep the
    # SAME bin so cool/warm alpha stay comparable)
    from cleanup import auto_alpha_bin
    for d in ("cool", "warm"):
        m = br == d
        if m.sum() < 30:
            continue
        eff, note_ab = auto_alpha_bin(T[m], alpha_bin)
        if eff > alpha_bin:
            alpha_bin = eff
            print(f"{note_ab} [{d} branch]")

    # ── gate 2: paramagnetic alpha, virgin cycle ────────────────────────────
    alpha = {d: alpha_of(T[br == d], y[br == d], 200, 290) * 1e6
             for d in ("cool", "warm")}
    applicable = TRANSITION.active and all(np.isfinite(a)
                                           for a in alpha.values())
    if applicable:
        ok_a = alpha["cool"] > -5 and alpha["warm"] > -5
    else:
        ok_a = True                      # N/A never counts as a failure
    gates["alpha_paramagnetic_virgin"] = {
        "cool_1e6_per_K": round(alpha["cool"], 2),
        "warm_1e6_per_K": round(alpha["warm"], 2),
        "applicable": bool(applicable),
        "verified_ref": ((VERIFIED["alpha_cool"], VERIFIED["alpha_warm"])
                         if VERIFIED else None),
        "pass": bool(ok_a)}
    ref = (f"(verified +{VERIFIED['alpha_cool']}/+{VERIFIED['alpha_warm']}) "
           if VERIFIED else "")
    status = ("PASS" if ok_a else "FAIL") if applicable else "N/A"
    print(f"GATE paramagnetic alpha (200-290 K, virgin): "
          f"cool {alpha['cool']:+.1f}, warm {alpha['warm']:+.1f} x1e-6/K "
          f"{ref}-> {status}")

    # ── gate 3: virgin cool-warm state-function consistency ─────────────────
    gaps = {}
    for q in PROBE_T:
        v = {}
        for d in ("cool", "warm"):
            m = (br == d) & (np.abs(T - q) < 3)
            v[d] = np.median(y[m]) if m.sum() > 5 else np.nan
        gaps[q] = (v["cool"] - v["warm"]) * 1e3
    # the gate needs BOTH branches at some probe T; a run that legitimately
    # lacks one (warming-only / cooling-only) is N/A, not a data failure
    applicable_g = any(np.isfinite(g) for g in gaps.values())
    max_gap = (np.nanmax(np.abs(list(gaps.values()))) if applicable_g
               else float("nan"))
    ok_g = (max_gap <= 0.20) if applicable_g else True
    gates["virgin_cool_warm_consistency"] = {
        "gaps_1e3": {k: round(g, 3) for k, g in gaps.items()},
        "max_abs_1e3": round(float(max_gap), 3),
        "applicable": bool(applicable_g), "pass": bool(ok_g)}
    status_g = ("PASS" if ok_g else "FAIL") if applicable_g else \
        "N/A (needs both cool and warm branches)"
    print(f"GATE virgin cool-warm consistency: max |gap| {max_gap:.3f} x1e-3 "
          f"-> {status_g}")

    # ── gate 4: span regression ─────────────────────────────────────────────
    # Span = virgin WARM branch at 300 K: one continuous 2->300 K sweep
    # anchored at the T_min reference. The virgin cool value (branch residual
    # ~0.1e-3 higher) goes to provenance.
    span = {}
    for d in ("cool", "warm"):
        sel = y[(br == d) & (T > 297)]
        span[d] = np.nanmedian(sel) * 1e3 if sel.size else float("nan")
    applicable_s = bool(VERIFIED) and np.isfinite(span["warm"])
    ok_s = (abs(span["warm"] - VERIFIED["span_1e3"]) < 0.10
            if applicable_s else True)
    gates["span_regression"] = {"span_warm_1e3": round(float(span["warm"]), 3),
                                "span_cool_1e3": round(float(span["cool"]), 3),
                                "verified_1e3": (VERIFIED["span_1e3"]
                                                 if VERIFIED else None),
                                "applicable": bool(applicable_s),
                                "pass": bool(ok_s)}
    if VERIFIED and not applicable_s:
        print(f"GATE span(virgin warm): no warm branch at 300 K "
              f"(cool {span['cool']:+.2f} x1e-3) -> N/A")
    elif VERIFIED:
        print(f"GATE span(virgin warm): {span['warm']:+.2f} x1e-3 "
              f"(cool {span['cool']:+.2f}; verified +{VERIFIED['span_1e3']}) "
              f"-> {'PASS' if ok_s else 'FAIL'}")
    else:
        print(f"GATE span(virgin warm): {span['warm']:+.2f} x1e-3 "
              f"(cool {span['cool']:+.2f}; no VERIFIED reference -> "
              "recorded, not gated)")

    # ── magnetic-history / drift offsets of post-field segments ─────────────
    # (real physics below T_C — remanent magnetostriction after 13 T loops —
    #  plus multi-day cell-zero drift; tabulated, not gated)
    history = {}
    vc = {}
    for q in PROBE_T:
        sel = y[(br == "cool") & (np.abs(T - q) < 3)]
        vc[q] = np.median(sel) if sel.size else np.nan
    post = temp_dep[temp_dep["Rel Time"] >= t_first_field]
    for mi, g in post.groupby("Mode_Index", sort=False):
        if len(g) < 200:
            continue
        Tg, yg = g["T PPMS [K]"].values, g["(del_L/L_0)_Sam"].values
        d = "cool" if mi.split("_")[0] == "1" else "warm"
        offs = {}
        for q in PROBE_T:
            m = np.abs(Tg - q) < 3
            if m.sum() > 5 and np.isfinite(vc.get(q, np.nan)):
                offs[q] = round((np.median(yg[m]) - vc[q]) * 1e3, 3)
        history[f"{mi} ({d})"] = offs
    print("post-field segment offsets vs virgin cool [1e-3] "
          "(remanent magnetostriction below T_C + cell drift):")
    for k, offs in history.items():
        print(f"  {k:>14}: " + "  ".join(f"{q}K:{v:+.2f}"
                                         for q, v in offs.items()))

    # ── T-curves: per-cycle colors, dwell filter ────────────────────────────
    t_curves, _ = S.build_curves(field_dep, temp_dep, angle_deg=0)
    dropped = []
    # settling filter: inside a cool (warm) sweep, drop rows that retrace
    # >0.7 K above the running minimum (below the running maximum) —
    # removes dwell/stabilisation oscillations (e.g. the 100-120 K
    # square-wave on the 0.5 T cool) without touching monotonic data
    t_curves, cstats = clean_t_curves(
        t_curves, S.Curve,
        jump_thr=(0.03e-3 if args.stitch_t_curves else None),
        dwell=args.dwell_removal, smooth=args.smooth_outside_protect)
    if not (args.stitch_t_curves and args.dwell_removal
            and args.smooth_outside_protect and args.stitch_b_loops):
        off = [n for n, v in (("stitch_t_curves", args.stitch_t_curves),
                              ("stitch_b_loops", args.stitch_b_loops),
                              ("dwell_removal", args.dwell_removal),
                              ("smooth_outside_protect",
                               args.smooth_outside_protect)) if not v]
        print(f"cleanup toggles OFF for this run: {', '.join(off)}")
    n_settle, n_dwell = cstats["settling_rows"], cstats["dwell_rows"]
    t_fixes = cstats["stitched"]
    print(f"  T-curve cleanup: {n_settle} settling + {n_dwell} dwell rows "
          f"dropped; {sum(len(v) for v in t_fixes.values())} steps "
          f"stitched; curves split at >600 s gaps")
    color_by_sweep = {}
    for c in t_curves:
        base = c.label.lstrip("_")
        if base not in color_by_sweep:
            color_by_sweep[base] = S.COLORS[len(color_by_sweep)
                                            % len(S.COLORS)]
        c.color = color_by_sweep[base]   # gap segments share the color
        span_k = c.raw_df["T PPMS [K]"].max() - c.raw_df["T PPMS [K]"].min()
        if span_k < DWELL_MIN_SPAN_K:
            c.enabled = False
            dropped.append((c.label, f"T-span {span_k:.1f} K < "
                            f"{DWELL_MIN_SPAN_K} K (dwell/settling)"))

    # ── B-curves: loops split into legs, loop-referenced, fragments out ─────
    fb = cal["meta"]["field_background"]
    b_curves, b_dropped, loop_steps = build_b_loop_curves(
        field_dep, fb=fb, L0_cm=L0,
        stitch_thr=(0.04e-3 if args.stitch_b_loops else None))
    if fb and fb.get("p2_scale_at_9T_1e6cm"):
        c13 = fb["p2_scale_at_9T_1e6cm"] * (13.0 / 9.0) ** 2
        print(f"field correction: smooth P2-scale quadratic subtracted "
              f"({c13:.2f}e-6 cm at 13 T = {c13 * 1e-6 / L0 * 1e3:.4f}e-3); "
              f"snap component enveloped, not corrected")
    dropped += b_dropped
    for label, why in dropped:
        print(f"  disabled: {label} — {why}")
    for loop, steps in loop_steps.items():
        for s in steps:
            print(f"  STEP in {loop}: {s['step_1e3']:+.3f} x1e-3 at "
                  f"B={s['B_T']} T — stitched out (recorded in provenance)")
    n_t = sum(c.enabled for c in t_curves)
    print(f"curves: {n_t}/{len(t_curves)} T-sweeps, {len(b_curves)} B-sweep "
          f"legs ({len(b_dropped)} fragments dropped)")

    header = [
        "reduce_str_batch.py (2026-07-02) — branch-aware registry reduction",
        f"source: {safe_relpath(SOURCE, PROC)} (raw PPMS columns only; "
        "the file's computed columns are stale and were ignored)",
        f"sample stem: {STEM}; str cell; "
        + (f"transition split at {TRANSITION.value} K"
           if TRANSITION.active else "no transition (single-panel figures)"),
        f"L0 = {L0} cm — {L0_SOURCE}",
        "calibration: calibrations.json, " + "; ".join(
            f"{br}: {m['mode']} {m['records']}"
            for br, m in cal["meta"]["per_branch"].items()),
        "formula: (delta_l - cell_cal_branch(T))*1e-6/L0 + cu_lit*1e-6, "
        "referenced to T_min (P18 Eq. 6)",
        "T-sweeps: global T_min reference; post-field segments carry "
        "remanent-magnetostriction/drift offsets below T_C (see "
        "_provenance.json history_offsets)",
        "cleanup: dwell rows removed; fast instrumental level shifts "
        "stitched (list in provenance); rolling-median smoothing applied "
        "OUTSIDE 150-190 K only",
        "B-sweeps: each 0->13->0 T loop re-referenced to its own B~0 start",
        "field background: smooth P2-scale quadratic SUBTRACTED from B-loops "
        "(0.57e-6 cm at 13 T); stick-slip snap component enveloped "
        "(0.1-2.1e-6 cm at T<=50 K), not correctable",
        "gates: " + ", ".join(
            f"{k}={'PASS' if v['pass'] else 'FAIL'}"
            for k, v in gates.items()),
        "read with: pandas.read_csv(path, comment='#')",
    ]

    # ── flagship figure: virgin cycle only ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for d, color in (("cool", "tab:blue"), ("warm", "tab:red")):
        m = br == d
        if not m.any():          # single-branch run: no empty legend entry
            continue
        o = np.argsort(T[m])
        ax.plot(T[m][o], y[m][o] * 1e3, color=color,
                label=f"{d} (virgin, $B=0$)",
                **marker_kw(color, int(m.sum()), shape=BRANCH_MARKER[d]))
    if TRANSITION.active:
        ax.axvline(TRANSITION.value, color="k", lw=0.8, ls=":")
        yl = ax.get_ylim()
        ax.text(TRANSITION.value + 4, yl[0] + 0.03 * (yl[1] - yl[0]),
                TRANSITION.axis_text(), fontsize=13)
    ax.set_xlabel(r"$T$ (K)")
    ax.set_ylabel(r"$\Delta L / L_0 \;(\times 10^{-3})$")
    ax.set_xlim(0, 305)         # dilatometry convention: T axis from 0 K
    ax.legend(loc="upper left")
    S._style_axes(ax)
    fig.savefig(f"{out_prefix}_Tdep_virgin.png", dpi=200,
                bbox_inches="tight")
    print(f"  Saved: {out_prefix}_Tdep_virgin.png")

    if t_curves:
        # ':.0f' in build_curves prints B=0.1..0.4 T curves as 'B=0T'
        for c in t_curves:
            if 0 < c.param_value < 1:
                c.label = c.label.replace("B=0T", f"B={c.param_value:g}T")
        fig = S.plot_temperature_dep(t_curves, 0, out_prefix,
                                     spacing_k=dl_spacing)
        if dl_spacing > 0:
            header = header + [f"dL/L0 thinned to ~1 point per {dl_spacing:g} K"]
        cap_legend(fig)
        fig.axes[0].set_xlim(0, 305)
        fig.savefig(f"{out_prefix}_Tdep_clean.png", dpi=200,
                    bbox_inches="tight")
        prepend_header(f"{out_prefix}_Tdep_clean.csv", header)

    # ── magnetostriction: two panels split at the transition (single panel
    # when there is none) ───────────────────────────────────────────────────
    # excluded stays [] for T-sweep-only runs (no field loops) so the
    # provenance below never references an undefined name.
    excluded = []
    # Kind-aware wording for the selected-loop provenance/CSV/print text.
    # For kind == "T_C" these reproduce the original wording byte-for-byte.
    _sym = TRANSITION.symbol                       # 'T_C'/'T_N'/'T_{CDW}' | None
    sub_step_note = (f"a sharp sub-{_sym} step" if TRANSITION.active
                     else "a sharp step")
    if TRANSITION.active:
        sel_short = f"T<{_sym}"
        _phase_note = ("higher-T-phase loops excluded"
                       if TRANSITION.kind == "T_CDW"
                       else "paramagnetic loops excluded")
        sel_criterion = (f"T < {_sym} ({_phase_note}); "
                         "instrumental steps stitched out, not a "
                         "selection criterion since 2026-07-02")
    else:
        sel_short = "all loops (no transition split)"
        sel_criterion = ("all loops (no transition split); instrumental "
                         "steps stitched out, not a selection criterion "
                         "since 2026-07-02")
    if b_curves:
        split = TRANSITION.split_active
        if split:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
            panels = {True: axes[0], False: axes[1]}
        else:
            fig, ax_single = plt.subplots(figsize=(8, 6))
            axes = [ax_single]
        temps = sorted({c.param_value for c in b_curves})
        color_map = {t: S.COLORS[i % len(S.COLORS)]
                     for i, t in enumerate(temps)}
        shape_map = {t: MARKER_SHAPES[i % len(MARKER_SHAPES)]
                     for i, t in enumerate(temps)}
        loop_count, csv_rows = {}, []
        for c in sorted(b_curves, key=lambda cc: (cc.param_value,
                                                  cc.mode_index)):
            ax = (panels[c.param_value < TRANSITION.value] if split
                  else axes[0])
            if c.direction == "up":
                loop_count[c.param_value] = \
                    loop_count.get(c.param_value, 0) + 1
            n = loop_count.get(c.param_value, 1)
            n_loops = sum(1 for cc in b_curves
                          if cc.param_value == c.param_value
                          and cc.direction == "up")
            lbl = (f"T={c.param_value:.0f} K"
                   + (f" ({chr(96 + n)})" if n_loops > 1 else "")
                   if c.direction == "up" else None)
            df_c = c.cleaned()
            ax.plot(df_c["B [T]"], df_c["(del_L/L_0)_Sam"] * 1e3,
                    color=color_map[c.param_value], label=lbl,
                    **marker_kw(color_map[c.param_value], len(df_c),
                                shape=shape_map[c.param_value],
                                open_face=(c.direction != "up"),
                                n_markers=120))
            for _, row in df_c.iterrows():
                csv_rows.append({
                    "B_T": row["B [T]"],
                    "lambda_dL_L0": row["(del_L/L_0)_Sam"],
                    "lambda_raw_dL_L0": row.get("lambda_raw", np.nan),
                    "T_K": c.param_value, "direction": c.direction,
                    "mode_index": c.mode_index})
        if split:
            lo_title, hi_title = TRANSITION.panel_titles()
            axes[0].set_title(lo_title, fontsize=12)
            if TRANSITION.kind == "T_C":
                axes[1].set_title(hi_title + "\nresidual staircase = "
                                  "instrumental snaps, not correctable "
                                  "(see provenance)", fontsize=10)
            else:
                axes[1].set_title(hi_title, fontsize=12)
        for ax in axes:
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel(r"$B$ (T)")
            ax.legend(fontsize=9, loc="best")
            S._style_axes(ax)
        axes[0].set_ylabel(r"$\lambda = \Delta L(B)/L_0 \;(\times 10^{-3})$")
        fig.text(0.5, 0.965,
                 "filled: field up   open: field down   "
                 "(each loop referenced to its $B\\approx0$ start)",
                 ha="center", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(f"{out_prefix}_Bdep_clean.png", dpi=200,
                    bbox_inches="tight")
        pd.DataFrame(csv_rows).to_csv(f"{out_prefix}_Bdep_clean.csv",
                                      index=False, encoding="utf-8")
        prepend_header(f"{out_prefix}_Bdep_clean.csv",
                       header + ["lambda_dL_L0 = lambda(B), loop-referenced "
                                 "(0 at each loop's B~0 start)",
                                 "lambda_raw_dL_L0 = same WITHOUT step "
                                 "stitching — " + sub_step_note + " could be "
                                 "a real domain avalanche; audit the "
                                 "stitched-step list in provenance"])
        print(f"  Saved: {out_prefix}_Bdep_clean.{{png,csv}}")

        # ── selected loops: below the transition, no step > 0.1e-3
        # (publication set); with no transition, all loops are kept ────────
        def loop_ok(c):
            # instrumental steps are stitched out now -> selection is purely
            # physical: below the transition, where the magnetostriction
            # signal lives
            return (c.param_value < TRANSITION.value
                    if TRANSITION.active else True)

        selected = sorted((c for c in b_curves if loop_ok(c)),
                          key=lambda cc: (cc.param_value, cc.mode_index))
        excluded = sorted({f"T={c.param_value:.0f}K" for c in b_curves
                           if not loop_ok(c)})
        fig, ax = plt.subplots(figsize=(8, 6))
        for c in selected:
            df_c = c.cleaned()
            ax.plot(df_c["B [T]"], df_c["(del_L/L_0)_Sam"] * 1e3,
                    color=color_map[c.param_value],
                    label=(f"T={c.param_value:.0f} K"
                           if c.direction == "up" else None),
                    **marker_kw(color_map[c.param_value], len(df_c),
                                shape=shape_map[c.param_value],
                                open_face=(c.direction != "up"),
                                n_markers=120))
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(r"$B$ (T)")
        ax.set_ylabel(r"$\lambda = \Delta L(B)/L_0 \;(\times 10^{-3})$")
        ax.set_xlim(0, 13.5)
        ax.legend(loc="upper left", fontsize=10)
        ax.text(0.98, 0.02, "filled: up   open: down",
                transform=ax.transAxes, ha="right", fontsize=9,
                color="0.35", style="italic")
        S._style_axes(ax)
        fig.savefig(f"{out_prefix}_Bdep_selected.png", dpi=200,
                    bbox_inches="tight")
        print(f"  Saved: {out_prefix}_Bdep_selected.png "
              f"(selection: {sel_short}, steps stitched; excluded {excluded})")

    # ── alpha(T) figure + CSV (virgin cycle, step-aware) ────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    rows, all_steps = [], {}
    rt = virgin["Rel Time"].values
    for d, color in (("cool", "tab:blue"), ("warm", "tab:red")):
        m = br == d
        if not m.any():          # single-branch run: no empty legend entry
            continue
        # Origin-style: outlier-robust smoothing before differentiating —
        # stitch persistent level shifts, then rolling median (11 pts)
        if args.stitch_t_curves:
            y_st, steps_info = stitch_steps(rt[m], y[m], T=T[m])
        else:
            y_st, steps_info = y[m], []
        o_ = np.argsort(rt[m])
        y_sm = np.empty_like(y_st)
        y_sm[o_] = pd.Series(y_st[o_]).rolling(
            11, center=True, min_periods=3).median().values
        aT, aV, _ = binned_alpha(rt[m], T[m], y_sm, bin_k=alpha_bin,
                                 min_span=min(2.0, alpha_bin * 2.0 / 3.0))
        all_steps[d] = steps_info
        ax.plot(aT, aV, "o-", ms=3, lw=1, color=color, label=f"{d} (virgin)")
        rows += [{"T_K": t, "alpha_1e-6_per_K": a, "direction": d}
                 for t, a in zip(aT, aV)]
    for d, infos in all_steps.items():
        for s in infos:
            print(f"  instrumental step ({d}, virgin): {s['step_1e3']:+.3f} "
                  f"x1e-3 stitched before alpha")
    ax.axhline(0, color="k", lw=0.6)
    if TRANSITION.active:
        ax.axvline(TRANSITION.value, color="k", lw=0.8, ls=":")
        ax.text(TRANSITION.value + 4, 0.9 * ax.get_ylim()[1],
                TRANSITION.axis_text(), fontsize=12)
    ax.set_xlabel(r"$T$ (K)")
    ax.set_ylabel(r"$\alpha$ ($10^{-6}\,\mathrm{K}^{-1}$)")
    ax.set_xlim(0, 305)
    ax.legend()
    S._style_axes(ax)
    fig.savefig(f"{out_prefix}_alpha.png", dpi=200, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(f"{out_prefix}_alpha.csv", index=False, encoding="utf-8")
    prepend_header(f"{out_prefix}_alpha.csv",
                   header + [f"alpha = MAD-clipped linear slope in {alpha_bin:g} K bins "
                             "after step-stitching + rolling-median(11) "
                             "smoothing, virgin (pre-field) cycle only"])
    print(f"  Saved: {out_prefix}_alpha.{{png,csv}}")


    # ── transition-window zoom: hysteresis + field influence around the
    # entered transition (T ± 50 K). Only meaningful when a transition is
    # set — with kind "none" there is nothing to zoom on, so the figure is
    # skipped (it used to be a hardcoded US-specific 100-200 K window).
    if not TRANSITION.active:
        print("  transition-window zoom skipped (no transition set — use "
              "--transition/--transition-type, or Find transition in the app)")
    else:
        zoom_lo = TRANSITION.value - 50.0
        zoom_hi = TRANSITION.value + 50.0
        zoom_tag = f"Tdep_zoom_{zoom_lo:.0f}_{zoom_hi:.0f}K"
        enabled_t = sorted((c for c in t_curves if c.enabled),
                           key=lambda cc: (cc.param_value,
                                           cc.raw_df["Rel Time"].iloc[0]))
        fields = sorted({c.param_value for c in enabled_t})
        fcolors = {b: S.COLORS[i % len(S.COLORS)]
                   for i, b in enumerate(fields)}
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        n_in_window = 0
        for c in enabled_t:
            d = c.raw_df
            w = d[(d["T PPMS [K]"] > zoom_lo - 5)
                  & (d["T PPMS [K]"] < zoom_hi + 5)]
            if len(w) < 30:
                continue
            n_in_window += 1
            ls = "-" if c.direction == "cool" else "--"
            col = fcolors[c.param_value]
            kw = marker_kw(col, len(w), shape=BRANCH_MARKER[c.direction])
            kw["markevery"] = list(S._thin_idx(w["T PPMS [K]"].values,
                                               dT=1.1))
            axes[0].plot(w["T PPMS [K]"], w["(del_L/L_0)_Sam"] * 1e3,
                         color=col, label=c.label, **kw)
            if args.stitch_t_curves:
                y_st, _ = stitch_steps(w["Rel Time"].values,
                                       w["(del_L/L_0)_Sam"].values,
                                       T=w["T PPMS [K]"].values)
            else:
                y_st = w["(del_L/L_0)_Sam"].values
            y_sm = pd.Series(y_st).rolling(11, center=True,
                                           min_periods=3).median().values
            aT, aV, _ = binned_alpha(w["Rel Time"].values,
                                     w["T PPMS [K]"].values, y_sm, bin_k=4.0)
            axes[1].plot(aT, aV, ls, marker="o", ms=3, lw=1.1, color=col,
                         label=c.label)
        for ax in axes:
            ax.axvline(TRANSITION.value, color="k", lw=0.8, ls=":")
            ax.set_xlim(zoom_lo, zoom_hi)
            ax.set_xlabel(r"$T$ (K)")
            S._style_axes(ax)
        axes[0].set_ylabel(r"$\Delta L / L_0 \;(\times 10^{-3})$")
        axes[1].set_ylabel(r"$\alpha$ ($10^{-6}\,\mathrm{K}^{-1}$)")
        axes[1].axhline(0, color="k", lw=0.5)
        axes[0].set_title("cool squares / warm diamonds; color = field",
                          fontsize=11)
        axes[1].set_title(r"$\alpha(T)$, 4 K bins (offset-immune)",
                          fontsize=11)
        axes[1].legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_{zoom_tag}.png", dpi=200,
                    bbox_inches="tight")
        print(f"  Saved: {out_prefix}_{zoom_tag}.png "
              f"({n_in_window} curves in window)")

    prov = {"source": SOURCE, "L0_cm": L0, "L0_source": L0_SOURCE,
            "calibration": cal["meta"], "gates": gates,
            "instrumental_steps_virgin": all_steps,
            "instrumental_steps_b_loops_stitched": loop_steps,
            "instrumental_steps_t_curves_stitched": t_fixes,
            "t_curve_cleanup": {"settling_rows": n_settle,
                                "dwell_rows": n_dwell,
                                "smoothing": "rolling median 21 outside "
                                "150-190 K only"},
            "field_correction": {"applied": bool(
                fb and fb.get("p2_scale_at_9T_1e6cm")),
                "form": "p2_scale_at_9T * (B/9)^2, subtracted per loop",
                "p2_scale_at_9T_1e6cm": (fb or {}).get(
                    "p2_scale_at_9T_1e6cm"),
                "snap_component": "enveloped only (registry envelope_vs_T)"},
            "history_offsets_1e3_vs_virgin_cool": history,
            "curves": {"settling_rows_dropped": n_settle,
                       "T_enabled": n_t, "T_total": len(t_curves),
                       "B_legs": len(b_curves),
                       "B_selected_criterion": sel_criterion,
                       "B_selected_excluded": excluded,
                       "disabled": [{"label": l, "reason": r}
                                    for l, r in dropped]}}
    with open(f"{out_prefix}_provenance.json", "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2)
    print(f"  Saved: {out_prefix}_provenance.json")

    failed = [k for k, v in gates.items() if not v["pass"]]
    if failed:
        print(f"\nGATES FAILED: {failed}")
        return 1
    print("\nall gates PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
