"""
cu_calibration_builder.py — build the cell-background calibration registry
from raw Cu reference measurements.

Pipeline (per 2026-07-02-cu-calibration-design.md):
    segment -> repair -> fit -> decompose -> QC

The Cu-run list comes from a cu_runs.json placed in the data folder (or
passed via --runs) — same convention as reduce_mini_batch.py's
angle_runs.json. A worked example (the runs behind the shipped
calibrations.json) is scripts/cu_runs.example.json:

  {"runs": [{"key": "str_1mm", "path": "Cu_1mm_str_dil.dat",
             "cell": "str_dil", "cu_length_mm": 1.0, "t_max": 300},
            {"key": "field_1mm", "path": "Cu_field_1mm.dat",
             "cell": "str_dil", "cu_length_mm": 1.0, "kind": "field"}],
   "transfer_pairs":   [["str_1mm/c1w", "str_2mm/c1w", "cross-length"]],
   "eq7_pairs":        {"str_dil": {"warm": ["str_1mm/c1w", "str_2mm/c1w"]}},
   "hysteresis_pairs": [["str_1mm/c1c", "str_1mm/c1w"]]}

Per run: key (your id — record ids become <key>/c<cycle><w|c>, also the keys
of calibration_config.json overrides), path (.dat file, relative to --data),
cell (free cell id, e.g. str_dil / mini_dil), cu_length_mm; optional t_max
(informational) and kind: "field" for a fixed-T field-sweep run feeding the
field-background stage. Optional top-level pair lists (record ids):
transfer_pairs (gate 3; omitted -> no transfer section — declare same-length
/ cross-length pairs to quantify transfer uncertainty), eq7_pairs (omitted ->
auto-derived: per cell/branch the cycle-1 records of the two shortest Cu
lengths), hysteresis_pairs (QC figure only; omitted -> every complete
file/cycle cool+warm pair).

Run:  python3 cu_calibration_builder.py --data DIR   # all runs, registry +
                                                     # QC to fig_calibration_QC/
      python3 cu_calibration_builder.py --data DIR --file KEY  # one run, QC only
"""

import os
import re
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

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QC_DIR = os.path.join(PROJECT, "fig_calibration_QC")
DATA_DIR = PROJECT       # default; overridden by --data in main()

DL_COL = "delta l [1E-6 cm]"
T_COL = "T PPMS [K]"


def _resolve(path):
    """Run paths from cu_runs.json resolve against the data directory."""
    return path if os.path.isabs(path) else os.path.join(DATA_DIR, path)


def load_runs_spec(runs_path, data_dir):
    """Resolve the Cu-run spec: --runs JSON > <data>/cu_runs.json.
    Returns ({key: meta}, {"transfer": ..., "eq7": ..., "hysteresis": ...})
    with pair lists None when the spec omits them (see module docstring)."""
    path = runs_path or os.path.join(data_dir, "cu_runs.json")
    if not os.path.isfile(path):
        if runs_path:
            raise SystemExit(f"--runs {runs_path!r}: file not found")
        raise SystemExit(
            f"no Cu-run spec: put a cu_runs.json in {data_dir!r} or pass "
            "--runs <spec.json>.\nFormat: module docstring of "
            "cu_calibration_builder.py; worked example: "
            "scripts/cu_runs.example.json (the runs behind the shipped "
            "registry).")
    with open(path, encoding="utf-8") as fh:
        spec = json.load(fh)
    runs = {}
    try:
        for r in spec["runs"]:
            key = str(r["key"])
            if key in runs:
                raise SystemExit(f"{path}: duplicate run key {key!r}")
            meta = {"path": str(r["path"]), "cell": str(r["cell"]),
                    "cu_length_mm": float(r["cu_length_mm"])}
            if "t_max" in r:
                meta["t_max"] = r["t_max"]     # informational (QC summary)
            if "kind" in r:
                meta["kind"] = str(r["kind"])
            if "t_targets" in r:               # field runs: fixed-T setpoints
                meta["t_targets"] = [float(t) for t in r["t_targets"]]
            runs[key] = meta
    except (KeyError, TypeError, ValueError) as e:
        raise SystemExit(
            f"{path}: each entry in 'runs' needs key, path, cell, "
            f"cu_length_mm (optional t_max, kind). Problem: {e}")
    if not runs:
        raise SystemExit(f"{path}: 'runs' list is empty")
    pairs = {"transfer": spec.get("transfer_pairs"),
             "eq7": spec.get("eq7_pairs"),
             "hysteresis": spec.get("hysteresis_pairs")}
    for p in pairs["transfer"] or []:
        if len(p) != 3 or p[2] not in ("same-length", "cross-length"):
            raise SystemExit(
                f"{path}: transfer_pairs entries are [reduced_id, "
                f"calibration_id, 'same-length'|'cross-length']; got {p!r}")
    print(f"cu runs: {path} ({len(runs)} runs: {', '.join(runs)})")
    return runs, pairs


# ═════════════════════════════════════════════════════════════════════════════
# Loading
# ═════════════════════════════════════════════════════════════════════════════

def parse_header(path):
    """First line of PPMS dilatometry .dat, e.g.
    'dilatometry PPMS. C_max = 100.000000 pF.'"""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        line = fh.readline()
    m = re.search(r"C_max\s*=\s*([\d.]+)\s*pF", line)
    return {"header": line.strip(),
            "cmax_ppms_pF": float(m.group(1)) if m else None}


def load_dat(path):
    header = parse_header(path)
    df = pd.read_csv(path, sep="\t", skiprows=1, encoding="utf-8", encoding_errors="replace")
    df = df[np.isfinite(df[DL_COL]) & np.isfinite(df[T_COL])].reset_index(drop=True)
    return df, header


# ═════════════════════════════════════════════════════════════════════════════
# Segmentation: cycles and branches
# ═════════════════════════════════════════════════════════════════════════════

def segment_branches(df, rate_window=25, rate_threshold=1e-3,
                     min_points=300, min_span_K=20.0, time_gap_s=600):
    """
    Split a Cu run into directional branches.

    Direction from the rolling mean of per-point dT: 'cool' (dT < -thr),
    'warm' (dT > +thr), 'dwell' otherwise. Contiguous same-direction
    stretches shorter than min_points or spanning < min_span_K are merged
    into 'dwell'. A new cycle starts at each cool branch (a full cycle is
    cool -> warm) or after a large time gap.

    Returns a list of dicts:
      {branch_id, cycle, direction, i0, i1, T_start, T_end, T_min, T_max,
       C_start, n_points}
    """
    T = df[T_COL].values
    dT = pd.Series(T).diff().rolling(rate_window, center=True).mean().values
    direction = np.where(dT < -rate_threshold, -1,
                np.where(dT > rate_threshold, 1, 0))

    # time gaps break stretches regardless of direction
    if "Rel Time" in df.columns:
        gaps = df["Rel Time"].diff().abs().values > time_gap_s
    else:
        gaps = np.zeros(len(df), bool)

    # contiguous stretches
    stretches = []
    start = 0
    for i in range(1, len(df)):
        if direction[i] != direction[start] or gaps[i]:
            stretches.append((start, i))
            start = i
    stretches.append((start, len(df)))

    # demote short / narrow stretches to dwell, then merge adjacent dwells
    labeled = []
    for i0, i1 in stretches:
        d = direction[i0]
        span = abs(T[i1 - 1] - T[i0])
        if d != 0 and (i1 - i0 < min_points or span < min_span_K):
            d = 0
        labeled.append([i0, i1, d])
    merged = []
    for seg in labeled:
        if merged and merged[-1][2] == seg[2]:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)

    branches, cycle, seen_cool = [], 0, False
    for bid, (i0, i1, d) in enumerate(merged):
        if d == 0:
            continue
        name = "cool" if d == -1 else "warm"
        if name == "cool":
            cycle += 1
            seen_cool = True
        elif not seen_cool:
            cycle = max(cycle, 1)   # file starting mid-cycle with a warming
        branches.append({
            "branch_id": bid, "cycle": cycle, "direction": name,
            "i0": int(i0), "i1": int(i1),
            "T_start": float(T[i0]), "T_end": float(T[i1 - 1]),
            "T_min": float(T[i0:i1].min()), "T_max": float(T[i0:i1].max()),
            "C_start": float(df["C [pF]"].values[i0]) if "C [pF]" in df else None,
            "n_points": int(i1 - i0),
        })
    return branches


# ═════════════════════════════════════════════════════════════════════════════
# Artifact repair: offset-step stitching + exclusion windows
# ═════════════════════════════════════════════════════════════════════════════

def load_config():
    """Per-file/branch manual overrides. Keys: '<filekey>/c<cycle><w|c>' ->
    {manual_jump_rows: [...], exclude_T_windows: [[T1,T2],...], use: bool}"""
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "calibration_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def branch_key(key, b):
    return f"{key}/c{b['cycle']}{b['direction'][0]}"


def detect_steps(dl, abs_threshold=1.0, k_mad=10.0, window=51):
    """Indices i where dl[i]-dl[i-1] is an offset step: exceeds abs_threshold
    [1e-6 cm] AND k_mad x the local robust scale of point-to-point changes."""
    d = np.diff(dl)
    med = pd.Series(d).rolling(window, center=True, min_periods=11).median().values
    mad = pd.Series(np.abs(d - med)).rolling(window, center=True,
                                             min_periods=11).median().values
    mad = np.maximum(mad, 1e-3)
    flag = (np.abs(d - med) > abs_threshold) & (np.abs(d - med) > k_mad * mad)
    return np.where(flag)[0] + 1, d, med


def stitch_steps(dl, step_idx, d, med):
    """Remove each detected step by subtracting (observed jump - expected local
    change) from everything after it. Returns repaired array and a log."""
    out = dl.astype(float).copy()
    log = []
    for i in step_idx:
        jump = d[i - 1] - (med[i - 1] if np.isfinite(med[i - 1]) else 0.0)
        out[i:] -= jump
        log.append({"row_in_branch": int(i), "jump_1e6cm": float(jump)})
    return out, log


def robust_polyfit(T, y, deg=9, n_iter=3, clip=4.0):
    """Iteratively sigma-clipped polynomial fit (numpy Polynomial, scaled
    domain). Returns (poly, inlier_mask)."""
    mask = np.isfinite(T) & np.isfinite(y)
    for _ in range(n_iter):
        p = np.polynomial.Polynomial.fit(T[mask], y[mask], deg)
        res = y - p(T)
        s = 1.4826 * np.median(np.abs(res[mask] - np.median(res[mask])))
        new = mask & (np.abs(res - np.median(res[mask])) < clip * max(s, 1e-3))
        if new.sum() == mask.sum():
            break
        mask = new
    return p, mask


def find_exclusion_windows(T, y, poly, k=6.0, min_pts=25, pad_K=1.0):
    """T-windows where |residual| stays above k x global MAD for >= min_pts
    consecutive points -> transient excursions / noise patches to exclude."""
    res = y - poly(T)
    s = 1.4826 * np.median(np.abs(res - np.median(res)))
    bad = np.abs(res - np.median(res)) > k * max(s, 1e-3)
    windows, start = [], None
    for i, b in enumerate(bad):
        if b and start is None:
            start = i
        elif not b and start is not None:
            if i - start >= min_pts:
                lo, hi = sorted((T[start], T[i - 1]))
                windows.append([float(lo - pad_K), float(hi + pad_K)])
            start = None
    if start is not None and len(bad) - start >= min_pts:
        lo, hi = sorted((T[start], T[-1]))
        windows.append([float(lo - pad_K), float(hi + pad_K)])
    # merge overlapping
    merged = []
    for w in sorted(windows):
        if merged and w[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], w[1])
        else:
            merged.append(w)
    return merged


def in_windows(T, windows):
    m = np.zeros(len(T), bool)
    for lo, hi in windows:
        m |= (T >= lo) & (T <= hi)
    return m


def repair_branch(df, b, overrides=None):
    """Full repair for one branch. Returns dict with T, dl_raw, dl_repaired,
    exclusion windows, fit mask, step log, and the pre-fit polynomial."""
    ov = overrides or {}
    sl = slice(b["i0"], b["i1"])
    T = df[T_COL].values[sl].astype(float)
    dl_raw = df[DL_COL].values[sl].astype(float)

    step_idx, d, med = detect_steps(dl_raw)
    manual = [int(r) for r in ov.get("manual_jump_rows", [])]
    step_idx = np.unique(np.concatenate([step_idx, np.array(manual, int)])
                         ) if manual else step_idx
    dl_rep, step_log = stitch_steps(dl_raw, step_idx, d, med)

    poly, inliers = robust_polyfit(T, dl_rep)
    auto_windows = find_exclusion_windows(T, dl_rep, poly)
    windows = auto_windows + [list(map(float, w))
                              for w in ov.get("exclude_T_windows", [])]

    # Steps inside an exclusion window are the edges of a transient, not real
    # offsets — un-stitch them (keep only steps on trusted baseline).
    if len(step_idx) and windows:
        inside = in_windows(T[np.clip(step_idx, 0, len(T) - 1)], windows)
        if inside.any():
            step_idx = step_idx[~inside]
            dl_rep, step_log = stitch_steps(dl_raw, step_idx, d, med)
            poly, _ = robust_polyfit(T, dl_rep)
            auto_windows = find_exclusion_windows(T, dl_rep, poly)
            windows = auto_windows + [list(map(float, w))
                                      for w in ov.get("exclude_T_windows", [])]

    fit_mask = ~in_windows(T, windows)

    # second-pass pre-fit without excluded windows (better window estimate)
    if windows:
        poly, _ = robust_polyfit(T[fit_mask], dl_rep[fit_mask])

    return {"branch": b, "T": T, "dl_raw": dl_raw, "dl_repaired": dl_rep,
            "steps": step_log, "auto_windows": auto_windows,
            "manual_windows": ov.get("exclude_T_windows", []),
            "windows": windows, "fit_mask": fit_mask, "prefit": poly,
            "use": ov.get("use", True)}


# ═════════════════════════════════════════════════════════════════════════════
# QC plotting
# ═════════════════════════════════════════════════════════════════════════════

BRANCH_COLORS = {"cool": "tab:blue", "warm": "tab:red"}

def qc_plot_segmentation(key, df, branches, out_dir=None):
    out_dir = QC_DIR if out_dir is None else out_dir
    os.makedirs(out_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax1.plot(df.index, df[T_COL], "-", color="0.8", lw=0.8, label="_dwell")
    for b in branches:
        sl = slice(b["i0"], b["i1"])
        ax1.plot(df.index[sl], df[T_COL].iloc[sl], ".", ms=1,
                 color=BRANCH_COLORS[b["direction"]])
        ax2.plot(df[T_COL].iloc[sl], df[DL_COL].iloc[sl], ".", ms=1,
                 color=BRANCH_COLORS[b["direction"]])
        Tmid = 0.5 * (b["T_start"] + b["T_end"])
        ax2.annotate(f"c{b['cycle']}{b['direction'][0]}",
                     (Tmid, np.interp(Tmid,
                                      df[T_COL].iloc[sl].iloc[::max(1,(b['i1']-b['i0'])//50)],
                                      df[DL_COL].iloc[sl].iloc[::max(1,(b['i1']-b['i0'])//50)])),
                     fontsize=8, color=BRANCH_COLORS[b["direction"]])
    ax1.set_xlabel("row"); ax1.set_ylabel("T (K)")
    ax1.set_title(f"{key}: T(t) — blue cool / red warm / grey dwell", fontsize=10)
    ax2.set_xlabel("T (K)"); ax2.set_ylabel(DL_COL)
    ax2.set_title(f"{key}: branches (label = cycle+direction)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(out_dir, f"seg_{key}.png")
    fig.savefig(out, dpi=115); plt.close(fig)
    return out


def qc_plot_repair(key, repairs, out_dir=None):
    """One figure per file: each used branch gets a row (overlay + residual)."""
    out_dir = QC_DIR if out_dir is None else out_dir
    used = [r for r in repairs if r["use"]]
    if not used:
        return None
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(len(used), 2, figsize=(13, 3.6 * len(used)),
                             squeeze=False)
    for row, r in enumerate(used):
        b = r["branch"]
        ax, axr = axes[row]
        ax.plot(r["T"], r["dl_raw"], ".", ms=1, color="0.75", label="raw")
        ax.plot(r["T"], r["dl_repaired"], ".", ms=1,
                color=BRANCH_COLORS[b["direction"]], label="repaired")
        Tf = np.linspace(r["T"].min(), r["T"].max(), 400)
        ax.plot(Tf, r["prefit"](Tf), "k-", lw=1, label="pre-fit")
        for lo, hi in r["windows"]:
            ax.axvspan(lo, hi, color="gold", alpha=0.35)
        for s in r["steps"]:
            ax.axvline(r["T"][min(s["row_in_branch"], len(r["T"]) - 1)],
                       color="green", lw=0.8, ls=":")
        ax.set_title(f"{key} c{b['cycle']}{b['direction'][0]}  "
                     f"steps={len(r['steps'])}  windows={len(r['windows'])}",
                     fontsize=10)
        ax.set_xlabel("T (K)"); ax.set_ylabel(DL_COL)
        ax.legend(fontsize=7, markerscale=8)
        res = r["dl_repaired"] - r["prefit"](r["T"])
        axr.plot(r["T"][r["fit_mask"]], res[r["fit_mask"]], ".", ms=1,
                 color=BRANCH_COLORS[b["direction"]])
        axr.plot(r["T"][~r["fit_mask"]], res[~r["fit_mask"]], ".", ms=1,
                 color="gold")
        axr.axhline(0, color="k", lw=0.6)
        rms = np.sqrt(np.mean(res[r["fit_mask"]] ** 2))
        axr.set_title(f"residual vs pre-fit  RMS={rms:.2f} [1e-6 cm]",
                      fontsize=10)
        axr.set_xlabel("T (K)"); axr.set_ylabel("resid [1e-6 cm]")
    fig.tight_layout()
    out = os.path.join(out_dir, f"repair_{key}.png")
    fig.savefig(out, dpi=115); plt.close(fig)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Final fits + calibration registry + anchor gate
# ═════════════════════════════════════════════════════════════════════════════

# Legacy polynomials currently hardcoded in the analysis scripts (power basis,
# c0..c9 = r6,r7,r8,r12..r18). Used only for the anchor gate.
LEGACY_POLYS = {
    "_0_42mm": [413.53191, 0.12944, -0.01064, -3.13017e-4, 6.81607e-6,
                -6.1787e-8, 3.16305e-10, -9.5141e-13, 1.56896e-15, -1.0956e-18],
}

def _load_legacy_from_script(tag, script_name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
    src = open(path, encoding="utf-8").read()
    m = re.search(r"def cell_with_cu_fit\(.*?\):", src, re.S)
    vals = dict(re.findall(r"(r\d+)\s*=\s*([-\d.eE]+)", src[m.start():m.start() + 800]))
    LEGACY_POLYS[tag] = [float(vals[k]) for k in
                         ["r6", "r7", "r8", "r12", "r13", "r14",
                          "r15", "r16", "r17", "r18"]]

def legacy_eval(tag, T):
    return sum(c * np.asarray(T, float) ** i
               for i, c in enumerate(LEGACY_POLYS[tag]))


def fit_branch_final(r, deg=9):
    """Final calibration fit on repaired, window-masked data.
    Returns power-basis coefficients c0..c9 plus fit metrics."""
    T, y = r["T"][r["fit_mask"]], r["dl_repaired"][r["fit_mask"]]
    p = np.polynomial.Polynomial.fit(T, y, deg)
    coef = p.convert().coef                      # power basis, natural T
    coef = np.pad(coef, (0, deg + 1 - len(coef)))
    res = y - p(T)
    return {"coefficients": [float(c) for c in coef],
            "rms_1e6cm": float(np.sqrt(np.mean(res ** 2))),
            "max_resid_1e6cm": float(np.max(np.abs(res))),
            "T_fit_min": float(T.min()), "T_fit_max": float(T.max()),
            "n_fit_points": int(len(T))}


def build_registry(results):
    """Assemble the registry dict from processed files (branch records;
    eq7/transfer/field sections are added by later stages before saving)."""
    records = []
    for res in results:
        if res.get("repairs") is None:
            continue
        meta = res["meta"]
        for rp in res["repairs"]:
            if not rp["use"]:
                continue
            b = rp["branch"]
            fit = fit_branch_final(rp)
            records.append({
                "id": branch_key(res["key"], b),
                "cell": meta["cell"],
                "cu_length_mm": meta["cu_length_mm"],
                "branch": b["direction"],
                "cycle": b["cycle"],
                "C_start_pF": b["C_start"],
                "source_file": meta["path"],
                "cmax_ppms_pF": res["header"]["cmax_ppms_pF"],
                "corrections": {"steps": rp["steps"],
                                "auto_windows": rp["auto_windows"],
                                "manual_windows": rp["manual_windows"]},
                **fit,
            })
    registry = {"_doc": "Cell-background calibrations built by "
                        "cu_calibration_builder.py. delta_l convention: "
                        "positive = sample expansion (P18 Eqs. 3-6); subtract "
                        "record polynomial directly from measured delta_l.",
                "records": records}
    return registry


def save_registry(registry, out_path=None):
    if out_path is None:
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "calibrations.json")
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as fh:
                if json.load(fh).get("_example_registry"):
                    print("\nreplacing the shipped EXAMPLE registry with your "
                          "own build — the example-registry banner goes away")
        except (OSError, ValueError):
            pass
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=1)
    print(f"\nregistry: {out_path}  ({len(registry['records'])} records)")


ANCHORS = [
    # (record id, legacy tag, tolerance on max|Δ| [1e-6 cm], compare mode)
    ("mini_1mm/c1w",   "_1mm_mini_dil", 2.0, "absolute"),
    ("str_0.42mm/c1w", "_0_42mm",       7.0, "absolute"),
    ("str_1mm/c1w",    "_3mm",          5.0, "shape"),
]

def anchor_gate(registry):
    """Gate 1: new warming fits must reproduce the legacy script polynomials.
    Applies only to the authors' setup — skipped (PASS) when the legacy
    scripts referenced in ANCHORS are not present."""
    try:
        _load_legacy_from_script("_1mm_mini_dil", "Dilatometer_V9_fit_1mm_mini_dil.py")
        _load_legacy_from_script("_3mm", "Dilatometer_V9_fit_3mm.py")
    except FileNotFoundError:
        print("\nANCHOR GATE skipped: no legacy scripts to compare against "
              "(only relevant to the original setup) -> PASS")
        return True
    recs = {r["id"]: r for r in registry["records"]}
    print("\nANCHOR GATE (new warming fits vs legacy script polynomials)")
    all_pass = True
    for rid, tag, tol, mode in ANCHORS:
        r = recs.get(rid)
        if r is None:
            print(f"  {rid:16s} MISSING record -> FAIL"); all_pass = False
            continue
        T = np.linspace(max(5.0, r["T_fit_min"]), min(295.0, r["T_fit_max"]), 400)
        new = sum(c * T ** i for i, c in enumerate(r["coefficients"]))
        old = legacy_eval(tag, T)
        diff = new - old
        if mode == "shape":
            diff = diff - np.mean(diff)
        mx = float(np.max(np.abs(diff)))
        ok = mx <= tol
        all_pass &= ok
        print(f"  {rid:16s} vs {tag:14s} ({mode:8s}) "
              f"max|Δ|={mx:5.2f}  tol={tol}  -> {'PASS' if ok else 'FAIL'}")
    print(f"  => anchor gate {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ═════════════════════════════════════════════════════════════════════════════
# Eq. (7) decomposition + gates 2-3 (round-trip / transfer) + hysteresis QC
# ═════════════════════════════════════════════════════════════════════════════

def cu_lit_data_fit(T, r3=16549.7494, r4=-578.02392, r5=56.45939):
    """Cu literature relative thermal expansion [1e-6], same constants as the
    analysis scripts (referencing to T_min happens in the reduction)."""
    return r3 * np.exp(r4 / (np.asarray(T, float) + r5))


def eval_record(rec, T):
    return sum(c * np.asarray(T, float) ** i
               for i, c in enumerate(rec["coefficients"]))


def round_trip_gate(results, registry):
    """Gate 2: each Cu run reduced with its own calibration must return Cu
    literature. dev(T) = resid(T) - resid(T_min) because the reduction
    references at T_min; the centered RMS must match the fit RMS (gate),
    while the T_min anchor offset is bounded by the recorded max_resid and
    reported, not gated (it is <~3e-6 cm, negligible vs transfer errors)."""
    recs = {r["id"]: r for r in registry["records"]}
    print("\nROUND-TRIP GATE (each Cu run reduced with its own calibration)")
    all_pass = True
    for res in results:
        if res.get("repairs") is None:
            continue
        for rp in res["repairs"]:
            if not rp["use"]:
                continue
            rid = branch_key(res["key"], rp["branch"])
            rec = recs[rid]
            L = rec["cu_length_mm"] / 10.0          # cm
            T = rp["T"][rp["fit_mask"]]
            dl = rp["dl_repaired"][rp["fit_mask"]]
            resid = dl - eval_record(rec, T)        # [1e-6 cm]
            anchor = float(resid[np.argmin(T)])     # offset the reduction adds
            dev = resid - anchor
            rms_c = float(np.sqrt(np.mean((dev - dev.mean()) ** 2)))
            mx = float(np.max(np.abs(dev)))
            tol = 1.2 * rec["rms_1e6cm"] + 0.05
            ok = rms_c <= tol
            all_pass &= ok
            print(f"  {rid:18s} RMS={rms_c:5.2f} (tol {tol:.2f})  "
                  f"max|Δ|={mx:5.2f}  T_min anchor={anchor:+5.2f} [1e-6 cm]  "
                  f"-> {'PASS' if ok else 'FAIL'}")
    print(f"  => round-trip gate {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def transfer_gate(results, registry, pairs):
    """Gate 3 (honest uncertainty): reduce each Cu run with the same cell's
    other calibration; deviation from Cu literature -> transfer error.
    Writes registry['transfer'] and the per-cell QC figures.

    pairs = cu_runs.json "transfer_pairs": [reduced_id, calibration_id, kind]
    with kind "same-length" (same cell+length, different cooldowns — isolates
    pure cell irreproducibility) or "cross-length" (measures the
    closest-length-mode error, which contains [cu_lit - ΔL''] x length
    mismatch by construction). Which pairs are meaningful is a judgment call
    on your run set, so there is no automatic default: pairs=None skips the
    gate and the registry carries no transfer section (consumers treat that
    as 'uncertainty not quantified')."""
    if not pairs:
        print("\nTRANSFER GATE skipped: no transfer_pairs in cu_runs.json — "
              "declare same-length / cross-length record pairs to quantify "
              "the cell's transfer uncertainty (see module docstring).")
        return None
    data = {}
    for res in results:
        if res.get("repairs") is None:
            continue
        for rp in res["repairs"]:
            if rp["use"]:
                data[branch_key(res["key"], rp["branch"])] = rp
    recs = {r["id"]: r for r in registry["records"]}

    print("\nTRANSFER GATE (each Cu run reduced with the cell's other calibration)")
    by_cell = {}
    curves = {}
    for tid, cid, kind in pairs:
        if tid not in data or cid not in recs:
            print(f"  {tid} <- {cid}: missing, skipped")
            continue
        rp, rec_t, rec_c = data[tid], recs[tid], recs[cid]
        lo = max(rec_t["T_fit_min"], rec_c["T_fit_min"])
        hi = min(rec_t["T_fit_max"], rec_c["T_fit_max"])
        m = rp["fit_mask"] & (rp["T"] >= lo) & (rp["T"] <= hi)
        T, dl = rp["T"][m], rp["dl_repaired"][m]
        L_t = rec_t["cu_length_mm"] / 10.0
        dev = (dl - eval_record(rec_c, T)) / L_t    # ΔL/L deviation [1e-6]
        dev -= dev[np.argmin(T)]
        dev_cm = dev * L_t                          # cell-level [1e-6 cm]
        entry = {"reduced": tid, "with": cid, "kind": kind,
                 "T_range": [float(T.min()), float(T.max())],
                 "rms_1e6cm": float(np.sqrt(np.mean(dev_cm ** 2))),
                 "max_1e6cm": float(np.max(np.abs(dev_cm))),
                 "max_dLoverL_1e6": float(np.max(np.abs(dev)))}
        mism_cm = abs(rec_t["cu_length_mm"] - rec_c["cu_length_mm"]) / 10.0
        if kind == "cross-length":
            entry["max_1e6cm_per_cm_mismatch"] = entry["max_1e6cm"] / mism_cm
        cell = rec_t["cell"]
        by_cell.setdefault(cell, []).append(entry)
        curves.setdefault(cell, []).append((f"{tid} ← {cid} ({kind})", T, dev_cm))
        extra = (f"  ({entry['max_1e6cm_per_cm_mismatch']:.0f}/cm mismatch)"
                 if kind == "cross-length" else "")
        print(f"  {tid:16s} <- {cid:18s} {kind:12s} "
              f"RMS={entry['rms_1e6cm']:6.2f}  max|Δ|={entry['max_1e6cm']:6.2f} "
              f"[1e-6 cm]{extra}")

    transfer = {}
    for cell, entries in by_cell.items():
        summ = {"pairs": entries, "per_branch": {}}
        for br, letter in (("warm", "w"), ("cool", "c")):
            bre = [e for e in entries if e["reduced"].endswith(letter)]
            same = [e for e in bre if e["kind"] == "same-length"]
            cross = [e for e in bre if e["kind"] == "cross-length"]
            b = {"same_length_rms_1e6cm": max(e["rms_1e6cm"] for e in same)
                                          if same else None,
                 "same_length_max_1e6cm": max(e["max_1e6cm"] for e in same)
                                          if same else None,
                 "cross_length_max_1e6cm": max(e["max_1e6cm"] for e in cross)
                                           if cross else None}
            if same:
                b["transfer_uncertainty_1e6cm"] = b["same_length_max_1e6cm"]
                b["basis"] = ("same-length pairs (cell irreproducibility "
                              "across cooldowns)")
            else:
                b["transfer_uncertainty_1e6cm"] = b["cross_length_max_1e6cm"]
                b["basis"] = ("cross-length pairs only — includes the "
                              "[cu_lit - ΔL''] x length-mismatch term; scale "
                              "by actual sample/Cu mismatch via "
                              "max_1e6cm_per_cm_mismatch")
            summ["per_branch"][br] = b
            print(f"  => {cell}/{br}: transfer_uncertainty = "
                  f"{b['transfer_uncertainty_1e6cm']:.1f} x1e-6 cm ({b['basis']})")
        transfer[cell] = summ
    registry["transfer"] = transfer

    for cell, cvs in curves.items():
        os.makedirs(QC_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, T, dev_cm in cvs:
            ax.plot(T, dev_cm, lw=1, label=label)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("T (K)"); ax.set_ylabel("transfer deviation [1e-6 cm]")
        ax.set_title(f"{cell}: Cu run reduced with other calibration "
                     f"(deviation from Cu literature)", fontsize=10)
        ax.legend(fontsize=7)
        fig.tight_layout()
        out = os.path.join(QC_DIR, f"transfer_{cell}.png")
        fig.savefig(out, dpi=115); plt.close(fig)
        print(f"  QC: {out}")
    return transfer


def derive_eq7_pairs(records):
    """Default eq7_pairs when cu_runs.json declares none: per cell and
    branch, the cycle-1 records of the two SHORTEST distinct Cu lengths
    (shorter first; ties broken by run order, i.e. first record of that
    length in the registry). Cells with a single Cu length get no pair —
    the reduction falls back to closest-length records."""
    pairs = {}
    cells = list(dict.fromkeys(r["cell"] for r in records))
    for cell in cells:
        for br in ("warm", "cool"):
            by_len = {}
            for r in records:
                if (r["cell"] == cell and r["branch"] == br
                        and r["cycle"] == 1
                        and r["cu_length_mm"] not in by_len):
                    by_len[r["cu_length_mm"]] = r["id"]
            if len(by_len) >= 2:
                la, lb = sorted(by_len)[:2]
                pairs.setdefault(cell, {})[br] = (by_len[la], by_len[lb])
    return pairs


def eq7_decompose(registry, pairs=None):
    """P18 Eq. (7): ΔL_cell(T; L) = ΔL'(T) + ΔL''(T)·L from the two Cu lengths
    per cell and branch, where ΔL_cell = cu_lit·L - δl_Cu.

    Both parts are exact given the branch polynomials p_A, p_B: with
    D = (p_B - p_A)/(L_B - L_A) [1e-6 cm per cm],
        ΔL'(T)  = D(T)·L_A - p_A(T)          (exact degree-9 polynomial)
        ΔL''(T) = cu_lit(T) - D(T)           (stored as D; cu_lit is analytic)
    so the virtual Cu curve for sample length L_s reduces to
        δl_virt(T; L_s) = p_A(T) + D(T)·(L_s - L_A)
    i.e. length-interpolation of the two measured calibrations.

    Resolved test: constant offsets in ΔL'' are arbitrary (each run's δl zero
    is arbitrary), so the signal is the T-variation std of ΔL''; noise is the
    larger of fit-RMS propagation and same-length irreproducibility (from the
    transfer gate), both / |L_B - L_A|."""
    recs = {r["id"]: r for r in registry["records"]}
    if pairs is None:
        pairs = derive_eq7_pairs(registry["records"])
    transfer = registry.get("transfer", {})
    eq7 = {"_doc": "delta_l_virt(T;L_s)[1e-6 cm] = dl1 poly + "
                   "(cu_lit(T) - dl2 poly)*L_s subtracted per the standard "
                   "formula; equivalently p_A(T) + D(T)*(L_s - L_A), L in cm. "
                   "dl2_poly_coefficients = D; DL''(T) = cu_lit(T) - D(T)."}
    print("\nEQ.(7) LENGTH DECOMPOSITION")
    if not pairs:
        print("  no eq7 pairs (a cell needs two Cu lengths) — reductions "
              "will use closest-length records")
    for cell, branches in pairs.items():
        eq7[cell] = {}
        for br, (ida, idb) in branches.items():
            missing = [i for i in (ida, idb) if i not in recs]
            if missing:
                raise SystemExit(
                    f"eq7_pairs {cell}/{br}: record(s) {missing} not in the "
                    f"registry; available: {', '.join(sorted(recs))}")
            ra, rb = recs[ida], recs[idb]
            La, Lb = ra["cu_length_mm"] / 10.0, rb["cu_length_mm"] / 10.0
            lo = max(ra["T_fit_min"], rb["T_fit_min"])
            hi = min(ra["T_fit_max"], rb["T_fit_max"])
            pa = np.array(ra["coefficients"])
            pb = np.array(rb["coefficients"])
            D = (pb - pa) / (Lb - La)
            dl1 = D * La - pa                      # ΔL' coefficients [1e-6 cm]

            T = np.linspace(lo, hi, 800)
            dl2 = cu_lit_data_fit(T) - sum(c * T ** i for i, c in enumerate(D))
            signal = float(np.std(dl2 - dl2.mean()))
            noise_stat = float(np.hypot(ra["rms_1e6cm"], rb["rms_1e6cm"])
                               / abs(Lb - La))
            same = (transfer.get(cell, {}).get("per_branch", {})
                    .get(br, {}).get("same_length_rms_1e6cm"))
            noise_sys = (float(np.sqrt(2) * same / abs(Lb - La))
                         if same is not None else None)
            noise = max(noise_stat, noise_sys or 0.0)
            resolved = signal > 3.0 * noise
            note = None
            if same is None:
                note = ("no same-length pair for this cell: noise is fit-RMS "
                        "propagation only (lower bound on the true "
                        "run-to-run systematic)")
            eq7[cell][br] = {
                "records": [ida, idb], "T_range": [float(lo), float(hi)],
                "dl1_coefficients": [float(c) for c in dl1],
                "dl2_poly_coefficients": [float(c) for c in D],
                "dl2_signal_std_1e6": signal,
                "dl2_noise_stat_1e6": noise_stat,
                "dl2_noise_sys_1e6": noise_sys,
                "resolved": bool(resolved),
                **({"note": note} if note else {}),
            }
            print(f"  {cell}/{br}: ΔL'' variation std={signal:.1f}, noise "
                  f"stat={noise_stat:.1f} sys="
                  f"{'-' if noise_sys is None else f'{noise_sys:.1f}'} [1e-6] "
                  f"-> {'RESOLVED' if resolved else 'NOT resolved '
                       '(reduction falls back to closest-length)'}")
    registry["eq7"] = eq7
    qc_plot_eq7(registry)
    return eq7


def qc_plot_eq7(registry):
    recs = {r["id"]: r for r in registry["records"]}
    for cell, branches in registry["eq7"].items():
        if cell == "_doc":
            continue
        rows = [(br, branches[br]) for br in branches]
        fig, axes = plt.subplots(len(rows), 3, figsize=(13, 3.8 * len(rows)),
                                 squeeze=False)
        for row, (br, e) in enumerate(rows):
            ax_c, ax1, ax2 = axes[row]
            T = np.linspace(*e["T_range"], 500)
            for rid in e["records"]:
                r = recs[rid]
                L = r["cu_length_mm"] / 10.0
                ce = cu_lit_data_fit(T) * L - eval_record(r, T)
                ax_c.plot(T, ce - ce[0], lw=1,
                          label=f"{rid} (L={r['cu_length_mm']} mm)")
            ax_c.set_title(f"{cell} {br}: ΔL_cell (referenced)", fontsize=9)
            ax_c.legend(fontsize=7)
            dl1 = sum(c * T ** i for i, c in enumerate(e["dl1_coefficients"]))
            ax1.plot(T, dl1 - dl1[0], "k-", lw=1)
            ax1.set_title("ΔL' (length-independent, referenced)", fontsize=9)
            dl2 = cu_lit_data_fit(T) - sum(
                c * T ** i for i, c in enumerate(e["dl2_poly_coefficients"]))
            dl2c = dl2 - dl2.mean()
            n = 3 * max(e["dl2_noise_stat_1e6"], e["dl2_noise_sys_1e6"] or 0)
            ax2.plot(T, dl2c, "k-", lw=1)
            ax2.axhspan(-n, n, color="tab:orange", alpha=0.25,
                        label="±3×noise")
            ax2.set_title(f"ΔL'' (mean-removed) — "
                          f"{'RESOLVED' if e['resolved'] else 'NOT resolved'}",
                          fontsize=9)
            ax2.legend(fontsize=7)
            for ax in (ax_c, ax1, ax2):
                ax.set_xlabel("T (K)"); ax.set_ylabel("[1e-6 cm]")
            ax2.set_ylabel("[1e-6 cm / cm]")
        fig.tight_layout()
        out = os.path.join(QC_DIR, f"eq7_{cell}.png")
        fig.savefig(out, dpi=115); plt.close(fig)
        print(f"  QC: {out}")


def derive_hysteresis_pairs(records):
    """Default hysteresis_pairs when cu_runs.json declares none: every
    (file, cycle) that carries both a cool and a warm record, in record
    order, as (cool_id, warm_id)."""
    ids = {r["id"] for r in records}
    pairs = []
    for r in records:
        if r["branch"] == "cool":
            wid = r["id"][:-1] + "w"
            if wid in ids:
                pairs.append((r["id"], wid))
    return pairs


def qc_plot_hysteresis(registry, pairs=None):
    """Gate 4 artifact: cooling - warming difference per (file, cycle),
    referenced at the common T_min — the branches are contiguous in time at
    the turnaround, so any offset there is stitch bookkeeping, not physics;
    what remains is the hysteresis loop opening. Expected: str cell tens of
    1e-6 cm, mini cell near zero. QC figure only — never written to the
    registry. pairs from cu_runs.json "hysteresis_pairs" (lets you exclude
    e.g. a mechanically-shifted re-cooling cycle); None -> derived."""
    recs = {r["id"]: r for r in registry["records"]}
    if pairs is None:
        pairs = derive_hysteresis_pairs(registry["records"])
    cells = sorted({r["cell"] for r in registry["records"]})
    fig, axes = plt.subplots(1, len(cells), figsize=(6.5 * len(cells), 4.8),
                             squeeze=False)
    axmap = dict(zip(cells, axes[0]))
    print("\nHYSTERESIS QC (cooling - warming opening, referenced at T_min)")
    for cid, wid in pairs:
        if cid not in recs or wid not in recs:
            continue
        rc, rw = recs[cid], recs[wid]
        lo = max(rc["T_fit_min"], rw["T_fit_min"])
        hi = min(rc["T_fit_max"], rw["T_fit_max"])
        T = np.linspace(lo, hi, 500)
        diff = eval_record(rc, T) - eval_record(rw, T)
        diff -= diff[0]                    # reference at common T_min
        ax = axmap[rc["cell"]]
        ax.plot(T, diff, lw=1, label=f"{cid} − {wid}")
        print(f"  {cid:18s} - {wid:18s}  opening max|Δ|="
              f"{np.max(np.abs(diff)):5.1f} [1e-6 cm]")
    for cell, ax in axmap.items():
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("T (K)"); ax.set_ylabel("δl_cool − δl_warm [1e-6 cm]")
        ax.set_title(f"{cell}: cooling/warming hysteresis", fontsize=10)
        ax.legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(QC_DIR, "hysteresis.png")
    fig.savefig(out, dpi=115); plt.close(fig)
    print(f"  QC: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Field background (Cu_field_1_mm): a(T)·|B|^n per P2/P20
# ═════════════════════════════════════════════════════════════════════════════

FIELD_T_TARGETS = [2, 5, 10, 15, 20, 30, 50, 100, 120, 150, 300]
FIELD_N_FIT_TMAX = 50      # ≥100 K sweeps are drift-dominated: excluded from
                           # the global-n fit, a(T) still reported + flagged

def extract_field_sweeps(df, t_targets=None):
    """Split the field file into fixed-T plateaus, each containing one
    0→9→0 T sweep with B≈0 dwells before/after. Per plateau: remove drift
    linear in time anchored on the pre/post B≈0 dwell means, reference to
    the pre-sweep dwell. Returns list of sweep dicts. t_targets = the run's
    fixed-T setpoints in K (cu_runs.json "t_targets" on the field run);
    default FIELD_T_TARGETS."""
    t_targets = FIELD_T_TARGETS if t_targets is None else t_targets
    T = df[T_COL].values
    B = df["B [T]"].values
    t = df["Rel Time"].values
    dl = df[DL_COL].values
    lab = np.full(len(df), -1)
    for k, tg in enumerate(t_targets):
        lab[np.abs(T - tg) < max(0.02 * tg, 0.35)] = k
    sweeps = []
    s = 0
    for i in range(1, len(df) + 1):
        if i == len(df) or lab[i] != lab[s]:
            if lab[s] >= 0 and i - s > 50:
                tg = t_targets[lab[s]]
                b = B[s:i]
                on = np.where(np.abs(b) > 0.05)[0]
                if len(on) > 30:                     # has a sweep
                    j0, j1 = s + on[0], s + on[-1] + 1
                    pre = slice(max(s, j0 - 40), j0)
                    post = slice(j1, min(i, j1 + 40))
                    if j0 - pre.start < 5 or post.stop - j1 < 5:
                        continue
                    t0, y0 = t[pre].mean(), np.median(dl[pre])
                    t1, y1 = t[post].mean(), np.median(dl[post])
                    slope = (y1 - y0) / (t1 - t0)
                    sl = slice(j0, j1)
                    corr = dl[sl] - (y0 + slope * (t[sl] - t0))
                    sweeps.append({
                        "T_target": tg, "T_mean": float(T[sl].mean()),
                        "B": B[sl], "dl_corr": corr, "t": t[sl],
                        "drift_slope_1e6cm_per_h": float(slope * 3600),
                        "closure_1e6cm": float(y1 - y0),
                        "up": B[sl][np.r_[np.diff(B[sl]) > 0, False]],
                    })
            s = i
    return sweeps


def symmetrize_sweep(sw, n_grid=60):
    """Interpolate up and down half-sweeps on a common B grid. The
    symmetrized average (up+down)/2 cancels drift linear in time (the ramp
    is time-symmetric about the 9 T apex); (up-down)/2 isolates drift and
    dB/dt-proportional effects."""
    B, y = sw["B"], sw["dl_corr"]
    iapex = int(np.argmax(B))
    Bg = np.linspace(0.3, 0.98 * B.max(), n_grid)
    iu = np.argsort(B[:iapex]); idn = np.argsort(B[iapex:])
    up = np.interp(Bg, B[:iapex][iu], y[:iapex][iu])
    dn = np.interp(Bg, B[iapex:][idn], y[iapex:][idn])
    sym = 0.5 * (up + dn)
    return Bg, sym - sym[0], 0.5 * (up - dn)


def field_background(registry, field_runs):
    """Step 5 — measured outcome (2026-07-02): the smooth reversible
    background a(T)·|B|^n of P2/P20 (~0.3e-6 cm at 9 T) is NOT resolvable in
    this run. After drift removal and up/down symmetrization the sweeps show
    (a) a noise floor of ~0.15-0.25e-6 cm where they are featureless
    (15/20/50 K) and (b) discrete stick-slip snaps of 0.4-2e-6 cm at specific
    fields (2/5/10/30 K), one sweep per T so their reproducibility cannot be
    validated. Fitting a·|B|^n to snap-contaminated data (best fit n≈0.75 vs
    P2/P20 n=1.7-1.8, amplitude ~100x P2) would add error, not remove it.
    Per the design spec ('flag, don't silently apply') the record is written
    with apply=false and a per-T uncertainty ENVELOPE for magnetostriction
    reductions instead of a correction.

    field_runs = [(key, meta)] for the spec runs with kind == "field". None
    -> section skipped; more than one -> the first is used (the registry
    format carries a single field_background section)."""
    if not field_runs:
        print("\nFIELD BACKGROUND skipped: no kind=\"field\" run in "
              "cu_runs.json — the registry carries no field_background "
              "section (magnetostriction reductions then state no "
              "cell-background envelope).")
        return None
    if len(field_runs) > 1:
        print(f"\nFIELD BACKGROUND: multiple field runs "
              f"({', '.join(k for k, _ in field_runs)}) — using the first; "
              f"the registry format carries a single field_background "
              f"section.")
    key, meta = field_runs[0]
    df, header = load_dat(_resolve(meta["path"]))
    sweeps = extract_field_sweeps(df, meta.get("t_targets"))
    print(f"\nFIELD BACKGROUND ({key}, {len(sweeps)} sweeps) — "
          f"envelope, correction NOT applied")
    env = []
    for sw in sweeps:
        Bg, sym, asym = symmetrize_sweep(sw)
        # noise floor: robust scale of point-to-point sym changes
        floor = float(1.4826 * np.median(np.abs(np.diff(sym))) / np.sqrt(2))
        span = float(np.max(np.abs(sym)))
        # snap detection: grid-step changes far above the floor
        d = np.diff(sym)
        snaps = [{"B_T": float(0.5 * (Bg[i] + Bg[i + 1])),
                  "step_1e6cm": float(d[i])}
                 for i in np.where(np.abs(d) > max(6 * floor, 0.15))[0]]
        drift_dom = sw["T_target"] > FIELD_N_FIT_TMAX
        env.append({"T_K": sw["T_target"],
                    "envelope_1e6cm": span,
                    "sym_noise_floor_1e6cm": floor,
                    "snaps": snaps,
                    "drift_slope_1e6cm_per_h": sw["drift_slope_1e6cm_per_h"],
                    "drift_dominated": drift_dom})
        print(f"  T={sw['T_target']:5.0f} K  envelope={span:5.2f}  "
              f"floor={floor:5.2f}  snaps={len(snaps)}  "
              f"drift={sw['drift_slope_1e6cm_per_h']:+7.2f}/h [1e-6 cm]"
              f"{'  [drift-dominated]' if drift_dom else ''}")
    p2_at_9T = 5.2e-3 * 9.0 ** 1.8
    print(f"  P2-scale smooth background at 9 T = {p2_at_9T:.2f} x1e-6 cm — "
          f"comparable to the residual-drift envelope at the snap-free "
          f"temperatures (0.1-0.25) and buried under snaps elsewhere: "
          f"unresolved with one sweep per T. Correction NOT applied; "
          f"magnetostriction reductions carry the envelope as uncertainty.")

    registry["field_background"] = {
        "_doc": "Field background could NOT be resolved as a smooth "
                "a(T)*|B|^n correction in this Cu run: sweeps show discrete "
                "stick-slip snaps (0.4-2e-6 cm) and drift, while the "
                "P2/P20-scale smooth background (~0.3e-6 cm at 9 T) is "
                "comparable to the 0.1-0.25e-6 cm residual-drift envelope at "
                "the snap-free temperatures. apply=false: magnetostriction "
                "reductions subtract nothing and state envelope_1e6cm(T) as "
                "the cell-background uncertainty at 9 T.",
        "cell": meta["cell"], "cu_length_mm": meta["cu_length_mm"],
        "source_file": meta["path"],
        "apply": False,
        "p2_scale_at_9T_1e6cm": p2_at_9T,
        "envelope_vs_T": env,
    }
    qc_plot_field(sweeps, env)
    return registry["field_background"]


def qc_plot_field(sweeps, env, out_dir=None):
    """Per T: raw up/down (drift-removed) + symmetrized curve with snaps and
    the P2-scale reference; last panel = envelope(T) vs P2 scale."""
    out_dir = QC_DIR if out_dir is None else out_dir
    os.makedirs(out_dir, exist_ok=True)
    ncol = 4
    nrow = int(np.ceil((len(sweeps) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.9 * nrow))
    axes = axes.ravel()
    emap = {e["T_K"]: e for e in env}
    for ax, sw in zip(axes, sweeps):
        e = emap[sw["T_target"]]
        up = np.r_[np.diff(sw["B"]) > 0, False]
        ax.plot(sw["B"][up], sw["dl_corr"][up], ".", ms=1.5, color="mistyrose")
        ax.plot(sw["B"][~up], sw["dl_corr"][~up], ".", ms=1.5,
                color="lightblue")
        Bg, sym, _ = symmetrize_sweep(sw)
        ax.plot(Bg, sym, "k.-", ms=2.5, lw=0.9, label="symmetrized")
        Bf = np.linspace(0, 9, 100)
        ax.plot(Bf, 5.2e-3 * Bf ** 1.8, "g--", lw=1, label="P2 scale")
        for s in e["snaps"]:
            ax.axvline(s["B_T"], color="tab:orange", lw=0.8, ls=":")
        ax.set_title(f"T={sw['T_target']} K  env={e['envelope_1e6cm']:.2f}  "
                     f"snaps={len(e['snaps'])}"
                     f"{' [drift]' if e['drift_dominated'] else ''}",
                     fontsize=8)
        ax.set_xlabel("B (T)", fontsize=8); ax.set_ylabel("[1e-6 cm]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, markerscale=3)
    ax = axes[len(sweeps)]
    good = [e for e in env if not e["drift_dominated"]]
    bad = [e for e in env if e["drift_dominated"]]
    ax.plot([e["T_K"] for e in good], [e["envelope_1e6cm"] for e in good],
            "o-", color="tab:red", ms=4, label="envelope (uncertainty)")
    ax.plot([e["T_K"] for e in bad], [e["envelope_1e6cm"] for e in bad],
            "s", color="tab:orange", ms=4, label="drift-dominated")
    ax.plot([e["T_K"] for e in good],
            [e["sym_noise_floor_1e6cm"] for e in good], "^-",
            color="0.5", ms=3, label="noise floor")
    ax.axhline(5.2e-3 * 9 ** 1.8, color="g", ls="--", lw=1, label="P2 at 9 T")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("field-background envelope at 9 T (NOT applied)", fontsize=8)
    ax.set_xlabel("T (K)", fontsize=8); ax.set_ylabel("[1e-6 cm]", fontsize=8)
    ax.tick_params(labelsize=7); ax.legend(fontsize=6)
    for ax in axes[len(sweeps) + 1:]:
        ax.axis("off")
    fig.tight_layout()
    out = os.path.join(out_dir, "fieldbg_str_dil.png")
    fig.savefig(out, dpi=115); plt.close(fig)
    print(f"  QC: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# Driver
# ═════════════════════════════════════════════════════════════════════════════

def process_file(key, meta, config=None):
    path = _resolve(meta["path"])
    df, header = load_dat(path)
    branches = segment_branches(df)
    out = qc_plot_segmentation(key, df, branches)
    print(f"\n{key}  ({meta['cell']}, Cu {meta['cu_length_mm']} mm)  "
          f"Cmax(PPMS)={header['cmax_ppms_pF']} pF  rows={len(df)}")
    for b in branches:
        print(f"  cycle {b['cycle']} {b['direction']:4s}  "
              f"T {b['T_start']:6.1f} -> {b['T_end']:6.1f} K  "
              f"C0={b['C_start']:.2f} pF  n={b['n_points']}")
    print(f"  QC: {out}")

    repairs = None
    if meta.get("kind") != "field":
        config = config or {}
        repairs = [repair_branch(df, b, config.get(branch_key(key, b)))
                   for b in branches]
        rout = qc_plot_repair(key, repairs)
        for r in repairs:
            b = r["branch"]
            flag = "" if r["use"] else "  [not used for fits]"
            print(f"    repair c{b['cycle']}{b['direction'][0]}: "
                  f"{len(r['steps'])} step(s) stitched, "
                  f"windows {r['windows']}{flag}")
        print(f"  repair QC: {rout}")
    return {"key": key, "meta": meta, "header": header, "branches": branches,
            "repairs": repairs}


def main(argv):
    import argparse
    global DATA_DIR, QC_DIR
    ap = argparse.ArgumentParser(
        prog="cu_calibration_builder.py",
        description="Build the cell-background calibration registry "
                    "(calibrations.json) from raw Cu reference runs "
                    "described by a cu_runs.json (see the module docstring; "
                    "worked example: scripts/cu_runs.example.json).")
    ap.add_argument("--data", metavar="DIR", default=DATA_DIR,
                    help="data directory: cu_runs.json is looked up here and "
                         "relative run paths resolve against it "
                         "(default: the repo root)")
    ap.add_argument("--runs", metavar="SPEC", default=None,
                    help="run-spec JSON (default: <data>/cu_runs.json)")
    ap.add_argument("--file", metavar="KEY", default=None,
                    help="process a single Cu run by key (QC only — the "
                         "registry is built only on a full run)")
    ap.add_argument("--out", metavar="REGISTRY", default=None,
                    help="registry output path "
                         "(default: scripts/calibrations.json — the file the "
                         "reduction scripts read; see also DILAT_CALIBRATIONS)")
    ap.add_argument("--qc-dir", metavar="DIR", default=QC_DIR,
                    help="QC figure directory (default: %(default)s)")
    args = ap.parse_args(argv)
    DATA_DIR = args.data
    QC_DIR = args.qc_dir
    runs, pairs = load_runs_spec(args.runs, DATA_DIR)
    only = args.file
    if only is not None and only not in runs:
        raise SystemExit(f"--file {only!r}: not in the run spec; available: "
                         f"{', '.join(runs)}")
    config = load_config()
    results = []
    for key, meta in runs.items():
        if only and key != only:
            continue
        results.append(process_file(key, meta, config))
    summary = os.path.join(QC_DIR, "segmentation_summary.json")
    os.makedirs(QC_DIR, exist_ok=True)
    with open(summary, "w", encoding="utf-8") as fh:
        json.dump([{"key": r["key"], "meta": r["meta"], "header": r["header"],
                    "branches": r["branches"],
                    "repairs": [{"branch": branch_key(r["key"], rp["branch"]),
                                 "steps": rp["steps"],
                                 "windows": rp["windows"],
                                 "use": rp["use"]}
                                for rp in (r["repairs"] or [])]}
                   for r in results], fh, indent=1)
    print(f"\nsummary: {summary}")

    if only is None:                # full run -> registry + gates 1-3 + eq7
        registry = build_registry(results)
        g1 = anchor_gate(registry)
        g2 = round_trip_gate(results, registry)
        transfer_gate(results, registry, pairs["transfer"])
        eq7_decompose(registry, pairs["eq7"])
        qc_plot_hysteresis(registry, pairs["hysteresis"])
        field_background(registry, [(k, m) for k, m in runs.items()
                                    if m.get("kind") == "field"])
        save_registry(registry, args.out)
        print(f"\nGates: anchor {'PASS' if g1 else 'FAIL'}, "
              f"round-trip {'PASS' if g2 else 'FAIL'}, "
              f"transfer recorded per cell (see registry['transfer']).")


if __name__ == "__main__":
    main(sys.argv[1:])
