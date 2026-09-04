"""
calibration_bridge.py — minimal, dependency-light bridge to calibrations.json.

Lets ANY analysis script (legacy or current, any OS) replace its hardcoded legacy
cell-background polynomial with a registry calibration:

    from calibration_bridge import registry_cell_fit
    cell_with_cu_fit = registry_cell_fit(record_id="str_0.42mm/c1w")
    # or, resolved automatically:
    cell_with_cu_fit = registry_cell_fit(cell="str_dil", branch="warm",
                                         sample_thickness_cm=0.042)

The returned callable maps T [K] -> cell+Cu background delta_l [1e-6 cm],
exactly like the legacy cell_with_cu_fit(T) polynomials it replaces.

Legacy (pre-registry) scripts are not branch-aware: use branch="warm" (the legacy polynomials
were warming-branch fits). Branch-aware cool/warm reduction needs the current
scripts. calibrations.json is looked up next to this file; override with
the DILAT_CALIBRATIONS environment variable or the path= argument.

Registry built and gated by cu_calibration_builder.py — see
2026-07-02-cu-calibration-design.md. Requires only numpy + stdlib.
"""

import json
import os

import numpy as np

_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
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


def load_registry(path=None):
    path = (path or os.environ.get("DILAT_CALIBRATIONS") or _DEFAULT_PATH)
    with open(path, encoding="utf-8") as fh:
        reg = json.load(fh)
    _example_registry_banner(reg, path)
    return reg, path


def registry_cell_fit(cell=None, branch="warm", sample_thickness_cm=None,
                      record_id=None, t_max_needed=None, path=None,
                      verbose=True):
    """Return f(T) -> delta_l_cell [1e-6 cm] from the calibration registry.

    Resolution order:
      1. record_id given -> that exact record.
      2. Eq. 7 (P18) resolved for (cell, branch) and sample_thickness_cm
         given -> virtual Cu curve at the sample thickness.
      3. Closest-Cu-length record for (cell, branch); records not covering
         t_max_needed are deprioritized.
    """
    reg, used_path = load_registry(path)
    recs = {r["id"]: r for r in reg["records"]}

    if record_id is not None:
        r = recs[record_id]
        if verbose:
            print(f"  [calibration_bridge] record {record_id} "
                  f"({r['cell']}, {r['branch']}, Cu {r['cu_length_mm']} mm, "
                  f"T {r['T_fit_min']:.0f}-{r['T_fit_max']:.0f} K) "
                  f"from {os.path.basename(used_path)}")
        return _poly_eval(r["coefficients"])

    if cell is None:
        raise ValueError("give either record_id or cell=")

    e = reg.get("eq7", {}).get(cell, {}).get(branch)
    if (e and isinstance(e, dict) and e.get("resolved")
            and sample_thickness_cm is not None):
        ra = recs[e["records"][0]]
        La = ra["cu_length_mm"] / 10.0
        pa = np.asarray(ra["coefficients"], float)
        D = np.asarray(e["dl2_poly_coefficients"], float)
        if verbose:
            print(f"  [calibration_bridge] eq7 virtual Cu curve at "
                  f"L={sample_thickness_cm} cm from {e['records']} "
                  f"({cell}/{branch})")
        return _poly_eval(pa + D * (sample_thickness_cm - La))

    cands = [r for r in reg["records"]
             if r["cell"] == cell and r["branch"] == branch]
    if not cands:
        raise ValueError(f"no {cell}/{branch} records in {used_path}")

    def rank(r):
        # Tiebreak reconciled with the QC load_calibration rank (review #5):
        # the trailing -span term is only consulted when (covers, dist, 350K,
        # cycle) tie, which does not occur for any current lookup — verified by
        # running both resolvers on the real str_dil/mini_dil lookups. Kept
        # identical so the two implementations can never silently diverge.
        covers = (t_max_needed is None
                  or r["T_fit_max"] >= t_max_needed - 0.5)
        dist = (abs(r["cu_length_mm"] / 10.0 - sample_thickness_cm)
                if sample_thickness_cm is not None else 0.0)
        return (not covers, dist, "350K" in r["source_file"], r["cycle"],
                -(r["T_fit_max"] - r["T_fit_min"]))

    best = sorted(cands, key=rank)[0]
    if verbose:
        print(f"  [calibration_bridge] closest-length record {best['id']} "
              f"(Cu {best['cu_length_mm']} mm) for {cell}/{branch}")
    return _poly_eval(best["coefficients"])
