"""detect.py — cell recognition from a raw dilatometry file (plan P2).

Review #9: this module runs the Pott-Schefzyk plate-constant fit ONLY. It does
NOT segment the run (that stays in reduce.separate_data, called once). It looks
at the recorded (C, delta_l) pairs, recovers the effective plate constant K the
PPMS used when converting capacitance to delta_l, converts K to an effective
plate radius r_eff, and from r_eff proposes which cell produced the file.

Contract (SEAMS, plan): detect(path) -> dict with keys
    r_eff_mm         effective plate radius from the fitted K (float | None)
    K                fitted plate constant [1e-6 cm * pF]      (float | None)
    C_med_pF         median working capacitance                (float | None)
    n_rows           usable (C, delta_l) rows after windowing   (int)
    cell_guess       best single guess for the dropdown default: one of
                     "str_dil" / "mini_dil" / "unknown"
    candidates       human-readable list of the possibilities in play
    rescale_factor   per-file mis-conversion factor (float | None). Populated
                     ONLY in the ambiguous ~7 mm band, where the file could be
                     a mini run the PPMS converted with its 7 mm default; then
                     factor = MINI_DIL plate_K_ref / K_fitted (multiply delta_l
                     by it to undo the mis-conversion). None otherwise.
    confidence       "confident" | "ambiguous" | "unknown"
    note             one-line explanation of the guess for the GUI/log
    dl_p2p_1e6cm     peak-to-peak of raw delta_l over the file [1e-6 cm]; the
                     GUI uses it for the before/after rescale preview
    error            present only on failure (string); other numeric fields None

The r_eff bands come from real fits (see cells.py header): mini true ~4.87 mm,
str / PPMS 7 mm-default ~7.03 mm, a mis-converted mini also ~6.8-7 mm. So r_eff
~7 mm CANNOT by itself separate a genuine str run from a mis-converted mini
(review #1) — the median capacitance is a weak, sample-dependent, NON-decisive
hint (str US ~7.5 pF vs mini ~18.5 pF) used only to pre-select the dropdown.

Detection never blocks reduction: on failure it returns confidence "unknown"
and cell_guess "unknown", and the GUI forces a manual cell choice.

CLI: `python3 detect.py <path>` prints the dict as one JSON line (this is how
dilat_app.py invokes it across the interpreter split; the in-process contract
on a single-interpreter machine is identical).
"""

import json
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

import cells
import plate_constant_audit as ps

# r_eff classification bands [mm].
R_MINI_MAX = 5.9     # below -> confidently the mini cell (correctly converted)
R_AMBIG_MAX = 7.6    # 5.9..7.6 -> str_dil OR a mini mis-converted at 7 mm
# C_med midpoint between the two cells' working windows; a NON-decisive hint
# only used to choose the dropdown default inside the ambiguous band.
C_HINT_SPLIT_PF = 0.5 * (cells.STR_DIL["C_hint_pF"] + cells.MINI_DIL["C_hint_pF"])


def _r_from_K(K):
    # K [1e-6 cm * pF] -> SI (1e-8 m * 1e-12 F = 1e-20 F*m) -> r = sqrt(K/(eps0*pi))
    return float(np.sqrt(K * 1e-20 / (ps.EPS0 * np.pi)) * 1e3)


def _blank(**over):
    d = {"r_eff_mm": None, "K": None, "C_med_pF": None, "n_rows": 0,
         "cell_guess": "unknown", "candidates": ["unknown"],
         "rescale_factor": None, "confidence": "unknown", "note": "",
         "dl_p2p_1e6cm": None}
    d.update(over)
    return d


def detect(path):
    """See module docstring. Pure function of the file; no side effects."""
    kind = "csv" if str(path).lower().endswith(".csv") else "dat"
    try:
        C, dl = ps.load_pairs(path, kind)
    except Exception as e:  # noqa: BLE001 — detection must never raise to caller
        return _blank(note=f"could not read pairs: {e}", error=str(e))

    n_rows = int(len(C))
    if n_rows < 300:
        return _blank(n_rows=n_rows,
                      note=f"too few usable rows ({n_rows}) to fit K",
                      error="insufficient rows")

    K, n_diff = ps.fit_K(C, dl)
    if not np.isfinite(K) or n_diff < 100:
        return _blank(n_rows=n_rows, C_med_pF=round(float(np.median(C)), 3),
                      note="plate-constant fit did not converge",
                      error="fit failed")

    r = _r_from_K(K)
    C_med = float(np.median(C))
    dl_p2p = float(np.nanmax(dl) - np.nanmin(dl))
    base = _blank(r_eff_mm=round(r, 3), K=round(float(K), 1),
                  C_med_pF=round(C_med, 3), n_rows=n_rows,
                  dl_p2p_1e6cm=round(dl_p2p, 3))

    if r < R_MINI_MAX:
        base.update(
            cell_guess="mini_dil", candidates=["mini_dil"],
            confidence="confident", rescale_factor=None,
            note=(f"r_eff = {r:.2f} mm matches the mini cell (~4.87 mm) — "
                  "correctly converted, no rescale."))
    elif r < R_AMBIG_MAX:
        factor = float(cells.MINI_DIL["plate_K_ref"] / K)
        best = ("str_dil" if C_med < C_HINT_SPLIT_PF else "mini_dil")
        hint = (f"C_med = {C_med:.1f} pF leans "
                f"{'str_dil' if best == 'str_dil' else 'mini_dil'} "
                f"(str US ~{cells.STR_DIL['C_hint_pF']:.1f}, "
                f"mini ~{cells.MINI_DIL['C_hint_pF']:.1f} pF) — "
                "sample-dependent, NOT decisive")
        base.update(
            cell_guess=best,
            candidates=["str_dil (genuine standard-cell run)",
                        "mini_dil converted with the PPMS 7 mm default "
                        f"(needs delta_l x {factor:.4f})"],
            confidence="ambiguous", rescale_factor=round(factor, 4),
            note=(f"r_eff = {r:.2f} mm (~7 mm): a genuine str run OR a mini "
                  f"run mis-converted at 7 mm. {hint}."))
    else:
        base.update(
            cell_guess="unknown",
            candidates=["unknown (r_eff outside known cell ranges)"],
            confidence="unknown", rescale_factor=None,
            note=(f"r_eff = {r:.2f} mm is outside both cells' ranges "
                  "(~4.87 mm mini, ~7.0 mm str) — choose the cell manually."))
    return base


def main(argv):
    if len(argv) != 2:
        print(json.dumps(_blank(note="usage: detect.py <path>",
                                error="bad args")))
        return 2
    print(json.dumps(detect(argv[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
