"""transition.py — the sample transition used to label figures and split the
magnetostriction panels. Generalises the old hard-coded ferromagnetic T_C to
any of T_C / T_N / T_CDW, or none (no line, single panel).

Pure module: no matplotlib, no I/O. find_transition (assisted detection)
depends only on numpy/scipy.
"""

import numpy as np
from scipy.signal import find_peaks

KINDS = ("T_C", "T_N", "T_CDW", "none")

# LaTeX symbols (without $...$) per kind.
_SYMBOL = {"T_C": "T_C", "T_N": "T_N", "T_CDW": "T_{CDW}"}


class Transition:
    def __init__(self, value, kind):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.value = None if value is None else float(value)
        self.kind = kind

    @property
    def active(self):
        return self.value is not None and self.kind != "none"

    @property
    def split_active(self):
        # two magnetostriction panels only make sense with a real transition
        return self.active

    @property
    def symbol(self):
        return _SYMBOL.get(self.kind)

    @property
    def label(self):
        s = self.symbol
        return None if s is None else f"${s}$"

    def axis_text(self):
        return self.label if self.active else None

    def panel_titles(self):
        if self.kind == "T_C":
            return (r"$T < T_C$ (ferromagnetic)", r"$T \geq T_C$ (paramagnetic)")
        s = self.symbol
        return (rf"$T < {s}$", rf"$T \geq {s}$")

    def __repr__(self):
        return f"Transition(value={self.value}, kind={self.kind!r})"


def _alpha_curve(T, dLL, bin_k=3.0):
    """Bin (T, dLL) into bin_k-wide T bins (median), then alpha = d(dLL)/dT by
    central differences. Returns (T_centres, alpha) with NaNs dropped."""
    T = np.asarray(T, float)
    y = np.asarray(dLL, float)
    good = np.isfinite(T) & np.isfinite(y)
    T, y = T[good], y[good]
    if T.size < 30:
        return np.array([]), np.array([])
    lo, hi = np.nanmin(T), np.nanmax(T)
    edges = np.arange(lo, hi + bin_k, bin_k)
    idx = np.clip(np.digitize(T, edges) - 1, 0, len(edges) - 2)
    tb, ab = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= 3:
            tb.append(0.5 * (edges[b] + edges[b + 1]))
            ab.append(np.median(y[m]))
    tb, yb = np.asarray(tb), np.asarray(ab)
    if tb.size < 5:
        return np.array([]), np.array([])
    alpha = np.gradient(yb, tb)          # 1/K; ~1e-5 scale -> *1e6 below
    return tb, alpha * 1e6               # units 1e-6/K


def _rank_peaks(sig, tb, min_prominence, max_candidates, edge):
    """Rank prominence-filtered peaks of the anomaly signal. Peaks within
    `edge` bins of either end are trimmed: the zero-padded boxcar baseline is
    biased there, so a candidate in that zone is untrustworthy by
    construction. A peak must clear BOTH min_prominence (absolute floor,
    kills numerical noise on flat data) and 6x the robust noise of sig
    (relative floor — makes detection scale-aware, so a signal in arbitrary
    units, e.g. raw delta-l, does not produce huge-prominence garbage)."""
    noise = 1.4826 * np.median(np.abs(sig - np.median(sig)))
    floor = max(min_prominence, 6.0 * noise)
    peaks, props = find_peaks(sig, prominence=floor)
    keep = (peaks >= edge) & (peaks < len(sig) - edge)
    peaks, proms = peaks[keep], props["prominences"][keep]
    if peaks.size == 0:
        return []
    order = np.argsort(proms)[::-1][:max_candidates]
    out = []
    for j in order:
        p = int(peaks[j])
        out.append({"T": round(float(tb[p]), 2),
                    "prominence": round(float(proms[j]), 3),
                    "alpha_jump": round(float(sig[p]), 3)})
    return out


def find_transition(T, dLL, *, bin_k=3.0, min_prominence=0.15,
                    max_candidates=3):
    """Assisted transition detection. Returns ranked candidates or [] when the
    alpha curve has no peak clearing min_prominence (flat data -> 'none').
    Never raises; on any failure returns []."""
    try:
        tb, alpha = _alpha_curve(T, dLL, bin_k)
        if tb.size < 5:
            return []
        # anomaly signal = |alpha - smooth baseline|; a transition is a peak.
        # The baseline window must be WIDER than a lambda anomaly (~10-15 K),
        # else the anomaly is absorbed into the baseline and splits into a
        # weak double peak (seen on the US T_C = 177 K reference data).
        k = max(3, int(round(30.0 / bin_k)) | 1)         # odd window ~30 K
        base = np.convolve(alpha, np.ones(k) / k, mode="same")
        sig = np.abs(alpha - base)
        return _rank_peaks(sig, tb, min_prominence, max_candidates, k // 2)
    except Exception:                                    # detection never blocks
        return []


def _main(argv):
    import json
    import pandas as pd
    if len(argv) != 2:
        print(json.dumps({"error": "usage: transition.py <archive.csv|.dat>"}))
        return 2
    # reuse the reducer's loader path via a light read: expect T + dLL columns
    path = argv[1]
    sep = "," if path.lower().endswith(".csv") else "\t"
    skip = 0 if path.lower().endswith(".csv") else 1
    df = pd.read_csv(path, sep=sep, skiprows=skip, comment="#",
                     encoding="utf-8", encoding_errors="replace")
    # keep only the virgin (pre-field) segment: field loops and post-field
    # remanence offsets otherwise swamp the thermal-expansion anomaly
    if "B [T]" in df.columns:
        b = pd.to_numeric(df["B [T]"], errors="coerce").fillna(0.0).abs()
        in_field = np.flatnonzero(b.values > 0.1)
        if in_field.size:
            df = df.iloc[:in_field[0]]
    tcol = next((c for c in df.columns if "T PPMS" in c or c.strip() == "T [K]"),
                None)
    # detect on ONE branch: T-binned medians over mixed cool+warm rows flip
    # between branches (hysteresis/drift offsets) and fake alpha steps, so
    # keep only the longest monotonic-T stretch (dwells inherit direction)
    if tcol is not None and len(df) > 100:
        Ts = pd.Series(df[tcol].values).rolling(
            51, center=True, min_periods=1).median().values
        d = np.diff(Ts)
        sgn = np.where(d > 0.002, 1, np.where(d < -0.002, -1, 0))
        for i in range(1, sgn.size):
            if sgn[i] == 0:
                sgn[i] = sgn[i - 1]
        best_len, best_lo, lo = 0, 0, 0
        for i in range(1, sgn.size + 1):
            if i == sgn.size or (sgn[i] != sgn[lo] and sgn[i] != 0):
                if i - lo > best_len:
                    best_len, best_lo = i - lo, lo
                lo = i
        df = df.iloc[best_lo:best_lo + best_len + 1]
    # prefer the reducer's calibrated sample signal when present; a raw .dat
    # has neither this column nor "del_L/L_0", so behavior there is unchanged.
    ycol = (next((c for c in df.columns if "(del_L/L_0)_Sam" in c), None)
            or next((c for c in df.columns if "del_L/L_0" in c), None)
            or next((c for c in df.columns if "delta l" in c), None))
    if tcol is None or ycol is None:
        print(json.dumps({"error": "no T / dLL column found",
                          "columns": list(df.columns)}))
        return 1
    out = {"candidates": find_transition(df[tcol].values, df[ycol].values)}
    if "delta l" in ycol:
        out["note"] = ("scanned RAW delta-l (cell background NOT removed) — "
                       "candidates are low-confidence; anomalies of the cell "
                       "itself can appear. Reduce first and re-scan the "
                       "reduced archive for a reliable result.")
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main(sys.argv))
