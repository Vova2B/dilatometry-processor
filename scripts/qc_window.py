"""qc_window.py — shared interactive QC window and figure-drawing layer
for qc_str_cell.py and qc_mini_cell.py (single source; previously the same
code was byte-duplicated in both scripts).

Symbol-only curve style per the project figure conventions: cooling =
squares, warming = diamonds; field loops filled = up leg, open = down leg;
markers thinned evenly in T. matplotlib widget imports stay local to the
methods that need them so headless use of the plotting helpers works.
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _style_axes(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)
    ax.tick_params(direction="in", top=True, right=True, length=4)


def _thin_idx(T, dT=1.9):
    """Marker indices spaced evenly in T for symbol-only curves — index-based
    markevery piles markers into black bands wherever sampling is time-dense
    (dwells, slow ramps near the transition). dT fixed in axis units so short
    segments get the same density as full sweeps (~160 markers per 305 K)."""
    T = np.asarray(T, float)
    keep, last = [0], T[0]
    for i in range(1, len(T)):
        if np.isfinite(T[i]) and abs(T[i] - last) >= dT:
            keep.append(i)
            last = T[i]
    return keep


def _draw_curves_T(ax, curves, selected_idx=None, interactive=False,
                   use_raw=False, stride=None, spacing_k=0.0):
    """Draw T-sweep curves on ax. Returns list of Line2D artists (one per curve).

    In interactive mode: selected curve is bold; others are dimmed.
    In export mode (interactive=False): only enabled curves, full opacity.
    use_raw=True draws the pre-cleanup separated+recalculated curve (raw_df),
    bypassing trim/neg-threshold/smoothing.
    stride sets the interactive marker density (markevery); None keeps the
    legacy ~35-marker thinning. Export mode always uses T-spaced _thin_idx.
    """
    artists = []
    for i, c in enumerate(curves):
        is_selected = interactive and (i == selected_idx)
        if not c.enabled and not is_selected:
            artists.append(None)
            continue
        df = c.raw_df if use_raw else c.cleaned()
        if df.empty:
            artists.append(None)
            continue

        ls  = "-" if c.direction == "cool" else "--"
        lw  = 2.5 if is_selected else 1.5
        if not interactive:
            alpha = 1.0
        elif is_selected and not c.enabled:
            alpha = 0.45
        elif is_selected:
            alpha = 1.0
        else:
            alpha = 0.55

        lbl = c.label if c.enabled else f"[off] {c.label}"
        # export: symbol-only journal style (journal convention — black-edged
        # squares cooling / diamonds warming); interactive keeps the line
        line, = ax.plot(df["T PPMS [K]"], df["(del_L/L_0)_Sam"] * 1e3,
                        color=c.color, alpha=alpha, label=lbl,
                        ls=ls if interactive else "none", lw=lw,
                        marker="s" if c.direction == "cool" else "D",
                        ms=5.5 if is_selected else 4,
                        markerfacecolor=c.color, markeredgecolor="k",
                        markeredgewidth=0.6,
                        markevery=(max(1, stride if stride else len(df) // 35)
                                   if interactive
                                   else _thin_idx(df["T PPMS [K]"].values,
                                                  dT=spacing_k) if spacing_k > 0
                                   else _thin_idx(df["T PPMS [K]"].values)),
                        picker=interactive, pickradius=5)
        line._curve_idx = i
        artists.append(line)

    return artists


def _draw_curves_B(ax, curves, selected_idx=None, interactive=False,
                   use_raw=False, stride=None):
    """Draw B-sweep curves on ax. Returns list of Line2D artists.

    use_raw=True draws the pre-cleanup separated+recalculated curve (raw_df).
    stride sets the interactive marker density (markevery); None shows all.
    """
    artists = []
    for i, c in enumerate(curves):
        is_selected = interactive and (i == selected_idx)
        if not c.enabled and not is_selected:
            artists.append(None)
            continue
        df = c.raw_df if use_raw else c.cleaned()
        if df.empty:
            artists.append(None)
            continue

        lw = 2.5 if is_selected else 1.5
        if not interactive:
            alpha = 1.0
        elif is_selected and not c.enabled:
            alpha = 0.45
        elif is_selected:
            alpha = 1.0
        else:
            alpha = 0.55

        lbl = c.label if c.enabled else f"[off] {c.label}"
        line, = ax.plot(df["B [T]"], df["(del_L/L_0)_Sam"] * 1e3,
                        color=c.color, lw=lw, alpha=alpha,
                        marker="o", markersize=2,
                        markevery=(max(1, stride) if (interactive and stride)
                                   else None),
                        label=lbl,
                        picker=interactive, pickradius=5)
        line._curve_idx = i
        artists.append(line)

    return artists


def plot_temperature_dep(curves, angle_deg, out_prefix, use_raw=False,
                         spacing_k=0.0):
    """Publication-quality T-dep plot from a list of Curve objects.
    Saves <out_prefix>_Tdep_clean.{png,csv}, or _Tdep_raw.{png,csv} when
    use_raw=True (pre-cleanup: separated+recalculated, no trim/smoothing).
    """
    suffix = "raw" if use_raw else "clean"
    enabled = [c for c in curves if c.enabled]
    if not enabled:
        print("  No enabled T-curves to plot.")
        return None

    fig, ax = plt.subplots(figsize=(9.5, 6))
    # Reserve the right margin for an OUTSIDE legend so it never sits on data.
    fig.subplots_adjust(left=0.13, right=0.76, top=0.93, bottom=0.13)

    _draw_curves_T(ax, curves, interactive=False, use_raw=use_raw,
                   spacing_k=spacing_k)

    angle_str = f"{angle_deg:+d}" if angle_deg != 0 else "0"
    ax.set_xlabel(r"$T$ (K)")
    ax.set_ylabel(r"$\Delta L / L_0 \;(\times 10^{-3})$")
    if angle_deg != 0:
        ax.text(0.98, 0.975, fr"$\theta = {angle_str}\,°$  (w.r.t. $B$ field)",
                transform=ax.transAxes, fontsize=12,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor="0.7", alpha=0.85))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=8, framealpha=0.9, borderaxespad=0.0)
    _style_axes(ax)
    ax.set_xlim(left=0)          # project convention: T axes start at 0 K

    # CSV — one row per data point, provenance columns included
    csv_rows = []
    for c in enabled:
        src = c.raw_df if use_raw else c.cleaned()
        if spacing_k > 0 and len(src):
            # thin to ~one point per spacing_k Kelvin (figure + CSV in step)
            src = src.iloc[_thin_idx(src["T PPMS [K]"].values, dT=spacing_k)]
        for _, row in src.iterrows():
            csv_rows.append({
                "T_K":         row["T PPMS [K]"],
                "dL_L0":       row["(del_L/L_0)_Sam"],
                "B_T":         c.param_value,
                "direction":   c.direction,
                "mode_index":  c.mode_index,
                "angle_deg":   c.angle_deg,
                "smooth_win":  c.smooth_window,
                "trim_start":  c.trim_start,
                "trim_end":    c.trim_end,
            })

    pd.DataFrame(csv_rows).to_csv(f"{out_prefix}_Tdep_{suffix}.csv", index=False, encoding="utf-8")
    fig.savefig(f"{out_prefix}_Tdep_{suffix}.png", dpi=200, bbox_inches="tight")
    print(f"  Saved: {out_prefix}_Tdep_{suffix}.{{png,csv}}")
    return fig


def plot_field_dep(curves, angle_deg, out_prefix, use_raw=False):
    """Publication-quality B-dep plot from a list of Curve objects.
    Saves <out_prefix>_Bdep_clean.{png,csv}, or _Bdep_raw.{png,csv} when
    use_raw=True (pre-cleanup: separated+recalculated, no trim/smoothing).
    """
    suffix = "raw" if use_raw else "clean"
    enabled = [c for c in curves if c.enabled]
    if not enabled:
        print("  No enabled B-curves to plot.")
        return None

    fig, ax = plt.subplots(figsize=(9.5, 6))
    # Reserve the right margin for an OUTSIDE legend so it never sits on data.
    fig.subplots_adjust(left=0.13, right=0.76, top=0.93, bottom=0.13)

    _draw_curves_B(ax, curves, interactive=False, use_raw=use_raw)

    angle_str = f"{angle_deg:+d}" if angle_deg != 0 else "0"
    ax.set_xlabel(r"$B$ (T)")
    ax.set_ylabel(r"$\Delta L / L_0 \;(\times 10^{-3})$")
    if angle_deg != 0:
        ax.text(0.98, 0.975, fr"$\theta = {angle_str}\,°$  (w.r.t. $B$ field)",
                transform=ax.transAxes, fontsize=12,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor="0.7", alpha=0.85))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=8, framealpha=0.9, borderaxespad=0.0)
    _style_axes(ax)

    csv_rows = []
    for c in enabled:
        src = c.raw_df if use_raw else c.cleaned()
        for _, row in src.iterrows():
            csv_rows.append({
                "B_T":         row["B [T]"],
                "dL_L0":       row["(del_L/L_0)_Sam"],
                "T_K":         c.param_value,
                "direction":   c.direction,
                "mode_index":  c.mode_index,
                "angle_deg":   c.angle_deg,
                "smooth_win":  c.smooth_window,
                "trim_start":  c.trim_start,
                "trim_end":    c.trim_end,
            })

    pd.DataFrame(csv_rows).to_csv(f"{out_prefix}_Bdep_{suffix}.csv", index=False, encoding="utf-8")
    fig.savefig(f"{out_prefix}_Bdep_{suffix}.png", dpi=200, bbox_inches="tight")
    print(f"  Saved: {out_prefix}_Bdep_{suffix}.{{png,csv}}")
    return fig


class QCWindow:
    """Interactive per-curve QC window using matplotlib widgets.

    Usage:
        win = QCWindow(t_curves, kind="T", angle_deg=0, out_prefix="...")
        win.show()   # blocks until user clicks Export or closes window
    """

    # ── layout constants (figure-normalised coordinates) ─────────────────────
    # Grouped right-hand control column (x≈0.67–0.97), top→bottom:
    # SELECT CURVE · VIEW · EDIT SELECTED CURVE · MARKERS · ACTIONS.
    _AX_PREVIEW  = [0.05, 0.10, 0.60, 0.83]
    _AX_SEL      = [0.67, 0.695, 0.30, 0.26]   # RadioButtons / Prev-Next
    _AX_VIEWROW  = [0.67, 0.615, 0.30, 0.05]   # Enabled | Raw | Legend (thirds)
    _AX_TSTART   = [0.67, 0.520, 0.30, 0.035]
    _AX_TEND     = [0.67, 0.462, 0.30, 0.035]
    _AX_SMOOTH   = [0.67, 0.404, 0.30, 0.035]
    _AX_NEGTH    = [0.82, 0.355, 0.15, 0.035]
    _AX_MARKEV   = [0.82, 0.272, 0.15, 0.035]
    _AX_XLIM     = [0.745, 0.198, 0.095, 0.035]  # AXES group: "min:max" boxes
    _AX_YLIM     = [0.875, 0.198, 0.095, 0.035]
    _AX_APPLYALL = [0.67, 0.115, 0.30, 0.042]
    _AX_RESET    = [0.67, 0.068, 0.30, 0.042]
    _AX_EXPORT   = [0.67, 0.018, 0.30, 0.045]
    # section header baselines (fig-normalized y)
    _HDR = {"select": 0.965, "view": 0.665, "edit": 0.585,
            "markers": 0.320, "axes": 0.245, "actions": 0.170}
    _SEL_LIST_H  = 0.215   # fig-fraction height of the radio-list window

    # ── DPI/size-robust font scaling ──────────────────────────────────────────
    # The rectangles above are figure-FRACTION coordinates, so widgets never
    # overlap each other geometrically at any figure size. What broke on other
    # screens/DPI was TEXT: several labels picked up the module-wide 14 pt
    # rcParams["font.size"] instead of a size tuned to their small panel, so a
    # label could overflow its own button/textbox once the figure got smaller
    # (fraction-sized panel shrinks, absolute point-sized font does not).
    # Every font below is expressed as a base size tuned at this reference
    # figure size, then scaled by the actual figure size and floored so text
    # never becomes unreadable.
    _REF_W, _REF_H = 14.0, 7.0
    _FS_FLOOR    = 8      # never render smaller than this, any panel
    _FS_WIDGET   = 9      # Button / TextBox labels
    _FS_HEADER   = 8      # section headers, slider labels, small captions
    _FS_RADIO    = 8      # curve-list radio labels
    _FS_FILENAME = 9      # top-left filename/kind caption
    _FS_PINNED   = 8.5    # "current: <curve>" pinned line
    _FS_HINT     = 7      # "wheel scrolls" hint
    _FS_ANGLE    = 11     # on-plot theta annotation
    _FS_LEGEND   = 8      # curve legend
    _MIN_ROW_PT  = 10.7   # min comfortable row pitch for an 8 pt radio label
                          # (calibrated so the historical cap of 12 rows still
                          # fits exactly at the _REF_W x _REF_H reference size)

    # breathing room from every figure edge (fig fraction): the whole layout
    # is affinely remapped into [_PAD, 1-_PAD] on both axes, so elements no
    # longer sit on the window edges; a pure shrink+shift preserves the
    # relative geometry (and its no-overlap guarantee).
    _PAD = 0.02

    def _pf(self, v):
        """Remap a figure-fraction POSITION into the padded area."""
        return self._PAD + v * (1.0 - 2.0 * self._PAD)

    def _sf(self, v):
        """Scale a figure-fraction SIZE/OFFSET for the padded area."""
        return v * (1.0 - 2.0 * self._PAD)

    def _pad_rect(self, r):
        return [self._pf(r[0]), self._pf(r[1]), self._sf(r[2]), self._sf(r[3])]

    def __init__(self, curves, kind, angle_deg, out_prefix, figsize=(14, 7)):
        cls = type(self)
        for name in ("_AX_PREVIEW", "_AX_SEL", "_AX_VIEWROW", "_AX_TSTART",
                     "_AX_TEND", "_AX_SMOOTH", "_AX_NEGTH", "_AX_MARKEV",
                     "_AX_XLIM", "_AX_YLIM",
                     "_AX_APPLYALL", "_AX_RESET", "_AX_EXPORT"):
            setattr(self, name, self._pad_rect(getattr(cls, name)))
        self._HDR = {k: self._pf(v) for k, v in cls._HDR.items()}
        self._SEL_LIST_H = self._sf(cls._SEL_LIST_H)
        self.curves     = curves      # list[Curve]
        self.kind       = kind        # "T" or "B"
        self.angle_deg  = angle_deg
        self.out_prefix = out_prefix
        self._figsize   = figsize     # (w, h) inches; default matches the
                                       # original hardcoded window size
        self.sel_idx    = 0           # index of selected curve
        self._updating  = False       # guard: suppress callbacks during programmatic updates
        self.exported   = False
        self.show_raw   = False       # RAW toggle: draw pre-cleanup raw_df
        self.show_legend = True       # Legend on/off toggle (VIEW group)
        # Marker density: markevery stride for the interactive preview. Seeded
        # to show ~150 markers on the longest curve (far denser than the old
        # fixed ~35), user-editable via the MARKERS textbox. 1 = every point.
        _npts = max((len(c.raw_df) for c in curves), default=150)
        self.marker_stride = max(1, _npts // 150)

        if not curves:
            return

        self._build_figure()
        self._build_widgets()
        self._full_redraw()
        self._sync_controls()

    # ── figure construction ───────────────────────────────────────────────────

    def _fs(self, base_pt):
        """Scale a base font size (pt, tuned at the _REF_W x _REF_H reference
        figure) to the actual figure size, floored at _FS_FLOOR."""
        return max(self._FS_FLOOR, round(base_pt * self._scale))

    def _compute_sel_K(self, n_labels):
        """How many curves the SELECT CURVE window shows at once.

        The radio list occupies a fixed FRACTION (_SEL_LIST_H) of the figure
        height, so its available height in points shrinks on a smaller
        figure while the (floored) row font size does not shrink below
        _FS_FLOOR. Cap the visible row count so each row keeps >= _MIN_ROW_PT
        of height at the actual figure size. This only ever reduces the
        on-screen count below the historical default of 12 on a
        smaller-than-reference figure — every curve stays reachable via
        scroll — and never raises it above 12, so behavior at the reference
        size (and larger) is unchanged.
        """
        avail_pt = self._SEL_LIST_H * self._figsize[1] * 72.0
        k_cap = max(1, int(avail_pt // self._MIN_ROW_PT))
        return max(1, min(12, n_labels, k_cap))

    def _build_figure(self):
        kind_label = "T-dep (x=T)" if self.kind == "T" else "B-dep (x=B)"
        self._scale = min(self._figsize[0] / self._REF_W,
                          self._figsize[1] / self._REF_H)
        self.fig = plt.figure(figsize=self._figsize)
        fname = os.path.basename(self.out_prefix)
        self.fig.canvas.manager.set_window_title(
            f"QC [{fname}] — {kind_label}  |  θ={self.angle_deg:+d}°")
        self.fig.text(self._pf(0.035), self._pf(0.972),
                      f"{fname} — {kind_label}   θ={self.angle_deg:+d}°",
                      fontsize=self._fs(self._FS_FILENAME), color="0.35",
                      ha="left", va="top")
        self.ax = self.fig.add_axes(self._AX_PREVIEW)
        _style_axes(self.ax)
        self.ax.set_xlabel(r"$T$ (K)" if self.kind == "T" else r"$B$ (T)")
        self.ax.set_ylabel(r"$\Delta L / L_0 \;(\times 10^{-3})$")
        self._artists = []        # Line2D for each curve (main)
        self._raw_art = None      # thin grey raw overlay for selected curve
        self.fig.canvas.mpl_connect("pick_event", self._on_pick)

    def _section_header(self, key, text):
        """Draw a group header + a thin separator rule above it (fig coords)."""
        y = self._HDR[key]
        self.fig.text(self._AX_SEL[0], y, text, fontsize=self._fs(self._FS_HEADER),
                      fontweight="bold", color="0.4", ha="left", va="bottom")
        self.fig.add_artist(plt.Line2D(
            [self._AX_SEL[0], self._AX_SEL[0] + self._AX_SEL[2]],
            [y + self._sf(0.018)] * 2, color="0.8", lw=0.8,
            transform=self.fig.transFigure))

    def _build_widgets(self):
        from matplotlib.widgets import (RadioButtons, Slider, TextBox, Button)
        labels = [c.label for c in self.curves]
        px = self._AX_SEL[0]   # panel left x

        for _k, _t in (("select", "SELECT CURVE"), ("view", "VIEW"),
                       ("edit", "EDIT SELECTED CURVE"),
                       ("markers", "MARKERS — show every Nth point"),
                       ("axes", "AXES — min:max (blank = auto)"),
                       ("actions", "ACTIONS")):
            self._section_header(_k, _t)

        # ── GROUP: SELECT CURVE  (windowed, scrollable radio list) ─────────────
        # Always a radio list; when there are more curves than fit (K), a scroll
        # window shows K at a time with wheel + ▲/▼ + an "i–j / N" counter.
        self._sel_offset = 0
        self._sel_K = self._compute_sel_K(len(labels))
        self._sel_scroll = len(labels) > self._sel_K
        if self._sel_scroll:
            self._sel_counter = self.fig.text(
                px + self._AX_SEL[2], self._HDR["select"], "",
                fontsize=self._fs(self._FS_HEADER), color="0.4",
                ha="right", va="bottom")
            self._sel_pinned = self.fig.text(
                px, self._pf(0.928), "", fontsize=self._fs(self._FS_PINNED),
                color="#1f5fd0", ha="left", va="bottom")
            self.fig.text(px + self._AX_SEL[2], self._pf(0.928),
                          "wheel ⬍ scrolls",
                          fontsize=self._fs(self._FS_HINT), color="0.55",
                          ha="right", va="bottom")
            aw = self._sf(0.045)
            # flush with the top of the radio list, whatever its height
            ax_up = self.fig.add_axes([px + self._AX_SEL[2] - aw,
                                       self._AX_SEL[1] + self._SEL_LIST_H - aw,
                                       aw, aw])
            ax_dn = self.fig.add_axes([px + self._AX_SEL[2] - aw,
                                       self._AX_SEL[1], aw, aw])
            self._up_btn = Button(ax_up, "▲")
            self._dn_btn = Button(ax_dn, "▼")
            self._up_btn.on_clicked(lambda _e: self._scroll_by(-1))
            self._dn_btn.on_clicked(lambda _e: self._scroll_by(+1))
            for _b in (self._up_btn, self._dn_btn):
                _b.label.set_fontsize(self._fs(self._FS_WIDGET))
            sr = [px, self._AX_SEL[1], self._AX_SEL[2] - self._sf(0.05),
                  self._SEL_LIST_H]
        else:
            self._sel_counter = self._sel_pinned = None
            self._up_btn = self._dn_btn = None
            sr = list(self._AX_SEL)
        self._ax_sel = self.fig.add_axes(sr)
        self._render_select_window()
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

        # ── GROUP: VIEW  (Enabled | Cleaned/Raw | Legend) ──────────────────────
        x0, y0, w, h = self._AX_VIEWROW
        bw, gap = w * 0.31, w * 0.035
        ax_en = self.fig.add_axes([x0, y0, bw, h])
        self._w_enabled = Button(ax_en, "Curve: on", color="#c8e6c9")
        self._w_enabled.on_clicked(self._on_enabled)
        ax_raw = self.fig.add_axes([x0 + bw + gap, y0, bw, h])
        self._w_raw = Button(ax_raw, "View: clean", color="#e0e0e0")
        self._w_raw.on_clicked(self._on_raw_toggle)
        ax_leg = self.fig.add_axes([x0 + 2 * (bw + gap), y0, bw, h])
        self._w_legend = Button(ax_leg, "Legend: on", color="#c8e6c9")
        self._w_legend.on_clicked(self._on_legend_toggle)
        for _b in (self._w_enabled, self._w_raw, self._w_legend):
            _b.label.set_fontsize(self._fs(self._FS_WIDGET))

        # ── GROUP: EDIT SELECTED CURVE ─────────────────────────────────────────
        n   = len(self.curves[0].raw_df)
        cap = max(1, n - 1)
        ax_ts = self.fig.add_axes(self._AX_TSTART)
        self._w_tstart = Slider(ax_ts, "Trim start (idx)", 0, cap,
                                valinit=0, valstep=1, valfmt="%d")
        self._w_tstart.on_changed(self._on_trim_start)
        ax_te = self.fig.add_axes(self._AX_TEND)
        self._w_tend = Slider(ax_te, "Trim end (idx)", 0, cap,
                              valinit=0, valstep=1, valfmt="%d")
        self._w_tend.on_changed(self._on_trim_end)
        ax_sm = self.fig.add_axes(self._AX_SMOOTH)
        self._w_smooth = Slider(ax_sm, "Smooth window (pts)", 0, 51,
                                valinit=0, valstep=2, valfmt="%d")
        self._w_smooth.on_changed(self._on_smooth)
        # Slider name labels default to the LEFT of the track (x=-0.02,
        # ha='right'); at panel x=0.67 that crosses the plot's right spine
        # (~x=0.65). Move each label ABOVE its slider, inside the control column.
        for _sl in (self._w_tstart, self._w_tend, self._w_smooth):
            _sl.label.set_position((0.0, 1.0))
            _sl.label.set_horizontalalignment("left")
            _sl.label.set_verticalalignment("bottom")
            _sl.label.set_fontsize(self._fs(self._FS_HEADER))
            _sl.valtext.set_fontsize(self._fs(self._FS_WIDGET))

        # neg-threshold textbox (label to its left, inside the panel)
        nx, ny, nw, nh = self._AX_NEGTH
        axnl = self.fig.add_axes([px, ny, nx - px - 0.01, nh]); axnl.axis("off")
        axnl.text(1.0, 0.5, "Drop δl below [1e-6 cm]:", ha="right", va="center",
                  transform=axnl.transAxes, fontsize=self._fs(self._FS_HEADER))
        ax_neg = self.fig.add_axes(self._AX_NEGTH)
        self._w_negth = TextBox(ax_neg, "", initial="")
        self._w_negth.on_submit(self._on_neg_thresh)
        self._w_negth.text_disp.set_fontsize(self._fs(self._FS_WIDGET))

        # ── GROUP: MARKERS  (display density) ──────────────────────────────────
        mx, my, mw, mh = self._AX_MARKEV
        axml = self.fig.add_axes([px, my, mx - px - 0.01, mh]); axml.axis("off")
        axml.text(1.0, 0.5, "N =", ha="right", va="center",
                  transform=axml.transAxes, fontsize=self._fs(self._FS_HEADER))
        ax_mk = self.fig.add_axes(self._AX_MARKEV)
        self._w_markev = TextBox(ax_mk, "", initial=str(self.marker_stride))
        self._w_markev.on_submit(self._on_markevery)
        self._w_markev.text_disp.set_fontsize(self._fs(self._FS_WIDGET))

        # ── GROUP: AXES  (view limits, "min:max", blank side = auto) ──────────
        # T-dependences default to a 0 K axis start (project convention).
        self._w_axlim = {}
        for axis, rect, init in (("x", self._AX_XLIM,
                                  "0:" if self.kind == "T" else ""),
                                 ("y", self._AX_YLIM, "")):
            lx = rect[0] - self._sf(0.035)
            axl = self.fig.add_axes([lx, rect[1], rect[0] - lx - 0.005,
                                     rect[3]])
            axl.axis("off")
            axl.text(1.0, 0.5, f"{axis.upper()}:", ha="right", va="center",
                     transform=axl.transAxes,
                     fontsize=self._fs(self._FS_HEADER))
            box = TextBox(self.fig.add_axes(rect), "", initial=init)
            box.on_submit(lambda _s, a=axis: self._on_axis_limits(a))
            box.text_disp.set_fontsize(self._fs(self._FS_WIDGET))
            self._w_axlim[axis] = box

        # ── GROUP: ACTIONS ─────────────────────────────────────────────────────
        ax_app = self.fig.add_axes(self._AX_APPLYALL)
        self._w_applyall = Button(ax_app, "Apply to all (same B/T)")
        self._w_applyall.on_clicked(self._on_apply_all)
        ax_res = self.fig.add_axes(self._AX_RESET)
        self._w_reset = Button(ax_res, "Reset this curve")
        self._w_reset.on_clicked(self._on_reset)
        ax_exp = self.fig.add_axes(self._AX_EXPORT)
        self._w_export = Button(ax_exp, "EXPORT", color="#c8e6c9")
        self._w_export.on_clicked(self._on_export)
        for _b in (self._w_applyall, self._w_reset, self._w_export):
            _b.label.set_fontsize(self._fs(self._FS_WIDGET))

    # ── drawing ───────────────────────────────────────────────────────────────

    def _full_redraw(self):
        """Redraw all curves."""
        self.ax.cla()
        self._raw_art = None   # ax.cla() detached it; reset reference
        _style_axes(self.ax)
        self.ax.set_xlabel(r"$T$ (K)" if self.kind == "T" else r"$B$ (T)")
        self.ax.set_ylabel(r"$\Delta L / L_0 \;(\times 10^{-3})$")
        if self.kind == "T":
            self._artists = _draw_curves_T(
                self.ax, self.curves,
                selected_idx=self.sel_idx, interactive=True,
                use_raw=self.show_raw, stride=self.marker_stride)
        else:
            self._artists = _draw_curves_B(
                self.ax, self.curves,
                selected_idx=self.sel_idx, interactive=True,
                use_raw=self.show_raw, stride=self.marker_stride)
        # in RAW mode the main lines already are raw_df — skip the grey overlay
        if not self.show_raw:
            self._draw_raw_overlay()
        if self.angle_deg != 0:
            angle_str = f"{self.angle_deg:+d}"
            self.ax.text(0.98, 0.975,
                         fr"$\theta = {angle_str}\,°$  (w.r.t. $B$ field)",
                         transform=self.ax.transAxes, fontsize=self._fs(self._FS_ANGLE),
                         ha="right", va="top",
                         bbox=dict(boxstyle="round,pad=0.25",
                                   facecolor="white", edgecolor="0.7", alpha=0.85))
        # Legend: upper-left (empty corner for a T-rise) and DRAGGABLE; the
        # control panel occupies the right so it can't sit outside the axes.
        # Hidden when the VIEW "Legend" toggle is off.
        if self.show_legend:
            leg = self.ax.legend(loc="upper left", ncol=2,
                                 fontsize=self._fs(self._FS_LEGEND),
                                 framealpha=0.9, columnspacing=1.0,
                                 handletextpad=0.4)
            if leg is not None:
                leg.set_draggable(True)
        self._apply_axis_limits()
        self.fig.canvas.draw_idle()

    @staticmethod
    def _parse_lim(s):
        """'min:max' -> (lo, hi); blank side -> None (auto); bad text -> None."""
        s = (s or "").strip()
        if not s:
            return None
        lo, _, hi = s.partition(":")
        try:
            lo_v = float(lo) if lo.strip() else None
            hi_v = float(hi) if hi.strip() else None
        except ValueError:
            return None
        if lo_v is None and hi_v is None:
            return None
        return (lo_v, hi_v)

    def _apply_axis_limits(self):
        for axis, box in getattr(self, "_w_axlim", {}).items():
            lim = self._parse_lim(box.text)
            if lim is None:
                continue
            if axis == "x":
                self.ax.set_xlim(left=lim[0], right=lim[1])
            else:
                self.ax.set_ylim(bottom=lim[0], top=lim[1])

    def _on_axis_limits(self, _axis):
        if self._updating:
            return
        # blanking a box must return that axis to autoscale: recompute the
        # auto limits first, then re-impose whatever boxes still hold
        self.ax.relim()
        self.ax.autoscale()
        self._apply_axis_limits()
        self.fig.canvas.draw_idle()

    def _draw_raw_overlay(self):
        """Grey thin line showing raw (unprocessed) data for selected curve."""
        if self._raw_art is not None:
            try:
                self._raw_art.remove()
            except (ValueError, NotImplementedError):
                try:
                    self.ax.lines.remove(self._raw_art)
                except ValueError:
                    pass
            self._raw_art = None
        c = self.curves[self.sel_idx]
        raw = c.raw_df
        x = raw["T PPMS [K]"] if self.kind == "T" else raw["B [T]"]
        y = raw["(del_L/L_0)_Sam"] * 1e3
        mask = np.isfinite(y)
        self._raw_art, = self.ax.plot(
            x[mask], y[mask], color="grey", lw=0.7, alpha=0.4, zorder=0)

    def _redraw_active(self):
        """Lightweight redraw: only update the selected curve's artist."""
        i = self.sel_idx
        c = self.curves[i]
        art = self._artists[i] if i < len(self._artists) else None
        if art is None:
            self._full_redraw()
            return
        df = c.cleaned()
        if df.empty:
            art.set_data([], [])
        else:
            x = df["T PPMS [K]"] if self.kind == "T" else df["B [T]"]
            art.set_data(x, df["(del_L/L_0)_Sam"] * 1e3)
        self._draw_raw_overlay()
        self.fig.canvas.draw_idle()

    # ── curve selection ───────────────────────────────────────────────────────

    def _select(self, idx):
        idx = max(0, min(idx, len(self.curves) - 1))
        self.sel_idx = idx
        # keep the selection visible in the scroll window (e.g. when picking a
        # curve on the plot that is currently scrolled out of the radio list)
        K = getattr(self, "_sel_K", None)
        if K:
            if idx < self._sel_offset:
                self._sel_offset = idx
            elif idx >= self._sel_offset + K:
                self._sel_offset = idx - K + 1
            self._render_select_window()   # rebuilds radio w/ correct highlight
        self._full_redraw()
        self._sync_controls()

    # ── windowed / scrollable curve selector ──────────────────────────────────

    def _render_select_window(self):
        """(Re)build the windowed RadioButtons for the current scroll offset."""
        from matplotlib.widgets import RadioButtons
        N, K = len(self.curves), self._sel_K
        self._sel_offset = max(0, min(self._sel_offset, max(0, N - K)))
        off = self._sel_offset
        self._ax_sel.clear()
        self._ax_sel.set_axis_off()
        win = [c.label for c in self.curves[off:off + K]]
        local = self.sel_idx - off
        in_win = 0 <= local < len(win)
        self._w_sel = RadioButtons(self._ax_sel, win,
                                   active=local if in_win else 0)
        for t in self._w_sel.labels:
            t.set_fontsize(self._fs(self._FS_RADIO))
        self._w_sel.on_clicked(self._on_select_radio)
        if not in_win and hasattr(self._w_sel, "_buttons"):
            # show no fill without EMPTYING the facecolor array (an empty array
            # breaks the widget's own set_active on the next click).
            # `_buttons` is private matplotlib API (3.7+); on any mismatch the
            # except degrades gracefully — the pinned "current" line stays the
            # authoritative selection indicator.
            try:
                self._w_sel._buttons.set_facecolor(["none"] * len(win))
            except Exception:
                pass
        if self._sel_counter is not None:
            self._sel_counter.set_text(f"{off + 1}–{min(off + K, N)} / {N}")
        if self._sel_pinned is not None:
            self._sel_pinned.set_text(
                f"▶ current:  {self.curves[self.sel_idx].label}")
        if self._up_btn is not None:
            self._set_btn_enabled(self._up_btn, off > 0)
            self._set_btn_enabled(self._dn_btn, off < N - K)

    def _scroll_by(self, delta):
        N, K = len(self.curves), self._sel_K
        if N <= K:
            return
        new = max(0, min(self._sel_offset + delta, N - K))
        if new != self._sel_offset:
            self._sel_offset = new
            self._render_select_window()
            self.fig.canvas.draw_idle()

    def _on_scroll(self, event):
        if getattr(self, "_ax_sel", None) is None or event.inaxes is not self._ax_sel:
            return
        self._scroll_by(-3 if event.button == "up" else 3)

    @staticmethod
    def _set_btn_enabled(btn, enabled):
        btn.ax.set_facecolor("0.88" if enabled else "0.96")
        btn.label.set_color("0.15" if enabled else "0.7")

    def _on_select_radio(self, label):
        if self._updating:
            return
        off, K = self._sel_offset, self._sel_K
        win = [c.label for c in self.curves[off:off + K]]
        if label in win:
            self._select(off + win.index(label))

    def _on_pick(self, event):
        art = event.artist
        if hasattr(art, "_curve_idx"):
            self._select(art._curve_idx)

    # ── control sync ──────────────────────────────────────────────────────────

    def _sync_controls(self):
        """Update all widgets to reflect the selected curve's current state."""
        self._updating = True
        c = self.curves[self.sel_idx]
        n   = len(c.raw_df)
        cap = max(1, n - 1)

        # enabled toggle button
        if c.enabled:
            self._w_enabled.label.set_text("Curve: on")
            self._w_enabled.ax.set_facecolor("#c8e6c9")
        else:
            self._w_enabled.label.set_text("Curve: OFF")
            self._w_enabled.ax.set_facecolor("#ffcdd2")

        # update trim slider maxima to match this curve's length
        self._w_tstart.valmax = cap
        self._w_tstart.ax.set_xlim(0, cap)
        self._w_tstart.set_val(c.trim_start)

        self._w_tend.valmax = cap
        self._w_tend.ax.set_xlim(0, cap)
        self._w_tend.set_val(c.trim_end)

        self._w_smooth.set_val(c.smooth_window)

        thresh_str = "" if c.neg_threshold is None else str(c.neg_threshold)
        self._w_negth.set_val(thresh_str)

        # the radio selector (window + highlight + pinned line + counter) is
        # rebuilt by _render_select_window whenever selection or offset changes.

        self._updating = False

    # ── widget callbacks ──────────────────────────────────────────────────────

    def _on_enabled(self, event):
        if self._updating:
            return
        c = self.curves[self.sel_idx]
        c.enabled = not c.enabled
        c._cache_key = None
        if c.enabled:
            self._w_enabled.label.set_text("Curve: on")
            self._w_enabled.ax.set_facecolor("#c8e6c9")
        else:
            self._w_enabled.label.set_text("Curve: OFF")
            self._w_enabled.ax.set_facecolor("#ffcdd2")
        self._full_redraw()

    def _on_raw_toggle(self, event):
        """Flip every curve between cleaned() and pre-cleanup raw_df."""
        self.show_raw = not self.show_raw
        if self.show_raw:
            self._w_raw.label.set_text("View: RAW")
            self._w_raw.ax.set_facecolor("#ffe0b2")
        else:
            self._w_raw.label.set_text("View: clean")
            self._w_raw.ax.set_facecolor("#e0e0e0")
        self._full_redraw()

    def _on_legend_toggle(self, event):
        """Show/hide the on-plot (draggable) legend."""
        self.show_legend = not self.show_legend
        if self.show_legend:
            self._w_legend.label.set_text("Legend: on")
            self._w_legend.ax.set_facecolor("#c8e6c9")
        else:
            self._w_legend.label.set_text("Legend: off")
            self._w_legend.ax.set_facecolor("#e0e0e0")
        self._full_redraw()

    def _on_markevery(self, text):
        """MARKERS textbox: set interactive marker density (markevery stride).
        N=1 shows every point; larger N thins. Invalid input reverts."""
        try:
            n = int(float(text))
            if n < 1:
                raise ValueError
        except (TypeError, ValueError):
            self._w_markev.set_val(str(self.marker_stride))  # revert
            return
        self.marker_stride = n
        self._full_redraw()

    def _on_trim_start(self, val):
        if self._updating:
            return
        c = self.curves[self.sel_idx]
        c.trim_start = int(val)
        c._cache_key = None
        self._redraw_active()

    def _on_trim_end(self, val):
        if self._updating:
            return
        c = self.curves[self.sel_idx]
        c.trim_end = int(val)
        c._cache_key = None
        self._redraw_active()

    def _on_smooth(self, val):
        if self._updating:
            return
        c = self.curves[self.sel_idx]
        c.smooth_window = int(val)
        c._cache_key = None
        self._redraw_active()

    def _on_neg_thresh(self, text):
        if self._updating:
            return
        c = self.curves[self.sel_idx]
        text = text.strip()
        if text == "":
            c.neg_threshold = None
        else:
            try:
                c.neg_threshold = float(text)
                self._w_negth.color = "white"
            except ValueError:
                self._w_negth.color = "#ffcccc"
                return
        c._cache_key = None
        self._redraw_active()

    def _on_apply_all(self, event):
        if self._updating:
            return
        src = self.curves[self.sel_idx]
        for c in self.curves:
            if c is not src and abs(c.param_value - src.param_value) < 0.15:
                c.trim_start    = src.trim_start
                c.trim_end      = src.trim_end
                c.smooth_window = src.smooth_window
                c.neg_threshold = src.neg_threshold
                c._cache_key    = None
        self._full_redraw()

    def _on_reset(self, event):
        if self._updating:
            return
        self.curves[self.sel_idx].reset()
        self._full_redraw()
        self._sync_controls()

    def _on_export(self, event):
        if self.exported:
            return
        self.exported = True
        self._w_export.label.set_text("Saving…")
        self._w_export.ax.set_facecolor("#fff59d")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        try:
            if self.kind == "T":
                plot_temperature_dep(self.curves, self.angle_deg, self.out_prefix)
                tag = "Tdep"
            else:
                plot_field_dep(self.curves, self.angle_deg, self.out_prefix)
                tag = "Bdep"
            print(f"  [EXPORT] Saved {self.out_prefix}_{tag}_clean.{{png,csv}}")
            self._w_export.label.set_text("Saved ✓ — click to close")
            self._w_export.ax.set_facecolor("#a5d6a7")
        except Exception as e:
            print(f"  [EXPORT] FAILED: {e}")
            self._w_export.label.set_text("Save failed — see console")
            self._w_export.ax.set_facecolor("#ef9a9a")
            self.exported = False
            self.fig.canvas.draw_idle()
            return
        self._w_export.disconnect_events()
        self._w_export.on_clicked(lambda evt: plt.close(self.fig))
        self.fig.canvas.draw_idle()

    # ── public API ────────────────────────────────────────────────────────────

    def show(self):
        """Open the QC window and block until Export is clicked or closed."""
        if not self.curves:
            print("  No curves — skipping QC window.")
            return
        plt.show()


def save_qc_state(t_curves, b_curves, path):
    state = {
        "T": [c.to_state_dict() for c in t_curves],
        "B": [c.to_state_dict() for c in b_curves],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    print(f"  QC state saved: {path}")


def load_qc_state(t_curves, b_curves, path):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        state = json.load(f)
    label_to_curve = {c.label: c for c in t_curves + b_curves}
    for d in state.get("T", []) + state.get("B", []):
        c = label_to_curve.get(d["label"])
        if c is not None:
            c.apply_state_dict(d)
    print(f"  QC state loaded: {path}")
