"""
dilat_app.py — unified dilatometry GUI (plan docs/plans/2026-07-02-unified-dilat-gui.md).

P0 (walking skeleton, DONE): file picker -> detect -> label -> Run subprocess.
P2 (DONE): detection wired to detect.py (Pott-Schefzyk fit_K, review #9 — no
  segmentation); a user-confirmable CELL dropdown; a GATED, default-OFF rescale
  SUGGESTION for a mini run the PPMS mis-converted at its 7 mm default.
P3 (DONE):
  * a Sample & parameters panel driven by samples.json (samples.py loader):
    sample dropdown, editable L0 (prefilled from the selection), angle, and
    per-cell cleanup toggles whose DEFAULTS flip when the cell changes;
  * a multi-run RESULTS list built AFTER a headless Run from the provenance
    JSONs, with an "Open QC" button per row that launches the interactive
    script for THAT run as a subprocess ON DEMAND (review #10 — never a chain
    of auto-opened windows);
  * the P2 rescale offer moved to its own full-width row so its long label +
    factor text no longer clip at the panel edge.

UI pass (cosmetic — no science/behavior change): ttk theme picked at runtime
  (aqua on macOS / vista on Windows / clam fallback), one spacing constant,
  labeled section frames, grid weights so the Results and Log panes grow on
  resize and nothing clips at any width >= the minimum, a monospaced/badged
  detection panel, a busy Progressbar + status during a Run, a Treeview results
  list with PASS/SUSPECT-tagged rows (Open QC via button or double-click), and
  keyboard basics (Return in the path box = re-detect, Cmd/Ctrl-O = browse).

Wiring honesty (review — "never let a toggle silently do nothing"): the cleanup
toggles are LIVE — every Run passes each one to the batch driver as an explicit
--<toggle>/--no-<toggle> flag (P5 landed 2026-07-09). Selecting a cell resets
them to its safe per-cell defaults (mini: stitching OFF). L0 is honored on both
the str Run (--L0) and the Open-QC path (--thickness). The mini rescale offer
is LIVE too (2026-07-09): when checked, Run passes --rescale-file/
--rescale-factor; a rescale already set in angle_runs.json wins with a warning.
The angle box is disabled — no driver consumes it (mini angles come from
angle_runs.json), and a live dead control was the audited failure mode.

Design notes:
  * This GUI is stdlib-only (tkinter) so it can run under stock Python on
    Windows. cells.py + samples.py are pure-stdlib data/loader, imported here
    directly; the science stack (numpy/pandas/scipy) may live in a DIFFERENT
    interpreter, so detection (detect.py) and the workers run as SUBPROCESSES
    against the science interpreter. On a single-interpreter machine
    sys.executable carries both and the same subprocess seam applies.
  * Detection is SUGGESTION-only. Nothing — least of all the rescale — is ever
    auto-applied (review #1).
"""

import glob
import json
import os
import queue
import subprocess
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

import threading
import time
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk, scrolledtext

import cells     # pure-stdlib parameter data (safe under the GUI interpreter)
import samples   # pure-stdlib samples.json loader

HERE = os.path.dirname(os.path.abspath(__file__))          # scripts/
PROC = os.path.dirname(HERE)                               # Dilatometry processor/

# Optional convenience: a file the app pre-selects on startup when it
# exists (leave pointing at a non-existent path to boot blank + Browse).
HARDCODED_FILE = os.path.join(PROC, "Data", "your_run.dat")

# Batch reduction worker per confirmed cell (both headless, known-good).
REDUCE_WORKER = {
    "str_dil":  os.path.join(HERE, "reduce_str_batch.py"),
    "mini_dil": os.path.join(HERE, "reduce_mini_batch.py"),
}
DETECT_WORKER = os.path.join(HERE, "detect.py")
TRANSITION_WORKER = os.path.join(HERE, "transition.py")
# Mirrors transition.KINDS. NOT imported directly: transition.py needs
# numpy/scipy (find_peaks), which may not be present in this stdlib-only GUI
# interpreter — same cross-interpreter subprocess seam as detect.py above.
# "none" is intentionally reordered to the FRONT (transition.KINDS ends with
# it) so it reads as the default first item in the dropdown — not a typo.
TRANSITION_KINDS = ("none", "T_C", "T_N", "T_CDW")

# interactive QC script per cell (launched ON DEMAND from a results row).
QC_SCRIPT = {
    "str_dil":  os.path.join(HERE, "qc_str_cell.py"),
    "mini_dil": os.path.join(HERE, "qc_mini_cell.py"),
}

# Where each cell's batch driver writes its provenance JSON(s) (see the drivers).
# Both cells are discovered by glob so input-named runs (the reducer derives the
# output stem from the picked file) all show up, not just the default US run.
STR_PROV_GLOB = os.path.join(PROC, "Output", "str", "*_provenance.json")
MINI_PROV_GLOB = os.path.join(PROC, "Output", "mini", "*_provenance.json")
PROVENANCE_GLOB = {"str_dil": STR_PROV_GLOB, "mini_dil": MINI_PROV_GLOB}
DATA_DIR = os.path.join(PROC, "Data")   # where the QC scripts look for raw .dat files

# Cell dropdown values, from cells.py (str first so it is the neutral default).
CELL_CHOICES = [cells.STR_DIL["registry_key"], cells.MINI_DIL["registry_key"]]

# Cleanup toggles (review scope): stitch T-curves / B-loops default ON for str,
# OFF for mini; dwell removal + smoothing outside the protect window default ON.
CLEANUP_TOGGLES = [
    ("stitch_t_curves",        "stitch T-curves"),
    ("stitch_b_loops",         "stitch B-loops"),
    ("dwell_removal",          "dwell removal"),
    ("smooth_outside_protect", "smooth (outside protect)"),
]

SCIENCE_CANDIDATES = [
    sys.executable, "/opt/homebrew/bin/python3", "python3", "python",
    "py",   # Windows launcher — stock python.org installs ship no python3.exe
]

# Windows: a GUI launched via pythonw.exe otherwise flashes a console window
# for every worker subprocess (detect / find-transition / Run / QC / open).
POPEN_NOWINDOW = ({"creationflags": subprocess.CREATE_NO_WINDOW}
                  if os.name == "nt" else {})

# ── One spacing constant + a small palette (secondary text / status badges) ──
PAD = 8
BADGE = {
    "confident": ("#1b5e20", "#dcefdc"),
    "ambiguous": ("#7a5a00", "#fbedc4"),
    "unknown":   ("#7a1f1f", "#f6d9d9"),
}
SECONDARY = "#666666"
MUTED = "#8a8a8a"
# Results-tree row colors. Foregrounds are set EXPLICITLY with each
# background: without them a dark-mode system supplies LIGHT text on these
# light pastels — unreadable (live smoke-test feedback 2026-07-09).
TREE_PASS_BG = "#e3f1e3"
TREE_PASS_FG = "#1b5e20"
TREE_SUSPECT_BG = "#fbedc4"
TREE_SUSPECT_FG = "#6b4c00"
TREE_ACTIVE_BG = "#cfe2ff"   # the run whose QC is open in an auto-open queue
TREE_ACTIVE_FG = "#0d3a75"


def cleanup_defaults(cell_key, sample_overrides=None):
    """Per-cell cleanup toggle defaults, with optional per-sample overrides
    from samples.json. Stitching is data-driven (mini: never — that is the
    seam the B-loop remanence bug lived in)."""
    base = {
        "stitch_t_curves":        cell_key == "str_dil",
        "stitch_b_loops":         cell_key == "str_dil",
        "dwell_removal":          True,
        "smooth_outside_protect": True,
    }
    for k, v in (sample_overrides or {}).get(cell_key, {}).items():
        if k in base:
            base[k] = bool(v)
    return base


def pick_writable_out_base(data_folder):
    """A folder that can actually hold Output/. Writing inside the app dir
    fails on read-only or policy-locked installs (e.g. a managed Mac that
    blocks ~/Downloads), so prefer next to the data (the user reaches it),
    then the app dir, then the home directory, then the system temp dir."""
    candidates = []
    if data_folder and os.path.isdir(data_folder):
        candidates.append(os.path.join(data_folder, "dilat_output"))
    candidates.append(os.path.join(PROC, "Output"))
    candidates.append(os.path.join(os.path.expanduser("~"), "dilatometry_output"))
    import tempfile
    candidates.append(os.path.join(tempfile.gettempdir(), "dilatometry_output"))
    for base in candidates:
        try:
            os.makedirs(base, exist_ok=True)
            probe = os.path.join(base, ".write_test")
            with open(probe, "w", encoding="utf-8") as fh:
                fh.write("ok")
            os.remove(probe)
            return base
        except OSError:
            continue
    return candidates[-1]   # temp dir; if even this fails the run will report it


def find_science_python():
    """First candidate interpreter that can import numpy+pandas, else None."""
    seen = set()
    for cand in SCIENCE_CANDIDATES:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        try:
            r = subprocess.run([cand, "-c", "import numpy, pandas"],
                               capture_output=True, timeout=30,
                               **POPEN_NOWINDOW)
            if r.returncode == 0:
                return cand
        except (OSError, subprocess.SubprocessError):
            continue
    return None


def _pick_mono_family(root):
    """A monospaced family that actually exists on this platform."""
    try:
        fams = set(tkfont.families(root))
    except tk.TclError:
        fams = set()
    for cand in ("Menlo", "Consolas", "DejaVu Sans Mono", "Courier New"):
        if cand in fams:
            return cand
    return "TkFixedFont"


class _Tooltip:
    """Minimal hover tooltip (stdlib only)."""
    def __init__(self, widget, text):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _e=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left",
                 background="#ffffe0", relief="solid", borderwidth=1,
                 font=("TkDefaultFont", 9), wraplength=360).pack()

    def _hide(self, _e=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class DilatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dilatometry Processor")
        # Open no taller/wider than the screen (Windows laptops + display
        # scaling otherwise push the Results/QC buttons below the taskbar).
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = min(900, sw - 40), min(820, sh - 80)
        self.root.geometry(f"{w}x{h}")
        # Small floor; the whole form scrolls, so it never traps content.
        self.root.minsize(560, 420)
        self._apply_theme()
        self.mono = (_pick_mono_family(root), 11)
        self.mono_sm = (self.mono[0], 10)

        self.science_py = find_science_python()
        # Preload the default run only if it exists on this machine; on a
        # fresh clone start blank and point the user at Browse instead.
        self.selected_path = tk.StringVar(
            value=HARDCODED_FILE if os.path.isfile(HARDCODED_FILE) else "")
        self.cell_var = tk.StringVar(value=CELL_CHOICES[0])
        self.rescale_var = tk.BooleanVar(value=False)   # default OFF (review #1)
        self.detection = None                           # last detect.py dict
        self._rescale_shown_for = None                  # (path, cell) preview flag
        self.qc_by_item = {}                            # tree item id -> (cmd, prov_path)
        self._out_dirs = {}                             # cell -> output folder of the last Run
        self._base_tags = {}                            # tree item id -> base row tag

        # Auto-open QC after a successful Run (P3.5). str -> one window; mini ->
        # a SEQUENTIAL queue (one QC subprocess at a time). The queue runs in a
        # worker thread; UI updates go through self._ui_q so the mainloop never
        # blocks. "Skip rest" cancels the remaining launches without killing an
        # already-open window.
        self.auto_open_var = tk.BooleanVar(value=True)  # default ON
        self._skip_event = None                         # threading.Event while a queue runs
        self._queue_active = False

        # sample registry (samples.json). A newer schema is REFUSED loudly.
        self.samples_reg = {"schema": samples.SCHEMA, "samples": {}}
        self._samples_error = None
        try:
            self.samples_reg = samples.load_samples()
        except samples.SamplesError as e:
            self._samples_error = str(e)
        self.sample_names = ["(manual)"] + list(self.samples_reg["samples"])
        self.sample_var = tk.StringVar(value=self.sample_names[0])
        self.l0_var = tk.StringVar(value="")
        self.angle_var = tk.StringVar(value="0")
        # Output resolution knobs (WIRED to the batch Run, like the
        # cleanup toggles): alpha bin width and dL/L0 point spacing, both in K.
        self.alpha_bin_var = tk.StringVar(value="0.2")   # auto-coarsened when
        self.dl_spacing_var = tk.StringVar(value="0.2")  # the data are sparse
        # Assisted/manual transition (P6): value in K + kind, passed to the
        # batch reducer as --transition/--transition-type. Default "none".
        self.transition_value = tk.StringVar(value="")
        self.transition_kind = tk.StringVar(value="none")
        self.transition_window = None       # [lo, hi] protect window or None
        self.sample_overrides = {}          # per-cell cleanup overrides
        self.toggle_vars = {k: tk.BooleanVar() for k, _ in CLEANUP_TOGGLES}

        self._ui_q = queue.Queue()
        self._build()
        self._schedule_scroll_sync()          # size the scroll region once built
        self.root.bind_all("<Command-o>", lambda _e: self.pick_file())
        self.root.bind_all("<Control-o>", lambda _e: self.pick_file())
        self.root.after(100, self._drain_ui)
        if self.selected_path.get():
            self.root.after(200, self.run_detection)

    # ---- theme ---------------------------------------------------------
    def _apply_theme(self):
        self.style = ttk.Style()
        names = self.style.theme_names()
        for preferred in ("aqua", "vista", "clam"):
            if preferred in names:
                try:
                    self.style.theme_use(preferred)
                    break
                except tk.TclError:
                    continue
        # A slightly heavier font for section titles reads as grouping.
        self.style.configure("Section.TLabelframe.Label",
                             font=("TkDefaultFont", 11, "bold"))

    # ---- build ---------------------------------------------------------
    def _make_scrollable(self):
        """Wrap the whole form in a vertically scrollable canvas so no control
        is ever off-screen (Windows/scaled displays make the form taller than
        the screen). The interior keeps its natural height and the scrollbar/
        mouse-wheel reach everything; it only stretches to fill when the window
        is genuinely taller than the form, so it never clips the content."""
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.scroll_canvas = tk.Canvas(self.root, highlightthickness=0)
        self.scroll_canvas.grid(row=0, column=0, sticky="nsew")
        vbar = ttk.Scrollbar(self.root, orient="vertical",
                             command=self.scroll_canvas.yview)
        vbar.grid(row=0, column=1, sticky="ns")
        self.scroll_canvas.configure(yscrollcommand=vbar.set)
        self.body = ttk.Frame(self.scroll_canvas)
        self._body_id = self.scroll_canvas.create_window(
            (0, 0), window=self.body, anchor="nw")
        # Sync runs deferred (after_idle) so the body's requested height is read
        # AFTER layout settles — reading it mid-storm gave a stale, too-short
        # value that clipped the bottom of the form.
        self.body.bind("<Configure>", lambda _e: self._schedule_scroll_sync())
        self.scroll_canvas.bind("<Configure>",
                                lambda _e: self._schedule_scroll_sync())

        def _wheel(evt):
            node = evt.widget                       # let Tree/Log scroll self
            while node is not None:
                if node is getattr(self, "tree", None) or \
                        node is getattr(self, "log", None):
                    return
                node = getattr(node, "master", None)
            if getattr(evt, "num", None) == 4:
                self.scroll_canvas.yview_scroll(-1, "units")
            elif getattr(evt, "num", None) == 5:
                self.scroll_canvas.yview_scroll(1, "units")
            elif evt.delta:
                self.scroll_canvas.yview_scroll(
                    -1 if evt.delta > 0 else 1, "units")
        self.scroll_canvas.bind_all("<MouseWheel>", _wheel)   # Win/Mac
        self.scroll_canvas.bind_all("<Button-4>", _wheel)     # Linux up
        self.scroll_canvas.bind_all("<Button-5>", _wheel)     # Linux down

    def _schedule_scroll_sync(self):
        if getattr(self, "_scroll_sync_pending", False):
            return
        self._scroll_sync_pending = True
        self.scroll_canvas.after_idle(self._scroll_sync)

    def _scroll_sync(self):
        self._scroll_sync_pending = False
        nat = self.body.winfo_reqheight()            # settled natural height
        cw = self.scroll_canvas.winfo_width()
        ch = self.scroll_canvas.winfo_height()
        # Fill the viewport when the window is taller than the form; otherwise
        # keep the form's natural height and let the scrollbar reach the rest.
        self.scroll_canvas.itemconfigure(
            self._body_id, width=cw, height=max(ch, nat))
        self.scroll_canvas.configure(scrollregion=(0, 0, cw, max(ch, nat)))

    def _section(self, title, row, weight=0):
        """A titled section frame gridded into the scroll body at `row`.
        weight>0 lets the row grow when the window is taller than the form."""
        f = ttk.LabelFrame(self.body, text=title, style="Section.TLabelframe")
        sticky = "nsew" if weight else "ew"
        f.grid(row=row, column=0, sticky=sticky, padx=PAD, pady=(PAD // 2, 0))
        if weight:
            self.body.rowconfigure(row, weight=weight)
        return f

    def _build(self):
        self._make_scrollable()
        self.body.columnconfigure(0, weight=1)

        # ── File ────────────────────────────────────────────────────────────
        filef = self._section("File", 0)
        row = ttk.Frame(filef)
        row.pack(fill="x", padx=PAD, pady=PAD // 2)
        ttk.Label(row, text="Raw file:").pack(side="left")
        ttk.Button(row, text="Browse…", command=self.pick_file, width=9).pack(
            side="right", padx=(6, 0))
        self.path_entry = ttk.Entry(row, textvariable=self.selected_path,
                                    font=self.mono_sm)
        self.path_entry.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.path_entry.bind("<Return>", lambda _e: self.run_detection())

        # ── Detection ───────────────────────────────────────────────────────
        det = self._section("Detection (suggestion only)", 1)
        head = ttk.Frame(det)
        head.pack(fill="x", padx=PAD, pady=(PAD // 2, 0))
        self.detect_values = ttk.Label(head, text="(detecting…)",
                                       font=self.mono, anchor="w")
        self.detect_values.pack(side="left")
        self.conf_badge = tk.Label(head, text="", font=("TkDefaultFont", 9, "bold"),
                                   padx=8, pady=1, bd=0)
        self.conf_badge.pack(side="right")
        self.detect_guess = ttk.Label(det, text="", anchor="w",
                                      font=("TkDefaultFont", 11))
        self.detect_guess.pack(fill="x", padx=PAD, pady=(2, 0))
        self._wrap(self.detect_guess)
        self.cand_caption = ttk.Label(det, text="", foreground=SECONDARY,
                                      font=("TkDefaultFont", 9))
        self.cand_caption.pack(fill="x", padx=PAD)
        self.cand_frame = ttk.Frame(det)
        self.cand_frame.pack(fill="x", padx=(PAD * 3, PAD), pady=(0, PAD // 2))

        # ── Cell ────────────────────────────────────────────────────────────
        cellf = self._section("Cell (confirm or override)", 2)
        crow = ttk.Frame(cellf)
        crow.pack(fill="x", padx=PAD, pady=PAD // 2)
        ttk.Label(crow, text="Cell:").pack(side="left", padx=(0, 4))
        self.cell_combo = ttk.Combobox(
            crow, textvariable=self.cell_var, values=CELL_CHOICES,
            state="readonly", width=12)
        self.cell_combo.pack(side="left")
        self.cell_combo.bind("<<ComboboxSelected>>", self._on_cell_change)

        # ── Rescale offer (row 3, hidden until a mini r_eff~7 mm run) ────────
        self.rescale_frame = ttk.LabelFrame(
            self.root,
            text="Mis-conversion rescale (suggestion — OFF by default)",
            style="Section.TLabelframe")
        self.rescale_chk = ttk.Checkbutton(
            self.rescale_frame,
            text="apply rescale to this mini run",
            variable=self.rescale_var, command=self._on_rescale_toggle)
        self.rescale_chk.pack(side="left", padx=(PAD, 6), pady=PAD // 2)
        self.rescale_lbl = ttk.Label(self.rescale_frame, text="",
                                     foreground="#8c564b", anchor="w")
        self.rescale_lbl.pack(side="left", fill="x", expand=True, pady=PAD // 2)
        self._wrap(self.rescale_lbl)
        self.rescale_frame.grid(row=3, column=0, sticky="ew",
                                padx=PAD, pady=(PAD // 2, 0))
        self.rescale_frame.grid_remove()   # shown by _update_rescale_offer

        # ── Sample & parameters ─────────────────────────────────────────────
        self.paramf = self._section("Sample & parameters", 4)
        line1 = ttk.Frame(self.paramf)
        line1.pack(fill="x", padx=PAD, pady=(PAD // 2, 2))
        ttk.Label(line1, text="Sample:").pack(side="left")
        self.sample_combo = ttk.Combobox(
            line1, textvariable=self.sample_var, values=self.sample_names,
            state="readonly", width=12)
        self.sample_combo.pack(side="left", padx=(4, 16))
        self.sample_combo.bind("<<ComboboxSelected>>", self._on_sample_change)
        ttk.Label(line1, text="L0 (cm):").pack(side="left")
        self.l0_entry = ttk.Entry(line1, textvariable=self.l0_var, width=10)
        self.l0_entry.pack(side="left", padx=(4, 16))
        ttk.Label(line1, text="angle (deg):").pack(side="left")
        # Disabled on purpose: NO batch driver consumes an angle from the GUI
        # (str ignores angle; mini takes per-run angles from angle_runs.json).
        # A live box here would be a silent no-op — the audited failure mode.
        self.angle_entry = ttk.Entry(line1, textvariable=self.angle_var,
                                     width=6, state="disabled")
        self.angle_entry.pack(side="left", padx=(4, 0))
        _Tooltip(self.angle_entry,
                 "Not an input: str runs ignore angle; mini runs take "
                 "per-angle settings from angle_runs.json.")

        # Output resolution row — these ARE passed to the batch Run.
        line_res = ttk.Frame(self.paramf)
        line_res.pack(fill="x", padx=PAD, pady=(0, 2))
        ttk.Label(line_res, text="α bin (K):").pack(side="left")
        ttk.Entry(line_res, textvariable=self.alpha_bin_var, width=6).pack(
            side="left", padx=(4, 16))
        ttk.Label(line_res, text="ΔL/L₀ spacing (K):").pack(side="left")
        ttk.Entry(line_res, textvariable=self.dl_spacing_var, width=6).pack(
            side="left", padx=(4, 8))
        ttk.Label(line_res, text="(blank = every point)", foreground=MUTED,
                  font=("TkDefaultFont", 9)).pack(side="left")

        # Transition (P6) — value + type passed to the batch reducer, plus an
        # assisted "Find transition" button that runs transition.py against
        # the science interpreter and proposes the strongest alpha-peak.
        line_trans = ttk.Frame(self.paramf)
        line_trans.pack(fill="x", padx=PAD, pady=(0, 2))
        ttk.Label(line_trans, text="Transition (K):").pack(side="left")
        self.transition_entry = ttk.Entry(
            line_trans, textvariable=self.transition_value, width=8)
        self.transition_entry.pack(side="left", padx=(4, 16))
        ttk.Label(line_trans, text="type:").pack(side="left")
        self.transition_combo = ttk.Combobox(
            line_trans, textvariable=self.transition_kind,
            values=list(TRANSITION_KINDS), state="readonly", width=8)
        self.transition_combo.pack(side="left", padx=(4, 16))
        self.find_transition_btn = ttk.Button(
            line_trans, text="Find transition", command=self._on_find_transition)
        self.find_transition_btn.pack(side="left")

        self.protect_lbl = ttk.Label(self.paramf, text="", anchor="w",
                                     foreground=SECONDARY)
        self.protect_lbl.pack(fill="x", padx=PAD, pady=(0, 2))
        self._wrap(self.protect_lbl)

        tog = ttk.Frame(self.paramf)
        tog.pack(fill="x", padx=PAD, pady=(0, 2))
        ttk.Label(tog, text="cleanup:").pack(side="left", padx=(0, 6))
        tog_note = ("Applied to the Run via CLI flags. Selecting a cell resets "
                    "these to its safe defaults (mini: stitching OFF — the "
                    "below-T_C staircase on loop-dense runs is remanence "
                    "SIGNAL, and stitching rectifies it into fake drift).")
        for key, label in CLEANUP_TOGGLES:
            chk = ttk.Checkbutton(tog, text=label,
                                  variable=self.toggle_vars[key])
            chk.pack(side="left", padx=(0, 10))
            _Tooltip(chk, tog_note)
        cap = ttk.Label(self.paramf,
                        text="Cell change resets to per-cell defaults.",
                        foreground=MUTED, font=("TkDefaultFont", 9))
        cap.pack(anchor="w", padx=PAD, pady=(0, PAD // 2))

        # ── Actions ─────────────────────────────────────────────────────────
        actf = self._section("Actions", 5)
        arow = ttk.Frame(actf)
        arow.pack(fill="x", padx=PAD, pady=PAD // 2)
        self.redetect_btn = ttk.Button(arow, text="Re-detect",
                                       command=self.run_detection)
        self.redetect_btn.pack(side="left")
        self.run_btn = ttk.Button(arow, text="Run reduction",
                                  command=self.run_reduction)
        self.run_btn.pack(side="left", padx=6)
        self.auto_open_chk = ttk.Checkbutton(
            arow, text="auto-open plots after run",
            variable=self.auto_open_var)
        self.auto_open_chk.pack(side="left", padx=(12, 0))
        # "Skip rest" is packed ONLY while a mini auto-open queue is running.
        self.skip_btn = ttk.Button(arow, text="Skip rest",
                                   command=self._skip_queue)
        self.progress = ttk.Progressbar(arow, mode="indeterminate", length=160)
        self.run_status = ttk.Label(arow, text="", foreground=SECONDARY)
        # progress + status are packed only while a Run is in flight.

        # Persistent "what will Run actually do" line: the advisory-vs-
        # effective gap (L0/angle ignored by batch, mini's whole-folder
        # sweep) must be visible AT the Run button, not buried in the log.
        self.run_summary = ttk.Label(actf, text="",
                                     foreground=SECONDARY, justify="left")
        self.run_summary.pack(fill="x", padx=PAD, pady=(0, PAD // 2))
        self._wrap(self.run_summary)
        for var in (self.cell_var, self.l0_var, self.angle_var,
                    self.selected_path):
            var.trace_add("write", lambda *_a: self._update_run_summary())
        self._update_run_summary()

        # ── Results (grows on resize) ───────────────────────────────────────
        resf = self._section(
            "Results — open QC on demand (no auto-opened windows)", 6, weight=1)
        resf.columnconfigure(0, weight=1)
        resf.rowconfigure(0, weight=1)
        cols = ("run", "status", "summary")
        self.tree = ttk.Treeview(resf, columns=cols, show="headings",
                                 height=6, selectmode="browse")
        self.tree.heading("run", text="Run")
        self.tree.heading("status", text="Status")
        self.tree.heading("summary", text="Summary")
        self.tree.column("run", width=150, minwidth=90, stretch=False,
                         anchor="w")
        self.tree.column("status", width=80, minwidth=64, stretch=False,
                         anchor="center")
        self.tree.column("summary", width=420, minwidth=160, stretch=True,
                         anchor="w")
        self.tree.tag_configure("pass", background=TREE_PASS_BG,
                                foreground=TREE_PASS_FG)
        self.tree.tag_configure("suspect", background=TREE_SUSPECT_BG,
                                foreground=TREE_SUSPECT_FG)
        self.tree.tag_configure("placeholder", foreground=MUTED)
        self.tree.tag_configure("active", background=TREE_ACTIVE_BG,
                                foreground=TREE_ACTIVE_FG)
        self.tree.grid(row=0, column=0, sticky="nsew", padx=(PAD, 0),
                       pady=(PAD // 2, 0))
        vsb = ttk.Scrollbar(resf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.grid(row=0, column=1, sticky="ns", pady=(PAD // 2, 0))
        self.tree.bind("<<TreeviewSelect>>", self._on_result_select)
        legend = ttk.Label(
            resf, foreground=MUTED, font=("TkDefaultFont", 9),
            text="PASS = quality gates met · CHECK = a gate failed — inspect "
                 "in QC · PRELIM = mini, gates advisory · ERR = unreadable")
        legend.grid(row=2, column=0, columnspan=2, sticky="w", padx=PAD,
                    pady=(0, PAD // 2))
        self._wrap(legend)
        self.tree.bind("<Double-1>", lambda _e: self._open_selected_qc())
        rbtns = ttk.Frame(resf)
        rbtns.grid(row=1, column=0, columnspan=2, sticky="ew",
                   padx=PAD, pady=PAD // 2)
        self.open_qc_btn = ttk.Button(rbtns, text="Open QC",
                                      command=self._open_selected_qc,
                                      state="disabled")
        self.open_qc_btn.pack(side="left")
        self.precleanup_btn = ttk.Button(
            rbtns, text="Plot pre-cleanup",
            command=self._plot_selected_precleanup, state="disabled")
        self.precleanup_btn.pack(side="left", padx=(6, 0))
        ttk.Label(rbtns, text="select a run → Open QC (double-click) or "
                  "Plot pre-cleanup (saves _raw png/csv)",
                  foreground=MUTED, font=("TkDefaultFont", 9)).pack(
            side="left", padx=8)
        self._set_results_placeholder()

        # ── Log (grows on resize) ───────────────────────────────────────────
        logf = self._section("Log", 7, weight=2)
        logf.columnconfigure(0, weight=1)
        logf.rowconfigure(1, weight=1)
        ltop = ttk.Frame(logf)
        ltop.grid(row=0, column=0, sticky="ew", padx=4, pady=(2, 0))
        ttk.Button(ltop, text="Clear", command=self._clear_log, width=7).pack(
            side="right")
        self.log = scrolledtext.ScrolledText(logf, height=8, wrap="word",
                                             font=self.mono_sm)
        self.log.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        # ── Status bar ──────────────────────────────────────────────────────
        sci = self.science_py or "NONE FOUND"
        self.status_lbl = ttk.Label(
            self.root, text=f"science python: {sci}", anchor="w",
            relief="sunken", foreground=SECONDARY)
        self.status_lbl.grid(row=8, column=0, sticky="ew")
        self._wrap(self.status_lbl)

        if self.selected_path.get():
            self._log(f"Ready. Opening file:\n  {self.selected_path.get()}")
        else:
            self._log("Ready. No default run on this machine — use Browse… "
                      "(or Cmd/Ctrl-O) to pick a raw PPMS .dat or *_all.csv.")
        if self._samples_error:
            self._log("samples.json REFUSED: " + self._samples_error)
        else:
            n = len(self.samples_reg["samples"])
            self._log(f"samples.json: {n} sample(s) loaded "
                      f"({', '.join(self.samples_reg['samples']) or 'none'}).")
        if not self.science_py:
            self._log("WARNING: no interpreter with numpy/pandas found; "
                      "detection, Run and Open-QC will fail.")
        # Initialise the toggle defaults + L0 for the opening cell/sample.
        self._refresh_toggles()

    # ---- resize: reflow long labels so nothing clips when narrow --------
    def _wrap(self, lbl):
        """Make a fill=x label wrap to its OWN allocated width (canonical
        tkinter idiom) so long text reflows instead of clipping at any size."""
        def _rewrap(event, w=lbl):
            new = max(60, event.width - 4)
            if abs(new - (w.cget("wraplength") or 0)) > 2:
                w.config(wraplength=new)
        lbl.bind("<Configure>", _rewrap)

    # ---- thread-safe UI updates via one queue --------------------------
    def _log(self, msg):
        self._ui_q.put(("log", msg))

    def _clear_log(self):
        self.log.delete("1.0", "end")

    def _drain_ui(self):
        try:
            while True:
                kind, payload = self._ui_q.get_nowait()
                if kind == "log":
                    self.log.insert("end", payload + "\n")
                    self.log.see("end")
                elif kind == "detected":
                    self._apply_detection(payload)
                elif kind == "detect_text":
                    self._show_detect_error(payload)
                elif kind == "runbtn":
                    self.run_btn.config(state=payload)
                elif kind == "run_done":
                    cell, rc = payload
                    self._set_running(False)
                    self._populate_results(cell)
                    if rc == 0:
                        self._start_autoopen(cell)
                    elif rc is not None:
                        self._log("Run exited non-zero — auto-open skipped.")
                elif kind == "transition_found":
                    self._apply_transition_result(payload)
                elif kind == "transition_error":
                    self._show_transition_error(payload)
                elif kind == "tree_active":
                    self._set_active_row(payload)
                elif kind == "queue_done":
                    self._end_queue()
        except queue.Empty:
            pass
        self.root.after(100, self._drain_ui)

    # ---- file picker ---------------------------------------------------
    def pick_file(self):
        start = os.path.dirname(self.selected_path.get()) or PROC
        p = filedialog.askopenfilename(
            initialdir=start, title="Select raw PPMS file",
            filetypes=[("Data", "*.csv *.dat"), ("All", "*.*")])
        if p:
            self.selected_path.set(p)
            self._log(f"Selected: {p}")
            self.run_detection()   # warns about stale L0 for ALL entry paths

    # ---- detection (subprocess to detect.py under science python) ------
    def run_detection(self):
        path = self.selected_path.get()
        if not self.science_py:
            self._show_detect_error("detection unavailable (no science "
                                    "interpreter)")
            return
        if not os.path.exists(path):
            self._show_detect_error(f"file not found: {path}")
            return
        self._warn_stale_l0(path)   # here, not in pick_file: the path Entry
        #                             (Return key) reaches detection too
        self.detect_values.config(text="(detecting…)")
        self._set_badge(None, "")
        # Sequence token: rapid re-triggers (Return held in the path box) may
        # finish out of order; only the LATEST request may fill the UI.
        self._detect_seq = getattr(self, "_detect_seq", 0) + 1
        threading.Thread(target=self._detect_worker,
                         args=(path, self._detect_seq), daemon=True).start()

    def _detect_worker(self, path, seq):
        def post(kind, payload):
            if seq == getattr(self, "_detect_seq", seq):   # drop stale results
                self._ui_q.put((kind, payload))
        try:
            r = subprocess.run(
                [self.science_py, DETECT_WORKER, path],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace", cwd=HERE, timeout=120,
                **POPEN_NOWINDOW)
            if r.returncode != 0:
                post("detect_text", f"detection failed:\n{r.stderr.strip()}")
                return
            d = json.loads(r.stdout.strip().splitlines()[-1])
            d["_path"] = path
            post("detected", d)
        except Exception as e:  # noqa: BLE001 — GUI must not crash on detect
            post("detect_text", f"detection error: {e}")

    # ---- assisted transition finder (subprocess to transition.py) ------
    def _on_find_transition(self):
        path = self.selected_path.get()
        if not path:
            messagebox.showinfo("Find transition", "Select a data file first.")
            return
        if not self.science_py:
            self._log("Cannot Find transition: no science interpreter.")
            messagebox.showinfo("Find transition",
                                "No science interpreter available.")
            return
        if not os.path.exists(path):
            messagebox.showinfo("Find transition", f"File not found: {path}")
            return
        self.find_transition_btn.config(state="disabled")
        self._log(f"\n=== Find transition: {path} ===")
        threading.Thread(target=self._find_transition_worker, args=(path,),
                         daemon=True).start()

    def _find_transition_worker(self, path):
        try:
            r = subprocess.run(
                [self.science_py, TRANSITION_WORKER, path],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace", cwd=HERE, timeout=120,
                **POPEN_NOWINDOW)
            try:
                data = json.loads(r.stdout.strip().splitlines()[-1])
            except Exception:
                data = {}
            if r.returncode != 0 and not data:
                self._ui_q.put(("transition_error",
                                r.stderr.strip() or "transition.py failed"))
                return
            if data.get("error"):
                self._ui_q.put(("transition_error", data["error"]))
                return
            self._ui_q.put(("transition_found", data.get("candidates", [])))
        except Exception as e:  # noqa: BLE001 — GUI must not crash on find
            self._ui_q.put(("transition_error", str(e)))

    def _apply_transition_result(self, cands):
        """Main-thread: populate the value field from the top candidate, or
        report a clean 'no transition found' message. Never touches the
        type dropdown — the user still confirms T_C/T_N/T_CDW."""
        self.find_transition_btn.config(state="normal")
        if not cands:
            self._log("Find transition: no clear transition found (flat alpha).")
            messagebox.showinfo(
                "Find transition",
                "No clear transition found (flat alpha).\n"
                "Leave type = none, or enter a value manually.")
            return
        top = cands[0]
        self.transition_value.set(f"{top['T']:.1f}")
        self._log(f"Find transition: candidates={cands}")
        messagebox.showinfo(
            "Find transition",
            "Strongest candidate: {:.1f} K (prominence {}).\n"
            "Pick the type (T_C / T_N / T_CDW) to confirm.\n"
            "Note: on a raw .dat file this scan reads the uncalibrated "
            "raw signal, so treat the value as an assist, not a "
            "calibrated result."
            .format(top["T"], top.get("prominence")))

    def _show_transition_error(self, msg):
        self.find_transition_btn.config(state="normal")
        self._log(f"Find transition error: {msg}")
        messagebox.showinfo("Find transition", f"Could not run: {msg}")

    def _show_detect_error(self, msg):
        self.detect_values.config(text=msg)
        self._set_badge("unknown", "n/a")
        self.detect_guess.config(text="-> choose the cell manually below.")
        self.cand_caption.config(text="")
        for w in self.cand_frame.winfo_children():
            w.destroy()

    def _set_badge(self, kind, text):
        fg, bg = BADGE.get(kind, (SECONDARY, "#e8e8e8"))
        if not text:
            self.conf_badge.config(text="", background=self.root.cget("bg"))
        else:
            self.conf_badge.config(text=text, foreground=fg, background=bg)

    def _apply_detection(self, d):
        """Main-thread: show result, set dropdown default, refresh offer."""
        self.detection = d
        for w in self.cand_frame.winfo_children():
            w.destroy()
        if d.get("error") or d.get("r_eff_mm") is None:
            self.detect_values.config(text=d.get("note", "detection failed"))
            self._set_badge("unknown", "n/a")
            self.detect_guess.config(text="-> choose the cell manually below.")
            self.cand_caption.config(text="")
        else:
            self.detect_values.config(
                text=(f"r_eff = {d['r_eff_mm']} mm     "
                      f"C_med = {d['C_med_pF']} pF     {d['n_rows']} rows"))
            conf = str(d.get("confidence", "?"))
            self._set_badge(conf, conf)
            self.detect_guess.config(text=f"best guess:  {d.get('cell_guess','?')}")
            cands = d.get("candidates", [])
            self.cand_caption.config(text="candidates:" if cands else "")
            for c in cands:
                lbl = ttk.Label(self.cand_frame, text="•  " + c,
                                foreground=SECONDARY, font=self.mono_sm,
                                anchor="w", justify="left")
                lbl.pack(fill="x", anchor="w")
                self._wrap(lbl)
        self._log(f"Detection: r_eff={d.get('r_eff_mm')} mm, "
                  f"guess={d.get('cell_guess')}, "
                  f"confidence={d.get('confidence')}. {d.get('note', '')}")
        guess = d.get("cell_guess")
        if guess in CELL_CHOICES:
            self.cell_var.set(guess)
        self._rescale_shown_for = None
        self._refresh_toggles()
        self._update_rescale_offer()

    # ---- sample dropdown -----------------------------------------------
    def _on_sample_change(self, _evt=None):
        name = self.sample_var.get()
        entry = self.samples_reg["samples"].get(name)
        if entry is None:                       # "(manual)"
            self.transition_window = None
            self.sample_overrides = {}
            self.protect_lbl.config(text="manual entry — no sample defaults.")
            self._log("Sample: (manual) — enter L0/angle by hand; cleanup "
                      "defaults follow the cell only.")
            self._refresh_toggles()
            return
        # sample sets the default cell first (drives toggles + L0 + rescale)
        self.sample_overrides = entry.get("cleanup_overrides", {})
        self.transition_window = entry.get("transition_window_K")
        default_cell = entry.get("default_cell", self.cell_var.get())
        if default_cell in CELL_CHOICES:
            self.cell_var.set(default_cell)
        self._prefill_l0(entry)
        tw = self.transition_window
        self.protect_lbl.config(
            text=(f"{entry.get('display_name', name)}   |   protect window "
                  f"{tw[0]}–{tw[1]} K" if tw else
                  f"{entry.get('display_name', name)}   |   no protect window")
            + (f"   |   {entry['notes']}" if entry.get("notes") else ""))
        self._log(f"Sample: {name} -> cell {self.cell_var.get()}, "
                  f"protect window {tw}. {entry.get('notes', '')}")
        self._refresh_toggles()
        self._update_rescale_offer()

    def _prefill_l0(self, entry):
        opt = samples.l0_for_cell(entry, self.cell_var.get())
        if opt is None:
            return
        self.l0_var.set(str(opt["value"]))
        self._l0_prefilled_for = self.sample_var.get()
        conf = "CONFIRMED" if opt.get("confirmed") else "UNCONFIRMED"
        self._log(f"L0 prefilled: {opt['value']} cm [{conf}] — {opt.get('source','')}")

    def _warn_stale_l0(self, file_path):
        """A sample-prefilled L0 silently applied to an unrelated file gave a
        wrong-scale reduction and a (correct) gate FAIL on the Cu runs — flag
        the mismatch the moment the file is picked."""
        sample = getattr(self, "_l0_prefilled_for", None)
        if not sample or not self.l0_var.get().strip():
            return
        if sample.lower() not in os.path.basename(file_path).lower():
            self._log(f"WARNING: L0 = {self.l0_var.get()} cm was prefilled "
                      f"for sample '{sample}', but the selected file is "
                      f"'{os.path.basename(file_path)}'. If this is a "
                      "different sample, SET ITS OWN L0 (cm) before Run — "
                      "a wrong L0 scales dL/L0 and alpha and can fail the "
                      "cool-warm consistency gate.")

    # ---- cell dropdown + toggles + rescale -----------------------------
    def _on_cell_change(self, _evt=None):
        self._log(f"Cell confirmed: {self.cell_var.get()}")
        # re-prefill L0 for the new cell if a sample is selected
        entry = self.samples_reg["samples"].get(self.sample_var.get())
        if entry is not None:
            self._prefill_l0(entry)
        self._refresh_toggles()
        self._update_rescale_offer()

    def _refresh_toggles(self):
        """Reset the cleanup toggles to the per-cell (+ per-sample)
        defaults. They flip str<->mini so the user sees what the driver does."""
        defs = cleanup_defaults(self.cell_var.get(), self.sample_overrides)
        for key, _label in CLEANUP_TOGGLES:
            self.toggle_vars[key].set(defs[key])

    def _update_rescale_offer(self):
        """Show the rescale offer ONLY when the confirmed cell is mini_dil AND
        detection found r_eff ~ 7 mm (rescale_factor populated). Otherwise hide
        it and force the checkbox OFF."""
        d = self.detection or {}
        factor = d.get("rescale_factor")
        show = (self.cell_var.get() == "mini_dil" and factor is not None)
        if not show:
            self.rescale_var.set(False)
            self.rescale_frame.grid_remove()
            return
        self.rescale_lbl.config(
            text=f"factor × {factor:.4f}  (mini_dil, r_eff~7 mm — possible "
                 "7 mm mis-conversion; NEVER auto-applied)")
        self.rescale_frame.grid()   # re-show at its fixed row (3)
        key = (d.get("_path"), "mini_dil")
        if self._rescale_shown_for != key:
            self._rescale_shown_for = key
            p2p = d.get("dl_p2p_1e6cm")
            self._log(
                "Rescale SUGGESTION (mini_dil, r_eff~7 mm): factor × "
                f"{factor:.4f}. Preview of raw delta_l peak-to-peak:\n"
                f"    without rescale: {p2p} (1e-6 cm)\n"
                f"    with rescale:    {round(p2p * factor, 3) if p2p is not None else '?'} (1e-6 cm)\n"
                "Checkbox is OFF by default and is NEVER auto-applied.")

    def _on_rescale_toggle(self):
        state = "ON" if self.rescale_var.get() else "OFF"
        self._log(f"Rescale suggestion toggled {state} by user. When ON, Run "
                  "passes --rescale-file/--rescale-factor for the picked run; "
                  "a rescale already set in angle_runs.json wins (the reducer "
                  "warns instead of double-applying).")

    def _update_run_summary(self):
        """The persistent line under the Run button saying what Run will
        ACTUALLY do — including which UI fields it ignores."""
        sel = self.selected_path.get()
        if not sel:
            self.run_summary.config(
                text="No file selected — Browse… (Cmd/Ctrl-O) to pick a raw "
                     "PPMS .dat or *_all.csv, then Run.")
            self.run_btn.config(state="disabled")
            return
        if str(self.run_btn.cget("state")) == "disabled" and \
                not self.run_status.cget("text"):
            self.run_btn.config(state="normal")
        cell = self.cell_var.get()
        if cell == "mini_dil":
            scope = (f"ALL angle runs in {os.path.basename(os.path.dirname(sel)) or '/'}"
                     " (angle_runs.json there, or the built-in set) — not just "
                     "this file")
        else:
            scope = os.path.basename(sel)
        l0 = self.l0_var.get().strip()
        if cell == "mini_dil":
            l0_note = "L0 comes from angle_runs.json (this field is not used)"
        elif l0:
            l0_note = f"L0={l0} cm (used); angle not used by str batch"
        else:
            l0_note = "L0 blank → reducer default; set L0 (cm) for correct scale"
        self.run_summary.config(
            text=f"Run will use: {cell} · {scope} · {l0_note}.")

    # ---- Run reduction (subprocess to the batch worker for the cell) ---
    def _set_running(self, running):
        """Toggle the busy indicator + Run/Re-detect button state."""
        if running:
            self.run_btn.config(state="disabled")
            self.redetect_btn.config(state="disabled")
            self.run_status.config(text="Running reduction… (~1 min)")
            self.run_status.pack(side="left", padx=(10, 6))
            self.progress.pack(side="left")
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.pack_forget()
            self.run_status.pack_forget()
            self.run_status.config(text="")
            self.run_btn.config(state="normal")
            self.redetect_btn.config(state="normal")
            self._update_run_summary()   # re-disable Run if no file is picked

    def run_reduction(self):
        if not self.science_py:
            self._log("Cannot Run: no science interpreter.")
            return
        cell = self.cell_var.get()
        worker = REDUCE_WORKER.get(cell)
        if not worker:
            self._log(f"No batch worker for cell {cell!r}.")
            return
        sel = self.selected_path.get()
        if cell == "mini_dil":
            # The mini sweep reduces the whole FOLDER, not the picked file —
            # confirm so nobody expects a single-angle run.
            folder = os.path.dirname(sel)
            if not messagebox.askokcancel(
                    "Mini reduction sweeps the folder",
                    f"Mini reduces ALL angle runs found in\n{folder}\n"
                    "(per angle_runs.json there, or the built-in set) — "
                    "not just the picked file.\n\nContinue?"):
                self._log("Mini Run cancelled.")
                return
        self._set_running(True)
        # The picked file now drives the reduction input (was hardcoded before).
        cmd = [self.science_py, worker]
        if cell == "mini_dil":
            # mini is a multi-angle sweep: the picked file's FOLDER drives the
            # angle glob (a single --file would reduce just one angle).
            cmd += ["--data", os.path.dirname(sel)]
            self._log(f"NOTE: mini Run uses the FOLDER of the picked file "
                      f"({os.path.dirname(sel)}) to glob all angle archives; "
                      "the individual filename is not used for the batch sweep.")
        else:
            cmd += ["--data", os.path.dirname(sel), "--file", os.path.basename(sel)]
            self._log(f"Input: {sel}")
        # Results go to a WRITABLE folder (next to the data by default) — never
        # inside the app dir, which can be read-only / policy-locked.
        out_base = pick_writable_out_base(os.path.dirname(sel))
        out_dir = os.path.join(out_base, "mini" if cell == "mini_dil" else "str")
        self._out_dirs[cell] = out_dir
        cmd += ["--out", out_dir]
        self._log(f"Output → {out_dir}")
        self._log(f"Output written to: {out_base}")
        data_adjacent = os.path.join(os.path.dirname(sel), "dilat_output")
        if os.path.normpath(out_base) != os.path.normpath(data_adjacent):
            self._log("NOTE: the data folder was not writable, so output fell "
                      "back to a different location (see path above).")
        self._log("If you need results in Downloads (or another "
                  "protected folder), grant this app access: macOS System "
                  "Settings > Privacy & Security > Files and Folders; "
                  "Windows: allow it through Controlled Folder Access.")
        # Output resolution — WIRED to the batch driver (both cells accept it).
        ab = self.alpha_bin_var.get().strip()
        ds = self.dl_spacing_var.get().strip()
        try:
            if ab and float(ab) > 0:
                cmd += ["--alpha-bin", ab]
        except ValueError:
            self._log(f"α bin '{ab}' is not a number — ignored "
                      "(default 0.2 K, auto-coarsened if sparse).")
        try:
            if ds and float(ds) > 0:
                cmd += ["--dl-spacing", ds]
        except ValueError:
            self._log(f"ΔL/L₀ spacing '{ds}' is not a number — ignored "
                      "(every point kept).")
        # Cleanup toggles -> WIRED to both batch drivers as explicit flags,
        # so what the checkboxes show is exactly what the run does.
        for key, _label in CLEANUP_TOGGLES:
            flag = key.replace("_", "-")
            on = self.toggle_vars[key].get()
            cmd += [f"--{flag}" if on else f"--no-{flag}"]
        if cell == "mini_dil" and self.toggle_vars["stitch_b_loops"].get():
            self._log("WARNING: 'stitch B-loops' is ON for a mini run — on "
                      "loop-dense runs the below-T_C staircase is remanence "
                      "SIGNAL and stitching rectifies it into fake drift. "
                      "Only proceed if you have inspected the loops.")
        # L0 -> WIRED to the str batch driver via --L0 (mini takes L0 from
        # angle_runs.json's "L0_cm", so the field does not apply to mini).
        if cell == "str_dil":
            l0v = self.l0_var.get().strip()
            try:
                if l0v and float(l0v) > 0:
                    cmd += ["--L0", l0v]
                    self._log(f"NOTE: L0 = {l0v} cm passed to the str Run. "
                              "(angle is not used by the str batch driver.)")
                else:
                    self._log("NOTE: L0 field blank — the str reducer's default "
                              "L0 is used; set L0 (cm) for correct absolute "
                              "ΔL/L₀ and α.")
            except ValueError:
                self._log(f"L0 '{l0v}' is not a number — ignored; reducer "
                          "default L0 used.")
        else:
            self._log("NOTE: mini Run takes L0 (and per-angle settings) from "
                      "angle_runs.json; the L0/angle fields above are not used "
                      "by the mini batch sweep.")
        # Transition (P6) — WIRED to the str batch driver only. The mini
        # sweep has NO --transition CLI hook (reduce_mini_batch.py reads
        # transition_K / transition_type from angle_runs.json), and it uses
        # parse_known_args, so passing the flags to mini would be silently
        # swallowed — mirror the L0 cell-gating above and don't pretend.
        if cell == "str_dil":
            tr_kind = self.transition_kind.get()
            tr_val = self.transition_value.get().strip()
            if tr_kind != "none" and tr_val:
                cmd += ["--transition", tr_val, "--transition-type", tr_kind]
                self._log(f"Transition: {tr_val} K ({tr_kind}) passed to the "
                          "reducer.")
            else:
                cmd += ["--transition-type", "none"]
                if tr_kind != "none" and not tr_val:
                    self._log(f"NOTE: transition type '{tr_kind}' set but no "
                              "value given — treated as none (use Find "
                              "transition or enter a value).")
        else:
            self._log("NOTE: the mini batch sweep takes the transition from "
                      "angle_runs.json (transition_K / transition_type); the "
                      "transition fields above are not used for mini runs.")
        if self.rescale_var.get():
            factor = (self.detection or {}).get("rescale_factor")
            if cell == "mini_dil" and factor is not None:
                cmd += ["--rescale-file", os.path.basename(sel),
                        "--rescale-factor", f"{factor:.6f}"]
                self._log(f"Rescale: delta_l x {factor:.4f} passed for the "
                          f"picked run ({os.path.basename(sel)}). The reducer "
                          "skips it — with a warning — if angle_runs.json "
                          "already sets a rescale for that run.")
            else:
                self._log("NOTE: rescale checkbox is ON but no detection "
                          "factor is available for this cell/file — nothing "
                          "passed to the batch driver.")
        self._log(f"\n=== Run ({cell}): " + " ".join(cmd) + " ===")
        # Auto-open should pop ONLY this run's outputs, not every past run the
        # results glob now surfaces. Stamp the start; _start_autoopen filters
        # provenance files to those (re)written at or after this moment.
        self._run_start_time = time.time()
        threading.Thread(target=self._run_worker, args=(cmd, cell),
                         daemon=True).start()

    def _run_worker(self, cmd, cell):
        rc = None
        tail = []          # last few non-empty output lines, for a failure hint
        try:
            proc = subprocess.Popen(
                cmd, cwd=HERE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace", bufsize=1, **POPEN_NOWINDOW)
            for line in proc.stdout:
                line = line.rstrip("\n")
                self._log(line)
                if line.strip():
                    tail.append(line.strip())
                    tail = tail[-6:]
            proc.wait()
            rc = proc.returncode
            self._log(f"=== worker exit code: {rc} ===")
        except Exception as e:  # noqa: BLE001
            self._log(f"Run error: {e}")
            tail.append(f"could not start the reducer: {e}")
        finally:
            self._last_run_rc = rc
            self._last_run_tail = tail
            self._ui_q.put(("run_done", (cell, rc)))

    # ---- multi-run results list (review #10) ---------------------------
    def _provenance_files(self, cell):
        # Prefer the folder the last Run actually wrote to; fall back to the
        # app-dir default for runs made before this session.
        d = self._out_dirs.get(cell)
        if d:
            return sorted(glob.glob(os.path.join(d, "*_provenance.json")))
        pat = PROVENANCE_GLOB.get(cell)
        return sorted(glob.glob(pat)) if pat else []

    def _summarize(self, cell, prov, prov_path):
        """(run, status, summary, tag) for one run's Treeview row.

        The run name comes from the provenance filename stem (== the reducer's
        output STEM), so input-named str runs are distinguishable in the tree
        instead of all showing one default stem."""
        run_name = os.path.basename(prov_path)
        if run_name.endswith("_provenance.json"):
            run_name = run_name[: -len("_provenance.json")]
        if cell == "str_dil":
            g = prov.get("gates", {})
            a = g.get("alpha_paramagnetic_virgin", {})
            s = g.get("span_regression", {})
            v = g.get("virgin_cool_warm_consistency", {})
            allpass = all(x.get("pass") for x in g.values()) if g else False
            status = "PASS" if allpass else "CHECK"
            tag = "pass" if allpass else "suspect"
            summary = (
                f"α {a.get('cool_1e6_per_K','?')}/{a.get('warm_1e6_per_K','?')}"
                f"   span {s.get('span_cool_1e3','?')}/"
                f"{s.get('span_warm_1e3','?')}e-3"
                f"   gaps≤{v.get('max_abs_1e3','?')}e-3")
            return run_name, status, summary, tag
        ang = prov.get("angle_deg", "?")
        gl = prov.get("glitch", {})
        excl = gl.get("rows_excluded", gl.get("excluded", "?")) \
            if isinstance(gl, dict) else "?"
        alp = prov.get("alpha_185_215K_1e6")
        if isinstance(alp, dict):
            alp = f"{alp.get('cool','?')}/{alp.get('warm','?')}"
        nlam = len(prov.get("lambda_at_Bmax_by_T", {}) or {})
        run = f"θ={ang:+d}°" if isinstance(ang, int) else f"θ={ang}"
        summary = (f"C0={prov.get('C0_pF','?')}pF   α(185-215) c/w {alp}   "
                   f"{nlam} λ(Bmax)   glitch excl {excl}")
        # mini gates are advisory: flag amber, not a green gate.
        return run, "PRELIM", summary, "suspect"

    def _qc_command(self, cell, prov, prov_path):
        """Subprocess argv to open THIS run's QC in the interactive QC script.
        L0 IS honored here (--thickness). The raw source is taken from the
        provenance JSON — on this Mac that is the comma `_all.csv` archive the
        reduction ran on; QC load_data now sniffs it (P3.5), so a window opens
        without a raw .dat present. --data/--file point the QC script straight at it."""
        script = QC_SCRIPT[cell]
        try:
            l0 = str(float(self.l0_var.get()))
        except (TypeError, ValueError):
            l0 = str(prov.get("L0_cm", "0.058" if cell == "str_dil" else "0.02"))
        src = prov.get("source", "")
        data_dir = os.path.dirname(src) if src else DATA_DIR
        cmd = [self.science_py, script, "--thickness", l0,
               "--out", os.path.dirname(prov_path), "--data", data_dir]
        fname = os.path.basename(src)
        if fname:
            # str takes --file verbatim; mini treats it as a basename substring
            # (the full archive name uniquely selects this angle's archive).
            cmd += ["--file", fname]
        return cmd

    def _set_results_placeholder(self):
        self.tree.delete(*self.tree.get_children())
        self.qc_by_item = {}
        self._base_tags = {}
        self.tree.insert("", "end", iid="_placeholder",
                         values=("", "", "(run a reduction to list its runs here)"),
                         tags=("placeholder",))
        self.open_qc_btn.config(state="disabled")
        self.precleanup_btn.config(state="disabled")

    def _populate_results(self, cell):
        self.tree.delete(*self.tree.get_children())
        self.qc_by_item = {}
        self._base_tags = {}
        self.open_qc_btn.config(state="disabled")
        self.precleanup_btn.config(state="disabled")
        files = self._provenance_files(cell)
        if not files:
            rc = getattr(self, "_last_run_rc", None)
            tail = getattr(self, "_last_run_tail", []) or []
            if rc not in (0, None):
                # The reduction aborted before writing any output. Surface the
                # actual reason (e.g. a missing angle_runs.json, a bad input
                # file) instead of a misleading "provenance missing".
                reason = tail[-1] if tail else "see the log below"
                self.tree.insert("", "end", values=(
                    "", "FAILED",
                    f"reduction did not complete (exit {rc}): {reason}"),
                    tags=("suspect",))
                if len(tail) > 1:
                    for extra in tail[:-1]:
                        self.tree.insert("", "end", values=("", "", extra),
                                         tags=("placeholder",))
            else:
                self.tree.insert("", "end", values=(
                    "", "NO OUTPUT",
                    "the reducer produced no results — see the log below."),
                    tags=("suspect",))
            return
        for path in files:
            try:
                with open(path, encoding="utf-8") as f:
                    prov = json.load(f)
            except Exception as e:  # noqa: BLE001
                run, status, summary, tag = (
                    os.path.basename(path), "ERR", f"unreadable ({e})", "suspect")
                cmd = None
            else:
                run, status, summary, tag = self._summarize(cell, prov, path)
                cmd = self._qc_command(cell, prov, path)
            iid = self.tree.insert("", "end",
                                   values=(run, status, summary), tags=(tag,))
            self._base_tags[iid] = tag
            if cmd is not None:
                self.qc_by_item[iid] = (cmd, path)

    def _on_result_select(self, _evt=None):
        sel = self.tree.selection()
        ok = bool(sel) and sel[0] in self.qc_by_item
        state = "normal" if ok else "disabled"
        self.open_qc_btn.config(state=state)
        self.precleanup_btn.config(state=state)

    def _open_selected_qc(self):
        sel = self.tree.selection()
        if not sel or sel[0] not in self.qc_by_item:
            return
        cmd, path = self.qc_by_item[sel[0]]
        self._open_qc(cmd, path)

    def _plot_selected_precleanup(self):
        """Run the selected run's QC script with --precleanup: save the
        separated+recalculated _Tdep_raw/_Bdep_raw plots without opening QC."""
        sel = self.tree.selection()
        if not sel or sel[0] not in self.qc_by_item:
            return
        if not self.science_py:
            self._log("Cannot Plot pre-cleanup: no science interpreter.")
            return
        cmd, path = self.qc_by_item[sel[0]]
        cmd = list(cmd) + ["--precleanup"]
        self._log(f"\n=== Plot pre-cleanup ({os.path.basename(path)}): "
                  + " ".join(cmd) + " ===")
        threading.Thread(target=self._qc_worker, args=(cmd,),
                         daemon=True).start()

    def _open_qc(self, cmd, prov_path):
        """Launch ONE run's QC as a subprocess ON DEMAND from a results row
        (review #10). The auto-open queue below reuses the SAME per-run command;
        this manual path is unchanged."""
        if not self.science_py:
            self._log("Cannot Open QC: no science interpreter.")
            return
        self._log(f"\n=== Open QC ({os.path.basename(prov_path)}): "
                  + " ".join(cmd) + " ===")
        threading.Thread(target=self._qc_worker, args=(cmd,),
                         daemon=True).start()

    def _qc_worker(self, cmd):
        try:
            proc = subprocess.Popen(
                cmd, cwd=HERE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace", bufsize=1, **POPEN_NOWINDOW)
            for line in proc.stdout:
                self._log("[QC] " + line.rstrip("\n"))
            proc.wait()
            self._log(f"=== QC window closed (exit {proc.returncode}) ===")
        except Exception as e:  # noqa: BLE001
            self._log(f"Open QC error: {e}")

    # ---- auto-open plots after a successful Run (P3.5) ------------------
    def _start_autoopen(self, cell):
        """Called on the main thread after a Run exits 0. If auto-open is ON,
        launch the run's QC — str is a single window, mini a SEQUENTIAL queue
        over its five angle runs. Runs in a worker thread; the mainloop is never
        blocked. No-op (with a log line) when disabled or nothing is openable."""
        if not self.auto_open_var.get():
            self._log("Auto-open is OFF — open QC per row on demand.")
            return
        if not self.science_py:
            self._log("Auto-open: no science interpreter — skipped.")
            return
        # Only this run's freshly (re)written provenance files — the results
        # glob also lists earlier runs, which must NOT auto-open.
        since = getattr(self, "_run_start_time", 0.0) - 1.0
        items = []
        for iid in self.tree.get_children():
            entry = self.qc_by_item.get(iid)
            if entry is None:
                continue
            prov_path = entry[1]
            try:
                if os.path.getmtime(prov_path) >= since:
                    items.append(iid)
            except OSError:
                continue
        if not items:
            self._log("Auto-open: no QC-openable runs from this run.")
            return
        self._skip_event = threading.Event()
        self._queue_active = True
        is_queue = len(items) > 1
        if is_queue:
            # "Skip rest" is visible ONLY while a queue is active.
            self.skip_btn.pack(side="left", padx=(12, 0))
        self._log(f"Auto-open: {len(items)} QC window(s)"
                  + (" — sequential queue; close each to open the next."
                     if is_queue else "."))
        threading.Thread(target=self._autoopen_worker, args=(items,),
                         daemon=True).start()

    def _autoopen_worker(self, items):
        """Worker thread: open each run's QC in turn, waiting for one to exit
        before launching the next. A non-zero exit or a launch failure is one
        log line and the queue continues. 'Skip rest' cancels the remaining
        launches (an already-open window is left alone)."""
        try:
            for iid in items:
                if self._skip_event.is_set():
                    self._log("Auto-open: remaining runs skipped.")
                    break
                entry = self.qc_by_item.get(iid)
                if entry is None:        # results were repopulated meanwhile
                    continue
                cmd, prov_path = entry
                self._ui_q.put(("tree_active", iid))
                self._log(f"\n=== Auto-open QC ({os.path.basename(prov_path)}): "
                          + " ".join(cmd) + " ===")
                try:
                    proc = subprocess.Popen(
                        cmd, cwd=HERE, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace", bufsize=1, **POPEN_NOWINDOW)
                    for line in proc.stdout:
                        self._log("[QC] " + line.rstrip("\n"))
                    proc.wait()
                    if proc.returncode != 0:
                        self._log(f"Auto-open: QC exited {proc.returncode} "
                                  "— queue continues.")
                    else:
                        self._log("=== QC window closed (exit 0) ===")
                except Exception as e:  # noqa: BLE001
                    self._log(f"Auto-open: launch failed ({e}) — queue "
                              "continues.")
        finally:
            self._ui_q.put(("queue_done", None))

    def _skip_queue(self):
        """User clicked 'Skip rest': cancel the remaining launches. The window
        currently open stays open (nothing is killed)."""
        if self._skip_event is not None:
            self._skip_event.set()
        self._log("Skip rest: remaining auto-open runs cancelled "
                  "(any open window stays; nothing killed).")

    def _set_active_row(self, iid):
        """Highlight the run whose QC is currently open; base tag on the rest."""
        for i in self.tree.get_children():
            base = self._base_tags.get(i)
            if i == iid:
                self.tree.item(i, tags=("active",))
            elif base is not None:
                self.tree.item(i, tags=(base,))
        if self.tree.exists(iid):
            self.tree.see(iid)

    def _end_queue(self):
        """Queue finished (or was skipped): hide 'Skip rest', clear highlight."""
        self._queue_active = False
        self.skip_btn.pack_forget()
        for i in self.tree.get_children():
            base = self._base_tags.get(i)
            if base is not None:
                self.tree.item(i, tags=(base,))


def main():
    if sys.platform == "win32":
        # Without this, Tk renders blurry/undersized on scaled-DPI Windows
        # laptops and the screen-size math sees the virtualized resolution.
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:  # noqa: BLE001 — older Windows / no shcore
            pass
    root = tk.Tk()
    DilatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
