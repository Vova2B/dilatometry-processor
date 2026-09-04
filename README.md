# Capacitance Dilatometry Processor

*for Quantum Design PPMS® systems*

Reduction, quality control, and plotting for **capacitance dilatometry** data
measured in a Quantum Design PPMS with Küchler-type BeCu dilatometers
(standard cell: RSI **83**, 095102 (2012); uniaxial-stress cell: RSI **87**,
073903 (2016); mini cell: RSI **88**, 083903 (2017)).

From a raw PPMS `.dat` export it produces referenced thermal expansion
ΔL/L₀(T), magnetostriction ΔL(B) loops, the linear expansion coefficient
α(T), angle-resolved (polar) anisotropy figures, and a per-run provenance
JSON with physical quality gates.

## Install

Python ≥ 3.10 with the standard scientific stack:

```bash
pip install -r scripts/requirements.txt
```

Tkinter (bundled with python.org installers; `python3-tk` on Debian/Ubuntu)
is needed only for the GUI launcher.

The full user guide — install variants (incl. fully offline machines), every
workflow, troubleshooting — is `docs/guide.html` (open in any browser).

## Quick start

**GUI** — pick a file, detect the cell, run reduction, inspect results:

```bash
python3 scripts/dilat_app.py
```

**Standard/stress-cell run (single file):**

```bash
python3 scripts/reduce_str_batch.py --data /path/to/folder --file myrun.dat \
        --L0 0.058 --transition 100
```

`--L0` is the sample thickness in cm; `--transition` (optional) draws the
transition line and splits the ferro/para magnetostriction panels. Outputs
land in `Output/str/<input-stem>_*` (CSV + PNG + `_provenance.json` with the
gate results).

**Mini-cell rotation series (multi-angle):** describe your angle files once in
an `angle_runs.json` next to the data (required — there is no built-in run
list):

```json
{"stem": "MYSAMPLE_mini",
 "L0_cm": 0.02,
 "transition_K": 100.0,
 "runs": [{"angle_deg": 0,   "tag": "rot0",   "glob": "*rot0*.dat"},
          {"angle_deg": 45,  "tag": "plus45", "glob": "*plus45*.dat"},
          {"angle_deg": -45, "tag": "minus45","glob": "*minus45*.dat"}]}
```

```bash
python3 scripts/reduce_mini_batch.py --data /path/to/folder
```

Per-angle outputs plus combined overlay and polar-anisotropy figures are
written to `Output/mini/<stem>_*`. An optional per-run `"rescale"` factor
corrects raw δl converted with the wrong plate radius.

**Interactive QC** (trim, smooth, exclude curves, re-export) opens from the
GUI's results table, or directly:

```bash
python3 scripts/qc_str_cell.py --data /path/to/folder --file myrun.dat
```

## Calibration — bring your own cell

The empty-cell (Cu) background is read from `scripts/calibrations.json`.
**The shipped registry is a labelled example — the authors' dilatometers,
not yours.** Every script that loads it prints a banner and stamps
`example_registry: true` into the run's provenance JSON until you replace it.

Build your own from empty-cell Cu reference runs. Describe the runs once in
a `cu_runs.json` next to your Cu `.dat` files (same convention as
`angle_runs.json` above; full field reference in the module docstring, and
`scripts/cu_runs.example.json` is the worked example that produced the
shipped registry):

```json
{"runs": [
  {"key": "mycell_1mm", "path": "Cu_1mm_run.dat",
   "cell": "my_cell", "cu_length_mm": 1.0},
  {"key": "mycell_2mm", "path": "Cu_2mm_run.dat",
   "cell": "my_cell", "cu_length_mm": 2.0}]}
```

```bash
python3 scripts/cu_calibration_builder.py --data /path/to/cu/folder
```

This segments cool/warm branches, repairs offset steps, fits the per-branch
polynomial backgrounds, runs the round-trip gate (each Cu run reduced with
its own calibration must return Cu literature), writes QC figures to
`fig_calibration_QC/`, and saves the registry to `scripts/calibrations.json`
(or `--out`; point `DILAT_CALIBRATIONS` at it to keep several). Two Cu
lengths per cell enable the Eq.-(7) thickness-matched virtual curves;
optional `transfer_pairs` / `eq7_pairs` / `hysteresis_pairs` lists and a
`kind: "field"` run (field-background envelope) are described in
`cu_calibration_builder.py --help` and its module docstring. Per-branch
manual repairs (exclusion windows, forced step rows, `use: false`) go in
`scripts/calibration_config.json`, keyed `<key>/c<cycle><w|c>`.

The registry stores branch-aware (cool/warm) polynomial backgrounds, the
P18 Eq. (7) length decomposition for thickness-matched virtual curves, and a
field-background envelope. Selection at load time prefers an Eq.-(7) virtual
curve at your sample thickness when its fitted T-range covers the run
(≤ 5 K overhang tolerated), falling back to the closest-length record
otherwise — the choice is recorded in each run's provenance JSON.

## Layout

```
scripts/
  dilat_app.py                 Tkinter launcher (detect → reduce → QC)
  reduce_str_batch.py          headless reduction, standard/stress cell
  reduce_mini_batch.py         headless reduction, mini cell rotation series
  qc_str_cell.py               interactive QC, standard/stress cell
  qc_mini_cell.py              interactive QC, mini cell
  polar_figures.py             standalone polar/anisotropy figures
  reduce.py, cleanup.py, cells.py, detect.py, samples.py   shared core
  cu_calibration_builder.py    build calibrations.json from Cu runs
  calibration_bridge.py        minimal calibrations.json reader for any script
  plate_constant_audit.py      plate-constant audit (wrong-radius detector)
  calibrations.json            cell-background registry (EXAMPLE — see above)
  cu_runs.example.json         worked cu_runs.json (the runs behind the
                               shipped registry)
  samples.json                 sample registry (ships one EXAMPLE entry —
                               add your samples: T_C window, L0 hints)
```

**Convention:** `qc_str_cell.py` and `qc_mini_cell.py` are deliberate
standalone twins — no shared QC module. Any change to their shared logic
(QC window, plotting, calibration loading) must be replicated in both;
parity is part of review.

## Units

T in K; B in T; raw δl in 10⁻⁶ cm; sample length L₀ in cm inside the code
(mm in the GUI); ΔL/L₀ dimensionless (plots ×10⁻³); α in 10⁻⁶ K⁻¹.

## License and citation

Licensed under the **MIT License** (see `LICENSE`).
If this software contributes to a publication, cite it (see `CITATION.cff`)
together with the Küchler dilatometer papers above.
