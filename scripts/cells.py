"""
cells.py — per-cell reduction parameters for the US dilatometry pipeline.

Minimal P1 version (2026-07-03): just the parameter DATA the reduction needs,
so cell-specific numbers live in one place instead of being scattered as
module constants. Detection/rescale wiring is P2 and deliberately absent here.

Each entry:
  registry_key            calibrations.json cell key (load_calibration cell=)
  cmax                    None -> no Cmax correction (ratio is identically 1.0);
                          dict{K, cmax_true, cmax_ppms} -> apply the mini-cell
                          true-Cmax correction (reduce.cmax_ratio / estimate_C0)
  field_change_threshold  separate_data threshold used by the BATCH driver and
                          the new tool (|dB| between rows, in 1e-4 T units)
  field_change_threshold_interactive
                          value the legacy interactive main() passes; kept
                          only to preserve current behavior (see note below)
  plate_K_ref             Pott-Schefzyk plate constant K [1e-6 cm * pF] of the
                          cell's TRUE geometry, used only by detect.py to (a)
                          classify a file's fitted r_eff and (b) compute the
                          per-file rescale factor for a mini run mis-converted
                          with the PPMS 7 mm default (factor = plate_K_ref /
                          K_fitted). Not used by the reduction path.
  C_hint_pF               typical working capacitance of the US runs in this
                          cell; a NON-decisive tie-breaker for the detection
                          dropdown default when r_eff sits in the ambiguous
                          ~7 mm band (str US ~7.5 pF, mini ~18.5 pF). Never
                          used to gate a decision — only to pre-select.

detection reference values (Pott-Schefzyk fits, plate_constant_audit.py,
2026-07-03): str/PPMS 7 mm-default plate r_eff ~= 7.02-7.03 mm (K ~= 137 200);
mini true plate r_eff ~= 4.87 mm (K ~= 65 946, mean of the Cu_1mm / Cu_2mm
mini calibration fits: 66 062, 65 831). A sample run that fits r_eff
~= 6.8-7.0 mm on the MINI cell was PPMS-converted with the 7 mm default
radius; plate_K_ref / K_fitted then gives the delta_l rescale to repair it
(supply it per run via angle_runs.json "rescale").

field_change_threshold discrepancy (review #4, investigated 2026-07-03):
  str batch = mini batch = 100; mini interactive main() = 3. On the real mini
  data the Field_Change distribution has a clean empty gap (dwell noise
  <=0.12, real ramps >=~1012 in 1e-4 T units), so only ~3 ramp-boundary rows
  per run fall in 3<FC<=100, and separate_data's min_points=25 grouping
  discards those strays -> segmentation is effectively identical for 3 vs 100.
  RECOMMENDATION: 100 (physically a 0.01 T/step ramp detector; 3 = 0.3 mT is
  below field-stability noise and only catches ramp edges). Migrate the mini
  interactive main() to 100 in a later phase; NOT changed in P1 because the
  regression gate pins current behavior. Batch already uses 100.
"""

STR_DIL = {
    "registry_key": "str_dil",
    "cmax": None,                       # C ~ 7.5 pF; Cmax error <2%, not corrected
    "field_change_threshold": 100,
    "field_change_threshold_interactive": 100,
    "plate_K_ref": 137_200.0,           # r_eff ~ 7.02 mm (PPMS default plate)
    "C_hint_pF": 7.5,
}

MINI_DIL = {
    "registry_key": "mini_dil",
    "cmax": {"K": 136_300.0, "cmax_true": 50.0, "cmax_ppms": 100.0},
    "field_change_threshold": 100,      # batch / new-tool value (canonical)
    "field_change_threshold_interactive": 3,   # legacy interactive main() value
    "plate_K_ref": 65_946.0,            # r_eff ~ 4.87 mm (Cu mini calibration)
    "C_hint_pF": 18.5,
}

BY_KEY = {STR_DIL["registry_key"]: STR_DIL,
          MINI_DIL["registry_key"]: MINI_DIL}
