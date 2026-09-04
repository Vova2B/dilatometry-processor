"""
samples.py — per-sample parameter registry loader for the dilatometry GUI.

P3 (2026-07-03, plan docs/plans/2026-07-02-unified-dilat-gui.md, review #8).
Pure stdlib (json/os) so dilat_app.py can import it under the GUI interpreter
without the science stack. samples.json holds one entry per sample with the
few things the pipeline cannot derive from the raw file: candidate L0 values
(with provenance + a confirmed flag), the transition window to PROTECT during
cleanup, the default cell, per-cell cleanup overrides, and notes.

Schema rules (review #8, all mandatory):
  * top-level "schema": 1 (SCHEMA below is the highest this loader understands);
  * absent per-sample keys are filled with documented defaults on load;
  * a file whose "schema" is GREATER than SCHEMA is REFUSED with a clear
    message and no partial data (SamplesError) — never silently misread newer
    fields;
  * write-back re-dumps the loaded dict (reg["_raw"]) verbatim, so fields this
    loader does not know about survive a load -> save round-trip;
  * location = next to this file (scripts/samples.json), overridable via the
    DILAT_SAMPLES environment variable;
  * a MISSING file is not an error — the tool still runs with manual entry
    (empty registry), per the samples.json <-> everything seam contract.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Highest schema version this loader understands. Bump when a NEW required
# field is added that older code cannot safely ignore.
SCHEMA = 1

# Documented defaults applied for absent per-sample keys. display_name defaults
# to the sample key (handled in _merge_defaults, not here, since it is dynamic).
DEFAULTS = {
    "L0_cm": [],                 # list of {value, cell, source, confirmed}
    "transition_window_K": None, # [lo, hi] protect window, or None
    "default_cell": "str_dil",
    "cleanup_overrides": {},     # {cell_key: {toggle: bool, ...}}
    "notes": "",
}


class SamplesError(Exception):
    """Raised when samples.json cannot be trusted (e.g. newer schema)."""


def samples_path():
    """The one documented location, DILAT_SAMPLES env var takes precedence."""
    return os.environ.get("DILAT_SAMPLES") or os.path.join(HERE, "samples.json")


def _merge_defaults(name, entry):
    """Return a copy of entry with defaults filled AND unknown keys preserved."""
    merged = dict(entry)   # keep every field the file carried, known or not
    merged.setdefault("display_name", name)
    for key, default in DEFAULTS.items():
        if key not in merged:
            # copy mutable defaults so callers cannot mutate the shared literal
            merged[key] = default.copy() if isinstance(default, (list, dict)) \
                else default
    return merged


def load_samples(path=None):
    """Load the sample registry.

    Returns a dict:
      {"schema": <file schema>, "samples": {name: merged_entry, ...},
       "_raw": <verbatim loaded dict>, "_path": <resolved path>}

    Missing file -> empty registry (tool runs with manual entry).
    Schema newer than SCHEMA -> raises SamplesError (no partial run).
    """
    path = path or samples_path()
    if not os.path.exists(path):
        raw = {"schema": SCHEMA, "samples": {}}
        return {"schema": SCHEMA, "samples": {}, "_raw": raw, "_path": path}

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    file_schema = raw.get("schema", 1)
    if file_schema > SCHEMA:
        raise SamplesError(
            f"{path}: samples.json schema is {file_schema}, but this tool only "
            f"understands schema {SCHEMA}. Refusing to run so newer fields are "
            "not silently misread — update the tool.")

    samples = {name: _merge_defaults(name, entry)
               for name, entry in raw.get("samples", {}).items()}
    return {"schema": file_schema, "samples": samples,
            "_raw": raw, "_path": path}


def save_samples(reg, path=None):
    """Write the registry back by re-dumping reg["_raw"] verbatim, so unknown
    fields survive. Programmatic edits should mutate reg["_raw"]["samples"]
    (use update_sample) rather than the defaults-merged reg["samples"]."""
    path = path or reg.get("_path") or samples_path()
    raw = reg.get("_raw", {"schema": SCHEMA, "samples": {}})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)
        f.write("\n")
    return path


def update_sample(reg, name, key, value):
    """Set one field on a sample in BOTH the merged view and the raw dict, so a
    later save_samples persists it (and leaves every other field untouched)."""
    reg["_raw"].setdefault("samples", {}).setdefault(name, {})[key] = value
    if name in reg["samples"]:
        reg["samples"][name][key] = value


def l0_for_cell(entry, cell_key):
    """Best L0 option for a given cell: the first L0_cm entry whose "cell"
    matches, else the first option, else None. Returns the option dict."""
    opts = entry.get("L0_cm") or []
    for o in opts:
        if o.get("cell") == cell_key:
            return o
    return opts[0] if opts else None
