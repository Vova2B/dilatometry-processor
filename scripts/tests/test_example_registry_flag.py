"""The shipped calibrations.json is a labelled EXAMPLE, not a silent default:
loaders print a one-shot banner and stamp example_registry into the meta that
reaches every run's _provenance.json. A user-built registry (no marker key —
the builder never writes one) must stay silent and stamp False.
"""
import importlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)

REGISTRY = os.path.join(SCRIPTS, "calibrations.json")


def test_shipped_registry_carries_the_marker():
    reg = json.load(open(REGISTRY, encoding="utf-8"))
    assert reg["_example_registry"]
    assert len(reg["records"]) == 15   # records themselves stay


def _fresh(modname):
    mod = importlib.import_module(modname)
    mod._EXAMPLE_BANNER_SHOWN = False   # one-shot flag: reset per test
    return mod


def test_qc_twins_banner_and_provenance_stamp(capsys):
    for modname in ("qc_mini_cell", "qc_str_cell"):
        mod = _fresh(modname)
        cal = mod.load_calibration(sample_thickness_cm=0.1)
        out = capsys.readouterr().out
        assert "SHIPPED EXAMPLE calibration registry" in out
        assert cal["meta"]["example_registry"] is True
        # one-shot: a second load in the same process stays quiet
        mod.load_calibration(sample_thickness_cm=0.1)
        assert "SHIPPED EXAMPLE" not in capsys.readouterr().out


def test_user_registry_is_silent_and_stamps_false(tmp_path, capsys):
    reg = json.load(open(REGISTRY, encoding="utf-8"))
    del reg["_example_registry"]       # what a user-built registry looks like
    own = tmp_path / "calibrations.json"
    own.write_text(json.dumps(reg), encoding="utf-8")
    mod = _fresh("qc_mini_cell")
    cal = mod.load_calibration(sample_thickness_cm=0.1, path=str(own))
    assert "SHIPPED EXAMPLE" not in capsys.readouterr().out
    assert cal["meta"]["example_registry"] is False


def test_bridge_and_polar_loaders_banner(capsys):
    bridge = _fresh("calibration_bridge")
    bridge.registry_cell_fit(cell="str_dil", branch="warm",
                             sample_thickness_cm=0.042, verbose=False)
    assert "SHIPPED EXAMPLE calibration registry" in capsys.readouterr().out
    polar = _fresh("polar_figures")
    polar.load_cooling_calibration(0.1)
    assert "SHIPPED EXAMPLE calibration registry" in capsys.readouterr().out
