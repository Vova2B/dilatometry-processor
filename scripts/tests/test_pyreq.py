"""The interpreter floor must be stated once and enforced before any
third-party import, so an unsupported Python produces an explanation instead
of an AttributeError from inside argparse."""
import os
import re
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyreq  # noqa: E402

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REDUCERS = ("reduce_str_batch.py", "reduce_mini_batch.py")


def test_floor_matches_what_the_readme_promises():
    assert pyreq.MIN_PYTHON == (3, 10)
    assert all(isinstance(n, int) for n in pyreq.MIN_PYTHON)


@pytest.mark.parametrize("info,ok", [
    ((3, 7, 17), False),
    ((3, 8, 20), False),   # the PPMS computer: argparse has no
    ((3, 9, 25), False),   # BooleanOptionalAction below 3.9, and we need 3.10
    ((3, 10, 0), True),
    ((3, 10, 19), True),
    ((3, 13, 1), True),
    ((4, 0, 0), True),
])
def test_is_supported_draws_the_line_at_the_floor(info, ok):
    assert pyreq.is_supported(info) is ok


def test_is_supported_defaults_to_the_running_interpreter():
    assert pyreq.is_supported() is True  # the suite runs on a supported Python


def test_message_names_both_versions_and_what_to_do():
    msg = pyreq.message((3, 8, 20), program="reduce_str_batch.py")
    assert "3.8.20" in msg          # what they have
    assert "3.10" in msg            # what they need
    assert "reduce_str_batch.py" in msg
    # actionable, not just a complaint
    assert re.search(r"portable bundle", msg, re.I)


def test_message_is_plain_ascii_so_a_windows_codepage_can_print_it():
    msg = pyreq.message((3, 8, 20), program="x.py")
    msg.encode("ascii")             # raises if a glyph sneaks in


def test_require_is_silent_and_returns_none_on_a_supported_python(capsys):
    assert pyreq.require("x.py", info=(3, 10, 0)) is None
    assert capsys.readouterr().err == ""


def test_require_exits_3_with_the_message_on_stderr(capsys):
    with pytest.raises(SystemExit) as e:
        pyreq.require("x.py", info=(3, 8, 20))
    assert e.value.code == 3
    assert "3.8.20" in capsys.readouterr().err


def test_pyreq_imports_nothing_but_the_standard_library():
    """It is imported before numpy exists, so it may not need numpy."""
    src = open(os.path.join(SCRIPTS, "pyreq.py"), encoding="utf-8").read()
    imported = set(re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)",
                              src, re.M))
    assert imported <= {"sys"}, f"pyreq.py must stay stdlib-only, got {imported}"


@pytest.mark.parametrize("name", REDUCERS)
def test_reducers_guard_before_importing_numpy(name):
    """The guard is worthless if numpy is imported first: on an old
    interpreter the numpy import is what fails, with an unrelated error."""
    src = open(os.path.join(SCRIPTS, name), encoding="utf-8").read()
    guard = src.index("pyreq.require(")
    numpy = src.index("import numpy")
    assert guard < numpy, f"{name} imports numpy before calling pyreq.require()"


@pytest.mark.parametrize("name", REDUCERS)
def test_reducers_name_themselves_in_the_guard(name):
    src = open(os.path.join(SCRIPTS, name), encoding="utf-8").read()
    call = src[src.index("pyreq.require("):][:120]
    assert name in call


def test_probe_code_accepts_the_running_interpreter():
    r = subprocess.run([sys.executable, "-c", pyreq.probe_code()],
                       capture_output=True, timeout=60)
    assert r.returncode == 0, r.stderr.decode()


def test_probe_code_checks_the_version_and_the_science_stack():
    code = pyreq.probe_code()
    assert "numpy" in code and "pandas" in code
    assert "version_info" in code
    compile(code, "<probe>", "exec")
