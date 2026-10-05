"""
pyreq.py — the supported-Python floor, stated once.

Imported by every entry point BEFORE numpy/pandas/matplotlib, so that an
unsupported interpreter is reported in words instead of failing later with an
error that points at the wrong thing. The motivating case: Python 3.8 has no
`argparse.BooleanOptionalAction` (added in 3.9), so both reducers used to die
with `AttributeError` while building their parser — before reading a single
argument, and even for `--help`. The message blamed argparse; the cause was
the interpreter.

Stdlib-only, and `sys` is the only import it is allowed to grow: on an old or
incomplete interpreter this module has to work when nothing else does. Its
messages stay plain ASCII because a Windows console on a non-UTF-8 codepage
cannot print the glyphs the rest of the tool uses.
"""

import sys

# The floor the README promises and CI enforces. 3.9 would be enough for
# argparse alone, but numpy >= 1.24 (requirements.txt) needs 3.9+ and the
# tested matrix is 3.10 and 3.13 — so 3.10 is the honest number.
MIN_PYTHON = (3, 10)

EXIT_UNSUPPORTED_PYTHON = 3   # distinct from 0 (ok) and 1 (a gate failed)


def version_text(info=None):
    """'3.8.20' for a version tuple or sys.version_info."""
    info = tuple(sys.version_info[:3] if info is None else info)[:3]
    return ".".join(str(n) for n in info)


def is_supported(info=None):
    """True when `info` (default: this interpreter) meets MIN_PYTHON."""
    info = sys.version_info[:3] if info is None else info
    return tuple(info)[:len(MIN_PYTHON)] >= MIN_PYTHON


def message(info=None, program=None):
    """The whole explanation, in the words a user needs to act on."""
    have = version_text(info)
    need = ".".join(str(n) for n in MIN_PYTHON)
    who = program or "This tool"
    return (
        "\n"
        "%s needs Python %s or newer, but this is Python %s.\n"
        "\n"
        "This is not a problem with your data or your measurement. Nothing\n"
        "was read and nothing was written.\n"
        "\n"
        "What to do, easiest first:\n"
        "  1. Use the portable bundle. It carries its own Python and all the\n"
        "     libraries inside it, so there is nothing to install, it needs no\n"
        "     internet, and it will not disturb the Python that your other\n"
        "     instrument software depends on.\n"
        "  2. Or install Python %s or newer from python.org and run the tool\n"
        "     with that interpreter instead.\n"
        "\n"
        "If you are launching from the app rather than the command line, the\n"
        "app picks the interpreter for you -- point it at the newer one, or\n"
        "use the bundle, which already has it.\n"
        % (who, need, have, need)
    )


def require(program=None, info=None, stream=None):
    """Exit with a readable explanation when the interpreter is too old.

    Returns None on a supported interpreter so callers can simply call it.
    """
    if is_supported(info):
        return None
    (stream or sys.stderr).write(message(info, program))
    raise SystemExit(EXIT_UNSUPPORTED_PYTHON)


def probe_code():
    """Source for `<interpreter> -c ...` that answers: can this one run the
    science workers? Exits 0 when yes, non-zero when the version is below the
    floor or the science stack is missing. Used by the app to choose an
    interpreter, so that a Python carrying numpy and pandas but too old to run
    the reducers is rejected up front rather than at Run time."""
    return (
        "import sys\n"
        "if sys.version_info[:2] < %r:\n"
        "    sys.stdout.write('.'.join(str(n) for n in sys.version_info[:3]))\n"
        "    sys.exit(%d)\n"
        "import numpy, pandas\n"
        % (MIN_PYTHON, EXIT_UNSUPPORTED_PYTHON)
    )
