"""The bundle zip is what a user on a blocked machine actually downloads, so
it has to be compressed — and it has to keep the launcher executable."""
import os
import stat
import sys

import pytest
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import make_portable_bundle as mpb  # noqa: E402


def _make_bundle(tmp_path):
    bundle = tmp_path / "DilatometryProcessor-test"
    (bundle / "scripts").mkdir(parents=True)
    # highly compressible, like the .py/.pyi/.txt bulk of a real bundle
    (bundle / "scripts" / "big.py").write_bytes(b"x = 1\n" * 40000)
    launcher = bundle / "run_app.command"
    launcher.write_text("#!/bin/sh\necho hi\n")
    launcher.chmod(0o755)
    return bundle


def test_every_entry_is_deflated(tmp_path):
    """ZipFile(compression=) is ignored by writestr() when a ZipInfo is
    passed: the ZipInfo's own compress_type wins, and from_file() leaves it
    STORED. That silently tripled the download."""
    bundle = _make_bundle(tmp_path)
    zpath = str(bundle) + ".zip"
    mpb.write_bundle_zip(str(bundle), zpath, str(tmp_path))

    with zipfile.ZipFile(zpath) as z:
        kinds = {i.compress_type for i in z.infolist()}
    assert kinds == {zipfile.ZIP_DEFLATED}, f"not all deflated: {kinds}"


def test_compression_actually_shrinks_the_payload(tmp_path):
    bundle = _make_bundle(tmp_path)
    zpath = str(bundle) + ".zip"
    mpb.write_bundle_zip(str(bundle), zpath, str(tmp_path))

    raw = sum(os.path.getsize(os.path.join(r, f))
              for r, _d, fs in os.walk(str(bundle)) for f in fs)
    assert os.path.getsize(zpath) < raw / 2


@pytest.mark.skipif(os.name == "nt", reason="Windows has no POSIX mode bits: "
                    "chmod(0o755) is a no-op there, so this property cannot "
                    "hold. The bundles are built on macOS, where it must.")
def test_launcher_keeps_its_executable_bit(tmp_path):
    """The reason the writer uses ZipInfo at all — do not regress it while
    fixing the compression."""
    bundle = _make_bundle(tmp_path)
    zpath = str(bundle) + ".zip"
    mpb.write_bundle_zip(str(bundle), zpath, str(tmp_path))

    with zipfile.ZipFile(zpath) as z:
        info = next(i for i in z.infolist()
                    if i.filename.endswith("run_app.command"))
    assert (info.external_attr >> 16) & stat.S_IXUSR


def test_paths_are_relative_to_the_bundles_root(tmp_path):
    bundle = _make_bundle(tmp_path)
    zpath = str(bundle) + ".zip"
    mpb.write_bundle_zip(str(bundle), zpath, str(tmp_path))

    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
    assert all(n.startswith("DilatometryProcessor-test/") for n in names)
    assert not any(n.startswith("/") or ".." in n for n in names)
