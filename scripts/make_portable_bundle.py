#!/usr/bin/env python3
"""make_portable_bundle.py — build a copy-anywhere app folder (Python inside).

    python3 make_portable_bundle.py --target macos-arm64
    python3 make_portable_bundle.py --target windows-x64
    ... [--zip]

Produces  <share>/3_portable_bundles/DilatometryProcessor-<target>/ :

    run_app.command / run_app.bat   double-clickable launcher
    README.txt                      first-run notes (Gatekeeper / SmartScreen)
    python/                         standalone CPython (astral-sh
                                    python-build-standalone, install_only) with
                                    numpy/pandas/scipy/matplotlib installed
    scripts/                        the same file set sync_export.py ships

The bundle is per-CPU-architecture: macos-arm64 covers Apple Silicon only;
windows-x64 covers 64-bit Intel/AMD Windows. The Windows bundle is assembled
on any host via `pip install --platform win_amd64 --target` (wheels only) —
verify it on a real Windows machine before handing it out.

Maintainer tool — deliberately in sync_export.py's SKIP set, so it is never
shipped in the offline copy or the update zip.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SHARE = os.path.dirname(os.path.dirname(HERE))       # .../for opensourcing and sharing
BUNDLES = os.path.join(SHARE, "3_portable_bundles")
CACHE = os.path.join(BUNDLES, "_download_cache")

# Pinned interpreter build (https://github.com/astral-sh/python-build-standalone)
PBS_TAG = "20260623"
PY_VER = "3.12.13"
TARGETS = {
    "macos-arm64":  f"cpython-{PY_VER}+{PBS_TAG}-aarch64-apple-darwin-install_only.tar.gz",
    "windows-x64":  f"cpython-{PY_VER}+{PBS_TAG}-x86_64-pc-windows-msvc-install_only.tar.gz",
}
PBS_URL = ("https://github.com/astral-sh/python-build-standalone/releases/"
           "download/{tag}/{asset}")

README = """Dilatometry Processor — portable bundle ({target})
=====================================================

Version:  {version}
Built:    {build_date}

Nothing to install: this folder carries its own Python and libraries.

Licensing
---------
The processor itself is MIT-licensed (LICENSE in this folder). The bundled
CPython interpreter and scientific libraries keep their own licences —
see THIRD-PARTY-LICENSES.txt.

Start the app
-------------
macOS:    double-click run_app.command
Windows:  double-click run_app.bat

First-run notes
---------------
* macOS Gatekeeper: if the folder arrived by web download / email, macOS
  quarantines it and the first launch is blocked. Fix once, in Terminal:
      xattr -dr com.apple.quarantine "<path to this folder>"
  (AirDrop / USB-stick copies are not quarantined.)
* macOS may also ask to allow running a program from an unidentified
  developer: right-click run_app.command -> Open -> Open.
* Windows SmartScreen may warn on first run: More info -> Run anyway.
* Windows, app does not start or errors: run run_app_debug.bat instead —
  it keeps a console window open with the full error text to report.

Architecture
------------
macos-arm64 runs on Apple Silicon (M1 and later) only.
windows-x64 runs on 64-bit Intel/AMD Windows 10/11.

Updating
--------
Replace the scripts/ subfolder with a newer dilat_code_update_*.zip's
contents; python/ can stay as is.
"""


def _fetch(asset):
    os.makedirs(CACHE, exist_ok=True)
    dest = os.path.join(CACHE, asset)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"cached: {asset}")
        return dest
    url = PBS_URL.format(tag=PBS_TAG, asset=asset)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest + ".part")
    os.replace(dest + ".part", dest)
    return dest


def _ship_files():
    """Same ship set as sync_export.py (its SKIP + extension rules)."""
    sys.path.insert(0, HERE)
    import sync_export
    skip = set(sync_export.SKIP) | {os.path.basename(__file__)}
    for name in sorted(os.listdir(HERE)):
        if name in skip or name.startswith("."):
            continue
        if name.endswith(sync_export.COPY_EXT):
            yield name


def _run(cmd, **kw):
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def _version_stamp():
    """Repo state at build time: git describe when available (tags first,
    else short hash, -dirty when uncommitted), else 'unversioned'."""
    try:
        p = subprocess.run(["git", "describe", "--tags", "--always",
                            "--dirty"], cwd=HERE, capture_output=True,
                           text=True)
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip()
    except OSError:
        pass
    return "unversioned"


def _write_third_party_licenses(bundle, pydir, out_path=None):
    """THIRD-PARTY-LICENSES.txt at the bundle root: a pointer to every
    licence file pip installed under site-packages/*.dist-info/ (they ship
    with the wheels — only this top-level index is new) plus CPython's own
    licence. Returns the number of licence files indexed. out_path lets a
    dry-run write the index elsewhere without touching a built bundle."""
    site = None
    for cand in (os.path.join(pydir, "lib", "python%s.%s" %
                              tuple(PY_VER.split(".")[:2]), "site-packages"),
                 os.path.join(pydir, "Lib", "site-packages")):
        if os.path.isdir(cand):
            site = cand
            break
    if site is None:
        raise SystemExit(f"no site-packages under {pydir} — cannot index "
                         "third-party licences")
    entries = []
    for di in sorted(os.listdir(site)):
        if not di.endswith(".dist-info"):
            continue
        hits = []
        for root, _dirs, files in os.walk(os.path.join(site, di)):
            for fn in sorted(files):
                up = fn.upper()
                if ("LICENSE" in up or "LICENCE" in up or "COPYING" in up
                        or up.startswith("AUTHORS") or up.startswith("NOTICE")):
                    hits.append(os.path.relpath(os.path.join(root, fn),
                                                bundle).replace(os.sep, "/"))
        if hits:
            entries.append((di[:-len(".dist-info")], hits))
    n = sum(len(h) for _p, h in entries)
    py_lic = None      # python-build-standalone: lib/python3.12/LICENSE.txt
    for cand in (os.path.join(os.path.dirname(site), "LICENSE.txt"),
                 os.path.join(pydir, "LICENSE.txt"),
                 os.path.join(pydir, "Lib", "LICENSE.txt")):
        if os.path.isfile(cand):
            py_lic = cand
            break
    out = out_path or os.path.join(bundle, "THIRD-PARTY-LICENSES.txt")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("Third-party licences in this bundle\n")
        f.write("===================================\n\n")
        f.write("The Dilatometry Processor scripts are MIT-licensed (see "
                "LICENSE).\nThis bundle additionally redistributes, "
                "unmodified:\n\n")
        f.write("* CPython %s (PSF License Agreement): %s\n\n" % (
            PY_VER,
            os.path.relpath(py_lic, bundle).replace(os.sep, "/")
            if py_lic else
            "python/ (see https://docs.python.org/3/license.html)"))
        f.write("* the Python packages below, each under its own licence,\n"
                "  with the licence texts pip installed alongside them:\n\n")
        for pkg, hits in entries:
            f.write("  %s\n" % pkg)
            for h in hits:
                f.write("    %s\n" % h)
        f.write("\n(%d licence files across %d packages)\n"
                % (n, len(entries)))
    print(f"third-party licences: {out} ({n} files, {len(entries)} packages)")
    return n


def build(target, make_zip):
    asset = TARGETS[target]
    bundle = os.path.join(BUNDLES, f"DilatometryProcessor-{target}")
    if os.path.exists(bundle):
        raise SystemExit(f"refusing to overwrite existing bundle: {bundle}\n"
                         "Move it away first (never deleted automatically).")
    os.makedirs(bundle)

    # 1. interpreter --------------------------------------------------------
    tar = _fetch(asset)
    print(f"extracting -> {bundle}/python")
    with tarfile.open(tar) as tf:
        tf.extractall(bundle)          # archives carry a top-level 'python/'
    pydir = os.path.join(bundle, "python")
    if not os.path.isdir(pydir):
        raise SystemExit("unexpected archive layout: no top-level python/ dir")

    # 2. science libraries --------------------------------------------------
    req = os.path.join(HERE, "requirements.txt")
    if target == "macos-arm64":
        bpy = os.path.join(pydir, "bin", "python3")
        _run([bpy, "-m", "pip", "install", "--no-warn-script-location",
              "-r", req])
        _run([bpy, "-c",
              "import tkinter, numpy, pandas, scipy, matplotlib; "
              "print('bundle imports OK', numpy.__version__)"])
    else:  # windows-x64 — cross-assembled: unpack win_amd64 wheels from here
        site = os.path.join(pydir, "Lib", "site-packages")
        _run([sys.executable, "-m", "pip", "install",
              "--platform", "win_amd64", "--implementation", "cp",
              "--python-version", PY_VER, "--only-binary=:all:",
              "--target", site, "-r", req])

    # 3. app scripts ---------------------------------------------------------
    sdir = os.path.join(bundle, "scripts")
    os.makedirs(sdir)
    names = list(_ship_files())
    for name in names:
        shutil.copy2(os.path.join(HERE, name), os.path.join(sdir, name))
    print(f"scripts: {len(names)} files")

    # 4. launcher + readme ---------------------------------------------------
    if target == "macos-arm64":
        launcher = os.path.join(bundle, "run_app.command")
        with open(launcher, "w", encoding="utf-8", newline="\n") as f:
            f.write('#!/bin/bash\ncd "$(dirname "$0")"\n'
                    'exec ./python/bin/python3 scripts/dilat_app.py\n')
        os.chmod(launcher, 0o755)
    else:
        # newline="\r\n" translates the \n's — do NOT also write literal \r\n
        # (that produced \r\r\n lines on the first build).
        # PYTHONUTF8=1 (PEP 540): force UTF-8 for EVERY file/pipe/console
        # encoding in the app and all its worker subprocesses (they inherit
        # the env) — Windows otherwise defaults to the ANSI codepage, which
        # cannot encode the Greek/degree glyphs this tool prints and reads.
        launcher = os.path.join(bundle, "run_app.bat")
        with open(launcher, "w", encoding="utf-8", newline="\r\n") as f:
            # pythonw = no console window; dilat_app logs into its own pane
            f.write('@echo off\ncd /d "%~dp0"\nset PYTHONUTF8=1\n'
                    'start "" "%~dp0python\\pythonw.exe" '
                    '"%~dp0scripts\\dilat_app.py"\n')
        debug = os.path.join(bundle, "run_app_debug.bat")
        with open(debug, "w", encoding="utf-8", newline="\r\n") as f:
            # console stays open -> startup tracebacks are visible/copyable
            f.write('@echo off\ncd /d "%~dp0"\nset PYTHONUTF8=1\n'
                    '"%~dp0python\\python.exe" "%~dp0scripts\\dilat_app.py"\n'
                    'pause\n')
    shutil.copy2(os.path.join(os.path.dirname(HERE), "LICENSE"),
                 os.path.join(bundle, "LICENSE"))
    _write_third_party_licenses(bundle, pydir)
    with open(os.path.join(bundle, "README.txt"), "w", encoding="utf-8") as f:
        f.write(README.format(target=target, version=_version_stamp(),
                              build_date=__import__("datetime").date.today()
                              .isoformat()))

    # 5. zip -----------------------------------------------------------------
    if make_zip:
        zpath = bundle + ".zip"
        print(f"zipping -> {zpath} (takes a minute)")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(bundle):
                for fn in files:
                    p = os.path.join(root, fn)
                    arc = os.path.relpath(p, BUNDLES).replace(os.sep, "/")
                    zi = zipfile.ZipInfo.from_file(p, arc)
                    zi.external_attr = (os.stat(p).st_mode & 0xFFFF) << 16
                    with open(p, "rb") as src:
                        zf.writestr(zi, src.read())
        print(f"zip size: {os.path.getsize(zpath) / 1e6:.0f} MB")

    print(f"\nbundle ready: {bundle}")
    return bundle


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("--zip", action="store_true",
                    help="also write DilatometryProcessor-<target>.zip")
    args = ap.parse_args()
    build(args.target, args.zip)


if __name__ == "__main__":
    main()
