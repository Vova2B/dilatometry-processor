#!/usr/bin/env python3
"""sync_export.py — regenerate the offline copy and the update zip from THIS
canonical scripts/ folder. 2_offline_sharing/app/scripts is generated output:
never hand-edit it. Run from 1_github_publication/scripts/.

    python3 sync_export.py [--date YYYY-MM-DD]

--date is required-explicit because Date.now() is unavailable in some harnesses;
default is read from the system clock when run as a normal CLI.
"""
import argparse
import datetime as _dt
import os
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))            # .../1_github.../scripts
PUB = os.path.dirname(HERE)                                  # .../1_github_publication
SHARE = os.path.dirname(PUB)                                 # .../for opensourcing and sharing
USE_ROOT = os.path.dirname(SHARE)                            # .../USe
OFFLINE = os.path.join(SHARE, "2_offline_sharing", "app", "scripts")

COPY_EXT = (".py", ".json", ".txt")
SKIP = {"sync_export.py", "make_portable_bundle.py", "__pycache__"}


def _files():
    for name in sorted(os.listdir(HERE)):
        if name in SKIP or name.startswith("."):
            continue
        if name.endswith(COPY_EXT):
            yield name


def sync_offline():
    os.makedirs(OFFLINE, exist_ok=True)
    written = []
    current = set()
    for name in _files():
        shutil.copy2(os.path.join(HERE, name), os.path.join(OFFLINE, name))
        written.append(name)
        current.add(name)
    pruned = []
    if os.path.isdir(OFFLINE):
        for name in sorted(os.listdir(OFFLINE)):
            if name in SKIP or name.startswith("."):
                continue
            if not name.endswith(COPY_EXT):
                continue
            if name in current:
                continue
            path = os.path.join(OFFLINE, name)
            if os.path.isfile(path):
                os.remove(path)
                pruned.append(name)
    return written, pruned


def build_zip(date_str):
    zpath = os.path.join(USE_ROOT, f"dilat_code_update_{date_str}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for name in _files():
            z.write(os.path.join(HERE, name), arcname=f"scripts/{name}")
    return zpath


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None,
                    help="date stamp for the zip (YYYY-MM-DD)")
    args = ap.parse_args(argv)
    date_str = args.date or _dt.date.today().isoformat()
    written, pruned = sync_offline()
    zpath = build_zip(date_str)
    print(f"synced {len(written)} files -> {OFFLINE}")
    for n in written:
        print(f"  {n}")
    if pruned:
        print(f"pruned {len(pruned)} stale file(s) from {OFFLINE}:")
        for n in pruned:
            print(f"  {n}")
    print(f"zip -> {zpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
