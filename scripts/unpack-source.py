#!/usr/bin/env python3
"""Update changed source files from an archive using the build host's timestamps."""

import argparse
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not args.archive.resolve().is_relative_to(root):
        raise SystemExit("Keep the source archive inside this project")
    changed = 0
    with tarfile.open(args.archive) as archive:
        for member in archive:
            path = root / member.name
            if not member.isfile() or not path.resolve().is_relative_to(root) or path.is_symlink():
                raise SystemExit(f"Unsafe source archive entry: {member.name}")
            data = archive.extractfile(member).read()
            if path.exists() and path.read_bytes() == data:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(member.mode & 0o777)
            changed += 1
    print(f"Updated {changed} source files")


if __name__ == "__main__":
    main()
