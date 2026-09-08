#!/usr/bin/env python3
"""Apply the project's dependency patches without resetting existing work."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def git(repo, *args, check=True):
    return subprocess.run(["git", "-c", f"safe.directory={repo.resolve()}", "-C", str(repo), *args],
                          check=check, text=True, capture_output=True)


def series_is_applied(repo, patches):
    # Later patches can change an earlier patch's context. Reverse the whole
    # series in a disposable copy instead of touching the user's working tree.
    with tempfile.TemporaryDirectory() as temporary:
        copy = Path(temporary)
        git(copy, "init", "--quiet")
        paths = set()
        for patch in patches:
            for line in git(repo, "apply", "--numstat", str(patch)).stdout.splitlines():
                path = Path(line.split("\t", 2)[2])
                if path.is_absolute() or ".." in path.parts:
                    raise SystemExit(f"Unsafe patch path: {path}")
                paths.add(path)
        for path in paths:
            source = repo / path
            if source.is_file():
                (copy / path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, copy / path)
        for patch in reversed(patches):
            if git(copy, "apply", "--reverse", str(patch), check=False).returncode:
                return False
        return True


def main():
    root = Path(__file__).resolve().parents[1]
    dependencies = json.loads((root / "dependencies.json").read_text())
    for name, dependency in dependencies.items():
        repo = root / "third_party" / name
        if git(repo, "rev-parse", "HEAD").stdout.strip() != dependency["revision"]:
            raise SystemExit(f"{name}: expected the pinned revision; inspect local changes before updating")
        patches = sorted((root / "patches" / name).glob("*.patch"))
        if not patches:
            print(f"{name}: pinned revision needs no local patches")
            continue
        applied = next((n for n in range(len(patches), 0, -1) if series_is_applied(repo, patches[:n])), 0)
        if applied == len(patches):
            print(f"{name}: patch series already applied")
            continue
        for patch in patches[applied:]:
            if git(repo, "apply", "--reverse", "--check", str(patch), check=False).returncode == 0:
                print(f"{name}: {patch.name} already applied")
                continue
            result = git(repo, "apply", "--check", str(patch), check=False)
            if result.returncode:
                raise SystemExit(f"{name}: patch does not apply cleanly:\n{result.stderr}")
            branch = git(repo, "branch", "--show-current").stdout.strip()
            if branch != dependency["branch"]:
                exists = git(repo, "show-ref", "--verify", "--quiet", "refs/heads/" + dependency["branch"], check=False)
                switch_args = [dependency["branch"]] if exists.returncode == 0 else ["-c", dependency["branch"]]
                git(repo, "switch", *switch_args)
            git(repo, "apply", str(patch))
            print(f"{name}: applied {patch.name}")


if __name__ == "__main__":
    main()
