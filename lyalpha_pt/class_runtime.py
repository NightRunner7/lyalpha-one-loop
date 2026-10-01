"""Record the installed CLASS wrapper and its declared source checkout."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


def class_runtime(source_dir=None, expected_commit=None, expected_wrapper=None):
    spec = importlib.util.find_spec("classy")
    if spec is None or not spec.origin:
        raise RuntimeError("classy is missing from this Python environment; install the custom wrapper first.")
    wrapper = Path(spec.origin).resolve()
    wrapper_hash = hashlib.sha256(wrapper.read_bytes()).hexdigest()
    expected_wrapper = expected_wrapper or os.environ.get("LYA_CLASS_WRAPPER_SHA256")
    if expected_wrapper and expected_wrapper != wrapper_hash:
        raise RuntimeError("Installed classy changed since campaign preparation; use a new campaign directory.")
    runtime = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "classy_file": str(wrapper),
        "classy_sha256": wrapper_hash,
    }
    source_dir = source_dir or os.environ.get("LYA_CLASS_SOURCE_DIR")
    expected_commit = expected_commit or os.environ.get("LYA_CLASS_SOURCE_COMMIT")
    if source_dir:
        source = Path(source_dir).expanduser().resolve()
        if not (source / "source" / "input.c").is_file():
            raise FileNotFoundError(f"Not a CLASS source directory: {source}")
        def git(*args):
            return subprocess.check_output(
                ["git", "-C", str(source), *args], text=True, stderr=subprocess.PIPE
            ).strip()
        commit = git("rev-parse", "HEAD")
        if expected_commit and commit != expected_commit:
            raise RuntimeError("CLASS source commit changed since campaign preparation; use a new campaign directory.")
        runtime["declared_source"] = {
            "directory": str(source), "commit": commit,
            "branch": git("branch", "--show-current"),
            "tracked_files_modified": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    return runtime
