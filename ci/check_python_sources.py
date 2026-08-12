#!/usr/bin/env python3
"""Compile Python sources without leaving __pycache__ in the checkout."""

from __future__ import annotations

import py_compile
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    sources = sorted(path for path in ROOT.rglob("*.py") if ".git" not in path.parts)
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="modelzoo-pycompile-") as cache:
        cache_root = Path(cache)
        for source in sources:
            relative = source.relative_to(ROOT)
            target = cache_root / relative.with_suffix(".pyc")
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                py_compile.compile(str(source), cfile=str(target), doraise=True)
            except py_compile.PyCompileError as exc:
                failures.append(f"{relative}: {exc.msg}")

    if failures:
        print("Python compile failures:", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1

    print(f"Compiled {len(sources)} Python sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

