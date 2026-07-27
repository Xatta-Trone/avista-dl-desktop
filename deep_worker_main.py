"""Frozen entry point for the GUI-free AVISTA deep-learning worker."""

from __future__ import annotations

import multiprocessing
import sys


def main() -> int:
    if "--packaging-smoke-test" in sys.argv[1:]:
        try:
            kind = sys.argv[sys.argv.index("--packaging-smoke-test") + 1]
            output_path = sys.argv[sys.argv.index("--smoke-output") + 1]
        except (ValueError, IndexError):
            return 2
        from app.core.packaging_smoke import run_packaging_smoke

        return run_packaging_smoke(kind, output_path)

    from app.training.run_torch_model import main as run_worker

    return run_worker()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
