"""Frozen entry point for the GUI-free AVISTA deep-learning worker."""

from __future__ import annotations

import multiprocessing

from app.training.run_torch_model import main


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
