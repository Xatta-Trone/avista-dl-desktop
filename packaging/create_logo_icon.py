"""Generate a Windows ICO file from the AVISTA PNG logo."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image


ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def build_icon(source_path: Path, destination_path: Path) -> None:
    source = Image.open(source_path).convert("RGBA")
    master_size = max(ICON_SIZES)
    master = Image.new("RGBA", (master_size, master_size), (0, 0, 0, 0))
    image = source.copy()
    image.thumbnail((master_size, master_size), Image.Resampling.LANCZOS)
    x = (master_size - image.width) // 2
    y = (master_size - image.height) // 2
    master.alpha_composite(image, (x, y))

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    master.save(
        destination_path,
        format="ICO",
        sizes=[(size, size) for size in ICON_SIZES],
    )


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: create_logo_icon.py <source-logo.png> <destination-logo.ico>")
        return 2
    build_icon(Path(sys.argv[1]), Path(sys.argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
