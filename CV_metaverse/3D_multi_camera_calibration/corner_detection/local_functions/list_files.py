from __future__ import annotations

import glob
from pathlib import Path


def list_files(root: str | Path, file_type: str) -> list[str]:
    """List all files in *root* matching the given *file_type* extension.

    Args:
        root: Directory to search.
        file_type: Extension without the dot (e.g. ``"png"``).

    Returns:
        Sorted list of matching file paths.
    """
    pattern = Path(root) / f"*.{file_type}"
    return sorted(glob.glob(str(pattern)))


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[3] / "dataSet"
    if data_dir.exists():
        print(list_files(data_dir, "png"))
    else:
        print(f"Data directory not found: {data_dir}")
