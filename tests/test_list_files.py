"""Tests for the local_functions package."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from list_files import list_files


def test_list_files_returns_sorted_paths() -> None:
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "b.png").write_text("")
        (tmp_path / "a.png").write_text("")
        (tmp_path / "c.txt").write_text("")

        result = list_files(tmp_path, "png")

        assert len(result) == 2
        assert result == sorted(result)
        assert all("png" in p for p in result)


def test_list_files_empty_directory() -> None:
    with TemporaryDirectory() as tmp:
        result = list_files(tmp, "jpg")
        assert result == []


def test_list_files_no_matching_extension() -> None:
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "image.png").write_text("")

        result = list_files(tmp_path, "bmp")
        assert result == []
