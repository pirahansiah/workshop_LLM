"""Tests for ImageProcessing class."""

from __future__ import annotations

import cv2
import numpy as np

from farshid import ImageProcessing


def test_cartoon_image_returns_correct_shape() -> None:
    proc = ImageProcessing()
    img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    result = proc.cartoon_image(img)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_save_image_creates_file(tmp_path) -> None:
    proc = ImageProcessing()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = tmp_path / "test_output.jpg"
    proc.save_image(str(out), img)
    assert out.exists()
    loaded = cv2.imread(str(out))
    assert loaded is not None


def test_save_image_default_filename(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    proc = ImageProcessing()
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    proc.save_image("", img)
    assert (tmp_path / "farshid.jpg").exists()
