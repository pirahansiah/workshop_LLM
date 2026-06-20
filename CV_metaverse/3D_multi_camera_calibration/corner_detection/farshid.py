"""Computer vision utilities for the CV metaverse workshop.

Author: Farshid Pirahansiah — 8 August 2019
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


class ImageProcessing:
    """OpenCV-based image processing utilities."""

    HARCASCADE_DIR: Path = Path(__file__).resolve().parent / "haarcascades"

    def show_image_plt(self, image: NDArray[np.uint8]) -> None:
        """Display an image using matplotlib."""
        from matplotlib import pyplot as plt

        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_img)
        plt.title("ORIGINAL")
        plt.show()

    def show_image_opencv(self, image: NDArray[np.uint8], title: str = "farshid") -> None:
        """Display an image with OpenCV."""
        cv2.imshow(title, image)
        cv2.waitKey(1000)

    def save_image(self, filename: str, img: NDArray[np.uint8]) -> None:
        """Save an image to disk."""
        if not filename:
            filename = "farshid.jpg"
        cv2.imwrite(filename, img)

    def cartoon_image(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Apply a cartoon effect to the image.

        Uses bilateral filtering for colour smoothing and adaptive thresholding
        for edge detection.
        """
        num_down = 5
        num_bilateral = 9

        img_color = image.copy()
        for _ in range(num_down):
            img_color = cv2.pyrDown(img_color)
        for _ in range(num_bilateral):
            img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
        for _ in range(num_down):
            img_color = cv2.pyrUp(img_color)

        h, w = image.shape[:2]
        img_color = cv2.resize(img_color, (w, h))

        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.medianBlur(img_gray, 7)
        img_edge = cv2.adaptiveThreshold(
            img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2
        )
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        return cv2.bitwise_and(img_color, img_edge)

    def _load_cascade(self, name: str) -> cv2.CascadeClassifier:
        path = self.HARCASCADE_DIR / name
        cascade = cv2.CascadeClassifier(str(path))
        if cascade.empty():
            raise FileNotFoundError(f"Failed to load cascade: {path}")
        return cascade

    def face_detection_webcam(self) -> None:
        """Run real-time face+eye detection from the default webcam."""
        face_cascade = self._load_cascade("haarcascade_frontalface_default.xml")
        eye_cascade = self._load_cascade("haarcascade_eye.xml")

        cap = cv2.VideoCapture(0)
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                img = frame.copy()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for x, y, w, h in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    roi_gray = gray[y : y + h, x : x + w]
                    roi_color = img[y : y + h, x : x + w]
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    for ex, ey, ew, eh in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, "original", (10, 10), font, 0.3, (255, 0, 0), 2)
                cv2.putText(img, "face detection", (10, 10), font, 0.3, (255, 0, 0), 2)

                img_concat = np.concatenate((frame, img), axis=1)
                cv2.imshow("Computer Vision Workshop", img_concat)
                if cv2.waitKey(1) == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def face_detection(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Detect faces and eyes in a single image and return the annotated copy."""
        face_cascade = self._load_cascade("haarcascade_frontalface_default.xml")
        eye_cascade = self._load_cascade("haarcascade_eye.xml")

        result = img.copy()
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y : y + h, x : x + w]
            roi_color = result[y : y + h, x : x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for ex, ey, ew, eh in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        return result

    def thresholding(self) -> None:
        """Compare various thresholding methods on webcam feed."""
        cap = cv2.VideoCapture(0)
        size_image = 300
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, th1 = cv2.threshold(frame_gray, 10, 255, cv2.THRESH_BINARY)
                _, th2 = cv2.threshold(frame_gray, 10, 255, cv2.THRESH_OTSU)
                _, th3 = cv2.threshold(frame_gray, 10, 255, cv2.THRESH_TOZERO_INV)
                th4 = cv2.adaptiveThreshold(
                    frame_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
                )
                th5 = cv2.adaptiveThreshold(
                    frame_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )

                sz = (size_image, size_image)
                frame = cv2.resize(frame, sz)
                frame_gray = cv2.cvtColor(cv2.resize(frame_gray, sz), cv2.COLOR_GRAY2BGR)
                th1 = cv2.cvtColor(cv2.resize(th1, sz), cv2.COLOR_GRAY2BGR)
                th2 = cv2.cvtColor(cv2.resize(th2, sz), cv2.COLOR_GRAY2BGR)
                th3 = cv2.cvtColor(cv2.resize(th3, sz), cv2.COLOR_GRAY2BGR)
                th4 = cv2.cvtColor(cv2.resize(th4, sz), cv2.COLOR_GRAY2BGR)
                th5 = cv2.cvtColor(cv2.resize(th5, sz), cv2.COLOR_GRAY2BGR)

                font = cv2.FONT_HERSHEY_SIMPLEX
                pos = (10, 10)
                for text, img in [
                    ("THRESH_BINARY", th1),
                    ("THRESH_OTSU", th2),
                    ("THRESH_TOZERO_INV", th3),
                    ("ADAPTIVE_THRESH_MEAN_C", th4),
                    ("ADAPTIVE_THRESH_GAUSSIAN_C", th5),
                    ("Original", frame),
                ]:
                    cv2.putText(img, text, pos, font, 0.3, (255, 0, 0), 2)

                row1 = cv2.vconcat([frame, frame_gray])
                row2 = cv2.vconcat([th1, th2])
                row3 = cv2.vconcat([th5, th4])
                img_all = cv2.hconcat([row1, row2, row3])
                cv2.imshow("Thresholding Comparison", img_all)
                if cv2.waitKey(1) == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def color_spaces(self) -> None:
        """Show a comparison of colour-space conversions from webcam feed."""
        cap = cv2.VideoCapture(0)
        size_image = 300
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                original = cv2.resize(frame, (size_image, size_image))
                conversions = [
                    ("COLOR_BGR2GRAY", cv2.COLOR_BGR2GRAY),
                    ("COLOR_BGR2HSV", cv2.COLOR_BGR2HSV),
                    ("COLOR_BGR2YUV", cv2.COLOR_BGR2YUV),
                    ("COLOR_BGR2YCrCb", cv2.COLOR_BGR2YCrCb),
                    ("COLOR_BGR2HLS", cv2.COLOR_BGR2HLS),
                    ("COLOR_BGR2Luv", cv2.COLOR_BGR2Luv),
                ]

                font = cv2.FONT_HERSHEY_SIMPLEX
                pos = (10, 10)
                converted: list[NDArray[np.uint8]] = []
                for label, code in conversions:
                    out = cv2.cvtColor(original, code)
                    if len(out.shape) == 2:
                        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
                    cv2.putText(out, label, pos, font, 0.3, (255, 0, 0), 2)
                    converted.append(out)

                img_concat = np.concatenate(
                    [np.concatenate([original, converted[0]], axis=0)]
                    + [
                        np.concatenate([converted[i], converted[i + 1]], axis=0)
                        for i in range(1, len(converted), 2)
                    ],
                    axis=1,
                )
                cv2.imshow("Colour Space Comparison", img_concat)
                if cv2.waitKey(1) == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def cartoon_webcam(self) -> None:
        """Apply cartoon effect to webcam feed in real time."""
        cap = cv2.VideoCapture(0)
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                cartoon = self.cartoon_image(frame)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, "original", (10, 10), font, 0.3, (255, 0, 0), 2)
                cv2.putText(cartoon, "cartoon", (10, 10), font, 0.3, (255, 0, 0), 2)
                img_concat = np.concatenate((frame, cartoon), axis=1)
                cv2.imshow("Cartoon Effect", img_concat)
                if cv2.waitKey(1) == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def main_demo(self) -> None:
        """Quick demonstration: capture one frame and apply thresholding."""
        print("Farshid Pirahansiah")
        print(cv2.__version__)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam")
            return

        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to capture frame")
                return

            cv2.imshow("original", frame)
            cv2.waitKey(1000)

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            th2 = cv2.adaptiveThreshold(
                frame_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
            )
            th3 = cv2.adaptiveThreshold(
                frame_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            self.show_image_plt(th2)
            self.show_image_opencv(th3)
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    processor = ImageProcessing()
    processor.color_spaces()
