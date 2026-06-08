"""Camera stream reader for the ESP32-CAM MJPEG endpoint."""

from __future__ import annotations

import cv2

try:
    from config import STREAM_PATH, STREAM_PORT
except ImportError:
    from .config import STREAM_PATH, STREAM_PORT


class CameraStream:
    """Read frames from http://<robot-ip>:81/stream with OpenCV."""

    def __init__(self, ip: str, port: int = STREAM_PORT, path: str = STREAM_PATH) -> None:
        self.ip = ip
        self.port = port
        self.path = path if path.startswith("/") else f"/{path}"
        self._capture = None

    @property
    def url(self) -> str:
        return f"http://{self.ip}:{self.port}{self.path}"

    def open(self) -> None:
        self.close()
        self._capture = cv2.VideoCapture(self.url)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._capture.isOpened():
            raise RuntimeError(f"Cannot open camera stream: {self.url}")

    def read(self):
        if self._capture is None:
            raise RuntimeError("Camera stream is not open. Call open() first.")
        return self._capture.read()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> CameraStream:
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
