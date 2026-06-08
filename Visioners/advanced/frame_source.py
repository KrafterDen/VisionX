"""Frame source adapters for camera, image, and video inputs."""

from __future__ import annotations

from pathlib import Path

import cv2

from .camera_stream import CameraStream
from .config import AdvancedConfig


class CameraFrameSource:
    """Read frames from the robot ESP32-CAM stream."""

    def __init__(self, config: AdvancedConfig) -> None:
        self.camera = CameraStream(config.robot_ip, port=config.stream_port, path=config.stream_path)

    def open(self) -> None:
        self.camera.open()

    def read(self):
        return self.camera.read()

    def close(self) -> None:
        self.camera.close()


class ImageFrameSource:
    """Return a fresh copy of one saved image on every read."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.frame = None

    def open(self) -> None:
        self.frame = cv2.imread(self.path)
        if self.frame is None:
            raise RuntimeError(f"Cannot read image source: {self.path}")

    def read(self):
        if self.frame is None:
            self.open()
        return True, self.frame.copy()

    def close(self) -> None:
        self.frame = None


class OpenCVFrameSource:
    """Read frames from a webcam index, video file, or stream URL."""

    def __init__(self, source: str) -> None:
        self.source = int(source) if source.isdigit() else source
        self.capture = None

    def open(self) -> None:
        self.capture = cv2.VideoCapture(self.source)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

    def read(self):
        if self.capture is None:
            self.open()
        return self.capture.read()

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


def build_frame_source(config: AdvancedConfig):
    """Create the right frame source for the current config."""

    if not config.video_source:
        return CameraFrameSource(config)
    suffix = Path(config.video_source).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return ImageFrameSource(config.video_source)
    return OpenCVFrameSource(config.video_source)
