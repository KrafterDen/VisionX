"""CLI entry point for the advanced autonomous kegelring controller."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


if __package__ in {None, ""}:
    PACKAGE_DIR = Path(__file__).resolve().parent
    if str(PACKAGE_DIR.parent) not in sys.path:
        sys.path.insert(0, str(PACKAGE_DIR.parent))
    __package__ = PACKAGE_DIR.name

from .camera_stream import CameraStream
from .config import AdvancedConfig, VALID_TARGET_COLORS
from .controller import AdvancedController
from .debug import RunLogger
from .robot_io import RobotIO


class CameraFrameSource:
    def __init__(self, config: AdvancedConfig) -> None:
        self.camera = CameraStream(config.robot_ip, port=config.stream_port, path=config.stream_path)

    def open(self) -> None:
        self.camera.open()

    def read(self):
        return self.camera.read()

    def close(self) -> None:
        self.camera.close()


class ImageFrameSource:
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
    if not config.video_source:
        return CameraFrameSource(config)
    suffix = Path(config.video_source).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return ImageFrameSource(config.video_source)
    return OpenCVFrameSource(config.video_source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced autonomous kegelring controller.")
    parser.add_argument("--ip", default=None, help="ESP32-CAM robot IP address.")
    parser.add_argument(
        "--target-color",
        default="blue",
        choices=VALID_TARGET_COLORS,
        help="Target pin color to knock down.",
    )
    parser.add_argument("--target-count", type=int, default=2)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", "--no-robot", action="store_true", dest="dry_run")
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Only run the initial scan, save the map, stop, and exit.",
    )
    parser.add_argument("--video-source", help="Camera index, video file, image file, or stream URL.")
    parser.add_argument(
        "--save-log",
        nargs="?",
        const="advanced_kegelring_log.jsonl",
        help="Write JSONL run log. Optional path; default advanced_kegelring_log.jsonl.",
    )
    parser.add_argument("--k-distance", type=float, default=None)
    parser.add_argument("--scan-pwm", type=int, default=None)
    parser.add_argument("--scan-left-pwm", type=int, default=None)
    parser.add_argument("--scan-right-pwm", type=int, default=None)
    parser.add_argument("--scan-left-angular-speed", type=float, default=None)
    parser.add_argument("--scan-right-angular-speed", type=float, default=None)
    parser.add_argument(
        "--scan-angle",
        type=float,
        default=None,
        help="Total initial scan angle in degrees. 180 means -90 to +90.",
    )
    parser.add_argument("--approach-pwm", type=int, default=None)
    parser.add_argument("--attack-pwm", type=int, default=None)
    parser.add_argument(
        "--map-output-dir",
        default=None,
        help="Directory for saved map JSON files.",
    )
    parser.add_argument(
        "--no-save-map",
        action="store_true",
        help="Disable automatic map JSON saving.",
    )
    parser.add_argument(
        "--detection-debug-dir",
        default=None,
        help="Directory for accepted-detection debug images.",
    )
    parser.add_argument(
        "--no-save-detection-debug",
        action="store_true",
        help="Disable accepted-detection debug image saving.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> AdvancedConfig:
    config = AdvancedConfig(
        target_color=args.target_color,
        target_count=args.target_count,
        debug=args.debug,
        dry_run=args.dry_run,
        scan_only=args.scan_only,
        save_log=args.save_log,
        video_source=args.video_source,
        save_map=not args.no_save_map,
        save_detection_debug_images=not args.no_save_detection_debug,
    )
    if args.ip is not None:
        config.robot_ip = args.ip
    if args.k_distance is not None:
        config.k_distance = args.k_distance
    if args.scan_pwm is not None:
        config.scan_pwm = args.scan_pwm
        config.scan_left_pwm = args.scan_pwm
        config.scan_right_pwm = args.scan_pwm
    if args.scan_left_pwm is not None:
        config.scan_left_pwm = args.scan_left_pwm
    if args.scan_right_pwm is not None:
        config.scan_right_pwm = args.scan_right_pwm
    if args.scan_left_angular_speed is not None:
        config.scan_left_angular_speed_deg_per_sec = args.scan_left_angular_speed
    if args.scan_right_angular_speed is not None:
        config.scan_right_angular_speed_deg_per_sec = args.scan_right_angular_speed
    if args.scan_angle is not None:
        config.max_scan_angle_deg = args.scan_angle
    if args.approach_pwm is not None:
        config.approach_pwm = args.approach_pwm
    if args.attack_pwm is not None:
        config.attack_pwm = args.attack_pwm
    if args.map_output_dir is not None:
        config.map_output_dir = args.map_output_dir
    if args.detection_debug_dir is not None:
        config.detection_debug_dir = args.detection_debug_dir
    return config


def main() -> int:
    args = parse_args()
    config = config_from_args(args)
    logger = RunLogger(config.save_log)
    robot = RobotIO(config)
    frame_source = build_frame_source(config)
    controller = AdvancedController(
        config=config,
        robot=robot,
        frame_source=frame_source,
        logger=logger,
    )
    try:
        controller.run()
        return 0
    finally:
        try:
            robot.stop()
        finally:
            frame_source.close()
            logger.close()
            if config.debug:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
