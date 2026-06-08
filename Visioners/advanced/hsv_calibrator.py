"""Interactive HSV mask calibration helper.

Run examples:
    python Visioners/advanced/hsv_calibrator.py
    python Visioners/advanced/hsv_calibrator.py --image Visioners/temporary/all_keg_colors_frame.jpg
    python Visioners/advanced/hsv_calibrator.py --source 0
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


if __package__ in {None, ""}:
    PACKAGE_DIR = Path(__file__).resolve().parent
    if str(PACKAGE_DIR.parent) not in sys.path:
        sys.path.insert(0, str(PACKAGE_DIR.parent))
    __package__ = PACKAGE_DIR.name

from .camera_stream import CameraStream
from .config import ROBOT_IP
from .vision import COLOR_RANGES


COLOR_NAMES = tuple(COLOR_RANGES)
WINDOW = "HSV calibration"
TRACKBARS = {
    "H min": 179,
    "H max": 179,
    "S min": 255,
    "S max": 255,
    "V min": 255,
    "V max": 255,
}
PICK_REGION_RADIUS = 3
PICK_H_MARGIN = 8
PICK_S_MARGIN = 35
PICK_V_MARGIN = 35


@dataclass
class PickerState:
    hsv_frame: np.ndarray | None = None
    frame_size: tuple[int, int] | None = None
    picked_point: tuple[int, int] | None = None
    picked_hsv: tuple[int, int, int] | None = None


class FrameReader:
    def open(self) -> None:
        raise NotImplementedError

    def read(self):
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class ImageReader(FrameReader):
    def __init__(self, path: str) -> None:
        self.path = path
        self.frame = None

    def open(self) -> None:
        self.frame = cv2.imread(self.path)
        if self.frame is None:
            raise RuntimeError(f"Cannot read image: {self.path}")

    def read(self):
        if self.frame is None:
            self.open()
        return True, self.frame.copy()

    def close(self) -> None:
        self.frame = None


class OpenCVReader(FrameReader):
    def __init__(self, source: str) -> None:
        self.source = int(source) if source.isdigit() else source
        self.capture = None

    def open(self) -> None:
        self.capture = cv2.VideoCapture(self.source)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open source: {self.source}")

    def read(self):
        if self.capture is None:
            self.open()
        return self.capture.read()

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class RobotCameraReader(FrameReader):
    def __init__(self, ip: str) -> None:
        self.camera = CameraStream(ip)

    def open(self) -> None:
        self.camera.open()

    def read(self):
        return self.camera.read()

    def close(self) -> None:
        self.camera.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune HSV masks with OpenCV trackbars.")
    parser.add_argument("--ip", default=ROBOT_IP, help="Robot camera IP. Used when --image/--source are omitted.")
    parser.add_argument("--image", help="Calibrate from one saved image.")
    parser.add_argument("--source", help="Camera index, video file, image file, or stream URL.")
    parser.add_argument("--color", default="blue", choices=COLOR_NAMES, help="Initial color to tune.")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "hsv_ranges.json"),
        help="Where to save calibrated HSV ranges as JSON.",
    )
    return parser.parse_args()


def build_reader(args: argparse.Namespace) -> FrameReader:
    if args.image:
        return ImageReader(args.image)
    if args.source:
        suffix = Path(args.source).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            return ImageReader(args.source)
        return OpenCVReader(args.source)
    return RobotCameraReader(args.ip)


def ranges_to_trackbar_values(ranges) -> dict[str, int]:
    if len(ranges) == 2 and ranges[0][0][0] == 0 and ranges[1][1][0] == 179:
        low_wrap, high_wrap = ranges
        return {
            "H min": high_wrap[0][0],
            "H max": low_wrap[1][0],
            "S min": min(low_wrap[0][1], high_wrap[0][1]),
            "S max": max(low_wrap[1][1], high_wrap[1][1]),
            "V min": min(low_wrap[0][2], high_wrap[0][2]),
            "V max": max(low_wrap[1][2], high_wrap[1][2]),
        }
    lower, upper = ranges[0]
    return {
        "H min": lower[0],
        "H max": upper[0],
        "S min": lower[1],
        "S max": upper[1],
        "V min": lower[2],
        "V max": upper[2],
    }


def color_to_trackbar_values(color: str) -> dict[str, int]:
    return ranges_to_trackbar_values(COLOR_RANGES[color])


def create_trackbars(initial_color: str) -> None:
    for name, maximum in TRACKBARS.items():
        cv2.createTrackbar(name, WINDOW, 0, maximum, lambda _value: None)
    set_trackbars(color_to_trackbar_values(initial_color))


def read_trackbars() -> dict[str, int]:
    return {name: cv2.getTrackbarPos(name, WINDOW) for name in TRACKBARS}


def set_trackbars(values: dict[str, int]) -> None:
    for name, value in values.items():
        cv2.setTrackbarPos(name, WINDOW, int(value))


def values_to_ranges(values: dict[str, int]) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    h_min = values["H min"]
    h_max = values["H max"]
    s_min = values["S min"]
    s_max = values["S max"]
    v_min = values["V min"]
    v_max = values["V max"]
    if h_min <= h_max:
        return [((h_min, s_min, v_min), (h_max, s_max, v_max))]
    return [
        ((0, s_min, v_min), (h_max, s_max, v_max)),
        ((h_min, s_min, v_min), (179, s_max, v_max)),
    ]


def make_mask(hsv_frame, ranges) -> np.ndarray:
    mask = None
    for lower, upper in ranges:
        current = cv2.inRange(hsv_frame, lower, upper)
        mask = current if mask is None else cv2.bitwise_or(mask, current)
    return mask


def hsv_sample_to_trackbar_values(hsv: tuple[int, int, int]) -> dict[str, int]:
    hue, saturation, value = hsv
    h_min = (hue - PICK_H_MARGIN) % 180
    h_max = (hue + PICK_H_MARGIN) % 180
    return {
        "H min": h_min,
        "H max": h_max,
        "S min": max(0, saturation - PICK_S_MARGIN),
        "S max": min(255, saturation + PICK_S_MARGIN),
        "V min": max(0, value - PICK_V_MARGIN),
        "V max": min(255, value + PICK_V_MARGIN),
    }


def pick_hsv_from_preview(state: PickerState, x: int, y: int) -> tuple[int, int, int] | None:
    if state.hsv_frame is None or state.frame_size is None:
        return None
    frame_width, frame_height = state.frame_size
    if frame_width <= 0 or frame_height <= 0:
        return None
    source_x = x % frame_width
    source_y = y % frame_height
    if x < 0 or y < 0 or x >= frame_width * 2 or y >= frame_height * 2:
        return None

    x0 = max(0, source_x - PICK_REGION_RADIUS)
    x1 = min(frame_width, source_x + PICK_REGION_RADIUS + 1)
    y0 = max(0, source_y - PICK_REGION_RADIUS)
    y1 = min(frame_height, source_y + PICK_REGION_RADIUS + 1)
    sample = state.hsv_frame[y0:y1, x0:x1]
    if sample.size == 0:
        return None
    median = np.median(sample.reshape(-1, 3), axis=0).astype(int)
    return int(median[0]), int(median[1]), int(median[2])


def on_mouse(event, x, y, _flags, userdata) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    state: PickerState = userdata
    hsv = pick_hsv_from_preview(state, x, y)
    if hsv is None:
        return
    state.picked_point = (x, y)
    state.picked_hsv = hsv
    set_trackbars(hsv_sample_to_trackbar_values(hsv))
    print(f"Picked HSV: H={hsv[0]} S={hsv[1]} V={hsv[2]}")


def format_python_range(color: str, ranges) -> str:
    body = ", ".join(f"(({lo[0]}, {lo[1]}, {lo[2]}), ({hi[0]}, {hi[1]}, {hi[2]}))" for lo, hi in ranges)
    return f'    "{color}": [{body}],'


def save_ranges(path: str, calibrated: dict[str, list]) -> None:
    output = {}
    for color, ranges in calibrated.items():
        output[color] = [[list(lower), list(upper)] for lower, upper in ranges]
    Path(path).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


def draw_help(frame, color: str, values: dict[str, int], ranges, picker_state: PickerState) -> None:
    picked = ""
    if picker_state.picked_hsv is not None:
        h, s, v = picker_state.picked_hsv
        picked = f"  PICKED H={h} S={s} V={v}"
    lines = [
        f"Color: {color}{picked}",
        f"H {values['H min']}..{values['H max']}  S {values['S min']}..{values['S max']}  V {values['V min']}..{values['V max']}",
        "Mouse: left-click any preview tile to pick HSV | Keys: 1-6 color | n/p next/prev",
        "Keys: r reset | space print | s save | q quit",
        "If H min > H max, hue wraps around red boundary and saves as two ranges.",
    ]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 112), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.58, frame, 0.42, 0)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (12, 22 + index * 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if len(ranges) == 2:
        cv2.putText(frame, "WRAP", (frame.shape[1] - 76, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    if picker_state.picked_point is not None:
        cv2.drawMarker(frame, picker_state.picked_point, (0, 255, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    reader = build_reader(args)
    color_index = COLOR_NAMES.index(args.color)
    current_color = COLOR_NAMES[color_index]
    calibrated = {color: list(ranges) for color, ranges in COLOR_RANGES.items()}
    picker_state = PickerState()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 720)
    create_trackbars(current_color)
    cv2.setMouseCallback(WINDOW, on_mouse, picker_state)

    try:
        reader.open()
        while True:
            ok, frame = reader.read()
            if not ok:
                print("Frame was not received")
                break

            values = read_trackbars()
            ranges = values_to_ranges(values)
            calibrated[current_color] = ranges

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_height, frame_width = frame.shape[:2]
            picker_state.hsv_frame = hsv
            picker_state.frame_size = (frame_width, frame_height)
            mask = make_mask(hsv, ranges)
            result = cv2.bitwise_and(frame, frame, mask=mask)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            top = np.hstack((frame, result))
            bottom = np.hstack((mask_bgr, cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)))
            preview = np.vstack((top, bottom))
            draw_help(preview, current_color, values, ranges, picker_state)
            cv2.imshow(WINDOW, preview)

            key = cv2.waitKey(30) & 0xFF
            if key in {ord("q"), ord("Q"), 27}:
                break
            if key in {ord(" "), ord("P")}:
                print(format_python_range(current_color, ranges))
            elif key in {ord("s"), ord("S")}:
                save_ranges(args.output, calibrated)
                print(f"Saved HSV ranges to {args.output}")
                print(format_python_range(current_color, ranges))
            elif key in {ord("r"), ord("R")}:
                set_trackbars(color_to_trackbar_values(current_color))
            elif key in {ord("n"), ord("N")}:
                color_index = (color_index + 1) % len(COLOR_NAMES)
                current_color = COLOR_NAMES[color_index]
                set_trackbars(ranges_to_trackbar_values(calibrated[current_color]))
            elif key in {ord("p")}:
                color_index = (color_index - 1) % len(COLOR_NAMES)
                current_color = COLOR_NAMES[color_index]
                set_trackbars(ranges_to_trackbar_values(calibrated[current_color]))
            elif ord("1") <= key <= ord(str(len(COLOR_NAMES))):
                color_index = key - ord("1")
                current_color = COLOR_NAMES[color_index]
                set_trackbars(ranges_to_trackbar_values(calibrated[current_color]))
    finally:
        reader.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
