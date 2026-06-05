"""Frame processing helpers.

Keep computer-vision logic here, away from robot movement code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2

try:
    from config import ALIGN_TOLERANCE_PIXELS, TARGET_COLOR
except ImportError:
    from .config import ALIGN_TOLERANCE_PIXELS, TARGET_COLOR


MIN_COLOR_AREA = 750
MAX_COLOR_AREA_RATIO = 0.45
MIN_KEG_WIDTH = 25
MIN_KEG_HEIGHT = 35
MIN_KEG_ASPECT_RATIO = 1.05
MAX_KEG_ASPECT_RATIO = 3.2
MIN_CLOSE_KEG_AREA = 9000
MIN_CLOSE_KEG_WIDTH = 70
MIN_CLOSE_KEG_HEIGHT = 45
MIN_CLOSE_KEG_ASPECT_RATIO = 0.45
MIN_CLOSE_KEG_BOTTOM_RATIO = 0.55
TOP_EDGE_MARGIN = 8
SIDE_EDGE_MARGIN = 4
MIN_SIDE_EDGE_AREA = 5000
COLOR_RANGES = {
    "red": [((0, 55, 35), (10, 255, 255)), ((176, 55, 35), (179, 255, 255))],
    "pink": [((145, 35, 35), (174, 255, 255))],
    "purple": [((113, 35, 30), (138, 255, 210))],
    "blue": [((96, 55, 35), (112, 255, 255))],
    "green": [((42, 55, 70), (85, 255, 255))],
    "yellow": [((20, 35, 90), (42, 255, 255))],
}
BOX_COLORS = {
    "red": (0, 0, 255),
    "pink": (180, 80, 255),
    "purple": (160, 0, 180),
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
}
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


@dataclass(frozen=True)
class KegDetection:
    """Structured data for one detected keg.

    __getitem__ keeps older code working while new code can use attributes.
    """

    color: str
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    area: float
    confidence: float

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]

    @property
    def cx(self) -> int:
        return self.center[0]

    @property
    def cy(self) -> int:
        return self.center[1]

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def aspect_ratio(self) -> float:
        return self.height / max(self.width, 1)

    def is_target(self, target_color: str | set[str] | tuple[str, ...]) -> bool:
        if isinstance(target_color, str):
            return self.color == target_color
        return self.color in target_color

    def is_enemy(self, target_color: str | set[str] | tuple[str, ...]) -> bool:
        return not self.is_target(target_color)

    def to_dict(self) -> dict[str, Any]:
        return {
            "color": self.color,
            "bbox": self.bbox,
            "center": self.center,
            "cx": self.cx,
            "cy": self.cy,
            "area": self.area,
            "confidence": self.confidence,
        }

    def __getitem__(self, key: str) -> Any:
        if key == "color":
            return self.color
        if key == "bbox":
            return self.bbox
        if key == "center":
            return self.center
        if key == "cx":
            return self.cx
        if key == "cy":
            return self.cy
        if key == "area":
            return self.area
        if key == "confidence":
            return self.confidence
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def _is_keg_candidate(area: float, x: int, y: int, width: int, height: int, frame_shape) -> bool:
    frame_height, frame_width = frame_shape[:2]
    frame_area = frame_width * frame_height
    aspect_ratio = height / max(width, 1)
    bottom = y + height

    if area > frame_area * MAX_COLOR_AREA_RATIO:
        return False
    if y <= TOP_EDGE_MARGIN:
        return False
    if (x <= SIDE_EDGE_MARGIN or x + width >= frame_width - SIDE_EDGE_MARGIN) and area < MIN_SIDE_EDGE_AREA:
        return False

    is_normal_keg = (
        width >= MIN_KEG_WIDTH
        and height >= MIN_KEG_HEIGHT
        and MIN_KEG_ASPECT_RATIO <= aspect_ratio <= MAX_KEG_ASPECT_RATIO
    )
    is_close_keg = (
        area >= MIN_CLOSE_KEG_AREA
        and width >= MIN_CLOSE_KEG_WIDTH
        and height >= MIN_CLOSE_KEG_HEIGHT
        and aspect_ratio >= MIN_CLOSE_KEG_ASPECT_RATIO
        and aspect_ratio <= MAX_KEG_ASPECT_RATIO
        and bottom >= frame_height * MIN_CLOSE_KEG_BOTTOM_RATIO
    )

    if not (is_normal_keg or is_close_keg):
        return False

    return True


def detect_kegs(frame, target_color: str | None = None, min_area: int = MIN_COLOR_AREA):
    """Return detected keg candidates as KegDetection objects."""

    detections = []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color_name, ranges in COLOR_RANGES.items():
        if target_color is not None and color_name != target_color:
            continue

        mask = None
        for lower, upper in ranges:
            current_mask = cv2.inRange(hsv, lower, upper)
            mask = current_mask if mask is None else cv2.bitwise_or(mask, current_mask)

        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            if not _is_keg_candidate(area, x, y, width, height, frame.shape):
                continue

            center_x = x + width // 2
            center_y = y + height // 2
            bbox_area = max(width * height, 1)
            confidence = min(1.0, area / bbox_area)
            detections.append(
                KegDetection(
                    color=color_name,
                    bbox=(x, y, width, height),
                    center=(center_x, center_y),
                    area=area,
                    confidence=confidence,
                )
            )

    detections.sort(key=lambda detection: detection["area"], reverse=True)
    return detections


def choose_target(detections, target_color: str = TARGET_COLOR):
    """Return the largest detection matching the requested target color."""

    matching_detections = [
        detection for detection in detections if detection["color"] == target_color
    ]
    if not matching_detections:
        return None

    return max(matching_detections, key=lambda detection: detection["area"])


def get_alignment(frame, target, tolerance: int = ALIGN_TOLERANCE_PIXELS):
    """Return horizontal alignment data for the selected target."""

    height, width = frame.shape[:2]
    frame_center = (width // 2, height // 2)

    if target is None:
        return {
            "status": "no_target",
            "command_hint": "stop",
            "error_x": None,
            "frame_center": frame_center,
            "target_center": None,
            "tolerance": tolerance,
        }

    target_center = target["center"]
    error_x = target_center[0] - frame_center[0]
    if abs(error_x) <= tolerance:
        status = "centered"
        command_hint = "stop"
    elif error_x < 0:
        status = "target_left"
        command_hint = "left"
    else:
        status = "target_right"
        command_hint = "right"

    return {
        "status": status,
        "command_hint": command_hint,
        "error_x": error_x,
        "frame_center": frame_center,
        "target_center": target_center,
        "tolerance": tolerance,
    }


def draw_detections(frame, detections):
    """Return a copy of frame with detection boxes and labels drawn."""

    output = frame.copy()

    for index, detection in enumerate(detections):
        color_name = detection["color"]
        x, y, width, height = detection["bbox"]
        center_x, center_y = detection["center"]
        area = detection["area"]
        box_color = BOX_COLORS[color_name]
        label_y = max(18, y - 8 - (index % 3) * 16)

        cv2.rectangle(output, (x, y), (x + width, y + height), box_color, 2)
        cv2.circle(output, (center_x, center_y), 4, box_color, -1)
        cv2.putText(
            output,
            f"{color_name} {int(area)}",
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2,
            cv2.LINE_AA,
        )

    return output


def draw_alignment(output, target, alignment, target_color: str = TARGET_COLOR):
    """Draw selected target and horizontal alignment hint."""

    frame_center = alignment["frame_center"]
    tolerance = alignment["tolerance"]
    status = alignment["status"]

    cv2.line(
        output,
        (frame_center[0] - tolerance, 0),
        (frame_center[0] - tolerance, output.shape[0]),
        (255, 255, 255),
        1,
    )
    cv2.line(
        output,
        (frame_center[0] + tolerance, 0),
        (frame_center[0] + tolerance, output.shape[0]),
        (255, 255, 255),
        1,
    )

    if target is not None:
        x, y, width, height = target["bbox"]
        target_center = target["center"]
        cv2.rectangle(output, (x, y), (x + width, y + height), (255, 255, 255), 3)
        cv2.line(output, frame_center, target_center, (255, 255, 255), 2)
        cv2.circle(output, target_center, 7, (255, 255, 255), 2)

    cv2.putText(
        output,
        f"TARGET {target_color}: {status}",
        (12, output.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def detect_colors(frame, min_area: int = MIN_COLOR_AREA):
    """Return a copy of frame with colored keg detections highlighted."""

    detections = detect_kegs(frame, min_area=min_area)
    return draw_detections(frame, detections)


def process_frame(frame, fps: float | None = None, command: str | None = None):
    """Return a processed copy of one camera frame."""

    detections = detect_kegs(frame)
    target = choose_target(detections)
    alignment = get_alignment(frame, target)

    output = draw_detections(frame, detections)
    height, width = output.shape[:2]
    center_x = width // 2
    center_y = height // 2

    cv2.line(output, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
    cv2.line(output, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)
    draw_alignment(output, target, alignment)

    label_parts = []
    if fps is not None:
        label_parts.append(f"FPS: {fps:.1f}")
    if command is not None:
        label_parts.append(f"CMD: {command}")
    label_parts.append(f"ALIGN: {alignment['command_hint']}")
    if alignment["error_x"] is not None:
        label_parts.append(f"DX: {alignment['error_x']}")

    if label_parts:
        cv2.putText(
            output,
            " | ".join(label_parts),
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return output
