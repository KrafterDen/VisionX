"""Debug overlay and lightweight JSONL logging for advanced autonomous runs."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .vision import BOX_COLORS, draw_detections


class RunLogger:
    def __init__(self, path: str | None) -> None:
        self.path = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8") if self.path else None

    def log(self, event: str, **data: Any) -> None:
        payload = {"time": time.time(), "event": event, **data}
        print(f"[{event}] {data}")
        if self._file is None:
            return
        self._file.write(json.dumps(_jsonable(payload), ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return value


def save_pin_map(
    pin_map,
    *,
    pose,
    config,
    stage: str,
    knocked_count: int,
) -> Path | None:
    if not config.save_map:
        return None

    output_dir = Path(config.map_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    timestamp = f"{timestamp}_{int((now % 1) * 1000):03d}"
    path = output_dir / f"map_{timestamp}_{stage}.json"
    image_path = output_dir / f"map_{timestamp}_{stage}.png"
    latest_path = output_dir / "latest_map.json"
    latest_image_path = output_dir / "latest_map.png"
    payload = {
        "saved_at": now,
        "stage": stage,
        "target_color": config.target_color,
        "target_count": config.target_count,
        "knocked_count": knocked_count,
        "pose": pose,
        "alive_count": len(pin_map.alive_pins()),
        "target_alive_count": len(pin_map.target_pins()),
        "obstacle_alive_count": len(pin_map.obstacle_pins()),
        "pins": pin_map.pins,
        "tuning": {
            "k_distance": config.k_distance,
            "merge_radius_cm": config.merge_radius_cm,
            "cross_color_merge_radius_cm": config.cross_color_merge_radius_cm,
            "map_cleanup_merge_radius_cm": config.map_cleanup_merge_radius_cm,
            "map_min_observations": config.map_min_observations,
            "map_target_min_votes": config.map_target_min_votes,
            "forbidden_radius_cm": config.forbidden_radius_cm,
            "camera_horizontal_fov_deg": config.camera_horizontal_fov_deg,
        },
    }
    text = json.dumps(_jsonable(payload), ensure_ascii=False, indent=2)
    path.write_text(text + "\n", encoding="utf-8")
    latest_path.write_text(text + "\n", encoding="utf-8")
    image = draw_pin_map_image(pin_map, pose=pose, config=config, stage=stage, knocked_count=knocked_count)
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(latest_image_path), image)
    return path


def save_detection_decision_image(
    frame,
    *,
    detection,
    pin,
    observation: dict,
    config,
    run_timestamp: str,
    sequence: int,
) -> Path | None:
    if not config.save_detection_debug_images:
        return None

    output_dir = Path(config.detection_debug_dir) / run_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    color = BOX_COLORS.get(pin.color, (255, 255, 255))
    x, y, width, height = detection["bbox"]
    image = frame.copy()

    cv2.rectangle(image, (x, y), (x + width, y + height), color, 3)
    cv2.circle(image, detection["center"], 6, color, -1, cv2.LINE_AA)

    lines = [
        f"PIN #{pin.id} DECIDED: {pin.color}",
        f"raw={detection['color']}  target={pin.is_target}",
        f"obs={pin.observations}  votes={pin.color_votes}",
        f"bbox=({x},{y},{width},{height}) area={detection['area']:.0f}",
        f"map x={pin.x:.1f} y={pin.y:.1f} dist={pin.distance_cm:.1f}cm",
        f"scan angle={observation['angle_deg']:.1f}",
    ]
    _draw_label_panel(image, lines)

    filename = (
        f"{sequence:04d}_pin{pin.id}_{pin.color}_"
        f"obs{pin.observations}_{int(time.time() * 1000)}.jpg"
    )
    path = output_dir / filename
    cv2.imwrite(str(path), image)
    return path


def _draw_label_panel(image, lines: list[str]) -> None:
    line_height = 22
    panel_height = 18 + line_height * len(lines)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], panel_height), (0, 0, 0), -1)
    image[:] = cv2.addWeighted(overlay, 0.62, image, 0.38, 0)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (12, 24 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_pin_map_image(
    pin_map,
    *,
    pose,
    config,
    stage: str,
    knocked_count: int,
):
    size = int(config.map_image_size_px)
    image = np.full((size, size, 3), 245, dtype=np.uint8)

    alive_pins = pin_map.alive_pins()
    points = [(0.0, 0.0), (float(pose.x), float(pose.y))]
    points.extend((pin.x, pin.y) for pin in pin_map.pins)
    max_abs_x = max(abs(point[0]) for point in points) + config.map_padding_cm
    max_y = max(point[1] for point in points) + config.map_padding_cm
    min_y = min(point[1] for point in points) - config.map_padding_cm
    world_width = max(80.0, max_abs_x * 2.0)
    world_height = max(100.0, max_y - min_y)
    scale = min((size - 90) / world_width, (size - 120) / world_height)
    center_x_px = size // 2
    bottom_margin_px = 65
    origin_y_px = int(size - bottom_margin_px + min(0.0, min_y) * scale)

    def world_to_px(x_cm: float, y_cm: float) -> tuple[int, int]:
        return int(center_x_px + x_cm * scale), int(origin_y_px - y_cm * scale)

    def cm_to_px(value_cm: float) -> int:
        return max(1, int(value_cm * scale))

    _draw_grid(image, world_to_px, scale, min_y, max_y, max_abs_x)
    _draw_axes(image, world_to_px, max_abs_x, min_y, max_y)

    for pin in alive_pins:
        if pin.is_target:
            continue
        center = world_to_px(pin.x, pin.y)
        radius = cm_to_px(config.forbidden_radius_cm)
        overlay = image.copy()
        cv2.circle(overlay, center, radius, (80, 80, 255), -1, cv2.LINE_AA)
        image[:] = cv2.addWeighted(overlay, 0.18, image, 0.82, 0)
        cv2.circle(image, center, radius, (80, 80, 220), 1, cv2.LINE_AA)

    for pin in pin_map.pins:
        center = world_to_px(pin.x, pin.y)
        color = BOX_COLORS.get(pin.color, (80, 80, 80))
        radius = 11 if pin.status == "alive" else 8
        thickness = -1 if pin.status == "alive" else 2
        cv2.circle(image, center, radius, color, thickness, cv2.LINE_AA)
        outline = (255, 255, 255) if pin.is_target else (40, 40, 40)
        cv2.circle(image, center, radius + 3, outline, 2, cv2.LINE_AA)
        label = f"#{pin.id} {pin.color}"
        if pin.status != "alive":
            label += f" {pin.status}"
        cv2.putText(
            image,
            label,
            (center[0] + 12, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"{pin.distance_cm:.0f}cm",
            (center[0] + 12, center[1] + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )

    _draw_robot(image, world_to_px, pose, config, scale)
    _draw_map_legend(image, config, stage, knocked_count, pin_map)
    return image


def _draw_grid(image, world_to_px, scale: float, min_y: float, max_y: float, max_abs_x: float) -> None:
    step_cm = 25.0 if scale >= 4.0 else 50.0
    x = -math.ceil(max_abs_x / step_cm) * step_cm
    while x <= max_abs_x:
        top = world_to_px(x, max_y)
        bottom = world_to_px(x, min_y)
        cv2.line(image, top, bottom, (225, 225, 225), 1)
        x += step_cm
    y = math.floor(min_y / step_cm) * step_cm
    while y <= max_y:
        left = world_to_px(-max_abs_x, y)
        right = world_to_px(max_abs_x, y)
        cv2.line(image, left, right, (225, 225, 225), 1)
        y += step_cm


def _draw_axes(image, world_to_px, max_abs_x: float, min_y: float, max_y: float) -> None:
    cv2.line(image, world_to_px(-max_abs_x, 0), world_to_px(max_abs_x, 0), (170, 170, 170), 2)
    cv2.line(image, world_to_px(0, min_y), world_to_px(0, max_y), (170, 170, 170), 2)
    cv2.putText(image, "FORWARD", world_to_px(4, max_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(image, "RIGHT", world_to_px(max_abs_x - 28, -5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1, cv2.LINE_AA)


def _draw_robot(image, world_to_px, pose, config, scale: float) -> None:
    center = np.array(world_to_px(float(pose.x), float(pose.y)), dtype=np.float32)
    heading = math.radians(float(pose.heading_deg))
    forward = np.array([math.sin(heading), -math.cos(heading)], dtype=np.float32)
    right = np.array([math.cos(heading), math.sin(heading)], dtype=np.float32)
    length = max(22.0, config.robot_length_cm * scale)
    width = max(18.0, config.robot_width_cm * scale)
    tip = center + forward * length * 0.7
    rear_left = center - forward * length * 0.45 - right * width * 0.5
    rear_right = center - forward * length * 0.45 + right * width * 0.5
    pts = np.array([tip, rear_right, rear_left], dtype=np.int32)
    cv2.fillConvexPoly(image, pts, (255, 255, 255), cv2.LINE_AA)
    cv2.polylines(image, [pts], True, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.circle(image, tuple(center.astype(int)), 4, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.putText(image, "ROBOT", tuple((center + np.array([10, 22])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)


def _draw_map_legend(image, config, stage: str, knocked_count: int, pin_map) -> None:
    lines = [
        f"MAP: {stage}",
        f"TARGET: {config.target_color}  KNOCKED: {knocked_count}/{config.target_count}",
        f"ALIVE: {len(pin_map.alive_pins())}  TARGETS: {len(pin_map.target_pins())}  OBSTACLES: {len(pin_map.obstacle_pins())}",
        f"K={config.k_distance:.0f}  forbidden={config.forbidden_radius_cm:.0f}cm",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (18, 28 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (20, 20, 20),
            2 if index == 0 else 1,
            cv2.LINE_AA,
        )


def draw_advanced_overlay(
    frame,
    detections,
    *,
    state: str,
    target_color: str,
    knocked_count: int,
    target_count: int,
    command: str,
    forward_allowed: bool,
    selected_pin=None,
    target_detection=None,
    pin_map=None,
):
    output = draw_detections(frame, detections)
    height, width = output.shape[:2]
    center_x = width // 2
    corridor_left = int(width * 0.35)
    corridor_right = int(width * 0.65)
    color = BOX_COLORS.get(target_color, (255, 255, 255))

    cv2.line(output, (center_x, 0), (center_x, height), (255, 255, 255), 1)
    cv2.line(output, (corridor_left, 0), (corridor_left, height), (0, 255, 255), 1)
    cv2.line(output, (corridor_right, 0), (corridor_right, height), (0, 255, 255), 1)

    if target_detection is not None:
        x, y, w, h = target_detection["bbox"]
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 255, 255), 3)
        cv2.circle(output, target_detection["center"], 7, (255, 255, 255), 2)

    lines = [
        f"STATE: {state}",
        f"TARGET: {target_color}  COUNT: {knocked_count}/{target_count}",
        f"CMD: {command}  FORWARD: {'OK' if forward_allowed else 'BLOCK'}",
    ]
    if selected_pin is not None:
        lines.append(
            f"PIN: #{selected_pin.id} {selected_pin.color} "
            f"x={selected_pin.x:.0f} y={selected_pin.y:.0f}"
        )
    if pin_map is not None:
        lines.append(f"MAP PINS: {len(pin_map.alive_pins())}")

    for index, text in enumerate(lines):
        cv2.putText(
            output,
            text,
            (12, 24 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )

    return output
