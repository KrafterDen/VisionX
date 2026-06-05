"""Approximate pin map built from scan observations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geometry import distance, normalize_angle_deg


CONFUSABLE_COLOR_PAIRS = {
    frozenset(("blue", "purple")),
    frozenset(("red", "pink")),
    frozenset(("pink", "purple")),
}


@dataclass
class RobotPose:
    x: float = 0.0
    y: float = 0.0
    heading_deg: float = 0.0

    @property
    def point(self) -> tuple[float, float]:
        return self.x, self.y


@dataclass
class Pin:
    id: int
    color: str
    x: float
    y: float
    angle_deg: float
    distance_cm: float
    confidence: float
    observations: int
    max_area: float
    status: str = "alive"
    is_target: bool = False
    color_votes: dict[str, int] = field(default_factory=dict)

    @property
    def point(self) -> tuple[float, float]:
        return self.x, self.y


@dataclass
class PinMap:
    pins: list[Pin] = field(default_factory=list)
    next_id: int = 1

    def alive_pins(self) -> list[Pin]:
        return [pin for pin in self.pins if pin.status == "alive"]

    def target_pins(self) -> list[Pin]:
        return [pin for pin in self.alive_pins() if pin.is_target]

    def obstacle_pins(self) -> list[Pin]:
        return [pin for pin in self.alive_pins() if not pin.is_target]

    def add_observation(
        self,
        *,
        color: str,
        x: float,
        y: float,
        angle_deg: float,
        distance_cm: float,
        confidence: float,
        area: float,
        target_color: str,
        merge_radius_cm: float,
        cross_color_merge_radius_cm: float | None = None,
        target_min_votes: int = 1,
    ) -> Pin:
        existing = self._find_merge_candidate(
            color,
            (x, y),
            merge_radius_cm,
            cross_color_merge_radius_cm,
        )
        if existing is None:
            pin = Pin(
                id=self.next_id,
                color=color,
                x=x,
                y=y,
                angle_deg=normalize_angle_deg(angle_deg),
                distance_cm=distance_cm,
                confidence=confidence,
                observations=1,
                max_area=area,
                is_target=color == target_color,
                color_votes={color: 1},
            )
            self.next_id += 1
            self.pins.append(pin)
            return pin

        self._merge_observation_into_pin(
            existing,
            color=color,
            x=x,
            y=y,
            angle_deg=angle_deg,
            distance_cm=distance_cm,
            confidence=confidence,
            area=area,
            target_color=target_color,
            target_min_votes=target_min_votes,
        )
        return existing

    def cleanup(
        self,
        *,
        target_color: str,
        same_color_radius_cm: float,
        cross_color_radius_cm: float,
        cleanup_merge_radius_cm: float,
        min_observations: int,
        target_min_votes: int,
    ) -> None:
        self._merge_close_pins(
            target_color=target_color,
            same_color_radius_cm=max(same_color_radius_cm, cleanup_merge_radius_cm),
            cross_color_radius_cm=cross_color_radius_cm,
            target_min_votes=target_min_votes,
        )
        self._drop_weak_pins(min_observations)
        self._merge_close_pins(
            target_color=target_color,
            same_color_radius_cm=max(same_color_radius_cm, cleanup_merge_radius_cm),
            cross_color_radius_cm=cross_color_radius_cm,
            target_min_votes=target_min_votes,
        )
        self._renumber_alive_pins()

    def mark_knocked(self, pin_id: int) -> None:
        for pin in self.pins:
            if pin.id == pin_id:
                pin.status = "knocked"
                return

    def _find_merge_candidate(
        self,
        color: str,
        point: tuple[float, float],
        merge_radius_cm: float,
        cross_color_merge_radius_cm: float | None,
    ) -> Pin | None:
        same_color_matches = [
            pin
            for pin in self.alive_pins()
            if pin.color == color and distance(pin.point, point) < merge_radius_cm
        ]
        if same_color_matches:
            return min(same_color_matches, key=lambda pin: distance(pin.point, point))

        if cross_color_merge_radius_cm is None:
            return None

        cross_color_matches = [
            pin
            for pin in self.alive_pins()
            if _colors_can_cross_merge(pin.color, color)
            and distance(pin.point, point) < cross_color_merge_radius_cm
        ]
        if not cross_color_matches:
            return None
        return min(cross_color_matches, key=lambda pin: distance(pin.point, point))

    def _merge_observation_into_pin(
        self,
        pin: Pin,
        *,
        color: str,
        x: float,
        y: float,
        angle_deg: float,
        distance_cm: float,
        confidence: float,
        area: float,
        target_color: str,
        target_min_votes: int,
    ) -> None:
        weight = pin.observations
        total = weight + 1
        pin.x = (pin.x * weight + x) / total
        pin.y = (pin.y * weight + y) / total
        pin.distance_cm = (pin.distance_cm * weight + distance_cm) / total
        pin.confidence = (pin.confidence * weight + confidence) / total
        pin.angle_deg = normalize_angle_deg(
            pin.angle_deg + normalize_angle_deg(angle_deg - pin.angle_deg) / total
        )
        pin.observations = total
        pin.max_area = max(pin.max_area, area)
        if not pin.color_votes:
            pin.color_votes = {pin.color: max(1, weight)}
        pin.color_votes[color] = pin.color_votes.get(color, 0) + 1
        self._refresh_pin_color(pin, target_color, target_min_votes)

    def _merge_pin_into_pin(self, keep: Pin, drop: Pin, *, target_color: str, target_min_votes: int) -> None:
        keep_weight = keep.observations
        drop_weight = drop.observations
        total = keep_weight + drop_weight
        keep.x = (keep.x * keep_weight + drop.x * drop_weight) / total
        keep.y = (keep.y * keep_weight + drop.y * drop_weight) / total
        keep.distance_cm = (keep.distance_cm * keep_weight + drop.distance_cm * drop_weight) / total
        keep.confidence = (keep.confidence * keep_weight + drop.confidence * drop_weight) / total
        keep.angle_deg = normalize_angle_deg(
            keep.angle_deg + normalize_angle_deg(drop.angle_deg - keep.angle_deg) * drop_weight / total
        )
        keep.observations = total
        keep.max_area = max(keep.max_area, drop.max_area)
        if not keep.color_votes:
            keep.color_votes = {keep.color: max(1, keep_weight)}
        if not drop.color_votes:
            drop.color_votes = {drop.color: max(1, drop_weight)}
        for color, votes in drop.color_votes.items():
            keep.color_votes[color] = keep.color_votes.get(color, 0) + votes
        self._refresh_pin_color(keep, target_color, target_min_votes)

    def _merge_close_pins(
        self,
        *,
        target_color: str,
        same_color_radius_cm: float,
        cross_color_radius_cm: float,
        target_min_votes: int,
    ) -> None:
        changed = True
        while changed:
            changed = False
            alive = self.alive_pins()
            for left_index, left in enumerate(alive):
                for right in alive[left_index + 1 :]:
                    if left.color == right.color:
                        radius = same_color_radius_cm
                    elif _colors_can_cross_merge(left.color, right.color):
                        radius = cross_color_radius_cm
                    else:
                        continue
                    if distance(left.point, right.point) > radius:
                        continue
                    keep, drop = self._best_pin_to_keep(left, right)
                    self._merge_pin_into_pin(keep, drop, target_color=target_color, target_min_votes=target_min_votes)
                    self.pins.remove(drop)
                    changed = True
                    break
                if changed:
                    break

    def _drop_weak_pins(self, min_observations: int) -> None:
        if min_observations <= 1:
            return
        self.pins = [
            pin
            for pin in self.pins
            if pin.status != "alive" or pin.observations >= min_observations
        ]

    def _renumber_alive_pins(self) -> None:
        for index, pin in enumerate(self.pins, start=1):
            pin.id = index
        self.next_id = len(self.pins) + 1

    def _best_pin_to_keep(self, left: Pin, right: Pin) -> tuple[Pin, Pin]:
        left_score = (left.observations, left.max_area, left.confidence)
        right_score = (right.observations, right.max_area, right.confidence)
        if left_score >= right_score:
            return left, right
        return right, left

    def _refresh_pin_color(self, pin: Pin, target_color: str, target_min_votes: int) -> None:
        if pin.color_votes.get(target_color, 0) >= target_min_votes:
            pin.color = target_color
            pin.is_target = True
            return
        pin.color = max(pin.color_votes, key=lambda color: pin.color_votes[color])
        pin.is_target = pin.color == target_color


def _colors_can_cross_merge(left: str, right: str) -> bool:
    if left == right:
        return True
    return frozenset((left, right)) in CONFUSABLE_COLOR_PAIRS


def estimate_distance_cm(detection, k_distance: float) -> float:
    return k_distance / max(float(detection["bbox"][3]), 1.0)


def detection_to_map_observation(
    detection,
    *,
    robot_pose: RobotPose,
    scan_heading_deg: float,
    frame_width: int,
    camera_horizontal_fov_deg: float,
    k_distance: float,
) -> dict:
    cx = float(detection["cx"])
    camera_offset_angle = ((cx - frame_width / 2.0) / frame_width) * camera_horizontal_fov_deg
    global_angle = normalize_angle_deg(scan_heading_deg + camera_offset_angle)
    distance_cm = estimate_distance_cm(detection, k_distance)
    angle_rad = math.radians(global_angle)
    x = robot_pose.x + distance_cm * math.sin(angle_rad)
    y = robot_pose.y + distance_cm * math.cos(angle_rad)
    return {
        "color": detection["color"],
        "x": x,
        "y": y,
        "angle_deg": global_angle,
        "distance_cm": distance_cm,
        "confidence": float(detection["confidence"]),
        "area": float(detection["area"]),
    }
