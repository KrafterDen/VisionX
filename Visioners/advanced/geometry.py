"""Geometry helpers for approximate robot mapping and path planning."""

from __future__ import annotations

import math
from typing import Protocol


Point = tuple[float, float]


class HasPosition(Protocol):
    x: float
    y: float


def as_point(value: Point | HasPosition) -> Point:
    if isinstance(value, tuple):
        return float(value[0]), float(value[1])
    return float(value.x), float(value.y)


def distance(a: Point | HasPosition, b: Point | HasPosition) -> float:
    ax, ay = as_point(a)
    bx, by = as_point(b)
    return math.hypot(ax - bx, ay - by)


def point_to_segment_distance(
    point: Point | HasPosition,
    segment_start: Point | HasPosition,
    segment_end: Point | HasPosition,
) -> float:
    px, py = as_point(point)
    ax, ay = as_point(segment_start)
    bx, by = as_point(segment_end)
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    closest = (ax + t * dx, ay + t * dy)
    return distance((px, py), closest)


def segment_intersects_forbidden_zone(
    segment_start: Point | HasPosition,
    segment_end: Point | HasPosition,
    obstacle_pin: Point | HasPosition,
    forbidden_radius: float,
) -> bool:
    return point_to_segment_distance(obstacle_pin, segment_start, segment_end) < forbidden_radius


def compute_heading(from_pos: Point | HasPosition, to_pos: Point | HasPosition) -> float:
    """Heading where 0 deg is +Y and positive is right."""

    fx, fy = as_point(from_pos)
    tx, ty = as_point(to_pos)
    return normalize_angle_deg(math.degrees(math.atan2(tx - fx, ty - fy)))


def normalize_angle_deg(angle: float) -> float:
    """Normalize angle to [-180, 180)."""

    return ((angle + 180.0) % 360.0) - 180.0


def turn_angle(current_heading: float, desired_heading: float) -> float:
    return normalize_angle_deg(desired_heading - current_heading)


def generate_waypoint_candidates(
    robot_pos: Point | HasPosition,
    target_pos: Point | HasPosition,
    blocking_obstacle: Point | HasPosition,
    forbidden_radius: float,
    waypoint_extra_margin: float,
) -> list[Point]:
    rx, ry = as_point(robot_pos)
    tx, ty = as_point(target_pos)
    ox, oy = as_point(blocking_obstacle)
    vx = tx - rx
    vy = ty - ry
    length = math.hypot(vx, vy)
    if length == 0:
        return []
    nx = -vy / length
    ny = vx / length
    offset = forbidden_radius + waypoint_extra_margin
    return [(ox + nx * offset, oy + ny * offset), (ox - nx * offset, oy - ny * offset)]

