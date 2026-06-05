"""Target selection and simple forbidden-zone path planning."""

from __future__ import annotations

from dataclasses import dataclass

from .config import AdvancedConfig
from .geometry import (
    Point,
    compute_heading,
    distance,
    generate_waypoint_candidates,
    point_to_segment_distance,
    turn_angle,
)
from .mapping import Pin, PinMap, RobotPose


@dataclass(frozen=True)
class PlannedPath:
    target_pin: Pin
    current_goal: Point
    waypoint: Point | None
    blocking_obstacle: Pin | None
    direct_path_safe: bool
    cost: float


def path_clearance(
    start: Point,
    end: Point,
    obstacle: Pin,
    forbidden_radius: float,
) -> float:
    return point_to_segment_distance(obstacle.point, start, end) - forbidden_radius


def path_blockers(
    start: Point,
    end: Point,
    obstacles: list[Pin],
    forbidden_radius: float,
) -> list[tuple[Pin, float]]:
    blockers = []
    for obstacle in obstacles:
        clearance = path_clearance(start, end, obstacle, forbidden_radius)
        if clearance < 0:
            blockers.append((obstacle, clearance))
    return sorted(blockers, key=lambda item: item[1])


def obstacle_risk(
    start: Point,
    end: Point,
    obstacles: list[Pin],
    config: AdvancedConfig,
) -> float:
    risk = 0.0
    for obstacle in obstacles:
        clearance = path_clearance(start, end, obstacle, config.forbidden_radius_cm)
        if clearance < 0:
            risk += config.obstacle_hard_risk + abs(clearance) * 20.0
        elif clearance < config.safety_margin_cm:
            risk += (config.safety_margin_cm - clearance) * 20.0
    return risk


def target_cost(
    pose: RobotPose,
    target: Pin,
    obstacles: list[Pin],
    config: AdvancedConfig,
) -> float:
    heading = compute_heading(pose.point, target.point)
    distance_cost = distance(pose.point, target.point)
    turn_cost = abs(turn_angle(pose.heading_deg, heading)) * 0.25
    return distance_cost + turn_cost + obstacle_risk(pose.point, target.point, obstacles, config)


def select_target(pin_map: PinMap, pose: RobotPose, config: AdvancedConfig) -> Pin | None:
    targets = pin_map.target_pins()
    if not targets:
        return None
    obstacles = pin_map.obstacle_pins()
    return min(targets, key=lambda pin: target_cost(pose, pin, obstacles, config))


def candidate_inside_forbidden_zone(
    candidate: Point,
    obstacles: list[Pin],
    forbidden_radius: float,
) -> bool:
    return any(distance(candidate, obstacle.point) < forbidden_radius for obstacle in obstacles)


def plan_path(
    pose: RobotPose,
    target: Pin,
    obstacles: list[Pin],
    config: AdvancedConfig,
) -> PlannedPath:
    start = pose.point
    end = target.point
    blockers = path_blockers(start, end, obstacles, config.forbidden_radius_cm)
    direct_cost = target_cost(pose, target, obstacles, config)
    if not blockers:
        return PlannedPath(
            target_pin=target,
            current_goal=end,
            waypoint=None,
            blocking_obstacle=None,
            direct_path_safe=True,
            cost=direct_cost,
        )

    blocking_obstacle = blockers[0][0]
    candidates = generate_waypoint_candidates(
        start,
        end,
        blocking_obstacle.point,
        config.forbidden_radius_cm,
        config.waypoint_extra_margin_cm,
    )
    best_candidate = None
    best_cost = float("inf")
    for candidate in candidates:
        if candidate_inside_forbidden_zone(candidate, obstacles, config.forbidden_radius_cm):
            continue
        route_cost = distance(start, candidate) + distance(candidate, end)
        route_cost += abs(turn_angle(pose.heading_deg, compute_heading(start, candidate))) * 0.25
        route_cost += obstacle_risk(start, candidate, obstacles, config) * 0.5
        route_cost += obstacle_risk(candidate, end, obstacles, config) * 0.5
        if route_cost < best_cost:
            best_cost = route_cost
            best_candidate = candidate

    if best_candidate is None:
        return PlannedPath(
            target_pin=target,
            current_goal=end,
            waypoint=None,
            blocking_obstacle=blocking_obstacle,
            direct_path_safe=False,
            cost=direct_cost,
        )

    return PlannedPath(
        target_pin=target,
        current_goal=best_candidate,
        waypoint=best_candidate,
        blocking_obstacle=blocking_obstacle,
        direct_path_safe=False,
        cost=best_cost,
    )

