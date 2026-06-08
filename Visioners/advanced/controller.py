"""Advanced MVP state machine for autonomous colored-pin kegelring."""

from __future__ import annotations

import math
import time
from enum import Enum

import cv2

from .config import AdvancedConfig
from .debug import (
    RunLogger,
    draw_advanced_overlay,
    save_pin_map,
)
from .detection_stabilizer import DetectionStabilizer
from .geometry import compute_heading, distance, turn_angle
from .mapping import (
    Pin,
    PinMap,
    RobotPose,
    estimate_distance_cm,
)
from .planner import PlannedPath, plan_path, select_target
from .robot_io import RobotIO
from .scanner import FieldScanner
from .vision import detect_kegs


class State(str, Enum):
    INIT = "INIT"
    INITIAL_SCAN = "INITIAL_SCAN"
    BUILD_MAP = "BUILD_MAP"
    SELECT_TARGET = "SELECT_TARGET"
    PLAN_PATH = "PLAN_PATH"
    TURN_TO_GOAL_SECTOR = "TURN_TO_GOAL_SECTOR"
    LOCAL_LOCK_TARGET = "LOCAL_LOCK_TARGET"
    GO_TO_WAYPOINT = "GO_TO_WAYPOINT"
    APPROACH_TARGET = "APPROACH_TARGET"
    ATTACK = "ATTACK"
    BACKUP_AND_UPDATE = "BACKUP_AND_UPDATE"
    RESCAN = "RESCAN"
    DONE = "DONE"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class AdvancedController:
    def __init__(
        self,
        *,
        config: AdvancedConfig,
        robot: RobotIO,
        frame_source,
        logger: RunLogger,
    ) -> None:
        self.config = config
        self.robot = robot
        self.frame_source = frame_source
        self.logger = logger
        self.pose = RobotPose()
        self.pin_map = PinMap()
        self.state = State.INIT
        self.knocked_count = 0
        self.last_frame_time = time.monotonic()
        self.last_command = "stop"
        self.forward_allowed = True
        self.detection_stabilizer = DetectionStabilizer(config)
        now = time.time()
        self.run_timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        self.run_timestamp = f"{self.run_timestamp}_{int((now % 1) * 1000):03d}"
        self.scanner = FieldScanner(
            config=config,
            robot=robot,
            pose=self.pose,
            logger=logger,
            detection_stabilizer=self.detection_stabilizer,
            read_frame=self._read_frame,
            debug_frame=self._debug_scan_frame,
            set_last_command=self._set_last_command,
            run_timestamp=self.run_timestamp,
        )

    def run(self) -> None:
        try:
            self._transition(State.INIT, "start")
            self.robot.connect()
            if hasattr(self.frame_source, "open"):
                self.frame_source.open()
            self._validate_camera()

            self.pin_map = self.initial_scan()
            self._transition(State.BUILD_MAP, "scan complete")
            self._resolve_target_count()
            self._log_map()
            self._save_map("initial_scan")
            if self.config.scan_only:
                self.logger.log("scan_only_done", pins=len(self.pin_map.alive_pins()))
                self._transition(State.DONE, "scan-only complete")
                self.robot.stop()
                return

            rescan_attempts = 0
            waypoint_failures = 0
            assert self.config.target_count is not None
            while self.knocked_count < self.config.target_count:
                target = self._select_target_or_rescan()
                if target is None:
                    break

                path = self._plan_for_target(target)
                if path.waypoint is not None:
                    if not self.go_to_waypoint(path):
                        waypoint_failures += 1
                        if waypoint_failures > self.config.max_waypoint_failures:
                            self.logger.log(
                                "stop_after_waypoint_failures",
                                failures=waypoint_failures,
                                max_failures=self.config.max_waypoint_failures,
                            )
                            break
                        self.pin_map = self.rescan()
                        continue
                    path = self._plan_for_target(target)

                self._transition(State.TURN_TO_GOAL_SECTOR, f"target #{target.id}")
                self.turn_to_point(path.current_goal)

                self._transition(State.LOCAL_LOCK_TARGET, f"target #{target.id}")
                target_detection = self.local_lock_target()
                if target_detection is None:
                    rescan_attempts += 1
                    if rescan_attempts > self.config.max_rescan_attempts:
                        self.logger.log(
                            "stop_after_rescan_attempts",
                            reason="local lock failed",
                            attempts=rescan_attempts,
                            max_attempts=self.config.max_rescan_attempts,
                        )
                        break
                    self.pin_map = self.rescan()
                    continue

                self._transition(State.APPROACH_TARGET, f"target #{target.id}")
                target_detection = self.approach_target()
                if target_detection is None:
                    rescan_attempts += 1
                    if rescan_attempts > self.config.max_rescan_attempts:
                        self.logger.log(
                            "stop_after_rescan_attempts",
                            reason="approach failed",
                            attempts=rescan_attempts,
                            max_attempts=self.config.max_rescan_attempts,
                        )
                        break
                    self.pin_map = self.rescan()
                    continue

                self._transition(State.ATTACK, f"target #{target.id}")
                if not self.attack(target_detection):
                    rescan_attempts += 1
                    if rescan_attempts > self.config.max_rescan_attempts:
                        self.logger.log(
                            "stop_after_rescan_attempts",
                            reason="attack gate failed",
                            attempts=rescan_attempts,
                            max_attempts=self.config.max_rescan_attempts,
                        )
                        break
                    self.pin_map = self.rescan()
                    continue

                self._transition(State.BACKUP_AND_UPDATE, f"target #{target.id}")
                self.backup()
                self.pin_map.mark_knocked(target.id)
                self.knocked_count += 1
                self._save_map("after_knock")
                rescan_attempts = 0
                waypoint_failures = 0
                self.logger.log("knocked", target_id=target.id, knocked_count=self.knocked_count)

            self._transition(State.DONE, "target count reached or no target")
            self._save_map("final")
            self.robot.stop()
        except KeyboardInterrupt:
            self._transition(State.EMERGENCY_STOP, "keyboard interrupt")
            self.robot.stop()
            raise
        except Exception as exc:
            self._transition(State.EMERGENCY_STOP, f"exception: {exc}")
            self.robot.stop()
            if self.config.stop_on_exception:
                raise

    def _resolve_target_count(self) -> None:
        if self.config.target_count is not None:
            return
        self.config.target_count = len(self.pin_map.target_pins())
        self.logger.log(
            "target_count_detected",
            target_color=self.config.target_color,
            target_count=self.config.target_count,
        )

    def initial_scan(self) -> PinMap:
        self._transition(State.INITIAL_SCAN, "sweeping field")
        return self.scanner.scan()

    def rescan(self) -> PinMap:
        self._transition(State.RESCAN, "rebuilding approximate map")
        pin_map = self.initial_scan()
        self._save_map("rescan", pin_map)
        return pin_map

    def go_to_waypoint(self, path: PlannedPath) -> bool:
        waypoint = path.waypoint
        if waypoint is None:
            return True
        self._transition(State.GO_TO_WAYPOINT, f"waypoint {waypoint}")
        self.turn_to_point(waypoint)
        deadline = time.monotonic() + max(1.0, distance(self.pose.point, waypoint) / 8.0)
        while distance(self.pose.point, waypoint) > 8.0 and time.monotonic() < deadline:
            ok, frame = self._read_frame()
            if not ok:
                return False
            detections = self._detect_stable_kegs(frame)
            if not self._detections_ready():
                self.robot.stop()
                self._debug(frame, detections, pin_map=self.pin_map)
                continue
            if not self.forward_is_safe(detections, target_detection=None):
                self.robot.stop()
                self.logger.log("waypoint_blocked")
                return False
            self.robot.safe_forward_pulse(self.config.waypoint_pulse_sec)
            self.last_command = "forward"
            self._advance_pose(self.config.waypoint_pulse_sec)
            self._debug(frame, detections, pin_map=self.pin_map)
        self.robot.stop()
        return distance(self.pose.point, waypoint) <= 12.0

    def turn_to_point(self, point: tuple[float, float]) -> None:
        desired = compute_heading(self.pose.point, point)
        delta = turn_angle(self.pose.heading_deg, desired)
        if abs(delta) < 4.0:
            self.pose.heading_deg = desired
            return
        command = "right" if delta > 0 else "left"
        duration = abs(delta) / max(self.config.angular_speed_deg_per_sec_at_scan_pwm, 1.0)
        self.robot.timed_command(command, duration, self.config.turn_pwm)
        self.last_command = command
        self.pose.heading_deg = desired

    def local_lock_target(self):
        self.detection_stabilizer.reset()
        stable_frames = 0
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            ok, frame = self._read_frame()
            if not ok:
                continue
            detections = self._detect_stable_kegs(frame)
            if not self._detections_ready():
                self.robot.stop()
                self._debug(frame, detections)
                continue
            target = self.choose_target_detection(detections)
            if target is None:
                self.robot.timed_command("left", self.config.turn_pulse_sec, self.config.turn_pwm)
                self.last_command = "left"
                continue
            error = self.alignment_error(target)
            if abs(error) <= self.config.center_tolerance:
                stable_frames += 1
                self.robot.stop()
                if stable_frames >= self.config.target_confirm_frames:
                    self._debug(frame, detections, target_detection=target)
                    return target
            else:
                stable_frames = 0
                command = "left" if error < 0 else "right"
                self.robot.timed_command(command, self.config.turn_pulse_sec, self.config.turn_pwm)
                self.last_command = command
            self._debug(frame, detections, target_detection=target)
        return None

    def approach_target(self):
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            ok, frame = self._read_frame()
            if not ok:
                continue
            detections = self._detect_stable_kegs(frame)
            if not self._detections_ready():
                self.robot.stop()
                self._debug(frame, detections)
                continue
            target = self.choose_target_detection(detections)
            if target is None:
                self.robot.stop()
                return None
            if self.attack_ready(target, detections):
                self.robot.stop()
                self._debug(frame, detections, target_detection=target)
                return target
            error = self.alignment_error(target)
            if abs(error) > self.config.center_tolerance:
                command = "left" if error < 0 else "right"
                self.robot.timed_command(command, self.config.turn_pulse_sec, self.config.turn_pwm)
                self.last_command = command
            elif self.forward_is_safe(detections, target):
                self.robot.safe_forward_pulse(self.config.forward_pulse_sec)
                self.last_command = "forward"
                self._advance_pose(self.config.forward_pulse_sec)
            else:
                self.robot.stop()
                self.logger.log("approach_blocked")
                return None
            self._debug(frame, detections, target_detection=target)
        return None

    def attack(self, target_detection) -> bool:
        ok, frame = self._read_frame()
        if not ok:
            return False
        detections = self._detect_stable_kegs(frame)
        target = self.choose_target_detection(detections) or target_detection
        if not self.attack_ready(target, detections):
            self.robot.stop()
            return False
        self.robot.timed_command("forward", self.config.attack_duration_sec, self.config.attack_pwm)
        self.last_command = "forward"
        self._advance_pose(self.config.attack_duration_sec, speed_cm_s=self.config.approximate_forward_speed_cm_per_sec_at_approach_pwm * 1.3)
        return True

    def backup(self) -> None:
        self.robot.timed_command("backward", self.config.backup_duration_sec, self.config.backup_pwm)
        heading_rad = math.radians(self.pose.heading_deg)
        distance_cm = self.config.approximate_forward_speed_cm_per_sec_at_approach_pwm * self.config.backup_duration_sec
        self.pose.x -= distance_cm * math.sin(heading_rad)
        self.pose.y -= distance_cm * math.cos(heading_rad)
        self.last_command = "backward"

    def choose_target_detection(self, detections):
        matches = [detection for detection in detections if detection["color"] == self.config.target_color]
        if not matches:
            return None
        return max(matches, key=lambda detection: (detection["bbox"][3], detection["area"]))

    def alignment_error(self, detection) -> float:
        return (float(detection["cx"]) - self.config.frame_width / 2.0) / (self.config.frame_width / 2.0)

    def attack_ready(self, target_detection, detections) -> bool:
        if target_detection is None:
            return False
        if abs(self.alignment_error(target_detection)) > self.config.center_tolerance:
            return False
        distance_to_pin = estimate_distance_cm(target_detection, self.config.k_distance)
        distance_to_front = distance_to_pin - self.config.robot_front_offset_cm
        bbox_h = target_detection["bbox"][3]
        close_enough = (
            distance_to_front <= self.config.attack_distance_cm
            or bbox_h >= self.config.attack_bbox_height_threshold_px
        )
        return close_enough and self.forward_is_safe(detections, target_detection)

    def forward_is_safe(self, detections, target_detection=None) -> bool:
        target_distance = (
            estimate_distance_cm(target_detection, self.config.k_distance)
            if target_detection is not None
            else float("inf")
        )
        left = self.config.frame_width * 0.35
        right = self.config.frame_width * 0.65
        for detection in detections:
            if detection["color"] == self.config.target_color:
                continue
            cx = detection["cx"]
            if not (left <= cx <= right):
                continue
            bad_distance = estimate_distance_cm(detection, self.config.k_distance)
            if bad_distance <= target_distance + 12.0:
                self.forward_allowed = False
                return False
        self.forward_allowed = True
        return True

    def _select_target_or_rescan(self) -> Pin | None:
        self._transition(State.SELECT_TARGET, "choosing alive target")
        target = select_target(self.pin_map, self.pose, self.config)
        if target is not None:
            self.logger.log("target_selected", target=target)
            return target
        self.pin_map = self.rescan()
        target = select_target(self.pin_map, self.pose, self.config)
        if target is None:
            self.logger.log("no_target")
        return target

    def _plan_for_target(self, target: Pin) -> PlannedPath:
        self._transition(State.PLAN_PATH, f"target #{target.id}")
        path = plan_path(self.pose, target, self.pin_map.obstacle_pins(), self.config)
        self.logger.log(
            "planned_path",
            target_id=target.id,
            goal=path.current_goal,
            waypoint=path.waypoint,
            direct=path.direct_path_safe,
            blocker=path.blocking_obstacle,
            cost=path.cost,
        )
        return path

    def _detect_stable_kegs(self, frame):
        detections = detect_kegs(frame, min_area=self.config.min_contour_area)
        return self.detection_stabilizer.filter(detections)

    def _detections_ready(self) -> bool:
        return self.detection_stabilizer.ready

    def _advance_pose(self, duration_sec: float, speed_cm_s: float | None = None) -> None:
        speed = self.config.approximate_forward_speed_cm_per_sec_at_approach_pwm if speed_cm_s is None else speed_cm_s
        distance_cm = speed * duration_sec
        heading_rad = math.radians(self.pose.heading_deg)
        self.pose.x += distance_cm * math.sin(heading_rad)
        self.pose.y += distance_cm * math.cos(heading_rad)

    def _read_frame(self):
        ok, frame = self.frame_source.read()
        if ok:
            self.last_frame_time = time.monotonic()
            return True, frame
        if time.monotonic() - self.last_frame_time > self.config.max_no_frame_time_sec:
            self._transition(State.EMERGENCY_STOP, "video frame timeout")
            self.robot.stop()
        return False, None

    def _validate_camera(self) -> None:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            ok, _frame = self._read_frame()
            if ok:
                return
        raise RuntimeError("No camera frames received")

    def _debug(self, frame, detections, *, target_detection=None, pin_map=None) -> None:
        if not self.config.debug:
            return
        overlay = draw_advanced_overlay(
            frame,
            detections,
            state=self.state.value,
            target_color=self.config.target_color,
            knocked_count=self.knocked_count,
            target_count=self.config.target_count,
            command=self.last_command,
            forward_allowed=self.forward_allowed,
            target_detection=target_detection,
            pin_map=pin_map or self.pin_map,
        )
        cv2.imshow("Advanced autonomous kegelring", overlay)
        key = cv2.waitKey(1) & 0xFF
        if key in {ord("q"), ord("Q"), 27, ord("x"), ord("X")}:
            raise KeyboardInterrupt

    def _debug_scan_frame(self, frame, detections, pin_map: PinMap) -> None:
        self._debug(frame, detections, pin_map=pin_map)

    def _set_last_command(self, command: str) -> None:
        self.last_command = command

    def _transition(self, next_state: State, reason: str) -> None:
        if self.state != next_state:
            self.logger.log("state", old=self.state.value, new=next_state.value, reason=reason)
        self.state = next_state

    def _log_map(self) -> None:
        for pin in self.pin_map.pins:
            self.logger.log("pin", pin=pin)

    def _save_map(self, stage: str, pin_map: PinMap | None = None) -> None:
        path = save_pin_map(
            self.pin_map if pin_map is None else pin_map,
            pose=self.pose,
            config=self.config,
            stage=stage,
            knocked_count=self.knocked_count,
        )
        if path is not None:
            self.logger.log("map_saved", stage=stage, path=str(path))
