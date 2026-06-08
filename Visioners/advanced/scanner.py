"""Initial field scan and map-building logic."""

from __future__ import annotations

import time
from collections.abc import Callable

from .config import AdvancedConfig
from .debug import RunLogger, save_detection_decision_image
from .detection_stabilizer import DetectionStabilizer
from .mapping import (
    Pin,
    PinMap,
    RobotPose,
    detection_to_map_observation,
)
from .robot_io import RobotIO
from .vision import detect_kegs


class FieldScanner:
    """Build an approximate pin map by sweeping the camera across the field."""

    def __init__(
        self,
        *,
        config: AdvancedConfig,
        robot: RobotIO,
        pose: RobotPose,
        logger: RunLogger,
        detection_stabilizer: DetectionStabilizer,
        read_frame: Callable,
        debug_frame: Callable,
        set_last_command: Callable[[str], None],
        run_timestamp: str,
    ) -> None:
        self.config = config
        self.robot = robot
        self.pose = pose
        self.logger = logger
        self.detection_stabilizer = detection_stabilizer
        self.read_frame = read_frame
        self.debug_frame = debug_frame
        self.set_last_command = set_last_command
        self.run_timestamp = run_timestamp
        self.saved_detection_decisions: set[tuple[int, str]] = set()
        self.detection_debug_sequence = 0

    def scan(self) -> PinMap:
        pin_map = PinMap()
        self.detection_stabilizer.reset()

        if self.config.dry_run and self.config.video_source:
            for _ in range(max(1, self.config.dry_run_scan_frames)):
                ok, frame = self.read_frame()
                if not ok:
                    break
                self._add_frame_observations(pin_map, frame, self.pose.heading_deg)
            self.robot.stop()
            self._cleanup_map(pin_map)
            return pin_map

        half_angle = self.config.max_scan_angle_deg / 2.0
        left_duration = half_angle / self.config.scan_left_angular_speed_deg_per_sec
        self.robot.timed_command("left", left_duration, self.config.scan_left_pwm)
        self.pose.heading_deg -= half_angle

        sweep_duration = self.config.max_scan_angle_deg / self.config.scan_right_angular_speed_deg_per_sec
        sweep_start = time.monotonic()
        last_command_at = 0.0
        self.robot.set_speed(self.config.scan_right_pwm)
        while time.monotonic() - sweep_start < sweep_duration:
            now = time.monotonic()
            if now - last_command_at >= self.config.command_interval_sec:
                self.robot.move_once("right")
                self.set_last_command("right")
                last_command_at = now
            ok, frame = self.read_frame()
            if not ok:
                continue
            elapsed = now - sweep_start
            scan_heading = -half_angle + self.config.scan_right_angular_speed_deg_per_sec * elapsed
            detections = self._detect_stable_kegs(frame)
            self._add_detections_to_map(pin_map, detections, scan_heading, frame=frame)
            self.debug_frame(frame, detections, pin_map)

        self.robot.stop()
        self.pose.heading_deg = half_angle
        self._cleanup_map(pin_map)
        return pin_map

    def _detect_stable_kegs(self, frame):
        detections = detect_kegs(frame, min_area=self.config.min_contour_area)
        return self.detection_stabilizer.filter(detections)

    def _add_frame_observations(self, pin_map: PinMap, frame, scan_heading: float) -> None:
        detections = self._detect_stable_kegs(frame)
        self._add_detections_to_map(pin_map, detections, scan_heading, frame=frame)

    def _add_detections_to_map(self, pin_map: PinMap, detections, scan_heading: float, *, frame=None) -> None:
        for detection in detections:
            observation = detection_to_map_observation(
                detection,
                robot_pose=self.pose,
                scan_heading_deg=scan_heading,
                frame_width=self.config.frame_width,
                camera_horizontal_fov_deg=self.config.camera_horizontal_fov_deg,
                k_distance=self.config.k_distance,
            )
            pin = pin_map.add_observation(
                **observation,
                target_color=self.config.target_color,
                merge_radius_cm=self.config.merge_radius_cm,
                cross_color_merge_radius_cm=self.config.cross_color_merge_radius_cm,
                target_min_votes=self.config.map_target_min_votes,
            )
            self._save_detection_decision_debug(frame, detection, pin, observation)
            self.logger.log("map_observation", pin_id=pin.id, detection=detection, observation=observation)

    def _save_detection_decision_debug(self, frame, detection, pin: Pin, observation: dict) -> None:
        if frame is None or pin.observations < self.config.map_min_observations:
            return
        key = (pin.id, pin.color)
        if key in self.saved_detection_decisions:
            return
        self.saved_detection_decisions.add(key)
        self.detection_debug_sequence += 1
        path = save_detection_decision_image(
            frame,
            detection=detection,
            pin=pin,
            observation=observation,
            config=self.config,
            run_timestamp=self.run_timestamp,
            sequence=self.detection_debug_sequence,
        )
        if path is not None:
            self.logger.log("detection_decision_saved", pin_id=pin.id, color=pin.color, path=str(path))

    def _cleanup_map(self, pin_map: PinMap) -> None:
        before = len(pin_map.alive_pins())
        pin_map.cleanup(
            target_color=self.config.target_color,
            same_color_radius_cm=self.config.merge_radius_cm,
            cross_color_radius_cm=self.config.cross_color_merge_radius_cm,
            cleanup_merge_radius_cm=self.config.map_cleanup_merge_radius_cm,
            min_observations=self.config.map_min_observations,
            target_min_votes=self.config.map_target_min_votes,
        )
        after = len(pin_map.alive_pins())
        if before != after:
            self.logger.log("map_cleanup", before=before, after=after)
