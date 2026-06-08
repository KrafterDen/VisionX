"""Tunable settings for the advanced kegelring autonomous controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .vision import COLOR_RANGES


ROBOT_IP = "10.85.194.75"
COMMAND_PORT = 80
COMMAND_PATH = "/ws"
STREAM_PORT = 81
STREAM_PATH = "/stream"
DEFAULT_SPEED = 170
MIN_SPEED = 85
MAX_SPEED = 255
MOVE_REPEAT_SECONDS = 0.2


VALID_TARGET_COLORS = ("red", "pink", "purple", "blue", "green", "yellow")
DEFAULT_MAP_OUTPUT_DIR = str(Path(__file__).resolve().parent / "maps")
DEFAULT_DETECTION_DEBUG_DIR = str(Path(__file__).resolve().parent / "detection_debug")


@dataclass
class AdvancedConfig:
    """Hackathon-tunable config values for mapping, planning, and control."""

    robot_ip: str = ROBOT_IP
    command_port: int = COMMAND_PORT
    command_path: str = COMMAND_PATH
    stream_port: int = STREAM_PORT
    stream_path: str = STREAM_PATH
    target_color: str = "blue"
    target_count: int | None = None

    frame_width: int = 320
    frame_height: int = 240
    camera_horizontal_fov_deg: float = 60.0
    roi_top_fraction: float = 0.25
    min_contour_area: int = 750
    max_contour_area: int = 35_000
    min_aspect_ratio: float = 0.45
    max_aspect_ratio: float = 3.4
    morphology_kernel_size: int = 5
    hsv_ranges: dict = field(default_factory=lambda: dict(COLOR_RANGES))

    k_distance: float = 2_954.0
    pin_real_height_cm: float = 12.0
    attack_distance_cm: float = 18.0
    attack_bbox_height_threshold_px: int = 105

    robot_width_cm: float = 13.0
    robot_length_cm: float = 16.0
    robot_front_offset_cm: float = 6.0
    pin_radius_cm: float = 3.5
    safety_margin_cm: float = 8.0

    scan_pwm: int = 165
    scan_left_pwm: int = 125
    scan_right_pwm: int = 125
    search_pwm: int = 140
    approach_pwm: int = 170
    attack_pwm: int = 255
    turn_pwm: int = 180
    backup_pwm: int = 140
    angular_speed_deg_per_sec_at_scan_pwm: float = 55.0
    scan_left_angular_speed_deg_per_sec: float = 55.0
    scan_right_angular_speed_deg_per_sec: float = 45.0
    approximate_forward_speed_cm_per_sec_at_approach_pwm: float = 22.0

    merge_radius_cm: float = 18.0
    cross_color_merge_radius_cm: float = 14.0
    map_cleanup_merge_radius_cm: float = 22.0
    map_min_observations: int = 2
    map_target_min_votes: int = 2
    waypoint_extra_margin_cm: float = 10.0
    obstacle_hard_risk: float = 10_000.0
    max_scan_angle_deg: float = 180.0
    local_scan_angle_deg: float = 35.0
    rescan_angle_deg: float = 70.0

    command_interval_sec: float = 0.12
    max_no_frame_time_sec: float = 1.0
    center_tolerance: float = 0.16
    target_confirm_frames: int = 3
    detection_confirm_frames: int = 3
    detection_track_radius_px: int = 45
    detection_track_max_age_sec: float = 0.5
    max_rescan_attempts: int = 2
    max_waypoint_failures: int = 2
    dry_run_scan_frames: int = 4
    forward_pulse_sec: float = 0.18
    turn_pulse_sec: float = 0.12
    waypoint_pulse_sec: float = 0.18
    attack_duration_sec: float = 0.75
    backup_duration_sec: float = 0.45
    stop_on_exception: bool = True

    debug: bool = False
    dry_run: bool = False
    scan_only: bool = False
    save_log: str | None = None
    video_source: str | None = None
    save_map: bool = True
    map_output_dir: str = DEFAULT_MAP_OUTPUT_DIR
    map_image_size_px: int = 900
    map_padding_cm: float = 35.0
    save_detection_debug_images: bool = True
    detection_debug_dir: str = DEFAULT_DETECTION_DEBUG_DIR

    @property
    def video_stream_url(self) -> str:
        path = self.stream_path if self.stream_path.startswith("/") else f"/{self.stream_path}"
        return f"http://{self.robot_ip}:{self.stream_port}{path}"

    @property
    def websocket_url(self) -> str:
        path = self.command_path if self.command_path.startswith("/") else f"/{self.command_path}"
        return f"ws://{self.robot_ip}:{self.command_port}{path}"

    @property
    def forbidden_radius_cm(self) -> float:
        return self.robot_width_cm / 2.0 + self.pin_radius_cm + self.safety_margin_cm

    @property
    def turn_safety_radius_cm(self) -> float:
        half_l = self.robot_length_cm / 2.0
        half_w = self.robot_width_cm / 2.0
        return (half_l * half_l + half_w * half_w) ** 0.5 + self.safety_margin_cm
