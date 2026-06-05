"""Simple autonomous colored-kegel knockdown MVP.

This script reuses the existing CameraStream, RobotClient, and vision helpers.
It intentionally stays simple: one target color, one camera stream, one robot,
and a small state machine.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from enum import Enum

import cv2

try:
    from camera_stream import CameraStream
    from config import (
        ATTACK_SPEED,
        KEY_COMMAND_HOLD_SECONDS,
        MANUAL_SPEED,
        ROBOT_IP,
        TARGET_COLOR,
    )
    from vision import BOX_COLORS, detect_kegs, draw_detections
except ImportError:
    from .camera_stream import CameraStream
    from .config import (
        ATTACK_SPEED,
        KEY_COMMAND_HOLD_SECONDS,
        MANUAL_SPEED,
        ROBOT_IP,
        TARGET_COLOR,
    )
    from .vision import BOX_COLORS, detect_kegs, draw_detections


VALID_TARGET_COLORS = ("red", "pink", "purple", "blue", "green", "yellow")
KEY_TO_COMMAND = {
    ord("w"): "forward",
    ord("W"): "forward",
    ord("s"): "backward",
    ord("S"): "backward",
    ord("a"): "left",
    ord("A"): "left",
    ord("d"): "right",
    ord("D"): "right",
}
AUTO_TOGGLE_KEYS = {ord("t"), ord("T")}
STOP_KEYS = {ord(" "), ord("x"), ord("X")}
QUIT_KEYS = {ord("q"), ord("Q"), 27}
DEBUG_QUIT_KEYS = QUIT_KEYS | {ord("x"), ord("X")}


class State(str, Enum):
    INIT = "INIT"
    MANUAL_CONTROL = "MANUAL_CONTROL"
    SEARCH = "SEARCH"
    LOCK_TARGET = "LOCK_TARGET"
    ALIGN = "ALIGN"
    APPROACH = "APPROACH"
    ATTACK = "ATTACK"
    BACKUP = "BACKUP"
    DONE = "DONE"
    EMERGENCY_STOP = "EMERGENCY_STOP"


STATE_DESCRIPTIONS = {
    State.INIT: "connect robot, open camera, set first speed",
    State.MANUAL_CONTROL: "keyboard control is active",
    State.SEARCH: "rotate slowly until the target color appears",
    State.LOCK_TARGET: "wait for the same target to be visible for several frames",
    State.ALIGN: "turn left/right until the target is close to frame center",
    State.APPROACH: "drive or correct heading while safety corridor is clear",
    State.ATTACK: "verify safety, then drive forward fast to knock the target",
    State.BACKUP: "reverse briefly after the hit",
    State.DONE: "stop after the requested target count",
    State.EMERGENCY_STOP: "stop immediately after error, video loss, or user abort",
}


@dataclass(frozen=True)
class AutonomousSettings:
    target_color: str
    target_count: int
    debug: bool
    dry_run: bool
    keyboard: bool = False
    manual_start: bool = False
    search_speed: int = 170
    align_speed: int = 170
    approach_speed: int = 190
    attack_speed: int = ATTACK_SPEED
    backup_speed: int = MANUAL_SPEED
    manual_speed: int = 200
    manual_turn_speed: int = 170
    command_interval_sec: float = 0.12
    center_tolerance: float = 0.16
    target_confirm_frames: int = 4
    attack_duration_sec: float = 0.7
    backup_duration_sec: float = 0.45
    attack_bbox_height_threshold: int = 105
    attack_area_threshold: int = 12_000
    attack_bottom_ratio: float = 0.85
    corridor_left_ratio: float = 0.35
    corridor_right_ratio: float = 0.65
    safety_bad_area_threshold: int = 2_500
    safety_bad_bottom_ratio: float = 0.68
    fallen_min_area: int = 2_000
    fallen_max_height_width_ratio: float = 0.85
    fallen_bottom_ratio: float = 0.58


class RobotIO:
    """Dry-run aware wrapper around the existing RobotClient."""

    def __init__(self, ip: str, dry_run: bool) -> None:
        self._dry_run = dry_run
        self._robot = None if dry_run else self._create_robot_client(ip)
        self._speed = None

    @staticmethod
    def _create_robot_client(ip: str):
        try:
            from robot_client import RobotClient
        except ImportError:
            from .robot_client import RobotClient

        return RobotClient(ip)

    def connect(self) -> None:
        if self._dry_run:
            print("[dry-run] robot connect skipped")
            return
        self._robot.connect()

    def set_speed(self, speed: int) -> None:
        if self._speed == speed:
            return
        self._speed = speed
        if self._dry_run:
            print(f"[dry-run] speed:{speed}")
            return
        self._robot.set_speed(speed)

    def stop(self) -> None:
        if self._dry_run:
            print("[dry-run] stop")
            return
        self._robot.stop()

    def move_once(self, command: str) -> None:
        if self._dry_run:
            print(f"[dry-run] {command}")
            return
        self._robot.move_once(command)

    def close(self) -> None:
        if self._dry_run:
            print("[dry-run] robot close skipped")
            return
        if self._robot is not None:
            self._robot.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple autonomous colored-kegel knockdown MVP."
    )
    parser.add_argument("--ip", default=ROBOT_IP, help="ESP32-CAM robot IP address.")
    parser.add_argument(
        "--target-color",
        default=TARGET_COLOR,
        choices=VALID_TARGET_COLORS,
        help="Kegel color to knock down.",
    )
    parser.add_argument(
        "--target-count",
        default=2,
        type=int,
        help="How many target-colored kegel detections to knock down before stopping.",
    )
    parser.add_argument("--debug", action="store_true", help="Show debug video overlay.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read camera and run decisions, but do not connect to or command robot.",
    )
    parser.add_argument(
        "--keyboard",
        action="store_true",
        help="Enable W/A/S/D keyboard override in the video window.",
    )
    parser.add_argument(
        "--manual-start",
        action="store_true",
        help="Start in keyboard control mode. Press T to start autonomous mode.",
    )
    return parser.parse_args()


def bbox_bottom(detection: dict) -> int:
    x, y, width, height = detection["bbox"]
    return y + height


def touches_frame_border(detection: dict, frame_shape, margin: int = 4) -> bool:
    frame_height, frame_width = frame_shape[:2]
    x, y, width, height = detection["bbox"]
    return (
        x <= margin
        or y <= margin
        or x + width >= frame_width - margin
        or y + height >= frame_height - margin
    )


def looks_like_fallen_keg(
    detection: dict,
    frame_shape,
    settings: AutonomousSettings,
) -> bool:
    frame_height = frame_shape[0]
    x, y, width, height = detection["bbox"]
    height_width_ratio = height / max(width, 1)
    bottom = y + height

    return (
        detection["area"] >= settings.fallen_min_area
        and height_width_ratio <= settings.fallen_max_height_width_ratio
        and bottom >= frame_height * settings.fallen_bottom_ratio
    )


def choose_best_target(
    detections: list[dict],
    target_color: str,
    frame_shape,
    settings: AutonomousSettings,
    allow_fallen: bool = False,
) -> dict | None:
    target_detections = [
        detection
        for detection in detections
        if detection["color"] == target_color
        and (allow_fallen or not looks_like_fallen_keg(detection, frame_shape, settings))
    ]
    if not target_detections:
        return None

    return max(
        target_detections,
        key=lambda detection: (
            not touches_frame_border(detection, frame_shape),
            detection["bbox"][3],
            detection["area"],
        ),
    )


def normalized_error_x(frame, target: dict) -> float:
    _, frame_width = frame.shape[:2]
    target_cx = target["center"][0]
    frame_center_x = frame_width / 2.0
    return (target_cx - frame_center_x) / frame_center_x


def is_close_enough(frame, target: dict, settings: AutonomousSettings) -> bool:
    frame_height = frame.shape[0]
    _, _, _, bbox_height = target["bbox"]
    return (
        bbox_height >= settings.attack_bbox_height_threshold
        or bbox_bottom(target) >= frame_height * settings.attack_bottom_ratio
        or target["area"] >= settings.attack_area_threshold
    )


def find_forward_blocker(
    detections: list[dict],
    target_color: str,
    frame_shape,
    settings: AutonomousSettings,
) -> dict | None:
    frame_height, frame_width = frame_shape[:2]
    corridor_left = frame_width * settings.corridor_left_ratio
    corridor_right = frame_width * settings.corridor_right_ratio

    for detection in detections:
        if detection["color"] == target_color:
            continue

        x, y, width, height = detection["bbox"]
        intersects_corridor = x < corridor_right and x + width > corridor_left
        appears_close = (
            detection["area"] >= settings.safety_bad_area_threshold
            or y + height >= frame_height * settings.safety_bad_bottom_ratio
        )
        if intersects_corridor and appears_close:
            return detection

    return None


def draw_debug_overlay(
    frame,
    detections: list[dict],
    target: dict | None,
    blocker: dict | None,
    state: State,
    last_command: str,
    knocked_count: int,
    settings: AutonomousSettings,
):
    output = draw_detections(frame, detections)
    frame_height, frame_width = output.shape[:2]
    center_x = frame_width // 2
    corridor_left = int(frame_width * settings.corridor_left_ratio)
    corridor_right = int(frame_width * settings.corridor_right_ratio)

    cv2.line(output, (center_x, 0), (center_x, frame_height), (255, 255, 255), 1)
    cv2.line(output, (corridor_left, 0), (corridor_left, frame_height), (0, 255, 255), 1)
    cv2.line(output, (corridor_right, 0), (corridor_right, frame_height), (0, 255, 255), 1)

    if target is not None:
        x, y, width, height = target["bbox"]
        cv2.rectangle(output, (x, y), (x + width, y + height), (255, 255, 255), 3)
        cv2.circle(output, target["center"], 7, (255, 255, 255), 2)

    if blocker is not None:
        x, y, width, height = blocker["bbox"]
        cv2.rectangle(output, (x, y), (x + width, y + height), (0, 0, 255), 3)
        cv2.putText(
            output,
            "BLOCK",
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    for detection in detections:
        if detection["color"] != settings.target_color:
            continue
        if not looks_like_fallen_keg(detection, output.shape, settings):
            continue
        x, y, width, height = detection["bbox"]
        cv2.putText(
            output,
            "FALLEN",
            (x, min(frame_height - 8, y + height + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    lines = [
        f"STATE: {state.value}",
        f"CMD: {last_command}",
        f"COUNT: {knocked_count}/{settings.target_count}",
        f"TARGET: {settings.target_color}",
    ]
    if settings.keyboard:
        lines.append("KEYS: W/A/S/D move | T auto/manual | Space stop | Q quit")
    for index, text in enumerate(lines):
        cv2.putText(
            output,
            text,
            (12, 24 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            BOX_COLORS.get(settings.target_color, (255, 255, 255)),
            2,
            cv2.LINE_AA,
        )

    return output


def send_motion(
    robot: RobotIO,
    command: str,
    now: float,
    last_command_time: float,
    settings: AutonomousSettings,
) -> float:
    if now - last_command_time < settings.command_interval_sec:
        return last_command_time
    robot.move_once(command)
    return now


def speed_for_manual_command(command: str, settings: AutonomousSettings) -> int:
    if command in {"left", "right"}:
        return settings.manual_turn_speed
    return settings.manual_speed


def transition(state: State, next_state: State, reason: str) -> State:
    if state != next_state:
        print(
            f"{state.value} -> {next_state.value}: "
            f"{reason} [{STATE_DESCRIPTIONS[next_state]}]"
        )
    return next_state


def run() -> None:
    args = parse_args()
    if args.target_count < 1:
        raise ValueError("--target-count must be at least 1")

    settings = AutonomousSettings(
        target_color=args.target_color,
        target_count=args.target_count,
        debug=args.debug,
        dry_run=args.dry_run,
        keyboard=args.keyboard or args.manual_start,
        manual_start=args.manual_start,
    )
    show_window = settings.debug or settings.keyboard
    robot = None
    camera = None
    state = State.INIT
    locked_frames = 0
    knocked_count = 0
    attack_started_at = None
    backup_started_at = None
    manual_command = None
    manual_until = 0.0
    last_command = "stop"
    last_command_time = 0.0

    try:
        robot = RobotIO(args.ip, settings.dry_run)
        camera = CameraStream(args.ip)
        robot.connect()
        camera.open()
        robot.set_speed(
            settings.manual_speed if settings.manual_start else settings.search_speed
        )
        print(
            "Autonomous kegelring started: "
            f"target={settings.target_color}, count={settings.target_count}, "
            f"dry_run={settings.dry_run}, keyboard={settings.keyboard}"
        )
        if settings.keyboard:
            print("Controls: W/A/S/D move, T auto/manual, Space or X stop, Q or Esc quit")
        first_state = State.MANUAL_CONTROL if settings.manual_start else State.SEARCH
        state = transition(state, first_state, "robot and camera ready")

        while True:
            ok, frame = camera.read()
            if not ok:
                state = transition(state, State.EMERGENCY_STOP, "video frame missing")
                break

            now = time.monotonic()
            detections = detect_kegs(frame)
            allow_fallen_target = state in {State.APPROACH, State.ATTACK}
            target = choose_best_target(
                detections,
                settings.target_color,
                frame.shape,
                settings,
                allow_fallen=allow_fallen_target,
            )
            blocker = find_forward_blocker(
                detections,
                settings.target_color,
                frame.shape,
                settings,
            )

            if state == State.MANUAL_CONTROL:
                locked_frames = 0
                attack_started_at = None
                backup_started_at = None

                if manual_command is not None and now > manual_until:
                    manual_command = None
                    if last_command != "stop":
                        robot.stop()
                    last_command = "stop"
                elif manual_command is not None:
                    robot.set_speed(speed_for_manual_command(manual_command, settings))
                    last_command_time = send_motion(
                        robot,
                        manual_command,
                        now,
                        last_command_time,
                        settings,
                    )
                    last_command = manual_command
                elif last_command != "stop":
                    robot.stop()
                    last_command = "stop"

            elif state == State.SEARCH:
                robot.set_speed(settings.search_speed)
                locked_frames = 0
                attack_started_at = None
                backup_started_at = None
                if target is None:
                    last_command_time = send_motion(
                        robot,
                        "left",
                        now,
                        last_command_time,
                        settings,
                    )
                    last_command = "left"
                else:
                    robot.stop()
                    last_command = "stop"
                    locked_frames = 1
                    state = transition(state, State.LOCK_TARGET, "target visible")

            elif state == State.LOCK_TARGET:
                robot.stop()
                last_command = "stop"
                if target is None:
                    locked_frames = 0
                    state = transition(state, State.SEARCH, "target lost while locking")
                else:
                    locked_frames += 1
                    if locked_frames >= settings.target_confirm_frames:
                        state = transition(state, State.ALIGN, "target confirmed")

            elif state == State.ALIGN:
                robot.set_speed(settings.align_speed)
                if target is None:
                    robot.stop()
                    last_command = "stop"
                    state = transition(state, State.SEARCH, "target lost during align")
                else:
                    error = normalized_error_x(frame, target)
                    if abs(error) <= settings.center_tolerance:
                        robot.stop()
                        last_command = "stop"
                        state = transition(state, State.APPROACH, f"aligned error={error:.2f}")
                    else:
                        command = "left" if error < 0 else "right"
                        last_command_time = send_motion(
                            robot,
                            command,
                            now,
                            last_command_time,
                            settings,
                        )
                        last_command = command

            elif state == State.APPROACH:
                if target is None:
                    robot.stop()
                    last_command = "stop"
                    state = transition(state, State.SEARCH, "target lost during approach")
                elif is_close_enough(frame, target, settings):
                    robot.stop()
                    last_command = "stop"
                    state = transition(state, State.ATTACK, "target close enough")
                    attack_started_at = None
                elif blocker is not None:
                    robot.stop()
                    last_command = "stop"
                    state = transition(
                        state,
                        State.SEARCH,
                        f"blocked by {blocker['color']} pin in corridor",
                    )
                else:
                    error = normalized_error_x(frame, target)
                    if error < -settings.center_tolerance:
                        command = "left"
                    elif error > settings.center_tolerance:
                        command = "right"
                    else:
                        command = "forward"
                    if command in {"left", "right"}:
                        robot.set_speed(settings.align_speed)
                    else:
                        robot.set_speed(settings.approach_speed)
                    last_command_time = send_motion(
                        robot,
                        command,
                        now,
                        last_command_time,
                        settings,
                    )
                    last_command = command

            elif state == State.ATTACK:
                if attack_started_at is None:
                    if target is None:
                        robot.stop()
                        last_command = "stop"
                        state = transition(state, State.SEARCH, "target lost before attack")
                        continue

                    error = normalized_error_x(frame, target)
                    if abs(error) > settings.center_tolerance:
                        robot.stop()
                        last_command = "stop"
                        state = transition(state, State.ALIGN, f"not centered error={error:.2f}")
                        continue

                    if blocker is not None:
                        robot.stop()
                        last_command = "stop"
                        state = transition(
                            state,
                            State.SEARCH,
                            f"attack blocked by {blocker['color']} pin",
                        )
                        continue

                    robot.set_speed(settings.attack_speed)
                    attack_started_at = now
                    print("ATTACK: full speed forward")

                if now - attack_started_at < settings.attack_duration_sec:
                    last_command_time = send_motion(
                        robot,
                        "forward",
                        now,
                        last_command_time,
                        settings,
                    )
                    last_command = "forward"
                else:
                    robot.stop()
                    last_command = "stop"
                    knocked_count += 1
                    state = transition(
                        state,
                        State.BACKUP,
                        f"attack complete knocked_count={knocked_count}",
                    )
                    backup_started_at = None

            elif state == State.BACKUP:
                if backup_started_at is None:
                    robot.set_speed(settings.backup_speed)
                    backup_started_at = now
                    print("BACKUP: reverse briefly")

                if now - backup_started_at < settings.backup_duration_sec:
                    last_command_time = send_motion(
                        robot,
                        "backward",
                        now,
                        last_command_time,
                        settings,
                    )
                    last_command = "backward"
                else:
                    robot.stop()
                    last_command = "stop"
                    if knocked_count >= settings.target_count:
                        state = transition(state, State.DONE, "target count reached")
                    else:
                        state = transition(state, State.SEARCH, "looking for next target")

            elif state == State.DONE:
                robot.stop()
                print(f"DONE: knocked_count={knocked_count}")
                break

            elif state == State.EMERGENCY_STOP:
                break

            if show_window:
                debug_frame = draw_debug_overlay(
                    frame,
                    detections,
                    target,
                    blocker,
                    state,
                    last_command,
                    knocked_count,
                    settings,
                )
                cv2.imshow("Simple autonomous kegelring", debug_frame)
                key = cv2.waitKey(1) & 0xFF
                if settings.keyboard and key in KEY_TO_COMMAND:
                    manual_command = KEY_TO_COMMAND[key]
                    manual_until = now + KEY_COMMAND_HOLD_SECONDS
                    locked_frames = 0
                    attack_started_at = None
                    backup_started_at = None
                    robot.set_speed(speed_for_manual_command(manual_command, settings))
                    if state != State.MANUAL_CONTROL:
                        robot.stop()
                        state = transition(
                            state,
                            State.MANUAL_CONTROL,
                            f"keyboard override {manual_command}",
                        )
                    robot.move_once(manual_command)
                    last_command_time = now
                    last_command = manual_command
                elif settings.keyboard and key in AUTO_TOGGLE_KEYS:
                    manual_command = None
                    robot.stop()
                    last_command = "stop"
                    locked_frames = 0
                    attack_started_at = None
                    backup_started_at = None
                    if state == State.MANUAL_CONTROL:
                        robot.set_speed(settings.search_speed)
                        state = transition(
                            state,
                            State.SEARCH,
                            "keyboard returned to autonomous mode",
                        )
                    else:
                        robot.set_speed(settings.manual_speed)
                        state = transition(
                            state,
                            State.MANUAL_CONTROL,
                            "keyboard paused autonomous mode",
                        )
                elif settings.keyboard and key in STOP_KEYS:
                    manual_command = None
                    robot.stop()
                    last_command = "stop"
                    locked_frames = 0
                    attack_started_at = None
                    backup_started_at = None
                    if state != State.MANUAL_CONTROL:
                        robot.set_speed(settings.manual_speed)
                        state = transition(state, State.MANUAL_CONTROL, "keyboard stop")
                elif key in (QUIT_KEYS if settings.keyboard else DEBUG_QUIT_KEYS):
                    state = transition(state, State.EMERGENCY_STOP, "debug key stop")
                    break

    except KeyboardInterrupt:
        state = transition(state, State.EMERGENCY_STOP, "keyboard interrupt")
    except Exception as exc:
        state = transition(state, State.EMERGENCY_STOP, f"exception: {exc}")
        raise
    finally:
        try:
            if robot is not None:
                robot.stop()
        finally:
            if camera is not None:
                camera.close()
            if robot is not None:
                robot.close()
            if show_window:
                cv2.destroyAllWindows()
        print(f"Stopped safely. final_state={state.value}, knocked_count={knocked_count}")


if __name__ == "__main__":
    run()
