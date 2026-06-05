"""Run camera preview and keyboard control together.

Keys:
    W - forward
    S - backward
    A - left
    D - right
    T - toggle automatic target knockdown
    Space or X - stop
    Q or Esc - quit
"""

from __future__ import annotations

import time

import cv2

from camera_stream import CameraStream
from config import (
    ALIGN_STABLE_FRAMES,
    ALIGN_SPEED,
    ATTACK_SPEED,
    AUTO_ATTACK_LOST_TARGET_SECONDS,
    AUTO_ATTACK_MAX_SECONDS,
    AUTO_ALIGN_ERROR_WORSE_MARGIN,
    AUTO_DISABLE_STEERING_AREA,
    AUTO_FINAL_FORWARD_AREA,
    AUTO_FINAL_FORWARD_SECONDS,
    KEY_COMMAND_HOLD_SECONDS,
    MANUAL_SPEED,
    MOVE_REPEAT_SECONDS,
    ROBOT_IP,
)
from robot_client import RobotClient
from vision import choose_target, detect_kegs, get_alignment, process_frame


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

AUTO_MODE_ALIGN = "align"
AUTO_MODE_ATTACK = "attack"
AUTO_MODE_FINAL_FORWARD = "final_forward"


def _reverse_turn(command: str) -> str:
    return "right" if command == "left" else "left"


def run() -> None:
    robot = RobotClient(ROBOT_IP)
    camera = CameraStream(ROBOT_IP)

    active_command = None
    active_until = 0.0
    last_motion_send = 0.0
    last_auto_command_sent = None
    auto_enabled = False
    auto_mode = AUTO_MODE_ALIGN
    auto_turn_command = None
    auto_last_error_abs = None
    auto_aligned_frames = 0
    auto_attack_started_at = 0.0
    auto_final_forward_started_at = 0.0
    auto_steering_disabled = False
    auto_target_lost_since = None
    last_frame_time = time.monotonic()
    fps = 0.0

    try:
        robot.connect()
        robot.set_speed(MANUAL_SPEED)
        camera.open()

        print("Controls: W/A/S/D move, T auto-knock, Space or X stop, Q or Esc quit")

        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera frame was not received")
                break

            now = time.monotonic()
            frame_delta = now - last_frame_time
            last_frame_time = now
            if frame_delta > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / frame_delta)

            detections = detect_kegs(frame)
            target = choose_target(detections)
            alignment = get_alignment(frame, target)
            display_command = active_command
            if auto_enabled:
                display_command = f"auto:{auto_mode}:{auto_turn_command or alignment['command_hint']}"

            cv2.imshow("Robot camera", process_frame(frame, fps, display_command))

            key = cv2.waitKey(1) & 0xFF
            if key in KEY_TO_COMMAND:
                if auto_enabled:
                    print("Auto-knock disabled by manual movement")
                    auto_enabled = False
                    auto_mode = AUTO_MODE_ALIGN
                    auto_turn_command = None
                    auto_last_error_abs = None
                    auto_aligned_frames = 0
                    auto_final_forward_started_at = 0.0
                    auto_steering_disabled = False
                    auto_target_lost_since = None
                    last_auto_command_sent = None
                    robot.set_speed(MANUAL_SPEED)
                active_command = KEY_TO_COMMAND[key]
                active_until = now + KEY_COMMAND_HOLD_SECONDS
            elif key in AUTO_TOGGLE_KEYS:
                auto_enabled = not auto_enabled
                active_command = None
                auto_mode = AUTO_MODE_ALIGN
                auto_turn_command = None
                auto_last_error_abs = None
                auto_aligned_frames = 0
                auto_attack_started_at = 0.0
                auto_final_forward_started_at = 0.0
                auto_steering_disabled = False
                auto_target_lost_since = None
                last_auto_command_sent = None
                robot.stop()
                robot.set_speed(ALIGN_SPEED if auto_enabled else MANUAL_SPEED)
                print("Auto-knock enabled" if auto_enabled else "Auto-knock disabled")
            elif key in STOP_KEYS:
                auto_enabled = False
                auto_mode = AUTO_MODE_ALIGN
                auto_turn_command = None
                auto_last_error_abs = None
                auto_aligned_frames = 0
                auto_final_forward_started_at = 0.0
                auto_steering_disabled = False
                auto_target_lost_since = None
                last_auto_command_sent = None
                active_command = None
                robot.stop()
                robot.set_speed(MANUAL_SPEED)
            elif key in QUIT_KEYS:
                break

            if not auto_enabled and active_command is not None and now > active_until:
                active_command = None
                robot.stop()

            if auto_enabled:
                command_hint = alignment["command_hint"]
                error_x = alignment["error_x"]

                if auto_mode == AUTO_MODE_FINAL_FORWARD:
                    if now - auto_final_forward_started_at >= AUTO_FINAL_FORWARD_SECONDS:
                        robot.stop()
                        robot.set_speed(MANUAL_SPEED)
                        auto_enabled = False
                        auto_mode = AUTO_MODE_ALIGN
                        auto_turn_command = None
                        auto_last_error_abs = None
                        auto_aligned_frames = 0
                        auto_final_forward_started_at = 0.0
                        auto_steering_disabled = False
                        auto_target_lost_since = None
                        last_auto_command_sent = "stop"
                        print("Auto-knock final forward finished")
                        continue

                    if now - last_motion_send >= MOVE_REPEAT_SECONDS:
                        robot.move_once("forward")
                        last_motion_send = now
                        last_auto_command_sent = "forward"
                    continue

                if target is not None and target["area"] >= AUTO_FINAL_FORWARD_AREA:
                    robot.set_speed(ATTACK_SPEED)
                    auto_mode = AUTO_MODE_FINAL_FORWARD
                    auto_final_forward_started_at = now
                    auto_turn_command = None
                    auto_last_error_abs = None
                    auto_aligned_frames = 0
                    auto_target_lost_since = None
                    auto_steering_disabled = False
                    print(f"Auto-knock final forward started: area={int(target['area'])}")
                    robot.move_once("forward")
                    last_motion_send = now
                    last_auto_command_sent = "forward"
                    continue

                if error_x is None:
                    auto_turn_command = None
                    auto_last_error_abs = None
                    auto_aligned_frames = 0
                    if auto_mode == AUTO_MODE_ATTACK:
                        if auto_steering_disabled:
                            robot.set_speed(ATTACK_SPEED)
                            auto_mode = AUTO_MODE_FINAL_FORWARD
                            auto_final_forward_started_at = now
                            auto_turn_command = None
                            auto_last_error_abs = None
                            auto_target_lost_since = None
                            print("Auto-knock final forward started: target lost after steering disabled")
                            robot.move_once("forward")
                            last_motion_send = now
                            last_auto_command_sent = "forward"
                            continue

                        if auto_target_lost_since is None:
                            auto_target_lost_since = now
                        if now - auto_target_lost_since < AUTO_ATTACK_LOST_TARGET_SECONDS:
                            if now - last_motion_send >= MOVE_REPEAT_SECONDS:
                                robot.move_once("forward")
                                last_motion_send = now
                                last_auto_command_sent = "forward"
                            continue
                        robot.stop()
                        robot.set_speed(MANUAL_SPEED)
                        auto_enabled = False
                        auto_mode = AUTO_MODE_ALIGN
                        auto_turn_command = None
                        auto_last_error_abs = None
                        auto_aligned_frames = 0
                        auto_final_forward_started_at = 0.0
                        auto_steering_disabled = False
                        last_auto_command_sent = "stop"
                        print("Auto-knock finished: target lost")
                        continue

                    if last_auto_command_sent != "stop":
                        robot.stop()
                        last_auto_command_sent = "stop"
                    continue

                auto_target_lost_since = None

                if (
                    auto_mode == AUTO_MODE_ATTACK
                    and target is not None
                    and target["area"] >= AUTO_DISABLE_STEERING_AREA
                ):
                    if not auto_steering_disabled:
                        print(f"Auto-knock steering disabled: area={int(target['area'])}")
                    auto_steering_disabled = True
                    auto_turn_command = None
                    auto_last_error_abs = None

                error_abs = abs(error_x)
                if auto_steering_disabled:
                    auto_aligned_frames = 0
                elif command_hint == "stop":
                    auto_turn_command = None
                    auto_last_error_abs = None
                    auto_aligned_frames += 1
                else:
                    auto_aligned_frames = 0
                    if auto_turn_command is None:
                        auto_turn_command = command_hint
                    elif (
                        auto_last_error_abs is not None
                        and error_abs > auto_last_error_abs + AUTO_ALIGN_ERROR_WORSE_MARGIN
                    ):
                        auto_turn_command = _reverse_turn(auto_turn_command)
                        auto_last_error_abs = None
                        print(f"Auto-knock reversed to {auto_turn_command}: dx={error_x}")
                    auto_last_error_abs = error_abs

                if auto_mode == AUTO_MODE_ALIGN:
                    if auto_aligned_frames >= ALIGN_STABLE_FRAMES:
                        robot.stop()
                        robot.set_speed(ATTACK_SPEED)
                        auto_mode = AUTO_MODE_ATTACK
                        auto_attack_started_at = now
                        auto_turn_command = None
                        auto_last_error_abs = None
                        auto_steering_disabled = False
                        last_auto_command_sent = "stop"
                        print("Auto-knock attack started")
                        continue

                    if command_hint == "stop":
                        if last_auto_command_sent != "stop":
                            robot.stop()
                            last_auto_command_sent = "stop"
                    elif now - last_motion_send >= MOVE_REPEAT_SECONDS:
                        robot.move_once(auto_turn_command)
                        last_motion_send = now
                        last_auto_command_sent = auto_turn_command

                elif auto_mode == AUTO_MODE_ATTACK:
                    if now - auto_attack_started_at >= AUTO_ATTACK_MAX_SECONDS:
                        robot.stop()
                        robot.set_speed(MANUAL_SPEED)
                        auto_enabled = False
                        auto_mode = AUTO_MODE_ALIGN
                        auto_turn_command = None
                        auto_last_error_abs = None
                        auto_aligned_frames = 0
                        auto_final_forward_started_at = 0.0
                        auto_steering_disabled = False
                        last_auto_command_sent = "stop"
                        print("Auto-knock finished by timeout")
                        continue

                    if auto_steering_disabled or command_hint == "stop":
                        attack_command = "forward"
                    else:
                        attack_command = auto_turn_command

                    if now - last_motion_send >= MOVE_REPEAT_SECONDS:
                        robot.move_once(attack_command)
                        last_motion_send = now
                        last_auto_command_sent = attack_command

            elif active_command is not None and now - last_motion_send >= MOVE_REPEAT_SECONDS:
                robot.move_once(active_command)
                last_motion_send = now

    finally:
        robot.close()
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
