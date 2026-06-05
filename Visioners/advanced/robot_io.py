"""Dry-run aware robot command wrapper for the advanced controller."""

from __future__ import annotations

import time

from .config import AdvancedConfig


class RobotIO:
    def __init__(self, config: AdvancedConfig) -> None:
        self.config = config
        self.dry_run = config.dry_run
        self._robot = None if self.dry_run else self._create_robot_client()
        self._speed: int | None = None
        self.last_command = "stop"

    def _create_robot_client(self):
        try:
            from robot_client import RobotClient
        except ImportError:
            from ..robot_client import RobotClient

        return RobotClient(
            self.config.robot_ip,
            port=self.config.command_port,
            path=self.config.command_path,
        )

    def connect(self) -> None:
        if self.dry_run:
            print("[dry-run] robot connect skipped")
            return
        self._robot.connect()

    def set_speed(self, pwm: int) -> None:
        pwm = int(pwm)
        if self._speed == pwm:
            return
        self._speed = pwm
        if self.dry_run:
            print(f"[dry-run] speed:{pwm}")
            return
        self._robot.set_speed(pwm)

    def send_command(self, command: str) -> None:
        self.last_command = command
        if self.dry_run:
            print(f"[dry-run] {command}")
            return
        if command == "stop":
            self._robot.stop()
        else:
            self._robot.move_once(command)

    def move_once(self, command: str) -> None:
        self.send_command(command)

    def stop(self) -> None:
        self.send_command("stop")

    def timed_command(self, command: str, duration_sec: float, pwm: int | None = None) -> None:
        if pwm is not None:
            self.set_speed(pwm)
        deadline = time.monotonic() + max(0.0, duration_sec)
        try:
            while time.monotonic() < deadline:
                self.send_command(command)
                time.sleep(self.config.command_interval_sec)
        finally:
            self.stop()

    def safe_forward_pulse(self, duration_sec: float | None = None) -> None:
        self.timed_command(
            "forward",
            self.config.forward_pulse_sec if duration_sec is None else duration_sec,
            self.config.approach_pwm,
        )

    def close(self) -> None:
        try:
            self.stop()
        finally:
            if not self.dry_run and self._robot is not None:
                self._robot.close()
