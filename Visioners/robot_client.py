"""WebSocket client for KPI Robot Vision Car movement commands."""

from __future__ import annotations

import threading
import time

from websocket import create_connection

from config import (
    COMMAND_PATH,
    COMMAND_PORT,
    DEFAULT_SPEED,
    MAX_SPEED,
    MIN_SPEED,
    MOVE_REPEAT_SECONDS,
)


MOTION_COMMANDS = {"forward", "backward", "left", "right", "stop"}


class RobotClient:
    """Small safe wrapper around the robot WebSocket API."""

    def __init__(self, ip: str, port: int = COMMAND_PORT, path: str = COMMAND_PATH) -> None:
        self.ip = ip
        self.port = port
        self.path = path if path.startswith("/") else f"/{path}"
        self._socket = None
        self._lock = threading.Lock()

    @property
    def ws_url(self) -> str:
        return f"ws://{self.ip}:{self.port}{self.path}"

    def connect(self, stop_on_connect: bool = True) -> None:
        """Open the WebSocket and optionally force motors into stopped state."""

        with self._lock:
            self._close_locked()
            self._socket = create_connection(self.ws_url, timeout=2)

        print("ping ->", self.send("ping"))
        if stop_on_connect:
            self.stop()

    def send(self, command: str) -> str:
        """Send one text command and return the robot response."""

        with self._lock:
            if self._socket is None:
                raise RuntimeError("Robot is not connected. Call connect() first.")

            self._socket.send(command)
            answer = str(self._socket.recv()).strip()

        print(f"{command} -> {answer}")
        return answer

    def set_speed(self, speed: int = DEFAULT_SPEED) -> None:
        speed = max(MIN_SPEED, min(MAX_SPEED, int(speed)))
        self.send(f"speed:{speed}")

    def stop(self) -> None:
        self.send("stop")

    def move_once(self, command: str) -> None:
        if command not in MOTION_COMMANDS or command == "stop":
            raise ValueError(f"Unsupported movement command: {command!r}")
        self.send(command)

    def move_for(self, command: str, seconds: float) -> None:
        """Move for a fixed time by refreshing the command until stop."""

        deadline = time.monotonic() + seconds
        try:
            while time.monotonic() < deadline:
                self.move_once(command)
                time.sleep(MOVE_REPEAT_SECONDS)
        finally:
            self.stop()

    def close(self) -> None:
        """Stop the robot and close the WebSocket."""

        try:
            if self._socket is not None:
                self.stop()
        finally:
            with self._lock:
                self._close_locked()

    def _close_locked(self) -> None:
        if self._socket is None:
            return
        try:
            self._socket.close()
        finally:
            self._socket = None

    def __enter__(self) -> RobotClient:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
