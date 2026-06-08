"""Shared settings for the robot and camera examples."""

ROBOT_IP = "10.85.194.75"

COMMAND_PORT = 80
COMMAND_PATH = "/ws"

STREAM_PORT = 81
STREAM_PATH = "/stream"

DEFAULT_SPEED = 170
MIN_SPEED = 85
MAX_SPEED = 255
MANUAL_SPEED = 120

# The ESP32 firmware stops the motors if movement commands are not refreshed.
MOVE_REPEAT_SECONDS = 0.2

# Used by the OpenCV keyboard demo. Holding a key usually repeats key events,
# and this timeout stops the robot when no fresh key event arrives.
KEY_COMMAND_HOLD_SECONDS = 0.6

# Autonomous targeting debug settings.
TARGET_COLOR = "blue"
ALIGN_TOLERANCE_PIXELS = 25
ALIGN_STABLE_FRAMES = 3
ALIGN_SPEED = 140
AUTO_ALIGN_ERROR_WORSE_MARGIN = 6
ATTACK_SPEED = 255
AUTO_ATTACK_MAX_SECONDS = 2.0
AUTO_ATTACK_LOST_TARGET_SECONDS = 0.45
AUTO_DISABLE_STEERING_AREA = 8000
AUTO_FINAL_FORWARD_AREA = 12000
AUTO_FINAL_FORWARD_SECONDS = 2
