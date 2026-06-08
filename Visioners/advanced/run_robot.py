"""One-click launcher for the advanced kegelring controller.

Edit the settings below, then run:
    python Visioners/advanced/run_robot.py
"""

from __future__ import annotations

import sys
from pathlib import Path


if __package__ in {None, ""}:
    PACKAGE_DIR = Path(__file__).resolve().parent
    if str(PACKAGE_DIR.parent) not in sys.path:
        sys.path.insert(0, str(PACKAGE_DIR.parent))
    __package__ = PACKAGE_DIR.name

from .cli import run_controller
from .config import AdvancedConfig


# ---------------------------------------------------------------------------
# Основні налаштування запуску
# ---------------------------------------------------------------------------

# IP-адреса робота/ESP32-CAM у Wi-Fi мережі.
# Міняй це, якщо робот отримав іншу адресу або ти підключився до іншого робота.
ROBOT_IP = "10.85.194.75"

# Колір кеглів, які робот має збивати.
# Доступні кольори: "red", "pink", "purple", "blue", "green", "yellow".
# Кількість таких кеглів робот визначає сам під час первинного сканування.
TARGET_COLOR = "blue"

# ---------------------------------------------------------------------------
# Режим роботи
# ---------------------------------------------------------------------------

# True показує debug-вікно OpenCV з камерою, детекціями, картою і станом робота.
# Корисно під час налаштування, але на слабкому комп'ютері може трохи гальмувати.
DEBUG = True

# True запускає алгоритм без реальних команд роботу.
# Камера/картинка читається, логіка працює, але рухи тільки друкуються в консоль.
# Для першої перевірки коду краще ставити True.
DRY_RUN = False

# True робить тільки первинне сканування поля, будує карту і зупиняється.
# Корисно, щоб перевірити детекцію кольорів і приблизну карту без атаки.
SCAN_ONLY = False

# Джерело відео замість камери робота.
# None означає брати stream з ESP32-CAM.
# Можна вказати шлях до фото/відео або індекс камери, наприклад:
# "Visioners/temporary/all_keg_colors_frame.jpg" або "0".
VIDEO_SOURCE = None


# ---------------------------------------------------------------------------
# Логи, карти і debug-файли
# ---------------------------------------------------------------------------

# True записує події запуску у JSONL-лог.
# Лог допомагає розібрати, на якому стані робот зупинився або чому зробив дію.
SAVE_LOG = True

# Шлях до JSONL-логу.
# Якщо SAVE_LOG = True, сюди записуються state changes, detections, map events.
SAVE_LOG_PATH = "advanced_kegelring_log.jsonl"

# True зберігає JSON і PNG карту після сканування, ресканів і фіналу.
# Корисно для аналізу, де робот "бачить" кеглі у своїй приблизній системі координат.
SAVE_MAP = True

# True зберігає картинки моментів, коли детекція була прийнята як реальна кегля.
# Корисно для перевірки, чи не плутає робот кольори або випадкові плями.
SAVE_DETECTION_DEBUG_IMAGES = True


# ---------------------------------------------------------------------------
# Калібрування і тюнінг руху
# None означає: використовувати значення за замовчуванням з config.py.
# ---------------------------------------------------------------------------

# Коефіцієнт оцінки дистанції до кеглі за висотою bbox:
# distance_cm = K_DISTANCE / bbox_height_px.
# Приклад: якщо кегля на 60 см має bbox height 80 px, тоді K_DISTANCE = 60 * 80 = 4800.
# Якщо дистанції на карті явно неправильні, спочатку калібруй саме це.
K_DISTANCE = 2954

# Загальний кут первинного сканування в градусах.
# 180 означає приблизно подивитися від -90 до +90 градусів перед роботом.
# Більше кут - ширший огляд, але довше сканування і більше похибки повороту.
SCAN_ANGLE_DEG = 180.0

# PWM швидкість повороту вліво під час початку сканування.
# Більше значення - швидше/сильніше крутиться, але може втратити точність.
SCAN_LEFT_PWM = None

# PWM швидкість повороту вправо під час основного sweep-сканування.
# Якщо карта розтягується або стискається по кутах, часто треба тюнити це разом зі швидкістю нижче.
SCAN_RIGHT_PWM = None

# Оцінка кутової швидкості при повороті вліво, градусів за секунду.
# Використовується не для мотора напряму, а для розрахунку, куди дивилася камера під час скану.
SCAN_LEFT_ANGULAR_SPEED = None

# Оцінка кутової швидкості при повороті вправо, градусів за секунду.
# Якщо кеглі на карті мають неправильні кути, це один з головних параметрів для тюнінгу.
SCAN_RIGHT_ANGULAR_SPEED = None

# PWM швидкість обережного під'їзду до цілі.
# Менше значення - повільніше і безпечніше, більше - швидше, але гірше контроль.
APPROACH_PWM = None

# PWM швидкість фінальної атаки.
# Зазвичай ставиться високою, бо робот має впевнено збити кеглю після наведення.
ATTACK_PWM = None


def build_config() -> AdvancedConfig:
    config = AdvancedConfig(
        target_color=TARGET_COLOR,
        debug=DEBUG,
        dry_run=DRY_RUN,
        scan_only=SCAN_ONLY,
        save_log=SAVE_LOG_PATH if SAVE_LOG else None,
        video_source=VIDEO_SOURCE,
        save_map=SAVE_MAP,
        save_detection_debug_images=SAVE_DETECTION_DEBUG_IMAGES,
    )
    config.robot_ip = ROBOT_IP

    if K_DISTANCE is not None:
        config.k_distance = K_DISTANCE
    if SCAN_ANGLE_DEG is not None:
        config.max_scan_angle_deg = SCAN_ANGLE_DEG
    if SCAN_LEFT_PWM is not None:
        config.scan_left_pwm = SCAN_LEFT_PWM
    if SCAN_RIGHT_PWM is not None:
        config.scan_right_pwm = SCAN_RIGHT_PWM
    if SCAN_LEFT_ANGULAR_SPEED is not None:
        config.scan_left_angular_speed_deg_per_sec = SCAN_LEFT_ANGULAR_SPEED
    if SCAN_RIGHT_ANGULAR_SPEED is not None:
        config.scan_right_angular_speed_deg_per_sec = SCAN_RIGHT_ANGULAR_SPEED
    if APPROACH_PWM is not None:
        config.approach_pwm = APPROACH_PWM
    if ATTACK_PWM is not None:
        config.attack_pwm = ATTACK_PWM

    return config


def main() -> int:
    return run_controller(build_config())


if __name__ == "__main__":
    raise SystemExit(main())
