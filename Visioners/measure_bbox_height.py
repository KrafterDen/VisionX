"""Tiny helper for measuring detected keg bbox height.

Use this for quick distance calibration:
    k_distance = known_distance_cm * bbox_height_px
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from camera_stream import CameraStream
from config import ROBOT_IP
from vision import BOX_COLORS, COLOR_RANGES, detect_kegs, draw_detections


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "temporary" / "bbox_height_debug.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure detected keg bbox heights.")
    parser.add_argument("--ip", default=ROBOT_IP, help="Robot camera IP.")
    parser.add_argument("--image", help="Read one saved image instead of the robot camera.")
    parser.add_argument(
        "--target-color",
        choices=sorted(COLOR_RANGES),
        help="Measure only one color. By default all detected colors are printed.",
    )
    parser.add_argument("--known-distance-cm", type=float, help="Real distance to the keg.")
    parser.add_argument("--min-area", type=int, default=750)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Where to save debug image.")
    parser.add_argument("--once", action="store_true", help="Read one frame, print values, and exit.")
    parser.add_argument("--show", action="store_true", help="Show preview window in one-frame mode.")
    return parser.parse_args()


def read_frame(args: argparse.Namespace):
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise RuntimeError(f"Cannot read image: {args.image}")
        return frame

    with CameraStream(args.ip) as camera:
        frame = None
        for _ in range(max(1, args.warmup_frames)):
            ok, current_frame = camera.read()
            if ok:
                frame = current_frame
        if frame is None:
            raise RuntimeError(f"Cannot read camera frame from {camera.url}")
        return frame


def print_measurements(detections, known_distance_cm: float | None) -> None:
    if not detections:
        print("No kegs detected.")
        return

    print("idx color   x   y   w   h_px   area    confidence   k_distance")
    print("--- ----- --- --- --- ------ ------- ---------- ------------")
    for index, detection in enumerate(detections, start=1):
        x, y, width, height = detection["bbox"]
        k_distance = ""
        if known_distance_cm is not None:
            k_distance = f"{known_distance_cm * height:.1f}"
        print(
            f"{index:>3} "
            f"{detection['color']:<6} "
            f"{x:>3} {y:>3} {width:>3} {height:>6} "
            f"{detection['area']:>7.1f} "
            f"{detection['confidence']:>10.2f} "
            f"{k_distance:>12}"
        )

    best = max(detections, key=lambda detection: detection["bbox"][3])
    best_height = best["bbox"][3]
    print()
    print(f"Best bbox height: {best_height}px ({best['color']})")
    if known_distance_cm is not None:
        print(f"Suggested k_distance: {known_distance_cm * best_height:.1f}")


def draw_height_overlay(frame, detections, known_distance_cm: float | None = None):
    output = draw_detections(frame, detections)
    best = None

    for detection in detections:
        x, y, width, height = detection["bbox"]
        color = BOX_COLORS.get(detection["color"], (255, 255, 255))
        label = f"h={height}px"
        if known_distance_cm is not None:
            label += f" K={known_distance_cm * height:.0f}"
        cv2.putText(
            output,
            label,
            (x, min(output.shape[0] - 8, y + height + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
        if best is None or height > best["bbox"][3]:
            best = detection

    if best is None:
        bottom_text = "NO KEG DETECTED"
    else:
        best_height = best["bbox"][3]
        bottom_text = f"BEST: {best['color']} h={best_height}px"
        if known_distance_cm is not None:
            bottom_text += f"  suggested K={known_distance_cm * best_height:.0f}"

    cv2.putText(
        output,
        bottom_text,
        (12, output.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def save_debug_image(path: str, output) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output)
    print(f"Debug image saved: {output_path}")


def run_once(args: argparse.Namespace) -> int:
    frame = read_frame(args)
    detections = detect_kegs(
        frame,
        target_color=args.target_color,
        min_area=args.min_area,
    )
    detections.sort(key=lambda detection: detection["bbox"][3], reverse=True)
    print_measurements(detections, args.known_distance_cm)
    output = draw_height_overlay(frame, detections, args.known_distance_cm)
    save_debug_image(args.output, output)
    if args.show:
        cv2.imshow("bbox height measurement", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def run_live(args: argparse.Namespace) -> int:
    print("Opening camera. Press q, x, or Esc to quit.")
    with CameraStream(args.ip) as camera:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera frame missing.")
                continue

            detections = detect_kegs(
                frame,
                target_color=args.target_color,
                min_area=args.min_area,
            )
            detections.sort(key=lambda detection: detection["bbox"][3], reverse=True)
            output = draw_height_overlay(frame, detections, args.known_distance_cm)
            cv2.imshow("bbox height measurement", output)

            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), ord("Q"), ord("x"), ord("X"), 27}:
                save_debug_image(args.output, output)
                break

    cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = parse_args()
    if args.image or args.once:
        return run_once(args)
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
