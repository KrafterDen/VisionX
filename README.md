# VisionX Robot Vision

Code for the kegelring robotics hackathon task.

## Project Map

- `Visioners/advanced/` - main autonomous kegelring implementation. This folder is self-contained.
- `Visioners/advanced/cli.py` - CLI entry point for scan-only, dry-run, and full autonomous modes.
- `Visioners/advanced/controller.py` - high-level state machine.
- `Visioners/advanced/config.py` - hackathon-tunable constants.
- `Visioners/advanced/vision.py` - HSV color detection used by the advanced controller.
- `Visioners/advanced/scanner.py` - initial field scan and map-building flow.
- `Visioners/advanced/detection_stabilizer.py` - confirms detections across consecutive frames.
- `Visioners/advanced/frame_source.py` - camera/image/video frame source adapters.
- `Visioners/advanced/camera_stream.py`, `robot_client.py`, `robot_io.py` - camera and robot IO.
- `Visioners/advanced/geometry.py`, `mapping.py`, `planner.py` - pure logic for pose, map building, and path planning.
- `Visioners/advanced/debug.py` - overlays, JSONL logging, and map/debug image saving.
- `Visioners/advanced/measure_bbox_height.py` - distance calibration helper.
- `Visioners/legacy/` - older manual preview and simple autonomous scripts kept as fallback/reference.

Generated files such as `*.jsonl`, `Visioners/temporary/`, `Visioners/advanced/maps/`,
`Visioners/advanced/detection_debug/`, `Visioners/advanced/temporary/`, and `__pycache__/`
are ignored by git.

## Install

```bash
python -m pip install -r Visioners/advanced/requirements.txt
```

## Run Advanced Mode

One-click launcher with editable settings at the top of the file:

```bash
python Visioners/advanced/run_robot.py
```

Scan and save a map:

```bash
python Visioners/advanced/cli.py --ip 10.85.194.75 --target-color blue --scan-only --debug --save-log
```

Full autonomous run:

```bash
python Visioners/advanced/cli.py --ip 10.85.194.75 --target-color blue --debug --save-log
```

Dry-run from a saved image:

```bash
python Visioners/advanced/cli.py --target-color blue --dry-run --video-source Visioners/temporary/all_keg_colors_frame.jpg --save-log
```

## Run Tests

```bash
python -m unittest Visioners.advanced.tests.test_geometry_planner
```

## Cleanup Notes

For hackathon work, treat `Visioners/advanced/` as the main code path. The old
manual and simple autonomous scripts live in `Visioners/legacy/`.
