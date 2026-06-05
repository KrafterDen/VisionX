# Advanced Kegelring Controller

Advanced MVP for the colored-pin hackathon task. It reuses the existing
`Visioners` camera, robot WebSocket, and HSV detection modules.

## Run

Mapping-only scan:

```bash
cd /home/krafterden9/VisionX_repo
python Visioners/advanced/autonomous_kegelring.py \
  --ip 10.85.194.75 \
  --target-color blue \
  --scan-only \
  --scan-angle 180 \
  --scan-left-pwm 165 \
  --scan-right-pwm 135 \
  --debug \
  --save-log
```

Full autonomous run:

```bash
cd /home/krafterden9/VisionX_repo
python Visioners/advanced/autonomous_kegelring.py \
  --ip 10.85.194.75 \
  --target-color blue \
  --target-count 2 \
  --debug \
  --save-log
```

Dry-run with a saved frame:

```bash
python Visioners/advanced/autonomous_kegelring.py \
  --target-color blue \
  --dry-run \
  --video-source Visioners/temporary/all_keg_colors_frame.jpg \
  --save-log \
  --map-output-dir Visioners/advanced/maps
```

Map JSON files are saved automatically to `Visioners/advanced/maps/`.
Open `latest_map.json` to inspect the newest map after a run.
Use `--no-save-map` to disable this.

Accepted detection debug images are saved automatically to
`Visioners/advanced/detection_debug/<run_timestamp>/`. These show the exact
camera frame where a pin first reached the "real enough" observation threshold.
Use `--no-save-detection-debug` to disable this.

## Tune First

- `k_distance`: distance estimate. Example: pin at 60 cm with 80 px height means `4800`.
- `scan-left-pwm`, `scan-right-pwm`, `scan-left-angular-speed`, and `scan-right-angular-speed`: initial scan timing.
- `scan-angle`: total initial sweep. `180` means the robot scans from left 90 degrees to right 90 degrees.
- `approach_pwm` and `approximate_forward_speed_cm_per_sec_at_approach_pwm`: waypoint/approach pose estimate.
- `forbidden_radius_cm` comes from robot width, pin radius, and safety margin.
- `max_rescan_attempts` and `max_waypoint_failures`: stop limits for repeated unsafe/blocked situations.

The map is approximate. Final attack always requires live camera confirmation.
