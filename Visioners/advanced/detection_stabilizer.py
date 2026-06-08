"""Stabilize frame detections across several consecutive frames."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .config import AdvancedConfig


@dataclass
class DetectionTrack:
    id: int
    color: str
    center: tuple[int, int]
    seen: int
    last_seen: float
    detection: object


class DetectionStabilizer:
    """Confirm detections only after seeing the same object several times."""

    def __init__(self, config: AdvancedConfig) -> None:
        self.config = config
        self.tracks: list[DetectionTrack] = []
        self.next_id = 1
        self.frame_count = 0

    def reset(self) -> None:
        self.tracks.clear()
        self.frame_count = 0

    @property
    def ready(self) -> bool:
        return self.frame_count >= self.config.detection_confirm_frames

    def filter(self, detections, now: float | None = None):
        now = time.monotonic() if now is None else now
        self.frame_count += 1
        self.tracks = [
            track
            for track in self.tracks
            if now - track.last_seen <= self.config.detection_track_max_age_sec
        ]
        confirmed = []
        matched_track_ids = set()

        for detection in detections:
            track = self._find_track(detection, matched_track_ids)
            if track is None:
                track = DetectionTrack(
                    id=self.next_id,
                    color=detection["color"],
                    center=detection["center"],
                    seen=0,
                    last_seen=now,
                    detection=detection,
                )
                self.next_id += 1
                self.tracks.append(track)

            matched_track_ids.add(track.id)
            track.seen += 1
            track.center = detection["center"]
            track.last_seen = now
            track.detection = detection
            if track.seen >= self.config.detection_confirm_frames:
                confirmed.append(detection)

        return confirmed

    def _find_track(self, detection, used_track_ids: set[int]) -> DetectionTrack | None:
        color = detection["color"]
        cx, cy = detection["center"]
        best_track = None
        best_distance = float("inf")
        for track in self.tracks:
            if track.id in used_track_ids or track.color != color:
                continue
            tx, ty = track.center
            center_distance = math.hypot(cx - tx, cy - ty)
            if center_distance < best_distance:
                best_track = track
                best_distance = center_distance
        if best_distance <= self.config.detection_track_radius_px:
            return best_track
        return None
