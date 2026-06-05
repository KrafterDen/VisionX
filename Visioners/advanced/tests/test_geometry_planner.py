from __future__ import annotations

import unittest

from Visioners.advanced.config import AdvancedConfig
from Visioners.advanced.geometry import (
    compute_heading,
    distance,
    generate_waypoint_candidates,
    normalize_angle_deg,
    point_to_segment_distance,
    segment_intersects_forbidden_zone,
    turn_angle,
)
from Visioners.advanced.mapping import PinMap, RobotPose
from Visioners.advanced.planner import plan_path, select_target


class GeometryTests(unittest.TestCase):
    def test_angle_helpers(self) -> None:
        self.assertEqual(normalize_angle_deg(190), -170)
        self.assertEqual(normalize_angle_deg(-190), 170)
        self.assertAlmostEqual(compute_heading((0, 0), (0, 10)), 0)
        self.assertAlmostEqual(compute_heading((0, 0), (10, 0)), 90)
        self.assertAlmostEqual(turn_angle(170, -170), 20)

    def test_point_to_segment_distance(self) -> None:
        self.assertAlmostEqual(point_to_segment_distance((5, 4), (0, 0), (10, 0)), 4)
        self.assertAlmostEqual(point_to_segment_distance((12, 0), (0, 0), (10, 0)), 2)

    def test_forbidden_zone_intersection(self) -> None:
        self.assertTrue(segment_intersects_forbidden_zone((0, 0), (10, 0), (5, 1), 2))
        self.assertFalse(segment_intersects_forbidden_zone((0, 0), (10, 0), (5, 4), 2))

    def test_waypoint_candidates(self) -> None:
        candidates = generate_waypoint_candidates((0, 0), (10, 0), (5, 0), 3, 2)
        self.assertEqual(len(candidates), 2)
        self.assertAlmostEqual(distance(candidates[0], (5, 0)), 5)
        self.assertAlmostEqual(distance(candidates[1], (5, 0)), 5)


class MappingPlannerTests(unittest.TestCase):
    def test_map_merging(self) -> None:
        pin_map = PinMap()
        first = pin_map.add_observation(
            color="blue",
            x=10,
            y=40,
            angle_deg=10,
            distance_cm=41,
            confidence=0.7,
            area=1000,
            target_color="blue",
            merge_radius_cm=18,
        )
        second = pin_map.add_observation(
            color="blue",
            x=14,
            y=44,
            angle_deg=12,
            distance_cm=46,
            confidence=0.9,
            area=1200,
            target_color="blue",
            merge_radius_cm=18,
        )
        self.assertEqual(first.id, second.id)
        self.assertEqual(len(pin_map.pins), 1)
        self.assertEqual(pin_map.pins[0].observations, 2)

    def test_map_cleanup_merges_color_jitter_and_drops_singletons(self) -> None:
        pin_map = PinMap()
        pin_map.add_observation(
            color="purple",
            x=10,
            y=50,
            angle_deg=10,
            distance_cm=51,
            confidence=0.8,
            area=1000,
            target_color="blue",
            merge_radius_cm=18,
        )
        pin_map.add_observation(
            color="blue",
            x=18,
            y=54,
            angle_deg=12,
            distance_cm=56,
            confidence=0.7,
            area=900,
            target_color="blue",
            merge_radius_cm=18,
        )
        pin_map.add_observation(
            color="blue",
            x=17,
            y=55,
            angle_deg=12,
            distance_cm=57,
            confidence=0.7,
            area=900,
            target_color="blue",
            merge_radius_cm=18,
        )
        pin_map.add_observation(
            color="purple",
            x=80,
            y=50,
            angle_deg=60,
            distance_cm=94,
            confidence=0.4,
            area=800,
            target_color="blue",
            merge_radius_cm=18,
        )
        pin_map.cleanup(
            target_color="blue",
            same_color_radius_cm=18,
            cross_color_radius_cm=14,
            cleanup_merge_radius_cm=22,
            min_observations=2,
            target_min_votes=2,
        )

        self.assertEqual(len(pin_map.alive_pins()), 1)
        self.assertEqual(pin_map.pins[0].color, "blue")
        self.assertTrue(pin_map.pins[0].is_target)
        self.assertEqual(pin_map.pins[0].color_votes["blue"], 2)

    def test_map_cleanup_does_not_merge_unrelated_colors(self) -> None:
        pin_map = PinMap()
        pin_map.add_observation(
            color="red",
            x=10,
            y=50,
            angle_deg=10,
            distance_cm=51,
            confidence=0.8,
            area=1000,
            target_color="blue",
            merge_radius_cm=18,
        )
        pin_map.add_observation(
            color="red",
            x=10,
            y=50,
            angle_deg=10,
            distance_cm=51,
            confidence=0.8,
            area=1000,
            target_color="blue",
            merge_radius_cm=18,
        )
        pin_map.add_observation(
            color="blue",
            x=16,
            y=54,
            angle_deg=12,
            distance_cm=56,
            confidence=0.7,
            area=900,
            target_color="blue",
            merge_radius_cm=18,
        )
        pin_map.add_observation(
            color="blue",
            x=16,
            y=54,
            angle_deg=12,
            distance_cm=56,
            confidence=0.7,
            area=900,
            target_color="blue",
            merge_radius_cm=18,
        )
        pin_map.cleanup(
            target_color="blue",
            same_color_radius_cm=18,
            cross_color_radius_cm=14,
            cleanup_merge_radius_cm=22,
            min_observations=2,
            target_min_votes=2,
        )

        self.assertEqual(len(pin_map.alive_pins()), 2)
        self.assertEqual({pin.color for pin in pin_map.alive_pins()}, {"red", "blue"})

    def test_target_selection_penalizes_blocked_path(self) -> None:
        config = AdvancedConfig(target_color="blue")
        pin_map = PinMap()
        blocked = pin_map.add_observation(
            color="blue",
            x=0,
            y=100,
            angle_deg=0,
            distance_cm=100,
            confidence=1,
            area=1000,
            target_color="blue",
            merge_radius_cm=1,
        )
        clear = pin_map.add_observation(
            color="blue",
            x=80,
            y=80,
            angle_deg=45,
            distance_cm=113,
            confidence=1,
            area=1000,
            target_color="blue",
            merge_radius_cm=1,
        )
        pin_map.add_observation(
            color="red",
            x=0,
            y=50,
            angle_deg=0,
            distance_cm=50,
            confidence=1,
            area=1000,
            target_color="blue",
            merge_radius_cm=1,
        )
        selected = select_target(pin_map, RobotPose(), config)
        self.assertEqual(selected.id, clear.id)
        self.assertNotEqual(selected.id, blocked.id)

    def test_plan_path_creates_waypoint_for_blocker(self) -> None:
        config = AdvancedConfig(target_color="blue")
        pin_map = PinMap()
        target = pin_map.add_observation(
            color="blue",
            x=0,
            y=100,
            angle_deg=0,
            distance_cm=100,
            confidence=1,
            area=1000,
            target_color="blue",
            merge_radius_cm=1,
        )
        obstacle = pin_map.add_observation(
            color="red",
            x=0,
            y=50,
            angle_deg=0,
            distance_cm=50,
            confidence=1,
            area=1000,
            target_color="blue",
            merge_radius_cm=1,
        )
        path = plan_path(RobotPose(), target, [obstacle], config)
        self.assertFalse(path.direct_path_safe)
        self.assertIsNotNone(path.waypoint)


if __name__ == "__main__":
    unittest.main()
