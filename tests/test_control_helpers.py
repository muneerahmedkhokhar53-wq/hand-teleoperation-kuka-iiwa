"""Regression tests for teleoperation control-state helpers.

Run with: python -m unittest discover -s tests -v
"""

import unittest

import numpy as np

import hand_tracker as ht
import teleoperation as teleop


class GestureAnalyzerTests(unittest.TestCase):
    def test_reset_clears_pinching_and_stale_votes(self):
        analyzer = ht.GestureAnalyzer()
        points = np.zeros((21, 2), dtype=np.float64)
        points[ht.L.WRIST] = (0.0, 0.0)
        points[ht.L.MIDDLE_FINGER_MCP] = (0.0, 1.0)
        points[ht.L.THUMB_TIP] = (0.5, 0.5)
        points[ht.L.INDEX_FINGER_TIP] = (0.5, 0.5)

        result = analyzer.analyze(points)
        self.assertTrue(result.pinching)
        self.assertTrue(analyzer._votes)

        analyzer.reset()

        self.assertFalse(analyzer._pinching)
        self.assertEqual(len(analyzer._votes), 0)


class HandCalibrationTests(unittest.TestCase):
    def test_calibration_waits_until_hand_size_is_available(self):
        mapper = teleop.HandToRobot()

        self.assertFalse(mapper.calibrate())
        self.assertEqual(mapper.neutral, teleop.DEFAULT_NEUTRAL_SIZE)

        mapper.update_size(0.31)
        self.assertTrue(mapper.calibrate())
        self.assertAlmostEqual(mapper.neutral, 0.31)


class TargetFilterTests(unittest.TestCase):
    def test_step_respects_cartesian_speed_limit(self):
        start = np.array([0.4, 0.0, 0.4], dtype=np.float64)
        target_filter = teleop.TargetFilter(start)
        desired = np.array([100.0, -100.0, 100.0], dtype=np.float64)
        dt = 0.01

        actual = target_filter.step(desired, dt)
        distance = float(np.linalg.norm(actual - start))

        self.assertLessEqual(distance, teleop.MAX_SPEED * dt + 1e-12)
        self.assertTrue(np.all(np.isfinite(actual)))

    def test_hand_mapping_returns_finite_three_axis_target(self):
        mapper = teleop.HandToRobot()
        mapper.update_size(0.2)

        target = mapper.map(np.array([0.5, 0.4], dtype=np.float64))

        self.assertEqual(target.shape, (3,))
        self.assertTrue(np.all(np.isfinite(target)))



class WorkspaceClampTests(unittest.TestCase):
    def assert_inside_workspace(self, target):
        point = teleop.rs.clamp_to_workspace(target)
        radius = float(np.linalg.norm(point - teleop.rs.SHOULDER))

        self.assertTrue(np.all(np.isfinite(point)))
        self.assertGreaterEqual(float(point[2]), teleop.rs.MIN_Z - 1e-12)
        self.assertGreaterEqual(float(np.hypot(point[0], point[1])), teleop.rs.MIN_RADIUS - 1e-12)
        self.assertLessEqual(radius, teleop.rs.MAX_REACH + 1e-9)

    def test_extreme_vertical_target_respects_inner_cylinder_and_reach(self):
        self.assert_inside_workspace([0.1, 0.0, 10.0])

    def test_origin_target_respects_all_workspace_constraints(self):
        self.assert_inside_workspace([0.0, 0.0, 0.0])

    def test_non_finite_target_is_rejected(self):
        with self.assertRaises(ValueError):
            teleop.rs.clamp_to_workspace([float("nan"), 0.0, 0.5])

    def test_wrong_target_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            teleop.rs.clamp_to_workspace([0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
