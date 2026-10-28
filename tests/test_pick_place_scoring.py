"""Regression tests for pick-and-place scoring."""

import unittest
from unittest.mock import patch

import robot_sim as rs


class DropZoneScoringTests(unittest.TestCase):
    def setUp(self):
        self.robot = rs.RobotArm()
        self.robot.props = [(1, [0, 0, 0]), (2, [0, 0, 0]), (3, [0, 0, 0])]
        cx, cy = rs.BOX_CENTER
        self.positions = {
            1: (cx, cy, rs.BOX_FLOOR_T + rs.CUBE_SIZE / 2.0, (0, 0, 0, 1)),
            2: (cx, cy, rs.BOX_FLOOR_T + rs.CUBE_SIZE / 2.0, (0, 0, 0, 1)),
            3: (cx + rs.BOX_INNER_HALF + 0.01, cy, rs.BOX_FLOOR_T + rs.CUBE_SIZE / 2.0, (0, 0, 0, 1)),
        }
        self.velocities = {
            1: ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            2: ((0.5, 0.0, 0.0), (0.0, 0.0, 0.0)),
            3: ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        }

    def test_counts_only_settled_cubes_inside_drop_zone(self):
        with (
            patch.object(rs.p, "getBasePositionAndOrientation",
                         side_effect=lambda body: (self.positions[body][:3], self.positions[body][3])),
            patch.object(rs.p, "getBaseVelocity",
                         side_effect=lambda body: self.velocities[body]),
        ):
            self.assertEqual(self.robot.cubes_in_drop_zone(), 1)

    def test_moving_cube_is_not_counted_until_it_settles(self):
        with (
            patch.object(rs.p, "getBasePositionAndOrientation",
                         return_value=((rs.BOX_CENTER[0], rs.BOX_CENTER[1], 0.07), (0, 0, 0, 1))),
            patch.object(rs.p, "getBaseVelocity",
                         return_value=((0.0, 0.0, 0.06), (0.0, 0.0, 0.0))),
        ):
            self.robot.props = [(1, [0, 0, 0])]
            self.assertEqual(self.robot.cubes_in_drop_zone(), 0)

    def test_cube_outside_container_is_not_counted_even_when_settled(self):
        with (
            patch.object(rs.p, "getBasePositionAndOrientation",
                         return_value=((rs.BOX_CENTER[0] + rs.BOX_INNER_HALF + 0.01,
                                        rs.BOX_CENTER[1], 0.07), (0, 0, 0, 1))),
            patch.object(rs.p, "getBaseVelocity",
                         return_value=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
        ):
            self.robot.props = [(1, [0, 0, 0])]
            self.assertEqual(self.robot.cubes_in_drop_zone(), 0)


if __name__ == "__main__":
    unittest.main()
