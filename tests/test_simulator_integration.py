"""Headless PyBullet integration smoke tests for the simulator."""

import unittest

import numpy as np
import pybullet as p

import robot_sim as rs


class SimulatorIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.client = p.connect(p.DIRECT)
        self.assertGreaterEqual(self.client, 0)
        self.robot = rs.build_scene()

    def tearDown(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)

    def test_scene_loads_seven_joint_arm_and_pick_place_props(self):
        self.assertEqual(len(self.robot.joints), 7)
        self.assertEqual(self.robot.cube_total, len(rs.CUBE_SPOTS))
        self.assertFalse(self.robot.control_enabled)
        self.assertAlmostEqual(
            self.robot.gripper_opening, rs.GRIPPER_MAX_OPENING, delta=0.02
        )

    def test_locked_arm_ignores_joint_commands(self):
        before = np.asarray(self.robot.joint_positions(), dtype=float)
        self.robot.command([q + 0.2 for q in before])
        for _ in range(10):
            p.stepSimulation()
        after = np.asarray(self.robot.joint_positions(), dtype=float)

        self.assertFalse(self.robot.control_enabled)
        np.testing.assert_allclose(after, before, atol=0.03)

    def test_reset_relocks_arm_and_restores_open_gripper(self):
        self.robot.enable_control()
        self.robot.close_gripper()
        self.robot.reset()

        self.assertFalse(self.robot.control_enabled)
        self.assertAlmostEqual(
            self.robot._gripper_target, rs.GRIPPER_MAX_OPENING, places=6
        )
        np.testing.assert_allclose(
            self.robot.joint_positions(), rs.HOME_Q, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
