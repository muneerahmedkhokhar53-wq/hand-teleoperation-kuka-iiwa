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

    def test_gripper_motors_physically_open_and_close_fingers(self):
        self.robot.close_gripper()
        for _ in range(120):
            p.stepSimulation()
        self.assertLess(self.robot.gripper_opening, 0.03)

        self.robot.open_gripper()
        for _ in range(120):
            p.stepSimulation()
        self.assertGreater(
            self.robot.gripper_opening, rs.GRIPPER_MAX_OPENING * 0.8
        )

    def test_gripper_contacts_cube_with_both_fingers_and_releases_it(self):
        """Exercise real finger/cube collision, not only motor joint travel."""
        # Keep the deliberately suspended test cube in place so gravity does
        # not move it out of the gripper before the close command is tested.
        p.setGravity(0, 0, 0)
        body = self.robot.props[0][0]
        gripper_pos, gripper_orn = p.getBasePositionAndOrientation(
            self.robot.gripper_id
        )
        cube_pos, cube_orn = p.multiplyTransforms(
            gripper_pos,
            gripper_orn,
            [0.0, 0.0, rs.FINGER_CENTER_Z],
            [0.0, 0.0, 0.0, 1.0],
        )
        p.resetBasePositionAndOrientation(body, cube_pos, cube_orn)
        p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])

        self.robot.open_gripper()
        for _ in range(120):
            p.stepSimulation()

        open_contacts = p.getContactPoints(
            bodyA=self.robot.gripper_id, bodyB=body
        )
        self.assertFalse(
            any(contact[3] in (0, 1) for contact in open_contacts),
            "An open gripper should leave clearance around the test cube.",
        )

        self.robot.close_gripper()
        for _ in range(240):
            p.stepSimulation()

        contacts = p.getContactPoints(bodyA=self.robot.gripper_id, bodyB=body)
        contacting_fingers = {
            contact[3] for contact in contacts if contact[3] in (0, 1)
        }
        self.assertEqual(
            contacting_fingers,
            {0, 1},
            "A closed gripper should contact the cube with both fingers.",
        )

        self.robot.open_gripper()
        for _ in range(120):
            p.stepSimulation()
        released_contacts = p.getContactPoints(
            bodyA=self.robot.gripper_id, bodyB=body
        )
        self.assertFalse(
            any(contact[3] in (0, 1) for contact in released_contacts),
            "Opening the gripper should release the cube from both fingers.",
        )

    def test_drop_zone_counts_only_a_settled_cube_inside_container(self):
        body = self.robot.props[0][0]
        inside_position = [
            rs.BOX_CENTER[0],
            rs.BOX_CENTER[1],
            rs.BOX_FLOOR_T + rs.CUBE_SIZE / 2.0,
        ]
        p.resetBasePositionAndOrientation(body, inside_position, [0, 0, 0, 1])
        p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
        self.assertEqual(self.robot.cubes_in_drop_zone(), 1)

        p.resetBaseVelocity(body, [0.2, 0, 0], [0, 0, 0])
        self.assertEqual(self.robot.cubes_in_drop_zone(), 0)

        p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
        p.resetBasePositionAndOrientation(
            body, [rs.BOX_CENTER[0] + rs.BOX_INNER_HALF + 0.01,
                   rs.BOX_CENTER[1], inside_position[2]], [0, 0, 0, 1]
        )
        self.assertEqual(self.robot.cubes_in_drop_zone(), 0)


if __name__ == "__main__":
    unittest.main()
