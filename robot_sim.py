"""
Vision Teleoperation - KUKA iiwa Robot Simulation
=================================================

PyBullet simulation of a 7-DOF KUKA iiwa arm with a physically simulated
two-finger parallel gripper and a clean pick-and-place workcell with a
large open container box.

Coordinate frame
----------------
    X = forward / back
    Y = left / right
    Z = up / down

Safety model
------------
The arm is created *locked*: motors actively hold the current pose, and any
call to `command()` is ignored until `enable_control()` is called.
`disable_control()` re-locks the arm and holds wherever it currently is
(the motors are never released, so the arm cannot collapse under gravity).

Standalone demo (keyboard, click the PyBullet window first):
    E enable   D disable   G open   C close   R reset   Q quit
    I/K  X +/-     J/L  Y +/-     U/O  Z +/-
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p
import pybullet_data


# ============================================================================
# SIMULATION
# ============================================================================

TIME_STEP = 1.0 / 240.0
MAX_SUBSTEPS = 30
TRAIL_LIFETIME = 2.5

# A cube only counts as delivered after it settles inside the drop container.
DROP_ZONE_REST_SPEED = 0.05  # m/s
DROP_ZONE_REST_ANGULAR_SPEED = 0.10  # rad/s


# ============================================================================
# ROBOT
# ============================================================================

HOME_Q = (0.0, 0.55, 0.0, -1.45, 0.0, 1.10, 0.0)

# Fallback limits (KUKA iiwa 14) used if the URDF reports none.
FALLBACK_LOWER = (-2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054)
FALLBACK_UPPER = (2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054)

JOINT_MAX_VELOCITY = 2.0  # rad/s, keeps motion smooth and realistic
JOINT_MIN_FORCE = 200.0  # N*m

# End-effector orientation: tool axis (link +Z) points straight down.
TOOL_DOWN = p.getQuaternionFromEuler([0.0, -math.pi, 0.0])


# ============================================================================
# GRIPPER GEOMETRY
# (defined before the workspace because MIN_Z depends on TOOL_LENGTH)
# ============================================================================

GRIPPER_MAX_OPENING = 0.19  # gap between finger faces when fully open (m)
GRIPPER_FORCE = 60.0  # N per finger
GRIPPER_SPEED = 0.30  # m/s per finger

PALM_HALF = (0.040, 0.125, 0.018)  # palm spans the wider fingers
PALM_CENTER_Z = 0.030

FINGER_HALF = (0.016, 0.009, 0.060)
FINGER_CENTER_Z = PALM_CENTER_Z + PALM_HALF[2] + FINGER_HALF[2]

TOOL_LENGTH = FINGER_CENTER_Z + FINGER_HALF[2]  # flange -> fingertip (~0.168 m)


# ============================================================================
# WORKSPACE
# ============================================================================

SHOULDER = np.array([0.0, 0.0, 0.36], dtype=np.float64)

MAX_REACH = 0.80  # max flange distance from the shoulder
MIN_RADIUS = 0.20  # keep the flange away from the robot's own base column

# Lowest flange height: fingertips stop ~1 cm above the floor, so the
# gripper can reach the cubes but never goes through the floor.
FINGERTIP_CLEARANCE = 0.010
MIN_Z = TOOL_LENGTH + FINGERTIP_CLEARANCE

SMOOTH_TAU = 0.08

# Base pedestal
BASE_PLATE_RADIUS = 0.24
BASE_RIM_RADIUS = 0.25

# Round work table, centred on the robot (reach ring on the floor is ~0.78 m)
TABLE_RADIUS = 0.855
TABLE_EDGE_RADIUS = 0.87


# ============================================================================
# COLORS
# ============================================================================

BACKGROUND = (0.030, 0.040, 0.065)
FLOOR_RGBA = (0.060, 0.070, 0.095, 1.0)

LINK_COLORS = (
    (0.93, 0.94, 0.96, 1.0),
    (1.00, 0.48, 0.10, 1.0),
)

ACCENT = (0.25, 0.80, 0.95)
TARGET_RGBA = (0.25, 0.80, 0.95, 0.85)

GRIPPER_PALM_RGBA = (0.10, 0.13, 0.17, 1.0)
GRIPPER_FINGER_RGBA = (0.00, 0.88, 1.00, 1.0)

GRID_MINOR = (0.11, 0.13, 0.18)
GRID_MAJOR = (0.17, 0.20, 0.27)

TABLE_RGBA = (0.115, 0.130, 0.165, 1.0)
TABLE_EDGE_RGBA = (0.25, 0.80, 0.95, 1.0)

CUBE_COLORS = (
    (1.00, 0.45, 0.10, 1.0),
    (0.25, 0.80, 0.95, 1.0),
    (0.95, 0.30, 0.65, 1.0),
)


# ============================================================================
# CUBES AND CONTAINER
# ============================================================================

# (x, y) of each cube. Staggered triangle with 12 cm+ between faces so the
# open gripper (0.23 m across) never touches a neighbouring cube.
CUBE_SIZE = 0.12
CUBE_MASS = 0.15
CUBE_SPOTS = ((0.44, 0.36), (0.44, 0.12), (0.62, 0.24))

# Container box (open top, real collision walls), fully inside the reach ring
BOX_CENTER = (0.46, -0.27)
BOX_INNER_HALF = 0.15   # inside is 0.30 x 0.30 m
BOX_WALL_T = 0.012
BOX_WALL_H = 0.09       # lower than a cube, so the gripper can drop cubes in
BOX_FLOOR_T = 0.01

DROP_ZONE = BOX_CENTER
DROP_ZONE_HALF = BOX_INNER_HALF


# ============================================================================
# WORKSPACE LIMITS
# ============================================================================

def clamp_to_workspace(target: Sequence[float]) -> np.ndarray:
    """Clamp a finite XYZ target to the intersection of the safe workspace.

    The feasible region is above MIN_Z, outside the base's inner cylinder,
    and inside the shoulder-centered reach sphere. Applying the constraints
    only once can violate the inner-cylinder limit when the reach projection
    moves a point back toward the base, so the final point is revalidated.
    """
    t = np.asarray(target, dtype=np.float64).copy()
    if t.shape != (3,):
        raise ValueError("Workspace target must contain exactly three XYZ values.")
    if not np.all(np.isfinite(t)):
        raise ValueError("Workspace target must contain only finite values.")

    # Floor: the fingertips must never go through it.
    t[2] = max(float(t[2]), MIN_Z)

    # First project into the outer reach sphere.
    offset = t - SHOULDER
    dist = float(np.linalg.norm(offset))
    if dist > MAX_REACH:
        t = SHOULDER + offset * (MAX_REACH / dist)

    # Enforce the inner cylinder after reach projection.
    r = math.hypot(float(t[0]), float(t[1]))
    if r < MIN_RADIUS:
        if r < 1e-9:
            t[0], t[1] = MIN_RADIUS, 0.0
        else:
            t[0] *= MIN_RADIUS / r
            t[1] *= MIN_RADIUS / r
        r = MIN_RADIUS

    # Moving outward from the base can push the point outside the sphere.
    # Clamp height to the sphere's valid vertical interval at this radius.
    z_span = math.sqrt(max(0.0, MAX_REACH * MAX_REACH - r * r))
    min_z_at_radius = float(SHOULDER[2]) - z_span
    max_z_at_radius = float(SHOULDER[2]) + z_span
    t[2] = max(MIN_Z, min(max_z_at_radius, max(min_z_at_radius, float(t[2]))))

    return t


# ============================================================================
# ROBOT ARM
# ============================================================================

class RobotArm:
    """KUKA iiwa + parallel gripper, with a hard control lock."""

    def __init__(self) -> None:
        self.id: Optional[int] = None
        self.joints: List[int] = []
        self.ee_link = -1

        self.lower: List[float] = []
        self.upper: List[float] = []
        self.ranges: List[float] = []
        self.forces: List[float] = []

        self.control_enabled = False

        self.gripper_id: Optional[int] = None
        self._mount: Optional[int] = None
        self._gripper_target = GRIPPER_MAX_OPENING

        self.props: List[Tuple[int, List[float]]] = []  # (body id, home position)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        self.id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=[0.0, 0.0, 0.0],
            useFixedBase=True,
        )

        self.joints = [
            i
            for i in range(p.getNumJoints(self.id))
            if p.getJointInfo(self.id, i)[2] == p.JOINT_REVOLUTE
        ]
        if len(self.joints) < 7:
            raise RuntimeError("KUKA iiwa did not provide 7 revolute joints.")
        self.joints = self.joints[:7]
        self.ee_link = self.joints[-1]

        infos = [p.getJointInfo(self.id, j) for j in self.joints]
        for k, info in enumerate(infos):
            lo, hi = info[8], info[9]
            if hi <= lo:
                lo, hi = FALLBACK_LOWER[k], FALLBACK_UPPER[k]
            self.lower.append(lo)
            self.upper.append(hi)
            self.ranges.append(hi - lo)
            self.forces.append(max(info[10], JOINT_MIN_FORCE))

        self._paint()
        self._create_gripper()
        self.reset(reset_props=False)

    # ------------------------------------------------------------------
    # Appearance
    # ------------------------------------------------------------------

    def _paint(self) -> None:
        for n, link in enumerate(range(-1, p.getNumJoints(self.id))):
            p.changeVisualShape(
                self.id,
                link,
                rgbaColor=LINK_COLORS[n % 2],
                specularColor=[0.5, 0.5, 0.5],
            )

    # ------------------------------------------------------------------
    # Safety lock
    # ------------------------------------------------------------------

    def enable_control(self) -> None:
        self.control_enabled = True

    def disable_control(self) -> None:
        """Lock the arm and hold its current pose."""
        self.control_enabled = False
        if self.id is not None:
            self._apply(self.joint_positions())

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def joint_positions(self) -> List[float]:
        return [s[0] for s in p.getJointStates(self.id, self.joints)]

    def get_end_effector_pose(self) -> Tuple[np.ndarray, Tuple[float, ...]]:
        """World pose of the flange (URDF link frame)."""
        state = p.getLinkState(
            self.id, self.ee_link, computeForwardKinematics=True
        )
        return np.array(state[4], dtype=np.float64), tuple(state[5])

    def get_end_effector_position(self) -> np.ndarray:
        return self.get_end_effector_pose()[0]

    def get_tool_tip_position(self) -> np.ndarray:
        """World position of the point between the fingertips."""
        pos, orn = self.get_end_effector_pose()
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        return pos + rot @ np.array([0.0, 0.0, TOOL_LENGTH])

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, reset_props: bool = True) -> None:
        for joint, q in zip(self.joints, HOME_Q):
            p.resetJointState(self.id, joint, q, targetVelocity=0.0)

        self.control_enabled = False
        self._apply(HOME_Q)

        # Re-seat the gripper on the flange and open it.
        pos, orn = self.get_end_effector_pose()
        p.resetBasePositionAndOrientation(self.gripper_id, pos.tolist(), orn)
        p.resetBaseVelocity(self.gripper_id, [0, 0, 0], [0, 0, 0])
        for j in (0, 1):
            p.resetJointState(
                self.gripper_id, j, GRIPPER_MAX_OPENING / 2.0, targetVelocity=0.0
            )
        self.open_gripper()

        if reset_props:
            self.reset_props()

    def reset_props(self) -> None:
        for body, pos in self.props:
            p.resetBasePositionAndOrientation(body, pos, [0, 0, 0, 1])
            p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])

    # ------------------------------------------------------------------
    # Pick-and-place scoring
    # ------------------------------------------------------------------

    def cubes_in_drop_zone(self) -> int:
        """Number of settled cubes inside the container."""
        cx, cy = BOX_CENTER
        count = 0
        for body, _ in self.props:
            pos, _ = p.getBasePositionAndOrientation(body)
            linear_velocity, angular_velocity = p.getBaseVelocity(body)
            settled = (
                np.linalg.norm(linear_velocity) <= DROP_ZONE_REST_SPEED
                and np.linalg.norm(angular_velocity) <= DROP_ZONE_REST_ANGULAR_SPEED
            )
            if (
                settled
                and abs(pos[0] - cx) < BOX_INNER_HALF
                and abs(pos[1] - cy) < BOX_INNER_HALF
                and pos[2] < BOX_FLOOR_T + CUBE_SIZE / 2.0 + 0.02
            ):
                count += 1
        return count

    @property
    def cube_total(self) -> int:
        return len(self.props)

    # ------------------------------------------------------------------
    # Inverse kinematics
    # ------------------------------------------------------------------

    def solve(self, target: Sequence[float]) -> List[float]:
        if self.id is None:
            raise RuntimeError("Robot has not been loaded.")

        target = clamp_to_workspace(target)

        kwargs = dict(
            lowerLimits=self.lower,
            upperLimits=self.upper,
            jointRanges=self.ranges,
            restPoses=list(HOME_Q),
            maxNumIterations=200,
            residualThreshold=1e-4,
        )

        try:
            solution = p.calculateInverseKinematics(
                self.id,
                self.ee_link,
                target.tolist(),
                TOOL_DOWN,
                currentPositions=self.joint_positions(),
                **kwargs,
            )
        except TypeError:  # older PyBullet without currentPositions
            solution = p.calculateInverseKinematics(
                self.id, self.ee_link, target.tolist(), TOOL_DOWN, **kwargs
            )

        return list(solution[:7])

    # ------------------------------------------------------------------
    # Motor commands
    # ------------------------------------------------------------------

    def _apply(self, q: Sequence[float]) -> None:
        for joint, target, force in zip(self.joints, q, self.forces):
            p.setJointMotorControl2(
                self.id,
                joint,
                p.POSITION_CONTROL,
                targetPosition=float(target),
                force=force,
                maxVelocity=JOINT_MAX_VELOCITY,
            )

    def command(self, joint_positions: Sequence[float]) -> None:
        """Send joint targets. Ignored while the arm is locked."""
        if not self.control_enabled or self.id is None:
            return
        if len(joint_positions) < 7:
            return
        self._apply(joint_positions)

    # ------------------------------------------------------------------
    # Gripper
    # ------------------------------------------------------------------

    def _create_gripper(self) -> None:
        pos, orn = self.get_end_effector_pose()

        # Palm
        palm_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=list(PALM_HALF),
            collisionFramePosition=[0, 0, PALM_CENTER_Z],
        )
        palm_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=list(PALM_HALF),
            rgbaColor=GRIPPER_PALM_RGBA,
            visualFramePosition=[0, 0, PALM_CENTER_Z],
        )

        # Fingers
        finger_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=list(FINGER_HALF)
        )
        finger_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=list(FINGER_HALF),
            rgbaColor=GRIPPER_FINGER_RGBA,
        )

        t = FINGER_HALF[1]  # at q = 0 the inner faces touch (gripper closed)
        z = FINGER_CENTER_Z

        self.gripper_id = p.createMultiBody(
            baseMass=0.35,
            baseCollisionShapeIndex=palm_col,
            baseVisualShapeIndex=palm_vis,
            basePosition=pos.tolist(),
            baseOrientation=orn,
            linkMasses=[0.06, 0.06],
            linkCollisionShapeIndices=[finger_col, finger_col],
            linkVisualShapeIndices=[finger_vis, finger_vis],
            linkPositions=[[0, -t, z], [0, t, z]],
            linkOrientations=[[0, 0, 0, 1]] * 2,
            linkInertialFramePositions=[[0, 0, 0]] * 2,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * 2,
            linkParentIndices=[0, 0],
            linkJointTypes=[p.JOINT_PRISMATIC, p.JOINT_PRISMATIC],
            linkJointAxis=[[0, -1, 0], [0, 1, 0]],
        )

        for j in (0, 1):
            p.changeDynamics(
                self.gripper_id,
                j,
                jointLowerLimit=0.0,
                jointUpperLimit=GRIPPER_MAX_OPENING / 2.0,
                lateralFriction=1.5,
                spinningFriction=0.02,
                rollingFriction=0.002,
                restitution=0.0,
            )

        # The gripper must not fight the arm it is mounted on.
        for link in range(-1, p.getNumJoints(self.id)):
            for g in (-1, 0, 1):
                p.setCollisionFilterPair(self.id, self.gripper_id, link, g, 0)

        # Rigidly mount the gripper to the flange. Constraint frames are
        # expressed relative to the link's inertial frame, so convert.
        state = p.getLinkState(
            self.id, self.ee_link, computeForwardKinematics=True
        )
        inv_pos, inv_orn = p.invertTransform(state[0], state[1])
        rel_pos, rel_orn = p.multiplyTransforms(
            inv_pos, inv_orn, state[4], state[5]
        )

        self._mount = p.createConstraint(
            parentBodyUniqueId=self.id,
            parentLinkIndex=self.ee_link,
            childBodyUniqueId=self.gripper_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=list(rel_pos),
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=rel_orn,
            childFrameOrientation=[0, 0, 0, 1],
        )
        p.changeConstraint(self._mount, maxForce=800)

    @property
    def gripper_opening(self) -> float:
        """Measured gap between the fingers in meters."""
        if self.gripper_id is None:
            return GRIPPER_MAX_OPENING
        left = p.getJointState(self.gripper_id, 0)[0]
        right = p.getJointState(self.gripper_id, 1)[0]
        return float(np.clip(left + right, 0.0, GRIPPER_MAX_OPENING))

    @property
    def gripper_fraction(self) -> float:
        """0.0 = closed, 1.0 = fully open."""
        return self.gripper_opening / GRIPPER_MAX_OPENING

    def set_gripper(self, opening: float) -> None:
        self._gripper_target = float(np.clip(opening, 0.0, GRIPPER_MAX_OPENING))
        if self.gripper_id is None:
            return
        for j in (0, 1):
            p.setJointMotorControl2(
                self.gripper_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=self._gripper_target / 2.0,
                force=GRIPPER_FORCE,
                maxVelocity=GRIPPER_SPEED,
            )

    def open_gripper(self) -> None:
        self.set_gripper(GRIPPER_MAX_OPENING)

    def close_gripper(self) -> None:
        self.set_gripper(0.0)

    def update(self) -> None:
        """Kept for API compatibility. The gripper is fully physical now."""
        return


# ============================================================================
# SCENE
# ============================================================================

def _draw_floor_grid(extent: float = 1.2, step: float = 0.2) -> None:
    """Soft, sparse floor grid."""
    n = int(round(extent / step))
    for i in range(-n, n + 1):
        major = i % 5 == 0
        color = GRID_MAJOR if major else GRID_MINOR
        width = 1.5 if major else 1.0
        c = i * step
        p.addUserDebugLine([c, -extent, 0.001], [c, extent, 0.001], color, width)
        p.addUserDebugLine([-extent, c, 0.001], [extent, c, 0.001], color, width)


def _draw_reach_ring(n: int = 144) -> None:
    """ONE dashed cyan ring = the arm's real reach on the floor."""
    dz = MIN_Z - SHOULDER[2]
    radius = math.sqrt(max(MAX_REACH**2 - dz**2, 0.0))
    for i in range(n):
        if i % 2:  # skip every other segment -> dashed look
            continue
        a0 = 2.0 * math.pi * i / n
        a1 = 2.0 * math.pi * (i + 1) / n
        p.addUserDebugLine(
            [radius * math.cos(a0), radius * math.sin(a0), 0.006],
            [radius * math.cos(a1), radius * math.sin(a1), 0.006],
            ACCENT,
            lineWidth=2.0,
        )


def _static_box(half, pos, rgba) -> int:
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
    return p.createMultiBody(
        baseMass=0, baseVisualShapeIndex=vis, basePosition=pos
    )


def _solid_box(half, pos, rgba) -> int:
    """Static box WITH collision (used for the container walls)."""
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
    body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos,
    )
    p.changeDynamics(body, -1, lateralFriction=1.0, restitution=0.0)
    return body


def _build_drop_box() -> None:
    cx, cy = BOX_CENTER
    i, t, h, f = BOX_INNER_HALF, BOX_WALL_T, BOX_WALL_H, BOX_FLOOR_T
    floor_color = (0.18, 0.80, 0.50, 1.0)
    wall_color = (0.10, 0.58, 0.38, 1.0)
    wall_z = f + h / 2.0

    # Floor
    _solid_box([i + t, i + t, f / 2.0], [cx, cy, f / 2.0], floor_color)

    # Four walls
    _solid_box([i + t, t / 2.0, h / 2.0], [cx, cy + i + t / 2.0, wall_z], wall_color)
    _solid_box([i + t, t / 2.0, h / 2.0], [cx, cy - i - t / 2.0, wall_z], wall_color)
    _solid_box([t / 2.0, i, h / 2.0], [cx + i + t / 2.0, cy, wall_z], wall_color)
    _solid_box([t / 2.0, i, h / 2.0], [cx - i - t / 2.0, cy, wall_z], wall_color)


def _build_pick_pads() -> None:
    """Flat colored pads under each cube so the spacing looks deliberate."""
    pad_half = CUBE_SIZE / 2.0 + 0.025
    for (x, y), color in zip(CUBE_SPOTS, CUBE_COLORS):
        dim = (color[0] * 0.45, color[1] * 0.45, color[2] * 0.45, 1.0)
        _static_box([pad_half, pad_half, 0.0006], [x, y, 0.0036], dim)


def _build_set_dressing() -> None:
    # Round work table centred on the robot, with a thin cyan edge.
    edge = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=TABLE_EDGE_RADIUS,
        length=0.002,
        rgbaColor=TABLE_EDGE_RGBA,
    )
    p.createMultiBody(0, baseVisualShapeIndex=edge, basePosition=[0, 0, 0.001])

    table = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=TABLE_RADIUS,
        length=0.003,
        rgbaColor=TABLE_RGBA,
    )
    p.createMultiBody(0, baseVisualShapeIndex=table, basePosition=[0, 0, 0.0015])

    # Base pedestal
    plate = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=BASE_PLATE_RADIUS,
        length=0.02,
        rgbaColor=(0.10, 0.12, 0.16, 1),
    )
    p.createMultiBody(0, baseVisualShapeIndex=plate, basePosition=[0, 0, 0.01])

    rim = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=BASE_RIM_RADIUS,
        length=0.004,
        rgbaColor=(*ACCENT, 1.0),
    )
    p.createMultiBody(0, baseVisualShapeIndex=rim, basePosition=[0, 0, 0.004])

    # Pick pads and container
    _build_pick_pads()
    _build_drop_box()


def _spawn_cubes(robot: RobotArm) -> None:
    half = CUBE_SIZE / 2.0
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half] * 3)

    for (x, y), color in zip(CUBE_SPOTS, CUBE_COLORS):
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[half] * 3,
            rgbaColor=color,
            specularColor=[0.4, 0.4, 0.4],
        )
        pos = [x, y, half + 0.002]
        body = p.createMultiBody(
            baseMass=CUBE_MASS,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
        )
        p.changeDynamics(
            body,
            -1,
            lateralFriction=2.0,
            spinningFriction=0.03,
            rollingFriction=0.002,
            restitution=0.0,
        )
        robot.props.append((body, pos))


def build_scene() -> RobotArm:
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(TIME_STEP)
    p.setRealTimeSimulation(0)

    for flag in (
        p.COV_ENABLE_RGB_BUFFER_PREVIEW,
        p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
        p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
        p.COV_ENABLE_GUI,
    ):
        p.configureDebugVisualizer(flag, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

    # Camera framing the whole round table
    p.resetDebugVisualizerCamera(
        cameraDistance=1.6,
        cameraYaw=50,
        cameraPitch=-35,
        cameraTargetPosition=[0.25, 0.0, 0.05],
    )

    floor = p.loadURDF("plane.urdf")
    p.changeVisualShape(floor, -1, textureUniqueId=-1, rgbaColor=FLOOR_RGBA)
    p.changeDynamics(floor, -1, lateralFriction=1.0)

    _draw_floor_grid()
    _draw_reach_ring()
    _build_set_dressing()

    robot = RobotArm()
    robot.load()
    _spawn_cubes(robot)

    # Always start locked.
    robot.disable_control()
    return robot


# ============================================================================
# TARGET MARKER
# ============================================================================

def make_target_marker(position: Sequence[float] = (0.0, 0.0, 1.0)) -> int:
    visual = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.022, rgbaColor=TARGET_RGBA
    )
    return p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual,
        basePosition=[float(v) for v in position],
    )


def update_target_marker(marker: int, position: Sequence[float]) -> None:
    p.resetBasePositionAndOrientation(
        marker, [float(v) for v in position], [0, 0, 0, 1]
    )


# ============================================================================
# STANDALONE DEMO
# ============================================================================

def _pressed(keys, char: str) -> bool:
    k = ord(char)
    return k in keys and bool(keys[k] & p.KEY_IS_DOWN)


def _triggered(keys, char: str) -> bool:
    k = ord(char)
    return k in keys and bool(keys[k] & p.KEY_WAS_TRIGGERED)


def main() -> None:
    print(__doc__)

    bg = " ".join(
        f"--background_color_{c}={v}"
        for c, v in zip(("red", "green", "blue"), BACKGROUND)
    )
    if p.connect(p.GUI, options=bg) < 0:
        raise RuntimeError("Could not connect to PyBullet.")

    try:
        robot = build_scene()
        target = robot.get_end_effector_position()
        marker = make_target_marker(target)
        speed = 0.35  # m/s

        while p.isConnected():
            keys = p.getKeyboardEvents()

            if _triggered(keys, "q"):
                break
            if _triggered(keys, "e"):
                target = robot.get_end_effector_position()
                robot.enable_control()
                print("Control ENABLED")
            if _triggered(keys, "d"):
                robot.disable_control()
                print("Control DISABLED")
            if _triggered(keys, "g"):
                robot.open_gripper()
            if _triggered(keys, "c"):
                robot.close_gripper()
            if _triggered(keys, "r"):
                robot.reset()
                target = robot.get_end_effector_position()
                print("Reset")

            if robot.control_enabled:
                d = speed * TIME_STEP
                target = target + np.array(
                    [
                        d * (_pressed(keys, "i") - _pressed(keys, "k")),
                        d * (_pressed(keys, "j") - _pressed(keys, "l")),
                        d * (_pressed(keys, "u") - _pressed(keys, "o")),
                    ]
                )
                target = clamp_to_workspace(target)
                robot.command(robot.solve(target))

            update_target_marker(marker, target)
            p.stepSimulation()
            time.sleep(TIME_STEP)

    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
