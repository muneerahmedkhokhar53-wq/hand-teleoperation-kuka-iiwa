"""
VISION TELEOPERATION  -  FULL HAND PICK & PLACE
===============================================

Control a simulated KUKA iiwa robot with your whole hand.

HAND
----
PALM CENTER    Left / Right -> Robot Y
               Up / Down    -> Robot Z
               Near / Far   -> Robot X   (hand size = depth)

GESTURES
--------
FIST        Clutch (robot holds its pose; reposition your hand freely)
PINCH       Close gripper (grab a cube)
OPEN PALM   Open gripper (release)

PICK & PLACE
------------
1. Move above a cube with an open palm.
2. Lower your hand until the gripper surrounds the cube.
3. PINCH to grab, lift, move to the green container.
4. OPEN PALM to release. Use FIST to re-center your hand mid-task.

KEYBOARD (camera window)
------------------------
Q / ESC   Quit            SPACE  Hold / Resume
R         Reset all       D      Depth ON / OFF
C         Calibrate depth H      HUD ON / OFF
N         Respawn cubes only

SAFETY
------
* The system starts in HOLD. Press SPACE once to hand over control.
* The robot is unlocked only while a hand is visible, HOLD is off and
  the fist clutch is released. Otherwise it locks and holds its pose.
* Targets are workspace-clamped, low-pass filtered and speed-limited.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pybullet as p

import hand_tracker as ht
import robot_sim as rs


# ============================================================================
# CONFIGURATION
# ============================================================================

WINDOW = "Vision Teleoperation - Camera"

# Camera control zone (normalized x0, y0, x1, y1). Smaller zone = less hand
# travel needed to cover the full robot range.
ZONE = (0.25, 0.15, 0.75, 0.70)

# Robot workspace mapping
X_CENTER = 0.52
X_RANGE = 0.20
Y_RANGE = 0.40          # also reaches the container box
Z_CENTER = 0.38
Z_RANGE = 0.30
LATERAL_SIGN = 1.0

# Sensitivity: >1 means a small hand move gives a big robot move.
XY_GAIN = 1.5

# Full-hand control point: wrist + the four finger knuckles (MCP joints).
# Averaging these gives a palm center that does NOT move when you pinch.
PALM_LANDMARKS = (0, 5, 9, 13, 17)

# Depth from apparent hand size
DEFAULT_NEUTRAL_SIZE = 0.20
DEPTH_GAIN = 5.0
DEPTH_DEADZONE = 0.03   # ignore tiny size jitter around neutral

# Motion
SMOOTH_TAU = 0.10  # seconds
MAX_SPEED = 0.9  # m/s cap on the target marker
MAX_SUBSTEPS = rs.MAX_SUBSTEPS

# Gripper gesture must be stable this many frames before it triggers
GESTURE_DEBOUNCE = 3

# HUD
AMBER = (60, 190, 255)
DIM_ACCENT = tuple(int(c * 0.55) for c in ht.ACCENT)
CHIP_W = 52
CHIP_GAP = 10


# ============================================================================
# HAND -> ROBOT MAPPING
# ============================================================================

class HandToRobot:
    """Maps the palm center and hand size to a Cartesian robot target."""

    def __init__(self) -> None:
        self.neutral = DEFAULT_NEUTRAL_SIZE
        self.depth_on = True  # needed to reach cubes
        self._size: Optional[float] = None

    def update_size(self, size: float) -> None:
        self._size = size if self._size is None else 0.8 * self._size + 0.2 * size

    def calibrate(self) -> bool:
        """Set the neutral hand size; return False until a hand is observed."""
        if self._size is None:
            print("Depth calibration pending: show your hand.")
            return False
        self.neutral = self._size
        print(f"Depth calibrated: {self.neutral:.4f}")
        return True

    @property
    def depth_ratio(self) -> float:
        if self._size is None or self.neutral <= 0:
            return 1.0
        return self._size / self.neutral

    @staticmethod
    def _unit(value: float, lo: float, hi: float) -> float:
        """Normalize to [-1, 1] across a zone axis."""
        return float(np.clip((value - lo) / (hi - lo) * 2.0 - 1.0, -1.0, 1.0))

    def _depth(self) -> float:
        if not self.depth_on:
            return 0.0
        raw = (self.depth_ratio - 1.0) * DEPTH_GAIN
        if abs(raw) < DEPTH_DEADZONE * DEPTH_GAIN:
            return 0.0
        return float(np.clip(raw, -1.0, 1.0))

    def map(self, palm_xy) -> np.ndarray:
        x0, y0, x1, y1 = ZONE

        ux = float(np.clip(self._unit(palm_xy[0], x0, x1) * XY_GAIN, -1.0, 1.0))
        uy = float(np.clip(self._unit(palm_xy[1], y0, y1) * XY_GAIN, -1.0, 1.0))

        target = np.array(
            [
                X_CENTER + self._depth() * X_RANGE,
                LATERAL_SIGN * ux * Y_RANGE,
                Z_CENTER - uy * Z_RANGE,
            ],
            dtype=np.float64,
        )
        return rs.clamp_to_workspace(target)


# ============================================================================
# TARGET FILTER (smoothing + speed limit)
# ============================================================================

class TargetFilter:
    """Exponential smoothing with a hard Cartesian speed limit."""

    def __init__(self, start: np.ndarray) -> None:
        self.value = np.asarray(start, dtype=np.float64).copy()

    def reset(self, position: np.ndarray) -> None:
        self.value = np.asarray(position, dtype=np.float64).copy()

    def step(self, desired: np.ndarray, dt: float) -> np.ndarray:
        """Move toward a finite target using a positive time step."""
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be a finite positive number")

        desired = np.asarray(desired, dtype=np.float64)
        if desired.shape != self.value.shape:
            raise ValueError("desired target must match the current position shape")
        if not np.all(np.isfinite(desired)):
            raise ValueError("desired target must contain only finite values")

        alpha = 1.0 - math.exp(-dt / SMOOTH_TAU)
        step = (desired - self.value) * alpha

        limit = MAX_SPEED * dt
        length = float(np.linalg.norm(step))
        if length > limit:
            step *= limit / length

        self.value = self.value + step
        return self.value


# ============================================================================
# GRIPPER GESTURE GATE (debounce)
# ============================================================================

class GripperGate:
    """Fires 'close' / 'open' only after a gesture is stable for N frames."""

    _MAP = {"PINCH": "close", "OPEN PALM": "open"}

    def __init__(self, frames: int = GESTURE_DEBOUNCE) -> None:
        if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
            raise ValueError("frames must be a positive integer")
        self.frames = frames
        self._pending: Optional[str] = None
        self._count = 0
        self._last_fired: Optional[str] = None

    def update(self, gesture: Optional[str]) -> Optional[str]:
        action = self._MAP.get(gesture)

        if action is None:
            self._pending, self._count = None, 0
            return None

        if action == self._pending:
            self._count += 1
        else:
            self._pending, self._count = action, 1

        if self._count >= self.frames and action != self._last_fired:
            self._last_fired = action
            return action
        return None

    def reset(self) -> None:
        self._pending, self._count, self._last_fired = None, 0, None


# ============================================================================
# TELEMETRY
# ============================================================================

@dataclass
class Telemetry:
    status: str
    status_color: tuple
    gesture: str
    fps: float
    target: np.ndarray
    ee: np.ndarray
    palm_xy: Optional[np.ndarray]
    depth_ratio: float
    depth_on: bool
    fingers: list
    gripper_fraction: float
    cubes_done: int
    cubes_total: int


# ============================================================================
# DRAWING HELPERS
# ============================================================================

def draw_zone(frame, active: bool) -> None:
    h, w = frame.shape[:2]
    x0, y0 = int(ZONE[0] * w), int(ZONE[1] * h)
    x1, y1 = int(ZONE[2] * w), int(ZONE[3] * h)
    color = ht.ACCENT if active else DIM_ACCENT

    ht.rounded_rect(frame, x0, y0, x1 - x0, y1 - y0, 18, color, 1)

    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    cv2.line(frame, (cx - 8, cy), (cx + 8, cy), DIM_ACCENT, 1, ht.AA)
    cv2.line(frame, (cx, cy - 8), (cx, cy + 8), DIM_ACCENT, 1, ht.AA)

    ht.put_text(frame, "CONTROL ZONE", (x0 + 16, y0 + 26), 0.45, color)


def draw_cursor(frame, tip, color) -> None:
    c = (int(tip[0]), int(tip[1]))
    cv2.circle(frame, c, 20, color, 2, ht.AA)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(
            frame,
            (c[0] + dx * 12, c[1] + dy * 12),
            (c[0] + dx * 28, c[1] + dy * 28),
            color,
            2,
            ht.AA,
        )


def draw_chips(frame, x: int, y: int, states) -> None:
    for i, name in enumerate(ht.FINGERS):
        cx = x + i * (CHIP_W + CHIP_GAP)
        label = name[0].upper()
        mid = (cx + CHIP_W // 2, y + 21)

        if states[i]:
            ht.rounded_rect(frame, cx, y, CHIP_W, 30, 8, ht.FINGER_COLORS[name])
            ht.put_text(frame, label, mid, 0.55, (20, 20, 20), 2, "center")
        else:
            ht.rounded_rect(frame, cx, y, CHIP_W, 30, 8, ht.PANEL_EDGE, 1)
            ht.put_text(frame, label, mid, 0.55, ht.MUTED, 1, "center")


def _row(frame, x: int, right: int, y: int, label: str, value: str, color) -> None:
    ht.put_text(frame, label, (x, y), 0.45, ht.MUTED)
    ht.put_text(frame, value, (right, y), 0.5, color, align="right")


def _bar(frame, x: int, y: int, w: int, fraction: float, color) -> None:
    h = 8
    cv2.rectangle(frame, (x, y), (x + w, y + h), ht.PANEL_EDGE, 1)
    fill = int((w - 2) * float(np.clip(fraction, 0.0, 1.0)))
    if fill > 0:
        cv2.rectangle(frame, (x + 1, y + 1), (x + 1 + fill, y + h - 1), color, -1)


def draw_hud(frame, t: Telemetry) -> None:
    h, w = frame.shape[:2]

    px, py, pw, ph, pad = 24, 24, 340, 565, 20

    hint = (
        "Q Quit   SPACE Hold   R Reset   "
        "D Depth   C Calibrate   N Cubes   H HUD"
    )
    hint_w = cv2.getTextSize(hint, ht.FONT, 0.45, 1)[0][0] + 44

    ht.blend_panels(
        frame,
        [
            (px, py, pw, ph, 16),
            ((w - hint_w) // 2, h - 56, hint_w, 36, 18),
        ],
    )
    ht.put_text(frame, hint, (w // 2, h - 33), 0.45, ht.MUTED, align="center")

    x = px + pad
    cw = pw - 2 * pad
    right = x + cw
    y = py + 38

    # Header
    ht.put_text(
        frame, "VISION TELEOPERATION", (x, y), 0.55, ht.TEXT, 1,
        font=cv2.FONT_HERSHEY_DUPLEX,
    )
    cv2.circle(frame, (right - 4, y - 6), 5, t.status_color, -1, ht.AA)
    ht.divider(frame, x, y + 14, cw)

    # Summary
    y += 44
    for label, value, color in (
        ("STATUS", t.status, t.status_color),
        ("GESTURE", t.gesture, ht.TEXT),
        ("FPS", f"{t.fps:4.1f}", ht.TEXT),
    ):
        _row(frame, x, right, y, label, value, color)
        y += 26
    y -= 12
    ht.divider(frame, x, y, cw)

    # Target / actual
    y += 28
    cols = (x, x + 100, x + 205)
    for c, title in zip(cols, ("AXIS", "TARGET", "ACTUAL")):
        ht.put_text(frame, title, (c, y), 0.4, ht.ACCENT)

    for k, axis in enumerate("XYZ"):
        y += 28
        ht.put_text(frame, axis, (cols[0], y), 0.5, ht.MUTED)
        ht.put_text(frame, f"{t.target[k]:+.3f}", (cols[1], y), 0.46, ht.TEXT)
        ht.put_text(frame, f"{t.ee[k]:+.3f}", (cols[2], y), 0.46, ht.TEXT)

    # Tracking error
    y += 28
    error_mm = 1000.0 * float(np.linalg.norm(t.target - t.ee))
    _row(
        frame, x, right, y, "TRACKING ERROR", f"{error_mm:5.1f} mm",
        ht.GOOD if error_mm < 25 else AMBER,
    )
    ht.divider(frame, x, y + 14, cw)

    # Hand
    y += 40
    palm_text = (
        f"{t.palm_xy[0]:.2f} / {t.palm_xy[1]:.2f}"
        if t.palm_xy is not None
        else "--"
    )
    _row(frame, x, right, y, "PALM X / Y", palm_text, ht.TEXT)

    y += 26
    depth_text = f"x{t.depth_ratio:.2f}" if t.depth_on else "OFF"
    _row(
        frame, x, right, y, "DEPTH", depth_text,
        ht.TEXT if t.depth_on else ht.MUTED,
    )

    # Gripper (measured from the physical fingers)
    y += 28
    opening_mm = t.gripper_fraction * rs.GRIPPER_MAX_OPENING * 1000.0
    _row(
        frame, x, right, y, "GRIPPER",
        f"{opening_mm:5.1f} mm  {t.gripper_fraction * 100:3.0f}%", ht.TEXT,
    )
    y += 12
    _bar(frame, x, y, cw, t.gripper_fraction, ht.ACCENT)

    # Pick & place score
    y += 30
    done = t.cubes_done == t.cubes_total and t.cubes_total > 0
    _row(
        frame, x, right, y, "CUBES IN ZONE",
        f"{t.cubes_done} / {t.cubes_total}",
        ht.GOOD if done else ht.TEXT,
    )

    # Finger states
    y += 20
    draw_chips(frame, x, y, t.fingers)


# ============================================================================
# MAIN
# ============================================================================

def print_banner() -> None:
    print()
    print("==========================================")
    print(" VISION TELEOPERATION - FULL HAND PICK&PLACE")
    print(" KUKA IIWA + HAND + GRIPPER")
    print("==========================================")
    print()
    print("PALM CENTER  = ROBOT MOVEMENT")
    print("FIST         = CLUTCH")
    print("PINCH        = CLOSE GRIPPER")
    print("OPEN PALM    = OPEN GRIPPER")
    print()
    print("Q / ESC quit | SPACE hold/resume | R reset | N respawn cubes")
    print("D depth | C calibrate depth | H HUD")
    print()
    print("System starts in HOLD. Press SPACE to enable hand control.")
    print()


def main() -> None:
    print_banner()

    cap = None
    tracker = None

    try:
        # ------------------------------------------------------------
        # Camera and vision
        # ------------------------------------------------------------
        print("Starting camera...")
        cap = ht.open_camera()
        tracker = ht.HandTracker()
        analyzer = ht.GestureAnalyzer()
        mapper = HandToRobot()
        gate = GripperGate()
        cam_trail = deque(maxlen=ht.TRAIL_LENGTH)

        # ------------------------------------------------------------
        # Simulation
        # ------------------------------------------------------------
        print("Starting PyBullet...")
        if p.isConnected():
            p.disconnect()

        bg = " ".join(
            f"--background_color_{c}={v}"
            for c, v in zip(("red", "green", "blue"), rs.BACKGROUND)
        )
        if p.connect(p.GUI, options=bg) < 0:
            raise RuntimeError("Could not connect to PyBullet.")

        robot = rs.build_scene()
        robot.disable_control()

        home = robot.get_end_effector_position()
        target = TargetFilter(home)
        marker = rs.make_target_marker(home)

        # ------------------------------------------------------------
        # State
        # ------------------------------------------------------------
        hold = True
        was_engaged = False
        show_hud = True
        first_resume = True  # auto-calibrate depth on the first SPACE

        fps = 0.0
        last_time = time.perf_counter()
        sim_clock = 0.0
        frame_no = 0

        last_trail = home.copy()
        link_line = -1

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

        # ------------------------------------------------------------
        # Main loop
        # ------------------------------------------------------------
        while p.isConnected():

            ok, frame = cap.read()
            if not ok:
                print("ERROR: Could not read camera.")
                break

            frame = cv2.flip(frame, 1)  # mirror
            hand = tracker.process(frame, mirrored=True)
            frame = ht.fit_height(frame, ht.DISPLAY_HEIGHT)
            h, w = frame.shape[:2]

            # Timing
            now = time.perf_counter()
            dt = float(np.clip(now - last_time, 1e-3, 0.1))
            last_time = now
            inst = 1.0 / dt
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst

            # Hand data
            info = None
            palm_norm = None
            palm_px = None
            px = None

            if hand is not None:
                pts = hand.norm[:, :2] * (w, h)
                px = np.rint(pts).astype(np.int32)

                info = analyzer.analyze(pts)

                # Full-hand control point: palm center (stable while pinching)
                palm_norm = hand.norm[list(PALM_LANDMARKS), :2].mean(axis=0)
                palm_px = (palm_norm * (w, h)).astype(np.int32)

                mapper.update_size(float(np.linalg.norm(pts[0] - pts[9]) / h))
                cam_trail.append(tuple(palm_px))

                # If SPACE was pressed before a hand was visible, defer the
                # first depth calibration until the first active hand frame.
                if not hold and first_resume and mapper.calibrate():
                    first_resume = False
            else:
                cam_trail.clear()
                # Do not carry a previous hand's gesture votes into a new
                # tracking session; stale votes could briefly unlock motion.
                analyzer.reset()
                gate.update(None)

            gesture = info[3] if info is not None else "NO HAND"

            # Status
            if hand is None:
                status, status_color = "NO HAND - HOLDING", ht.BAD
            elif hold:
                status, status_color = "PAUSED", AMBER
            elif gesture == "FIST":
                status, status_color = "CLUTCH (FIST)", AMBER
            else:
                status, status_color = "ENGAGED", ht.GOOD

            # The robot may move ONLY when a hand is seen, HOLD is off,
            # the fist clutch is released and the palm position is known.
            engaged = (
                hand is not None
                and not hold
                and gesture != "FIST"
                and palm_norm is not None
            )

            # Engagement edges: lock / unlock the arm and re-sync the target
            if engaged and not was_engaged:
                target.reset(robot.get_end_effector_position())
                robot.enable_control()
            elif was_engaged and not engaged:
                robot.disable_control()
                target.reset(robot.get_end_effector_position())
            was_engaged = engaged

            if engaged:
                desired = mapper.map(palm_norm)
                smoothed = target.step(desired, dt)
                robot.command(robot.solve(smoothed))

            # Gripper (debounced, only with a visible hand).
            # The state is latched: the gripper keeps holding a cube while
            # you clutch with a fist or momentarily lose the gesture.
            if hand is not None and not hold:
                action = gate.update(gesture)
                if action == "close":
                    robot.close_gripper()
                elif action == "open":
                    robot.open_gripper()

            # Physics
            sim_clock = min(sim_clock + dt, rs.TIME_STEP * MAX_SUBSTEPS)
            for _ in range(int(sim_clock / rs.TIME_STEP)):
                p.stepSimulation()
                sim_clock -= rs.TIME_STEP

            ee = robot.get_end_effector_position()

            # Simulation overlays
            rs.update_target_marker(marker, target.value)

            link_line = p.addUserDebugLine(
                ee.tolist(),
                target.value.tolist(),
                (1, 1, 1),
                lineWidth=1,
                replaceItemUniqueId=link_line,
            )

            frame_no += 1
            if frame_no % 2 == 0 and np.linalg.norm(ee - last_trail) > 0.004:
                p.addUserDebugLine(
                    last_trail.tolist(),
                    ee.tolist(),
                    rs.ACCENT,
                    lineWidth=3,
                    lifeTime=rs.TRAIL_LIFETIME,
                )
                last_trail = ee.copy()

            # Camera overlay
            draw_zone(frame, engaged)

            if hand is not None:
                ht.draw_trail(frame, cam_trail)
                ht.draw_hand(frame, px, info[2], info[1])
                draw_cursor(frame, palm_px, status_color)

            if show_hud:
                draw_hud(
                    frame,
                    Telemetry(
                        status=status,
                        status_color=status_color,
                        gesture=gesture,
                        fps=fps,
                        target=target.value.copy(),
                        ee=ee.copy(),
                        palm_xy=palm_norm.copy() if palm_norm is not None else None,
                        depth_ratio=mapper.depth_ratio,
                        depth_on=mapper.depth_on,
                        fingers=info[0] if info is not None else [False] * 5,
                        gripper_fraction=robot.gripper_fraction,
                        cubes_done=robot.cubes_in_drop_zone(),
                        cubes_total=robot.cube_total,
                    ),
                )

            cv2.imshow(WINDOW, frame)

            # Keyboard
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            elif key == ord(" "):
                hold = not hold
                print("Manual HOLD enabled." if hold else "Manual HOLD disabled.")
                # First depth calibration happens when an active hand
                # is visible, even if SPACE was pressed before the hand appeared.

            elif key == ord("r"):
                print("Resetting robot...")
                robot.reset()
                home = robot.get_end_effector_position()
                target.reset(home)
                last_trail = home.copy()
                tracker.reset()
                cam_trail.clear()
                gate.reset()
                hold = True  # always resume paused
                was_engaged = False
                print("Reset complete. Press SPACE to enable control.")

            elif key == ord("n"):
                robot.reset_props()
                print("Cubes respawned.")

            elif key == ord("d"):
                mapper.depth_on = not mapper.depth_on
                print("Depth: " + ("ON" if mapper.depth_on else "OFF"))

            elif key == ord("c"):
                mapper.calibrate()

            elif key == ord("h"):
                show_hud = not show_hud

    except KeyboardInterrupt:
        print("\nStopped by user.")

    except p.error as error:
        print("\nPyBullet error:")
        print(error)

    except Exception as error:
        print("\nUnexpected error:")
        print(error)
        raise

    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        if tracker is not None:
            try:
                tracker.close()
            except Exception:
                pass

        if p.isConnected():
            p.disconnect()

        print()
        print("Vision teleoperation stopped.")


if __name__ == "__main__":
    main()
