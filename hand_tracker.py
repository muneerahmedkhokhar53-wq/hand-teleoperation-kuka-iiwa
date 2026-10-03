"""
Vision Teleoperation - Hand Tracking Module
============================================

Tracks one hand using MediaPipe and provides:

- 21 smoothed hand landmarks (One Euro filter, per-landmark adaptive)
- Angle-based finger detection (rotation invariant)
- Gesture recognition with majority-vote stabilisation
- Pinch detection with hysteresis
- Futuristic HUD drawing (glow skeleton, brackets, status panel)

Public API (backward compatible)
--------------------------------
    HandTracker, Hand, GestureAnalyzer, OneEuroFilter, finger_states,
    put_text, rounded_rect, blend_panels, divider, draw_trail, draw_hand,
    open_camera, fit_height

New
---
    GestureResult, FpsMeter, TrailBuffer, draw_hud, draw_bar
"""

from __future__ import annotations

import math
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, NamedTuple, Optional, Sequence

import cv2
import mediapipe as mp
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

WINDOW = "Vision Teleoperation"

CAMERA_INDEX = 0
CAPTURE_SIZE = (640, 480)
DISPLAY_HEIGHT = 720

TRAIL_LENGTH = 28

PINCH_ENTER = 0.28
PINCH_EXIT = 0.38
PINCH_RANGE = 0.90          # ratio at which pinch strength reaches 0

FINGER_EXTEND_ANGLE = 140.0  # degrees at the PIP / IP joint
GESTURE_VOTE_FRAMES = 5


# ============================================================================
# COLORS - BGR FORMAT
# ============================================================================

PANEL_BG = (28, 22, 20)
PANEL_EDGE = (78, 68, 62)

TEXT = (245, 245, 245)
MUTED = (150, 146, 142)

ACCENT = (240, 205, 70)

GOOD = (130, 230, 130)
BAD = (90, 90, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA


# ============================================================================
# MEDIAPIPE
# ============================================================================

mp_hands = mp.solutions.hands
L = mp_hands.HandLandmark

FINGERS = ("thumb", "index", "middle", "ring", "pinky")

FINGER_COLORS = {
    "thumb": (60, 160, 255),
    "index": (255, 205, 70),
    "middle": (130, 230, 130),
    "ring": (70, 220, 255),
    "pinky": (230, 120, 255),
}

FINGER_CHAINS = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

PALM_BONES = (
    (0, 1),
    (0, 5),
    (0, 17),
    (5, 9),
    (9, 13),
    (13, 17),
)

FINGERTIPS = (4, 8, 12, 16, 20)

TRACKED = (
    ("WRIST", L.WRIST),
    ("THUMB TIP", L.THUMB_TIP),
    ("INDEX TIP", L.INDEX_FINGER_TIP),
    ("MIDDLE TIP", L.MIDDLE_FINGER_TIP),
)

_GREY = (150, 150, 150)


# ============================================================================
# SMALL UTILITIES
# ============================================================================

def _pt(point) -> tuple[int, int]:
    """numpy point -> plain Python int tuple (required by recent OpenCV)."""
    return int(point[0]), int(point[1])


class FpsMeter:
    """Exponentially smoothed frames-per-second counter."""

    def __init__(self, smoothing: float = 0.9):
        self._s = smoothing
        self._t: Optional[float] = None
        self.fps = 0.0

    def tick(self) -> float:
        now = time.perf_counter()

        if self._t is not None:
            inst = 1.0 / max(now - self._t, 1e-6)
            self.fps = inst if self.fps == 0.0 else self._s * self.fps + (1 - self._s) * inst

        self._t = now
        return self.fps


class TrailBuffer:
    """Fixed-length fingertip trail."""

    def __init__(self, length: int = TRAIL_LENGTH):
        self._pts: Deque[tuple[int, int]] = deque(maxlen=length)

    def push(self, point):
        self._pts.append(_pt(point))

    def clear(self):
        self._pts.clear()

    def __iter__(self):
        return iter(self._pts)

    def __len__(self):
        return len(self._pts)

    def __getitem__(self, i):
        return list(self._pts)[i]


# ============================================================================
# ONE EURO FILTER
# ============================================================================

class OneEuroFilter:
    """
    Adaptive smoothing filter.

    When the hand is still:   strong smoothing (no jitter)
    When the hand moves:      higher cutoff (no lag)

    Speed is measured per landmark (vector norm) so smoothing does not
    depend on the direction of motion.
    """

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 12.0,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.reset()

    def reset(self):
        self._x: Optional[np.ndarray] = None
        self._dx: Optional[np.ndarray] = None
        self._t = 0.0

    @staticmethod
    def _alpha(cutoff, dt: float):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:

        if self._x is None:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            self._t = t
            return x.copy()

        dt = max(t - self._t, 1e-4)

        dx = (x - self._x) / dt

        a_d = self._alpha(self.d_cutoff, dt)

        self._dx = a_d * dx + (1.0 - a_d) * self._dx

        speed = np.linalg.norm(self._dx, axis=-1, keepdims=True)

        cutoff = self.min_cutoff + self.beta * speed

        a = self._alpha(cutoff, dt)

        self._x = a * x + (1.0 - a) * self._x
        self._t = t

        return self._x.copy()


# ============================================================================
# HAND DATA
# ============================================================================

@dataclass
class Hand:
    """Smoothed hand. `norm` is (21, 3) in MediaPipe normalised coordinates."""

    norm: np.ndarray
    label: str
    score: float

    @property
    def palm_center(self) -> np.ndarray:
        """Stable control point: mean of wrist + the four finger MCPs."""
        return self.norm[[0, 5, 9, 13, 17]].mean(axis=0)

    @property
    def palm_size(self) -> float:
        """Wrist -> middle MCP length (normalised). Grows as the hand nears the camera."""
        return float(np.linalg.norm(self.norm[L.WRIST][:2] - self.norm[L.MIDDLE_FINGER_MCP][:2]))

    def pixels(self, width: int, height: int) -> np.ndarray:
        """Landmarks as int32 (21, 2) pixel coordinates."""
        px = np.rint(self.norm[:, :2] * (width, height))
        return np.clip(px, -width, 2 * width).astype(np.int32)


# ============================================================================
# HAND TRACKER
# ============================================================================

class HandTracker:

    def __init__(
        self,
        model_complexity: int = 0,
        detection_confidence: float = 0.6,
        tracking_confidence: float = 0.6,
    ):
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=model_complexity,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        self._filter = OneEuroFilter()
        self.inference_ms = 0.0

    def reset(self):
        self._filter.reset()

    def process(self, bgr: np.ndarray, mirrored: bool) -> Optional[Hand]:

        t0 = time.perf_counter()

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        result = self._hands.process(rgb)

        self.inference_ms = (time.perf_counter() - t0) * 1000.0

        if not result.multi_hand_landmarks:
            self._filter.reset()
            return None

        raw = np.array(
            [[pt.x, pt.y, pt.z] for pt in result.multi_hand_landmarks[0].landmark],
            dtype=np.float64,
        )

        classification = result.multi_handedness[0].classification[0]

        label = classification.label

        # MediaPipe assumes a mirrored (selfie) image
        if not mirrored:
            label = "Left" if label == "Right" else "Right"

        smoothed = self._filter(raw, time.perf_counter())

        return Hand(smoothed, label, float(classification.score))

    def close(self):
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ============================================================================
# FINGER DETECTION
# ============================================================================

def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle in degrees at vertex b."""
    v1, v2 = a - b, c - b
    cosine = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def finger_states(pts: np.ndarray) -> list[bool]:
    """
    [thumb, index, middle, ring, pinky] -> extended?

    A finger counts as extended when its joint is nearly straight AND the
    tip is farther from the wrist than the middle joint. Using the joint
    angle makes this independent of hand rotation.
    """

    def dist(a, b) -> float:
        return float(np.linalg.norm(pts[a] - pts[b]))

    thumb = bool(
        _angle(pts[L.THUMB_MCP], pts[L.THUMB_IP], pts[L.THUMB_TIP]) > FINGER_EXTEND_ANGLE
        and dist(L.THUMB_TIP, L.PINKY_MCP) > dist(L.THUMB_IP, L.PINKY_MCP)
    )

    others = [
        bool(
            _angle(pts[mcp], pts[pip], pts[tip]) > FINGER_EXTEND_ANGLE
            and dist(tip, L.WRIST) > dist(pip, L.WRIST)
        )
        for mcp, pip, tip in (
            (5, 6, 8),
            (9, 10, 12),
            (13, 14, 16),
            (17, 18, 20),
        )
    ]

    return [thumb, *others]


# ============================================================================
# GESTURE ANALYZER
# ============================================================================

class GestureResult(NamedTuple):
    """Unpacks like the original 4-tuple."""
    states: list
    strength: float
    pinching: bool
    gesture: str


class GestureAnalyzer:

    def __init__(self):
        self._pinching = False
        self._votes: Deque[str] = deque(maxlen=GESTURE_VOTE_FRAMES)
        self.ratio = 1.0

    def reset(self):
        self._pinching = False
        self._votes.clear()

    def analyze(self, pts: np.ndarray) -> GestureResult:

        reference = float(np.linalg.norm(pts[L.WRIST] - pts[L.MIDDLE_FINGER_MCP])) + 1e-6

        self.ratio = float(
            np.linalg.norm(pts[L.THUMB_TIP] - pts[L.INDEX_FINGER_TIP]) / reference
        )

        # Hysteresis: easy to hold, hard to flicker
        if self._pinching:
            self._pinching = self.ratio < PINCH_EXIT
        else:
            self._pinching = self.ratio < PINCH_ENTER

        strength = float(np.clip(1.0 - self.ratio / PINCH_RANGE, 0.0, 1.0))

        states = finger_states(pts)

        self._votes.append(self._name(states))

        if self._pinching:
            gesture = "PINCH"  # no voting delay for the control gesture
        else:
            gesture = Counter(self._votes).most_common(1)[0][0]

        return GestureResult(states, strength, self._pinching, gesture)

    @staticmethod
    def _name(states: Sequence[bool]) -> str:

        thumb, index, middle, ring, pinky = states

        if all(states):
            return "OPEN PALM"

        if not any(states[1:]):
            return "THUMB" if thumb else "FIST"

        if index and not (middle or ring or pinky):
            return "POINTING"

        if index and middle and not (ring or pinky):
            return "PEACE"

        if index and pinky and not (middle or ring):
            return "ROCK"

        return "TRACKING"


# ============================================================================
# DRAWING PRIMITIVES
# ============================================================================

def put_text(
    img,
    text,
    org,
    scale=0.5,
    color=TEXT,
    thickness=1,
    align="left",
    font=FONT,
):
    x, y = org

    if align != "left":
        width = cv2.getTextSize(text, font, scale, thickness)[0][0]
        x -= width if align == "right" else width // 2

    cv2.putText(img, text, (int(x), int(y)), font, scale, color, thickness, AA)


def rounded_rect(img, x, y, w, h, r, color, thickness=-1):

    x, y, w, h = int(x), int(y), int(w), int(h)
    r = int(min(r, w // 2, h // 2))

    if thickness < 0:

        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)

        for cx, cy in (
            (x + r, y + r),
            (x + w - r, y + r),
            (x + r, y + h - r),
            (x + w - r, y + h - r),
        ):
            cv2.circle(img, (cx, cy), r, color, -1, AA)

        return

    for p1, p2 in (
        ((x + r, y), (x + w - r, y)),
        ((x + r, y + h), (x + w - r, y + h)),
        ((x, y + r), (x, y + h - r)),
        ((x + w, y + r), (x + w, y + h - r)),
    ):
        cv2.line(img, p1, p2, color, thickness, AA)

    for (cx, cy), angle in (
        ((x + r, y + r), 180),
        ((x + w - r, y + r), 270),
        ((x + w - r, y + h - r), 0),
        ((x + r, y + h - r), 90),
    ):
        cv2.ellipse(img, (cx, cy), (r, r), 0, angle, angle + 90, color, thickness, AA)


def blend_panels(frame, panels, alpha=0.72):
    """Translucent rounded panels: [(x, y, w, h, r), ...]."""

    overlay = frame.copy()

    for x, y, w, h, r in panels:
        rounded_rect(overlay, x, y, w, h, r, PANEL_BG)

    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    for x, y, w, h, r in panels:
        rounded_rect(frame, x, y, w, h, r, PANEL_EDGE, 1)


def divider(frame, x, y, w):
    cv2.line(frame, (int(x), int(y)), (int(x + w), int(y)), PANEL_EDGE, 1, AA)


def draw_bar(frame, x, y, w, h, value, color, marker: Optional[float] = None):
    """Horizontal progress bar. `value` and `marker` are in 0..1."""

    value = float(np.clip(value, 0.0, 1.0))

    rounded_rect(frame, x, y, w, h, h // 2, (52, 44, 40))

    if value > 0.02:
        rounded_rect(frame, x, y, max(h, int(w * value)), h, h // 2, color)

    if marker is not None:
        mx = int(x + w * float(np.clip(marker, 0.0, 1.0)))
        cv2.line(frame, (mx, y - 3), (mx, y + h + 3), TEXT, 1, AA)


# ============================================================================
# HAND DRAWING
# ============================================================================

def bones():
    """Yield (a, b, color) for every skeleton segment exactly once."""

    for a, b in PALM_BONES:
        yield a, b, FINGER_COLORS["pinky"] if (a, b) == (0, 17) else _GREY

    for name, chain in FINGER_CHAINS.items():
        for a, b in zip(chain[:-1], chain[1:]):
            yield a, b, FINGER_COLORS[name]


def draw_trail(frame, trail):

    pts = [_pt(t) for t in trail]
    n = len(pts)

    for i in range(1, n):

        f = i / n

        color = tuple(int(c * (0.15 + 0.85 * f)) for c in FINGER_COLORS["index"])

        cv2.line(frame, pts[i - 1], pts[i], color, max(1, int(1 + 5 * f)), AA)


def draw_hand(
    frame,
    px: np.ndarray,
    pinching: bool,
    pinch_strength: float,
    label: Optional[str] = None,
    score: float = 0.0,
):
    """Glow skeleton, corner brackets, joints and pinch indicator."""

    h, w = frame.shape[:2]

    px = np.asarray(px).astype(np.int32)

    # ---- glow (computed on a cropped ROI for speed) ----------------------

    pad = 48

    x0 = int(max(px[:, 0].min() - pad, 0))
    y0 = int(max(px[:, 1].min() - pad, 0))
    x1 = int(min(px[:, 0].max() + pad, w))
    y1 = int(min(px[:, 1].max() + pad, h))

    if x1 > x0 and y1 > y0:

        roi = frame[y0:y1, x0:x1]
        glow = np.zeros_like(roi)
        local = px - (x0, y0)

        for a, b, color in bones():
            cv2.line(glow, _pt(local[a]), _pt(local[b]), color, 10, AA)

        glow = cv2.GaussianBlur(glow, (0, 0), 9)

        frame[y0:y1, x0:x1] = cv2.add(roi, glow)

    # ---- corner brackets --------------------------------------------------

    bx0, by0 = px.min(axis=0) - 28
    bx1, by1 = px.max(axis=0) + 28

    ln = 26

    for (cx, cy), (dx, dy) in (
        ((bx0, by0), (1, 1)),
        ((bx1, by0), (-1, 1)),
        ((bx0, by1), (1, -1)),
        ((bx1, by1), (-1, -1)),
    ):
        cv2.line(frame, (int(cx), int(cy)), (int(cx + dx * ln), int(cy)), ACCENT, 2, AA)
        cv2.line(frame, (int(cx), int(cy)), (int(cx), int(cy + dy * ln)), ACCENT, 2, AA)

    if label:
        tag = f"{label.upper()} HAND  {score * 100:.0f}%"
        put_text(frame, tag, (bx0, by0 - 10), 0.45, ACCENT, 1)

    # ---- skeleton ---------------------------------------------------------

    for a, b, color in bones():
        cv2.line(frame, _pt(px[a]), _pt(px[b]), color, 2, AA)

    # ---- joints -----------------------------------------------------------

    for i, point in enumerate(px):

        if i in FINGERTIPS:
            color = FINGER_COLORS[FINGERS[FINGERTIPS.index(i)]]
            cv2.circle(frame, _pt(point), 9, color, 2, AA)

        cv2.circle(frame, _pt(point), 4, (255, 255, 255), -1, AA)

    # ---- pinch indicator --------------------------------------------------

    thumb = px[L.THUMB_TIP]
    index = px[L.INDEX_FINGER_TIP]

    mid = ((thumb + index) // 2).astype(int)

    color = GOOD if pinching else (170, 170, 170)

    cv2.line(frame, _pt(thumb), _pt(index), color, 2 if pinching else 1, AA)

    radius = int(5 + 7 * pinch_strength)

    if pinching:
        pulse = 0.5 + 0.5 * math.sin(time.perf_counter() * 8.0)
        cv2.circle(frame, _pt(mid), radius + 6 + int(4 * pulse), color, 1, AA)

    cv2.circle(frame, _pt(mid), radius, color, 2 if pinching else 1, AA)


# ============================================================================
# HUD PANEL
# ============================================================================

def draw_hud(
    frame,
    hand: Optional[Hand],
    result: Optional[GestureResult],
    fps: float = 0.0,
    lines: Sequence[str] = (),
    x: int = 16,
    y: int = 16,
):
    """Status panel: tracking state, gesture, finger chips, pinch bar, coordinates."""

    w = 272
    tracking = hand is not None and result is not None

    h = (316 + 20 * len(lines)) if tracking else 92

    blend_panels(frame, [(x, y, w, h, 14)])

    # Header
    cv2.circle(frame, (x + 22, y + 24), 6, GOOD if tracking else BAD, -1, AA)
    put_text(frame, "VISION TELEOPERATION", (x + 38, y + 29), 0.5, TEXT, 1)
    put_text(frame, f"{fps:3.0f} FPS", (x + w - 16, y + 29), 0.42, MUTED, 1, "right")
    divider(frame, x + 14, y + 44, w - 28)

    if not tracking:
        put_text(frame, "NO HAND DETECTED", (x + 20, y + 74), 0.55, BAD, 1)
        return

    states, strength, pinching, gesture = result

    # Gesture
    put_text(frame, "GESTURE", (x + 20, y + 68), 0.38, MUTED)
    put_text(frame, gesture, (x + 20, y + 98), 0.85, GOOD if pinching else ACCENT, 2, font=FONT_BOLD)
    put_text(
        frame,
        f"{hand.label.upper()}  {hand.score * 100:.0f}%",
        (x + w - 18, y + 98),
        0.42,
        MUTED,
        1,
        "right",
    )

    # Finger chips
    for i, (name, letter) in enumerate(zip(FINGERS, "TIMRP")):

        cx = x + 20 + i * 48
        cy = y + 116
        color = FINGER_COLORS[name]

        if states[i]:
            rounded_rect(frame, cx, cy, 40, 26, 8, color)
            put_text(frame, letter, (cx + 20, cy + 18), 0.5, PANEL_BG, 2, "center")
        else:
            rounded_rect(frame, cx, cy, 40, 26, 8, PANEL_EDGE, 1)
            put_text(frame, letter, (cx + 20, cy + 18), 0.5, MUTED, 1, "center")

    # Pinch bar
    put_text(frame, "PINCH", (x + 20, y + 170), 0.38, MUTED)
    put_text(frame, f"{strength * 100:3.0f}%", (x + w - 18, y + 170), 0.38, MUTED, 1, "right")

    draw_bar(
        frame,
        x + 20,
        y + 178,
        w - 40,
        10,
        strength,
        GOOD if pinching else ACCENT,
        marker=1.0 - PINCH_ENTER / PINCH_RANGE,
    )

    divider(frame, x + 14, y + 206, w - 28)

    # Tracked coordinates
    put_text(frame, "LANDMARK", (x + 20, y + 228), 0.36, MUTED)
    for dx, axis in ((168, "X"), (212, "Y"), (w - 18, "Z")):
        put_text(frame, axis, (x + dx, y + 228), 0.36, MUTED, 1, "right")

    for row, (name, idx) in enumerate(TRACKED):

        ry = y + 252 + row * 24
        px_, py_, pz_ = hand.norm[idx]

        put_text(frame, name, (x + 20, ry), 0.4, TEXT)
        put_text(frame, f"{px_:5.2f}", (x + 168, ry), 0.4, ACCENT, 1, "right")
        put_text(frame, f"{py_:5.2f}", (x + 212, ry), 0.4, ACCENT, 1, "right")
        put_text(frame, f"{pz_:+5.2f}", (x + w - 18, ry), 0.4, ACCENT, 1, "right")

    # Extra caller lines (e.g. robot target, reach %)
    for i, text in enumerate(lines):
        put_text(frame, text, (x + 20, y + 322 + i * 20), 0.4, MUTED)


# ============================================================================
# CAMERA
# ============================================================================

def open_camera():

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise SystemExit("ERROR: Camera could not be opened.")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always the freshest frame, minimal lag

    return cap


def fit_height(frame, height):

    h, w = frame.shape[:2]

    if h == height:
        return frame

    return cv2.resize(
        frame,
        (int(round(w * height / h)), height),
        interpolation=cv2.INTER_LINEAR,
    )


# ============================================================================
# END
# ============================================================================

if __name__ == "__main__":

    print("hand_tracker.py is a module.")
    print("Run teleoperation.py instead.")
