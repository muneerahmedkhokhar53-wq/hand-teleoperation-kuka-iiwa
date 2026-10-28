# Vision Teleoperation: Hand-Controlled KUKA iiwa Pick & Place

Control a simulated **7-DOF KUKA iiwa** robot arm with your **bare hand and a webcam**. Hand landmarks are tracked in real time with MediaPipe, mapped to a Cartesian target, and solved with inverse kinematics inside a **PyBullet** physics simulation. A physically simulated two-finger gripper lets you pick up cubes and drop them into a container box.

<!-- Add a real, checked-in demo at docs/demo.gif when available. Do not use a placeholder image. -->

## Features

- **Full-hand control:** the palm center (wrist + four knuckles) drives the robot, so the control point stays steady while you pinch.
- **Depth from hand size:** move your hand toward or away from the camera to move the robot forward and back.
- **Gesture control:** pinch to close the gripper, open palm to release, fist to clutch.
- **Physical gripper:** two prismatic fingers with friction and force limits, not a "magic" attach.
- **Smooth, safe motion:** One Euro filtering on landmarks, low-pass filtering plus a speed limit on the target, and workspace clamping.
- **Safety lock:** the arm starts on HOLD and moves only while a hand is visible, HOLD is off, and the fist clutch is released.
- **Live HUD:** status, gesture, FPS, target vs. actual position, tracking error, gripper opening, and cube score.

## Controls

### Hand

| Hand motion | Robot response |
|---|---|
| Palm left / right | Robot Y |
| Palm up / down | Robot Z |
| Hand nearer / farther from camera | Robot X (apparent hand size = depth) |

### Gestures

| Gesture | Action |
|---|---|
| Fist | Clutch: the robot holds its pose so you can reposition your hand |
| Pinch (thumb + index) | Close gripper |
| Open palm | Open gripper |

### Keyboard (camera window)

| Key | Action |
|---|---|
| `SPACE` | Hold / resume control |
| `R` | Reset robot and scene |
| `N` | Respawn cubes only |
| `D` | Depth control on / off |
| `C` | Calibrate neutral depth |
| `H` | Show / hide HUD |
| `Q` / `ESC` | Quit |

## How to Pick and Place

1. Press `SPACE` to hand over control (depth is auto-calibrated on the first press).
2. Move above a cube with an open palm.
3. Lower your hand until the gripper surrounds the cube.
4. **Pinch** to grab, lift, and move to the green container.
5. **Open your palm** to release the cube. Use a **fist** any time to re-center your hand.

## Project Structure

```
Vision_Teleoperation/
├── teleoperation.py   # Entry point: camera loop, hand-to-robot mapping, HUD
├── hand_tracker.py    # MediaPipe tracking, One Euro filter, gesture recognition, drawing
├── robot_sim.py       # PyBullet scene, KUKA iiwa, gripper, cubes, container, IK
├── requirements.txt
├── LICENSE
└── README.md
```

## Requirements

This repository describes a **simulation** workflow; it does not connect to or control a physical KUKA robot. Validate any real-robot adaptation separately with the robot manufacturer’s safety procedures.

- Python 3.9 to 3.12
- A webcam
- Packages: `numpy`, `opencv-python`, `mediapipe`, `pybullet`

## Installation

```bash
git clone https://github.com/muneerahmedkhokhar53-wq/hand-teleoperation-kuka-iiwa.git
cd hand-teleoperation-kuka-iiwa

python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

> **Note:** On Windows, `pybullet` may need the Microsoft C++ Build Tools if no prebuilt wheel is available for your Python version.

## Usage

```bash
python teleoperation.py
```

Two windows open: the **PyBullet simulation** and the **camera view** with the HUD. The system starts in **HOLD**. Press `SPACE` in the camera window to begin.

To test the robot and gripper without the camera:

```bash
python robot_sim.py
```

Keys (click the PyBullet window first): `E` enable, `D` disable, `G` open gripper, `C` close gripper, `R` reset, `Q` quit, `I/K` move X, `J/L` move Y, `U/O` move Z.

## How It Works

1. **Capture:** frames are read from the webcam with OpenCV and mirrored.
2. **Track:** `hand_tracker.py` runs MediaPipe Hands, smooths the 21 landmarks with a One Euro filter, and classifies the gesture with angle-based finger detection and majority voting.
3. **Map:** the palm center is normalized inside a control zone and mapped to a robot target. Hand size sets the depth (X).
4. **Filter:** the target is low-pass filtered, speed-limited, and clamped to the reachable workspace.
5. **Solve:** `robot_sim.py` computes inverse kinematics with the tool pointing straight down and drives the joint motors.
6. **Grip:** debounced pinch / open-palm gestures command the physical two-finger gripper.

## Configuration

Common settings live at the top of each file.

| Setting | File | Purpose |
|---|---|---|
| `ZONE` | `teleoperation.py` | Camera region that maps to the robot range |
| `XY_GAIN`, `DEPTH_GAIN` | `teleoperation.py` | Hand-motion sensitivity |
| `SMOOTH_TAU`, `MAX_SPEED` | `teleoperation.py` | Target smoothing and speed limit |
| `CAMERA_INDEX` | `hand_tracker.py` | Which webcam to use |
| `PINCH_ENTER`, `PINCH_EXIT` | `hand_tracker.py` | Pinch sensitivity (with hysteresis) |
| `CUBE_SIZE`, `CUBE_MASS`, `CUBE_SPOTS` | `robot_sim.py` | Cube properties and layout |
| `BOX_CENTER`, `BOX_INNER_HALF` | `robot_sim.py` | Container position and size |

## Troubleshooting

- **Camera does not open:** close other apps using the webcam, or change `CAMERA_INDEX` in `hand_tracker.py`.
- **Robot does not move:** press `SPACE` to release HOLD, show your hand, and make sure it is not in a fist.
- **Jittery or lost tracking:** improve lighting and keep your hand fully in view against a plain background.
- **Cannot reach a cube:** press `C` to recalibrate depth with your hand at a comfortable distance.
- **`mediapipe` has no attribute `solutions`:** install a MediaPipe version that still ships the `solutions` API (for example `pip install "mediapipe==0.10.14"`).

## Future Improvements

- Orientation control (wrist rotation)
- More objects and task variations
- Voice command integration

## License

Released under the [MIT License](LICENSE).
