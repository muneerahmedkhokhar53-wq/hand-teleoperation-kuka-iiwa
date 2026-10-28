# Manual Teleoperation Test Plan

This checklist validates the interactive parts of the **simulated** hand-teleoperation project that automated tests cannot fully cover. It does not validate or authorize control of physical robot hardware.

## 1. Test environment

- [ ] Use 64-bit Python 3.11 and the repository's `.venv`.
- [ ] Install the pinned dependencies and confirm `python -m pip check` succeeds.
- [ ] Run `python -m unittest discover -s tests -v`; record any failures before testing the GUI.
- [ ] Close other apps that may be using the webcam.
- [ ] Place the camera at a stable position with the hand fully visible and reasonably even lighting.

On Windows, invoke the project interpreter explicitly if the global Python version differs:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 2. Simulator-only smoke test

Run `python robot_sim.py`.

- [ ] PyBullet opens and displays the robot and scene without an exception.
- [ ] Click the simulator window, then verify the documented enable/disable and target-movement keys.
- [ ] Verify the gripper open/close controls and reset key.
- [ ] Quit with the documented key and confirm the process exits cleanly.

**Pass criteria:** the GUI starts, the documented test controls respond, reset returns the scene to its initial state, and quitting does not leave the process running.

## 3. Webcam teleoperation test

Run `python teleoperation.py`.

- [ ] Both the PyBullet scene and camera/HUD window open.
- [ ] Confirm the robot starts in **HOLD** and does not begin uncontrolled motion.
- [ ] Press `SPACE` in the camera window to resume control.
- [ ] Show one hand clearly; check that the hand landmarks/HUD respond.
- [ ] Move the palm slowly in small increments and confirm the target follows smoothly without abrupt jumps.
- [ ] Make a fist to engage the clutch; move the hand while clutched and verify the robot holds its pose.
- [ ] Release the fist and confirm tracking resumes.
- [ ] Pinch thumb and index finger to close the gripper; open the palm to release/open it.
- [ ] Test the depth control and neutral-depth calibration only while a hand is visible.
- [ ] Press `SPACE` to hold again and confirm motion stops; test reset only when ready to return to the initial scene.
- [ ] Quit with `Q` or `ESC` and confirm both windows close.

**Pass criteria:** the app starts safely in HOLD, visible-hand tracking responds, clutching prevents hand repositioning from moving the robot, the gripper gestures are repeatable, and the app can be stopped cleanly. Record intermittent or inconsistent gesture results instead of marking them as a pass.

## 4. Pick-and-place task trial

Only attempt this after the basic checks pass. Start with one cube and slow, deliberate movements.

1. Resume control and approach the cube while keeping the target within the robot's reachable workspace.
2. Use pinch to close the gripper around the cube.
3. Move the hand gradually toward a clear drop location.
4. Open the palm to release the cube.
5. Repeat the task three times and record each result.

| Trial | Grasped cube | Transported cube | Released at target | Notes |
|---|---|---|---|---|
| 1 | [ ] | [ ] | [ ] | |
| 2 | [ ] | [ ] | [ ] | |
| 3 | [ ] | [ ] | [ ] | |

A trial passes only when all three boxes are checked. If the cube slips, the gripper closes without grasping, or the arm cannot reach the target, record the failure and conditions. Do not describe the task as reliable based on a single successful attempt.

## 5. Failure report template

When a test fails, record:

- Operating system and Python version:
- Command used:
- Test section and step:
- Expected behavior:
- Actual behavior:
- Error text or screenshot:
- Whether the same issue repeats:

## Scope notes

- GitHub Actions and unit tests validate selected logic and syntax; they do not prove camera access, GUI rendering, tracking quality, or successful pick-and-place.
- Hand size is an approximate depth cue, not metric depth sensing.
- This project is for simulation only. It is not safety-rated and must not be connected to physical robot hardware based on this checklist.
