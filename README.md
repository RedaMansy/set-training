# Volleyball Set Tracker

A live webcam app that watches you set a volleyball to yourself and:

- **counts your sets**
- **measures how high each set goes**
- **scores how consistent you are**, by checking whether each set peaks inside a target zone you choose

It runs in real time on an ordinary laptop webcam, with no special sensors or GPU.

## How it works

Each frame from the webcam goes through three stages:

1. **Body: pose estimation.** A pretrained [Ultralytics YOLO](https://docs.ultralytics.com/) pose model (`yolo26n-pose`) finds 17 body keypoints. Simple geometry on those points (wrists above the head, hands together, elbows bent and up) decides whether you're in setting position.
2. **Ball: colour tracking.** The ball is found by its colour in HSV space with plain OpenCV, not with a neural network. A pretrained "sports ball" detector couldn't see the ball through motion blur, but a blurred yellow ball is still yellow.
3. **Logic: a state machine.** A set is counted when the ball comes down, turns round above your shoulders while you're in setting position, and goes back up. The peak of each set is compared with the target height.

All distances are measured in **shoulder widths**, not pixels, so the app works however far you stand from the camera.

## What you need

- A webcam
- A ball with a **distinctive, bright colour** (developed with a yellow-and-green volleyball). The colour should be different from your skin, clothes and background.
- Python 3.12 or newer (the pinned NumPy needs it; developed on Python 3.13, Windows 11)
- Room to set the ball straight up without hitting the ceiling or the camera loll

## Setup

```bash
git clone https://github.com/RedaMansy/set-training.git
cd set-training
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

Install the dependencies (this includes PyTorch, so it's a large download, about 1–2 GB):

```bash
pip install -r requirements.txt
```

The YOLO model weights (`yolo26n.pt`, `yolo26n-pose.pt`, about 5–8 MB each) download automatically the first time you run a script.

**Optional: NVIDIA GPU.** The default install runs on the CPU, at roughly 20 FPS. If you have an NVIDIA GPU, installing a CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) makes the pose model faster. No code changes are needed.

## Quick start

### 1. Calibrate the ball colour

Lighting changes how the ball's colour looks to the camera, so calibrate in the room and light you'll use.

```bash
python ball_color.py
```

1. Hold the ball up and **click its main colour** in the video window.
2. **Shift+click** any other colours on the ball to add them to the range.
3. Check the **Mask** window: the ball should be a solid white blob and everything else black. If parts of the room (or your skin) show up white, raise `s_min` in the **Controls** window until they disappear.
4. Press **`s`** to save. This writes `ball_color.json`, which the set counter loads at startup.

### 2. Run the set counter

```bash
python set_counter.py
```

Stand back until the camera can see you from at least the waist up, with plenty of space above your head. If a red **"Step back"** message appears, the target zone doesn't fit in view; move further back or tilt the camera up. Then start setting.

## Controls

| Key | Action |
|---|---|
| `+` / `-` | Raise / lower the target set height |
| `r` | Reset the score (keeps your target height) |
| `d` | Show / hide debug info (FPS, detection state, why a set was or wasn't counted) |
| `m` | Show / hide the colour mask (what the app thinks is "ball") |
| `f` | Fullscreen |
| `q` or `Esc` | Quit |

## What's on screen

- **Skeleton:** your body as the pose model sees it.
- **Ball and trail:** the tracked ball and its recent path.
- **Target zone:** the box where an on-target set should peak. It turns **green** (in the zone) or **red** (missed) after each set.
- **Score panel:**
  - **SETS:** the total count
  - **SET HEIGHT:** how high the last set went, in shoulder widths above your hands, next to the current target
  - **IN ZONE:** how many sets peaked inside the zone, with a percentage

## Tuning

The main settings are constants at the top of [`set_counter.py`](set_counter.py), each with a comment:

| Setting | Default | What it does |
|---|---|---|
| `TARGET_SET_HEIGHT` | `2.5` | Target height above your hands, in shoulder widths (about 1 m). Also adjustable live with `+` / `-`. |
| `TARGET_HALF_HEIGHT` | `0.4` | How far above or below the target still counts as in the zone |
| `TARGET_HALF_WIDTH` | `0.75` | How far the ball can drift sideways and still count |
| `MIN_FALL` | `0.4` | How far the ball must fall before a contact counts, so a held, wobbling ball isn't counted as sets. Very low sets (under about 0.7) won't count. |
| `POSTURE_WINDOW` | `0.6` | Seconds you must have been in setting position before contact |
| `COOLDOWN` | `0.35` | Minimum seconds between two sets |

The setting-position rules (how high the wrists must be, the elbow angles) are at the top of [`set_posture.py`](set_posture.py).

## Troubleshooting

| Problem | Things to try |
|---|---|
| Ball not detected, or lost in flight | Recalibrate with `ball_color.py` in the current light. Use brighter light in front of you, not behind you. |
| Your face, hands or something else counted as the ball | Press `m` to see the mask. Recalibrate and raise `s_min`. Warm lamp light can make skin look like a yellow ball; daylight or white light helps. |
| Sets counted while holding the ball | Usually the ball's detected position jumping around because of a loose calibration. Recalibrate. |
| Sets not counted | Press `d` and read **Last:** on the third line: it tells you which rule failed (e.g. `not in set position`, `turned outside the contact zone`). |
| Wrong camera opens | Change `CAMERA_INDEX` in [`pose.py`](pose.py) (try 1, 2, …). |
| Low frame rate | Close other apps using the camera or CPU, or use an NVIDIA GPU (see Setup). |

## Project files

`set_counter.py` is the app. The other files are the steps it was built from, and each one runs on its own:

| File | What it does |
|---|---|
| [`webcam.py`](webcam.py) | Live webcam feed I was playing around with to make sure openCV ran - u can ignore for the most part |
| [`detect.py`](detect.py) | Live object detection with pretrained YOLO |
| [`pose.py`](pose.py) | Live pose estimation (skeleton) |
| [`set_posture.py`](set_posture.py) | Detects setting position, with a live checklist of each rule |
| [`ball.py`](ball.py) | Ball tracking with YOLO's "sports ball" class - this was the initial implementation that didn't work very well with a ball in motion |
| [`ball_color.py`](ball_color.py) | Ball tracking by colour, and the calibration tool |
| [`set_counter.py`](set_counter.py) | The set counter: pose, ball and set logic combined |
| [`hud.py`](hud.py) | Drawing for the set counter's display |

## Limitations

- Built for **setting to yourself straight up**, facing the camera. Other skills (bumps, spikes) and side-on views aren't supported.
- Colour tracking depends on the lighting and on nothing else in view sharing the ball's colour.
- Heights are in shoulder widths, not centimetres, and they're approximate: at about 20 FPS the camera may miss the exact top of a set by a little.
