"""Step 3b: detect when you're in setting position (hands above your forehead,
elbows bent and up). Shows each check live so you can tune the thresholds.
Press 'q' or Esc to quit."""

import time

import cv2
import numpy as np
from ultralytics import YOLO

from pose import CAMERA_INDEX, KP_CONF, MODEL, draw_person

WINDOW = "Set posture"

# Keypoint indices (COCO order, see KEYPOINT_NAMES in pose.py)
NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 0, 5, 6, 7, 8, 9, 10
NEEDED = [NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST]

# --- Thresholds to tune --------------------------------------------------------
# Distances are measured in "shoulder widths" rather than pixels, so the rules
# work whether you stand close to the camera or far away.
WRIST_ABOVE_NOSE = 0.3   # both wrists at least this far above the nose
MAX_WRIST_GAP = 1.5      # wrists no further apart than this (hands round the ball)
ELBOW_ANGLE_MIN = 40     # degrees at the elbow (shoulder-elbow-wrist).
ELBOW_ANGLE_MAX = 165    #   180 = arm dead straight, so this excludes straight arms
HOLD_FRAMES = 3          # posture must hold this many frames in a row (stops flicker)
# -------------------------------------------------------------------------------


def elbow_angle(shoulder, elbow, wrist):
    """Angle at the elbow in degrees, from three (x, y) points."""
    a = shoulder - elbow
    b = wrist - elbow
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def check_set_posture(xy, conf):
    """Return (in_position, checks) where checks is a list of (name, passed, detail)."""
    missing = [i for i in NEEDED if conf[i] < KP_CONF]
    if missing:
        return False, [("arms + head visible", False, f"{len(missing)} points missing")]

    shoulder_w = max(np.linalg.norm(xy[L_SHOULDER] - xy[R_SHOULDER]), 1.0)
    y = xy[:, 1]  # remember: smaller y = higher up in the image

    # 1. Both wrists above the nose
    rise = (y[NOSE] - max(y[L_WRIST], y[R_WRIST])) / shoulder_w
    wrists_up = rise > WRIST_ABOVE_NOSE

    # 2. Hands close together (shaped round the ball)
    gap = np.linalg.norm(xy[L_WRIST] - xy[R_WRIST]) / shoulder_w
    hands_together = gap < MAX_WRIST_GAP

    # 3. Elbows higher than shoulders
    elbows_up = y[L_ELBOW] < y[L_SHOULDER] and y[R_ELBOW] < y[R_SHOULDER]

    # 4. Elbows bent, not locked straight
    la = elbow_angle(xy[L_SHOULDER], xy[L_ELBOW], xy[L_WRIST])
    ra = elbow_angle(xy[R_SHOULDER], xy[R_ELBOW], xy[R_WRIST])
    bent = all(ELBOW_ANGLE_MIN <= a <= ELBOW_ANGLE_MAX for a in (la, ra))

    checks = [
        ("wrists above head", wrists_up, f"{rise:+.2f} (need > {WRIST_ABOVE_NOSE})"),
        ("hands together", hands_together, f"{gap:.2f} (need < {MAX_WRIST_GAP})"),
        ("elbows above shoulders", elbows_up, ""),
        ("elbows bent", bent, f"L {la:.0f} / R {ra:.0f} deg"),
    ]
    return all(passed for _, passed, _ in checks), checks


def main_person(result):
    """Index of the person with the biggest box (the one closest to the camera)."""
    if result.keypoints is None or result.keypoints.conf is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(np.argmax(areas))


def draw_panel(frame, checks, in_position):
    for row, (name, passed, detail) in enumerate(checks):
        colour = (0, 200, 0) if passed else (0, 0, 255)
        text = f"[{'x' if passed else ' '}] {name}  {detail}"
        cv2.putText(frame, text, (10, 55 + row * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

    if in_position:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 160, 0), -1)
        cv2.putText(frame, "SET POSITION", (w // 2 - 110, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)


def main():
    model = YOLO(MODEL)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}.")
        return

    print("Running. Press 'q' or Esc in the video window to quit.")
    streak = 0  # how many frames in a row the posture has passed
    prev = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read a frame from the camera.")
            break
        frame = cv2.flip(frame, 1)

        result = model(frame, verbose=False)[0]
        idx = main_person(result)

        if idx is None:
            checks, passed = [("person in frame", False, "")], False
        else:
            xy = result.keypoints.xy[idx].cpu().numpy()
            conf = result.keypoints.conf[idx].cpu().numpy()
            draw_person(frame, xy, conf)
            passed, checks = check_set_posture(xy, conf)

        streak = streak + 1 if passed else 0
        draw_panel(frame, checks, in_position=streak >= HOLD_FRAMES)

        now = time.perf_counter()
        fps = 1 / (now - prev)
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
