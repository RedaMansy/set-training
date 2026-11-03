"""Step 3a: live pose estimation on the webcam.
Draws your skeleton (17 keypoints). Press 'q' or Esc to quit."""

import time

import cv2
from ultralytics import YOLO

CAMERA_INDEX = 0
MODEL = "yolo26n-pose.pt"  # pose version of the nano model (~7 MB, downloads on first run)
KP_CONF = 0.5              # hide keypoints the model is less than 50% sure about
WINDOW = "Pose"

# The 17 COCO keypoints, in the order the model returns them
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Pairs of keypoint indices to join with lines to make a skeleton
SKELETON = [
    (5, 7), (7, 9),        # left arm: shoulder-elbow-wrist
    (6, 8), (8, 10),       # right arm
    (5, 6), (11, 12),      # shoulders, hips
    (5, 11), (6, 12),      # torso sides
    (11, 13), (13, 15),    # left leg
    (12, 14), (14, 16),    # right leg
]

# Keypoints that matter for setting: drawn bigger and labelled
HIGHLIGHT = {0: "nose", 7: "L elbow", 8: "R elbow", 9: "L wrist", 10: "R wrist"}


def draw_person(frame, xy, conf):
    """xy: 17x2 array of pixel positions, conf: 17 confidences (0-1)."""
    visible = conf > KP_CONF

    for a, b in SKELETON:
        if visible[a] and visible[b]:
            cv2.line(frame, tuple(map(int, xy[a])), tuple(map(int, xy[b])), (255, 200, 0), 2)

    for i, (x, y) in enumerate(xy):
        if not visible[i]:
            continue
        pt = (int(x), int(y))
        if i in HIGHLIGHT:
            cv2.circle(frame, pt, 7, (0, 0, 255), -1)
            cv2.putText(frame, HIGHLIGHT[i], (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.circle(frame, pt, 4, (0, 255, 0), -1)


def main():
    model = YOLO(MODEL)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}.")
        return

    print("Running. Press 'q' or Esc in the video window to quit.")
    prev = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read a frame from the camera.")
            break
        frame = cv2.flip(frame, 1)

        result = model(frame, verbose=False)[0]

        # result.keypoints holds one entry per detected person
        if result.keypoints is not None and result.keypoints.conf is not None:
            all_xy = result.keypoints.xy.cpu().numpy()      # shape: (people, 17, 2)
            all_conf = result.keypoints.conf.cpu().numpy()  # shape: (people, 17)
            for xy, conf in zip(all_xy, all_conf):
                draw_person(frame, xy, conf)
            people = len(all_xy)
        else:
            people = 0

        now = time.perf_counter()
        fps = 1 / (now - prev)
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}  People: {people}", (10, 25),
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
