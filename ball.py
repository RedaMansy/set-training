"""Step 3c: track the volleyball live with the pretrained YOLO "sports ball" class.
Shows the ball, its recent path, whether it's going up or down, and how often
it's being detected. Press 'q' or Esc to quit."""

import time
from collections import deque

import cv2
from ultralytics import YOLO

from pose import CAMERA_INDEX

WINDOW = "Ball tracking"
MODEL = "yolo26n.pt"
SPORTS_BALL = 32       # COCO class id for "sports ball"
BALL_CONF = 0.25       # lower than usual: missing the ball is worse than the odd false hit
TRAIL_LENGTH = 20      # how many past positions to draw
MOVE_THRESHOLD = 150   # pixels/second of vertical speed before we call it UP or DOWN
RATE_WINDOW = 60       # detection rate is measured over this many recent frames


def find_ball(model, frame):
    """Return (x, y, radius, confidence) of the most confident ball, or None."""
    result = model(frame, classes=[SPORTS_BALL], conf=BALL_CONF, verbose=False)[0]
    if len(result.boxes) == 0:
        return None
    best = result.boxes[int(result.boxes.conf.argmax())]
    x1, y1, x2, y2 = best.xyxy[0].tolist()
    return (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) / 2, float(best.conf)


def main():
    model = YOLO(MODEL)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}.")
        return

    trail = deque(maxlen=TRAIL_LENGTH)   # (time, x, y) of recent detections
    seen = deque(maxlen=RATE_WINDOW)     # True/False: was the ball found in each frame
    print("Running. Press 'q' or Esc in the video window to quit.")
    prev = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read a frame from the camera.")
            break
        frame = cv2.flip(frame, 1)
        now = time.perf_counter()

        ball = find_ball(model, frame)
        seen.append(ball is not None)

        if ball is not None:
            x, y, r, conf = ball
            trail.append((now, x, y))
            cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 255), 2)
            cv2.putText(frame, f"ball {conf:.2f}", (int(x - r), int(y - r - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Forget the path if the ball has been missing for a while
        if trail and now - trail[-1][0] > 0.5:
            trail.clear()

        # Path: older points fade out
        for i in range(1, len(trail)):
            p1 = (int(trail[i - 1][1]), int(trail[i - 1][2]))
            p2 = (int(trail[i][1]), int(trail[i][2]))
            cv2.line(frame, p1, p2, (0, 200, 255), max(1, i // 4))

        # Vertical speed from the last few detections. Remember y grows downwards,
        # so negative speed = moving up.
        direction, speed = "-", 0.0
        if len(trail) >= 3:
            t0, _, y0 = trail[-3]
            t1, _, y1 = trail[-1]
            speed = (y1 - y0) / max(t1 - t0, 1e-3)
            if speed < -MOVE_THRESHOLD:
                direction = "UP"
            elif speed > MOVE_THRESHOLD:
                direction = "DOWN"
            else:
                direction = "still / top"

        rate = 100 * sum(seen) / len(seen)
        fps = 1 / (now - prev)
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Ball found in {rate:.0f}% of last {len(seen)} frames", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
        cv2.putText(frame, f"Direction: {direction}  ({-speed:+.0f} px/s up)", (10, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)

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
