"""Step 3d: track the volleyball by its colour instead of with YOLO.

Calibrate:  click the yellow part of the ball   -> sets the colour range
            shift+click the green part          -> adds it to the range
            or drag the sliders in "Controls"
Keys:       s = save settings (step 4 loads them)   q / Esc = quit
"""

import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from pose import CAMERA_INDEX

SETTINGS_FILE = Path(__file__).with_name("ball_color.json")
DEFAULTS = {  # a guess for bright yellow; click the ball to calibrate properly
    "h_min": 20, "h_max": 40, "s_min": 100, "s_max": 255, "v_min": 80, "v_max": 255,
    "exposure": 0,  # 0 = automatic, N = 2^-N seconds (5 = 1/32 s, 7 = 1/128 s)
}
SLIDER_MAX = {"h_min": 179, "h_max": 179, "s_min": 255, "s_max": 255,
              "v_min": 255, "v_max": 255, "exposure": 10}
MIN_AREA = 80          # ignore colour blobs smaller than this many pixels
MOVE_THRESHOLD = 150   # pixels/second of vertical speed before we call it UP or DOWN
CLICK_MARGIN = (8, 60, 60)  # how far either side of a clicked colour to accept (H, S, V)

WINDOW, CONTROLS, MASK_WINDOW = "Ball (colour)", "Controls", "Mask"


def load_settings():
    if SETTINGS_FILE.exists():
        return {**DEFAULTS, **json.loads(SETTINGS_FILE.read_text())}
    return dict(DEFAULTS)


def find_ball_by_color(frame, s, ignore=()):
    """Return ((x, y, radius) or None, mask). s holds the HSV range.
    ignore: ellipses ((cx, cy), (half_w, half_h)) to black out of the mask, e.g. your face."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (s["h_min"], s["s_min"], s["v_min"]),
                       (s["h_max"], s["s_max"], s["v_max"]))
    # Clean up: "open" removes small specks, "close" fills gaps (e.g. the
    # ball's other-coloured panels or seams) so the ball is one solid blob.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    for centre, axes in ignore:  # paint ignored areas black: "not ball"
        cv2.ellipse(mask, centre, axes, 0, 0, 360, 0, -1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
    if not contours:
        return None, mask
    biggest = max(contours, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(biggest)
    return (x, y, r), mask


class BallTrail:
    """Remembers recent ball positions and works out vertical speed/direction."""

    def __init__(self, length=20, forget_after=0.5):
        self.points = deque(maxlen=length)  # (time, x, y)
        self.forget_after = forget_after

    def update(self, now, ball):
        if ball is not None:
            self.points.append((now, ball[0], ball[1]))
        if self.points and now - self.points[-1][0] > self.forget_after:
            self.points.clear()

    def vertical_speed(self):
        """Pixels/second, positive = moving UP. None if not enough data."""
        if len(self.points) < 3:
            return None
        t0, _, y0 = self.points[-3]
        t1, _, y1 = self.points[-1]
        return -(y1 - y0) / max(t1 - t0, 1e-3)  # minus: image y grows downwards

    def direction(self):
        v = self.vertical_speed()
        if v is None:
            return "-"
        return "UP" if v > MOVE_THRESHOLD else "DOWN" if v < -MOVE_THRESHOLD else "still / top"

    def draw(self, frame):
        pts = self.points
        for i in range(1, len(pts)):
            cv2.line(frame, (int(pts[i - 1][1]), int(pts[i - 1][2])),
                     (int(pts[i][1]), int(pts[i][2])), (0, 200, 255), max(1, i // 4))


def apply_exposure(cap, value):
    if value == 0:
        cap.set(cv2.CAP_PROP_EXPOSURE, -5)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, -value)


def main():
    settings = load_settings()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}.")
        return

    cv2.namedWindow(WINDOW)
    cv2.namedWindow(MASK_WINDOW)
    cv2.namedWindow(CONTROLS)
    for key, maximum in SLIDER_MAX.items():
        cv2.createTrackbar(key, CONTROLS, settings[key], maximum, lambda _: None)

    state = {"hsv": None}

    def on_click(event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN or state["hsv"] is None:
            return
        patch = state["hsv"][max(y - 3, 0):y + 4, max(x - 3, 0):x + 4].reshape(-1, 3)
        h, s, v = np.median(patch, axis=0).astype(int)
        lo = [max(h - CLICK_MARGIN[0], 0), max(s - CLICK_MARGIN[1], 0), max(v - CLICK_MARGIN[2], 0)]
        hi = [min(h + CLICK_MARGIN[0], 179), 255, 255]
        if flags & cv2.EVENT_FLAG_SHIFTKEY:  # extend the current range
            lo = [min(lo[i], cv2.getTrackbarPos(k, CONTROLS)) for i, k in enumerate(("h_min", "s_min", "v_min"))]
            hi[0] = max(hi[0], cv2.getTrackbarPos("h_max", CONTROLS))
        for k, val in zip(("h_min", "s_min", "v_min", "h_max", "s_max", "v_max"), lo + hi):
            cv2.setTrackbarPos(k, CONTROLS, int(val))
        print(f"Clicked colour H={h} S={s} V={v} -> range H {lo[0]}-{hi[0]}, S {lo[1]}+, V {lo[2]}+")

    cv2.setMouseCallback(WINDOW, on_click)

    trail = BallTrail()
    seen = deque(maxlen=60)
    exposure_applied = None
    print(__doc__)
    prev = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read a frame from the camera.")
            break
        frame = cv2.flip(frame, 1)
        now = time.perf_counter()

        settings = {k: cv2.getTrackbarPos(k, CONTROLS) for k in SLIDER_MAX}
        if settings["exposure"] != exposure_applied:
            apply_exposure(cap, settings["exposure"])
            exposure_applied = settings["exposure"]

        state["hsv"] = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        ball, mask = find_ball_by_color(frame, settings)
        trail.update(now, ball)
        seen.append(ball is not None)

        if ball is not None:
            x, y, r = ball
            cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 255), 2)
        trail.draw(frame)

        v = trail.vertical_speed()
        fps = 1 / (now - prev)
        prev = now
        lines = [
            f"FPS: {fps:.1f}",
            f"Ball found in {100 * sum(seen) / len(seen):.0f}% of last {len(seen)} frames",
            f"Direction: {trail.direction()}" + (f"  ({v:+.0f} px/s up)" if v is not None else ""),
        ]
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2 if i == 0 else 1)

        cv2.imshow(WINDOW, frame)
        cv2.imshow(MASK_WINDOW, mask)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            SETTINGS_FILE.write_text(json.dumps(settings, indent=2))
            print(f"Saved to {SETTINGS_FILE.name}: {settings}")
        if key in (ord("q"), 27):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    if exposure_applied:  # hand the camera back on automatic exposure
        apply_exposure(cap, 0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
