"""Step 4: count your volleyball sets live.

A rep is counted when the ball comes DOWN, turns round above your shoulders
(near your hands), and goes back UP, while you were in setting position.
Keys: r = reset count, m = show/hide colour mask, q / Esc = quit.
Calibrate the ball colour first with ball_color.py (press 's' there to save).
"""

import time

import cv2
from ultralytics import YOLO

from ball_color import (MOVE_THRESHOLD, SETTINGS_FILE, BallTrail, apply_exposure,
                        find_ball_by_color, load_settings)
from pose import CAMERA_INDEX, KP_CONF, MODEL, draw_person
from set_posture import L_SHOULDER, NOSE, R_SHOULDER, check_set_posture, main_person

WINDOW = "Set counter"

# --- Rep rules to tune -----------------------------------------------------------
POSTURE_WINDOW = 0.6   # seconds: you must have been in set position this recently.
                       #   (at contact the ball often hides your hands, so the pose
                       #    check can fail on that exact frame)
ZONE_HALF_WIDTH = 1.5  # contact zone: this many shoulder widths either side of centre
ZONE_HEIGHT = 3.0      #   and from shoulder height up to this many shoulder widths
BODY_MEMORY = 0.5      # seconds to keep using your last known position if pose drops out
COOLDOWN = 0.35        # seconds after a rep before another can count (no double counts)
MIN_FALL = 0.4         # shoulder widths the ball must fall before contact. Stops a held,
                       #   wobbling ball (noisy detections) being counted as sets
TARGET_RADIUS = 40     # pixels: a set whose peak lands within this of the target is consistent
FACE_HALF_WIDTH = 0.3  # face area ignored by the ball tracker (skin can look yellow
FACE_HALF_HEIGHT = 0.4 #   under lamps): an ellipse round your nose, in shoulder widths
# ---------------------------------------------------------------------------------


def face_area(xy, conf, shoulder_width):
    """Ellipse ((cx, cy), (half_w, half_h)) covering the face, or None if the nose isn't seen."""
    if conf[NOSE] < KP_CONF:
        return None
    cx, cy = xy[NOSE]
    return ((int(cx), int(cy)),
            (int(FACE_HALF_WIDTH * shoulder_width), int(FACE_HALF_HEIGHT * shoulder_width)))


class RepCounter:
    """State machine: WAITING -> FALLING (ball coming down) -> rep if it turns UP in the zone."""

    def __init__(self):
        self.count = 0
        self.state = "WAITING"
        self.lowest = None          # (time, x, y) lowest point while falling
        self.fall_start = None      # (time, x, y) where the current fall began
        self.last_posture_ok = -1e9
        self.last_rep = -1e9
        self.body = None            # (time, shoulder_mid_x, shoulder_y, shoulder_width)
        self.last_reason = ""
        self.highest_point = None
        self.last_set_height = None
        self.contact_point = None
        self.target = None

    def update_body(self, now, xy, conf, posture_ok, kp_conf):
        if posture_ok:
            self.last_posture_ok = now
        if conf[L_SHOULDER] > kp_conf and conf[R_SHOULDER] > kp_conf:
            ls, rs = xy[L_SHOULDER], xy[R_SHOULDER]
            width = max(abs(ls[0] - rs[0]), 1.0)
            self.body = (now, (ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2, width)

    def contact_zone(self, now):
        """(x1, y1, x2, y2) box above your shoulders where contact should happen."""
        if self.body is None or now - self.body[0] > BODY_MEMORY:
            return None
        _, cx, sy, w = self.body
        return (cx - ZONE_HALF_WIDTH * w, sy - ZONE_HEIGHT * w, cx + ZONE_HALF_WIDTH * w, sy)

    def update_ball(self, now, ball_xy, speed):
        """speed: px/s, positive = up (None if unknown). Returns True when a rep is counted."""
        if speed is None:
            return False

        if self.state == "WAITING":
            if self.highest_point is not None and ball_xy is not None and ball_xy[1] < self.highest_point[2]:
                self.highest_point = (now, *ball_xy)
            if speed < -MOVE_THRESHOLD and ball_xy is not None:
                self.state, self.lowest = "FALLING", (now, *ball_xy)
                self.fall_start = self.lowest
                if self.highest_point is not None:
                    height = (self.contact_point[2] - self.highest_point[2]) / self.body[3]
                    self.last_set_height = height
                    if self.target is None:
                        self.target = self.highest_point
                        print("target set: ", self.target)   
                    print(self.last_set_height)
                    self.highest_point = None
            return False

        # FALLING: remember the lowest point (largest y), wait for it to turn up
        if ball_xy is not None and ball_xy[1] > self.lowest[2]:
            self.lowest = (now, *ball_xy)
        if speed <= MOVE_THRESHOLD:
            return False

        # The ball has turned round and is going up again. Was it a set?
        self.state = "WAITING"
        t, x, y = self.lowest
        zone = self.contact_zone(now)
        if zone is None:
            self.last_reason = "no body seen"
        elif (y - self.fall_start[2]) / self.body[3] < MIN_FALL:
            self.last_reason = "moved too little (wobble?)"
        elif not (zone[0] <= x <= zone[2] and zone[1] <= y <= zone[3]):
            self.last_reason = "turned outside the contact zone (bounce?)"
        elif now - self.last_posture_ok > POSTURE_WINDOW:
            self.last_reason = "not in set position"
        elif now - self.last_rep < COOLDOWN:
            self.last_reason = "too soon after last rep"
        else:
            self.count += 1
            self.last_rep = now
            self.last_reason = "SET!"
            self.contact_point = t, x, y
            self.highest_point = self.contact_point
            return True
        return False


def main():
    if not SETTINGS_FILE.exists():
        print("Warning: no ball_color.json - using default yellow. "
              "Run ball_color.py and press 's' to save a proper calibration.")
    colour = load_settings()

    model = YOLO(MODEL)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}.")
        return
    if colour["exposure"]:
        apply_exposure(cap, colour["exposure"])

    counter = RepCounter()
    trail = BallTrail()
    flash_until = 0.0
    face = None        # (time, ellipse) of the last place we saw your face
    show_mask = False
    print(__doc__)
    prev = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read a frame from the camera.")
            break
        frame = cv2.flip(frame, 1)
        now = time.perf_counter()

        # 1. Body: pose model -> posture check
        result = model(frame, verbose=False)[0]
        idx = main_person(result)
        posture_ok = False
        if idx is not None:
            xy = result.keypoints.xy[idx].cpu().numpy()
            conf = result.keypoints.conf[idx].cpu().numpy()
            draw_person(frame, xy, conf)
            posture_ok, _ = check_set_posture(xy, conf)
            counter.update_body(now, xy, conf, posture_ok, KP_CONF)
            if counter.body is not None:
                area = face_area(xy, conf, counter.body[3])
                if area is not None:
                    face = (now, area)
        if face is not None and now - face[0] > BODY_MEMORY:
            face = None  # not seen for a while: stop ignoring that spot

        # 2. Ball: colour tracking (ignoring your face) -> position and vertical speed
        ball, mask = find_ball_by_color(frame, colour, ignore=[face[1]] if face else [])
        trail.update(now, ball)
        speed = trail.vertical_speed()

        # 3. Combine: did a set just happen?
        if counter.update_ball(now, ball[:2] if ball else None, speed):
            flash_until = now + 0.4
            print(f"Rep {counter.count}")

        # --- Drawing ---
        if face:
            cv2.ellipse(frame, face[1][0], face[1][1], 0, 0, 360, (160, 160, 160), 1)
        zone = counter.contact_zone(now)
        if zone:
            recent_posture = now - counter.last_posture_ok <= POSTURE_WINDOW
            cv2.rectangle(frame, (int(zone[0]), int(zone[1])), (int(zone[2]), int(zone[3])),
                          (0, 200, 0) if recent_posture else (120, 120, 120), 1)
        if counter.target is not None:
            _, tx, ty = counter.target
            cv2.circle(frame, (int(tx), int(ty)), TARGET_RADIUS, (255, 255, 255), 2)
        if ball:
            cv2.circle(frame, (int(ball[0]), int(ball[1])), int(ball[2]), (0, 255, 255), 2)
        trail.draw(frame)

        h, w = frame.shape[:2]
        if now < flash_until:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 0), 12)
        cv2.putText(frame, f"SETS: {counter.count}", (w - 230, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
        if counter.last_set_height is not None:
            cv2.putText(frame, f"HEIGHT: {counter.last_set_height:.1f}", (w - 230, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        fps = 1 / (now - prev)
        prev = now
        lines = [
            f"FPS: {fps:.1f}",
            f"Set position: {'YES' if posture_ok else 'no'}   Ball: {trail.direction()}",
            f"State: {counter.state}   Last: {counter.last_reason}",
        ]
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 255), 2 if i == 0 else 1)

        cv2.imshow(WINDOW, frame)
        if show_mask:
            cv2.imshow("Mask", mask)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            counter.count = 0
        if key == ord("m"):  # toggle the colour mask window, to check what counts as "ball"
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow("Mask")
        if key in (ord("q"), 27):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    if colour["exposure"]:
        apply_exposure(cap, 0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
