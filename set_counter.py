"""Step 4: count your volleyball sets live.

A rep is counted when the ball comes DOWN, turns round above your shoulders
(near your hands), and goes back UP, while you were in setting position.
Keys: r = reset score, + / - = raise / lower target height, d = debug info, f = fullscreen,
      m = show/hide colour mask, q / Esc = quit.
Calibrate the ball colour first with ball_color.py (press 's' there to save).
"""

import time

import cv2
from ultralytics import YOLO

import hud
from ball_color import (MOVE_THRESHOLD, SETTINGS_FILE, BallTrail, apply_exposure,
                        find_ball_by_color, load_settings)
from pose import CAMERA_INDEX, KP_CONF, MODEL
from set_posture import L_SHOULDER, NOSE, R_SHOULDER, check_set_posture, main_person

WINDOW = "Set counter"
DISPLAY_SCALE = 2      # show the 640x480 camera at 2x (1280x960) so text and lines look sharp

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
TARGET_SET_HEIGHT = 2.5   # target: how high above your hands the ball should go, in
                          #   shoulder widths (2.5 ~ 1 m: just under the ceiling at home,
                          #   from real sets measuring ~2.6). Change it live with + / -
TARGET_STEP = 0.25        #   ...by this much per key press
TARGET_MIN, TARGET_MAX = 0.5, 4.0
TARGET_HALF_HEIGHT = 0.4  # a set is "in the zone" if its height is within this of the
TARGET_HALF_WIDTH = 0.75  #   target, and its peak is within this sideways of your centre
CONTACT_ABOVE_SHOULDERS = 1.0  # where to draw the zone before your first set: a typical
                               #   contact height above the shoulders (then the real one)
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
        self.target_height = TARGET_SET_HEIGHT  # shoulder widths above your hands
        self.contact_dy = CONTACT_ABOVE_SHOULDERS  # last contact's height above shoulders
        self.judged = 0           # sets whose peak has been checked against the target
        self.in_zone_count = 0      #   ...and how many of those peaked inside the zone
        self.last_in_zone = None    # result for the most recent set (None = none yet)
        self.last_judged_at = -1e9  # when that result came in (for the on-screen pop-up)

    def reset_session(self):
        """Start again: clear the score (the target height you chose is kept)."""
        self.count = self.judged = self.in_zone_count = 0
        self.last_in_zone = self.last_set_height = None
        self.last_judged_at = -1e9

    def change_target(self, steps):
        """Raise (+) or lower (-) the target height by TARGET_STEP per step."""
        new = self.target_height + steps * TARGET_STEP
        self.target_height = min(max(new, TARGET_MIN), TARGET_MAX)

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

    def body_offset(self, point):
        """Where a (time, x, y) point is relative to your shoulders, in shoulder widths:
        dx = sideways (+ = right on screen), dy = height above your shoulders.
        Works at any distance from the camera."""
        _, cx, sy, w = self.body
        _, x, y = point
        return float((x - cx) / w), float((sy - y) / w)

    def target_zone(self, now):
        """(x1, y1, x2, y2) pixel box where an on-target set peaks: target height above
        your hands, centred over you. Placed from where you are NOW."""
        if self.body is None or now - self.body[0] > BODY_MEMORY:
            return None
        _, cx, sy, w = self.body
        y = sy - (self.contact_dy + self.target_height) * w  # shoulder widths -> pixels
        return (cx - TARGET_HALF_WIDTH * w, y - TARGET_HALF_HEIGHT * w,
                cx + TARGET_HALF_WIDTH * w, y + TARGET_HALF_HEIGHT * w)

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
                    dx, _ = self.body_offset(self.highest_point)
                    in_zone = bool(abs(height - self.target_height) <= TARGET_HALF_HEIGHT
                                   and abs(dx) <= TARGET_HALF_WIDTH)
                    self.judged += 1
                    if in_zone:
                        self.in_zone_count += 1
                    self.last_in_zone = in_zone
                    self.last_judged_at = now
                    print(f"height {height:.2f}  in zone: {in_zone}  "
                          f"({self.in_zone_count}/{self.judged})")
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
            self.contact_dy = self.body_offset(self.contact_point)[1]  # zone follows your hands
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
    show_mask = debug = fullscreen = False
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # resizable
    cv2.resizeWindow(WINDOW, 640 * DISPLAY_SCALE, 480 * DISPLAY_SCALE)
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
        person = None
        if idx is not None:
            xy = result.keypoints.xy[idx].cpu().numpy()
            conf = result.keypoints.conf[idx].cpu().numpy()
            person = (xy, conf)
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

        # --- Drawing: on an upscaled copy, after all the processing above ---
        s = DISPLAY_SCALE
        display = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        if person:
            hud.draw_body(display, *person, s)
        hud.draw_target(display, counter.target_zone(now), counter.last_in_zone, s,
                        now, counter.last_judged_at)
        hud.draw_ball(display, ball, trail.points, s)

        fps = 1 / (now - prev)
        prev = now
        if debug:
            hud.draw_debug(display, s, [
                f"FPS: {fps:.1f}",
                f"Set position: {'YES' if posture_ok else 'no'}   Ball: {trail.direction()}",
                f"State: {counter.state}   Last: {counter.last_reason}",
                "keys: r reset  +/- target  d debug  f fullscreen  m mask  q quit",
            ], face[1] if face else None, counter.contact_zone(now),
                now - counter.last_posture_ok <= POSTURE_WINDOW)
        else:
            hud.draw_title(display)
        hud.draw_stats(display, counter, flash=now < flash_until)  # last, so it's on top

        cv2.imshow(WINDOW, display)
        if show_mask:
            cv2.imshow("Mask", mask)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            counter.reset_session()
        if key in (ord("+"), ord("=")):  # '=' is the same key as '+' without shift
            counter.change_target(+1)
        if key in (ord("-"), ord("_")):
            counter.change_target(-1)
        if key == ord("d"):
            debug = not debug
        if key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
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
