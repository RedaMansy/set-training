"""Drawing for set_counter.py: the clean presentation view and the debug overlay.

Everything is drawn on an upscaled copy of the camera frame (crisper text), so
positions coming in are in camera pixels and `s` scales them to display pixels.
"""

import os
from functools import lru_cache
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pose import KP_CONF, SKELETON

AA = cv2.LINE_AA  # anti-aliased (smooth) lines

# Text is drawn with Pillow and real TrueType fonts: OpenCV's built-in (Hershey)
# fonts have no proper weights or spacing. First font file found wins.
_WIN_FONTS = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
_MPL_FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
FONT_FILES = {  # weight -> candidates
    "semibold": [_WIN_FONTS / "seguisb.ttf", _WIN_FONTS / "arial.ttf", _MPL_FONTS / "DejaVuSans.ttf"],
    "bold": [_WIN_FONTS / "segoeuib.ttf", _WIN_FONTS / "arialbd.ttf", _MPL_FONTS / "DejaVuSans-Bold.ttf"],
    "black": [_WIN_FONTS / "seguibl.ttf", _WIN_FONTS / "arialbd.ttf", _MPL_FONTS / "DejaVuSans-Bold.ttf"],
}
PX_PER_SCALE = 32  # font size in px for "scale 1" (keeps the old OpenCV-style sizes)

# Colours are BGR
WHITE, GREY, DARK = (255, 255, 255), (175, 175, 175), (60, 60, 60)
GREEN, RED = (90, 215, 90), (80, 80, 235)
YELLOW, ORANGE = (0, 230, 255), (0, 150, 255)
BODY = (255, 205, 130)       # light blue skeleton
VERDICT_SECONDS = 1.0        # how long "IN ZONE!" / "MISSED" stays up
ZONE_COLOURS = {None: WHITE, True: GREEN, False: RED}


def _p(pt, s):
    return int(pt[0] * s), int(pt[1] * s)


def shade(img, x1, y1, x2, y2, colour=(0, 0, 0), alpha=0.55):
    """Semi-transparent rectangle (a see-through panel behind text)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(int(x1), 0), max(int(y1), 0), min(int(x2), w), min(int(y2), h)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    img[y1:y2, x1:x2] = cv2.addWeighted(np.full_like(roi, colour), alpha, roi, 1 - alpha, 0)


def _weight(thickness):
    return "semibold" if thickness <= 1 else "bold" if thickness == 2 else "black"


@lru_cache(maxsize=None)
def _font(weight, size):
    for path in FONT_FILES[weight]:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


@lru_cache(maxsize=2048)
def _render(string, weight, size, colour):
    """Draw text once into a small RGBA image (cached: most labels never change).
    Returns (BGR pixels, alpha 0-1, x/y offset of the image from the text's baseline start)."""
    font = _font(weight, size)
    x0, y0, x1, y1 = font.getbbox(string, anchor="ls")
    patch = Image.new("RGBA", (max(x1 - x0, 1), max(y1 - y0, 1)), (0, 0, 0, 0))
    b, g, r = colour
    ImageDraw.Draw(patch).text((-x0, -y0), string, font=font, fill=(r, g, b, 255), anchor="ls")
    rgba = np.asarray(patch, dtype=np.float32)
    return rgba[..., 2::-1], rgba[..., 3:] / 255.0, x0, y0


def text(img, string, org, scale, colour=WHITE, thickness=1):
    """Draw text with its baseline starting at org (like cv2.putText). thickness picks
    the weight: 1 = semibold, 2 = bold, 3+ = black."""
    bgr, alpha, dx, dy = _render(string, _weight(thickness), round(scale * PX_PER_SCALE), tuple(colour))
    x, y = int(org[0]) + dx, int(org[1]) + dy
    h, w = alpha.shape[:2]
    # clip to the image so text near an edge doesn't crash
    ix0, iy0, ix1, iy1 = max(x, 0), max(y, 0), min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if ix1 <= ix0 or iy1 <= iy0:
        return
    px0, py0 = ix0 - x, iy0 - y
    a = alpha[py0:py0 + iy1 - iy0, px0:px0 + ix1 - ix0]
    roi = img[iy0:iy1, ix0:ix1].astype(np.float32)
    src = bgr[py0:py0 + iy1 - iy0, px0:px0 + ix1 - ix0]
    img[iy0:iy1, ix0:ix1] = (src * a + roi * (1 - a)).astype(np.uint8)


def text_width(string, scale, thickness=1):
    return int(_font(_weight(thickness), round(scale * PX_PER_SCALE)).getlength(string))


def draw_body(img, xy, conf, s):
    visible = conf > KP_CONF
    for a, b in SKELETON:
        if visible[a] and visible[b]:
            cv2.line(img, _p(xy[a], s), _p(xy[b], s), BODY, 3, AA)
    for i in range(5, 17):  # body joints only; the face points just add clutter
        if visible[i]:
            cv2.circle(img, _p(xy[i], s), 5, WHITE, -1, AA)


def draw_ball(img, ball, trail_points, s):
    pts = [_p((x, y), s) for _, x, y in trail_points]
    for i in range(1, len(pts)):
        cv2.line(img, pts[i - 1], pts[i], ORANGE, max(2, i // 2), AA)  # newer = thicker
    if ball:
        cv2.circle(img, _p(ball, s), int(ball[2] * s), YELLOW, 3, AA)


def draw_target(img, box, in_zone, s, now, judged_at):
    """The target zone, tinted by the last verdict, plus a short "IN ZONE!"/"MISSED" pop-up."""
    if box is None:
        return
    x1, y1, x2, y2 = (v * s for v in box)
    colour = ZONE_COLOURS[in_zone]
    shade(img, x1, y1, x2, y2, colour, 0.18)
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2, AA)
    text(img, "TARGET", (x1 + 8, y1 + 22), 0.6, colour)

    if y1 < 0:  # the top of the zone is off screen: peaks there can't be seen
        warning = "Step back - the target zone doesn't fit in view"
        tw = text_width(warning, 0.7)
        cx = img.shape[1] / 2
        shade(img, cx - tw / 2 - 16, img.shape[0] - 130, cx + tw / 2 + 16, img.shape[0] - 90,
              (0, 0, 120), 0.7)
        text(img, warning, (cx - tw / 2, img.shape[0] - 102), 0.7, WHITE)

    if in_zone is not None and now - judged_at < VERDICT_SECONDS:
        label = "IN ZONE!" if in_zone else "MISSED"
        tw = text_width(label, 1.2, 2)
        cx = (x1 + x2) / 2
        top = y1 - 60 if y1 > 70 else y2 + 12  # above the zone, or below it if no room
        top = min(max(top, 10), img.shape[0] - 60)  # always fully on screen
        shade(img, cx - tw / 2 - 14, top, cx + tw / 2 + 14, top + 48, (0, 0, 0), 0.6)
        text(img, label, (cx - tw / 2, top + 36), 1.2, colour, 2)


def draw_stats(img, counter, flash):
    """Score panel in the top-right corner."""
    w = img.shape[1]
    x1, y1, x2, y2 = w - 340, 20, w - 20, 325
    shade(img, x1, y1, x2, y2)
    if flash:  # a set was just counted
        cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 3, AA)
    x = x1 + 22

    text(img, "SETS", (x, y1 + 36), 0.7, GREY)
    text(img, str(counter.count), (x, y1 + 110), 2.6, WHITE, 4)

    text(img, "SET HEIGHT", (x, y1 + 150), 0.6, GREY)
    goal = f"target {counter.target_height:.2f}"
    text(img, goal, (x2 - 22 - text_width(goal, 0.6), y1 + 150), 0.6, WHITE)
    if counter.last_set_height is None:
        text(img, "-", (x, y1 + 188), 1.1, WHITE, 2)
    else:
        value = f"{counter.last_set_height:.1f}"
        text(img, value, (x, y1 + 188), 1.1, WHITE, 2)
        text(img, "x shoulder width", (x + text_width(value, 1.1, 2) + 10, y1 + 188), 0.55, GREY)

    text(img, "IN ZONE", (x, y1 + 226), 0.6, GREY)
    if counter.judged:
        pct = counter.in_zone_count / counter.judged
        text(img, f"{counter.in_zone_count}/{counter.judged}", (x, y1 + 262), 1.1, WHITE, 2)
        pct_text = f"{pct:.0%}"
        text(img, pct_text, (x2 - 22 - text_width(pct_text, 1.1, 2), y1 + 262), 1.1,
             GREEN if pct >= 0.7 else YELLOW if pct >= 0.4 else RED, 2)
        bar_x2 = x2 - 22
        cv2.rectangle(img, (x, y2 - 28), (bar_x2, y2 - 18), DARK, -1)
        cv2.rectangle(img, (x, y2 - 28), (x + int((bar_x2 - x) * pct), y2 - 18), GREEN, -1)
    else:
        text(img, "-", (x, y1 + 262), 1.1, WHITE, 2)


def draw_title(img):
    h = img.shape[0]
    title = "VOLLEYBALL SET TRACKER"
    shade(img, 20, h - 70, 52 + text_width(title, 0.75), h - 20)
    text(img, title, (36, h - 48), 0.75, WHITE, 1)
    text(img, "pose estimation + colour tracking", (36, h - 28), 0.5, GREY)


def draw_debug(img, s, lines, face, contact_zone, posture_recent):
    """What the counter is thinking: tuning info, hidden in the clean view."""
    if face:
        (cx, cy), (aw, ah) = face
        cv2.ellipse(img, _p((cx, cy), s), (int(aw * s), int(ah * s)), 0, 0, 360, GREY, 1, AA)
    if contact_zone:
        x1, y1, x2, y2 = (int(v * s) for v in contact_zone)
        cv2.rectangle(img, (x1, y1), (x2, y2), GREEN if posture_recent else GREY, 1, AA)
    shade(img, 20, 20, 640, 30 + 26 * len(lines))
    for i, line in enumerate(lines):
        text(img, line, (32, 45 + 26 * i), 0.55, WHITE)
