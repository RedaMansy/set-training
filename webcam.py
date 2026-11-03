"""Step 1: show a live webcam feed. Press 'q' or Esc to quit."""

import cv2

CAMERA_INDEX = 0  # 0 = default webcam; try 1, 2... if you have several cameras


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}.")
        return

    print("Webcam running. Press 'q' or Esc in the video window to quit.")
    while True:
        ok, frame = cap.read()  # frame is a NumPy array (height x width x 3, BGR)
        if not ok:
            print("Failed to read a frame from the camera.")
            break

        frame = cv2.flip(frame, 1)  # mirror the image so it feels natural
        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF  # also lets the window redraw
        if key in (ord("q"), 27):
            break
        # Stop if the user closed the window with the X button
        if cv2.getWindowProperty("Webcam", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
