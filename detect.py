"""Step 2: live object detection on the webcam with a pretrained YOLO model.
Press 'q' or Esc to quit."""

import time

import cv2
from ultralytics import YOLO

CAMERA_INDEX = 0
MODEL = "yolo26n.pt"  # 'n' = nano: smallest and fastest. Try 's' or 'm' for more accuracy.
CONFIDENCE = 0.5      # ignore detections the model is less than 50% sure about
WINDOW = "Object detection"


def main():
    model = YOLO(MODEL)  # downloads the weights the first time (~5 MB)

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

        # Run the model on this frame. It returns one result per image.
        result = model(frame, conf=CONFIDENCE, verbose=False)[0]

        # Each detection has a box (x1, y1, x2, y2), a class id and a confidence.
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls)]
            conf = float(box.conf)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Frames per second, so you can see how fast the model runs
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
