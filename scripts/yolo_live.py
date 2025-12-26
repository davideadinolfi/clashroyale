import cv2
import time
from ultralytics import YOLO
from capture import Capture

# ---- CONFIG ----
MODEL_PATH = "runs/detect/yolo10dataset2_800iterMEDIO/weights/best.pt"
CONFIDENCE = 0.1
MAX_HEIGHT = 720
# ----------------

def resize_by_height(frame, max_height):
    h, w = frame.shape[:2]
    scale = max_height / h
    return cv2.resize(frame, None, fx=scale, fy=scale)

def main():
    cap = Capture("BlueStacks")
    model = YOLO(MODEL_PATH)

    print("[INFO] YOLO live avviato")
    last_time = time.time()

    while True:
        frame = cap.grab()

        # ---- YOLO INFERENCE ----
        results = model.predict(
            source=frame,
            conf=CONFIDENCE,
            imgsz=640,
            verbose=False
        )

        annotated = results[0].plot()

        # ---- FPS ----
        now = time.time()
        fps = 1 / (now - last_time)
        last_time = now

        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        display = resize_by_height(annotated, MAX_HEIGHT)
        cv2.imshow("YOLO Live", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()