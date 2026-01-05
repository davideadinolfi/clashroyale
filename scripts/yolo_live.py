from collections import deque

import cv2
import time
from ultralytics import YOLO
from capture import Capture
import easyocr

from scripts.utilities import getSecondi

# ---- CONFIG ----
MODEL_PATH = "runs/detect/yolo10dataset3_150iterMEDIO/weights/best.pt"
CONFIDENCE = 0.1
MAX_HEIGHT = 720
# ----------------LA RISOLUZIONE DI BLUESTACKS è 1280x640

def resize_by_height(frame, max_height):
    h, w = frame.shape[:2]
    scale = max_height / h
    return cv2.resize(frame, None, fx=scale, fy=scale)

def getTimer(frame, reader):
    show_roi(frame, x=350, y=5, w=100, h=45, win_name="Slot 2")
    # Leggi il numero
    roi = frame[5:50, 350:450]  # ritaglia la zona del timer
    result = reader.readtext(roi, detail=0, allowlist='0123456789:')
    if result:
        return result[-1]


def getElixir(frame):
    show_roi(frame, x=100, y=777, w=330, h=110, win_name="Slot 1")
    cv2.waitKey(1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    x_coords = [420, 400, 365, 335, 305, 275, 240, 210, 180, 145]
    lower_pink = (100, 30, 120)
    upper_pink = (179, 255, 255)

    for i, x in enumerate(x_coords, 1):
        px = hsv[889, x]
        if all(lower_pink[j] <= px[j] <= upper_pink[j] for j in range(3)):
            return 11 - i  # restituisce 10, 9, 8... fino a 1

    return 0

def match_templates_on_positions(frame, threshold=0.4):
    matches = []
    templates = {
        "knight": cv2.imread("../images/knight.png", cv2.IMREAD_GRAYSCALE),
        "archer": cv2.imread("../images/archer.png", cv2.IMREAD_GRAYSCALE),
        "mini-pekka": cv2.imread("../images/mini-pekka.png", cv2.IMREAD_GRAYSCALE),
        "giant": cv2.imread("../images/giant.png", cv2.IMREAD_GRAYSCALE),
        "wizard": cv2.imread("../images/wizard.png", cv2.IMREAD_GRAYSCALE),
        "cannon": cv2.imread("../images/cannon.png", cv2.IMREAD_GRAYSCALE),
        "bomber": cv2.imread("../images/bomber.png", cv2.IMREAD_GRAYSCALE),
        "skeletons": cv2.imread("../images/skeletons.png", cv2.IMREAD_GRAYSCALE),
    }
    positions = {
        1: (113, 777, 54, 63),  # x, y, w, h (percentuali)
        2: (204, 777, 54, 63),
        3: (281, 777, 54, 63),
        4: (363, 777, 54, 63),
    }

#  positions = {
#     1: (0.25, 0.86, 0.12, 0.07),  # x, y, w, h (percentuali)
#     2: (0.45, 0.86, 0.12, 0.07),
#     3: (0.62, 0.86, 0.12, 0.07),
#     4: (0.80, 0.86, 0.12, 0.07),
# }

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for tmpl_name, tmpl_img in templates.items():
        best = {
            "template": tmpl_name,
            "position": None,
            "score": -1,
            "location": None,
        }

        for pos_id, (x, y, w, h) in positions.items():
            x, y, w, h = map(int, (x, y, w, h))  # sicurezza
            tmpl_img = tmpl_img[:, :]
            roi = gray[y:y + h, x:x + w]

            if roi.size == 0:
                continue
            px, py, pw, ph = map(int, (x, y, w, h))
            roi = gray[py:py + ph, px:px + pw]

            tmpl_resized = cv2.resize(tmpl_img, (roi.shape[1], roi.shape[0]))
            res = cv2.matchTemplate(roi, tmpl_resized, cv2.TM_CCOEFF_NORMED)
            #cv2.imshow(f"ROI {pos_id}", roi)
            roi = gray[y:y+h, x:x+w]


            _, score, _, loc = cv2.minMaxLoc(res)

            if score > best["score"]:
                best.update({
                    "position": pos_id,
                    "score": score,
                    "location": (loc[0] + x, loc[1] + y),
                })

        if best["score"] >= threshold:
            matches.append(best)

    return matches

def main():
    cap = Capture("BlueStacks")
    model = YOLO(MODEL_PATH)
    reader = easyocr.Reader(['en'], gpu=False)
    print("[INFO] YOLO live avviato")
    last_time = time.time()
    card_cycle = deque(maxlen=8)
    while True:
        frame = cap.grab()

        # ---- YOLO INFERENCE ----
        results = model.predict(
            source=frame,
            conf=CONFIDENCE,
            imgsz=640,
            verbose=False
        )
        matches=match_templates_on_positions(frame)
        slot_to_card = {
            m["position"]: m["template"]
            for m in matches
        }

        for slot,item in slot_to_card.items():
            if item not in card_cycle:
                card_cycle.append(item)
        print(getElixir(frame))
        print(getSecondi(getTimer(frame, reader)))
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

#debug
def get_positions(frame):
    H, W = frame.shape[:2]

    positions = {
        "slot_1": (0.25, 0.86, 0.12, 0.07),  # x, y, w, h (percentuali)
        "slot_2": (0.45, 0.86, 0.12, 0.07),
        "slot_3": (0.62, 0.86, 0.12, 0.07),
        "slot_4": (0.80, 0.86, 0.12, 0.07),
    }

    rois = {}
    for name, (x, y, w, h) in positions.items():
        rois[name] = (
            int(x * W),
            int(y * H),
            int(w * W),
            int(h * H),
        )

    return rois


def show_roi(frame, x, y, w, h, win_name="ROI"):
    """
    x, y = coordinate top-left (pixel)
    w, h = larghezza e altezza (pixel)
    """
    roi = frame[y:y+h, x:x+w]

    if roi.size == 0:
        print("[WARN] ROI vuota")
        return

    cv2.imshow(win_name, roi)



if __name__ == "__main__":
    main()