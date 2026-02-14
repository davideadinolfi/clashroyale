from collections import deque
import sys
import cv2
import time
from ultralytics import YOLO
from capture import Capture
import easyocr
from scripts.utilities import getSecondi

# ---- CONFIG ----
MODEL_PATH = "runs/detect/generato2/weights/last.pt"
CONFIDENCE = 0.6
MAX_HEIGHT = 720
# ----------------LA RISOLUZIONE DI BLUESTACKS è 1280x640

def resize_by_height(frame, max_height):
    h, w = frame.shape[:2]
    scale = max_height / h
    return cv2.resize(frame, None, fx=scale, fy=scale)

def getTimer(frame, reader):
#    show_roi(frame, x=350, y=5, w=100, h=45, win_name="Slot 2")
    # Leggi il numero
    roi = frame[5:50, 350:450]  # ritaglia la zona del timer
    result = reader.readtext(roi, detail=0, allowlist='0123456789:')
    if result:
        return result[-1]


def getElixir(frame):
#    show_roi(frame, x=100, y=777, w=330, h=110, win_name="Slot 1")
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


def create_grid_matrix(results, frame_shape, grid_rows=72, grid_cols=128):
    """
    Crea una matrice 72x128 con i nomi degli oggetti classificati.
    Ogni cella contiene il nome dell'oggetto presente in quella zona.
    """
    h, w = frame_shape[:2]
    cell_h = h / grid_rows
    cell_w = w / grid_cols

    # Matrice vuota
    grid = [[None for _ in range(grid_cols)] for _ in range(grid_rows)]

    # Per ogni detection
    boxes = results[0].boxes
    for box in boxes:
        # Coordinate del bounding box
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls = int(box.cls[0])
        name = results[0].names[cls]

        # Centro del bounding box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Converti in coordinate griglia
        grid_x = int(cx / cell_w)
        grid_y = int(cy / cell_h)

        # Assicurati che sia dentro i limiti
        grid_x = min(grid_x, grid_cols - 1)
        grid_y = min(grid_y, grid_rows - 1)

        grid[grid_y][grid_x] = name

    return grid


def get_detections_info(results):
    """
    Restituisce lista con posizioni e nomi degli oggetti.
    """
    detections = []
    boxes = results[0].boxes

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls = int(box.cls[0])
        name = results[0].names[cls]
        conf = float(box.conf[0])

        detections.append({
            'name': name,
            'position': (int((x1 + x2) / 2), int((y1 + y2) / 2)),  # centro
            'bbox': (int(x1), int(y1), int(x2), int(y2)),
            'confidence': conf
        })

    return detections


def update(cap , model , reader):
    last_time = time.time()

    frame = cap.grab()

    # ---- YOLO INFERENCE ----
    results = model.predict(
        source=frame,
        conf=CONFIDENCE,
        imgsz=640,
        verbose=False
    )

    # Ottieni lista detections
    detections = get_detections_info(results)

    # Crea matrice 72x128
    grid_matrix = create_grid_matrix(results, frame.shape)

    matches = match_templates_on_positions(frame)
    slot_to_card = {
        m["position"]: m["template"]
        for m in matches
    }

    return grid_matrix, getElixir(frame), getSecondi(getTimer(frame, reader)), slot_to_card


def main():
    cap = Capture("BlueStacks")
    model = YOLO(MODEL_PATH)
    print(sys.executable)

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

        # Ottieni lista detections
        detections = get_detections_info(results)

        # Crea matrice 72x128
        grid_matrix = create_grid_matrix(results, frame.shape)

        matches = match_templates_on_positions(frame)
        slot_to_card = {
            m["position"]: m["template"]
            for m in matches
        }

        for slot, item in slot_to_card.items():
            if item not in card_cycle:
                card_cycle.append(item)

        # Stampa info (opzionale)
        print(f"Detections: {len(detections)}")
        #print(f"Elixir: {getElixir(frame)}")
       # print(f"Timer: {getSecondi(getTimer(frame, reader))}")

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

        cv2.imshow("yolo live", display)

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