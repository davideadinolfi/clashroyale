from ultralytics import YOLO
import cv2
from pathlib import Path


def extract_frame_and_predict(video_path, model_path, frame_number=0, output_path="prediction.jpg"):
    """
    Estrae un frame da un video, applica YOLO e salva il risultato.

    Args:
        video_path: percorso del video
        model_path: percorso del modello YOLO
        frame_number: quale frame estrarre (default: 0 = primo frame)
        output_path: dove salvare l'immagine con predizioni
    """

    # Carica modello
    model = YOLO(model_path)

    # Apri video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"❌ Impossibile aprire il video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frame totali")

    # Vai al frame desiderato
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Leggi frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"❌ Impossibile leggere il frame {frame_number}")
        return

    print(f"✓ Frame {frame_number} estratto: {frame.shape}")

    # YOLO prediction
    results = model.predict(frame, conf=0.4, verbose=True)

    # Frame con bounding box
    annotated = results[0].plot()

    # Salva
    cv2.imwrite(output_path, annotated)

    print(f"✓ Immagine salvata in: {output_path}")
    print(f"  Detections: {len(results[0].boxes)}")

    # Mostra dettagli detections
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = results[0].names[cls]
        print(f"    - {name}: {conf:.2f}")

    return annotated


# ========== USO ==========

# Primo frame
extract_frame_and_predict(
    video_path=Path(r"C:\Users\Davide Adinolfi\Downloads\porcodio.mp4"),
    model_path="runs/detect/generato5/weights/best.pt",
    frame_number=1,
    output_path="runs/images/pred.jpg"
)
