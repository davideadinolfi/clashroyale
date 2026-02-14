from ultralytics import YOLO
import cv2
from pathlib import Path

# ========== VERSIONE BASE ==========

def process_video_basic(video_path, model_path, output_path):
    """Processa video con YOLO e salva risultato"""

    # Carica modello
    model = YOLO(model_path)
    print(video_path)
    print(f"Esiste: {video_path.exists()}")
    # Apri video
    cap = cv2.VideoCapture(video_path)
    print(f"Video aperto: {cap.isOpened()}")
    # Parametri video output
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writer per salvare output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model.predict(frame, conf=0.5, verbose=False)

        # Frame annotato
        annotated = results[0].plot()

        # Salva frame
        out.write(annotated)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processati {frame_count} frame...")

    # Cleanup
    cap.release()
    out.release()
    print(f"Video salvato in: {output_path}")


# Uso
process_video_basic(
    video_path=Path(r"C:\Users\Davide Adinolfi\Downloads\Video senza titolo - Realizzato con Clipchamp (1).mp4"),
    model_path="runs/detect/generato2/weights/last.pt",
    output_path="runs/video_output.mp4"
)