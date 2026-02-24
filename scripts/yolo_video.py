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

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # PROVA DIVERSI CODEC
    # Opzione 1: .avi con XVID (più compatibile)
    output_path_avi = str(output_path).replace('.mp4', '.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path_avi, fourcc, fps, (width, height))

    # Opzione 2: se vuoi .mp4, prova 'avc1'
    # fourcc = cv2.VideoWriter_fourcc(*'avc1')
    # out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        print("ERRORE: impossibile creare video writer")
        cap.release()
        return

    print(f"✓ Video writer creato")

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Fine video al frame {frame_count}")
            break

        # YOLO inference
        results = model.predict(frame, conf=0.4, verbose=False)

        # Frame annotato
        annotated = results[0].plot(
            line_width=1,
            font_size=10
        )

        # IMPORTANTE: verifica che annotated abbia le dimensioni giuste
        if annotated.shape[:2] != (height, width):
            annotated = cv2.resize(annotated, (width, height))

        # Salva frame
        out.write(annotated)

        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processati {frame_count}/{total_frames} frame ({progress:.1f}%)...")

    # Cleanup
    cap.release()
    out.release()

    # Verifica dimensione file
    from pathlib import Path
    output_file = Path(output_path_avi)
    if output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"✓ Video salvato: {output_file}")
        print(f"  Dimensione: {size_mb:.2f} MB")
        print(f"  Frames: {frame_count}")
    else:
        print("✗ ERRORE: file output non creato!")


# Uso
process_video_basic(
    video_path=Path(r"C:\Users\Davide Adinolfi\Downloads\porcodio.mp4"),
    model_path="runs/detect/generato6/weights/last.pt",
    output_path="runs/video/video_output.mp4"
)