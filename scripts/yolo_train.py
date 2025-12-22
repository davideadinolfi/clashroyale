from ultralytics import YOLO
# Modello di base (leggero)
model = YOLO('yolov8n.pt')

# Addestramento
model.train(
    data='../dataset/1/data.yaml',  # percorso al file yaml
    epochs=100,
    imgsz=640,
    batch=16,
    device='cpu',  # o '0' se hai GPU Nvidia
    name='cr_custom',  # nome cartella output
)