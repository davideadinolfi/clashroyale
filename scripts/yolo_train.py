from ultralytics import YOLO
# Modello di base (leggero)
model = YOLO('yolov8n.pt')

# Addestramento
model.train(
    data='../dataset/2/data.yaml',  # percorso al file yaml
    epochs=50,
    imgsz=640,
    batch=16,
    device='0',  # o '0' se hai GPU Nvidia
    name='dataset2_50iter',  # nome cartella output
)
model.train(
    data='../dataset/2/data.yaml',  # percorso al file yaml
    epochs=200,
    imgsz=640,
    batch=16,
    device='0',  # o '0' se hai GPU Nvidia
    name='dataset2_200iter',  # nome cartella output
)
model.train(
    data='../dataset/3/data.yaml',  # percorso al file yaml
    epochs=50,
    imgsz=640,
    batch=16,
    device='0',  # o '0' se hai GPU Nvidia
    name='dataset3_50iter',  # nome cartella output
)
model.train(
    data='../dataset/3/data.yaml',  # percorso al file yaml
    epochs=200,
    imgsz=640,
    batch=16,
    device='0',  # o '0' se hai GPU Nvidia
    name='dataset3_200iter',  # nome cartella output
)