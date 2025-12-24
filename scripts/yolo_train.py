from ultralytics import YOLO
def main():
    model = YOLO('yolov10m.pt')
    model.train(
        data='../dataset/3/data.yaml',  # percorso al file yaml
        epochs=800,
        imgsz=640,
        batch=16,
        device='0',  # o '0' se hai GPU Nvidia
        name='yolo10dataset3_800iterMEDIO',  # nome cartella output
        mosaic=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=10,
        scale=0.0,
        erasing=0.0,
    )
if __name__ == "__main__":
    main()