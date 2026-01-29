from ultralytics import YOLO
def main():
    model = YOLO('yolov10s.pt' )
    model.train(
        data='../generation/dataset_output/dataset.yaml',  # percorso al file yaml
        epochs=100,
        imgsz=640,
        batch=16,
        device='0',  # o '0' se hai GPU Nvidia
        name='datasetgenerativo1',  # nome cartella output
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=10,
        erasing=0.0,
    )
if __name__ == "__main__":
    main()