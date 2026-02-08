from ultralytics import YOLO
def main():
    model = YOLO('yolov10m.pt' )
    model.train(
        data='../dataset/generato2/dataset.yaml',  # percorso al file yaml
        epochs=100,
        batch=16,
        device='0',  # o '0' se hai GPU Nvidia
        name='generato2',  # nome cartella output
    )
if __name__ == "__main__":
    main()