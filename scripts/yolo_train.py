from ultralytics import YOLO

from katacr.utils.detection.data import transform_affine


def main():
    model = YOLO('yolov10m.pt' )
    model.train(
        data='../dataset/generato5/dataset.yaml',  # percorso al file yaml
        epochs=100,
        batch=16,
        device='0',  # o '0' se hai GPU Nvidia
        name='generato5',  # nome cartella output
        hsv_h=0.003,
        hsv_s=0.2,
        hsv_v=0.1,
        degrees=5,
        translate=0.2,
        mosaic=0.4,
        erasing=0.2
    )
if __name__ == "__main__":
    main()