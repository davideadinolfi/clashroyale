from ultralytics import YOLO

model = YOLO("runs/detect/generato7/weights/last.pt")

metrics = model.val(
    data="../dataset/aiuto/dataset.yaml",   # stesso yaml usato nel training
    split="val",          # 'val' di default, metti 'test' se hai quella sezione
    imgsz=640,
    batch=1,
    conf=0.001,            # soglia bassa → curve complete (come nel training)
    iou=0.4,
    save_json=True,        # salva predictions in formato COCO
    plots=True,            # genera confusion matrix, PR curve, ecc.
)

print(metrics.box.map)      # mAP50-95
print(metrics.box.map50)    # mAP50
print(metrics.box.mp)       # mean precision
print(metrics.box.mr)       # mean recall