from ultralytics import YOLO

model = YOLO('yolo26n.pt') 

model.train(
    data='data.yaml', 
    epochs=50, 
    imgsz=640,
    hsv_s=0.4,
    hsv_v=0.4,
    degrees=30,
    translate=0.2,
    scale=0.5,
    mosaic=0.0,
    workers=8,
    rect=True,
    cache='ram',
    patience=15
  )