from ultralytics import YOLO
# Load your original PyTorch weights, then re-export optimized
model = YOLO('best1.pt') 
model.export(format='openvino', half=True) # Enables FP16 optimization