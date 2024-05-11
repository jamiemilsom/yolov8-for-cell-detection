from ultralytics import YOLO


model = YOLO('yolov8n.yaml')
results = model.train(data='config.yaml', epochs=100,batch=1)