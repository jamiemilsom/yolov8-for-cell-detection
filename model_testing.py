from ultralytics import YOLO
import cv2
import os

model = YOLO('runs/detect/train4/weights/best.pt')

images = ['cell_data/images/test/' + image for image in os.listdir('cell_data/images/test')]
results = model(images)


for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    probs = result.probs  # Probs object for classification outputs

    im_array = result.plot()  
    cv2.imshow('Image with Predicted Bounding Boxes', im_array)
    if cv2.waitKey(0) == ord('q'):
        break  