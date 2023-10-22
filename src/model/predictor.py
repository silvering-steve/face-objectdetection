import cv2
import numpy as np
from ultralytics import YOLO

class Predict:
    def __init__(self):
        self.model = YOLO("models/face.pt")

    def __call__(self, image, *args, **kwargs):
        image_bytes = np.asarray(bytearray(image), dtype=np.uint8)
        image_array = cv2.imdecode(image_bytes, 1)

        result = self.model.predict(image_array, conf=kwargs['conf'], iou=kwargs['iou'])

        return result[0].plot()


predict = Predict()