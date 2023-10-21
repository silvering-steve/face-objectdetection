from ultralytics import YOLO


class Model:
    def __init__(self):
        self.model = YOLO("models/face.pt")



model = Model()
