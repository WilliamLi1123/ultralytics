from ultralytics import YOLO
import os
os.environ["CUDA_VISIBLE_DEVICES"] = " "
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    model = YOLO("yolov8n.yaml").load("yolov8n.pt")

    model.train(data= "data.yaml", imgsz=640, epochs=2, batch=16, workers=0)