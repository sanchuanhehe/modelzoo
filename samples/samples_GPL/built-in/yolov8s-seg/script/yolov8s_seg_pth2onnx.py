from ultralytics import YOLO

# 加载预训练模型
model = YOLO("../model/yolov8s-seg.pt")
model.eval()
model.export(format="onnx", dynamic=False, imgsz=(640, 640), opset=11)
