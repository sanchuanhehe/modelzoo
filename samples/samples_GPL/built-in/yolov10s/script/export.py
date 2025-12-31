from ultralytics import YOLOv10

# 加载 YOLOv10 模型（确保是你的 pt 权重）
model = YOLOv10("../model/yolov10s.pt")

# 导出 ONNX（仅传递必要参数，后处理已通过源码禁用）
success = model.export(
    format="onnx",          # 导出格式为 ONNX
    opset=13,               # 兼容旧版本框架，稳定性最高
    simplify=True,         # 已在源码强制禁用，这里显式设置更保险
    imgsz=640,              # 与你 PT 推理时的输入尺寸一致（如 640×640）
    dynamic=False,          # 禁用动态维度，输出维度固定为 (1, 84, 2646)
    verbose=True ,
    nms=False           # 打印日志，方便排查
)

print("ONNX 导出成功！" if success else "ONNX 导出失败！")
