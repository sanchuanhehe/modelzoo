import cv2
import re
import os
import numpy as np
import argparse

# 1. 定义 YOLO 80 类名称
YOLO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def get_color(class_id):
    """根据类ID生成固定的随机颜色"""
    np.random.seed(class_id)
    return tuple(np.random.randint(0, 255, size=3).tolist())

def draw_from_files(image_path, txt_path, conf_threshold=0.01, show=False, output_dir=None):
    # 加载图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法加载图片 {image_path}")
        return

    # 读取 txt 文件内容
    if not os.path.exists(txt_path):
        print(f"错误: 找不到文件 {txt_path}")
        return

    with open(txt_path, 'r', encoding='utf-8') as f:
        result_text = f.read()

    # 2. 正则表达式解析文本
    pattern = r"Class (\d+) \| Score: ([\d.]+) \| Box: \[([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+)\]"
    matches = re.findall(pattern, result_text)

    count = 0
    for match in matches:
        cls_id = int(match[0])
        score = float(match[1])

        # 置信度过滤
        if score < conf_threshold:
            continue

        # 坐标解析
        x1, y1, x2, y2 = map(int, [float(x) for x in match[2:]])

        # 获取该类别的固定颜色
        color = get_color(cls_id)
        label_name = YOLO_CLASSES[cls_id] if cls_id < len(YOLO_CLASSES) else f"ID:{cls_id}"
        display_txt = f"{label_name} {score:.2f}"

        # 3. 绘图
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (w, h), baseline = cv2.getTextSize(display_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(img, display_txt, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        count += 1

    # 4. 保存
    base_name = os.path.basename(image_path)
    save_path = os.path.join(output_dir if output_dir else "", "res_" + base_name)
    cv2.imwrite(save_path, img)
    print(f"检测完成！共绘制了 {count} 个目标")
    print(f"结果已保存至: {save_path}")

    # 5. 可选展示
    if show:
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 结果可视化工具")

    # 添加命令行参数
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--txt", type=str, required=True, help="结果 txt 文件路径")
    parser.add_argument("--conf", type=float, default=0.01, help="置信度阈值 (默认: 0.01)")
    parser.add_argument("--show", action="store_true", help="是否在窗口显示结果图片")
    parser.add_argument("--output", type=str, default="vis_results", help="结果保存目录 (默认: 当前目录)")

    args = parser.parse_args()

    # 处理路径中可能存在的引号
    img_path = args.image.replace('"', '').replace("'", "")
    txt_path = args.txt.replace('"', '').replace("'", "")

    draw_from_files(img_path, txt_path, conf_threshold=args.conf, show=args.show, output_dir=args.output)
