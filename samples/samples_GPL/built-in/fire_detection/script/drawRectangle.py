import cv2
import re
import os
import numpy as np

def get_color(class_id):
    """根据类ID生成固定的随机颜色"""
    # 使用随机种子确保同一个ID每次颜色都一样
    np.random.seed(class_id)
    return tuple(np.random.randint(0, 255, size=3).tolist())

def draw_from_files(image_path, txt_path, conf_threshold=0.01):
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

        label_name = "fire"
        display_txt = f"{label_name} {score:.2f}"

        # 3. 绘图
        # 画矩形框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 绘制标签背景（颜色与框一致）
        (w, h), baseline = cv2.getTextSize(display_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1), color, -1)

        # 写文字（白色或黑色，取决于背景深浅，这里统一用白色）
        cv2.putText(img, display_txt, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        count += 1

    # 4. 展示与保存
    save_name = "res_" + os.path.basename(image_path)
    cv2.imwrite(save_name, img)
    print(f"检测完成！共绘制了 {count} 个目标")
    print(f"结果已保存至: {save_name}")

    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("--- 目标检测结果可视化工具 (多颜色版) ---")
    img_in = input("请输入图片路径: ").strip().replace('"', '').replace("'", "")
    txt_in = input("请输入结果txt路径: ").strip().replace('"', '').replace("'", "")

    # 建议阈值设为 0.01 来看你的 Class 0 结果
    draw_from_files(img_in, txt_in, conf_threshold=0.01)
