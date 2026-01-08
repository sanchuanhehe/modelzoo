# Copyright (c) ModelZoo. 2025-2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np, cv2, argparse, sys

SCORE_THRESHOLD, BOX_STRIP_CHARS = 0.25, "[]"
NMS_IOU_THRESHOLD_DEFAULT, NMS_EPSILON = 0.45, 1e-6
YOLO_CLASS_NUM, CIRCLE_RADIUS, BORDER_THICKNESS = 80, 3, 2
FONT_SCALE, TEXT_THICKNESS, TEXT_OFFSET_X, TEXT_OFFSET_Y = 0.35, 1, 5, 4
BASELINE_OFFSET, BACKGROUND_PADDING, BACKGROUND_WIDTH_OFFSET = 4, 8, 10
BG_OVERLAY_ALPHA, BG_IMAGE_ALPHA = 0.85, 0.15
TEXT_COLOR_MAIN, TEXT_COLOR_SHADOW_1, TEXT_COLOR_SHADOW_2, BACKGROUND_COLOR = (240, 240, 240), (20, 20, 20), (
    50, 50, 50), (245, 245, 245)
BORDER_COLOR_R_COEFF, BORDER_COLOR_G_BASE, BORDER_COLOR_G_COEFF = 255, 100, 155
OUTPUT_FILENAME, CLASSES_TXT = "demo_show.jpg", "person|bicycle|car|motorcycle|airplane|bus|train|truck|boat|traffic light|fire hydrant|stop sign|parking meter|bench|bird|cat|dog|horse|sheep|cow|elephant|bear|zebra|giraffe|backpack|umbrella|handbag|tie|suitcase|frisbee|skis|snowboard|sports ball|kite|baseball bat|baseball glove|skateboard|surfboard|tennis racket|bottle|wine glass|cup|fork|knife|spoon|bowl|banana|apple|sandwich|orange|broccoli|carrot|hot dog|pizza|donut|cake|chair|couch|potted plant|bed|dining table|toilet|tv|laptop|mouse|remote|keyboard|cell phone|microwave|oven|toaster|sink|refrigerator|book|clock|vase|scissors|teddy bear|hair drier|toothbrush"


def parse(file_path):
    bboxes, scores, classes = [], [], []
    for line in open(file_path):
        line = line.strip()
        if not line: continue
        parts = line.split('|')
        score = float(parts[1].split(':')[1].strip())
        if score < SCORE_THRESHOLD: continue
        scores.append(score)
        classes.append(int(parts[0].split()[1]))
        bboxes.append([float(x.strip()) for x in parts[2].split(':')[1].strip().strip(BOX_STRIP_CHARS).split(',')])
    return np.array(bboxes), np.array(scores), np.array(classes)


def nms(bboxes, scores, iou_thresh):
    if len(bboxes) == 0: return np.array([]), np.array([])
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (y2 - y1) * (x2 - x1)
    keep, order = [], scores.argsort()[::-1]
    while order.size > 0:
        i, order = order[0], order[1:]
        keep.append(i)
        if order.size == 0: break
        xx1 = np.maximum(x1[i], x1[order])
        yy1 = np.maximum(y1[i], y1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy2 = np.minimum(y2[i], y2[order])
        w = np.maximum(NMS_EPSILON, xx2 - xx1 + 1)
        h = np.maximum(NMS_EPSILON, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order] - inter)
        order = order[np.where(iou <= iou_thresh)[0]]
    return bboxes[keep], scores[keep]


def process(input_file, iou_thresh=NMS_IOU_THRESHOLD_DEFAULT):
    bboxes, scores, classes = parse(input_file)

    keep_bboxes, keep_scores = nms(bboxes, scores, iou_thresh)
    return [{'class': classes[np.where((bboxes == b).all(axis=1))[0][0]], 'score': s, 'box': b} for b, s in
            zip(keep_bboxes, keep_scores)]


def draw(file_path, boxes, output_path="demo_show.jpg"):
    CLASS_MAPPING = dict(zip(range(YOLO_CLASS_NUM), CLASSES_TXT.split('|')))
    img = cv2.imread(file_path)
    for bbox in boxes:
        class_id, score, box = bbox['class'], bbox['score'], bbox['box']
        x1, y1, x2, y2 = map(int, box)
        border_color = (
            int(BORDER_COLOR_R_COEFF * score), int(BORDER_COLOR_G_BASE + BORDER_COLOR_G_COEFF * (1 - score)))
        cv2.rectangle(img, (x1, y1), (x2, y2), border_color, BORDER_THICKNESS)
        for corner in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]: cv2.circle(img, corner, CIRCLE_RADIUS, border_color, -1)
        label = f"{CLASS_MAPPING[class_id]} | {score:.2%}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                                                              TEXT_THICKNESS)
        overlay, bg_y1, bg_y2 = img.copy(), y1 - text_height - baseline - BACKGROUND_PADDING, y1 - baseline - 2
        if bg_y1 < 0: bg_y1 = 0
        cv2.rectangle(overlay, (x1, bg_y1), (x1 + text_width + BACKGROUND_WIDTH_OFFSET, bg_y2), BACKGROUND_COLOR, -1)
        cv2.addWeighted(overlay, BG_OVERLAY_ALPHA, img, BG_IMAGE_ALPHA, 0, img)
        text_pos = (x1 + TEXT_OFFSET_X, y1 - baseline - BASELINE_OFFSET)
        for color, offset in [(TEXT_COLOR_SHADOW_1, 3), (TEXT_COLOR_SHADOW_2, 1), (TEXT_COLOR_MAIN, 0)]:
            cv2.putText(img, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, TEXT_THICKNESS + offset,
                        cv2.LINE_AA)
    cv2.imwrite(output_path, img)

def main():
    parser = argparse.ArgumentParser(
        description="YOLO检测结果可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python3 drawRectangle.py -i image.jpg -r result.txt -o 000000000139_show.jpg --iou 0.5
        """
    )

    parser.add_argument('-i', '--image', required=True, help='输入图片路径')
    parser.add_argument('-r', '--result', required=True, help='检测结果文件路径')
    parser.add_argument('-t', '--iou', type=float, default=NMS_IOU_THRESHOLD_DEFAULT,
                       help=f'IOU阈值 (默认: {NMS_IOU_THRESHOLD_DEFAULT})')
    parser.add_argument('-o', '--output', default="../out/demo_show.jpg",
                       help='输出文件名 (默认: demo_show.jpg)')

    args = parser.parse_args()

    # 参数验证
    if args.iou < 0 or args.iou > 1:
        print("错误: IOU阈值必须在0到1之间")
        sys.exit(1)
    # 处理流程
    try:
        boxes = process(args.result, args.iou)
        if not boxes:
            print("未检测到任何对象，跳过绘图")
            return

        print(f"检测到 {len(boxes)} 个边界框")
        draw(args.image, boxes, args.output)
        print("处理完成！")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()