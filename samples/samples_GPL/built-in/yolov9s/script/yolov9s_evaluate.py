# Copyright (c) ModelZoo. 2025-2026. All rights reserved.
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

"""
YOLOv9s 评估脚本
支持两种输入格式：
  - bin: 原始模型输出文件（需要 NMS 后处理）
  - txt: 后处理结果文件（已包含检测框、置信度、类别）

自动检测输入目录中的文件类型，并使用 ultralytics DetectionValidator 进行评估。
"""

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import TQDM
from ultralytics.utils.checks import check_imgsz
from ultralytics.cfg import get_cfg

TXT_DET_PATTERN = re.compile(
    r"Class\s+(\d+)\s*\|\s*Score:\s*([\d.]+)\s*\|\s*Box:\s*\[([\d.,\s]+)\]"
)
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ============================================================================
# 通用函数
# ============================================================================

def detect_input_type(input_dir: str) -> Optional[str]:
    """
    自动检测输入目录中的文件类型。

    Args:
        input_dir: 输入目录路径

    Returns:
        'bin' 或 'txt'，如果无法确定则返回 None
    """
    input_path = Path(input_dir)
    bin_count = sum(1 for _ in input_path.glob("*.bin"))
    txt_count = sum(1 for _ in input_path.glob("*_result.txt"))
    has_bin = bin_count > 0
    has_txt = txt_count > 0

    if has_bin and not has_txt:
        return "bin"
    if has_txt and not has_bin:
        return "txt"
    if has_bin and has_txt:
        # 两者都存在，选择数量更多的
        return "bin" if bin_count >= txt_count else "txt"
    return None


def empty_prediction(device: torch.device) -> Dict[str, torch.Tensor]:
    """Create an empty prediction dict compatible with DetectionValidator.update_metrics()."""
    return {
        "bboxes": torch.empty((0, 4), dtype=torch.float32, device=device),
        "conf": torch.empty((0,), dtype=torch.float32, device=device),
        "cls": torch.empty((0,), dtype=torch.float32, device=device),
        "extra": torch.empty((0, 0), dtype=torch.float32, device=device),
    }


def build_filtered_dataloader(
    validator: DetectionValidator, val_path: str, input_stems: Set[str]
) -> str:
    """构建只包含对应输入文件的 dataloader 图片列表。"""
    print("Filtering dataloader to only include images with input files...")

    dataset_path = Path(validator.data.get("path", "."))
    val_path = Path(val_path)
    if val_path.is_dir():
        image_paths = [p for p in val_path.rglob("*") if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES]
    else:
        with open(val_path, "r", encoding="utf-8") as f:
            image_paths = [
                path if path.is_absolute() else dataset_path / path
                for line in f
                if (line_str := line.strip())
                for path in [Path(line_str)]
            ]

    filtered_images = [p for p in image_paths if p.stem in input_stems]
    if not filtered_images:
        raise ValueError("No images found with corresponding input files!")

    print(f"Filtered to {len(filtered_images)} images with input files")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as temp_list_file:
        temp_list_file.writelines(f"{img_path.resolve()}\n" for img_path in filtered_images)
        return temp_list_file.name


def build_validator_overrides(args: argparse.Namespace, conf: float) -> Dict[str, Any]:
    """Build common validator override arguments."""
    return {
        "model": args.model,
        "data": args.data,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "rect": False,
        "save_json": True,
        "task": "detect",
        "mode": "val",
        "half": False,
        "device": args.device,
        "conf": conf,
        "iou": 0.7,
        "max_det": args.max_det,
        "verbose": True,
        "plots": False,
    }


def create_model_and_validator(
    args: argparse.Namespace, conf: float
) -> Tuple[YOLO, DetectionValidator]:
    """Create YOLO model and initialized DetectionValidator."""
    print(f"Loading model {args.model}...")
    model = YOLO(args.model)

    validator = DetectionValidator(args=get_cfg(overrides=build_validator_overrides(args, conf)))
    validator.training = False

    if hasattr(model, "model") and hasattr(model.model, "stride"):
        validator.stride = int(model.model.stride.max())
    else:
        validator.stride = 32

    validator.data = check_det_dataset(args.data)
    validator.device = torch.device(args.device)
    validator.args.imgsz = check_imgsz(args.imgsz, stride=validator.stride)
    return model, validator


def prepare_dataloader(
    validator: DetectionValidator, args: argparse.Namespace, input_stems: Set[str]
) -> str:
    """Prepare dataloader filtered by available input stems."""
    print(f"Building dataloader for {args.data}...")
    val_path = validator.data.get("val")
    if not val_path:
        raise ValueError(f"Validation path not found in {args.data}")

    filtered_val_path = build_filtered_dataloader(validator, val_path, input_stems)
    validator.dataloader = validator.get_dataloader(filtered_val_path, args.batch)
    return filtered_val_path


def finalize_and_save_results(validator: DetectionValidator) -> Dict[str, Any]:
    """Finalize metrics and save json results when enabled."""
    stats = validator.get_stats()
    validator.finalize_metrics()
    validator.print_results()

    if validator.args.save_json and validator.jdict:
        pred_json = validator.save_dir / "predictions.json"
        with open(str(pred_json), "w", encoding="utf-8") as f:
            print(f"Saving {f.name}...")
            json.dump(validator.jdict, f)
        stats = validator.eval_json(stats)

    if validator.args.save_json:
        print(f"Results saved to {validator.save_dir}")

    return stats

# ============================================================================
# BIN 模式评估函数
# ============================================================================

def evaluate_from_bin(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Evaluate YOLOv9s metrics (mAP) using pre-computed raw output files (*.bin).
    """
    model, validator = create_model_and_validator(args, conf=0.001)

    # Pre-scan bin directory to get available bin files
    bin_dir = Path(args.input)
    bin_stems = {f.stem for f in bin_dir.glob("*.bin")}
    if not bin_stems:
        print(f"错误: No .bin files found in {bin_dir}!")
        return {}

    print(f"Found {len(bin_stems)} .bin files in {bin_dir}")
    
    filtered_val_path: Optional[str] = None
    try:
        filtered_val_path = prepare_dataloader(validator, args, bin_stems)

        # Initialize metrics and class names
        validator.init_metrics(model.model)

        num_classes = len(validator.data["names"])
        out_channels = num_classes + 4

        print(f"Starting evaluation from {bin_dir}...")
        bar = TQDM(validator.dataloader, desc=validator.get_desc(), total=len(validator.dataloader))

        for batch_i, batch in enumerate(bar):
            # Preprocess batch (moves to device, normalizes images)
            batch = validator.preprocess(batch)

            # Preserve per-image alignment: missing/invalid bin keeps empty prediction for that image.
            batch_preds = [empty_prediction(validator.device) for _ in batch["im_file"]]
            valid_indices = []
            valid_raw_preds = []

            for si, im_file in enumerate(batch["im_file"]):
                stem = Path(im_file).stem
                bin_path = bin_dir / f"{stem}.bin"
                if not bin_path.exists():
                    print(f"警告: Missing bin file: {bin_path}, treated as empty prediction")
                    continue

                raw = np.fromfile(bin_path, dtype=np.float32)
                if raw.size == 0:
                    print(f"警告: Empty bin file: {bin_path}, treated as empty prediction")
                    continue
                if raw.size % out_channels != 0:
                    print(
                        f"警告: Bin file {bin_path} size {raw.size} not divisible by {out_channels}, treated as empty prediction"
                    )
                    continue

                num_anchors = raw.size // out_channels
                valid_raw_preds.append(torch.from_numpy(raw.reshape(1, out_channels, num_anchors)))
                valid_indices.append(si)

            if valid_raw_preds:
                # Stack valid raw tensors -> (N_valid, out_channels, N)
                try:
                    preds = torch.cat(valid_raw_preds, dim=0).to(validator.device)
                except RuntimeError as e:
                    print(f"错误: Failed to stack predictions in batch {batch_i}: {e}")
                    validator.update_metrics(batch_preds, batch)
                    continue

                # Postprocess (NMS) and place predictions back to original image index
                processed_valid_preds = validator.postprocess(preds)
                for si, pred in zip(valid_indices, processed_valid_preds):
                    batch_preds[si] = pred

            validator.update_metrics(batch_preds, batch)

        return finalize_and_save_results(validator)
    finally:
        if filtered_val_path and Path(filtered_val_path).exists():
            Path(filtered_val_path).unlink()


# ============================================================================
# TXT 模式评估函数
# ============================================================================

def parse_txt_detection(txt_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Parse a single txt file with detections.
    Format: "Class {id} | Score: {conf} | Box: [x1, y1, x2, y2]"
    
    Returns:
        dict with keys: 'bboxes', 'conf', 'cls', 'extra'
    """
    bboxes = []
    confs = []
    cls_ids = []

    if not txt_path.exists():
        return empty_prediction(device)

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse: "Class {id} | Score: {conf} | Box: [x1, y1, x2, y2]"
            match = TXT_DET_PATTERN.match(line)
            if not match:
                continue

            cls_id = int(match.group(1))
            conf = float(match.group(2))
            box_coords = np.fromstring(match.group(3), sep=",", dtype=np.float32)
            if box_coords.size != 4:
                continue

            bboxes.append(box_coords)
            confs.append(conf)
            cls_ids.append(cls_id)

    if not bboxes:
        return empty_prediction(device)

    bbox_array = np.stack(bboxes)
    return {
        "bboxes": torch.from_numpy(bbox_array).to(device=device, dtype=torch.float32),
        "conf": torch.tensor(confs, dtype=torch.float32, device=device),
        "cls": torch.tensor(cls_ids, dtype=torch.float32, device=device),
        "extra": torch.empty((len(bboxes), 0), dtype=torch.float32, device=device),
    }


def scale_boxes_to_model_space(
    pred_dict: Dict[str, torch.Tensor], pbatch: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    """
    Scale predictions from original image space back to model input space.
    This is the inverse of validator.scale_preds().
    """
    if pred_dict["bboxes"].shape[0] == 0:
        return pred_dict

    ori_h, ori_w = pbatch["ori_shape"]
    img_h, img_w = pbatch["imgsz"]

    # Step 1: pre_r — 预缩放比例，使长边缩放到模型输入尺寸（不放大）
    pre_r = float(max(img_w, img_h)) / float(max(ori_w, ori_h))
    resized_w, resized_h = ori_w, ori_h
    if pre_r != 1.0:
        resized_w = min(int(np.ceil(float(ori_w) * pre_r)), int(img_w))
        resized_h = min(int(np.ceil(float(ori_h) * pre_r)), int(img_h))

    # Step 2: letterbox_r — letterbox 再次缩放比例
    # 通常 pre_r 已使图片适应模型输入，此处 letterbox_r 大多为 1.0。
    # 但在极端宽高比等边界情况下，ceil 取整可能导致 resized 尺寸略超出
    # 模型输入，此时 letterbox_r 会略小于 1.0 进行修正。
    letterbox_r = min(float(img_w) / float(resized_w), float(img_h) / float(resized_h))
    letterbox_r = min(letterbox_r, 1.0)
    unpad_w = int(round(float(resized_w) * letterbox_r))
    unpad_h = int(round(float(resized_h) * letterbox_r))

    # Step 3: 计算 padding 偏移（居中填充，与预处理的 letterbox 对齐）
    pad_w = int(round((float(img_w) - float(unpad_w)) / 2.0 - 0.1))
    pad_h = int(round((float(img_h) - float(unpad_h)) / 2.0 - 0.1))
    scale = pre_r * letterbox_r

    boxes = pred_dict["bboxes"].clone()
    boxes[..., 0] = boxes[..., 0] * scale + pad_w  # x1
    boxes[..., 1] = boxes[..., 1] * scale + pad_h  # y1
    boxes[..., 2] = boxes[..., 2] * scale + pad_w  # x2
    boxes[..., 3] = boxes[..., 3] * scale + pad_h  # y2

    return {
        **pred_dict,
        "bboxes": boxes,
    }


def evaluate_from_txt(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Evaluate YOLOv9s metrics (mAP) using post-processed txt files.
    txt format: "Class {id} | Score: {conf} | Box: [x1, y1, x2, y2]"
    """
    model, validator = create_model_and_validator(args, conf=args.conf)

    # Pre-scan txt directory to get available txt files
    txt_dir = Path(args.input)

    # Build mapping: image stem -> txt path (expects a_result.txt for image a.jpg)
    txt_stems = {txt_path.name.replace("_result.txt", ""): txt_path for txt_path in txt_dir.glob("*_result.txt")}

    if not txt_stems:
        print(f"错误: No .txt files found in {txt_dir}!")
        return {}

    print(f"Found {len(txt_stems)} .txt files in {txt_dir}")

    filtered_val_path: Optional[str] = None
    try:
        filtered_val_path = prepare_dataloader(validator, args, set(txt_stems.keys()))

        # Initialize metrics and class names
        validator.init_metrics(model.model)

        print(f"Starting evaluation from {txt_dir}...")
        bar = TQDM(validator.dataloader, desc=validator.get_desc(), total=len(validator.dataloader))

        for batch in bar:
            # Preprocess batch (moves to device, normalizes images)
            batch = validator.preprocess(batch)

            # Load predictions from .txt files
            batch_preds = []

            for si, im_file in enumerate(batch["im_file"]):
                stem = Path(im_file).stem

                if stem not in txt_stems:
                    print(f"警告: Missing txt file for: {stem}")
                    batch_preds.append(empty_prediction(validator.device))
                    continue

                txt_path = txt_stems[stem]
                pred_dict = parse_txt_detection(txt_path, validator.device)

                # Prepare batch info for this image to get coordinate scaling parameters
                pbatch = validator._prepare_batch(si, batch)

                # CRITICAL: Scale predictions from original image space back to model input space
                pred_dict = scale_boxes_to_model_space(pred_dict, pbatch)

                if pred_dict["conf"].shape[0] > args.max_det:
                    keep = torch.argsort(pred_dict["conf"], descending=True)[:args.max_det]
                    pred_dict = {
                        "bboxes": pred_dict["bboxes"][keep],
                        "conf": pred_dict["conf"][keep],
                        "cls": pred_dict["cls"][keep],
                        "extra": pred_dict["extra"][keep],
                    }

                batch_preds.append(pred_dict)

            # Update metrics - detections are already post-processed (NMS done)
            validator.update_metrics(batch_preds, batch)

        return finalize_and_save_results(validator)
    finally:
        if filtered_val_path and Path(filtered_val_path).exists():
            Path(filtered_val_path).unlink()


# ============================================================================
# 主函数
# ============================================================================

def main() -> Optional[Dict[str, Any]]:
    parser = argparse.ArgumentParser(
        description="YOLOv9s 评估脚本 - 支持自动检测 bin/txt 输入格式"
    )

    # 输入输出参数
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入目录路径（包含 .bin 或 .txt 文件）",
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["auto", "bin", "txt"],
        default="auto",
        help="评估模式：auto=自动检测, bin=bin文件, txt=txt文件（默认: auto）",
    )
    # 模型和数据参数
    parser.add_argument(
        "--model",
        type=str,
        default="yolov9s.pt",
        help="模型文件路径（默认: yolov9s.pt）",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="coco.yaml",
        help="数据集配置文件路径（默认: coco.yaml）",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="批大小（默认: 16）",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="输入图像大小（默认: 640）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="设备（cpu 或 cuda，默认: cpu）",
    )

    # txt 模式特定参数
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="置信度阈值（txt 模式，默认: 0.001）",
    )
    parser.add_argument(
        "--max_det",
        type=int,
        default=300,
        help="每张图最大检测数（默认: 300）",
    )

    args = parser.parse_args()

    # 统一校验输入目录
    if not Path(args.input).exists():
        print(f"错误: 输入目录不存在: {args.input}")
        raise SystemExit(1)

    # 检测或确认输入类型
    if args.mode == "auto":
        input_type = detect_input_type(args.input)
        if input_type is None:
            print(f"错误: 无法在 {args.input} 中找到 .bin 或 .txt 文件")
            raise SystemExit(1)
        print(f"自动检测到输入类型: {input_type}")
    else:
        input_type = args.mode

    # 执行评估
    print("\n" + "=" * 60)
    print(f"评估模式: {input_type}")
    print(f"输入目录: {args.input}")
    print(f"模型: {args.model}")
    print(f"数据集: {args.data}")
    print("=" * 60 + "\n")

    if input_type == "bin":
        stats = evaluate_from_bin(args)
    else:  # txt
        stats = evaluate_from_txt(args)

    return stats


if __name__ == "__main__":
    main()
