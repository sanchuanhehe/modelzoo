# Copyright (c) ModelZoo. 2026-2026. All rights reserved.
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSORT 模型导出脚本
将 YOLOv5s 检测模型和自定义 ReID 特征提取网络导出为 ONNX 格式。

用法 (请在 deepsort_env conda 环境下运行):
    conda activate deepsort_env
    python export_onnx.py [--output_dir OUTPUT_DIR]

生成文件:
    - onnx_output/yolov5s.onnx        (输入: 1x3x640x640)
    - onnx_output/reid_net.onnx       (输入: 1x3x128x64, 输出: 1x512)
"""

import os
import sys
import argparse
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# YOLOv5s 导出
# ============================================================
def export_yolov5(output_dir: str, opset: int = 11):
    print("=" * 60)
    print("[1/2] 开始导出 YOLOv5s 检测模型...")
    print("=" * 60)

    yolo_dir = os.path.join(SCRIPT_DIR, "detector", "YOLOv5")
    yolo_weight = os.path.join(yolo_dir, "yolov5s.pt")

    if not os.path.exists(yolo_weight):
        print(f"[错误] YOLOv5s 权重文件不存在: {yolo_weight}")
        return False

    output_path = os.path.join(output_dir, "yolov5s.onnx")

    # ---------------------------------------------------------
    # 核心修复点：绕过 YOLOv5 循环导入 (The Circular Import Fix)
    #
    # 当把 yolo_dir 加入 sys.path[0] 后，`import detector` 
    # 会触发连环包裹文件导入。由于模型里硬编码了：
    # `from detector.YOLOv5.utils.datasets import ...`
    # 我们在这短暂的瞬间将所有干涉路径解析的包裹文件全部隐藏，
    # 使其降级为纯粹的普通命名空间目录。
    # ---------------------------------------------------------
    detector_init_file = os.path.join(SCRIPT_DIR, "detector", "__init__.py")
    detector_init_bak = os.path.join(SCRIPT_DIR, "detector", "__init__.py.bak")
    
    yolo_init_file = os.path.join(yolo_dir, "__init__.py")
    yolo_init_bak = os.path.join(yolo_dir, "__init__.py.bak")
    
    yolo_detector_file = os.path.join(yolo_dir, "detector.py")
    yolo_detector_bak = os.path.join(yolo_dir, "detector.py.bak")

    # 临时备份冲突文件
    try:
        if os.path.exists(detector_init_file):
            shutil.move(detector_init_file, detector_init_bak)
        if os.path.exists(yolo_init_file):
            shutil.move(yolo_init_file, yolo_init_bak)
        if os.path.exists(yolo_detector_file):
            shutil.move(yolo_detector_file, yolo_detector_bak)
            
        # 把 YOLO_DIR 加入 sys.path 以便能裸导 `models`
        if yolo_dir not in sys.path:
            sys.path.insert(0, yolo_dir)

        import torch
        from models.experimental import attempt_load

        device = torch.device("cpu")
        model = attempt_load(yolo_weight, map_location=device)
        model.eval()

        for k, m in model.named_modules():
            m._non_persistent_buffers_set = set()  # pytorch 1.6 遗留问题的保护特性
            if isinstance(m, torch.nn.Upsample):
                m.recompute_scale_factor = None
            if hasattr(m, "onnx_dynamic"):
                m.onnx_dynamic = False
            if hasattr(m, "export"):
                m.export = True
            if type(m).__name__ == "Detect":
                m.inplace = False
                m.training = True  # 核心：强强制将该层切换到训练模式，阻止它执行 Tensor 的 NPU 不兼容重组操作

        dummy_input = torch.randn(1, 3, 640, 640, device=device)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            verbose=False,
            opset_version=opset,
            input_names=["images"],
            output_names=["output0", "output1", "output2"],
            do_constant_folding=True,
        )

        print(f"[成功] YOLOv5s ONNX 已保存至: {output_path}")
        print(f"       输入形状: 1x3x640x640, opset: {opset}")

        # 尝试简化
        try:
            import onnxsim, onnx
            print("[信息] 正在简化 ONNX...")
            m = onnx.load(output_path)
            m, ok = onnxsim.simplify(m)
            assert ok
            onnx.save(m, output_path)
            print("[成功] 简化完毕。")
        except ImportError:
            print("[提示] 未安装 onnx-simplifier，跳过简化。")
        except Exception as e:
            print(f"[警告] 简化失败: {e}")
            
        yolo_ok = True

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[失败] YOLOv5s 导出异常: {e}")
        yolo_ok = False

    finally:
        # 恢复物理文件
        if os.path.exists(detector_init_bak):
            shutil.move(detector_init_bak, detector_init_file)
        if os.path.exists(yolo_init_bak):
            shutil.move(yolo_init_bak, yolo_init_file)
        if os.path.exists(yolo_detector_bak):
            shutil.move(yolo_detector_bak, yolo_detector_file)

    return yolo_ok


# ============================================================
# ReID 自定义 Net 导出
# ============================================================
def export_reid(output_dir: str, opset: int = 11):
    print("\n" + "=" * 60)
    print("[2/2] 开始导出 ReID 特征提取网络...")
    print("=" * 60)

    import torch

    reid_dir = os.path.join(SCRIPT_DIR, "deep_sort", "deep")
    reid_weight = os.path.join(reid_dir, "checkpoint", "ckpt.t7")

    if not os.path.exists(reid_weight):
        print(f"[错误] ReID 权重文件不存在: {reid_weight}")
        return False

    if reid_dir not in sys.path:
        sys.path.insert(0, reid_dir)

    from model import Net

    net = Net(reid=True)
    state_dict = torch.load(reid_weight, map_location="cpu")
    if isinstance(state_dict, dict) and 'net_dict' in state_dict:
        state_dict = state_dict['net_dict']
    net.load_state_dict(state_dict, strict=False)
    net.eval()

    dummy_input = torch.randn(1, 3, 128, 64)
    output_path = os.path.join(output_dir, "reid_net.onnx")

    torch.onnx.export(
        net,
        dummy_input,
        output_path,
        verbose=False,
        opset_version=opset,
        input_names=["input"],
        output_names=["features"],
        do_constant_folding=True,
    )

    print(f"[成功] ReID Net ONNX 已保存至: {output_path}")
    print(f"       输入形状: 1x3x128x64, 输出形状: 1x512, opset: {opset}")

    try:
        import onnxsim, onnx
        print("[信息] 正在简化 ONNX...")
        m = onnx.load(output_path)
        m, ok = onnxsim.simplify(m)
        assert ok
        onnx.save(m, output_path)
        print("[成功] 简化完毕。")
    except ImportError:
        print("[提示] 未安装 onnx-simplifier，跳过简化。")
    except Exception as e:
        print(f"[警告] 简化失败: {e}")

    return True


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="DeepSORT 模型 ONNX 导出工具")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(SCRIPT_DIR, "onnx_output"),
                        help="ONNX 输出目录 (默认: ./onnx_output)")
    parser.add_argument("--opset", type=int, default=11,
                        help="ONNX opset 版本 (默认: 11)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"ONNX 输出目录: {args.output_dir}\n")

    yolo_ok = export_yolov5(args.output_dir, args.opset)
    reid_ok = export_reid(args.output_dir, args.opset)

    print("\n" + "=" * 60)
    print("导出汇总:")
    print(f"  YOLOv5s  -> {'✓ 成功' if yolo_ok else '✗ 失败'}")
    print(f"  ReID Net -> {'✓ 成功' if reid_ok else '✗ 失败'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
