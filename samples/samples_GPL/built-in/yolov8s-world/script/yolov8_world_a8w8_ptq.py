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


import os
import ssl
import stat
import argparse
import json
from pathlib import Path

import torch
import torch.fx
import hotwheels.amct_pytorch as amct

from ultralytics.utils.checks import check_yaml, print_args
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import LOGGER,TQDM,callbacks,colorstr
from ultralytics.utils.ops import Profile
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics import YOLOWorld
from deploy.utils import convert_to_deploy, set_classes
from deploy import modules as deploy_modules


PATH, _ = os.path.split(os.path.realpath(__file__))
WORK_DIR = os.path.join(PATH, "work_dir")
os.makedirs(WORK_DIR, exist_ok=True)
FILE_MODE = stat.S_IRUSR + stat.S_IWUSR + stat.S_IRGRP
FILE_ACCESS_FLAG = os.O_WRONLY | os.O_CREAT | os.O_TRUNC

ssl._create_default_https_context = ssl._create_unverified_context


class QuantDetectionValidator(DetectionValidator):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.data = check_det_dataset(self.args.data)
        self.args.rect = False
        self.dataset_path = self.data.get(self.args.split)
        self.stride = 32
        self.args.batch = 16
        self.dataloader = self.get_dataloader(self.dataset_path, self.args.batch)
        self.device = model.args.get('device')
        self.names = model.names
        self.model_name = Path(self.args.model).stem
        dir_name = self.model_name
        if deploy_modules.TEXT_OFFLINE:
            dir_name = f"{dir_name}_text_offline"
        if deploy_modules.CUSTOM_RPN:
            dir_name = f"{dir_name}_with_NMS"
        if os.path.basename(self.data["yaml_file"]) == "lvis.yaml":
            dir_name = f"{dir_name}_lvis"
        self.save_dir = Path(os.path.join(WORK_DIR, dir_name))
        LOGGER.info(f"==> [AMCT] Save dir: {self.save_dir}")

    @torch.no_grad()
    def __call__(self, net, text_features=None):
        # === 基本状态初始化 ===
        self.training = False
        net.names = self.names
        callbacks.add_integration_callbacks(self)

        # 禁用 half 精度（验证阶段统一用 float32）
        self.args.half = False

        # CPU / MPS 下关闭多线程 dataloader
        if self.device.type in ("cpu", "mps"):
            self.args.workers = 0

        # 切换模型到 eval 模式
        net.eval()

        # === 验证开始回调 ===
        self.run_callbacks("on_val_start")

        # 四段耗时统计（预处理 / 推理 / 预留 / 后处理）
        timers = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )

        # 构建进度条
        data_iter = self.dataloader
        progress = TQDM(data_iter, desc=self.get_desc(), total=len(data_iter))

        # 初始化评估指标
        self.init_metrics(net)

        # json 结果缓存
        self.jdict = []

        # === 主循环 ===
        for step_idx, batch_data in enumerate(progress):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = step_idx

            # ---- 数据预处理 ----
            with timers[0]:
                batch_data = self.preprocess(batch_data)

            # ---- 前向推理 ----
            with timers[1]:
                input_imgs = batch_data["img"]

                if text_features is not None:
                    infer_out = net(input_imgs, text_features)
                else:
                    infer_out = net(input_imgs)

            # ---- 后处理 ----
            with timers[3]:
                infer_out = self.postprocess(infer_out)

            # ---- 更新指标 ----
            self.update_metrics(infer_out, batch_data)

            # ---- 可视化（仅前3个 batch）----
            if self.args.plots and step_idx < 3:
                self.plot_val_samples(batch_data, step_idx)
                self.plot_predictions(batch_data, infer_out, step_idx)

            self.run_callbacks("on_val_batch_end")

        # === 统计指标 ===
        results = self.get_stats()
        self.check_stats(results)

        # 计算平均耗时（ms / sample）
        dataset_size = len(self.dataloader.dataset)
        self.speed = dict(
            zip(
                self.speed.keys(),
                (t.t / dataset_size * 1e3 for t in timers)
            )
        )

        # 收尾处理
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")

        # 打印速度日志
        LOGGER.info(
            "Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess"
            % tuple(self.speed.values())
        )

        # === 保存 JSON 结果 ===
        if self.args.save_json and self.jdict:
            save_file = os.path.join(self.save_dir, "predictions.json")

            fd = os.open(save_file, FILE_ACCESS_FLAG, FILE_MODE)
            with os.fdopen(fd, "w") as fp:
                LOGGER.info(f"Saving {fp.name}...")
                json.dump(self.jdict, fp)

            # COCO 等评估
            results = self.eval_json(results)

        # 输出路径提示
        if self.args.plots or self.args.save_json:
            LOGGER.info(f"Results saved to {self.save_dir}")

        return results


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = convert_to_deploy(set_classes(args.data, YOLOWorld(args.model).to(device)))

    validator = QuantDetectionValidator(model, args = vars(args))
    save_dir, model_name = validator.save_dir, validator.model_name
    text_feature = model.txt_feats.float().to(device)
    
    LOGGER.info(f"==> [AMCT] create quant config..")
    config_file = f'{save_dir}/qconfig.json'
    amct.create_quant_config_fx(config_file, model, f"./data/{model_name}_mix_quantize.yml")

    LOGGER.info(f"==> [AMCT] create quantize model..")
    quant_model = amct.create_quant_model_fx(config_file, model)

    LOGGER.info(f"==> [AMCT] do calibration..")
    amct.enable_quantization(quant_model, calibration=True)
    for data in validator.dataloader:
        imgs = data["img"].to(device, non_blocking=True).float() / 255
        quant_model(imgs, text_feature)
        break

    LOGGER.info(f"==> [AMCT] save quant model...")
    torch.save(quant_model.state_dict(), f"{save_dir}/{model_name}_quant_model.pt")
    text_feature.detach().cpu().numpy().tofile(f"{save_dir}/text_feature.bin")
    onnx_export_setting = {'input_names': ['images', 'text_feature'], 'output_names': ['output']}
    amct.save_quant_model_fx(quant_model, f"{save_dir}/{model_name}", 
                             (torch.randn(1, 3, args.imgsz, args.imgsz).to(device), text_feature), 
                             onnx_export_setting=onnx_export_setting)
    



def parse_opt(default_pt="pretrain/yolov8m-worldv2.pt", default_data="coco.yaml"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=default_data, type=str, help="data name")
    parser.add_argument("--model", default=default_pt, type=str, help="model name")
    parser.add_argument("--imgsz", "--img-size", "--img", default=640, type=int, help="imgsz")
    parser.add_argument("--custom_rpn", action="store_true", help="custom rpn")
    parser.add_argument("--custom_rpn_topk", default=300, type=int, help="custom rpn topk")
    parser.add_argument("--text_offline", action="store_true", help="text offline threshold")

    args = parser.parse_args()
    deploy_modules.CUSTOM_RPN = args.__dict__.pop("custom_rpn")
    deploy_modules.CUSTOM_RPN_TOPK = args.__dict__.pop("custom_rpn_topk")
    deploy_modules.TEXT_OFFLINE = args.__dict__.pop("text_offline")

    args.task = "detect"
    args.rect = True
    args.mode = "val"
    args.data = check_yaml(args.data)
    print_args(vars(args))
    return args


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
