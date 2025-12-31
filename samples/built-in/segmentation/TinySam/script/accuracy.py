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

from tqdm import tqdm
import numpy as np
import torch
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util
import os
import stat
import json
import argparse
import torch.nn.functional as F

FILE_MODE = stat.S_IRUSR + stat.S_IWUSR + stat.S_IRGRP # mode is 640
FILE_ACCESS_FLAG = os.O_WRONLY | os.O_CREAT | os.O_TRUNC # writeOnly + create if not exist + truncate


def parse_args(argv=None):
    parser = argparse.ArgumentParser("MobileSAM Benchmark!")
    parser.add_argument('--image_dir', type=str, default="./data/coco/val2017",
                        help='coco image dir')
    parser.add_argument('--anno_path', type=str,
                        default="./data/coco/instances_val2017.json",
                        help='path to val2017 annotation json file')
    parser.add_argument('--vit_det_file_path', type=str, default="./data/eval/coco_instances_results_vitdet.json",
                        help='path to vitdet detection results json file')
    parser.add_argument('--res_dir', type=str, default="./out/result",
                        help='path to result dir')
    arguments = parser.parse_args(argv)
    return arguments


def eval(args):
    image_dir = args.image_dir
    anno_path = args.anno_path
    vit_det_file_path = args.vit_det_file_path
    res_dir =args.res_dir
    with open(vit_det_file_path) as f:
        res = json.load(f)

    pre_img_id = 0
    for i in tqdm(range(len(res))):
        res_ins = res[i]
        img_id = res_ins['image_id']
        img_file_name = f'{img_id:012d}' + '.jpg'

        if pre_img_id != img_id:
            image = cv2.imread(os.path.join(image_dir, img_file_name))
            ori_img_shape = image.shape[:2]
            scale = min(448 / ori_img_shape[0], 448 / ori_img_shape[1])
            target_img_shape = int(scale * ori_img_shape[0] + 0.5), int(scale * ori_img_shape[1] + 0.5)
            pre_img_id = img_id

        out2_name = os.path.join(res_dir, str(i))
        masks_np = np.fromfile(out2_name, dtype=np.float32)
        masks_np = masks_np.reshape(1, 1, 448, 448)
        masks = torch.from_numpy(masks_np)

        masks2 = masks[..., :target_img_shape[0], :target_img_shape[1]]
        masks3 = F.interpolate(masks2, size=ori_img_shape, mode='bilinear', align_corners=False)
        masks3 = masks3 > 0.0

        masks = masks3[0]

        new_seg = mask_util.encode(np.array(masks.cpu().numpy()[0],
                                order="F", dtype="uint8"))
        new_seg["counts"] = new_seg["counts"].decode("utf-8")
        res[i]["segmentation"] = new_seg

    for c in res:
        c.pop("bbox", None)
    save_res_json_file = os.path.join("out", 'res_tinysam.json')

    with os.fdopen(os.open(save_res_json_file, FILE_ACCESS_FLAG, FILE_MODE), 'w') as f:
        json.dump(res, f)

    gt = COCO(anno_path)
    dt = gt.loadRes(res)
    eval = COCOeval(gt, dt, "segm")
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

if __name__ == "__main__":
    args = parse_args()
    eval(args)
