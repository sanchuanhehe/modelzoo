import yaml
import json
import argparse
import re
import os
import fnmatch
import cv2

import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path

def find_matching_key_regex(annotations, filename):
    for key in annotations.keys():
        if key.startswith("images/") and f"/{filename}" in key:
            return key
    return None

def evaluate(opt):
    print("Evaluating accuracy...")

    with open('../data/DA-2K/annotations.json', 'r') as f:
        annotations = json.load(f)

    with open('../data/file_list.json', 'r') as f:
        file_list_data = json.load(f)
        lines = [item[0] for item in file_list_data.get('fileList', [])]

    points_num = 0
    acc_num = 0
    for i in tqdm(range(len(lines))):
        img_rel_path = lines[i]
        out_path = Path(img_rel_path).stem
        file_path = out_path
        out_filepath = f"{opt.output}/{file_path}_0.bin"

        # Read the original image to get its shape
        original_img = cv2.imread(img_rel_path)
        if original_img is None:
            print(f"Failed to read image: {img_rel_path}")
            continue
        h, w = original_img.shape[:2]

        # Load the inference result
        try:
            inference_result = np.fromfile(out_filepath, dtype=np.float32)
        except Exception as e:
            print(f"Failed to read {out_filepath}: {e}")
            continue

        if inference_result.size != h * w:
            print(f"Shape mismatch for {out_filepath}. Expected {h*w} but got {inference_result.size}")
            continue

        depth = inference_result.reshape((h, w))

        matching_key = find_matching_key_regex(annotations, out_path)
        if not matching_key:
            print(f"No annotation found for {out_path}")
            continue

        points = annotations[matching_key]
        for point in points:
            points_num += 1
            point_gt1 = point['point1']
            point_gt2 = point['point2']

            # Use original h, w indices for depth array
            point_infer1 = depth[point_gt1[0], point_gt1[1]]
            point_infer2 = depth[point_gt2[0], point_gt2[1]]

            if point_infer1 > point_infer2:
                acc_num += 1

    if points_num > 0:
        accuracy = acc_num / points_num
        print(f"Accuracy: {accuracy:.4f} ({acc_num}/{points_num})")
        return accuracy
    else:
        print("No valid points found to evaluate.")
        return 0.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Accuracy Evaluation')
    parser.add_argument('--output',
                        default="../out/result/bin",
                        type=str)

    opt = parser.parse_args()

    evaluate(opt)
