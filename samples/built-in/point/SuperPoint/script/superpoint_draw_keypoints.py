# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm


def pts_to_bbox(points, patch_size):
    shift_l = (patch_size + 1) / 2
    shift_r = patch_size - shift_l
    pts_l = points - shift_l
    pts_r = points + shift_r + 1
    bbox = torch.stack((pts_l[:, 1], pts_l[:, 0], pts_r[:, 1], pts_r[:, 0]), dim=1)
    return bbox

def _roi_pool(pred_heatmap, rois, patch_size=8):
    from torchvision.ops import roi_pool
    return roi_pool(pred_heatmap, rois.float(), (patch_size, patch_size), spatial_scale=1.0)

def norm_patches(patches):
    patch_size = patches.shape[-1]
    patches = patches.view(-1, 1, patch_size * patch_size)
    d = torch.sum(patches, dim=-1).unsqueeze(-1) + 1e-6
    patches = patches / d
    return patches.view(-1, 1, patch_size, patch_size)

def extract_patch_from_points(heatmap, points, patch_size=5):
    if isinstance(heatmap, torch.Tensor):
        heatmap = to_numpy(heatmap)
    
    heatmap = heatmap.squeeze()  # [H, W]
    pad_size = int(patch_size / 2)
    heatmap = np.pad(heatmap, pad_size, 'constant')
    
    patches = []
    for i in range(points.shape[0]):
        x = int(round(points[i, 0]))
        y = int(round(points[i, 1]))
        patch = heatmap[y:y + patch_size, x:x + patch_size]
        patches.append(patch)
    
    return patches

def to_numpy(tensor):
    if tensor is None:
        return None
    return tensor.detach().cpu().numpy()

class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, s_height, s_width, s_depth)
        return output.permute(0, 3, 1, 2)

# --- 核心后处理类 ---

class SuperPointPostProcessor:
    def __init__(self, nms_dist=4, conf_thresh=0.015, border_remove=4, device='cpu'):
        self.device = device
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.cell_size = 8  # 固定参数，与模型设计相关
        self.border_remove = border_remove
        
        self.heatmap = None
        self.keypoints = None
        self.descriptors = None
        self.coarse_desc = None

    def nms_fast(self, corners, H, W):
        grid = np.zeros((H, W), dtype=int)
        inds = np.zeros((H, W), dtype=int)
        
        inds1 = np.argsort(-corners[2, :])
        corners = corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)

        if rcorners.shape[1] == 0:
            return np.zeros((3, 0))
        if rcorners.shape[1] == 1:
            return np.vstack((rcorners, corners[2])).reshape(3, 1)

        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i

        pad = self.nms_dist
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')

        for i, rc in enumerate(rcorners.T):
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:
                grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
                grid[pt[1], pt[0]] = -1

        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        
        return out[:, np.argsort(-out[2, :])]

    def subpixel_refinement(self, keypoints, patch_size=5):
        pts = keypoints[0].transpose().copy()  # [N, 3]
        patches = extract_patch_from_points(self.heatmap, pts[:, :2], patch_size=patch_size)
        patches = np.stack(patches)
        
        patches_torch = torch.tensor(patches, dtype=torch.float32).unsqueeze(0).to(self.device)
        patches_torch = norm_patches(patches_torch)
        patches_torch = torch.log(patches_torch + 1e-6) # 使用log确保数值稳定
        
        dxdy = tgm.contrib.SpatialSoftArgmax2d(normalized_coordinates=False)(patches_torch)
        pts[:, :2] += dxdy.cpu().numpy().squeeze() - patch_size // 2
        
        return [pts.transpose().copy()]

    def extract_descriptors(self, keypoints):
        if keypoints[0].shape[1] == 0:
            return [np.zeros((256, 0))]
            
        H, W = self.coarse_desc.shape[2] * self.cell_size, self.coarse_desc.shape[3] * self.cell_size
        samp_pts = torch.from_numpy(keypoints[0][:2, :].copy()).to(self.device)
        
        # 归一化坐标到[-1, 1]范围
        samp_pts[0, :] = (samp_pts[0, :] / (W / 2.)) - 1.
        samp_pts[1, :] = (samp_pts[1, :] / (H / 2.)) - 1.
        
        samp_pts = samp_pts.transpose(0, 1).contiguous().view(1, 1, -1, 2).float()
        desc = F.grid_sample(self.coarse_desc, samp_pts, align_corners=True)
        
        desc = desc.data.cpu().numpy().reshape(256, -1)
        return [desc / np.linalg.norm(desc, axis=0, keepdims=True)]

    def load_bin_results(self, semi_path, desc_path):
        semi = np.fromfile(semi_path, dtype=np.float32).reshape(1, 65, 30, 40)
        desc = np.fromfile(desc_path, dtype=np.float32).reshape(1, 256, 30, 40)
        
        self.coarse_desc = torch.tensor(desc).to(self.device)
        return torch.tensor(semi).to(self.device)

    def generate_heatmap(self, semi):
        dense = nn.functional.softmax(semi, dim=1)
        nodust = dense[:, :-1, :, :]
        
        depth2space = DepthToSpace(8)
        self.heatmap = depth2space(nodust)
        return to_numpy(self.heatmap)

    def detect_keypoints(self):
        heatmap_np = to_numpy(self.heatmap)
        self.keypoints = [
            self.nms_fast(
                self._extract_initial_keypoints(h),
                h.shape[1],
                h.shape[2]
            ) for h in heatmap_np
        ]
        return self.keypoints

    def _extract_initial_keypoints(self, heatmap):
        heatmap = heatmap.squeeze()
        H, W = heatmap.shape
        ys, xs = np.where(heatmap >= self.conf_thresh)
        
        if len(xs) == 0:
            return np.zeros((3, 0))
            
        pts = np.zeros((3, len(xs)))
        pts[0, :] = xs
        pts[1, :] = ys
        pts[2, :] = heatmap[ys, xs]
        
        bord = self.border_remove
        toremove = np.logical_or(
            np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord)),
            np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        )
        return pts[:, ~toremove]

    def process_descriptors(self):
        self.descriptors = self.extract_descriptors(self.keypoints)
        return self.descriptors


# --- 主流程 ---

def main(args):
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    
    # 根据参数选择设备
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    logging.info(f"Processing on device: {device}")
    
    # 打印当前使用的配置，方便调试
    logging.info("--- Running with the following parameters ---")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")
    logging.info("--------------------------------------------")
    
    # 使用命令行参数初始化后处理器
    processor = SuperPointPostProcessor(
        nms_dist=args.nms_dist,
        conf_thresh=args.confidence_threshold,
        border_remove=args.border_remove,
        device=device
    )
    
    try:
        # 核心处理步骤
        semi = processor.load_bin_results(args.semi_bin, args.desc_bin)
        processor.generate_heatmap(semi)
        processor.detect_keypoints()
        
        if processor.keypoints and processor.keypoints[0].shape[1] > 0:
            processor.subpixel_refinement(processor.keypoints)
            processor.process_descriptors()
        else:
            logging.warning("No keypoints detected, skipping subpixel refinement and descriptor extraction.")
            return
        
        # 可视化结果
        h,w = 240,320
        img_gray = np.fromfile(args.image, dtype=np.float32).reshape(h, w) # 板端预处理后的
        # 1. 将浮点型数据归一化到0-255并转换为uint8
        img = ((img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-8) * 255).astype(np.uint8)

        # img = cv2.imread(args.image)
        if img is None:
            raise FileNotFoundError(f"Original image not found at: {args.image}")
        
        for x, y, _ in processor.keypoints[0].T:
            cv2.circle(img, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        cv2.imwrite(args.output, img)
        logging.info(f"Successfully processed!")
        logging.info(f"  - Detected keypoints: {processor.keypoints[0].shape[1]}")
        logging.info(f"  - Result saved to: {args.output}")
        
    except FileNotFoundError as e:
        logging.error(f"File not found error: {e}")
        logging.error("Please check the file paths in the script or provide them via command line arguments.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SuperPoint Post-Processing (No Config File, Debug Friendly)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter) # 显示默认值
    
    # --- 调试友好：为所有参数设置了默认值 ---
    # 输入输出参数
    parser.add_argument("--semi_bin", type=str, 
                        default="./out/result_bak/bin/v_adam_1_0.bin",
                        help="Path to the semi (heatmap) output .bin file.")
    parser.add_argument("--desc_bin", type=str, 
                        default="./out/result_bak/bin/v_adam_1_1.bin",
                        help="Path to the descriptor output .bin file.")
    parser.add_argument("--image", type=str, 
                        default="./data/hpatches_preprocessed/v_adam_1.bin", 
                        help="Path to the preprocessed image for visualization.")
    parser.add_argument("--output", type=str, 
                        default="./out/v_adam_1_keypoints.jpg",
                        help="Path to save the visualized result image.")
    
    # 算法参数
    parser.add_argument("--nms_dist", type=int, default=4,
                        help="Non Maximum Suppression (NMS) distance in pixels.")
    parser.add_argument("--confidence_threshold", type=float, default=0.015,
                        help="Confidence threshold for keypoint detection. Lower values detect more points.")
    parser.add_argument("--border_remove", type=int, default=4,
                        help="Number of pixels to exclude from the image borders.")
    
    # 设备参数
    parser.add_argument("--force_cpu", action='store_true',
                        help="Force using CPU even if CUDA is available. Useful for debugging.")
    
    args = parser.parse_args()
    
    # 在main函数中进行实际的路径检查和处理
    main(args)