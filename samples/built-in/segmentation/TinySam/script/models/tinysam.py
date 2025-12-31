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

from typing import Tuple, Union

import cv2
import numpy as np
import torch
import torch.fx
import torch.nn.functional as F
from torch import nn
from numpy.typing import NDArray
from torch import Tensor
from torchvision.transforms.functional import to_pil_image, resize

from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoderBox, PromptEncoderPoint
from .tinyvit import TinyVit


def preprocess(img: NDArray, image_size: Union[list, tuple]) -> Tuple[Tensor, float, Tuple[int, int], Tuple[int, int]]:
    # get preprocess H, W
    ori_img_shape = img.shape[:2]
    scale = min(image_size[0] / ori_img_shape[0], image_size[1] / ori_img_shape[1])
    target_img_shape = int(scale * ori_img_shape[0] + 0.5), int(scale * ori_img_shape[1] + 0.5)
    # resize
    img_resized = np.array(resize(to_pil_image(img), target_img_shape))  # type: ignore

    img_f32 = img_resized.astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    img_f32 = (img_f32 - mean) / std
    # padding
    canvas = np.zeros([image_size[0], image_size[1], 3], dtype=np.float32)
    canvas[:target_img_shape[0], :target_img_shape[1]] = img_f32
    canvas = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0)
    output_tuple = (canvas, scale, target_img_shape, ori_img_shape)
    return output_tuple


def boxes_scale(boxes, scale_h, scale_w):
    boxes[..., 0] *= scale_w
    boxes[..., 2] *= scale_w
    boxes[..., 1] *= scale_h
    boxes[..., 3] *= scale_h
    return boxes


def points_scale(points, scale_h, scale_w):
    points = points.float()
    points[..., 0] *= torch.tensor(scale_w)
    points[..., 1] *= torch.tensor(scale_h)
    return points


class SAMBoxPredictor(nn.Module):
    dummy: Tensor

    def __init__(self, image_size: Union[int, list, tuple], downsample_rate: int = 16):
        super().__init__()
        prompt_embed_dim = 256
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        self.downsample_rate = downsample_rate
        self.image_embedding_size = (
            self.image_size[0] // self.downsample_rate, self.image_size[1] // self.downsample_rate
        )
        self.image_encoder = TinyVit(3, embed_dims=[64, 128, 160, 320], depths=[2, 2, 6, 2], heads=[2, 4, 5, 10],
                                     window_sizes=[7, 7, 14, 7])
        self.prompt_encoder = PromptEncoderBox(
            prompt_embed_dim,
            self.image_embedding_size,
            self.image_size,
            mask_in_chans=16,
        )
        self.no_mask_embed = self.prompt_encoder.no_mask_embed
        self.pe_layer = self.prompt_encoder.pe_layer

        self.mask_decoder = MaskDecoder(256, self.image_size)
        self.dumb_param = self.register_buffer('dummy', torch.zeros(1), persistent=False)
        self.feature = None
        self.scale = None
        self.ori_img_shape = None
        self.target_img_shape = None
        self.scale_h = None
        self.scale_w = None

    def set_mask_decoder_image_pe(self):
        self.mask_decoder.set_image_pe(self.prompt_encoder.get_dense_pe())

    def set_mask_decoder_dense_embedding(self):
        self.mask_decoder.set_dense_embedding(self.no_mask_embed.weight.reshape(1, -1, 1, 1))

    @torch.no_grad()
    def set_image(self, img_in: NDArray):
        img, scale, target_img_shape, ori_img_shape = preprocess(img_in, self.image_size)
        img = img.to(self.dummy.device)
        self.feature = self.image_encoder(img)
        self.scale = scale
        self.ori_img_shape = ori_img_shape
        self.target_img_shape = target_img_shape
        self.scale_h = target_img_shape[0] / ori_img_shape[0]
        self.scale_w = target_img_shape[1] / ori_img_shape[1]

    @torch.no_grad()
    def predict(self, boxes: Tensor) -> Tuple[Tensor, Tensor]:
        boxes = boxes.unsqueeze(0).float()
        boxes = boxes_scale(boxes, self.scale_h, self.scale_w)
        boxes = boxes.to(self.dummy.device)
        # encode boxes
        sparse_embedding = self.prompt_encoder(boxes[:, :2], boxes[:, 2:])
        # predict mask
        iou_output, masks = self.mask_decoder(self.feature, sparse_embedding)
        # clip & resize back to original size
        masks2 = masks[..., :self.target_img_shape[0], :self.target_img_shape[1]]
        masks3 = F.interpolate(masks2, size=self.ori_img_shape, mode='bilinear', align_corners=False)
        masks3 = masks3 > 0.0
        return iou_output.squeeze(), masks3[0]

    @torch.no_grad()
    def forward(self, img: NDArray, boxes: Tensor):
        self.set_image(img)
        return self.predict(boxes)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        state_dict["pe_layer.positional_encoding_gaussian_matrix"] \
            = state_dict.get('prompt_encoder.pe_layer.positional_encoding_gaussian_matrix')
        state_dict["no_mask_embed.weight"] = state_dict.get('prompt_encoder.no_mask_embed.weight')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


class SAMPointPredictor(nn.Module):
    dummy: Tensor

    def __init__(self, image_size: Union[int, list, tuple], downsample_rate: int = 16):
        super().__init__()
        prompt_embed_dim = 256
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        self.downsample_rate = downsample_rate
        self.image_embedding_size = (
            self.image_size[0] // self.downsample_rate, self.image_size[1] // self.downsample_rate
        )
        self.image_encoder = TinyVit(3, embed_dims=[64, 128, 160, 320], depths=[2, 2, 6, 2], heads=[2, 4, 5, 10],
                                     window_sizes=[7, 7, 14, 7])
        self.prompt_encoder = PromptEncoderPoint(
            prompt_embed_dim,
            self.image_embedding_size,
            self.image_size,
            mask_in_chans=16,
        )
        self.no_mask_embed = self.prompt_encoder.no_mask_embed
        self.pe_layer = self.prompt_encoder.pe_layer

        self.mask_decoder = MaskDecoder(256, self.image_size)
        self.dumb_param = self.register_buffer('dummy', torch.zeros(1), persistent=False)
        self.feature = None
        self.scale = None
        self.ori_img_shape = None
        self.target_img_shape = None
        self.scale_h = None
        self.scale_w = None

    def set_mask_decoder_image_pe(self):
        self.mask_decoder.set_image_pe(self.prompt_encoder.get_dense_pe())

    def set_mask_decoder_dense_embedding(self):
        self.mask_decoder.set_dense_embedding(self.no_mask_embed.weight.reshape(1, -1, 1, 1))

    @torch.no_grad()
    def set_image(self, img_in: NDArray):
        img, scale, target_img_shape, ori_img_shape = preprocess(img_in, self.image_size)
        img = img.to(self.dummy.device)
        self.feature = self.image_encoder(img)
        self.scale = scale
        self.ori_img_shape = ori_img_shape
        self.target_img_shape = target_img_shape
        self.scale_h = target_img_shape[0] / ori_img_shape[0]
        self.scale_w = target_img_shape[1] / ori_img_shape[1]

    @torch.no_grad()
    def predict(self, points: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        points = points_scale(points, self.scale_h, self.scale_w)
        points = points.to(self.dummy.device)

        labels = labels.to(self.dummy.device)
        # encode boxes
        sparse_embedding = self.prompt_encoder(points, labels)
        # predict mask
        iou_output, masks = self.mask_decoder(self.feature, sparse_embedding)
        # clip & resize back to original size
        masks2 = masks[..., :self.target_img_shape[0], :self.target_img_shape[1]]
        masks3 = F.interpolate(masks2, size=self.ori_img_shape, mode='bilinear', align_corners=False)
        masks3 = masks3 > 0.0
        return iou_output.squeeze(), masks3[0]

    @torch.no_grad()
    def forward(self, img: NDArray, points: Tensor, labels: Tensor):
        self.set_image(img)
        return self.predict(points, labels)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        state_dict["pe_layer.positional_encoding_gaussian_matrix"] \
            = state_dict.get('prompt_encoder.pe_layer.positional_encoding_gaussian_matrix')
        state_dict["no_mask_embed.weight"] = state_dict.get('prompt_encoder.no_mask_embed.weight')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)