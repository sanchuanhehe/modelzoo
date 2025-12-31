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

from typing import Tuple

import torch
import torch.fx
import torch.nn as nn

from TinySAM.tinysam.modeling.prompt_encoder import PositionEmbeddingRandom, PromptEncoder


class PicoPositionEmbeddingRandom(PositionEmbeddingRandom):
    def pe_encoding(self, coords):
        return self._pe_encoding(coords)


class PromptEncoderBox(PromptEncoder):
    def __init__(self, embed_dim: int, image_embedding_size: Tuple[int, int], input_image_size: Tuple[int, int],
                 mask_in_chans: int, activation=nn.GELU, num_point_embeddings: int = 4):
        super().__init__(embed_dim, image_embedding_size, input_image_size,
                         mask_in_chans, activation)
        self.num_point_embeddings = num_point_embeddings  # pos/neg point + 2 box corners
        self.pe_layer = PicoPositionEmbeddingRandom(embed_dim // 2)
        self.point_embeddings = nn.ModuleList([nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)])

    def forward(self, box_lt: torch.Tensor, box_rb: torch.Tensor) -> torch.Tensor:
        box_lt = box_lt + 0.5  # Shift to center of pixel
        box_rb = box_rb + 0.5  # Shift to center of pixel
        if isinstance(self.input_image_size, int):
            box_lt = box_lt / self.input_image_size
            box_rb = box_rb / self.input_image_size
        else:
            box_lt = box_lt / self.input_image_size[1]
            box_rb = box_rb / self.input_image_size[0]

        corner_embedding_lefttop = self.pe_layer.pe_encoding(box_lt)
        corner_embedding_rightbottom = self.pe_layer.pe_encoding(box_rb)
        corner_embedding_lefttop += self.point_embeddings[2].weight
        corner_embedding_rightbottom += self.point_embeddings[3].weight
        corner_embedding = torch.concat([corner_embedding_lefttop.unsqueeze(1),
                                         corner_embedding_rightbottom.unsqueeze(1)], dim=1)
        return corner_embedding


class PromptEncoderPoint(PromptEncoder):
    def __init__(self, embed_dim: int, image_embedding_size: Tuple[int, int], input_image_size: Tuple[int, int],
                 mask_in_chans: int, activation=nn.GELU):
        super().__init__(embed_dim, image_embedding_size, input_image_size,
                         mask_in_chans, activation)
        self.num_point_embeddings = 4  # pos/neg point + 2 box corners
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.embed_dim = embed_dim
        self.not_a_point_embed = nn.Embedding(1, self.embed_dim)
        self.mask_input_size = (4 * self.image_embedding_size[0], 4 * self.image_embedding_size[1])
        self.no_mask_embed = nn.Embedding(1, self.embed_dim)

        self.pe_layer = PicoPositionEmbeddingRandom(self.embed_dim // 2)

        self.point_embeddings = nn.ModuleList([nn.Embedding(1, self.embed_dim) for i in range(self.num_point_embeddings)])


    def forward(self, point_coords: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        point_coords = point_coords + 0.5
        point_coords = point_coords / torch.tensor(self.input_image_size[0])
        point_embedding = self.pe_layer.pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.not_a_point_embed.weight * (
                point_labels == -1
        )

        for i in range(self.num_point_embeddings):
            point_embedding = point_embedding + self.point_embeddings[
                i
            ].weight * (point_labels == i)
        point_embedding = point_embedding.reshape(1, 5, 256)
        return point_embedding
