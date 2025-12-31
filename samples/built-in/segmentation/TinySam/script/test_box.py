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

import os
import argparse
from pathlib import Path

import cv2
import torch
import numpy as np

from models.tinysam import SAMBoxPredictor
import draw_util

try:
    import hotwheels.amct_pytorch as amct
except ImportError as e:
    print('Failed to import the AMCT.')

PATH, _ = os.path.split(os.path.realpath(__file__))


def load_state_dict(resume):
    state_dict = torch.load(resume, map_location='cpu')
    new_state_dict = {}
    for k, v in state_dict.items():
        valid_key = (
                'image_encoder.norm_head' not in k and
                'image_encoder.head' not in k and
                'image_encoder.neck_64to32' not in k
        )
        if valid_key:
            new_state_dict[k] = v
    return new_state_dict


def create_model(resume, device, image_size=(448, 448)):
    predictor = SAMBoxPredictor(image_size)
    state_dict = load_state_dict(resume)
    predictor.load_state_dict(state_dict)
    predictor.to(device)
    predictor.set_mask_decoder_image_pe()
    predictor.set_mask_decoder_dense_embedding()
    predictor.eval()
    return predictor


def image_encoder():
    args = parse_args([])
    predictor = create_model(args.resume, args.device)
    return predictor.image_encoder


def prompt_encoder():
    args = parse_args([])
    predictor = create_model(args.resume, args.device)
    return predictor.prompt_encoder


def mask_decoder():
    args = parse_args([])
    predictor = create_model(args.resume, args.device)
    return predictor.mask_decoder


def load_image_and_boxes(image_path, boxes, device):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = torch.tensor(boxes).to(device)
    return img, boxes


def restore_quant_model(model, restore_path):
    print('==> [AMCT]: restore quant model..')
    model.image_encoder = amct.restore_quant_model_fx(
        '{}/quant_model/image_encoder/image_encoder_config.json'.format(restore_path), model.image_encoder,
        '{}/quant_model/image_encoder/image_encoder_quantized.pt'.format(restore_path)
    )

    model.prompt_encoder = amct.restore_quant_model_fx(
        '{}/quant_model/prompt_encoder/prompt_encoder_config.json'.format(restore_path), model.prompt_encoder,
        '{}/quant_model/prompt_encoder/prompt_encoder_quantized.pt'.format(restore_path)
    )

    model.mask_decoder = amct.restore_quant_model_fx(
        '{}/quant_model/mask_decoder/mask_decoder_config.json'.format(restore_path), model.mask_decoder,
        '{}/quant_model/mask_decoder/mask_decoder_quantized.pt'.format(restore_path)
    )
    return model


def quant_model_enable(quant_model, calibration=False, fake_quant=False, real_quant=False):
    amct.enable_quantization(quant_model.image_encoder, calibration, fake_quant, real_quant)
    amct.enable_quantization(quant_model.prompt_encoder, calibration, fake_quant, real_quant)
    amct.enable_quantization(quant_model.mask_decoder, calibration, fake_quant, real_quant)


def parse_args(argv=None):
    parser = argparse.ArgumentParser("MobileSAM Demo!")
    construct_args_parser(parser)
    arguments = parser.parse_args(argv)
    arguments.device = f'cuda:{arguments.gpu_id}' if torch.cuda.is_available() else "cpu"
    if isinstance(arguments.test_box[0], float):
        arguments.test_box = np.asarray(arguments.test_box).reshape(-1, 4)
    print(f"==> [AMCT]: use {arguments.device}")
    arguments.work_path = os.path.join(PATH, '{}_box_results'.format(Path(arguments.resume).stem))
    os.makedirs(arguments.work_path, exist_ok=True)
    print(f"==> [AMCT]: work dir {arguments.work_path }")
    return arguments


def args_check(args):
    """Verify the validity of input parameters
    """
    if not os.path.exists(args.resume):
        raise FileNotFoundError(f'No such file or directory: {args.resume}')
    if not os.path.exists(args.test_image):
        raise FileNotFoundError(f'No such file or directory: {args.test_image}')


def construct_args_parser(parser):
    parser.add_argument('--resume', dest='resume', default='./model/tinysam.pth',
                        help='resume from checkpoint')
    parser.add_argument('--test_image', type=str, default='./data/images/test_image.jpg',
                        metavar="FILE", help='path to image file')
    parser.add_argument('--test_box', help='boxes of targets', type=float, nargs='+',
                        default=[
                            [15.799736, 131.609863, 184.381500, 192.262024],
                            [269.248138, 13.451981, 410.820587, 201.923019],
                            [0.000000, 280.281860, 220.173828, 438.155640],
                            [270.982300, 279.010773, 401.805176, 442.789215],                           
                        ])
    
    parser.add_argument('--image_size', dest='image_size', help='image size', 
                        default=(448, 448), type=int, nargs='+')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use', default=0, type=int)


def main():
    args = parse_args()
    predictor = create_model(args.resume, args.device, args.image_size)

    img, boxes = load_image_and_boxes(args.test_image, args.test_box, args.device)
    image_name = os.path.basename(args.test_image)
    draw_util.draw_boxes(img.copy(), boxes, image_name, args.work_path)
    quant_predictor = restore_quant_model(predictor, args.work_path)

    for status in ['float', 'fakequant', 'realquant']:
        if status != 'float':
            predictor = quant_predictor
            quant_model_enable(predictor, fake_quant=(status == 'fakequant'), real_quant=(status == 'realquant'))

        last_mask = torch.zeros((1, img.shape[0], img.shape[1])).to(args.device)
        print('==> [AMCT]: model predict...')
        predictor.set_image(img)
        for box in boxes:
            ious, mask = predictor.predict(box.clone())
            print('score: {}'.format(ious.cpu().numpy()))
            last_mask = torch.clamp(last_mask + mask, 0, 1)
        last_mask = last_mask.cpu().numpy()
        result_image = os.path.join(args.work_path, f'{status}_{image_name}')
        draw_util.draw_mask(img, last_mask, result_image)
        print('mask save to {}'.format(result_image))


if __name__ == '__main__':
    main()
