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
from test_box import load_image_and_boxes, create_model, construct_args_parser, quant_model_enable

import torch
import numpy as np
from models.tinysam import preprocess
import hotwheels.amct_pytorch as amct
import draw_util


PATH, _ = os.path.split(os.path.realpath(__file__))


def parse_args(argv=None):
    parser = argparse.ArgumentParser("MobileSAM Calibration!")
    construct_args_parser(parser)
    arguments = parser.parse_args(argv)
    arguments.device = f'cuda:{arguments.gpu_id}' if torch.cuda.is_available() else "cpu"
    arguments.save_path = os.path.join(PATH, '{}_box_results'.format(Path(arguments.resume).stem))
    os.makedirs(arguments.save_path, exist_ok=True)
    print(f"==> [AMCT]: save dir {arguments.save_path }")
    return arguments


if __name__ == '__main__':
    args = parse_args()

    predictor = create_model(args.resume, args.device, args.image_size)

    quant_model_path = os.path.join(args.save_path, 'quant_model')
    image_encoder_output_path = os.path.join(quant_model_path, 'image_encoder')
    prompt_encoder_output_path = os.path.join(quant_model_path, 'prompt_encoder')
    mask_decoder_output_path = os.path.join(quant_model_path, 'mask_decoder')

    print('==> [AMCT]: create quant config...')
    image_encoder_config_file = os.path.join(image_encoder_output_path, 'image_encoder_config.json')
    amct.create_quant_config_fx(image_encoder_config_file, predictor.image_encoder,
                                os.path.join(PATH, 'qconfigs/ifmr_quantize_image_encoder.yml'))

    prompt_encoder_config_file = os.path.join(prompt_encoder_output_path, 'prompt_encoder_config.json')
    amct.create_quant_config_fx(prompt_encoder_config_file, predictor.prompt_encoder,
                                os.path.join(PATH, 'qconfigs/ifmr_quantize_prompt_encoder.yml'))

    mask_decoder_config_file = os.path.join(mask_decoder_output_path, 'mask_decoder_config.json')
    amct.create_quant_config_fx(mask_decoder_config_file, predictor.mask_decoder,
                                os.path.join(PATH, 'qconfigs/ifmr_quantize_mask_decoder.yml'))

    print('==> [AMCT]: create quant model...')
    predictor.image_encoder = amct.create_quant_model_fx(image_encoder_config_file, predictor.image_encoder)
    predictor.prompt_encoder = amct.create_quant_model_fx(prompt_encoder_config_file, predictor.prompt_encoder)
    predictor.mask_decoder = amct.create_quant_model_fx(mask_decoder_config_file, predictor.mask_decoder)

    print('==> [AMCT]: do calibration...')
    quant_model_enable(predictor, calibration=True)

    img, boxes = load_image_and_boxes(args.test_image, args.test_box, args.device)

    draw_util.draw_boxes(img.copy(), boxes, args.test_image, args.save_path)
    for index in range(16):
        ious, mask = predictor(img, boxes[index % len(boxes)])

    print('==> [AMCT]: save quant model...')
    torch.save(predictor.image_encoder.state_dict(),
               os.path.join(image_encoder_output_path, 'image_encoder_quantized.pt'))
    image_encoder_export_setting = {'input_names': ['image'], 'output_names': ['image_embeddings']}
    input_data = preprocess(img, predictor.image_size)[0].to(args.device)
    np.save(os.path.join(args.save_path,
                        '{}_preprocessed.npy'.format(os.path.splitext(os.path.basename(args.test_image))[0])),
                        input_data.cpu())
    amct.save_quant_model_fx(predictor.image_encoder, os.path.join(image_encoder_output_path, 'image_encoder'),
                             input_data, onnx_export_setting=image_encoder_export_setting)

    prompt_encoder_export_setting = {'input_names': ['box_lt', 'box_rb'], 'output_names': ['sparse_embedding']}
    torch.save(predictor.prompt_encoder.state_dict(),
               os.path.join(prompt_encoder_output_path, 'prompt_encoder_quantized.pt'))
    amct.save_quant_model_fx(predictor.prompt_encoder, os.path.join(prompt_encoder_output_path, 'prompt_encoder'),
                             (boxes[0][:2].reshape(1, -1), boxes[0][2:].reshape(1, -1)),
                             onnx_export_setting=prompt_encoder_export_setting)

    torch.save(predictor.mask_decoder.state_dict(),
               os.path.join(mask_decoder_output_path, 'mask_decoder_quantized.pt'))
    mask_decoder_export_setting = {
        'input_names': ['image_embeddings', 'sparse_embedding'],
        'output_names': ['iou_predictions', 'low_res_masks']
    }
    sparse_embedding = torch.randn(1, 2, 256).to(args.device)
    amct.save_quant_model_fx(predictor.mask_decoder, os.path.join(mask_decoder_output_path, 'mask_decoder'),
                             (predictor.feature, sparse_embedding),
                             onnx_export_setting=mask_decoder_export_setting)

