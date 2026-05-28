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
import argparse
import torch
import torch.nn as nn
import numpy as np
from funasr import AutoModel
from funasr.models.fsmn_vad_streaming import encoder as fsmn_encoder


class KWSModelWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder=encoder

    def forward(self, speech_features):
        result=self.encoder(speech_features, cache=None)
        if isinstance(result, tuple):
            return result[0]
        return result

def infer_audio(args):
    print("infer:  ===== ")
    model = AutoModel(
        model=args.model_path,
        keywords=args.keywords,
        output_dir=args.output_dir,
        device=args.device,
        disable_update=True,
        trust_remote_code=True
    )

    res = model.generate(input=args.input, cache={})
    
    # Save preprocess result for calibration
    
    print(f"Inference result: {res}")
    if res and len(res) > 0:
        for item in res:
            text = item.get("text", "")
            if text.startswith("detected"):
                print(f"✓ Keyword detected: {text}")
            else:
                print(f"✗ No keyword detected: {text}")
    
def main():
    parser = argparse.ArgumentParser(description="Export speech_charctc_kws_phone-xiaoyun model to ONNX")
    parser.add_argument("--model_path", type=str, 
                        default="./iic/speech_charctc_kws_phone-xiaoyun",
                        help="Model path or model hub name")
    parser.add_argument("--output_dir", type=str, 
                        default="./outputs/onnx",
                        help="Output directory for ONNX model")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--quantize", type=bool, default=False, help="Whether to quantize")
    parser.add_argument("--opset_version", type=int, default=14, help="ONNX opset version")
    parser.add_argument("--onnx_path", type=str, default="./model/speech_charctc_kws_phone-xiaoyun.onnx", help="ONNX opset version")
    parser.add_argument("--input", type=str, default="./iic/speech_charctc_kws_phone-xiaoyun/example/kws_xiaoyunxiaoyun.wav", help="Input npy file path")
    parser.add_argument("--keywords", type=str, default="小云小云", help="Input keywords")
    args = parser.parse_args()
    onnx_file = args.onnx_path
    os.makedirs(args.output_dir, exist_ok=True)
    infer_audio(args)
    model = AutoModel(
        model=args.model_path,
        keywords=args.keywords,
        device=args.device,
        batch_size=1,
        disable_update=True,
        trust_remote_code=True
    )

    encoder = model.model.encoder
    encoder.eval()

    wrap_model = KWSModelWrapper(encoder)
    wrap_model.eval()
    abs_path = os.getcwd()
    print(abs_path)
    input_npy = abs_path + '/kws.npy'
    raw_data = np.load(input_npy)
    dummpy_input = torch.from_numpy(raw_data).unsqueeze(0) if raw_data.ndim == 2 else torch.from_numpy(raw_data)
    print(f"input shape: {dummpy_input.shape}")

    try:
        with torch.no_grad():
            encoder_output = wrap_model(dummpy_input)
    except Exception as e:
        print("error： ", e)
        raise
    
    torch.onnx.export(
        wrap_model,
        dummpy_input,
        onnx_file,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['speech_features'],
        output_names=['encoder_output'],
        dynamic_axes=None,
        verbose=False,
    )

    print(f"ONNX model saved to: {args.onnx_path}")


if __name__ == "__main__":
    main()