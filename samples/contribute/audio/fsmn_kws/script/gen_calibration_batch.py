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

import sys
import os
import argparse
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "FunASR"))

import numpy as np
from funasr import AutoModel

def main():
    parser = argparse.ArgumentParser(description="Export speech_charctc_kws_phone-xiaoyun model to ONNX")
    parser.add_argument("--model_path", type=str, 
                        default="./iic/speech_charctc_kws_phone-xiaoyun",
                        help="Model path or model hub name")
    parser.add_argument("--output_dir", type=str, 
                        default="./data/calibration",
                        help="Output directory for data")
    parser.add_argument("--input", type=str, default="./iic/speech_charctc_kws_phone-xiaoyun/unittest/example_kws/wav/", help="wav dir")
    parser.add_argument("--target_frames", type=int, default=151, help="input shape")
    parser.add_argument("--keywords", type=str, default="小云小云", help="Input keywords")
    args = parser.parse_args()
    
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
        print(f"Cleaned output_dir: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Only process these specific files
    target_files = [
        '20200707_spk57db_storenoise52db_40cm_xiaoyun_sox_37.wav',
        '20200707_spk57db_storenoise52db_40cm_xiaoyun_sox_50.wav'
    ]
    
    model = AutoModel(
        model=args.model_path,
        keywords=args.keywords,
        output_dir=args.output_dir,
        device="cpu",
        disable_update=True,
        trust_remote_code=True
    )
    
    calib_data_list = []
    target_frames = args.target_frames
    
    for wav_file in target_files:
        wav_path = os.path.join(args.input, wav_file)
        basename = os.path.splitext(wav_file)[0]
        print(f"Processing {wav_path}...")
        
        res = model.generate(input=wav_path, cache={})
        
        # Load saved preprocess result
        abs_path = os.getcwd()
        print(abs_path)
        preprocess_npy = abs_path + '/kws.npy'
        if os.path.exists(preprocess_npy):
            data = np.load(preprocess_npy)
            frames = data.shape[1]
            
            # Pad to target_frames using last frame
            if frames < target_frames:
                last_frame = data[0, -1, :]
                padded_data = np.zeros((1, target_frames, 400), dtype=np.float32)
                padded_data[0, :frames, :] = data[0, :, :]
                for i in range(frames, target_frames):
                    padded_data[0, i, :] = last_frame
                processed_data = padded_data
                print(f"  Padded: {frames} -> {target_frames} frames")
            elif frames > target_frames:
                processed_data = data[0, :target_frames, :].reshape(1, target_frames, 400)
                print(f"  Truncated: {frames} -> {target_frames} frames")
            else:
                processed_data = data
                print(f"  Exact: {frames} frames")
            
            # Save individual file
            output_file = os.path.join(args.output_dir, f'calib_{basename}_{target_frames}frames.bin')
            processed_data.astype(np.float32).tofile(output_file)
            print(f"  Saved: {output_file}")
            
            calib_data_list.append(processed_data)
    
    # Create batch file
    if calib_data_list:
        batch_size = len(calib_data_list)
        batch_data = np.zeros((batch_size, 1, target_frames, 400), dtype=np.float32)
        
        for i, data in enumerate(calib_data_list):
            batch_data[i, 0, :, :] = data[0, :, :]
        
        batch_file = os.path.join(args.output_dir, f'calib_batch_{batch_size}x{target_frames}frames.bin')
        batch_data.astype(np.float32).tofile(batch_file)
        
        # Save as txt
        txt_file = os.path.join(args.output_dir, 'calib_batch.txt')
        batch_2d = batch_data.reshape(-1)
        np.savetxt(txt_file, batch_2d, fmt='%.6f', delimiter=' ')
        
        print(f"\n=== Output ===")
        print(f"Batch bin: {batch_file}")
        print(f"Batch txt: {txt_file}")
        print(f"Batch shape: {batch_data.shape}")

if __name__ == "__main__":
    main()