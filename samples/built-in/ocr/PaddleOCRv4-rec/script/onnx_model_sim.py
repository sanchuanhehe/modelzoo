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

import onnx
from onnxsim import simplify
import onnxruntime as ort

def onnx_sim(model_path, output_path):
        if output_path is None:
            output_path = model_path.replace('.onnx', '_optimized.onnx')
        model_sim, check = simplify(model_path)
        onnx.save(model_sim, output_path)
        if check:
             print("sucess")
        else:
             print("sim failed")
    
        
# 使用示例
if __name__ == "__main__":
    
    onnx_sim(
        "ch_ptocr_v4_det_infer.pth_3x960x960.onnx", 
        "ch_ptocr_v4_det_simplified.onnx",
    )
    onnx_sim(
        "ch_ptocr_v4_rec_infer.pth_3x48x320.onnx", 
        "ch_ptocr_v4_rec_simplified.onnx",
    )