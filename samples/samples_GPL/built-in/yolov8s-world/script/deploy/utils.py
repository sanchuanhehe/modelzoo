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
import types
import yaml

from ultralytics.nn.modules import Detect
from ultralytics.utils import LOGGER
from .modules import DEPLOY_MODULE_MAPPINGS


FILE_ACCESS_FLAG = os.O_WRONLY | os.O_TRUNC | os.O_CREAT
FILE_ACCESS_MODE = 0o755


def convert_to_deploy(yolo_model):
    model = yolo_model.model

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    model = model.fuse()

    for m in model.modules():
        if isinstance(m, Detect):
            m.dynamic = False
            m.export = True
            m.format = 'onnx'
    LOGGER.info('Model converted to deploy format')

    for k1, v1 in DEPLOY_MODULE_MAPPINGS.items():
        for k2, v2 in v1.items():
            LOGGER.info(f"Replace {k1.__name__}.{k2} with {v2.__name__}")
    
    modules = dict(model.named_modules(remove_duplicate=False))
    for _, module in modules.items():
        if type(module) not in DEPLOY_MODULE_MAPPINGS:
            continue
        replace_dict = DEPLOY_MODULE_MAPPINGS.get(type(module))
        for k, v in replace_dict.items():
            if hasattr(module, k):
                setattr(module, k, types.MethodType(v, module))
    return model.eval()

def set_classes(data_config_file, model):
    with open(data_config_file, 'r') as f:
        data_config = yaml.safe_load(f)
    classes_names = data_config['names']
    names_list = [name.split('/')[0] for name in classes_names.values()]
    if model.names != classes_names:
        LOGGER.info(f"Set model classes to {names_list}")
        model.set_classes(names_list)
        model.model.names = classes_names
    return model




