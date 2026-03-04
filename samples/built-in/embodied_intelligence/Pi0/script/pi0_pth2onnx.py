# Copyright (c) ModelZoo. 2025-2026. All rights reserved.
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

# Copyright (c) 2025 Syslong Technology Co., Ltd. All Rights Reserved.
# Copyright (c) 2025 Shanghai Jiao Tong University
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.onnx

from lerobot.policies.pi0.modeling_pi0_all import PI0Policy

LOGGER = logging.getLogger(__name__)

def _normalize_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def _move_policy_to_device(policy: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    policy = policy.to(device)
    for module in policy.modules():
        for buffer in module.buffers():
            if buffer is not None and buffer.device != device:
                buffer.data = buffer.data.to(device)
    return policy


def _prepare_base_tensors(
    device: torch.device, batch_size: int, lang_tokens_len: int, seed: int
) -> dict[str, torch.Tensor]:
    torch.manual_seed(int(seed))
    state = torch.zeros((batch_size, 14), dtype=torch.float32, device=device)
    image = torch.zeros((batch_size, 3, 480, 640), dtype=torch.float32, device=device)
    lang_tokens = torch.randint(0, 1000, (batch_size, lang_tokens_len), dtype=torch.long, device=device)
    lang_masks = torch.ones((batch_size, lang_tokens_len), dtype=torch.bool, device=device)

    return {
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
        "observation.state": state,
        "observation.images.top": image,
    }

class ONNXWrapper(torch.nn.Module):
    """Expose policy.select_action as a single forward op for ONNX export."""

    def __init__(self, policy: PI0Policy, example_observation: dict, device: torch.device):
        super().__init__()
        self.policy = policy.to(device)
        self.device = device
        self._keys = list(example_observation.keys())

    def forward(self, *args):
        if len(args) != len(self._keys):
            raise ValueError(f"Expected {len(self._keys)} inputs, got {len(args)}")
        input_dict = {}
        for key, tensor in zip(self._keys, args, strict=False):
            if key == "lang_tokens":
                input_dict[key] = tensor.to(torch.long)
            elif key == "lang_masks":
                input_dict[key] = tensor.to(torch.bool)
            else:
                input_dict[key] = tensor.to(torch.float32)

        self.policy.eval()
        with torch.no_grad():
            actions = self.policy.select_action(input_dict)
            return actions


def export_onnx(
    *,
    wrapper: ONNXWrapper,
    observation: dict,
    onnx_output_path: Path,
    opset: int,
    do_constant_folding: bool,
) -> None:
    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_keys = list(observation.keys())
    observation_values = [observation[k] for k in dummy_keys]

    LOGGER.info("Exporting ONNX to %s", onnx_output_path)
    torch.onnx.export(
        wrapper,
        tuple(observation_values),
        str(onnx_output_path),
        opset_version=int(opset),
        input_names=dummy_keys,
        output_names=["action"],
        do_constant_folding=bool(do_constant_folding),
        verbose=False,
        dynamo=True,
    )

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export PI0 model to ONNX and optionally validate.")
    p.add_argument(
        "--pretrained-policy-path", type=str, required=True, help="Path or repo with config+weights"
    )
    p.add_argument("--output", type=str, default="../model/pi0.onnx", help="Output ONNX path")
    p.add_argument("--seed", type=int, default=42, help="Seed for dummy inputs and ORT")
    p.add_argument("--batch-size", type=int, default=1, help="Dummy batch size")
    p.add_argument("--lang-tokens-len", type=int, default=48, help="Dummy language token length")
    p.add_argument(
        "--local-files-only", action="store_true", default=True, help="Load policy without network"
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s",)

    policy_path = args.pretrained_policy_path
    onnx_output_path = _normalize_path(args.output)
    device = torch.device("cpu")

    LOGGER.info("Loading PI0Policy from %s", policy_path)
    policy = PI0Policy.from_pretrained(
        policy_path, local_files_only=bool(args.local_files_only), strict=False
    )
    policy = _move_policy_to_device(policy, device)
    # keep half precision as in original script
    try:
        policy.model = policy.model.half()
    except Exception:
        LOGGER.warning("Failed to convert policy.model to half; continuing in float32")
    policy.eval()

    # Optional prefix embeddings
    prefix_embs_path = Path("prefix_embs.pt")
    if prefix_embs_path.exists():
        try:
            _ = torch.load(prefix_embs_path, map_location=device)
        except Exception:
            LOGGER.warning("Failed to load prefix_embs.pt; continuing without it.")

    observation = _prepare_base_tensors(
        device, int(args.batch_size), int(args.lang_tokens_len), int(args.seed)
    )
    wrapper = ONNXWrapper(policy, observation, device)

    export_onnx(
        wrapper=wrapper,
        observation=observation,
        onnx_output_path=onnx_output_path,
        opset=14,
        do_constant_folding=True,
    )
    LOGGER.info("ONNX export finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
