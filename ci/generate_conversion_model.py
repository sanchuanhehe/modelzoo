#!/usr/bin/env python3
"""Generate the deterministic, redistributable ONNX used by conversion CI."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from onnx import TensorProto, helper

EXPECTED_SHA256 = "153e867604c1c2a0c441d3a2591474e6238052b034716d105e08ce578b4b2f48"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--print-sha", action="store_true")
    args = parser.parse_args()
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 4])
    graph = helper.make_graph([helper.make_node("Relu", ["input"], ["output"])], "modelzoo_ci_relu", [x], [y])
    model = helper.make_model(graph, producer_name="modelzoo-ci", opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 7
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model.SerializeToString())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.print_sha:
        print(digest)
    elif EXPECTED_SHA256 == "PLACEHOLDER" or digest != EXPECTED_SHA256:
        raise SystemExit(f"deterministic ONNX SHA mismatch: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
