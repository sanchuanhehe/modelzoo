"""
ACT worker client for the SVP_NNN binary protocol.
"""

import argparse
import os
import struct
import subprocess
import sys
import threading
from typing import Iterable, List, Sequence, Tuple

import numpy as np


PROTOCOL_MAGIC = 0x53565031
PROTOCOL_VERSION = 1

REQUEST_HEADER_STRUCT = struct.Struct("<IHHII")
INPUT_ENTRY_STRUCT = struct.Struct("<III")
RESPONSE_HEADER_STRUCT = struct.Struct("<IHHIIIiI")
OUTPUT_ENTRY_STRUCT = struct.Struct("<IIIIII")
DIM_STRUCT = struct.Struct("<Q")

WORKER_STATUS_OK = 0

WORKER_ELEM_UNKNOWN = 0
WORKER_ELEM_FLOAT32 = 1
WORKER_ELEM_FLOAT16 = 2
WORKER_ELEM_INT8 = 3
WORKER_ELEM_UINT8 = 4
WORKER_ELEM_INT32 = 5
WORKER_ELEM_INT64 = 6


def _read_exact(stream, size: int) -> bytes:
    chunks: List[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise RuntimeError("worker stream closed unexpectedly")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _dtype_from_elem_type(elem_type: int):
    if elem_type == WORKER_ELEM_FLOAT32:
        return np.float32
    if elem_type == WORKER_ELEM_FLOAT16:
        return np.float16
    if elem_type == WORKER_ELEM_INT8:
        return np.int8
    if elem_type == WORKER_ELEM_UINT8:
        return np.uint8
    if elem_type == WORKER_ELEM_INT32:
        return np.int32
    if elem_type == WORKER_ELEM_INT64:
        return np.int64
    raise RuntimeError(f"unsupported worker element type: {elem_type}")


class ACT3403Policy:
    def __init__(self, cpp_executable: str, model_path: str):
        super().__init__()
        self.cpp_executable = cpp_executable
        self.cpp_dir = os.path.dirname(cpp_executable)
        self._worker_env = os.environ.copy()
        self._worker_env["SVP_MODEL_PATH"] = model_path
        self._request_id = 0
        self._process = None
        self._stderr_thread = None
        self._start_process()

    def _start_process(self):
        self._process = subprocess.Popen(
            [self.cpp_executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cpp_dir,
            env=self._worker_env,
            text=False,
            bufsize=0,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(self._process.stderr,),
            daemon=True,
        )
        self._stderr_thread.start()

    def _drain_stderr(self, pipe):
        try:
            while True:
                chunk = pipe.readline()
                if not chunk:
                    break
                try:
                    sys.stderr.buffer.write(chunk)
                    sys.stderr.buffer.flush()
                except Exception:
                    sys.stderr.write(chunk.decode("utf-8", errors="replace"))
                    sys.stderr.flush()
        except Exception:
            pass

    def close(self):
        if self._process is None:
            return
        try:
            if self._process.stdin:
                self._process.stdin.close()
        except Exception:
            pass
        try:
            self._process.terminate()
            self._process.wait(timeout=2)
        except Exception:
            try:
                self._process.kill()
            except Exception:
                pass
        self._process = None

    def _worker_exit_message(self) -> str:
        if self._process is None:
            return "worker process is not running"

        return_code = self._process.poll()
        stderr_text = ""
        try:
            if self._process.stderr is not None:
                stderr_text = self._process.stderr.read().decode("utf-8", errors="replace").strip()
        except Exception:
            stderr_text = ""

        msg = f"worker exited unexpectedly"
        if return_code is not None:
            msg += f" (returncode={return_code})"
        if stderr_text:
            msg += f"\nworker stderr:\n{stderr_text}"
        return msg

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _write_request(self, input_arrays: Sequence[np.ndarray]) -> int:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("worker process is not running")
        if self._process.poll() is not None:
            raise RuntimeError(self._worker_exit_message())

        request_id = self._next_request_id()
        header = REQUEST_HEADER_STRUCT.pack(
            PROTOCOL_MAGIC,
            PROTOCOL_VERSION,
            len(input_arrays),
            request_id,
            0,
        )
        try:
            self._process.stdin.write(header)

            for idx, arr in enumerate(input_arrays):
                contiguous = np.ascontiguousarray(arr)
                payload = contiguous.tobytes()
                entry = INPUT_ENTRY_STRUCT.pack(idx, len(payload), 0)
                self._process.stdin.write(entry)
                self._process.stdin.write(payload)

            self._process.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError(self._worker_exit_message()) from exc
        return request_id

    def _read_response(self, expected_request_id: int):
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("worker process is not running")
        if self._process.poll() is not None:
            raise RuntimeError(self._worker_exit_message())

        header_bytes = _read_exact(self._process.stdout, RESPONSE_HEADER_STRUCT.size)
        (
            magic,
            version,
            status,
            request_id,
            output_count,
            latency_us,
            error_code,
            error_msg_size,
        ) = RESPONSE_HEADER_STRUCT.unpack(header_bytes)

        if magic != PROTOCOL_MAGIC or version != PROTOCOL_VERSION:
            raise RuntimeError(
                f"unexpected response header: magic=0x{magic:x}, version={version}"
            )
        if request_id != expected_request_id:
            raise RuntimeError(
                f"mismatched response id: expected {expected_request_id}, got {request_id}"
            )

        outputs = {}
        for _ in range(output_count):
            entry_bytes = _read_exact(self._process.stdout, OUTPUT_ENTRY_STRUCT.size)
            output_index, elem_type, elem_count, byte_size, dim_count, _reserved = OUTPUT_ENTRY_STRUCT.unpack(entry_bytes)
            dims = [
                DIM_STRUCT.unpack(_read_exact(self._process.stdout, DIM_STRUCT.size))[0]
                for _ in range(dim_count)
            ]
            payload = _read_exact(self._process.stdout, byte_size)
            dtype = _dtype_from_elem_type(elem_type)
            data = np.frombuffer(payload, dtype=dtype, count=elem_count)
            if dims:
                data = data.reshape(tuple(int(dim) for dim in dims))
            outputs[output_index] = data.copy()

        error_msg = ""
        if error_msg_size:
            error_msg = _read_exact(self._process.stdout, error_msg_size).decode(
                "utf-8", errors="replace"
            )

        if status != WORKER_STATUS_OK:
            raise RuntimeError(
                f"worker inference failed (error_code={error_code}): {error_msg or 'unknown error'}"
            )

        return outputs, latency_us

    def predict(self, batch: Sequence[np.ndarray]) -> np.ndarray:
        request_id = self._write_request(batch)
        outputs, _latency_us = self._read_response(request_id)
        if 2 not in outputs:
            raise RuntimeError("worker response does not contain output index 2")
        action = outputs[2]
        if action.ndim == 2:
            action = action[np.newaxis, ...]
        return action.astype(np.float32, copy=False)


def read_bin_file(file_path, dtype=np.float32):
    try:
        with open(file_path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=dtype)
        return data
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}")
        return None
    except Exception as e:
        print(f"读取bin文件失败：{str(e)}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_list", type=str, required=True)
    parser.add_argument(
        "--model_path",
        type=str,
        default="../model/act_distill_fp32_for_mindcmd_simp_release.om",
        help="Path to OM model file for worker process",
    )
    args = parser.parse_args()

    image_list = [item for item in args.image_list.split(";")]
    batch = []
    for bin_path in image_list:
        batch.append(read_bin_file(bin_path))

    om_model_path = "../out/main"
    with ACT3403Policy(om_model_path, args.model_path) as model:
        action = model.predict(batch)[0]
    action = np.around(action, 4)
    with open("result.txt", "w", encoding="utf-8") as f:
        for num in action:
            f.write(str(num))
            f.write("\n")
