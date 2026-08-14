#!/usr/bin/env python3
"""Capture a 115200 8N1 UART with PID locking and graceful shutdown."""

from __future__ import annotations

import argparse
import fcntl
import os
import select
import signal
import sys
import termios
import time
from datetime import UTC, datetime
from pathlib import Path

STOP = False


def request_stop(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def configure_uart(fd: int) -> None:
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0
    attrs[1] = 0
    attrs[2] = termios.CS8 | termios.CLOCAL | termios.CREAD
    attrs[3] = 0
    attrs[4] = termios.B115200
    attrs[5] = termios.B115200
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 1
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    termios.tcflush(fd, termios.TCIFLUSH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pid-file", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=0, help="seconds; zero means until signalled")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def main() -> int:
    args = parse_args()
    if args.timeout < 0:
        print("timeout cannot be negative", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.pid_file.parent.mkdir(parents=True, exist_ok=True)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)
    try:
        with args.pid_file.open("w", encoding="ascii") as pid_stream:
            fcntl.flock(pid_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pid_stream.write(f"{os.getpid()}\n")
            pid_stream.flush()
            fd = os.open(args.device, os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)
            try:
                configure_uart(fd)
                deadline = time.monotonic() + args.timeout if args.timeout else None
                with args.output.open("ab", buffering=0) as output:
                    output.write(f"\n--- UART capture started {timestamp()} ---\n".encode())
                    while not STOP and (deadline is None or time.monotonic() < deadline):
                        readable, _, _ = select.select([fd], [], [], 0.25)
                        if readable:
                            chunk = os.read(fd, 4096)
                            if chunk:
                                output.write(chunk)
                    output.write(f"\n--- UART capture stopped {timestamp()} ---\n".encode())
            finally:
                os.close(fd)
    except (BlockingIOError, OSError) as exc:
        print(f"UART capture failed: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            args.pid_file.unlink()
        except FileNotFoundError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
