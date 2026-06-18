#!/usr/bin/env python3
"""Launch the buddies host harness plus N firmware instances.

Each firmware instance is a `cargo run` (socket PAL), so every launch
recompiles the latest firmware before QEMU connects to the harness. The
harness arranges the N devices on a ring aimed at the world origin.

Ctrl-C tears the whole fleet down. Rerun to pick up recompiled code.

Usage (from `firmware/`):
    ./run.py [N]                       # N firmware instances (default 3)
    ./run.py 2 --panel 256x64          # build for the smaller bring-up panel
    ./run.py --features foo,bar        # enable extra cargo features
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

FIRMWARE_DIR = Path(__file__).resolve().parent

# Always-on cargo features for a harness run; the panel and any extras come from
# the CLI. `--no-default-features` keeps the semihosting PAL out of socket runs.
BASE_FEATURES = ["pal-socket"]
DEFAULT_PANEL = "rm67162"


def cargo_cmd(verb: str, features: list[str]) -> list[str]:
    return ["cargo", verb, "--no-default-features", "--features", ",".join(features)]


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Launch the buddies harness plus N firmware instances."
    )
    p.add_argument(
        "n", nargs="?", type=int, default=3,
        help="number of firmware instances (default 3)",
    )
    p.add_argument(
        "--panel", choices=["rm67162", "256x64"], default=DEFAULT_PANEL,
        help=f"display panel to build for (default {DEFAULT_PANEL})",
    )
    p.add_argument(
        "-F", "--features", action="append", default=[], metavar="FEAT",
        help="extra cargo feature(s) to enable; repeatable or comma-separated",
    )
    return p.parse_args(argv)


def main() -> int:
    args = parse_args(sys.argv[1:])
    n = args.n
    if n < 1:
        print("N must be >= 1", file=sys.stderr)
        return 2

    extras = [f for item in args.features for f in item.split(",") if f]
    features = [*BASE_FEATURES, f"panel-{args.panel}", *extras]
    cargo_build = cargo_cmd("build", features)
    cargo_run = cargo_cmd("run", features)

    # Compile once up front: a build error fails fast, and the instances below
    # hit a warm cache instead of racing on cargo's build lock.
    build = subprocess.run(cargo_build, cwd=FIRMWARE_DIR)
    if build.returncode != 0:
        return build.returncode

    procs: list[subprocess.Popen] = []

    def shutdown(*_) -> None:
        # Each child leads its own session, so signal the whole group to also
        # reach the QEMU process that `cargo run` spawned.
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 3.0
        for p in procs:
            try:
                p.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Harness first so the server is listening before instances connect.
    # (QEMU's reconnect-ms retries anyway, but this avoids the initial misses.)
    procs.append(
        subprocess.Popen(
            ["uv", "run", "--project", "host", "host/harness.py", str(n)],
            cwd=FIRMWARE_DIR,
            start_new_session=True,
        )
    )
    time.sleep(1.0)

    for _ in range(n):
        procs.append(
            subprocess.Popen(
                cargo_run,
                cwd=FIRMWARE_DIR,
                start_new_session=True,
                # Detached QEMU instances don't share the terminal's stdin.
                stdin=subprocess.DEVNULL,
            )
        )

    # If any part exits (harness window closed, a QEMU died), bring it all down.
    while True:
        for p in procs:
            if p.poll() is not None:
                shutdown()
        time.sleep(0.3)


if __name__ == "__main__":
    raise SystemExit(main())
