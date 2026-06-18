"""Headless listener for buddies firmware.

Stdlib-only counterpart to `harness.py`. Binds `127.0.0.1:5555`,
accepts firmware connections, and prints each frame tagged by buddy ID.

Usage (from `firmware/`, stdlib only so no uv needed):
    python3 host/listener.py
"""

import socket
import sys
import threading


HOST = "127.0.0.1"
PORT = 5555


def handle(conn: socket.socket, addr: tuple, buddy_id: int) -> None:
    print(f"buddy {buddy_id} connected from {addr[0]}:{addr[1]}", flush=True)
    try:
        conn.sendall(b"g")
        buf = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, _, rest = buf.partition(b"\n")
                buf = rest
                text = line.decode("ascii", errors="replace").strip()
                parts = text.split()
                if parts and parts[0] == "oled" and len(parts) == 4:
                    print(
                        f"buddy {buddy_id}: oled {parts[1]}x{parts[2]} "
                        f"({len(parts[3]) // 4} px)",
                        flush=True,
                    )
                elif parts and parts[0] == "strip" and (len(parts) - 1) % 3 == 0:
                    try:
                        vs = [int(x) for x in parts[1:]]
                    except ValueError:
                        print(f"buddy {buddy_id} raw: {text!r}", flush=True)
                        continue
                    triples = zip(vs[::3], vs[1::3], vs[2::3])
                    hexes = "  ".join(
                        f"#{r:02x}{g:02x}{b:02x}" for r, g, b in triples
                    )
                    print(f"buddy {buddy_id}: {hexes}", flush=True)
                else:
                    print(f"buddy {buddy_id} raw: {text!r}", flush=True)
    finally:
        conn.close()
        print(f"buddy {buddy_id} disconnected", flush=True)


def main() -> int:
    next_id = 0
    with socket.create_server((HOST, PORT)) as srv:
        print(f"listening on {HOST}:{PORT}", flush=True)
        while True:
            conn, addr = srv.accept()
            threading.Thread(
                target=handle, args=(conn, addr, next_id), daemon=True
            ).start()
            next_id += 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
