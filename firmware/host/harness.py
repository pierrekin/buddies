"""Qt harness for buddies firmware.

Hosts a TCP server on `127.0.0.1:5555`. Firmware instances connect as
clients (QEMU `-chardev socket,...,reconnect-ms=...`). Single window:
world view on top, one row per connected device below.

Usage (from `firmware/`):
    uv run --project host host/harness.py
"""

from __future__ import annotations

import math
import signal
import sys
from dataclasses import dataclass, field

import numpy as np

from channels import Channel, TestSignalChannel

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtNetwork import QHostAddress, QTcpServer, QTcpSocket
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)


HOST = "127.0.0.1"
PORT = 5555
N_LEDS = 8

WORLD_EXTENT_M = 4.0
WORLD_PX_PER_M = 80.0
WORLD_PX = int(WORLD_EXTENT_M * WORLD_PX_PER_M)

SAMPLE_RATE_HZ = 200_000
N_RX_CHANNELS = 4


@dataclass
class Buddy:
    id: int
    socket: QTcpSocket
    x: float = 0.0
    y: float = 0.0
    heading_deg: float = 0.0
    leds: list[tuple[int, int, int]] = field(
        default_factory=lambda: [(0, 0, 0)] * N_LEDS
    )
    buf: bytearray = field(default_factory=bytearray)
    channel: Channel = field(
        default_factory=lambda: TestSignalChannel(N_RX_CHANNELS, SAMPLE_RATE_HZ)
    )


class World(QObject):
    buddy_added = Signal(object)
    buddy_removed = Signal(object)
    buddy_updated = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.buddies: list[Buddy] = []

    def add(self, buddy: Buddy) -> None:
        self.buddies.append(buddy)
        self.buddy_added.emit(buddy)

    def remove(self, buddy: Buddy) -> None:
        try:
            self.buddies.remove(buddy)
        except ValueError:
            return
        self.buddy_removed.emit(buddy)

    def updated(self, buddy: Buddy) -> None:
        self.buddy_updated.emit(buddy)


class LedIndicator(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._color = QColor(0, 0, 0)
        self.setFixedSize(40, 40)

    def set_rgb(self, r: int, g: int, b: int) -> None:
        self._color = QColor(r, g, b)
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(40, 40, 40), 2))
        p.setBrush(QBrush(self._color))
        p.drawEllipse(4, 4, 32, 32)


class DeviceRow(QFrame):
    def __init__(self, buddy: Buddy) -> None:
        super().__init__()
        self._buddy = buddy
        self.setObjectName("row")
        self.setStyleSheet(
            """
            QFrame#row {
                background-color: #262626;
                border: 1px solid #383838;
                border-radius: 8px;
            }
            """
        )

        self._leds = [LedIndicator() for _ in range(N_LEDS)]
        led_row = QWidget()
        led_layout = QHBoxLayout(led_row)
        led_layout.setContentsMargins(0, 0, 0, 0)
        led_layout.setSpacing(6)
        for led in self._leds:
            led_layout.addWidget(led)

        label = QLabel(f"Device {buddy.id}")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #aaa; font-size: 11pt;")

        case = QFrame()
        case.setObjectName("case")
        case.setStyleSheet(
            """
            QFrame#case {
                background-color: #181818;
                border-radius: 18px;
                border: 1px solid #2a2a2a;
            }
            """
        )
        case_layout = QVBoxLayout(case)
        case_layout.setContentsMargins(20, 16, 20, 12)
        case_layout.setSpacing(10)
        case_layout.addWidget(led_row)
        case_layout.addWidget(label)

        self._debug = QLabel()
        self._debug.setStyleSheet(
            "color: #bbb; "
            "background-color: transparent; "
            "font-family: Menlo, Monaco, Courier, monospace; "
            "font-size: 10pt;"
        )
        self._debug.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )

        debug_box = QFrame()
        debug_box.setObjectName("debug")
        debug_box.setStyleSheet(
            """
            QFrame#debug {
                background-color: #1a1a1a;
                border: 1px solid #2c2c2c;
                border-radius: 6px;
            }
            """
        )
        debug_layout = QVBoxLayout(debug_box)
        debug_layout.setContentsMargins(14, 14, 14, 14)
        debug_layout.addWidget(self._debug)
        debug_layout.addStretch(1)

        row_layout = QHBoxLayout(self)
        row_layout.setContentsMargins(12, 12, 12, 12)
        row_layout.setSpacing(12)
        row_layout.addWidget(case)
        row_layout.addWidget(debug_box, stretch=1)

        self.refresh()

    def refresh(self) -> None:
        for i, (r, g, b) in enumerate(self._buddy.leds):
            if i < len(self._leds):
                self._leds[i].set_rgb(r, g, b)
        self._debug.setText(
            f"id       : {self._buddy.id}\n"
            f"position : ({self._buddy.x:+.2f}, {self._buddy.y:+.2f}) m\n"
            f"heading  : {self._buddy.heading_deg:.1f}°"
        )


class WorldView(QWidget):
    def __init__(self, world: World) -> None:
        super().__init__()
        self._world = world
        self.setFixedSize(WORLD_PX, WORLD_PX)
        self.setStyleSheet(
            "background-color: #0a1a2a; "
            "border: 1px solid #2a3a4a; "
            "border-radius: 6px;"
        )
        # mouseMoveEvent only fires without a button held when tracking is on,
        # which we need during the aim phase.
        self.setMouseTracking(True)
        self._mode = "idle"
        self._active: Buddy | None = None
        world.buddy_added.connect(self.update)
        world.buddy_removed.connect(self._on_buddy_removed)
        world.buddy_updated.connect(self.update)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx = self.width() / 2.0
        cy = self.height() / 2.0
        scale = WORLD_PX_PER_M
        half = WORLD_EXTENT_M / 2.0

        p.setPen(QPen(QColor(255, 255, 255, 25), 1))
        x = -half
        while x <= half + 1e-6:
            sx = cx + x * scale
            p.drawLine(int(sx), 0, int(sx), self.height())
            x += 0.5
        y = -half
        while y <= half + 1e-6:
            sy = cy - y * scale
            p.drawLine(0, int(sy), self.width(), int(sy))
            y += 0.5

        p.setPen(QPen(QColor(255, 255, 255, 80), 1))
        p.drawLine(int(cx) - 6, int(cy), int(cx) + 6, int(cy))
        p.drawLine(int(cx), int(cy) - 6, int(cx), int(cy) + 6)

        p.setPen(QPen(QColor(180, 180, 180), 1))
        p.drawText(int(cx) - 4, 14, "N")

        for buddy in self._world.buddies:
            sx = cx + buddy.x * scale
            sy = cy - buddy.y * scale

            p.save()
            p.translate(sx, sy)
            # Qt rotate is clockwise in degrees, matching our compass heading.
            p.rotate(buddy.heading_deg)
            p.setPen(QPen(QColor(180, 220, 255), 1))
            p.setBrush(QBrush(QColor(60, 80, 100)))
            p.drawRect(-12, -18, 24, 36)
            p.setPen(QPen(QColor(255, 200, 100), 2))
            p.drawLine(0, -18, 0, -28)
            p.restore()

            p.setPen(QPen(QColor(220, 220, 220), 1))
            p.drawText(int(sx) + 18, int(sy) + 5, f"D{buddy.id}")

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._mode == "aiming":
            self._mode = "idle"
            self._active = None
        buddy = self._buddy_under(event.position())
        if buddy is not None:
            self._mode = "dragging"
            self._active = buddy
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._mode == "idle" or self._active is None:
            return
        wx, wy = self._screen_to_world(event.position())
        if self._mode == "dragging":
            self._active.x = wx
            self._active.y = wy
            self._world.updated(self._active)
        elif self._mode == "aiming":
            dx = wx - self._active.x
            dy = wy - self._active.y
            # Cursor on the marker yields atan2(0, 0) = 0, which would snap
            # heading to north; preserve the previous heading instead.
            if abs(dx) > 1e-9 or abs(dy) > 1e-9:
                self._active.heading_deg = math.degrees(math.atan2(dx, dy)) % 360
                self._world.updated(self._active)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._mode == "dragging":
            self._mode = "aiming"

    def _screen_to_world(self, pos) -> tuple[float, float]:
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        return ((pos.x() - cx) / WORLD_PX_PER_M, (cy - pos.y()) / WORLD_PX_PER_M)

    def _buddy_under(self, pos) -> Buddy | None:
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        hit_radius_px = 24.0
        best = None
        best_dist = hit_radius_px
        for buddy in self._world.buddies:
            sx = cx + buddy.x * WORLD_PX_PER_M
            sy = cy - buddy.y * WORLD_PX_PER_M
            d = math.hypot(pos.x() - sx, pos.y() - sy)
            if d < best_dist:
                best_dist = d
                best = buddy
        return best

    def _on_buddy_removed(self, buddy: Buddy) -> None:
        if self._active is buddy:
            self._mode = "idle"
            self._active = None
        self.update()


class MainWindow(QMainWindow):
    def __init__(self, world: World) -> None:
        super().__init__()
        self.setWindowTitle("buddies harness")
        self._rows: dict[int, DeviceRow] = {}

        self._world_view = WorldView(world)

        root = QWidget()
        root.setObjectName("root")
        root.setStyleSheet("QWidget#root { background-color: #1e1e1e; }")
        self._stack = QVBoxLayout(root)
        self._stack.setContentsMargins(16, 16, 16, 16)
        self._stack.setSpacing(16)
        self._stack.addWidget(
            self._world_view, alignment=Qt.AlignmentFlag.AlignHCenter
        )
        self._stack.addStretch(1)
        self.setCentralWidget(root)
        self.resize(720, 600)

    def add_row(self, buddy: Buddy) -> None:
        row = DeviceRow(buddy)
        self._rows[buddy.id] = row
        self._stack.insertWidget(self._stack.count() - 1, row)

    def remove_row(self, buddy: Buddy) -> None:
        row = self._rows.pop(buddy.id, None)
        if row:
            self._stack.removeWidget(row)
            row.deleteLater()

    def refresh_row(self, buddy: Buddy) -> None:
        row = self._rows.get(buddy.id)
        if row:
            row.refresh()


class HarnessServer(QObject):
    buddy_connected = Signal(object)
    buddy_disconnected = Signal(object)

    def __init__(self, world: World) -> None:
        super().__init__()
        self._world = world
        self._next_id = 0
        self._tcp = QTcpServer(self)
        self._tcp.newConnection.connect(self._on_new_connection)

    def listen(self) -> bool:
        return self._tcp.listen(QHostAddress(HOST), PORT)

    def _on_new_connection(self) -> None:
        sock = self._tcp.nextPendingConnection()
        buddy = Buddy(
            id=self._next_id,
            socket=sock,
            x=float(self._next_id),
            y=0.0,
            heading_deg=0.0,
        )
        self._next_id += 1

        sock.write(b"g")
        sock.readyRead.connect(lambda b=buddy: self._on_data(b))
        sock.disconnected.connect(lambda b=buddy: self._on_disconnect(b))

        self._world.add(buddy)
        self.buddy_connected.emit(buddy)

    def _on_data(self, buddy: Buddy) -> None:
        chunk = bytes(buddy.socket.readAll())
        buddy.buf.extend(chunk)
        while b"\n" in buddy.buf:
            line, _, rest = buddy.buf.partition(b"\n")
            buddy.buf = bytearray(rest)
            self._parse_line(buddy, line.decode("ascii", errors="replace"))

    def _parse_line(self, buddy: Buddy, line: str) -> None:
        parts = line.strip().split()
        if not parts:
            return
        cmd = parts[0]
        if cmd == "strip":
            self._handle_strip(buddy, parts[1:])
        elif cmd == "scan":
            self._handle_scan(buddy)
        elif cmd == "rx":
            self._handle_rx(buddy, parts[1:])

    def _handle_strip(self, buddy: Buddy, values: list[str]) -> None:
        if len(values) % 3 != 0:
            return
        triples = list(zip(values[::3], values[1::3], values[2::3]))
        try:
            buddy.leds = [(int(r), int(g), int(b)) for r, g, b in triples[:N_LEDS]]
        except ValueError:
            return
        self._world.updated(buddy)

    def _handle_scan(self, buddy: Buddy) -> None:
        target = self._find_target(buddy)
        if target is None:
            buddy.socket.write(b"peer none\n")
            return
        bearing, range_m = self._compute_body_bearing_range(buddy, target)
        line = f"peer {target.id} {bearing:.2f} {range_m:.3f}\n"
        buddy.socket.write(line.encode("ascii"))

    def _find_target(self, buddy: Buddy) -> Buddy | None:
        other_ids = sorted(b.id for b in self._world.buddies if b.id != buddy.id)
        if not other_ids:
            return None
        higher = [i for i in other_ids if i > buddy.id]
        target_id = higher[0] if higher else other_ids[0]
        return next(
            (b for b in self._world.buddies if b.id == target_id), None
        )

    def _compute_body_bearing_range(
        self, viewer: Buddy, target: Buddy
    ) -> tuple[float, float]:
        dx = target.x - viewer.x
        dy = target.y - viewer.y
        range_m = math.hypot(dx, dy)
        world_bearing = math.degrees(math.atan2(dx, dy)) % 360
        body_bearing = ((world_bearing - viewer.heading_deg + 180) % 360) - 180
        return body_bearing, range_m

    def _handle_rx(self, buddy: Buddy, args: list[str]) -> None:
        if len(args) != 2:
            return
        try:
            n_samples = int(args[0])
            n_channels = int(args[1])
        except ValueError:
            return
        rx = buddy.channel.step(n_samples)
        if rx.shape != (n_channels, n_samples) or rx.dtype != np.float32:
            rx = np.ascontiguousarray(rx[:n_channels, :n_samples], dtype=np.float32)
        buddy.socket.write(rx.tobytes())

    def _on_disconnect(self, buddy: Buddy) -> None:
        self._world.remove(buddy)
        self.buddy_disconnected.emit(buddy)


def main() -> int:
    app = QApplication(sys.argv)

    signal.signal(signal.SIGINT, lambda *_: app.quit())
    # Wake the Python interpreter periodically so it can dispatch signals
    # while Qt's event loop holds the main thread.
    sigint_timer = QTimer()
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start(100)

    world = World()
    server = HarnessServer(world)
    window = MainWindow(world)
    window.show()

    server.buddy_connected.connect(window.add_row)
    server.buddy_disconnected.connect(window.remove_row)
    world.buddy_updated.connect(window.refresh_row)

    if not server.listen():
        print(f"could not bind {HOST}:{PORT}", file=sys.stderr)
        return 1
    print(f"harness listening on {HOST}:{PORT}")

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
