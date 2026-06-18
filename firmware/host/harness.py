"""Qt harness for buddies firmware.

Hosts a TCP server on `127.0.0.1:5555`. Firmware instances connect as
clients (QEMU `-chardev socket,...,reconnect-ms=...`). Single window:
world view on top, one row per connected device below.

Usage (from `firmware/`):
    uv run --project host host/harness.py [N]

`N` is the number of devices to expect; connected buddies are spread evenly
around a ring and aimed at the world origin. Omit it for the unarranged
fallback layout.
"""

from __future__ import annotations

import math
import signal
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from channels import (
    Channel,
    RX_POSITIONS_BODY,
    SinglePeerChirpChannel,
    SOUND_SPEED_M_PER_S,
    TestSignalChannel,
)

from PySide6.QtCore import QObject, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen
from PySide6.QtNetwork import QHostAddress, QTcpServer, QTcpSocket
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


HOST = "127.0.0.1"
PORT = 5555
N_LEDS = 16
LED_ROWS = 3
LED_CENTER = N_LEDS // 2

# Shared frame styling so the live device rows and the UI gallery match.
ROW_STYLE = """
    QFrame#row {
        background-color: #262626;
        border: 1px solid #383838;
        border-radius: 8px;
    }
"""

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
    # Full LED_ROWS x N_LEDS frame the firmware renders and streams over.
    leds: list[list[tuple[int, int, int]]] = field(
        default_factory=lambda: [[(0, 0, 0)] * N_LEDS for _ in range(LED_ROWS)]
    )
    buf: bytearray = field(default_factory=bytearray)
    channel: Channel = field(
        default_factory=lambda: SinglePeerChirpChannel(
            N_RX_CHANNELS, SAMPLE_RATE_HZ
        )
    )
    # Tap-button clicks waiting to be drained by the firmware's `taps` poll.
    pending_taps: int = 0
    # Latest OLED frame: dimensions plus a packed RGB888 buffer (row-major).
    oled_w: int = 0
    oled_h: int = 0
    oled_rgb: bytes = b""


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


class LedBar(QWidget):
    """The device's side panel: an N_LEDS x LED_ROWS matrix of small LEDs.

    `set_columns` drives the live in-plane view (one azimuth value per column,
    replicated down every row). `set_grid` addresses each cell directly, which
    the UI gallery uses to draw tees, pips and other multi-row glyphs.
    """

    COLS = N_LEDS
    ROWS = LED_ROWS
    GAP_PX = 2.0

    def __init__(self) -> None:
        super().__init__()
        self._grid = [[(0, 0, 0)] * self.COLS for _ in range(self.ROWS)]
        self.setMinimumSize(self.COLS * 14, self.ROWS * 14)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

    def set_columns(self, cols: list[tuple[int, int, int]]) -> None:
        cols = [tuple(c) for c in cols[: self.COLS]]
        cols += [(0, 0, 0)] * (self.COLS - len(cols))
        self._grid = [list(cols) for _ in range(self.ROWS)]
        self.update()

    def set_grid(self, grid: list[list[tuple[int, int, int]]]) -> None:
        self._grid = [
            [tuple(grid[r][c]) for c in range(self.COLS)]
            for r in range(self.ROWS)
        ]
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(10, 10, 10))

        gap = self.GAP_PX
        cell_w = (self.width() - gap * (self.COLS + 1)) / self.COLS
        cell_h = (self.height() - gap * (self.ROWS + 1)) / self.ROWS
        radius = min(cell_w, cell_h) * 0.3

        p.setPen(Qt.PenStyle.NoPen)
        for row in range(self.ROWS):
            y = gap + row * (cell_h + gap)
            for col in range(self.COLS):
                r, g, b = self._grid[row][col]
                lit = (r, g, b) != (0, 0, 0)
                color = QColor(r, g, b) if lit else QColor(26, 26, 26)
                x = gap + col * (cell_w + gap)
                p.setBrush(QBrush(color))
                p.drawRoundedRect(QRectF(x, y, cell_w, cell_h), radius, radius)


class OledView(QWidget):
    """The device's OLED screen: the RGB565 framebuffer the firmware streams,
    drawn on a mock of the RM67162 AMOLED dev module (rounded display corners in
    a dark board bezel). Nearest-neighbour scaling keeps the dot grid crisp. The
    screen aspect follows the streamed frame, so it tracks whichever panel the
    firmware was built for (256x64 bring-up or 536x240 RM67162)."""

    # RM67162-ish proportions: a wide screen in a slightly larger board.
    BEZEL_PX = 14
    BOARD_RADIUS = 22
    SCREEN_RADIUS = 14

    def __init__(self) -> None:
        super().__init__()
        self._img: QImage | None = None
        self.setMinimumSize(360, 190)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

    def set_frame(self, w: int, h: int, rgb: bytes) -> None:
        # Copy so the QImage owns its pixels; `rgb` is replaced every frame.
        self._img = QImage(
            rgb, w, h, w * 3, QImage.Format.Format_RGB888
        ).copy()
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(14, 15, 17))
        if self._img is None:
            return

        # Fit the screen inside the widget, leaving room for the bezel, while
        # preserving the streamed frame's aspect ratio.
        bez = self.BEZEL_PX
        avail_w = self.width() - 2 * bez - 8
        avail_h = self.height() - 2 * bez - 8
        iw, ih = self._img.width(), self._img.height()
        scale = min(avail_w / iw, avail_h / ih)
        sw, sh = int(iw * scale), int(ih * scale)
        sx = (self.width() - sw) // 2
        sy = (self.height() - sh) // 2

        # Board bezel: a dark rounded slab a little larger than the screen.
        board = QRectF(sx - bez, sy - bez, sw + 2 * bez, sh + 2 * bez)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(20, 20, 22))
        p.drawRoundedRect(board, self.BOARD_RADIUS, self.BOARD_RADIUS)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(45, 47, 52), 1))
        p.drawRoundedRect(board, self.BOARD_RADIUS, self.BOARD_RADIUS)

        # The AMOLED itself: rounded-corner screen clipping the framebuffer.
        screen = QRectF(sx, sy, sw, sh)
        path = QPainterPath()
        path.addRoundedRect(screen, self.SCREEN_RADIUS, self.SCREEN_RADIUS)
        p.save()
        p.setClipPath(path)
        p.fillRect(screen, QColor(0, 0, 0))
        scaled = self._img.scaled(
            sw, sh,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        p.drawImage(sx, sy, scaled)
        p.restore()


class LedPanel(QFrame):
    """The dark device 'case' wrapping an LedBar plus a caption."""

    def __init__(self, caption: str = "") -> None:
        super().__init__()
        self.setObjectName("case")
        self.setStyleSheet(
            """
            QFrame#case {
                background-color: #181818;
                border-radius: 18px;
                border: 1px solid #2a2a2a;
            }
            """
        )
        self.bar = LedBar()
        self._caption = QLabel(caption)
        self._caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._caption.setStyleSheet("color: #aaa; font-size: 11pt;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 12)
        layout.setSpacing(10)
        layout.addWidget(self.bar)
        layout.addWidget(self._caption)

    def set_caption(self, text: str) -> None:
        self._caption.setText(text)


class InfoBox(QFrame):
    """The text panel beside a device/screen (debug readout or explanation)."""

    def __init__(self, mono: bool = False) -> None:
        super().__init__()
        self.setObjectName("info")
        self.setStyleSheet(
            """
            QFrame#info {
                background-color: #1a1a1a;
                border: 1px solid #2c2c2c;
                border-radius: 6px;
            }
            """
        )
        font = (
            "font-family: Menlo, Monaco, Courier, monospace; " if mono else ""
        )
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setStyleSheet(
            f"color: #bbb; background-color: transparent; font-size: 10pt; {font}"
        )
        self.label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.addWidget(self.label)
        layout.addStretch(1)


class DeviceRow(QFrame):
    def __init__(self, buddy: Buddy) -> None:
        super().__init__()
        self._buddy = buddy
        self.setObjectName("row")
        self.setStyleSheet(ROW_STYLE)

        self._panel = LedPanel(f"Device {buddy.id}")
        self._oled = OledView()
        self._info = InfoBox(mono=True)

        tap_btn = QPushButton("Tap")
        tap_btn.setStyleSheet(
            "QPushButton {"
            " background-color: #2f2f2f; color: #ddd;"
            " border: 1px solid #444; border-radius: 6px; padding: 6px 0; }"
            "QPushButton:pressed { background-color: #4a90d9; color: #fff; }"
        )
        tap_btn.clicked.connect(self._on_tap)

        left = QWidget()
        left_col = QVBoxLayout(left)
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(8)
        left_col.addWidget(self._panel)
        left_col.addWidget(self._oled)
        left_col.addWidget(tap_btn)

        row_layout = QHBoxLayout(self)
        row_layout.setContentsMargins(12, 12, 12, 12)
        row_layout.setSpacing(12)
        row_layout.addWidget(left)
        row_layout.addWidget(self._info, stretch=1)

        self.refresh()

    def _on_tap(self) -> None:
        # Firmware drains this count on its next `taps` poll.
        self._buddy.pending_taps += 1

    def refresh(self) -> None:
        self._panel.bar.set_grid(self._buddy.leds)
        if self._buddy.oled_rgb:
            self._oled.set_frame(
                self._buddy.oled_w, self._buddy.oled_h, self._buddy.oled_rgb
            )
        self._info.label.setText(
            f"id       : {self._buddy.id}\n"
            f"position : ({self._buddy.x:+.2f}, {self._buddy.y:+.2f}) m\n"
            f"heading  : {self._buddy.heading_deg:.1f}°"
        )


# --- UI screen gallery -----------------------------------------------------
#
# Pure functions returning a LED_ROWS x N_LEDS grid of RGB tuples. Row 0 is the
# top lightpipe, row 1 the middle, row 2 the bottom. Rows are spatial: bearing
# rides all rows (the in-plane stem), identity rides the bottom, tracking the
# middle. A tap is modal: it commandeers the bar for a readout, then hands the
# rows back to the live position view.

OFF = (0, 0, 0)
GREEN = (0, 220, 0)
WHITE = (235, 235, 235)
BLUE = (70, 150, 255)


def _blank() -> list[list[tuple[int, int, int]]]:
    return [[OFF] * N_LEDS for _ in range(LED_ROWS)]


def _pip_cols(n: int, spacing: int = 2, center: int = LED_CENTER) -> list[int]:
    """Columns for `n` evenly spaced pips centred on `center`."""
    start = center - (n - 1) * spacing // 2
    return [start + i * spacing for i in range(n)]


def screen_position(col: int, color: tuple[int, int, int] = GREEN):
    """Live bearing: a full-height stem at azimuth column `col`."""
    grid = _blank()
    for r in range(LED_ROWS):
        grid[r][col] = color
    return grid


def screen_identity(n: int):
    """'Who am I' = white pips on the bottom row + stem rising above (a ⊥)."""
    grid = _blank()
    for c in _pip_cols(n):
        grid[2][c] = WHITE
    grid[0][LED_CENTER] = WHITE
    grid[1][LED_CENTER] = WHITE
    return grid


def screen_tracking(n: int):
    """'Who am I tracking' = blue pips on the top row + stem hanging below."""
    grid = _blank()
    for c in _pip_cols(n):
        grid[0][c] = BLUE
    grid[1][LED_CENTER] = BLUE
    grid[2][LED_CENTER] = BLUE
    return grid


def screen_reveal(target: int, bearing_col: int):
    """One-tap reveal: blue tracking pips on the top row laid over the live
    green bearing stem (which drops to the middle + bottom rows)."""
    grid = _blank()
    for r in (1, 2):
        grid[r][bearing_col] = GREEN
    for c in _pip_cols(target):
        grid[0][c] = BLUE
    return grid


def screen_combined(me: int, target: int):
    """Both readouts at once: blue tracking (top), white identity (bottom)."""
    grid = _blank()
    for c in _pip_cols(target):
        grid[0][c] = BLUE
    for c in _pip_cols(me):
        grid[2][c] = WHITE
    grid[1][LED_CENTER] = WHITE
    return grid


class GalleryRow(QFrame):
    """One gallery entry: a LED screen with an explanation beside it.

    With `flashing=True` the screen blinks on/off to show a "being set" state.
    """

    def __init__(
        self, grid, caption: str, explanation: str, flashing: bool = False
    ) -> None:
        super().__init__()
        self.setObjectName("row")
        self.setStyleSheet(ROW_STYLE)

        self._grid = grid
        self._panel = LedPanel(caption)
        self._panel.bar.set_grid(grid)
        info = InfoBox()
        info.label.setText(explanation)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        layout.addWidget(self._panel)
        layout.addWidget(info, stretch=1)

        if flashing:
            self._on = True
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._blink)
            self._timer.start(450)

    def _blink(self) -> None:
        self._on = not self._on
        self._panel.bar.set_grid(self._grid if self._on else _blank())


def build_gallery() -> QWidget:
    container = QWidget()
    container.setObjectName("root")
    container.setStyleSheet("QWidget#root { background-color: #1e1e1e; }")
    col = QVBoxLayout(container)
    col.setContentsMargins(16, 16, 16, 16)
    col.setSpacing(16)

    def header(text: str) -> None:
        label = QLabel(text)
        label.setStyleSheet("color: #ddd; font-size: 13pt; font-weight: 600;")
        col.addWidget(label)

    def row(grid, caption, explanation, flashing=False) -> None:
        col.addWidget(GalleryRow(grid, caption, explanation, flashing))

    header("Live position (green stem, all three rows)")
    row(screen_position(LED_CENTER), "ahead",
        "The peer is dead ahead, so the stem sits at the centre.")
    row(screen_position(LED_CENTER - 4), "left",
        "The peer is off to the left.")
    row(screen_position(0), "hard left",
        "The end column flags a peer past the frontal ±90° arc.")
    row(_blank(), "no peer",
        "Nothing is detected, so the bar stays dark.")

    header("Boot, setting identity (white, bottom row, flashing)")
    row(screen_identity(1), "I am 1 (setting)",
        "It powers up flashing here. Double-tap to step from 1 to 2 to 3, "
        "and it stops flashing once it commits on timeout. The pips are "
        "white so this never reads as a blue tracking screen.", flashing=True)
    row(screen_identity(3), "I am 3 (setting)",
        "The same flow stepped to 3. Odd counts sit a pip under the stem to "
        "form a tee, while even counts straddle it.", flashing=True)

    header("Default tracking (blue, top row, steady)")
    row(screen_tracking(2), "tracking 2",
        "After identity commits it drops into tracking mode. The screen is "
        "steady rather than flashing, because this is the committed state "
        "and not a prompt.")

    header("One tap, reveal who you are tracking (steady)")
    row(screen_reveal(2, LED_CENTER - 4), "reveal tracking 2",
        "Blue tracking pips ride the top row while the green bearing stem "
        "drops to the middle and bottom rows, so you still see direction. It "
        "holds for a few seconds and then returns to the plain bearing.")

    header("Two taps then pause, setting who you track (flashing)")
    row(screen_tracking(3), "setting tracking 3",
        "A double-tap enters set mode and the screen starts flashing. Keep "
        "double-tapping to cycle the target, and it commits on timeout.",
        flashing=True)

    header("Many taps, back to setting identity (flashing)")
    row(screen_identity(2), "re-set I am 2",
        "A long burst re-opens identity, which is rare. It uses the same "
        "white bottom flow as boot. A hold-shake entry would be safer here.",
        flashing=True)

    col.addStretch(1)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(container)
    scroll.setStyleSheet("QScrollArea { border: none; background-color: #1e1e1e; }")
    return scroll


class WorldView(QWidget):
    def __init__(self, world: World) -> None:
        super().__init__()
        self._world = world
        # The splitter handle resizes this pane, so grow to fill it rather
        # than locking to a fixed square. paintEvent already centres on the
        # live widget size.
        self.setMinimumSize(WORLD_PX, WORLD_PX)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
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
        scale = self._scale()
        half = WORLD_EXTENT_M / 2.0
        # World-square bounds; grid lines stay within these so a larger
        # viewport scales the grid up rather than spilling lines past it.
        left = cx - half * scale
        right = cx + half * scale
        top = cy - half * scale
        bottom = cy + half * scale

        p.setPen(QPen(QColor(255, 255, 255, 25), 1))
        x = -half
        while x <= half + 1e-6:
            sx = cx + x * scale
            p.drawLine(int(sx), int(top), int(sx), int(bottom))
            x += 0.5
        y = -half
        while y <= half + 1e-6:
            sy = cy - y * scale
            p.drawLine(int(left), int(sy), int(right), int(sy))
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

    def _scale(self) -> float:
        # Pixels per metre, sized so WORLD_EXTENT_M fills the smaller dimension.
        # Keeps the grid square and centred as the splitter resizes the pane.
        return min(self.width(), self.height()) / WORLD_EXTENT_M

    def _screen_to_world(self, pos) -> tuple[float, float]:
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        scale = self._scale()
        return ((pos.x() - cx) / scale, (cy - pos.y()) / scale)

    def _buddy_under(self, pos) -> Buddy | None:
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        scale = self._scale()
        hit_radius_px = 24.0
        best = None
        best_dist = hit_radius_px
        for buddy in self._world.buddies:
            sx = cx + buddy.x * scale
            sy = cy - buddy.y * scale
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

        # Device rows stack in their own pane below the splitter handle.
        devices = QWidget()
        self._devices = QVBoxLayout(devices)
        self._devices.setContentsMargins(0, 0, 0, 0)
        self._devices.setSpacing(16)
        self._devices.addStretch(1)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)
        splitter.setStyleSheet(
            """
            QSplitter::handle:vertical {
                background-color: #383838;
                margin: 2px 80px;
                border-radius: 2px;
            }
            QSplitter::handle:vertical:hover {
                background-color: #4a90d9;
            }
            """
        )
        splitter.addWidget(self._world_view)
        splitter.addWidget(devices)
        # Give spare space to the world view; the device rows keep their hint.
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([WORLD_PX, 200])

        root = QWidget()
        root.setObjectName("root")
        root.setStyleSheet("QWidget#root { background-color: #1e1e1e; }")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.addWidget(splitter)

        tabs = QTabWidget()
        tabs.addTab(root, "Live")
        tabs.addTab(build_gallery(), "UI Screens")
        self.setCentralWidget(tabs)
        self.resize(720, 600)

    def add_row(self, buddy: Buddy) -> None:
        row = DeviceRow(buddy)
        self._rows[buddy.id] = row
        self._devices.insertWidget(self._devices.count() - 1, row)

    def remove_row(self, buddy: Buddy) -> None:
        row = self._rows.pop(buddy.id, None)
        if row:
            self._devices.removeWidget(row)
            row.deleteLater()

    def refresh_row(self, buddy: Buddy) -> None:
        row = self._rows.get(buddy.id)
        if row:
            row.refresh()


class HarnessServer(QObject):
    buddy_connected = Signal(object)
    buddy_disconnected = Signal(object)

    def __init__(self, world: World, ring_count: int = 0) -> None:
        super().__init__()
        self._world = world
        self._ring_count = ring_count
        self._next_id = 0
        # Firmware emits a bearing line every loop; throttle the console echo.
        self._last_bearing_log: dict[int, float] = {}
        self._tcp = QTcpServer(self)
        self._tcp.newConnection.connect(self._on_new_connection)

    def listen(self) -> bool:
        return self._tcp.listen(QHostAddress(HOST), PORT)

    def _on_new_connection(self) -> None:
        sock = self._tcp.nextPendingConnection()
        x, y, heading_deg = self._initial_pose(self._next_id)
        buddy = Buddy(
            id=self._next_id,
            socket=sock,
            x=x,
            y=y,
            heading_deg=heading_deg,
        )
        self._next_id += 1

        sock.write(b"g")
        sock.readyRead.connect(lambda b=buddy: self._on_data(b))
        sock.disconnected.connect(lambda b=buddy: self._on_disconnect(b))

        self._world.add(buddy)
        self.buddy_connected.emit(buddy)

    def _initial_pose(self, index: int) -> tuple[float, float, float]:
        """Starting (x, y, heading) for the index-th device to connect.

        With a known ring count, devices sit evenly on a ring and face the
        world origin. Without one, fall back to a simple line along +x.
        """
        if self._ring_count <= 0:
            return float(index), 0.0, 0.0
        radius = 1.2
        theta = 2.0 * math.pi * (index % self._ring_count) / self._ring_count
        x = radius * math.sin(theta)
        y = radius * math.cos(theta)
        # Heading uses atan2(dx, dy) like the rest of the harness; aim the
        # vector from this device back at the origin.
        heading_deg = math.degrees(math.atan2(-x, -y)) % 360
        return x, y, heading_deg

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
        elif cmd == "oled":
            self._handle_oled(buddy, parts[1:])
        elif cmd == "rx":
            self._handle_rx(buddy, parts[1:])
        elif cmd == "bearing":
            self._handle_bearing(buddy, parts[1:])
        elif cmd == "taps":
            self._handle_taps(buddy)
        elif cmd == "heading":
            self._handle_heading(buddy)
        elif cmd == "log":
            print(f"buddy {buddy.id}: {line.strip()[4:]}", flush=True)

    def _handle_taps(self, buddy: Buddy) -> None:
        n = buddy.pending_taps
        buddy.pending_taps = 0
        buddy.socket.write(f"taps {n}\n".encode("ascii"))

    def _handle_heading(self, buddy: Buddy) -> None:
        # Mock the magnetometer with the device's true heading in the world, so
        # the firmware's compass tape scrolls as you re-aim it in the harness.
        buddy.socket.write(f"heading {buddy.heading_deg:.1f}\n".encode("ascii"))

    def _handle_strip(self, buddy: Buddy, values: list[str]) -> None:
        if len(values) != N_LEDS * LED_ROWS * 3:
            return
        try:
            flat = [int(v) for v in values]
        except ValueError:
            return
        px = list(zip(flat[::3], flat[1::3], flat[2::3]))
        buddy.leds = [px[r * N_LEDS:(r + 1) * N_LEDS] for r in range(LED_ROWS)]
        self._world.updated(buddy)

    def _handle_oled(self, buddy: Buddy, args: list[str]) -> None:
        # Line shape: `oled <w> <h> <hex>`, `<hex>` = 4 digits/pixel (big-endian
        # RGB565), row-major. Unpack to RGB888 with numpy for speed.
        if len(args) != 3:
            return
        try:
            w, h = int(args[0]), int(args[1])
        except ValueError:
            return
        hexblob = args[2]
        if w <= 0 or h <= 0 or len(hexblob) != w * h * 4:
            return
        try:
            raw = bytes.fromhex(hexblob)
        except ValueError:
            return
        v = np.frombuffer(raw, dtype=">u2").astype(np.uint32)
        r = (v >> 11) & 0x1F
        g = (v >> 5) & 0x3F
        b = v & 0x1F
        r8 = ((r << 3) | (r >> 2)).astype(np.uint8)
        g8 = ((g << 2) | (g >> 4)).astype(np.uint8)
        b8 = ((b << 3) | (b >> 2)).astype(np.uint8)
        buddy.oled_w = w
        buddy.oled_h = h
        buddy.oled_rgb = np.stack([r8, g8, b8], axis=1).tobytes()
        self._world.updated(buddy)

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
        delays = self._per_channel_delays(buddy)
        rx = buddy.channel.step(n_samples, per_channel_delays=delays)
        if rx.shape != (n_channels, n_samples) or rx.dtype != np.float32:
            rx = np.ascontiguousarray(rx[:n_channels, :n_samples], dtype=np.float32)
        buddy.socket.write(rx.tobytes())

    def _handle_bearing(self, buddy: Buddy, args: list[str]) -> None:
        if len(args) < 3:
            return
        try:
            bearing_deg = float(args[0])
            range_m = float(args[1])
            peak_avg = float(args[2])
        except ValueError:
            return
        now = time.monotonic()
        if now - self._last_bearing_log.get(buddy.id, 0.0) < 1.0:
            return
        self._last_bearing_log[buddy.id] = now
        target = self._find_target(buddy)
        if target is None:
            print(
                f"buddy {buddy.id}: bearing={bearing_deg:+.1f} "
                f"range={range_m:.2f} peak={peak_avg:.1f} (no target)",
                flush=True,
            )
            return
        true_bearing, true_range = self._compute_body_bearing_range(buddy, target)
        bearing_delta = ((bearing_deg - true_bearing + 180) % 360) - 180
        range_delta = range_m - true_range
        print(
            f"buddy {buddy.id}: bearing={bearing_deg:+6.1f}/{true_bearing:+6.1f} "
            f"(d={bearing_delta:+5.1f}) range={range_m:.2f}/{true_range:.2f} "
            f"(d={range_delta:+.2f}) peak={peak_avg:.1f}",
            flush=True,
        )

    def _per_channel_delays(self, buddy: Buddy) -> list[int] | None:
        target = self._find_target(buddy)
        if target is None:
            return None
        h_rad = math.radians(buddy.heading_deg)
        cos_h = math.cos(h_rad)
        sin_h = math.sin(h_rad)
        delays: list[int] = []
        for bx, by in RX_POSITIONS_BODY:
            wx = buddy.x + bx * cos_h + by * sin_h
            wy = buddy.y - bx * sin_h + by * cos_h
            range_m = math.hypot(target.x - wx, target.y - wy)
            delays.append(round(range_m / SOUND_SPEED_M_PER_S * SAMPLE_RATE_HZ))
        return delays

    def _on_disconnect(self, buddy: Buddy) -> None:
        self._world.remove(buddy)
        self.buddy_disconnected.emit(buddy)


def main() -> int:
    app = QApplication(sys.argv)

    ring_count = 0
    positionals = [a for a in sys.argv[1:] if not a.startswith("-")]
    if positionals:
        try:
            ring_count = int(positionals[0])
        except ValueError:
            print(f"ignoring non-integer count {positionals[0]!r}", file=sys.stderr)

    signal.signal(signal.SIGINT, lambda *_: app.quit())
    # Wake the Python interpreter periodically so it can dispatch signals
    # while Qt's event loop holds the main thread.
    sigint_timer = QTimer()
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start(100)

    world = World()
    server = HarnessServer(world, ring_count=ring_count)
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
