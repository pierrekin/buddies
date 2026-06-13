"""Viewer for processed artifacts: animated pressure field with channel
overlays (vector arrows, color markers) and synced scalar plots.

Frames are uint8 normalized to the level baked in at process time, so the
field maps straight through the colormap with no per-open level scan."""

import signal

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

DEFAULT_FPS = 60.0
DEFAULT_COLORMAP = "CET-D1A"  # diverging blue-white-red, if meta lacks one
OVERLAY_PEN = "g"
WINDOW_SIZE = (900, 950)
FIELD_ROW_STRETCH = 4  # field view height relative to each scalar plot row
# A vector channel's 95th-percentile magnitude is drawn at this fraction of
# the domain size.
VECTOR_LENGTH_FRACTION = 0.1
# Vertical distance (px) from the slider at which fine-scrub sensitivity is
# halved; sensitivity falls off as FALLOFF / (FALLOFF + distance).
FINE_SCRUB_FALLOFF = 80.0


class _JumpSliderStyle(QtWidgets.QProxyStyle):
    """Makes a left-click on the slider groove snap the handle to the cursor
    and treat the press as a drag, instead of the default page-step jump."""

    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QtWidgets.QStyle.StyleHint.SH_Slider_AbsoluteSetButtons:
            return int(QtCore.Qt.MouseButton.LeftButton.value)
        return super().styleHint(hint, option, widget, returnData)


class FineSlider(QtWidgets.QSlider):
    """A slider whose drag sensitivity drops as the cursor moves vertically
    away from the groove, allowing frame-accurate scrubbing. A click still
    snaps to the cursor (via the proxy style); from there the drag is relative,
    so moving the mouse up or down before dragging trades speed for precision."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._float_val = 0.0
        self._last_x = 0.0

    def mousePressEvent(self, ev):
        # Let the base class snap to the click and emit sliderPressed, then
        # anchor the relative accumulator at that position.
        super().mousePressEvent(ev)
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self._float_val = float(self.value())
            self._last_x = ev.position().x()

    def mouseMoveEvent(self, ev):
        if not self.isSliderDown():
            super().mouseMoveEvent(ev)
            return
        pos = ev.position()
        dx = pos.x() - self._last_x
        self._last_x = pos.x()
        # Distance of the cursor above or below the slider's own height.
        dy = max(0.0, -pos.y(), pos.y() - self.height())
        factor = FINE_SCRUB_FALLOFF / (FINE_SCRUB_FALLOFF + dy)
        span = self.maximum() - self.minimum()
        self._float_val += dx * (span / max(1, self.width())) * factor
        self._float_val = min(self.maximum(), max(self.minimum(), self._float_val))
        self.setValue(round(self._float_val))
        ev.accept()


class Viewer(QtWidgets.QWidget):
    def __init__(self, title, st, fps):
        super().__init__()
        self.st = st
        self.frames = st.frames
        self.nframes = len(st.frames)
        self.dt = st.dt
        self._frame_hooks = []  # called with the current time on frame change

        self.setWindowTitle(
            f"{title} | {self.nframes} frames | "
            f"{st.frames.shape[1]}x{st.frames.shape[2]} cells | "
            f"{len(st.channels)} channels"
        )
        self.resize(*WINDOW_SIZE)

        glw = pg.GraphicsLayoutWidget()
        self._build_field_view(glw)
        plot_row = 1
        for ch in st.channels:
            if len(ch.values) == 0:
                print(f"warning: channel {ch.name!r} is empty, skipping")
                continue
            if ch.kind == "scalar":
                self._add_scalar(glw, plot_row, ch)
                plot_row += 1
            elif ch.kind == "vector":
                self._add_vector(ch)
            elif ch.kind == "color":
                self._add_color(ch)
            else:
                raise ValueError(f"channel {ch.name!r} has unknown kind {ch.kind!r}")
        glw.ci.layout.setRowStretchFactor(0, FIELD_ROW_STRETCH)

        self.play_button = QtWidgets.QPushButton("Pause")
        self.play_button.clicked.connect(self.toggle)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Space), self, self.toggle)
        self.slider = FineSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.nframes - 1)
        self.slider.valueChanged.connect(self.set_frame)
        self.slider.sliderPressed.connect(self._scrub_start)
        self.slider.sliderReleased.connect(self._scrub_end)
        self._slider_style = _JumpSliderStyle(self.slider.style())
        self.slider.setStyle(self._slider_style)
        self._resume_after_scrub = False
        self.time_label = QtWidgets.QLabel()

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(self.slider, stretch=1)
        controls.addWidget(self.time_label)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(glw, stretch=1)
        layout.addLayout(controls)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(round(1000 / fps))
        self.timer.timeout.connect(self._advance)
        self.set_frame(0)
        self.timer.start()

    def _build_field_view(self, glw):
        _, nx, ny = self.frames.shape
        self.domain = (nx * self.st.dx, ny * self.st.dx)
        plot = glw.addPlot(row=0, col=0)
        plot.setAspectLocked(True)
        plot.setLabel("bottom", "x", units="m")
        plot.setLabel("left", "y", units="m")
        self.field_plot = plot

        self.cmap = pg.colormap.get(self.st.meta.get("colormap", DEFAULT_COLORMAP))
        self.img = pg.ImageItem(self.frames[0])
        self.img.setLookupTable(self.cmap.getLookupTable(nPts=256))
        # uint8 0..255 already spans the baked level; 128 = zero pressure.
        self.img.setLevels((0, 255))
        self.img.setRect(QtCore.QRectF(0, 0, *self.domain))
        plot.addItem(self.img)

        if self.st.overlay is not None:
            overlay = pg.ImageItem(self.st.overlay)
            overlay.setRect(QtCore.QRectF(0, 0, *self.domain))
            plot.addItem(overlay)

    def _label(self, ch):
        if not ch.name:
            return
        text = pg.TextItem(ch.name, color=OVERLAY_PEN, anchor=(0, 1))
        text.setPos(*ch.pos)
        self.field_plot.addItem(text)

    def _sampler(self, ch):
        values = np.asarray(ch.values)

        def at(t):
            return values[min(len(values) - 1, max(0, round(t / ch.dt)))]

        return at

    def _add_scalar(self, glw, row, ch):
        plot = glw.addPlot(row=row, col=0)
        plot.setLabel("left", ch.name)
        plot.setLabel("bottom", "t", units="s")
        t = np.arange(len(ch.values)) * ch.dt
        plot.plot(t, np.asarray(ch.values))
        cursor = pg.InfiniteLine(angle=90, pen=OVERLAY_PEN)
        plot.addItem(cursor)
        self._frame_hooks.append(cursor.setPos)
        if ch.pos is not None:
            marker = pg.ScatterPlotItem(
                [ch.pos[0]], [ch.pos[1]], symbol="o", size=8, pen=OVERLAY_PEN, brush=None
            )
            self.field_plot.addItem(marker)
            self._label(ch)

    def _add_vector(self, ch):
        if ch.scale is not None:
            scale = ch.scale
        else:
            magnitudes = np.hypot(*np.asarray(ch.values).T)
            ref = float(np.percentile(magnitudes, 95))
            scale = VECTOR_LENGTH_FRACTION * max(self.domain) / ref if ref > 0 else 0.0
        color = ch.color if ch.color is not None else OVERLAY_PEN
        shaft = pg.PlotCurveItem(pen=pg.mkPen(color, width=2))
        tip = pg.ScatterPlotItem(size=7, pen=None, brush=pg.mkBrush(color))
        self.field_plot.addItem(shaft)
        self.field_plot.addItem(tip)
        self._label(ch)
        x0, y0 = ch.pos
        sample = self._sampler(ch)

        def update(t):
            vx, vy = sample(t) * scale
            shaft.setData([x0, x0 + vx], [y0, y0 + vy])
            tip.setData([x0 + vx], [y0 + vy])

        self._frame_hooks.append(update)

    def _add_color(self, ch):
        values = np.asarray(ch.values)
        lo, hi = float(values.min()), float(values.max())
        span = hi - lo if hi > lo else 1.0
        marker = pg.ScatterPlotItem(
            [ch.pos[0]], [ch.pos[1]], symbol="s", size=14, pen=OVERLAY_PEN
        )
        self.field_plot.addItem(marker)
        self._label(ch)
        sample = self._sampler(ch)

        def update(t):
            marker.setBrush(self.cmap.map((float(sample(t)) - lo) / span, mode="qcolor"))

        self._frame_hooks.append(update)

    def set_frame(self, i):
        self._frame = i
        self.img.setImage(self.frames[i], autoLevels=False)
        t = i * self.dt
        for hook in self._frame_hooks:
            hook(t)
        self.slider.blockSignals(True)
        self.slider.setValue(i)
        self.slider.blockSignals(False)
        self.time_label.setText(f"{t * 1e3:.3f} ms  {i + 1}/{self.nframes}")

    def _advance(self):
        self.set_frame((self._frame + 1) % self.nframes)

    def _scrub_start(self):
        # Pause while dragging, remembering whether to resume on release.
        self._resume_after_scrub = self.timer.isActive()
        if self._resume_after_scrub:
            self.toggle()

    def _scrub_end(self):
        if self._resume_after_scrub:
            self.toggle()
            self._resume_after_scrub = False

    def toggle(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.timer.start()
            self.play_button.setText("Pause")


def launch(st, title="capture", fps=DEFAULT_FPS):
    app = pg.mkQApp("FDTD viewer")
    viewer = Viewer(title, st, fps)
    viewer.show()

    # Qt's event loop runs in C++ and won't deliver SIGINT to Python until the
    # interpreter regains control, so a bare Ctrl-C is ignored while the window
    # is up. Quit the app on SIGINT and keep a no-op timer ticking so the
    # interpreter wakes often enough to actually run the handler.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    sigint_timer = QtCore.QTimer()
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start(100)

    app.exec()
