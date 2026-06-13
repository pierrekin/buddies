"""Viewer for capture files: animated pressure field with channel overlays
(vector arrows, color markers) and synced scalar plots."""

import argparse
import sys

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from buddies import capture

DEFAULT_FPS = 60.0
# Display levels are set to this percentile of |p|. Scaling to the global
# maximum (the source peak) would render the decaying wavefront nearly
# invisible.
LEVEL_PERCENTILE = 99.5
# Computing the percentile over every frame of a multi-GB capture is slow;
# this many frames, evenly spaced, estimate it.
LEVEL_SAMPLE_FRAMES = 32
COLORMAP = "CET-D1A"  # diverging blue-white-red
OVERLAY_PEN = "g"
WINDOW_SIZE = (900, 950)
FIELD_ROW_STRETCH = 4  # field view height relative to each scalar plot row
# A vector channel's 95th-percentile magnitude is drawn at this fraction of
# the domain size.
VECTOR_LENGTH_FRACTION = 0.1


class Viewer(QtWidgets.QWidget):
    def __init__(self, title, cap, fps):
        super().__init__()
        self.cap = cap
        self.frames = cap.frames
        self.nframes = len(cap.frames)
        self._frame_hooks = []  # called with the current time on frame change

        self.setWindowTitle(
            f"{title} | {self.nframes} frames | "
            f"{cap.frames.shape[1]}x{cap.frames.shape[2]} cells | "
            f"{len(cap.channels)} channels"
        )
        self.resize(*WINDOW_SIZE)

        glw = pg.GraphicsLayoutWidget()
        self._build_field_view(glw)
        plot_row = 1
        for ch in cap.channels:
            if len(ch.values) == 0:
                print(f"warning: channel {ch.name!r} is empty, skipping", file=sys.stderr)
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
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.nframes - 1)
        self.slider.valueChanged.connect(self.set_frame)
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
        self.domain = (nx * self.cap.dx, ny * self.cap.dx)
        plot = glw.addPlot(row=0, col=0)
        plot.setAspectLocked(True)
        plot.setLabel("bottom", "x", units="m")
        plot.setLabel("left", "y", units="m")
        self.field_plot = plot

        sample = self.frames[:: max(1, self.nframes // LEVEL_SAMPLE_FRAMES)]
        lim = float(np.percentile(np.abs(sample), LEVEL_PERCENTILE))
        self.cmap = pg.colormap.get(COLORMAP)
        self.img = pg.ImageItem(self.frames[0])
        self.img.setLookupTable(self.cmap.getLookupTable(nPts=256))
        self.img.setLevels((-lim, lim))
        self.img.setRect(QtCore.QRectF(0, 0, *self.domain))
        plot.addItem(self.img)

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
        shaft = pg.PlotCurveItem(pen=pg.mkPen(OVERLAY_PEN, width=2))
        tip = pg.ScatterPlotItem(size=7, pen=None, brush=OVERLAY_PEN)
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
        t = i * self.cap.dt
        for hook in self._frame_hooks:
            hook(t)
        self.slider.blockSignals(True)
        self.slider.setValue(i)
        self.slider.blockSignals(False)
        self.time_label.setText(f"{t * 1e3:.3f} ms  {i + 1}/{self.nframes}")

    def _advance(self):
        self.set_frame((self._frame + 1) % self.nframes)

    def toggle(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.timer.start()
            self.play_button.setText("Pause")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", help="capture file to view")
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS, help="playback rate (frames/s)")
    args = ap.parse_args()

    app = pg.mkQApp("FDTD viewer")
    viewer = Viewer(args.run, capture.load(args.run), args.fps)
    viewer.show()
    app.exec()


if __name__ == "__main__":
    main()
