"""Viewer for processed artifacts: animated pressure field per shot, with
channel overlays and synced scalar plots.

The viewer opens one shot at a time; a combobox switches between them.
Each shot owns its own frames trajectory, channels, overlay, and extras
view, so switching rebuilds the layout from scratch."""

import re
import signal

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

# Channel names by convention end in " (unit)" -- e.g. "RX truth (V)". The
# y-axis gets just the unit so pyqtgraph can SI-prefix the tick labels
# (mV, µV, ...); the full name lives in the plot title.
_UNIT_SUFFIX = re.compile(r"\(([^()]+)\)\s*$")

DEFAULT_FPS = 60.0
DEFAULT_COLORMAP = "CET-D1A"  # diverging blue-white-red, if meta lacks one
OVERLAY_PEN = "g"
WINDOW_SIZE = (900, 950)
# Minimum heights for each row inside the scrollable layout. The layout's
# overall minimum height is the sum of its rows', so adding more channels
# pushes the stack past the viewport and a scrollbar appears. The field
# row is much taller because it carries the aspect-locked pressure image.
FIELD_ROW_MIN_HEIGHT = 480
SCALAR_ROW_MIN_HEIGHT = 170
EXTRA_ROW_MIN_HEIGHT = 280
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
        dy = max(0.0, -pos.y(), pos.y() - self.height())
        factor = FINE_SCRUB_FALLOFF / (FINE_SCRUB_FALLOFF + dy)
        span = self.maximum() - self.minimum()
        self._float_val += dx * (span / max(1, self.width())) * factor
        self._float_val = min(self.maximum(), max(self.minimum(), self._float_val))
        self.setValue(round(self._float_val))
        ev.accept()


class Viewer(QtWidgets.QWidget):
    def __init__(self, title, st, fps, extra_views=None):
        super().__init__()
        self.st = st
        self.title = title
        self.extra_views = extra_views
        self.dt = st.dt
        self.resize(*WINDOW_SIZE)

        # Per-shot state, populated by _load_shot.
        self.shot = None
        self.frames = None
        self.nframes = 0
        self.domain = None
        self.field_plot = None
        self.img = None
        self._frame_hooks = []
        self._frame = 0

        # Chrome built once; per-shot widgets live inside ``self.glw`` and
        # are torn down + rebuilt by _load_shot. The whole layout (field +
        # channels + extras) lives in one scroll area; row min-heights keep
        # plots readable and the field scrolls with the rest.
        self.glw = pg.GraphicsLayoutWidget()
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidget(self.glw)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self.shot_combo = QtWidgets.QComboBox()
        for name in st.shots:
            self.shot_combo.addItem(name)
        self.shot_combo.currentTextChanged.connect(self._on_shot_changed)
        # A single-shot artifact doesn't need a selector taking up the toolbar.
        self.shot_combo.setVisible(len(st.shots) > 1)

        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.clicked.connect(self.toggle)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Space), self, self.toggle)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Left), self, lambda: self.step(-1))
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Right), self, lambda: self.step(1))

        self.slider = FineSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.set_frame)
        self.slider.sliderPressed.connect(self._scrub_start)
        self.slider.sliderReleased.connect(self._scrub_end)
        # QProxyStyle takes ownership of any base style handed to it, so passing
        # the slider's shared application style would double-free it at teardown
        # (segfault on quit). The default constructor proxies the app style
        # without claiming ownership.
        self._slider_style = _JumpSliderStyle()
        self.slider.setStyle(self._slider_style)
        self._resume_after_scrub = False
        self.time_label = QtWidgets.QLabel()

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.shot_combo)
        controls.addWidget(self.play_button)
        controls.addWidget(self.slider, stretch=1)
        controls.addWidget(self.time_label)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.scroll, stretch=1)
        layout.addLayout(controls)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(round(1000 / fps))
        self.timer.timeout.connect(self._advance)

        # Initial shot = the first one the artifact lists.
        self._load_shot(next(iter(st.shots)))

    def _on_shot_changed(self, name):
        if name:
            self._load_shot(name)

    def _load_shot(self, name):
        # Pause playback before tearing down the field plot the timer drives.
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")

        self.shot = self.st.shots[name]
        self.frames = self.shot.frames
        self.nframes = len(self.frames) if self.frames is not None else 0
        self._frame_hooks = []
        self.field_plot = None
        self.img = None
        self.domain = None

        self.glw.clear()
        grid = self.glw.ci.layout

        plot_row = 0
        if self.frames is not None:
            self._build_field_view()
            grid.setRowMinimumHeight(plot_row, FIELD_ROW_MIN_HEIGHT)
            plot_row += 1

        for ch in self.shot.channels:
            if len(ch.values) == 0:
                print(f"warning: channel {ch.name!r} is empty, skipping")
                continue
            if ch.kind == "scalar":
                self._add_scalar(plot_row, ch)
                grid.setRowMinimumHeight(plot_row, SCALAR_ROW_MIN_HEIGHT)
                plot_row += 1
            elif ch.kind == "vector":
                if self.field_plot is not None:
                    self._add_vector(ch)
            elif ch.kind == "color":
                if self.field_plot is not None:
                    self._add_color(ch)
            else:
                raise ValueError(f"channel {ch.name!r} has unknown kind {ch.kind!r}")

        extras_start = plot_row
        if self.extra_views is not None:
            # Sim-specific hook: gets this viewer (carries the current shot
            # as ``viewer.shot``), the layout, and the next free row.
            self.extra_views(self, self.glw, plot_row)
            # Anything the extras added gets the extras min height -- they're
            # typically bar charts / spectra / heatmaps with their own legends
            # and axis ticks and want a bit more vertical room than a trace.
            for r in range(extras_start, grid.rowCount()):
                grid.setRowMinimumHeight(r, EXTRA_ROW_MIN_HEIGHT)

        # Force the GraphicsLayoutWidget tall enough that all rows hit their
        # minimum heights; the scroll area then provides a scrollbar instead
        # of compressing every plot into a few pixels. Ask the QGraphicsGrid
        # itself rather than summing row mins -- that way contents margins
        # and inter-row spacing are accounted for, otherwise the top/bottom
        # rows get clipped at the scroll extremes.
        grid.invalidate()
        hint = grid.effectiveSizeHint(QtCore.Qt.SizeHint.MinimumSize)
        # Small extra to absorb the QGraphicsView frame and pixel rounding.
        self.glw.setMinimumHeight(int(hint.height()) + 8)

        # Window title + slider range reflect the current shot.
        nch = sum(1 for c in self.shot.channels if len(c.values) > 0)
        if self.frames is not None:
            nx, ny = self.frames.shape[1:]
            field_part = f"{self.nframes} frames | {nx}x{ny} cells"
        else:
            field_part = "no frames"
        shot_part = f"shot: {name}" if len(self.st.shots) > 1 else ""
        self.setWindowTitle(" | ".join(
            x for x in [self.title, shot_part, field_part, f"{nch} channels"] if x
        ))

        playable = self.nframes > 1
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, self.nframes - 1))
        self.slider.blockSignals(False)
        self.slider.setEnabled(playable)
        self.play_button.setEnabled(playable)

        self.set_frame(0)

    def _build_field_view(self):
        _, nx, ny = self.frames.shape
        self.domain = (nx * self.st.dx, ny * self.st.dx)
        plot = self.glw.addPlot(row=0, col=0)
        plot.setAspectLocked(True)
        plot.setLabel("bottom", "x", units="m")
        plot.setLabel("left", "y", units="m")
        # Pin the view to the domain once. Overlay graphics (vector arrows,
        # markers, labels) can extend past the field, but auto-ranging to fit
        # them would rescale the axes mid-playback, so freeze the range here.
        plot.setRange(xRange=(0, self.domain[0]), yRange=(0, self.domain[1]), padding=0)
        plot.disableAutoRange()
        self.field_plot = plot

        self.cmap = pg.colormap.get(self.st.meta.get("colormap", DEFAULT_COLORMAP))
        self.img = pg.ImageItem(self.frames[0])
        self.img.setLookupTable(self.cmap.getLookupTable(nPts=256))
        self.img.setLevels((0, 255))
        self.img.setRect(QtCore.QRectF(0, 0, *self.domain))
        plot.addItem(self.img)

        if self.shot.overlay is not None:
            overlay = pg.ImageItem(self.shot.overlay)
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

    def _add_scalar(self, row, ch):
        plot = self.glw.addPlot(row=row, col=0)
        # Full descriptive name in the title (carries the meaning), short
        # unit on the y-axis so pyqtgraph SI-prefixes tick labels.
        plot.setTitle(ch.name, size="9pt")
        unit_match = _UNIT_SUFFIX.search(ch.name)
        if unit_match:
            plot.setLabel("left", units=unit_match.group(1))
        plot.setLabel("bottom", "t", units="s")
        t = np.arange(len(ch.values)) * ch.dt
        plot.plot(t, np.asarray(ch.values))
        cursor = pg.InfiniteLine(angle=90, pen=OVERLAY_PEN)
        plot.addItem(cursor)
        self._frame_hooks.append(cursor.setPos)
        if ch.pos is not None and self.field_plot is not None:
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
        if self.img is not None and self.nframes > 0:
            self.img.setImage(self.frames[i], autoLevels=False)
        t = i * self.dt
        for hook in self._frame_hooks:
            hook(t)
        self.slider.blockSignals(True)
        self.slider.setValue(i)
        self.slider.blockSignals(False)
        if self.nframes > 0:
            self.time_label.setText(f"{t * 1e3:.3f} ms  {i + 1}/{self.nframes}")
        else:
            self.time_label.setText("no frames")

    def _advance(self):
        if self.nframes > 0:
            self.set_frame((self._frame + 1) % self.nframes)

    def step(self, delta):
        if self.nframes == 0:
            return
        if self.timer.isActive():
            self.toggle()
        self.set_frame((self._frame + delta) % self.nframes)

    def _scrub_start(self):
        self._resume_after_scrub = self.timer.isActive()
        if self._resume_after_scrub:
            self.toggle()

    def _scrub_end(self):
        if self._resume_after_scrub:
            self.toggle()
            self._resume_after_scrub = False

    def toggle(self):
        if self.nframes == 0:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.timer.start()
            self.play_button.setText("Pause")


def launch(st, title="capture", fps=DEFAULT_FPS, extra_views=None):
    """Open the viewer on a Store. ``extra_views`` is an optional callable a
    sim's ``view.py`` can provide; it gets ``(viewer, layout, start_row)``
    each time a shot is loaded. It can read ``viewer.shot`` for the current
    shot's data, add its own pyqtgraph widgets, and register frame hooks
    via ``viewer._frame_hooks.append(...)``."""
    app = pg.mkQApp("FDTD viewer")
    viewer = Viewer(title, st, fps, extra_views=extra_views)
    viewer.show()

    signal.signal(signal.SIGINT, lambda *_: app.quit())
    sigint_timer = QtCore.QTimer()
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start(100)

    app.exec()
