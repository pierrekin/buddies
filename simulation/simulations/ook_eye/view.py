"""Eye-diagram view for ``ook_eye``.

Reads the raw mic channel from the Store, picks up ``bit_duration`` and
``first_arrival_sample`` from ``store.extras``, and folds the trace into
overlaid 2-bit-window traces -- the classic eye diagram. Demod stats from
``extras`` annotate the plot."""

import numpy as np
import pyqtgraph as pg


def extra_views(viewer, layout, start_row):
    shot = viewer.shot
    mic = next((c for c in shot.channels if c.name.startswith("mic")), None)
    if mic is None:
        return  # nothing to fold
    period = shot.extras.get("bit_duration")
    if period is None:
        return  # not an eye-capable shot

    first = int(shot.extras.get("first_arrival_sample", 0))
    values = np.asarray(mic.values, dtype=np.float32)[first:]
    dt = mic.dt
    spp = max(1, int(round(period / dt)))
    chunk_len = 2 * spp
    n_chunks = max(0, (len(values) - chunk_len) // spp + 1)
    t = np.arange(chunk_len) * dt

    plot = layout.addPlot(row=start_row, col=0)
    plot.setLabel("left", "RX (Pa)")
    plot.setLabel("bottom", "t (folded)", units="s")
    pen = pg.mkPen(color=(80, 200, 255, 80))
    for i in range(n_chunks):
        plot.plot(t, values[i * spp : i * spp + chunk_len], pen=pen)

    # Slicer time = the middle of the second bit window in the 2-period frame.
    plot.addItem(pg.InfiniteLine(
        pos=1.5 * period, angle=90,
        pen=pg.mkPen("g", style=pg.QtCore.Qt.PenStyle.DashLine),
    ))
    # Threshold from the demod, if the sim recorded one.
    threshold = shot.extras.get("slicer_threshold")
    if threshold is not None:
        plot.addItem(pg.InfiniteLine(
            pos=threshold, angle=0,
            pen=pg.mkPen("y", style=pg.QtCore.Qt.PenStyle.DashLine),
        ))
