"""Frequency-domain view for ``characterise``.

Pulls the FFT of the mic trace from ``extras`` and plots magnitude vs
frequency (log Y), with a vertical marker at the probe centre. Lets you
read off the channel's effective bandwidth at a glance."""

import numpy as np
import pyqtgraph as pg


def extra_views(viewer, layout, start_row):
    extras = viewer.shot.extras
    freqs = extras.get("spectrum_freqs")
    mag = extras.get("spectrum_mag")
    if freqs is None or mag is None:
        return

    plot = layout.addPlot(row=start_row, col=0)
    plot.setLabel("left", "|RX| (a.u., log)")
    plot.setLabel("bottom", "frequency", units="Hz")
    plot.setLogMode(x=False, y=True)
    # Drop the DC bin to avoid log(0); cap the upper end at 4x the probe
    # frequency so the interesting band fills the plot.
    probe_freq = float(extras.get("probe_freq", 0.0))
    upper = 4 * probe_freq if probe_freq > 0 else float(freqs[-1])
    keep = (freqs > 0) & (freqs <= upper)
    plot.plot(np.asarray(freqs[keep]), np.asarray(mag[keep]) + 1e-12)

    if probe_freq > 0:
        plot.addItem(pg.InfiniteLine(
            pos=probe_freq, angle=90,
            pen=pg.mkPen("g", style=pg.QtCore.Qt.PenStyle.DashLine),
        ))
