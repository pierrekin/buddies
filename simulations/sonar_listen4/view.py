"""Range-azimuth waterfall for ``sonar_listen4``.

Reads the (angle, range) energy map from ``extras`` and renders it as a 2D
image on a dB scale -- bearing across, range down. The same data already
appears rasterised onto the field overlay; this is the same map shown
without the Cartesian projection, so you can read off range bins exactly."""

import numpy as np
import pyqtgraph as pg


def extra_views(viewer, layout, start_row):
    extras = viewer.st.extras
    energy = extras.get("energy_map")
    angles = extras.get("angles_deg")
    ranges = extras.get("range_bins_m")
    if energy is None or angles is None or ranges is None:
        return

    energy = np.asarray(energy)
    emax = float(energy.max()) if energy.size else 0.0
    if emax <= 0:
        return

    db = 10 * np.log10(np.where(energy > 0, energy / emax, 1e-30))
    span = float(extras.get("color_span_db", 30.0))

    plot = layout.addPlot(row=start_row, col=0)
    plot.setLabel("left", "range", units="m")
    plot.setLabel("bottom", "bearing", units="deg")
    plot.invertY(True)  # range grows downward, sonar-display convention

    img = pg.ImageItem(db.T)  # T: (range, angle) so x=angle, y=range
    a0, a1 = float(angles[0]), float(angles[-1])
    r0, r1 = float(ranges[0]), float(ranges[-1])
    img.setRect(pg.QtCore.QRectF(a0, r0, a1 - a0, r1 - r0))
    img.setLevels((-span, 0.0))
    img.setColorMap(pg.colormap.get("CET-L4"))
    plot.addItem(img)
