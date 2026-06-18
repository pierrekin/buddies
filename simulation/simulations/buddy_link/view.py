"""View extras for ``buddy_link``.

Every shot carries both cross-shot sweeps, so the same two curves show on
every shot with a marker on the current one:

  * bearing error vs range (log x), point vs wide -- the headline: wide
    degrades in the near field and converges to point by a couple of metres.
  * bearing error vs heading at the fixed far range, point vs wide -- the
    omnidirectional check.

A one-line summary of the current shot sits below.
"""

import math

import numpy as np
import pyqtgraph as pg

POINT_PEN = (120, 200, 100)   # green: single-element pinger
WIDE_PEN = (220, 80, 80)      # red: all-four broadside pinger
MARKER_PEN = pg.mkPen("y", style=pg.QtCore.Qt.PenStyle.DashLine)
ZERO_PEN = pg.mkPen((180, 180, 180), style=pg.QtCore.Qt.PenStyle.DotLine)

MODE_PENS = {"point": POINT_PEN, "wide": WIDE_PEN}
MODE_SYMBOL = {"point": "o", "wide": "s"}


def extra_views(viewer, layout, start_row):
    extras = viewer.shot.extras
    row = start_row
    modes = extras.get("modes") or ["point", "wide"]
    exp = extras.get("this_experiment")

    if extras.get("range_m"):
        x = np.asarray(extras["range_m"], dtype=np.float64)
        cur = extras.get("this_range_m") if exp == "range" else None
        _curve(layout, row, x, extras, "range_bearing_err_deg", modes,
               f"Bearing error vs range (heading {extras.get('range_heading_deg', 0):.0f} deg)",
               "true range", "m", cur, logx=True); row += 1

    if extras.get("az_deg"):
        x = np.asarray(extras["az_deg"], dtype=np.float64)
        cur = extras.get("this_heading_deg") if exp == "azimuth" else None
        _curve(layout, row, x, extras, "az_bearing_err_deg", modes,
               f"Bearing error vs heading (range {extras.get('az_range_m', 0):.0f} m)",
               "true heading", "deg", cur); row += 1

    if extras.get("this_experiment"):
        _summary(layout, row, extras); row += 1


def _curve(layout, row, x, extras, key, modes, title, xlabel, xunits, current,
           logx=False):
    plot = layout.addPlot(row=row, col=0)
    nf = extras.get("nearfield_m")
    sub = f"  (near-field aperture^2/lambda = {nf*1e3:.0f} mm)" if nf else ""
    plot.setTitle(title + sub)
    plot.setLabel("left", "|bearing error|", units="deg")
    plot.setLabel("bottom", xlabel, units=xunits)
    plot.addLegend()
    if logx:
        plot.setLogMode(x=True, y=False)
    plot.addItem(pg.InfiniteLine(pos=0.0, angle=0, pen=ZERO_PEN))
    for m in modes:
        y = np.abs(np.asarray(extras.get(f"{key}_{m}", []), dtype=np.float64))
        if y.size != x.size:
            continue
        plot.plot(x, y, pen=pg.mkPen(MODE_PENS.get(m, (200, 200, 200)), width=2),
                  symbol=MODE_SYMBOL.get(m, "o"),
                  symbolBrush=MODE_PENS.get(m, (200, 200, 200)), name=m)
    if isinstance(current, (int, float)):
        pos = math.log10(current) if logx and current > 0 else current
        plot.addItem(pg.InfiniteLine(pos=float(pos), angle=90, pen=MARKER_PEN))


def _summary(layout, row, extras):
    sent = extras.get("bits_sent") or []
    rec = extras.get("this_bits_rx") or []
    plot = layout.addPlot(row=row, col=0)
    plot.setTitle(
        f"[{extras.get('this_experiment', '?')}] R={extras.get('this_range_m', 0):.2f} m  "
        f"heading={extras.get('this_heading_deg', 0):.0f} deg  {extras.get('this_mode', '?')}  |  "
        f"bearing_err={extras.get('this_bearing_err_deg', 0):+.2f} deg  "
        f"BER={extras.get('this_ber', 0):.2f} eye={extras.get('this_eye', 0):.2f}  |  "
        f"sent {''.join(str(int(b)) for b in sent)} recv {''.join(str(int(b)) for b in rec)}"
    )
    plot.hideAxis("left")
    plot.hideAxis("bottom")
