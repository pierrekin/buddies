"""View extras for ``model_comms_wreck``.

Per shot:

  * BER vs SNR waterfall (log y, dB x) with one curve per TX position.
  * BER vs noise sigma (log y, log x) with the same per-position curves.
  * Per-shot BER bar trio (phys, model, disagreement) with SNR + position
    label in the title.

Per-position colour code (consistent across both axes):

  * blue   = LOS
  * amber  = partial shadow
  * red    = NLOS
  * purple = floor"""

import math

import numpy as np
import pyqtgraph as pg

POS_COLORS = {
    "los":     (80, 140, 220),   # blue
    "partial": (220, 150, 60),   # amber
    "nlos":    (220, 80, 80),    # red
    "floor":   (180, 100, 220),  # purple
}
MARKER_PEN = pg.mkPen("y", style=pg.QtCore.Qt.PenStyle.DashLine)
BER_FLOOR = 1e-3


def extra_views(viewer, layout, start_row):
    extras = viewer.shot.extras
    role = extras.get("role")
    row = start_row

    if extras.get("sweep_sigmas"):
        _add_waterfall(layout, row, extras)
        row += 1
        _add_ber_vs_sigma(layout, row, extras)
        row += 1
    if role == "comms":
        _add_per_shot_bars(layout, row, extras)
        row += 1


def _pen(pname):
    return POS_COLORS.get(pname, (180, 180, 180))


def _add_waterfall(layout, row, extras):
    plot = layout.addPlot(row=row, col=0)
    plot.setTitle("BER vs SNR waterfall  (log y, dB x)  --  one curve per TX position")
    plot.setLabel("left", "BER")
    plot.setLabel("bottom", "SNR at RX", units="dB")
    plot.setLogMode(x=False, y=True)
    plot.addLegend()

    for pname, plabel in zip(extras["tx_pos_names"], extras["tx_pos_labels"]):
        snr = np.asarray(extras[f"sweep_{pname}_snr_db"], dtype=np.float64)
        ber = np.asarray(extras[f"sweep_{pname}_ber_phys"], dtype=np.float64)
        if snr.size == 0:
            continue
        ber_plot = np.where(ber > 0, ber, BER_FLOOR)
        pen = _pen(pname)
        plot.plot(snr, ber_plot, pen=pg.mkPen(pen, width=2),
                  symbol="o", symbolBrush=pen, name=plabel)

    if extras.get("role") == "comms":
        current_snr = extras.get("snr_db")
        if current_snr is not None and math.isfinite(current_snr):
            plot.addItem(pg.InfiniteLine(
                pos=float(current_snr), angle=90, pen=MARKER_PEN,
            ))


def _add_ber_vs_sigma(layout, row, extras):
    sigmas = np.asarray(extras["sweep_sigmas"], dtype=np.float64)
    sigmas_plot = np.where(sigmas > 0, sigmas, 1e-12)

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle("BER vs ambient noise sigma  (log y, log x)  --  "
                  "shadowed positions degrade earlier (lower SNR at same sigma)")
    plot.setLabel("left", "BER")
    plot.setLabel("bottom", "noise source RMS (volume rate)")
    plot.setLogMode(x=True, y=True)
    plot.addLegend()

    for pname, plabel in zip(extras["tx_pos_names"], extras["tx_pos_labels"]):
        ber = np.asarray(extras[f"sweep_{pname}_ber_phys"], dtype=np.float64)
        if ber.size == 0:
            continue
        ber_plot = np.where(ber > 0, ber, BER_FLOOR)
        pen = _pen(pname)
        plot.plot(sigmas_plot, ber_plot, pen=pg.mkPen(pen, width=2),
                  symbol="o", symbolBrush=pen, name=plabel)

    if extras.get("role") == "comms":
        sigma = extras.get("sigma")
        if sigma is not None and sigma > 0:
            plot.addItem(pg.InfiniteLine(
                pos=float(np.log10(sigma)), angle=90, pen=MARKER_PEN,
            ))


def _add_per_shot_bars(layout, row, extras):
    ber_p = float(extras["ber_phys"])
    ber_m = float(extras["ber_model"])
    disagree = 1.0 - float(extras["agreement"])
    snr = float(extras["snr_db"])
    sigma = float(extras["sigma"])
    pname = extras.get("pname", "")
    plabel = extras.get("label", "")
    pen = _pen(pname)

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle(
        f"per-shot BER  ({plabel}, sigma={sigma:.1e}, SNR={snr:+.1f} dB)"
    )
    plot.setLabel("left", "fraction")
    names = ["BER phys", "BER model", "disagreement"]
    values = [ber_p, ber_m, disagree]
    brushes = [pg.mkBrush(*pen), pg.mkBrush(*pen, 160),
               pg.mkBrush(140, 200, 100)]
    plot.addItem(pg.BarGraphItem(
        x=np.arange(len(names)), height=values, width=0.6, brushes=brushes,
    ))
    plot.setYRange(0.0, max(0.5, 1.1 * max(values)))
    plot.getAxis("bottom").setTicks([list(enumerate(names))])
