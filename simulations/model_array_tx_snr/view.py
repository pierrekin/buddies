"""View extras for ``model_array_tx_snr``.

Per shot:

  * Waterfall plot (log-y BER vs SNR dB) with single and broadside
    curves and a horizontal line at the array-gain target BER.
  * Per-shot BER bars (phys, model, disagreement) with the shot's SNR
    in the title.

The waterfall plot is the standard comms performance figure; the array
gain (dB at fixed BER) is annotated in the plot title."""

import math

import numpy as np
import pyqtgraph as pg

SINGLE_PEN = (80, 140, 220)
BROADSIDE_PEN = (220, 150, 60)
MARKER_PEN = pg.mkPen("y", style=pg.QtCore.Qt.PenStyle.DashLine)
TARGET_PEN = pg.mkPen((200, 200, 200), style=pg.QtCore.Qt.PenStyle.DotLine)
BER_FLOOR = 1e-3  # clip zero-BER points to this so log-y plotting works


def extra_views(viewer, layout, start_row):
    shot = viewer.shot
    extras = shot.extras
    role = extras.get("role")
    row = start_row

    _add_waterfall(layout, row, extras)
    row += 1
    _add_ber_vs_sigma(layout, row, extras)
    row += 1

    if role == "comms":
        _add_per_shot_bars(layout, row, extras)
        row += 1


def _add_waterfall(layout, row, extras):
    """The canonical BER vs SNR waterfall. Both configurations use the
    same OOK decoder so their curves should overlap; if they don't,
    something nontrivial is happening (mismatched bandwidth, decoder
    nonlinearity, ...)."""
    snr_s = np.asarray(extras["sweep_single_snr_db"], dtype=np.float64)
    ber_s = np.asarray(extras["sweep_single_ber_phys"], dtype=np.float64)
    snr_b = np.asarray(extras["sweep_broadside_snr_db"], dtype=np.float64)
    ber_b = np.asarray(extras["sweep_broadside_ber_phys"], dtype=np.float64)
    gain = extras.get("array_gain_db")
    gain_theory = extras.get("array_gain_theoretical_db")

    ber_s_plot = np.where(ber_s > 0, ber_s, BER_FLOOR)
    ber_b_plot = np.where(ber_b > 0, ber_b, BER_FLOOR)

    plot = layout.addPlot(row=row, col=0)
    title = "BER vs SNR waterfall  (log y, dB x)"
    if gain is not None and gain_theory is not None:
        title += (f"  --  array gain = {gain:+.1f} dB  "
                  f"(theory {gain_theory:+.1f} dB)")
    plot.setTitle(title)
    plot.setLabel("left", "BER")
    plot.setLabel("bottom", "SNR at RX", units="dB")
    plot.setLogMode(x=False, y=True)
    plot.addLegend()
    plot.plot(snr_s, ber_s_plot,
              pen=pg.mkPen(SINGLE_PEN, width=2),
              symbol="o", symbolBrush=SINGLE_PEN, name="single source")
    plot.plot(snr_b, ber_b_plot,
              pen=pg.mkPen(BROADSIDE_PEN, width=2),
              symbol="s", symbolBrush=BROADSIDE_PEN, name="broadside (8 elems)")

    if extras.get("role") == "comms":
        current_snr = extras.get("snr_db")
        if current_snr is not None and math.isfinite(current_snr):
            plot.addItem(pg.InfiniteLine(
                pos=float(current_snr), angle=90, pen=MARKER_PEN,
            ))


def _add_ber_vs_sigma(layout, row, extras):
    """Same data, plotted vs noise drive sigma. The two curves are
    horizontally offset by the array gain (broadside curve is to the
    right of single by ~20*log10(N) dB worth of sigma)."""
    sigmas = np.asarray(extras["sweep_sigmas"], dtype=np.float64)
    ber_s = np.asarray(extras["sweep_single_ber_phys"], dtype=np.float64)
    ber_b = np.asarray(extras["sweep_broadside_ber_phys"], dtype=np.float64)
    gain = extras.get("array_gain_db")

    # Replace zero sigma with a small positive value for log axis.
    sigmas_plot = np.where(sigmas > 0, sigmas, 1e-12)
    ber_s_plot = np.where(ber_s > 0, ber_s, BER_FLOOR)
    ber_b_plot = np.where(ber_b > 0, ber_b, BER_FLOOR)

    plot = layout.addPlot(row=row, col=0)
    title = "BER vs ambient noise sigma  (log y, log x)"
    if gain is not None:
        title += (f"  --  broadside curve shifted right by {gain:.1f} dB "
                  f"= {10**(gain/20):.1f}x sigma")
    plot.setTitle(title)
    plot.setLabel("left", "BER")
    plot.setLabel("bottom", "noise source RMS (volume rate)")
    plot.setLogMode(x=True, y=True)
    plot.addLegend()
    plot.plot(sigmas_plot, ber_s_plot,
              pen=pg.mkPen(SINGLE_PEN, width=2),
              symbol="o", symbolBrush=SINGLE_PEN, name="single source")
    plot.plot(sigmas_plot, ber_b_plot,
              pen=pg.mkPen(BROADSIDE_PEN, width=2),
              symbol="s", symbolBrush=BROADSIDE_PEN, name="broadside (8 elems)")

    if extras.get("role") == "comms":
        sigma = extras.get("sigma")
        if sigma is not None and sigma > 0:
            plot.addItem(pg.InfiniteLine(
                pos=float(np.log10(sigma)), angle=90, pen=MARKER_PEN,
            ))


def _add_per_shot_bars(layout, row, extras):
    ber_phys = float(extras["ber_phys"])
    ber_model = float(extras["ber_model"])
    disagree = 1.0 - float(extras["agreement"])
    snr = float(extras["snr_db"])
    config = extras.get("config", "")

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle(
        f"per-shot BER  (config = {config}, SNR = {snr:+.1f} dB)"
    )
    plot.setLabel("left", "fraction")
    names = ["BER phys", "BER model", "disagreement"]
    values = [ber_phys, ber_model, disagree]
    brushes = [pg.mkBrush(*SINGLE_PEN), pg.mkBrush(*BROADSIDE_PEN),
               pg.mkBrush(140, 200, 100)]
    plot.addItem(pg.BarGraphItem(
        x=np.arange(len(names)), height=values, width=0.6, brushes=brushes,
    ))
    plot.setYRange(0.0, max(0.5, 1.1 * max(values)))
    plot.getAxis("bottom").setTicks([list(enumerate(names))])
