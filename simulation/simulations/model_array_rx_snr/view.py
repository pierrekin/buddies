"""View extras for ``model_array_rx_snr``.

Per shot:

  * BER vs SNR waterfall (log y, dB x) with both pipelines (single,
    beamformed).
  * BER vs noise sigma (log y, log x) with the same two pipelines --
    array gain shows up as a horizontal shift between the curves.
  * Per-shot BER bars for comms shots."""

import math

import numpy as np
import pyqtgraph as pg

SINGLE_PEN = (80, 140, 220)
BEAM_PEN = (220, 150, 60)
MARKER_PEN = pg.mkPen("y", style=pg.QtCore.Qt.PenStyle.DashLine)
BER_FLOOR = 1e-3


def extra_views(viewer, layout, start_row):
    extras = viewer.shot.extras
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
    snr_s = np.asarray(extras["sweep_single_snr_db"], dtype=np.float64)
    ber_s = np.asarray(extras["sweep_single_ber_phys"], dtype=np.float64)
    snr_b = np.asarray(extras["sweep_beam_snr_db"], dtype=np.float64)
    ber_b = np.asarray(extras["sweep_beam_ber_phys"], dtype=np.float64)
    gain = extras.get("array_gain_db")
    gain_theory = extras.get("array_gain_theoretical_db")

    ber_s_plot = np.where(ber_s > 0, ber_s, BER_FLOOR)
    ber_b_plot = np.where(ber_b > 0, ber_b, BER_FLOOR)

    plot = layout.addPlot(row=row, col=0)
    title = "BER vs SNR waterfall  (log y, dB x)"
    if gain is not None and gain_theory is not None:
        title += (f"  --  RX array gain = {gain:+.1f} dB  "
                  f"(theory {gain_theory:+.1f} dB, uncorrelated noise)")
    plot.setTitle(title)
    plot.setLabel("left", "BER")
    plot.setLabel("bottom", "SNR at RX", units="dB")
    plot.setLogMode(x=False, y=True)
    plot.addLegend()
    plot.plot(snr_s, ber_s_plot, pen=pg.mkPen(SINGLE_PEN, width=2),
              symbol="o", symbolBrush=SINGLE_PEN, name="single element")
    plot.plot(snr_b, ber_b_plot, pen=pg.mkPen(BEAM_PEN, width=2),
              symbol="s", symbolBrush=BEAM_PEN, name="beamformed (8 elems)")

    if extras.get("role") == "comms":
        for key, pen in (("snr_single_db", SINGLE_PEN),
                          ("snr_beam_db", BEAM_PEN)):
            snr = extras.get(key)
            if snr is not None and math.isfinite(snr):
                plot.addItem(pg.InfiniteLine(
                    pos=float(snr), angle=90, pen=pg.mkPen(pen, width=1),
                ))


def _add_ber_vs_sigma(layout, row, extras):
    sigmas = np.asarray(extras["sweep_sigmas"], dtype=np.float64)
    ber_s = np.asarray(extras["sweep_single_ber_phys"], dtype=np.float64)
    ber_b = np.asarray(extras["sweep_beam_ber_phys"], dtype=np.float64)
    gain = extras.get("array_gain_db")

    sigmas_plot = np.where(sigmas > 0, sigmas, 1e-12)
    ber_s_plot = np.where(ber_s > 0, ber_s, BER_FLOOR)
    ber_b_plot = np.where(ber_b > 0, ber_b, BER_FLOOR)

    plot = layout.addPlot(row=row, col=0)
    title = "BER vs ambient noise sigma  (log y, log x)"
    if gain is not None:
        title += (f"  --  beamformed curve shifted right by {gain:.1f} dB "
                  f"= {10**(gain/20):.1f}x sigma")
    plot.setTitle(title)
    plot.setLabel("left", "BER")
    plot.setLabel("bottom", "noise source RMS (volume rate)")
    plot.setLogMode(x=True, y=True)
    plot.addLegend()
    plot.plot(sigmas_plot, ber_s_plot, pen=pg.mkPen(SINGLE_PEN, width=2),
              symbol="o", symbolBrush=SINGLE_PEN, name="single element")
    plot.plot(sigmas_plot, ber_b_plot, pen=pg.mkPen(BEAM_PEN, width=2),
              symbol="s", symbolBrush=BEAM_PEN, name="beamformed (8 elems)")

    if extras.get("role") == "comms":
        sigma = extras.get("sigma")
        if sigma is not None and sigma > 0:
            plot.addItem(pg.InfiniteLine(
                pos=float(np.log10(sigma)), angle=90, pen=MARKER_PEN,
            ))


def _add_per_shot_bars(layout, row, extras):
    sigma = float(extras["sigma"])
    snr_single = float(extras["snr_single_db"])
    snr_beam = float(extras["snr_beam_db"])

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle(
        f"per-shot BER  (sigma = {sigma:.1e}; "
        f"single SNR = {snr_single:+.1f} dB, "
        f"beam SNR = {snr_beam:+.1f} dB)"
    )
    plot.setLabel("left", "fraction")
    names = ["single phys", "single model", "single disagree",
             "beam phys", "beam model", "beam disagree"]
    values = [
        float(extras["ber_phys_single"]),
        float(extras["ber_model_single"]),
        1.0 - float(extras["agreement_single"]),
        float(extras["ber_phys_beam"]),
        float(extras["ber_model_beam"]),
        1.0 - float(extras["agreement_beam"]),
    ]
    brushes = [
        pg.mkBrush(*SINGLE_PEN), pg.mkBrush(*SINGLE_PEN, 160),
        pg.mkBrush(140, 200, 100),
        pg.mkBrush(*BEAM_PEN), pg.mkBrush(*BEAM_PEN, 160),
        pg.mkBrush(140, 200, 100),
    ]
    plot.addItem(pg.BarGraphItem(
        x=np.arange(len(names)), height=values, width=0.6, brushes=brushes,
    ))
    plot.setYRange(0.0, max(0.5, 1.1 * max(values)))
    plot.getAxis("bottom").setTicks([list(enumerate(names))])
