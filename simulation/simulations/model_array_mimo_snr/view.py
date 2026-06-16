"""View extras for ``model_array_mimo_snr``.

Per shot:

  * Per-stream BER vs SINR waterfall (log y, dB x) -- 4 curves (each
    stream x [single, beamformed]).
  * Per-stream BER vs noise sigma (log y, log x) -- same 4 curves,
    horizontally offset by the array gain.
  * Per-shot BER bars for joint shots, 8 bars (each stream x [phys/model]
    x [single/beamformed])."""

import math

import numpy as np
import pyqtgraph as pg

STREAM_A_PEN = (200, 100, 200)
STREAM_B_PEN = (90, 200, 220)
SINGLE_STYLE = pg.QtCore.Qt.PenStyle.DashLine
BEAM_STYLE = pg.QtCore.Qt.PenStyle.SolidLine
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
    sinr_a_s = np.asarray(extras["sweep_sinr_a_single_db"], dtype=np.float64)
    sinr_a_b = np.asarray(extras["sweep_sinr_a_beam_db"], dtype=np.float64)
    sinr_b_s = np.asarray(extras["sweep_sinr_b_single_db"], dtype=np.float64)
    sinr_b_b = np.asarray(extras["sweep_sinr_b_beam_db"], dtype=np.float64)
    ber_a_s = np.asarray(extras["sweep_ber_a_single_phys"], dtype=np.float64)
    ber_a_b = np.asarray(extras["sweep_ber_a_beam_phys"], dtype=np.float64)
    ber_b_s = np.asarray(extras["sweep_ber_b_single_phys"], dtype=np.float64)
    ber_b_b = np.asarray(extras["sweep_ber_b_beam_phys"], dtype=np.float64)
    gain = extras.get("array_gain_db")

    plot = layout.addPlot(row=row, col=0)
    title = "Per-stream BER vs SINR waterfall  (log y, dB x)"
    if gain is not None:
        title += f"  --  MIMO array gain = {gain:+.1f} dB (mean over streams)"
    plot.setTitle(title)
    plot.setLabel("left", "BER")
    plot.setLabel("bottom", "SINR at decoder", units="dB")
    plot.setLogMode(x=False, y=True)
    plot.addLegend()
    plot.plot(sinr_a_s, np.where(ber_a_s > 0, ber_a_s, BER_FLOOR),
              pen=pg.mkPen(STREAM_A_PEN, width=2, style=SINGLE_STYLE),
              symbol="o", symbolBrush=STREAM_A_PEN, name="A single")
    plot.plot(sinr_a_b, np.where(ber_a_b > 0, ber_a_b, BER_FLOOR),
              pen=pg.mkPen(STREAM_A_PEN, width=2, style=BEAM_STYLE),
              symbol="s", symbolBrush=STREAM_A_PEN, name="A beam")
    plot.plot(sinr_b_s, np.where(ber_b_s > 0, ber_b_s, BER_FLOOR),
              pen=pg.mkPen(STREAM_B_PEN, width=2, style=SINGLE_STYLE),
              symbol="o", symbolBrush=STREAM_B_PEN, name="B single")
    plot.plot(sinr_b_b, np.where(ber_b_b > 0, ber_b_b, BER_FLOOR),
              pen=pg.mkPen(STREAM_B_PEN, width=2, style=BEAM_STYLE),
              symbol="s", symbolBrush=STREAM_B_PEN, name="B beam")


def _add_ber_vs_sigma(layout, row, extras):
    sigmas = np.asarray(extras["sweep_sigmas"], dtype=np.float64)
    ber_a_s = np.asarray(extras["sweep_ber_a_single_phys"], dtype=np.float64)
    ber_a_b = np.asarray(extras["sweep_ber_a_beam_phys"], dtype=np.float64)
    ber_b_s = np.asarray(extras["sweep_ber_b_single_phys"], dtype=np.float64)
    ber_b_b = np.asarray(extras["sweep_ber_b_beam_phys"], dtype=np.float64)

    sigmas_plot = np.where(sigmas > 0, sigmas, 1e-12)

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle("Per-stream BER vs ambient noise sigma  (log y, log x)")
    plot.setLabel("left", "BER")
    plot.setLabel("bottom", "noise source RMS (volume rate)")
    plot.setLogMode(x=True, y=True)
    plot.addLegend()
    plot.plot(sigmas_plot, np.where(ber_a_s > 0, ber_a_s, BER_FLOOR),
              pen=pg.mkPen(STREAM_A_PEN, width=2, style=SINGLE_STYLE),
              symbol="o", symbolBrush=STREAM_A_PEN, name="A single")
    plot.plot(sigmas_plot, np.where(ber_a_b > 0, ber_a_b, BER_FLOOR),
              pen=pg.mkPen(STREAM_A_PEN, width=2, style=BEAM_STYLE),
              symbol="s", symbolBrush=STREAM_A_PEN, name="A beam")
    plot.plot(sigmas_plot, np.where(ber_b_s > 0, ber_b_s, BER_FLOOR),
              pen=pg.mkPen(STREAM_B_PEN, width=2, style=SINGLE_STYLE),
              symbol="o", symbolBrush=STREAM_B_PEN, name="B single")
    plot.plot(sigmas_plot, np.where(ber_b_b > 0, ber_b_b, BER_FLOOR),
              pen=pg.mkPen(STREAM_B_PEN, width=2, style=BEAM_STYLE),
              symbol="s", symbolBrush=STREAM_B_PEN, name="B beam")

    if extras.get("role") == "comms":
        sigma = extras.get("sigma")
        if sigma is not None and sigma > 0:
            plot.addItem(pg.InfiniteLine(
                pos=float(np.log10(sigma)), angle=90, pen=MARKER_PEN,
            ))


def _add_per_shot_bars(layout, row, extras):
    sigma = float(extras["sigma"])
    sinr_a_b = float(extras["sinr_a_beam_db"])
    sinr_b_b = float(extras["sinr_b_beam_db"])
    sinr_a_s = float(extras["sinr_a_single_db"])
    sinr_b_s = float(extras["sinr_b_single_db"])

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle(
        f"per-shot BER  (sigma = {sigma:.1e}; "
        f"A single SINR = {sinr_a_s:+.1f} dB, A beam = {sinr_a_b:+.1f} dB | "
        f"B single = {sinr_b_s:+.1f} dB, B beam = {sinr_b_b:+.1f} dB)"
    )
    plot.setLabel("left", "fraction")
    names = ["A sng phys", "A sng model", "A beam phys", "A beam model",
             "B sng phys", "B sng model", "B beam phys", "B beam model"]
    values = [
        float(extras["ber_a_single_phys"]), float(extras["ber_a_single_model"]),
        float(extras["ber_a_beam_phys"]),   float(extras["ber_a_beam_model"]),
        float(extras["ber_b_single_phys"]), float(extras["ber_b_single_model"]),
        float(extras["ber_b_beam_phys"]),   float(extras["ber_b_beam_model"]),
    ]
    brushes = [
        pg.mkBrush(*STREAM_A_PEN, 160), pg.mkBrush(*STREAM_A_PEN, 80),
        pg.mkBrush(*STREAM_A_PEN),       pg.mkBrush(*STREAM_A_PEN, 200),
        pg.mkBrush(*STREAM_B_PEN, 160), pg.mkBrush(*STREAM_B_PEN, 80),
        pg.mkBrush(*STREAM_B_PEN),       pg.mkBrush(*STREAM_B_PEN, 200),
    ]
    plot.addItem(pg.BarGraphItem(
        x=np.arange(len(names)), height=values, width=0.6, brushes=brushes,
    ))
    plot.setYRange(0.0, max(0.5, 1.1 * max(values)))
    plot.getAxis("bottom").setTicks([list(enumerate(names))])
