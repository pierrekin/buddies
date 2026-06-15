"""View extras for ``model_lora_noise``.

Per shot:

  * BER vs SNR waterfall with OOK and LoRa curves.
  * BER vs ambient noise sigma with both modulations.
  * Per-shot BER bars for comms shots."""

import math

import numpy as np
import pyqtgraph as pg

OOK_PEN = (80, 140, 220)
LORA_PEN = (220, 150, 60)
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
    snr_o = np.asarray(extras["sweep_ook_snr_db"], dtype=np.float64)
    ber_o = np.asarray(extras["sweep_ook_ber_phys"], dtype=np.float64)
    snr_l = np.asarray(extras["sweep_lora_snr_db"], dtype=np.float64)
    ber_l = np.asarray(extras["sweep_lora_ber_phys"], dtype=np.float64)
    pg_db = extras.get("css_processing_gain_db")

    ber_o_plot = np.where(ber_o > 0, ber_o, BER_FLOOR)
    ber_l_plot = np.where(ber_l > 0, ber_l, BER_FLOOR)

    plot = layout.addPlot(row=row, col=0)
    title = "BER vs SNR waterfall  (log y, dB x)"
    if pg_db is not None:
        title += f"  --  CSS processing gain (theoretical T*B): {pg_db:+.2f} dB"
    plot.setTitle(title)
    plot.setLabel("left", "BER")
    plot.setLabel("bottom", "SNR at RX", units="dB")
    plot.setLogMode(x=False, y=True)
    plot.addLegend()
    plot.plot(snr_o, ber_o_plot, pen=pg.mkPen(OOK_PEN, width=2),
              symbol="o", symbolBrush=OOK_PEN, name="OOK")
    plot.plot(snr_l, ber_l_plot, pen=pg.mkPen(LORA_PEN, width=2),
              symbol="s", symbolBrush=LORA_PEN, name="LoRa (binary CSS)")

    if extras.get("role") == "comms":
        current_snr = extras.get("snr_db")
        if current_snr is not None and math.isfinite(current_snr):
            plot.addItem(pg.InfiniteLine(
                pos=float(current_snr), angle=90, pen=MARKER_PEN,
            ))


def _add_ber_vs_sigma(layout, row, extras):
    sigmas = np.asarray(extras["sweep_sigmas"], dtype=np.float64)
    ber_o = np.asarray(extras["sweep_ook_ber_phys"], dtype=np.float64)
    ber_l = np.asarray(extras["sweep_lora_ber_phys"], dtype=np.float64)

    sigmas_plot = np.where(sigmas > 0, sigmas, 1e-12)
    ber_o_plot = np.where(ber_o > 0, ber_o, BER_FLOOR)
    ber_l_plot = np.where(ber_l > 0, ber_l, BER_FLOOR)

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle("BER vs ambient noise sigma  (log y, log x)")
    plot.setLabel("left", "BER")
    plot.setLabel("bottom", "noise source RMS (volume rate)")
    plot.setLogMode(x=True, y=True)
    plot.addLegend()
    plot.plot(sigmas_plot, ber_o_plot, pen=pg.mkPen(OOK_PEN, width=2),
              symbol="o", symbolBrush=OOK_PEN, name="OOK")
    plot.plot(sigmas_plot, ber_l_plot, pen=pg.mkPen(LORA_PEN, width=2),
              symbol="s", symbolBrush=LORA_PEN, name="LoRa (binary CSS)")

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
    modulation = extras.get("modulation", "")

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle(
        f"per-shot BER  (modulation = {modulation}, SNR = {snr:+.1f} dB)"
    )
    plot.setLabel("left", "fraction")
    names = ["BER phys", "BER model", "disagreement"]
    values = [ber_p, ber_m, disagree]
    pen = OOK_PEN if modulation == "ook" else LORA_PEN
    brushes = [pg.mkBrush(*pen), pg.mkBrush(*pen, 160),
               pg.mkBrush(140, 200, 100)]
    plot.addItem(pg.BarGraphItem(
        x=np.arange(len(names)), height=values, width=0.6, brushes=brushes,
    ))
    plot.setYRange(0.0, max(0.5, 1.1 * max(values)))
    plot.getAxis("bottom").setTicks([list(enumerate(names))])
