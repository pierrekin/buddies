"""View extras for ``model_link_noise``.

Sweep plot: BER and SNR (signal RMS / noise RMS) vs ambient-noise sigma,
log-x. The current shot's sigma gets a marker. Per-comm-shot also gets
a bar trio: BER phys, BER model, disagreement, plus a measured "noise
RMS at RX / signal RMS at RX" annotation in the title."""

import numpy as np
import pyqtgraph as pg

PHYS_PEN = (80, 140, 220)
MODEL_PEN = (220, 150, 60)
AGREE_PEN = (140, 200, 100)
SNR_PEN = (200, 100, 200)
MARKER_PEN = pg.mkPen("y", style=pg.QtCore.Qt.PenStyle.DashLine)


def extra_views(viewer, layout, start_row):
    shot = viewer.shot
    extras = shot.extras
    role = extras.get("role")
    row = start_row

    if extras.get("sweep_sigmas"):
        _add_sweep(layout, row, extras, role)
        row += 1

    if role == "comms":
        _add_per_shot_bars(layout, row, extras)
        row += 1


def _add_sweep(layout, row, extras, role):
    sigmas = np.asarray(extras["sweep_sigmas"], dtype=np.float64)
    ber_phys = np.asarray(extras["sweep_ber_phys"], dtype=np.float64)
    ber_model = np.asarray(extras["sweep_ber_model"], dtype=np.float64)
    agreement = np.asarray(extras["sweep_agreement"], dtype=np.float64)

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle("BER and disagreement vs ambient noise sigma (log x)")
    plot.setLabel("left", "fraction")
    plot.setLabel("bottom", "noise source RMS (volume rate)")
    plot.setLogMode(x=True, y=False)
    plot.addLegend()

    # Replace any zero sigma with a small positive value for log axis.
    sigmas_log = np.where(sigmas > 0, sigmas, 1e-12)

    plot.plot(sigmas_log, ber_phys, pen=pg.mkPen(PHYS_PEN, width=2),
              symbol="o", symbolBrush=PHYS_PEN, name="BER (physical)")
    plot.plot(sigmas_log, ber_model, pen=pg.mkPen(MODEL_PEN, width=2),
              symbol="o", symbolBrush=MODEL_PEN, name="BER (model, clean LTI)")
    plot.plot(sigmas_log, 1.0 - agreement,
              pen=pg.mkPen(AGREE_PEN, width=2,
                           style=pg.QtCore.Qt.PenStyle.DashLine),
              symbol="s", symbolBrush=AGREE_PEN, name="disagreement (phys vs model)")
    plot.setYRange(-0.02, 1.02)

    if role == "comms":
        current_sigma = float(extras.get("sigma", 0.0))
        if current_sigma > 0:
            plot.addItem(pg.InfiniteLine(
                pos=float(np.log10(current_sigma)), angle=90, pen=MARKER_PEN,
            ))


def _add_per_shot_bars(layout, row, extras):
    ber_phys = float(extras["ber_phys"])
    ber_model = float(extras["ber_model"])
    disagreement = 1.0 - float(extras["agreement"])
    sigma = float(extras["sigma"])
    rx_rms = float(extras["rx_phys_rms"])
    noise_rms = float(extras["noise_rms_at_rx"])
    snr_db = 20.0 * np.log10(rx_rms / noise_rms) if noise_rms > 0 else float("inf")

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle(
        f"per-shot BER  (sigma = {sigma:.1e}, "
        f"RX signal RMS = {rx_rms:.2e} V, noise RMS = {noise_rms:.2e} V, "
        f"SNR ~ {snr_db:.1f} dB)"
    )
    plot.setLabel("left", "fraction")
    names = ["BER phys", "BER model", "disagreement"]
    values = [ber_phys, ber_model, disagreement]
    brushes = [pg.mkBrush(*c) for c in (PHYS_PEN, MODEL_PEN, AGREE_PEN)]
    plot.addItem(pg.BarGraphItem(
        x=np.arange(len(names)), height=values, width=0.6, brushes=brushes,
    ))
    plot.setYRange(0.0, max(0.5, 1.1 * max(values)))
    plot.getAxis("bottom").setTicks([list(enumerate(names))])
