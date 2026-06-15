"""View extras for ``model_lora_sf``.

Per shot:

  * BER vs SNR waterfall with one curve per modulation (OOK, LoRa SF=k
    for each configured SF).
  * BER vs ambient noise sigma -- the curves shift left as SF grows.
  * Per-shot BER bar trio."""

import math

import numpy as np
import pyqtgraph as pg

OOK_PEN = (80, 140, 220)
LORA_SF7_PEN = (220, 150, 60)
LORA_SF9_PEN = (200, 100, 200)
MARKER_PEN = pg.mkPen("y", style=pg.QtCore.Qt.PenStyle.DashLine)
BER_FLOOR = 1e-3


def _pen_for(modulation):
    if modulation == "ook":
        return OOK_PEN
    if modulation == "lora_sf7":
        return LORA_SF7_PEN
    if modulation == "lora_sf9":
        return LORA_SF9_PEN
    return (180, 180, 180)


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
    sfs = list(extras.get("lora_sfs", [7, 9]))
    pgs = list(extras.get("lora_processing_gains_db", []))

    plot = layout.addPlot(row=row, col=0)
    title_bits = ["BER vs SNR waterfall  (log y, dB x)"]
    if pgs:
        title_bits.append("LoRa processing gains: " +
                          ", ".join(f"SF{sf}: {pg:+.1f} dB"
                                    for sf, pg in zip(sfs, pgs)))
    plot.setTitle("  --  ".join(title_bits))
    plot.setLabel("left", "BER")
    plot.setLabel("bottom", "SNR at RX", units="dB")
    plot.setLogMode(x=False, y=True)
    plot.addLegend()

    for mod, label in [("ook", "OOK"),
                        *[(f"lora_sf{sf}", f"LoRa SF={sf}") for sf in sfs]]:
        snr = np.asarray(extras.get(f"sweep_{mod}_snr_db", []), dtype=np.float64)
        ber = np.asarray(extras.get(f"sweep_{mod}_ber_phys", []), dtype=np.float64)
        if snr.size == 0:
            continue
        ber_plot = np.where(ber > 0, ber, BER_FLOOR)
        pen = _pen_for(mod)
        plot.plot(snr, ber_plot,
                  pen=pg.mkPen(pen, width=2),
                  symbol="o", symbolBrush=pen, name=label)

    if extras.get("role") == "comms":
        current_snr = extras.get("snr_db")
        if current_snr is not None and math.isfinite(current_snr):
            plot.addItem(pg.InfiniteLine(
                pos=float(current_snr), angle=90, pen=MARKER_PEN,
            ))


def _add_ber_vs_sigma(layout, row, extras):
    sfs = list(extras.get("lora_sfs", [7, 9]))
    sigmas = np.asarray(extras["sweep_sigmas"], dtype=np.float64)
    sigmas_plot = np.where(sigmas > 0, sigmas, 1e-12)

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle("BER vs ambient noise sigma  (log y, log x)")
    plot.setLabel("left", "BER")
    plot.setLabel("bottom", "noise source RMS (volume rate)")
    plot.setLogMode(x=True, y=True)
    plot.addLegend()

    for mod, label in [("ook", "OOK"),
                        *[(f"lora_sf{sf}", f"LoRa SF={sf}") for sf in sfs]]:
        ber = np.asarray(extras.get(f"sweep_{mod}_ber_phys", []), dtype=np.float64)
        if ber.size == 0:
            continue
        ber_plot = np.where(ber > 0, ber, BER_FLOOR)
        pen = _pen_for(mod)
        plot.plot(sigmas_plot, ber_plot,
                  pen=pg.mkPen(pen, width=2),
                  symbol="o", symbolBrush=pen, name=label)

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
    n_bits = int(extras.get("n_bits", 0))
    err_p = int(extras.get("errors_phys", 0))
    err_m = int(extras.get("errors_model", 0))

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle(
        f"per-shot BER  ({modulation}, SNR = {snr:+.1f} dB, "
        f"phys errors {err_p}/{n_bits}, model errors {err_m}/{n_bits})"
    )
    plot.setLabel("left", "fraction")
    names = ["BER phys", "BER model", "disagreement"]
    values = [ber_p, ber_m, disagree]
    pen = _pen_for(modulation)
    brushes = [pg.mkBrush(*pen), pg.mkBrush(*pen, 160),
               pg.mkBrush(140, 200, 100)]
    plot.addItem(pg.BarGraphItem(
        x=np.arange(len(names)), height=values, width=0.6, brushes=brushes,
    ))
    plot.setYRange(0.0, max(0.5, 1.1 * max(values)))
    plot.getAxis("bottom").setTicks([list(enumerate(names))])
