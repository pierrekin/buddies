"""View extras for ``model_lora_sync``.

Per LoRa SF, three BER curves on every plot:

  * **red, dashed**  -- no sync (matched filter ignores the preamble)
  * **amber, solid** -- preamble sync (real receiver)
  * **green, solid** -- oracle sync (cheat, upper bound)

The amber sits between red and green at moderate SNR -- when the
preamble is decodable, sync works and the curve hugs green; when the
preamble itself fails under noise, sync collapses and the curve drifts
back toward red. That's the educational point of having all three on
one axis."""

import math

import numpy as np
import pyqtgraph as pg

OOK_PEN = (80, 140, 220)
NOSYNC_PEN = (220, 80, 80)
PREAMBLE_PEN = (230, 180, 80)
ORACLE_PEN = (120, 200, 100)
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


def _plot_curve(plot, x, y, *, pen_color, dashed, symbol, label):
    y_plot = np.where(y > 0, y, BER_FLOOR)
    style = pg.QtCore.Qt.PenStyle.DashLine if dashed else pg.QtCore.Qt.PenStyle.SolidLine
    plot.plot(x, y_plot,
              pen=pg.mkPen(pen_color, width=2, style=style),
              symbol=symbol, symbolBrush=pen_color, name=label)


def _add_waterfall(layout, row, extras):
    sfs = list(extras.get("lora_sfs", [7, 9]))

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle("BER vs SNR  (log y, dB x)  --  "
                  "red = no sync, amber = preamble sync, green = oracle sync")
    plot.setLabel("left", "BER (payload only)")
    plot.setLabel("bottom", "SNR at RX", units="dB")
    plot.setLogMode(x=False, y=True)
    plot.addLegend()

    # OOK reference (one curve).
    snr = np.asarray(extras.get("sweep_ook_snr_db", []), dtype=np.float64)
    ber = np.asarray(extras.get("sweep_ook_ber_phys_nosync", []), dtype=np.float64)
    if snr.size:
        _plot_curve(plot, snr, ber, pen_color=OOK_PEN, dashed=False,
                    symbol="o", label="OOK")

    for sf in sfs:
        mod = f"lora_sf{sf}"
        snr = np.asarray(extras.get(f"sweep_{mod}_snr_db", []), dtype=np.float64)
        if snr.size == 0:
            continue
        _plot_curve(plot, snr,
                    np.asarray(extras[f"sweep_{mod}_ber_phys_nosync"], dtype=np.float64),
                    pen_color=NOSYNC_PEN, dashed=True, symbol="x",
                    label=f"SF={sf} no sync")
        _plot_curve(plot, snr,
                    np.asarray(extras[f"sweep_{mod}_ber_phys_preamble"], dtype=np.float64),
                    pen_color=PREAMBLE_PEN, dashed=False, symbol="t",
                    label=f"SF={sf} preamble sync")
        _plot_curve(plot, snr,
                    np.asarray(extras[f"sweep_{mod}_ber_phys_oracle"], dtype=np.float64),
                    pen_color=ORACLE_PEN, dashed=False, symbol="o",
                    label=f"SF={sf} oracle sync")

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
    plot.setTitle("BER vs ambient noise sigma  (log y, log x)  --  "
                  "red = no sync, amber = preamble sync, green = oracle sync")
    plot.setLabel("left", "BER (payload only)")
    plot.setLabel("bottom", "noise source RMS (volume rate)")
    plot.setLogMode(x=True, y=True)
    plot.addLegend()

    ber = np.asarray(extras.get("sweep_ook_ber_phys_nosync", []), dtype=np.float64)
    if ber.size:
        _plot_curve(plot, sigmas_plot, ber, pen_color=OOK_PEN, dashed=False,
                    symbol="o", label="OOK")

    for sf in sfs:
        mod = f"lora_sf{sf}"
        ber_n = np.asarray(extras.get(f"sweep_{mod}_ber_phys_nosync", []), dtype=np.float64)
        ber_p = np.asarray(extras.get(f"sweep_{mod}_ber_phys_preamble", []), dtype=np.float64)
        ber_o = np.asarray(extras.get(f"sweep_{mod}_ber_phys_oracle", []), dtype=np.float64)
        if ber_n.size == 0:
            continue
        _plot_curve(plot, sigmas_plot, ber_n,
                    pen_color=NOSYNC_PEN, dashed=True, symbol="x",
                    label=f"SF={sf} no sync")
        _plot_curve(plot, sigmas_plot, ber_p,
                    pen_color=PREAMBLE_PEN, dashed=False, symbol="t",
                    label=f"SF={sf} preamble sync")
        _plot_curve(plot, sigmas_plot, ber_o,
                    pen_color=ORACLE_PEN, dashed=False, symbol="o",
                    label=f"SF={sf} oracle sync")

    if extras.get("role") == "comms":
        sigma = extras.get("sigma")
        if sigma is not None and sigma > 0:
            plot.addItem(pg.InfiniteLine(
                pos=float(np.log10(sigma)), angle=90, pen=MARKER_PEN,
            ))


def _add_per_shot_bars(layout, row, extras):
    snr = float(extras["snr_db"])
    modulation = extras.get("modulation", "")
    kind = extras.get("kind", "")
    n_bits = int(extras.get("n_bits", 0))
    ber_n = float(extras["ber_phys_nosync"])
    ber_p = float(extras["ber_phys_preamble"])
    ber_o = float(extras["ber_phys_oracle"])
    pre_off = float(extras.get("preamble_offset_phys_us", 0.0))
    ora_off = float(extras.get("oracle_offset_phys_us", 0.0))
    pre_matches = int(extras.get("preamble_matches_phys", -1))

    plot = layout.addPlot(row=row, col=0)
    title = f"per-shot payload BER  ({modulation}, SNR = {snr:+.1f} dB"
    if kind == "lora":
        title += (f"; preamble matches {pre_matches}, "
                  f"preamble offset {pre_off:+.1f} us, "
                  f"oracle offset {ora_off:+.1f} us")
    title += ")"
    plot.setTitle(title)
    plot.setLabel("left", "BER (payload)")
    names = ["no sync", "preamble", "oracle"]
    values = [ber_n, ber_p, ber_o]
    brushes = [pg.mkBrush(*NOSYNC_PEN), pg.mkBrush(*PREAMBLE_PEN),
               pg.mkBrush(*ORACLE_PEN)]
    plot.addItem(pg.BarGraphItem(
        x=np.arange(len(names)), height=values, width=0.6, brushes=brushes,
    ))
    plot.setYRange(0.0, max(0.5, 1.1 * max(values)))
    plot.getAxis("bottom").setTicks([list(enumerate(names))])
