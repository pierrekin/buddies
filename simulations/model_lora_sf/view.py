"""View extras for ``model_lora_sf``.

Per shot:

  * BER vs SNR waterfall with two curves per LoRa SF: ``no sync`` (red,
    dashed -- failing case, matched filter without symbol-window
    alignment) and ``oracle sync`` (green, solid -- cheats by knowing
    the sent bits, isolates "alignment is the only issue"). OOK has
    one curve (its RMS slicer doesn't need sync).
  * BER vs ambient noise sigma with the same red/green semantic.
  * Per-shot BER bar group.

The red-vs-green spread per SF is the educational point: matched-filter
modulations like LoRa need a sync mechanism; without it they fail even
in a quiet channel, but if you *had* sync the processing gain is there."""

import math

import numpy as np
import pyqtgraph as pg

OOK_PEN = (80, 140, 220)
NOSYNC_PEN = (220, 80, 80)    # red -- the failing case
ORACLE_PEN = (120, 200, 100)  # green -- what's achievable with sync
MARKER_PEN = pg.mkPen("y", style=pg.QtCore.Qt.PenStyle.DashLine)
BER_FLOOR = 1e-3


def _pen_for(modulation):
    if modulation == "ook":
        return OOK_PEN
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
    title_bits = ["BER vs SNR  (log y, dB x)  --  red = no sync, green = oracle sync"]
    if pgs:
        title_bits.append("LoRa processing gains: " +
                          ", ".join(f"SF{sf}: {pg:+.1f} dB"
                                    for sf, pg in zip(sfs, pgs)))
    plot.setTitle("   ".join(title_bits))
    plot.setLabel("left", "BER")
    plot.setLabel("bottom", "SNR at RX", units="dB")
    plot.setLogMode(x=False, y=True)
    plot.addLegend()

    # OOK gets one curve (no sync needed).
    _plot_curve(plot, extras, "ook", "snr_db", "ber_phys_nosync",
                pen_color=OOK_PEN, dashed=False, label="OOK",
                x_log=False)

    # Each LoRa SF gets the red/green pair.
    for sf in sfs:
        mod = f"lora_sf{sf}"
        _plot_curve(plot, extras, mod, "snr_db", "ber_phys_nosync",
                    pen_color=NOSYNC_PEN, dashed=True,
                    label=f"LoRa SF={sf} (no sync)", x_log=False)
        _plot_curve(plot, extras, mod, "snr_db", "ber_phys_oracle",
                    pen_color=ORACLE_PEN, dashed=False,
                    label=f"LoRa SF={sf} (oracle sync)", x_log=False)

    if extras.get("role") == "comms":
        current_snr = extras.get("snr_db")
        if current_snr is not None and math.isfinite(current_snr):
            plot.addItem(pg.InfiniteLine(
                pos=float(current_snr), angle=90, pen=MARKER_PEN,
            ))


def _add_ber_vs_sigma(layout, row, extras):
    sfs = list(extras.get("lora_sfs", [7, 9]))
    sigmas = np.asarray(extras["sweep_sigmas"], dtype=np.float64)

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle("BER vs ambient noise sigma  (log y, log x)  --  red = no sync, green = oracle sync")
    plot.setLabel("left", "BER")
    plot.setLabel("bottom", "noise source RMS (volume rate)")
    plot.setLogMode(x=True, y=True)
    plot.addLegend()

    _plot_curve(plot, extras, "ook", None, "ber_phys_nosync",
                pen_color=OOK_PEN, dashed=False, label="OOK",
                x_sigmas=sigmas, x_log=True)
    for sf in sfs:
        mod = f"lora_sf{sf}"
        _plot_curve(plot, extras, mod, None, "ber_phys_nosync",
                    pen_color=NOSYNC_PEN, dashed=True,
                    label=f"LoRa SF={sf} (no sync)",
                    x_sigmas=sigmas, x_log=True)
        _plot_curve(plot, extras, mod, None, "ber_phys_oracle",
                    pen_color=ORACLE_PEN, dashed=False,
                    label=f"LoRa SF={sf} (oracle sync)",
                    x_sigmas=sigmas, x_log=True)

    if extras.get("role") == "comms":
        sigma = extras.get("sigma")
        if sigma is not None and sigma > 0:
            plot.addItem(pg.InfiniteLine(
                pos=float(np.log10(sigma)), angle=90, pen=MARKER_PEN,
            ))


def _plot_curve(plot, extras, mod, x_key, y_key, *, pen_color, dashed,
                label, x_sigmas=None, x_log=False):
    """Pull a series from the sweep dict and plot it."""
    y = np.asarray(extras.get(f"sweep_{mod}_{y_key}", []), dtype=np.float64)
    if y.size == 0:
        return
    if x_key is not None:
        x = np.asarray(extras.get(f"sweep_{mod}_{x_key}", []), dtype=np.float64)
    else:
        # x = sigmas (replace 0 with epsilon for log plot)
        x = np.where(x_sigmas > 0, x_sigmas, 1e-12)
    y_plot = np.where(y > 0, y, BER_FLOOR)
    style = pg.QtCore.Qt.PenStyle.DashLine if dashed else pg.QtCore.Qt.PenStyle.SolidLine
    plot.plot(x, y_plot,
              pen=pg.mkPen(pen_color, width=2, style=style),
              symbol="o" if not dashed else "x",
              symbolBrush=pen_color, name=label)


def _add_per_shot_bars(layout, row, extras):
    snr = float(extras["snr_db"])
    modulation = extras.get("modulation", "")
    kind = extras.get("kind", "")
    n_bits = int(extras.get("n_bits", 0))
    ber_p_n = float(extras["ber_phys_nosync"])
    ber_m_n = float(extras["ber_model_nosync"])
    ber_p_o = float(extras["ber_phys_oracle"])
    ber_m_o = float(extras["ber_model_oracle"])
    err_p_n = int(extras["errors_phys_nosync"])
    err_p_o = int(extras["errors_phys_oracle"])
    off_p = float(extras.get("oracle_offset_phys_us", 0.0))

    plot = layout.addPlot(row=row, col=0)
    title = (f"per-shot BER  ({modulation}, SNR = {snr:+.1f} dB; "
             f"phys nosync {err_p_n}/{n_bits}, oracle {err_p_o}/{n_bits}")
    if kind == "lora":
        title += f"; oracle offset = {off_p:+.1f} us"
    title += ")"
    plot.setTitle(title)
    plot.setLabel("left", "fraction")
    names = [
        "phys nosync", "model nosync",
        "phys oracle", "model oracle",
    ]
    values = [ber_p_n, ber_m_n, ber_p_o, ber_m_o]
    brushes = [
        pg.mkBrush(*NOSYNC_PEN), pg.mkBrush(*NOSYNC_PEN, 160),
        pg.mkBrush(*ORACLE_PEN), pg.mkBrush(*ORACLE_PEN, 160),
    ]
    plot.addItem(pg.BarGraphItem(
        x=np.arange(len(names)), height=values, width=0.6, brushes=brushes,
    ))
    plot.setYRange(0.0, max(0.5, 1.1 * max(values)))
    plot.getAxis("bottom").setTicks([list(enumerate(names))])
