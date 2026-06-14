"""View extras for ``model_link``.

Every shot carries the full sweep (bit_dur -> BER_phys, BER_model,
agreement), so the top extra plot is the same sweep curve everywhere
with a vertical marker on the current shot's bit duration. Below it:

  * link shots get a bar trio (BER_phys, BER_model, disagreement)
  * the train shot gets the fitted FIR impulse response

The sweep on the train shot is informational -- the FIR hasn't been
evaluated on chirp_train as a comm pipeline, but seeing the curve next
to the impulse response keeps both pieces of context in one place."""

import numpy as np
import pyqtgraph as pg

PHYS_PEN = (80, 140, 220)   # blue
MODEL_PEN = (220, 150, 60)  # orange
AGREE_PEN = (140, 200, 100)  # green
MARKER_PEN = pg.mkPen("y", style=pg.QtCore.Qt.PenStyle.DashLine)


def extra_views(viewer, layout, start_row):
    shot = viewer.shot
    extras = shot.extras
    row = start_row

    bit_durs = extras.get("sweep_bit_durs")
    if bit_durs and extras.get("sweep_ber_phys") is not None:
        _add_sweep(layout, row, extras)
        row += 1

    role = extras.get("role")
    if role == "test":
        _add_per_shot_bars(layout, row, extras)
        row += 1
    elif role == "train" and "fir_h" in extras:
        h = np.asarray(extras["fir_h"])
        sample_dt = shot.channels[0].dt if shot.channels else viewer.st.meta["dt"]
        _add_impulse_response(layout, row, h, sample_dt)
        row += 1


def _add_sweep(layout, row, extras):
    """BER_phys, BER_model, and agreement vs bit duration. The current
    shot's bit_dur (if any) gets a dashed vertical marker."""
    bit_durs = np.asarray(extras["sweep_bit_durs"], dtype=np.float64) * 1e6  # us
    ber_phys = np.asarray(extras["sweep_ber_phys"], dtype=np.float64)
    ber_model = np.asarray(extras["sweep_ber_model"], dtype=np.float64)
    agreement = np.asarray(extras.get("sweep_agreement", []), dtype=np.float64)

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle("BER and agreement vs bit duration (sweep)")
    plot.setLabel("left", "fraction")
    plot.setLabel("bottom", "bit duration", units="s")
    # Plot in seconds so pyqtgraph SI-prefixes the ticks; the data is in
    # microseconds above for human-friendly extras values, undo here.
    bit_durs_s = bit_durs * 1e-6
    plot.addLegend()
    plot.plot(bit_durs_s, ber_phys, pen=pg.mkPen(PHYS_PEN, width=2),
              symbol="o", symbolBrush=PHYS_PEN, name="BER (physical)")
    plot.plot(bit_durs_s, ber_model, pen=pg.mkPen(MODEL_PEN, width=2),
              symbol="o", symbolBrush=MODEL_PEN, name="BER (model)")
    if agreement.size:
        plot.plot(bit_durs_s, 1.0 - agreement, pen=pg.mkPen(AGREE_PEN, width=2,
                  style=pg.QtCore.Qt.PenStyle.DashLine),
                  symbol="s", symbolBrush=AGREE_PEN, name="disagreement (phys vs model)")
    plot.setYRange(-0.02, 1.02)

    current = extras.get("bit_dur")
    if current is not None:
        plot.addItem(pg.InfiniteLine(pos=float(current), angle=90, pen=MARKER_PEN))


def _add_per_shot_bars(layout, row, extras):
    """One bar per pipeline + disagreement bar. Clean at-a-glance check
    for whether the model's decisions match the physical ones."""
    ber_phys = float(extras["ber_phys"])
    ber_model = float(extras["ber_model"])
    disagreement = 1.0 - float(extras["agreement"])
    bit_dur_us = float(extras["bit_dur"]) * 1e6

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle(f"per-shot BER  (bit_dur = {bit_dur_us:.0f} us)")
    plot.setLabel("left", "fraction")

    names = ["BER phys", "BER model", "disagreement"]
    values = [ber_phys, ber_model, disagreement]
    brushes = [pg.mkBrush(*c) for c in (PHYS_PEN, MODEL_PEN, AGREE_PEN)]
    plot.addItem(pg.BarGraphItem(
        x=np.arange(len(names)), height=values, width=0.7, brushes=brushes,
    ))
    plot.setYRange(0.0, max(0.5, 1.1 * max(values)))
    plot.getAxis("bottom").setTicks([list(enumerate(names))])


def _add_impulse_response(layout, row, h, sample_dt):
    """Just the fitted FIR taps -- handy reminder of what the model is."""
    plot = layout.addPlot(row=row, col=0)
    plot.setTitle(f"fitted FIR impulse response  ({len(h)} taps)")
    plot.setLabel("left", "tap value")
    plot.setLabel("bottom", "lag", units="s")
    t = np.arange(len(h)) * sample_dt
    plot.plot(t, h, pen=pg.mkPen(MODEL_PEN, width=1.5))
