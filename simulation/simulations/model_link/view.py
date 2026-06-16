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
NRMSE_PEN = (200, 100, 200)  # magenta -- waveform-level mismatch
MARKER_PEN = pg.mkPen("y", style=pg.QtCore.Qt.PenStyle.DashLine)


def extra_views(viewer, layout, start_row):
    shot = viewer.shot
    extras = shot.extras
    role = extras.get("role")
    row = start_row

    # Pick the sweep axis that fits the current shot. Validate shots are
    # off-geometry by construction; their interesting axis is RX offset,
    # not bit duration.
    if role == "validate":
        if extras.get("validate_offsets"):
            _add_offset_sweep(layout, row, extras)
            row += 1
    elif extras.get("sweep_bit_durs"):
        _add_bit_dur_sweep(layout, row, extras)
        row += 1

    if role in ("test", "validate"):
        _add_per_shot_bars(layout, row, extras, role=role)
        row += 1
    elif role == "train" and "fir_h" in extras:
        h = np.asarray(extras["fir_h"])
        sample_dt = shot.channels[0].dt if shot.channels else viewer.st.meta["dt"]
        _add_impulse_response(layout, row, h, sample_dt)
        row += 1


def _add_bit_dur_sweep(layout, row, extras):
    """BER_phys, BER_model, disagreement, and waveform NRMSE vs bit
    duration. The current shot's bit_dur (if applicable) gets a marker."""
    _plot_sweep(
        layout, row,
        title="BER, disagreement, waveform NRMSE vs bit duration",
        x_label="bit duration", x_units="s",
        x=np.asarray(extras["sweep_bit_durs"], dtype=np.float64),
        ber_phys=extras["sweep_ber_phys"],
        ber_model=extras["sweep_ber_model"],
        agreement=extras.get("sweep_agreement"),
        waveform_nrmse=extras.get("sweep_waveform_nrmse"),
        marker_x=extras.get("bit_dur") if extras.get("role") == "test" else None,
    )


def _add_offset_sweep(layout, row, extras):
    """BER_phys, BER_model, disagreement, and waveform NRMSE vs RX y-offset."""
    _plot_sweep(
        layout, row,
        title="BER, disagreement, waveform NRMSE vs RX y-offset (validation: surrogate sees trained geometry only)",
        x_label="RX y-offset from trained position", x_units="m",
        x=np.asarray(extras["validate_offsets"], dtype=np.float64),
        ber_phys=extras["validate_ber_phys"],
        ber_model=extras["validate_ber_model"],
        agreement=extras.get("validate_agreement"),
        waveform_nrmse=extras.get("validate_waveform_nrmse"),
        marker_x=extras.get("rx_offset_m"),
    )


def _plot_sweep(layout, row, *, title, x_label, x_units, x, ber_phys, ber_model,
                agreement, waveform_nrmse, marker_x):
    """BER + disagreement + waveform NRMSE on a shared sweep axis.

    BER and disagreement are fractions in [0, 1]; waveform NRMSE can
    exceed 1 (it does when the surrogate is far from the truth), so the
    y-range grows to accommodate it."""
    plot = layout.addPlot(row=row, col=0)
    plot.setTitle(title)
    plot.setLabel("left", "fraction / NRMSE")
    plot.setLabel("bottom", x_label, units=x_units)
    plot.addLegend()
    plot.plot(x, np.asarray(ber_phys, dtype=np.float64),
              pen=pg.mkPen(PHYS_PEN, width=2),
              symbol="o", symbolBrush=PHYS_PEN, name="BER (physical)")
    plot.plot(x, np.asarray(ber_model, dtype=np.float64),
              pen=pg.mkPen(MODEL_PEN, width=2),
              symbol="o", symbolBrush=MODEL_PEN, name="BER (model)")
    y_top = 1.02
    if agreement is not None and len(agreement):
        plot.plot(x, 1.0 - np.asarray(agreement, dtype=np.float64),
                  pen=pg.mkPen(AGREE_PEN, width=2,
                               style=pg.QtCore.Qt.PenStyle.DashLine),
                  symbol="s", symbolBrush=AGREE_PEN, name="disagreement (phys vs model)")
    if waveform_nrmse is not None and len(waveform_nrmse):
        nrmse_arr = np.asarray(waveform_nrmse, dtype=np.float64)
        plot.plot(x, nrmse_arr,
                  pen=pg.mkPen(NRMSE_PEN, width=2),
                  symbol="t", symbolBrush=NRMSE_PEN,
                  name="waveform NRMSE (v_rx_phys vs v_rx_model)")
        y_top = max(y_top, 1.1 * float(nrmse_arr.max()))
    plot.setYRange(-0.02, y_top)
    if marker_x is not None:
        plot.addItem(pg.InfiniteLine(pos=float(marker_x), angle=90, pen=MARKER_PEN))


def _add_per_shot_bars(layout, row, extras, *, role):
    """One bar per pipeline + disagreement bar. Clean at-a-glance check
    for whether the model's decisions match the physical ones."""
    ber_phys = float(extras["ber_phys"])
    ber_model = float(extras["ber_model"])
    disagreement = 1.0 - float(extras["agreement"])
    bit_dur_us = float(extras["bit_dur"]) * 1e6

    plot = layout.addPlot(row=row, col=0)
    if role == "validate":
        offset_mm = float(extras["rx_offset_m"]) * 1000
        plot.setTitle(
            f"per-shot BER  (validation: bit_dur = {bit_dur_us:.0f} us, "
            f"RX offset = {offset_mm:.0f} mm)"
        )
    else:
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
