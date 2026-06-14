"""View extras for ``model_array_tx``.

Every shot carries the validation sweep (steering angle -> NRMSE), so
the top extra is the same sweep curve everywhere with a vertical marker
on the current shot's angle. Below it:

  * characterize shots get all eight fitted FIR impulse responses
    overlaid, with the current element's curve highlighted, so you can
    see how the per-element propagation delay shifts the impulse arrival.
  * validate shots get a single-bar 'waveform NRMSE' chart with a
    reference line at the per-element baseline, so it's obvious when
    superposition holds vs. drifts."""

import numpy as np
import pyqtgraph as pg

PHYS_PEN = (80, 140, 220)
MODEL_PEN = (220, 150, 60)
NRMSE_PEN = (200, 100, 200)
MARKER_PEN = pg.mkPen("y", style=pg.QtCore.Qt.PenStyle.DashLine)
HIGHLIGHT_PEN = pg.mkPen("w", width=2.5)
DIM_PENS = [
    pg.mkPen((80, 140, 220, 110), width=1.0),
    pg.mkPen((220, 150, 60, 110), width=1.0),
    pg.mkPen((140, 200, 100, 110), width=1.0),
    pg.mkPen((200, 100, 200, 110), width=1.0),
    pg.mkPen((220, 80, 80, 110), width=1.0),
    pg.mkPen((80, 200, 200, 110), width=1.0),
    pg.mkPen((180, 180, 100, 110), width=1.0),
    pg.mkPen((150, 100, 200, 110), width=1.0),
]


def extra_views(viewer, layout, start_row):
    shot = viewer.shot
    extras = shot.extras
    role = extras.get("role")
    row = start_row

    # Superposition sweep stays on every shot.
    if extras.get("sweep_angles_deg"):
        _add_sweep(layout, row, extras)
        row += 1

    # Comms sweeps stay on every shot too, but only matter when the
    # current shot is on one of those axes.
    if role == "comms":
        if extras.get("comms_steer_angles_deg"):
            _add_comms_steer_sweep(layout, row, extras)
            row += 1
        if extras.get("comms_bitdur_s"):
            _add_comms_bitdur_sweep(layout, row, extras)
            row += 1
        _add_comms_bars(layout, row, extras)
        row += 1
    elif role == "characterize":
        _add_fir_overlay(viewer, layout, row, extras)
        row += 1
    elif role == "validate":
        _add_validate_bar(layout, row, extras)
        row += 1


def _add_sweep(layout, row, extras):
    """Waveform NRMSE (superposition error) vs steering angle. Marker on
    the current shot's angle if it's a validate shot."""
    angles = np.asarray(extras["sweep_angles_deg"], dtype=np.float64)
    nrmse = np.asarray(extras["sweep_nrmse"], dtype=np.float64)

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle("superposition NRMSE vs steering angle  "
                  "(v_rx_phys vs sum_i FIR_i.predict)")
    plot.setLabel("left", "NRMSE")
    plot.setLabel("bottom", "steering angle", units="deg")
    plot.addLegend()
    plot.plot(angles, nrmse, pen=pg.mkPen(NRMSE_PEN, width=2),
              symbol="o", symbolBrush=NRMSE_PEN, name="waveform NRMSE")

    # Reference line at the per-element training baseline so it's clear
    # what 'matches' means -- superposition can't beat the per-element
    # fit error.
    train_baselines = extras.get("train_nrmse_per_element")
    if train_baselines:
        baseline = float(np.mean(train_baselines))
        plot.addItem(pg.InfiniteLine(
            pos=baseline, angle=0,
            pen=pg.mkPen((180, 180, 180), style=pg.QtCore.Qt.PenStyle.DotLine),
            label=f"per-element baseline ~{baseline:.3f}",
            labelOpts={"position": 0.92, "color": (180, 180, 180)},
        ))

    plot.setYRange(0.0, max(0.05, 1.1 * float(nrmse.max())))

    current = extras.get("angle_deg") if extras.get("role") == "validate" else None
    if current is not None:
        plot.addItem(pg.InfiniteLine(pos=float(current), angle=90, pen=MARKER_PEN))


def _add_fir_overlay(viewer, layout, row, extras):
    """All N FIRs on the same axis; highlight the current element."""
    plot = layout.addPlot(row=row, col=0)
    plot.setTitle("fitted FIR impulse responses (all elements; "
                  "current element highlighted)")
    plot.setLabel("left", "tap value")
    plot.setLabel("bottom", "lag", units="s")
    plot.addLegend()

    sample_dt = viewer.shot.channels[0].dt if viewer.shot.channels else viewer.st.meta["dt"]
    current_idx = int(extras.get("element_index", 0))

    h_arrays = []
    i = 0
    while f"fir_h_e{i}" in extras:
        h_arrays.append(np.asarray(extras[f"fir_h_e{i}"]))
        i += 1

    for i, h in enumerate(h_arrays):
        t = np.arange(len(h)) * sample_dt
        if i == current_idx:
            plot.plot(t, h, pen=HIGHLIGHT_PEN, name=f"e{i} (current)")
        else:
            plot.plot(t, h, pen=DIM_PENS[i % len(DIM_PENS)], name=f"e{i}")


def _add_comms_steer_sweep(layout, row, extras):
    """BER_phys, BER_model, disagreement vs steering angle (at the
    comms-sweep bit_dur). Marker on the current shot's angle."""
    angles = np.asarray(extras["comms_steer_angles_deg"], dtype=np.float64)
    _plot_ber_sweep(
        layout, row,
        title="comms: BER vs steering angle (1 ms bits)",
        x_label="steering angle", x_units="deg",
        x=angles,
        ber_phys=extras["comms_steer_ber_phys"],
        ber_model=extras["comms_steer_ber_model"],
        agreement=extras.get("comms_steer_agreement"),
        marker_x=extras.get("angle_deg"),
    )


def _add_comms_bitdur_sweep(layout, row, extras):
    """BER_phys, BER_model, disagreement vs bit duration (at broadside).
    Marker on the current shot's bit_dur if it's on this axis."""
    bit_durs = np.asarray(extras["comms_bitdur_s"], dtype=np.float64)
    marker = extras.get("bit_dur") if extras.get("angle_deg") == 0.0 else None
    _plot_ber_sweep(
        layout, row,
        title="comms: BER vs bit duration (broadside)",
        x_label="bit duration", x_units="s",
        x=bit_durs,
        ber_phys=extras["comms_bitdur_ber_phys"],
        ber_model=extras["comms_bitdur_ber_model"],
        agreement=extras.get("comms_bitdur_agreement"),
        marker_x=marker,
    )


def _plot_ber_sweep(layout, row, *, title, x_label, x_units, x,
                    ber_phys, ber_model, agreement, marker_x):
    plot = layout.addPlot(row=row, col=0)
    plot.setTitle(title)
    plot.setLabel("left", "fraction")
    plot.setLabel("bottom", x_label, units=x_units)
    plot.addLegend()
    plot.plot(x, np.asarray(ber_phys, dtype=np.float64),
              pen=pg.mkPen(PHYS_PEN, width=2),
              symbol="o", symbolBrush=PHYS_PEN, name="BER (physical)")
    plot.plot(x, np.asarray(ber_model, dtype=np.float64),
              pen=pg.mkPen(MODEL_PEN, width=2),
              symbol="o", symbolBrush=MODEL_PEN, name="BER (model)")
    if agreement is not None and len(agreement):
        plot.plot(x, 1.0 - np.asarray(agreement, dtype=np.float64),
                  pen=pg.mkPen((140, 200, 100), width=2,
                               style=pg.QtCore.Qt.PenStyle.DashLine),
                  symbol="s", symbolBrush=(140, 200, 100),
                  name="disagreement (phys vs model)")
    plot.setYRange(-0.02, 1.02)
    if marker_x is not None:
        plot.addItem(pg.InfiniteLine(pos=float(marker_x), angle=90, pen=MARKER_PEN))


def _add_comms_bars(layout, row, extras):
    """Per-shot BER bar trio: BER phys, BER model, disagreement."""
    ber_phys = float(extras["ber_phys"])
    ber_model = float(extras["ber_model"])
    disagreement = 1.0 - float(extras["agreement"])
    angle = float(extras["angle_deg"])
    bit_dur_us = float(extras["bit_dur"]) * 1e6

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle(
        f"per-shot BER  (steering = {angle:+.0f} deg, bit_dur = {bit_dur_us:.0f} us)"
    )
    plot.setLabel("left", "fraction")
    names = ["BER phys", "BER model", "disagreement"]
    values = [ber_phys, ber_model, disagreement]
    brushes = [pg.mkBrush(*c) for c in (PHYS_PEN, MODEL_PEN, (140, 200, 100))]
    plot.addItem(pg.BarGraphItem(
        x=np.arange(len(names)), height=values, width=0.6, brushes=brushes,
    ))
    plot.setYRange(0.0, max(0.5, 1.1 * max(values)))
    plot.getAxis("bottom").setTicks([list(enumerate(names))])


def _add_validate_bar(layout, row, extras):
    """Big single bar for this shot's superposition NRMSE next to the
    per-element baseline, so you can read at a glance whether the
    composition error is at-baseline (LTI holds) or above it (drift)."""
    nrmse = float(extras["waveform_nrmse"])
    baselines = extras.get("train_nrmse_per_element") or [0.0]
    baseline = float(np.mean(baselines))

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle(f"superposition NRMSE for this shot "
                  f"(steering = {float(extras['angle_deg']):+.0f} deg)")
    plot.setLabel("left", "NRMSE")
    names = ["per-element baseline (mean)", "this shot (sum vs phys)"]
    plot.addItem(pg.BarGraphItem(
        x=np.arange(2), height=[baseline, nrmse], width=0.6,
        brushes=[pg.mkBrush(180, 180, 180), pg.mkBrush(*NRMSE_PEN)],
    ))
    plot.setYRange(0.0, max(0.05, 1.15 * max(baseline, nrmse)))
    plot.getAxis("bottom").setTicks([list(enumerate(names))])
