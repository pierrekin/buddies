"""View extras for ``model_array_mimo``.

Per shot:

  * Beam pattern: composite RMS vs look-angle (phys vs model).
    Characterize: one peak at the trained TX direction.
    Joint: two peaks, one per stream -- the spatial multiplex made
    visible.
  * Per-RX NRMSE bars (training or validation).
  * BER bars: per-shot for single-stream shots; per-stream (A then B)
    for the joint shot."""

import numpy as np
import pyqtgraph as pg

PHYS_PEN = (80, 140, 220)
MODEL_PEN = (220, 150, 60)
EXPECTED_PEN = (140, 200, 100)
STREAM_A_PEN = (200, 100, 200)
STREAM_B_PEN = (90, 200, 220)
BASELINE_PEN = pg.mkPen((180, 180, 180), style=pg.QtCore.Qt.PenStyle.DotLine)


def extra_views(viewer, layout, start_row):
    extras = viewer.shot.extras
    role = extras.get("role")
    row = start_row

    if extras.get("beam_angles_deg") is not None:
        _add_beam_pattern(layout, row, extras, role)
        row += 1

    if role == "characterize":
        _add_per_rx_bars(
            layout, row, extras["train_nrmse_per_rx"],
            baseline=extras["train_nrmse_baseline"],
            title="per-RX training NRMSE",
        )
        row += 1
    elif role == "validate_alone":
        _add_per_rx_bars(
            layout, row, extras["per_rx_nrmse"],
            baseline=extras["train_nrmse_baseline"],
            title=("per-RX validation NRMSE  "
                   f"(mean = {extras['per_rx_nrmse_mean']:.4f})"),
        )
        row += 1
        _add_single_ber_bars(layout, row, extras)
        row += 1
    elif role == "joint":
        _add_per_rx_bars(
            layout, row, extras["per_rx_nrmse"],
            baseline=extras["train_nrmse_baseline"],
            title=("per-RX validation NRMSE on the mix  "
                   f"(mean = {extras['per_rx_nrmse_mean']:.4f})"),
        )
        row += 1
        _add_joint_ber_bars(layout, row, extras)
        row += 1


def _add_beam_pattern(layout, row, extras, role):
    angles = np.asarray(extras["beam_angles_deg"], dtype=np.float64)
    phys_rms = np.asarray(extras["beam_phys_rms"], dtype=np.float64)
    model_rms = np.asarray(extras["beam_model_rms"], dtype=np.float64)

    plot = layout.addPlot(row=row, col=0)
    plot.setLabel("left", "composite RMS")
    plot.setLabel("bottom", "look angle off broadside", units="deg")
    plot.addLegend()
    plot.plot(angles, phys_rms, pen=pg.mkPen(PHYS_PEN, width=2),
              name="phys composite")
    plot.plot(angles, model_rms, pen=pg.mkPen(MODEL_PEN, width=2,
                                              style=pg.QtCore.Qt.PenStyle.DashLine),
              name="model composite")

    if role == "joint":
        expected_a = float(extras["expected_a_deg"])
        expected_b = float(extras["expected_b_deg"])
        plot.setTitle(
            f"RX beam pattern (joint)  "
            f"TX_A direction = {expected_a:+.1f} deg, "
            f"TX_B = {expected_b:+.1f} deg -- two peaks expected"
        )
        plot.addItem(pg.InfiniteLine(
            pos=expected_a, angle=90,
            pen=pg.mkPen(STREAM_A_PEN, style=pg.QtCore.Qt.PenStyle.DotLine),
            label="TX_A", labelOpts={"position": 0.95},
        ))
        plot.addItem(pg.InfiniteLine(
            pos=expected_b, angle=90,
            pen=pg.mkPen(STREAM_B_PEN, style=pg.QtCore.Qt.PenStyle.DotLine),
            label="TX_B", labelOpts={"position": 0.05},
        ))
    else:
        expected = float(extras.get("expected_deg", 0.0))
        title_extra = ""
        if role == "validate_alone":
            look = float(extras.get("look_angle_deg", expected))
            title_extra = f" -- demix look = {look:+.1f} deg"
        plot.setTitle(f"RX beam pattern  expected = {expected:+.1f} deg{title_extra}")
        plot.addItem(pg.InfiniteLine(
            pos=expected, angle=90,
            pen=pg.mkPen(EXPECTED_PEN, style=pg.QtCore.Qt.PenStyle.DotLine),
            label="expected", labelOpts={"position": 0.95},
        ))


def _add_per_rx_bars(layout, row, nrmse_per_rx, *, baseline, title):
    values = [float(v) for v in nrmse_per_rx]
    plot = layout.addPlot(row=row, col=0)
    plot.setTitle(title)
    plot.setLabel("left", "NRMSE")
    plot.setLabel("bottom", "RX element index")
    plot.addItem(pg.BarGraphItem(
        x=np.arange(len(values)), height=values, width=0.7,
        brushes=[pg.mkBrush(*PHYS_PEN)] * len(values),
    ))
    if baseline > 0:
        plot.addItem(pg.InfiniteLine(
            pos=baseline, angle=0, pen=BASELINE_PEN,
            label=f"train baseline ~{baseline:.4f}",
            labelOpts={"position": 0.92, "color": (180, 180, 180)},
        ))
    plot.setYRange(0, max(0.05, 1.15 * max(values + [baseline])))


def _add_single_ber_bars(layout, row, extras):
    ber_phys = float(extras["ber_phys"])
    ber_model = float(extras["ber_model"])
    disagreement = 1.0 - float(extras["agreement"])
    label = extras.get("label", "")

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle(f"per-shot BER  ({label})")
    plot.setLabel("left", "fraction")
    names = ["BER phys", "BER model", "disagreement"]
    values = [ber_phys, ber_model, disagreement]
    brushes = [pg.mkBrush(*c) for c in (PHYS_PEN, MODEL_PEN, EXPECTED_PEN)]
    plot.addItem(pg.BarGraphItem(
        x=np.arange(len(names)), height=values, width=0.6, brushes=brushes,
    ))
    plot.setYRange(0.0, max(0.5, 1.1 * max(values)))
    plot.getAxis("bottom").setTicks([list(enumerate(names))])


def _add_joint_ber_bars(layout, row, extras):
    """Two stream rows of BER: stream A and stream B side by side."""
    ber_a_phys = float(extras["ber_a_phys"])
    ber_a_model = float(extras["ber_a_model"])
    disagreement_a = 1.0 - float(extras["agreement_a"])
    ber_b_phys = float(extras["ber_b_phys"])
    ber_b_model = float(extras["ber_b_model"])
    disagreement_b = 1.0 - float(extras["agreement_b"])

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle("joint demixed BER (stream A | stream B)")
    plot.setLabel("left", "fraction")
    names = [
        "A phys", "A model", "A disagree",
        "B phys", "B model", "B disagree",
    ]
    values = [ber_a_phys, ber_a_model, disagreement_a,
              ber_b_phys, ber_b_model, disagreement_b]
    brushes = [
        pg.mkBrush(*STREAM_A_PEN), pg.mkBrush(*STREAM_A_PEN, 180),
        pg.mkBrush(*EXPECTED_PEN),
        pg.mkBrush(*STREAM_B_PEN), pg.mkBrush(*STREAM_B_PEN, 180),
        pg.mkBrush(*EXPECTED_PEN),
    ]
    plot.addItem(pg.BarGraphItem(
        x=np.arange(len(names)), height=values, width=0.6, brushes=brushes,
    ))
    plot.setYRange(0.0, max(0.5, 1.1 * max(values)))
    plot.getAxis("bottom").setTicks([list(enumerate(names))])
