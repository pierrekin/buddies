"""View extras for ``model_array_rx``.

Per-shot extras (top to bottom of the scroll):

  1. Beam pattern: composite RMS vs look-angle for both phys and model
     traces, with vertical markers at the peak phys / model / expected
     directions. The peak separation = model's spatial blindness.
  2. Per-RX NRMSE bar chart (eight bars + a baseline reference line),
     so it's clear which channels drift first as TX moves.
  3. Cross-shot DOA sweep: TX displacement -> peak look-angle for phys
     vs model. Marker on the current shot when applicable."""

import numpy as np
import pyqtgraph as pg

PHYS_PEN = (80, 140, 220)
MODEL_PEN = (220, 150, 60)
EXPECTED_PEN = (140, 200, 100)
BASELINE_PEN = pg.mkPen((180, 180, 180), style=pg.QtCore.Qt.PenStyle.DotLine)
MARKER_PEN = pg.mkPen("y", style=pg.QtCore.Qt.PenStyle.DashLine)


def extra_views(viewer, layout, start_row):
    extras = viewer.shot.extras
    role = extras.get("role")
    row = start_row

    if extras.get("beam_angles_deg") is not None:
        _add_beam_pattern(layout, row, extras)
        row += 1

    if role == "characterize":
        _add_per_rx_bars(layout, row, extras["train_nrmse_per_rx"],
                        baseline=extras["train_nrmse_baseline"],
                        title="per-RX training NRMSE")
        row += 1
    elif role == "validate":
        _add_per_rx_bars(layout, row, extras["per_rx_nrmse"],
                        baseline=extras.get("char_train_baseline", 0.0),
                        title=("per-RX validation NRMSE  "
                               f"(mean = {extras['per_rx_nrmse_mean']:.4f})"))
        row += 1
        _add_ber_bars(layout, row, extras)
        row += 1

    if extras.get("sweep_tx_offsets_m"):
        _add_doa_sweep(layout, row, extras)
        row += 1
        _add_ber_sweep(layout, row, extras)
        row += 1


def _add_beam_pattern(layout, row, extras):
    """Beam-pattern curve: composite RMS vs look-angle, phys vs model."""
    angles = np.asarray(extras["beam_angles_deg"], dtype=np.float64)
    phys_rms = np.asarray(extras["beam_phys_rms"], dtype=np.float64)
    model_rms = np.asarray(extras["beam_model_rms"], dtype=np.float64)
    peak_phys = float(extras["peak_phys_deg"])
    peak_model = float(extras["peak_model_deg"])
    expected = float(extras["expected_deg"])

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle(
        f"RX beam pattern (delay-and-sum)  "
        f"peak phys = {peak_phys:+.1f} deg, model = {peak_model:+.1f} deg, "
        f"expected = {expected:+.1f} deg"
    )
    plot.setLabel("left", "composite RMS")
    plot.setLabel("bottom", "look angle off broadside", units="deg")
    plot.addLegend()
    plot.plot(angles, phys_rms, pen=pg.mkPen(PHYS_PEN, width=2),
              name="phys composite")
    plot.plot(angles, model_rms, pen=pg.mkPen(MODEL_PEN, width=2,
                                              style=pg.QtCore.Qt.PenStyle.DashLine),
              name="model composite")
    # Vertical markers: expected (green), peak phys (blue), peak model (orange).
    plot.addItem(pg.InfiniteLine(
        pos=expected, angle=90,
        pen=pg.mkPen(EXPECTED_PEN, style=pg.QtCore.Qt.PenStyle.DotLine),
        label="expected", labelOpts={"position": 0.95},
    ))
    plot.addItem(pg.InfiniteLine(
        pos=peak_phys, angle=90, pen=pg.mkPen(PHYS_PEN, width=1.2),
        label="phys peak", labelOpts={"position": 0.05},
    ))
    plot.addItem(pg.InfiniteLine(
        pos=peak_model, angle=90, pen=pg.mkPen(MODEL_PEN, width=1.2,
                                               style=pg.QtCore.Qt.PenStyle.DashLine),
        label="model peak", labelOpts={"position": 0.50},
    ))


def _add_per_rx_bars(layout, row, nrmse_per_rx, *, baseline, title):
    """Eight bars + reference dotted line at the per-RX training baseline."""
    values = [float(v) for v in nrmse_per_rx]
    plot = layout.addPlot(row=row, col=0)
    plot.setTitle(title)
    plot.setLabel("left", "NRMSE")
    plot.setLabel("bottom", "RX element index")
    plot.addItem(pg.BarGraphItem(
        x=np.arange(len(values)), height=values, width=0.7,
        brushes=[pg.mkBrush(80, 140, 220)] * len(values),
    ))
    if baseline > 0:
        plot.addItem(pg.InfiniteLine(
            pos=baseline, angle=0, pen=BASELINE_PEN,
            label=f"train baseline ~{baseline:.4f}",
            labelOpts={"position": 0.92, "color": (180, 180, 180)},
        ))
    y_top = max(0.05, 1.15 * max(values + [baseline]))
    plot.setYRange(0, y_top)


def _add_ber_bars(layout, row, extras):
    """Four BER bars: phys@correct, phys@trained, model@correct, model@trained,
    plus a natural-pair-agreement bar (1 - agree_natural_pair). The natural
    pair (phys@correct vs model@trained) is the apples-to-apples question:
    when each side decodes at its preferred look-angle, do they agree?"""
    bers = [
        ("phys@correct", float(extras["ber_phys_correct"])),
        ("phys@trained", float(extras["ber_phys_trained"])),
        ("model@correct", float(extras["ber_model_correct"])),
        ("model@trained", float(extras["ber_model_trained"])),
        ("natural pair disagree", 1.0 - float(extras["agree_natural_pair"])),
    ]
    look_correct = float(extras["look_correct_deg"])

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle(
        f"BER at two look angles  "
        f"(correct = {look_correct:+.0f} deg, trained = 0 deg)"
    )
    plot.setLabel("left", "fraction")
    names = [b[0] for b in bers]
    values = [b[1] for b in bers]
    brushes = [pg.mkBrush(*PHYS_PEN), pg.mkBrush(*PHYS_PEN, 160),
               pg.mkBrush(*MODEL_PEN), pg.mkBrush(*MODEL_PEN, 160),
               pg.mkBrush(*EXPECTED_PEN)]
    plot.addItem(pg.BarGraphItem(
        x=np.arange(len(names)), height=values, width=0.6, brushes=brushes,
    ))
    plot.setYRange(0.0, max(0.5, 1.1 * max(values)))
    plot.getAxis("bottom").setTicks([list(enumerate(names))])


def _add_ber_sweep(layout, row, extras):
    """BER vs TX y-displacement for each of the four decode pipelines.
    The phys@correct curve should stay low (real signal aligned); the
    model@trained curve should also stay low (model decodes its own
    geometry); the cross-pairs (phys@trained, model@correct) should
    climb as displacement grows."""
    offsets_m = np.asarray(extras["sweep_tx_offsets_m"], dtype=np.float64)
    series = [
        ("phys@correct", extras["sweep_ber_phys_correct"], pg.mkPen(PHYS_PEN, width=2)),
        ("phys@trained", extras["sweep_ber_phys_trained"],
         pg.mkPen(PHYS_PEN, width=2, style=pg.QtCore.Qt.PenStyle.DashLine)),
        ("model@correct", extras["sweep_ber_model_correct"],
         pg.mkPen(MODEL_PEN, width=2, style=pg.QtCore.Qt.PenStyle.DashLine)),
        ("model@trained", extras["sweep_ber_model_trained"], pg.mkPen(MODEL_PEN, width=2)),
    ]
    plot = layout.addPlot(row=row, col=0)
    plot.setTitle("BER vs TX y-displacement, by decode pipeline")
    plot.setLabel("left", "BER")
    plot.setLabel("bottom", "TX y-displacement", units="m")
    plot.addLegend()
    for name, vals, pen in series:
        plot.plot(offsets_m, np.asarray(vals, dtype=np.float64),
                  pen=pen, symbol="o", name=name)
    # Natural-pair disagreement -- the cleanest "do they agree?" number.
    agree = np.asarray(extras["sweep_agree_natural_pair"], dtype=np.float64)
    plot.plot(offsets_m, 1.0 - agree,
              pen=pg.mkPen(EXPECTED_PEN, width=2,
                           style=pg.QtCore.Qt.PenStyle.DotLine),
              symbol="s",
              name="natural-pair disagreement")
    plot.setYRange(-0.02, 1.02)
    current = extras.get("tx_offset_y_m")
    if current is not None and current > 0:
        plot.addItem(pg.InfiniteLine(pos=float(current), angle=90, pen=MARKER_PEN))


def _add_doa_sweep(layout, row, extras):
    """DOA estimate vs TX displacement: phys (correct), model (frozen at
    trained position), expected (ground truth)."""
    offsets_mm = np.asarray(extras["sweep_tx_offsets_m"], dtype=np.float64) * 1000.0
    phys_deg = np.asarray(extras["sweep_peak_phys_deg"], dtype=np.float64)
    model_deg = np.asarray(extras["sweep_peak_model_deg"], dtype=np.float64)
    expected_deg = np.asarray(extras["sweep_expected_deg"], dtype=np.float64)

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle("DOA estimate vs TX y-displacement  "
                  "(beamformer peak look-angle)")
    plot.setLabel("left", "peak look-angle", units="deg")
    plot.setLabel("bottom", "TX y-displacement", units="m")
    plot.addLegend()
    plot.plot(offsets_mm / 1000.0, expected_deg,
              pen=pg.mkPen(EXPECTED_PEN, width=2,
                           style=pg.QtCore.Qt.PenStyle.DotLine),
              symbol="d", symbolBrush=EXPECTED_PEN, name="expected")
    plot.plot(offsets_mm / 1000.0, phys_deg,
              pen=pg.mkPen(PHYS_PEN, width=2),
              symbol="o", symbolBrush=PHYS_PEN, name="phys peak")
    plot.plot(offsets_mm / 1000.0, model_deg,
              pen=pg.mkPen(MODEL_PEN, width=2,
                           style=pg.QtCore.Qt.PenStyle.DashLine),
              symbol="s", symbolBrush=MODEL_PEN, name="model peak")

    current = extras.get("tx_offset_y_m")
    if current is not None and current > 0:
        plot.addItem(pg.InfiniteLine(pos=float(current), angle=90, pen=MARKER_PEN))
