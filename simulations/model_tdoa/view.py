"""View extras for ``model_tdoa``.

Per shot:

  * Cross-shot bearing-error curves vs true azimuth: chirp-phys vs
    truth (blue), OOK-phys vs truth (orange), OOK-model vs truth
    (green dashed). The substitution test is whether orange and
    green-dashed sit on top of each other.
  * Cross-shot ``|model - phys|`` delta curve (red): the actual
    substitution error. Headline number in the title.
  * Per-shot title: this shot's truth, estimate(s), and deltas."""

import numpy as np
import pyqtgraph as pg

CHIRP_PEN = (80, 140, 220)    # blue: chirp phys vs truth
PHYS_PEN = (220, 150, 60)     # orange: ook phys vs truth
MODEL_PEN = (120, 200, 100)   # green: ook model vs truth
DELTA_PEN = (220, 80, 80)     # red: model vs phys delta
MARKER_PEN = pg.mkPen("y", style=pg.QtCore.Qt.PenStyle.DashLine)


def extra_views(viewer, layout, start_row):
    extras = viewer.shot.extras
    row = start_row

    if extras.get("az_true_deg"):
        _add_err_curves(layout, row, extras,
                        current=extras.get("this_az_true_deg"))
        row += 1
        _add_delta_curve(layout, row, extras,
                         current=extras.get("this_az_true_deg"))
        row += 1

    role = extras.get("role")
    if role in ("char", "val"):
        _add_shot_summary(layout, row, extras, role)
        row += 1


def _add_err_curves(layout, row, extras, current=None):
    az = np.asarray(extras["az_true_deg"], dtype=np.float64)
    chirp_err = np.asarray(extras["chirp_phys_err_deg"], dtype=np.float64)
    ook_phys_err = np.asarray(extras["ook_phys_err_deg"], dtype=np.float64)
    ook_model_err = np.asarray(extras["ook_model_err_deg"], dtype=np.float64)

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle(
        f"Bearing error vs true azimuth  --  "
        f"chirp phys mean |err|={extras['mean_abs_chirp_err_deg']:.2f} deg, "
        f"ook phys mean |err|={extras['mean_abs_ook_phys_err_deg']:.2f} deg"
    )
    plot.setLabel("left", "estimate - truth", units="deg")
    plot.setLabel("bottom", "true azimuth", units="deg")
    plot.addLegend()

    plot.plot(az, chirp_err, pen=pg.mkPen(CHIRP_PEN, width=2),
              symbol="o", symbolBrush=CHIRP_PEN, name="chirp phys")
    plot.plot(az, ook_phys_err, pen=pg.mkPen(PHYS_PEN, width=2),
              symbol="s", symbolBrush=PHYS_PEN, name="OOK phys")
    plot.plot(az, ook_model_err,
              pen=pg.mkPen(MODEL_PEN, width=2,
                           style=pg.QtCore.Qt.PenStyle.DashLine),
              symbol="t", symbolBrush=MODEL_PEN, name="OOK model")
    plot.addItem(pg.InfiniteLine(
        pos=0.0, angle=0,
        pen=pg.mkPen((180, 180, 180), style=pg.QtCore.Qt.PenStyle.DotLine),
    ))
    if isinstance(current, (int, float)):
        plot.addItem(pg.InfiniteLine(pos=float(current), angle=90, pen=MARKER_PEN))


def _add_delta_curve(layout, row, extras, current=None):
    az = np.asarray(extras["az_true_deg"], dtype=np.float64)
    delta = np.asarray(extras["ook_model_vs_phys_delta_deg"], dtype=np.float64)
    abs_delta = np.abs(delta)

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle(
        f"Substitution error |bearing_model - bearing_phys| vs azimuth  --  "
        f"mean = {extras['mean_abs_model_vs_phys_delta_deg']:.3f} deg, "
        f"max = {extras['max_abs_model_vs_phys_delta_deg']:.3f} deg"
    )
    plot.setLabel("left", "|model - phys|", units="deg")
    plot.setLabel("bottom", "true azimuth", units="deg")
    plot.plot(az, abs_delta, pen=pg.mkPen(DELTA_PEN, width=2),
              symbol="o", symbolBrush=DELTA_PEN)
    if isinstance(current, (int, float)):
        plot.addItem(pg.InfiniteLine(pos=float(current), angle=90, pen=MARKER_PEN))


def _add_shot_summary(layout, row, extras, role):
    az = float(extras.get("this_az_true_deg", 0.0))
    plot = layout.addPlot(row=row, col=0)
    if role == "char":
        bearing = float(extras["this_chirp_bearing_deg"])
        err = float(extras["this_chirp_err_deg"])
        nrmse = extras.get("train_nrmse_per_rx") or []
        title = (f"char shot: az_true={az:.1f} chirp_bearing={bearing:.2f} "
                 f"err={err:+.2f} deg  |  per-RX FIR train NRMSE: "
                 + ", ".join(f"{float(n):.4f}" for n in nrmse))
    else:
        phys_b = float(extras["this_phys_bearing_deg"])
        model_b = float(extras["this_model_bearing_deg"])
        delta = float(extras["this_model_vs_phys_delta_deg"])
        nrmse = extras.get("this_ook_nrmse_per_rx") or []
        title = (f"val shot: az_true={az:.1f} phys={phys_b:.2f} "
                 f"model={model_b:.2f} delta={delta:+.3f} deg  |  "
                 f"per-RX phys-vs-model NRMSE: "
                 + ", ".join(f"{float(n):.4f}" for n in nrmse))
    plot.setTitle(title)
    plot.hideAxis("left")
    plot.hideAxis("bottom")
