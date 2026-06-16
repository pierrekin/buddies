"""View extras for ``model_link_wreck``.

Per shot:

  * NRMSE bar across every validate shot, ordered (same-position
    waveforms first, then off-position chirps). The training NRMSE
    is drawn as a horizontal reference line so it's obvious which
    validate shots are at-baseline (LTI holds) and which are off.
  * Title labels each shot's role (waveform validation vs. position
    validation) so the cause of any rise is identifiable at a glance."""

import numpy as np
import pyqtgraph as pg

WAVEFORM_BRUSH = pg.mkBrush(80, 140, 220)   # blue   -- same-TX waveform sweep
POSITION_BRUSH = pg.mkBrush(220, 150, 60)   # orange -- displaced-TX sweep
BASELINE_PEN = pg.mkPen((180, 180, 180), style=pg.QtCore.Qt.PenStyle.DotLine)
MARKER_PEN = pg.mkPen("y", style=pg.QtCore.Qt.PenStyle.DashLine)


def extra_views(viewer, layout, start_row):
    extras = viewer.shot.extras
    row = start_row

    if extras.get("val_nrmse"):
        _add_nrmse_bars(layout, row, extras, current_name=viewer.shot.name)
        row += 1


REFIT_BRUSH = pg.mkBrush(120, 200, 100)  # green: per-geometry refit's NRMSE


def _add_nrmse_bars(layout, row, extras, current_name=None):
    names = extras["val_shot_names"]
    kinds = extras["val_kind"]
    nrmses_trained = [float(v) for v in extras["val_nrmse"]]
    nrmses_local = [float(v) for v in extras.get("val_nrmse_local", [])]
    baseline = float(extras["train_nrmse"])

    plot = layout.addPlot(row=row, col=0)
    plot.setTitle(
        f"Validation NRMSE  (training baseline {baseline:.4f}; "
        "left half = same TX, different waveforms; "
        "right half = displaced TX, chirp -- "
        "blue = trained-position FIR, green = per-geometry refit)"
    )
    plot.setLabel("left", "NRMSE")
    plot.setLabel("bottom", "shot")

    bar_w = 0.35
    xs = np.arange(len(names), dtype=np.float64)
    # Trained-FIR bars (one per shot).
    trained_brushes = [
        WAVEFORM_BRUSH if k == "validate_waveform" else POSITION_BRUSH
        for k in kinds
    ]
    plot.addItem(pg.BarGraphItem(
        x=xs - bar_w / 2, height=nrmses_trained,
        width=bar_w, brushes=trained_brushes,
    ))
    # Local-refit bars (only on off-position shots; nan elsewhere).
    if nrmses_local:
        refit_xs = [x + bar_w / 2 for x, v in zip(xs, nrmses_local)
                    if not (isinstance(v, float) and v != v)]
        refit_vals = [v for v in nrmses_local
                      if not (isinstance(v, float) and v != v)]
        if refit_vals:
            plot.addItem(pg.BarGraphItem(
                x=refit_xs, height=refit_vals, width=bar_w,
                brushes=[REFIT_BRUSH] * len(refit_vals),
            ))

    plot.addItem(pg.InfiniteLine(
        pos=baseline, angle=0, pen=BASELINE_PEN,
        label=f"train baseline {baseline:.4f}",
        labelOpts={"position": 0.92, "color": (180, 180, 180)},
    ))
    if current_name in names:
        i = names.index(current_name)
        plot.addItem(pg.InfiniteLine(pos=float(xs[i]), angle=90, pen=MARKER_PEN))

    y_top = max(0.05, 1.15 * max(nrmses_trained + [baseline]))
    plot.setYRange(0.0, y_top)
    plot.getAxis("bottom").setTicks(
        [list(zip(xs.tolist(),
                  [n.removeprefix("val_") for n in names]))]
    )
