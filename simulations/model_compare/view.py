"""View extras for ``model_compare``.

For every shot, render an NRMSE bar chart so you can see at a glance which
model wins on this signal. For the training (chirp) shot, also plot the
fitted impulse responses (M3 FIR taps) -- this is the data-driven channel
filter the model has learned."""

import numpy as np
import pyqtgraph as pg

BAR_BRUSHES = [
    (200, 80, 80),  # red — worst
    (220, 150, 60),
    (210, 200, 80),
    (140, 200, 100),
    (80, 200, 200),
    (80, 140, 220),  # blue — best
]


def extra_views(viewer, layout, start_row):
    shot = viewer.shot

    nrmse_map = shot.extras.get("nrmse")
    if nrmse_map:
        _add_nrmse_bars(layout, start_row, nrmse_map, shot.name)
        start_row += 1

    if shot.extras.get("role") == "train":
        # FIR coefficient arrays are stored under "<model>__h" keys.
        h_arrays = {
            k[: -len("__h")]: v
            for k, v in shot.extras.items()
            if k.endswith("__h") and hasattr(v, "shape")
        }
        if h_arrays:
            # The taps live at the sim's native sample dt, which equals any
            # channel's dt. Fall back to the artifact's frame dt if the shot
            # has no channels for some reason.
            sample_dt = shot.channels[0].dt if shot.channels else viewer.st.meta["dt"]
            _add_impulse_responses(layout, start_row, h_arrays, sample_dt)
            start_row += 1


def _add_nrmse_bars(layout, row, nrmse_map, shot_name):
    """Bar chart: one bar per model, ordered as the lineup was built."""
    plot = layout.addPlot(row=row, col=0)
    plot.setLabel("left", "NRMSE")
    plot.setLabel("bottom", "model")
    plot.setTitle(f"NRMSE for shot '{shot_name}' (lower is better; 1 = predicts nothing)")

    names = list(nrmse_map.keys())
    values = [float(nrmse_map[n]) for n in names]
    xs = np.arange(len(names))

    brushes = [pg.mkBrush(*BAR_BRUSHES[i % len(BAR_BRUSHES)]) for i in range(len(names))]
    bg = pg.BarGraphItem(x=xs, height=values, width=0.7, brushes=brushes)
    plot.addItem(bg)

    # Mark NRMSE=1 (the "predict zero" reference line).
    plot.addItem(pg.InfiniteLine(
        pos=1.0, angle=0, pen=pg.mkPen("y", style=pg.QtCore.Qt.PenStyle.DashLine),
    ))

    ax = plot.getAxis("bottom")
    ax.setTicks([list(zip(xs.tolist(), names))])


def _add_impulse_responses(layout, row, h_arrays, sample_dt):
    """Overlay the fitted FIR taps on a time axis (lag in seconds)."""
    plot = layout.addPlot(row=row, col=0)
    plot.setLabel("left", "tap value")
    plot.setLabel("bottom", "lag", units="s")
    plot.setTitle("Learned impulse responses h(t) — the M3 channels' filter")
    plot.addLegend()

    items = sorted(h_arrays.items(), key=lambda kv: len(kv[1]))
    for i, (name, h) in enumerate(items):
        h = np.asarray(h)
        t = np.arange(len(h)) * sample_dt
        pen = pg.mkPen(BAR_BRUSHES[(i + 3) % len(BAR_BRUSHES)], width=1.5)
        plot.plot(t, h, pen=pen, name=name)
