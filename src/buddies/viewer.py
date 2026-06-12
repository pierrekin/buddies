"""Interactive pyqtgraph viewer for capture files (see buddies.capture)."""

import argparse

import numpy as np
import pyqtgraph as pg

from buddies import capture

DEFAULT_FPS = 60.0
# Display levels are set to this percentile of |p|. Scaling to the global
# maximum (the source peak) would render the decaying wavefront nearly
# invisible.
LEVEL_PERCENTILE = 99.5
COLORMAP = "CET-D1A"  # diverging blue-white-red
WINDOW_SIZE = (800, 850)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", nargs="?", help="capture file to view")
    ap.add_argument(
        "--fps", type=float, default=DEFAULT_FPS, help="playback rate (frames/s)"
    )
    args = ap.parse_args()

    cap = capture.load(args.run)
    frames = cap.frames

    app = pg.mkQApp("FDTD viewer")
    imv = pg.ImageView()
    imv.setWindowTitle(
        f"{args.run} | {frames.shape[0]} frames | {frames.shape[1]}x{frames.shape[2]} cells"
    )

    lim = float(np.percentile(np.abs(frames), LEVEL_PERCENTILE))
    times_ms = np.arange(frames.shape[0]) * cap.dt * 1e3
    imv.setImage(frames, xvals=times_ms, autoLevels=False, levels=(-lim, lim))
    imv.setColorMap(pg.colormap.get(COLORMAP))

    imv.resize(*WINDOW_SIZE)
    imv.show()
    imv.play(args.fps)
    app.exec()


if __name__ == "__main__":
    main()
