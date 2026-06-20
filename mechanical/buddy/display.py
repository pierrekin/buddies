"""The side display stack: a bevelled sight glass over the OLED panel.

These are the separate solids that sit in the +Y wall pocket; the pocket and
bezel themselves are cut into the housing.
"""
from build123d import Box, chamfer

from . import dimensions as d
from . import materials as m


def build_display():
    """Return (sight_window, oled), the two solids in the display layup."""
    blank = Box(d.WIN_W - 0.2, d.WIN_GLASS_T, d.WIN_H - 0.2)
    sight_window = chamfer(
        [e for e in blank.edges() if e.bounding_box().center().Y > d.WIN_GLASS_T / 2 - 0.05],
        d.WIN_BEVEL,
    ).translate((0, d.Y_FACE - d.WIN_GLASS_T / 2, d.DISP_Z))
    oled = Box(d.DISP_W, d.DISP_T, d.DISP_H).translate(
        (0, d.Y_FACE - d.WIN_GLASS_T - d.DISP_AIR - d.DISP_T / 2, d.DISP_Z))

    sight_window.label, sight_window.color = "sight_window", m.GLASS.color
    oled.label, oled.color = "oled", m.OLED.color
    return sight_window, oled
