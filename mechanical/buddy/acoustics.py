"""The four acoustic elements that sit proud on the housing collars.

Each element is a hollow radial-mode PZT ceramic wrapped in a rho-c urethane
window over its OD and top cap; the open bore is air-backed through the collar.
Per element, inside out:  bore air | ceramic | urethane window | sea water.
"""
from build123d import Align, Cylinder

from . import dimensions as d
from . import materials as m


def build_acoustics():
    """Return (ceramics, windows), each a list of four solids in element order."""
    base = (Align.CENTER, Align.CENTER, Align.MIN)
    ceramic = (Cylinder(d.PZT_OD / 2, d.PZT_LEN, align=base)
               - Cylinder(d.PZT_ID / 2, d.PZT_LEN + 2, align=base))      # hollow, open bore
    window = (Cylinder(d.PZT_OD / 2 + d.WIN_T, d.PZT_LEN + d.WIN_T, align=base)
              - Cylinder(d.PZT_OD / 2, d.PZT_LEN, align=base))           # wall over OD + top cap

    ceramics, windows = [], []
    for i, (x, y) in enumerate(d.ELEMENT_XY):
        # translate (not Pos*) bakes the transform into the geometry, giving each
        # part an identity location so the glTF exporter can name it
        cer = ceramic.translate((x, y, d.TUBE_Z))
        win = window.translate((x, y, d.TUBE_Z))
        cer.label, cer.color = f"pzt_ceramic_{i}", m.CERAMIC.color
        win.label, win.color = f"pzt_window_{i}", m.URETHANE.color
        ceramics.append(cer)
        windows.append(win)
    return ceramics, windows
