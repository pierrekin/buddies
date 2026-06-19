"""Fuse the housing and its mounted parts into the buddy Compound.

The assembly nests into groups that mirror the modules: housing on its own,
the acoustic elements under "acoustics", the display stack under "display".
That nesting is what shapes the exported glTF tree.
"""
from build123d import Compound

from .acoustics import build_acoustics
from .display import build_display
from .housing import build_housing


def build_buddy():
    housing = build_housing()
    ceramics, windows = build_acoustics()
    sight_window, oled = build_display()
    acoustics = Compound(label="acoustics", children=[*ceramics, *windows])
    display = Compound(label="display", children=[sight_window, oled])
    return Compound(label="buddy", children=[housing, acoustics, display])
