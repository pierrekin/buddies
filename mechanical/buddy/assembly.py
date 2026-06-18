"""Fuse the housing and its mounted parts into the buddy Compound.

Child order is fixed: housing, then ceramics, windows, sight window, OLED.
The glTF node names are assigned by this order, so keep it stable.
"""
from build123d import Compound

from .acoustics import build_acoustics
from .display import build_display
from .housing import build_housing


def build_buddy():
    housing = build_housing()
    ceramics, windows = build_acoustics()
    sight_window, oled = build_display()
    return Compound(
        label="buddy",
        children=[housing, *ceramics, *windows, sight_window, oled],
    )
