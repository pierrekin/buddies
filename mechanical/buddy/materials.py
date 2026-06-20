"""Semantic material palette: one colour per material class.

These colours exist for CAD-viewer legibility and to seed each part's glTF
material. The colour is the identity key: OCCT maps one colour to one glTF
material, so every part sharing a class lands on a single material, which the
exporter renames to the class name. The actual look (PBR: roughness, metallic,
textures) is authored in Blender against those names, not here.

Edit a colour here and it changes everywhere that class is used. Keep the
colours distinct per class so the label->material mapping stays unambiguous.
"""
from dataclasses import dataclass

from build123d import Color


@dataclass(frozen=True)
class Material:
    name: str
    rgba: tuple  # (r, g, b) opaque, or (r, g, b, a) for translucent parts

    @property
    def color(self):
        return Color(*self.rgba)


HOUSING = Material("housing", (0.90, 0.10, 0.10))           # red
CERAMIC = Material("ceramic", (0.10, 0.80, 0.10))           # green
URETHANE = Material("urethane", (0.10, 0.30, 1.00, 0.50))   # blue, translucent
GLASS = Material("glass", (0.10, 0.90, 0.90, 0.35))         # cyan, translucent
OLED = Material("oled", (0.95, 0.85, 0.10))                 # yellow

ALL = (HOUSING, CERAMIC, URETHANE, GLASS, OLED)

# Map a part label to its material. A label is either the class name itself
# ("housing") or "<class>_<index>" ("pzt_window_0"); the prefixes below carry
# that mapping for the per-element parts.
_BY_PREFIX = {
    "housing": HOUSING,
    "pzt_ceramic": CERAMIC,
    "pzt_window": URETHANE,
    "sight_window": GLASS,
    "oled": OLED,
}


def material_for_label(label):
    """The Material for a part/node label, or None if it maps to no class."""
    for prefix, mat in _BY_PREFIX.items():
        if label == prefix or label.startswith(prefix + "_"):
            return mat
    return None
