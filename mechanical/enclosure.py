"""Broadest-form enclosure for the dive-computer-style buddy.

Landscape pillow body (X = long axis, Z = screen-facing) with four ears
protruding along Y (two top, two bottom) for bungee attachment. Each ear
has a horizontal through-hole whose axis runs along X.
"""
from build123d import (
    Axis,
    Box,
    Cylinder,
    Plane,
    Pos,
    Rot,
    SlotOverall,
    export_stl,
    extrude,
    fillet,
)

# --- body, landscape ---
WIDTH = 81               # along X
HEIGHT = 60              # along Y
THICKNESS = 24           # along Z

CORNER_RADIUS = 12       # vertical (Z-parallel) corner rounding
RIM_RADIUS = 5           # front/back rim rounding

# --- ears (4 total: 2 top, 2 bottom) ---
EAR_PROTRUSION = 8       # tip distance from body face, along Y
EAR_WIDTH = 12           # along X (post-rotation: ear Z dim)
EAR_THICKNESS = 16       # along Z (post-rotation: ear X dim)
EAR_OVERLAP = 10         # ear sinks back into body so the rounded inner
                         # half of the stadium is hidden; must exceed
                         # EAR_WIDTH/2 for a clean rectangular junction
EAR_X_PITCH = 40         # centre-to-centre spacing of paired ears
EAR_Z_OFFSET = -THICKNESS / 4   # bias ears toward back half of top/bottom face

# --- bungee through-hole (axis along X) ---
HOLE_DIAMETER = 6
HOLE_INSET = 4           # from outer tip of ear, along Y

# === body ===
body = Box(WIDTH, HEIGHT, THICKNESS)
body = fillet(body.edges().filter_by(Axis.Z), CORNER_RADIUS)
body = fillet(body.faces().sort_by(Axis.Z)[-1].edges(), RIM_RADIUS)
body = fillet(body.faces().sort_by(Axis.Z)[0].edges(), RIM_RADIUS)

# === ear prototype: stadium with its long axis along Y; rotated so the
# broad flat faces are perpendicular to the hole axis (i.e. facing ±X) ===
ear_total_len = EAR_PROTRUSION + EAR_OVERLAP
ear_proto = Rot(0, 90, 0) * extrude(
    Plane.XY * SlotOverall(ear_total_len, EAR_WIDTH, rotation=90),
    amount=EAR_THICKNESS / 2,
    both=True,
)

# === hole prototype: cylinder, axis along X (rotate Z-axis cylinder 90° about Y) ===
hole_proto = Rot(0, 90, 0) * Cylinder(HOLE_DIAMETER / 2, EAR_THICKNESS * 2)

# === place ears + cut holes ===
edge_y = HEIGHT / 2
ear_cy_offset = (EAR_PROTRUSION - EAR_OVERLAP) / 2

for sign_y in (+1, -1):
    cy_ear = sign_y * (edge_y + ear_cy_offset)
    cy_hole = sign_y * (edge_y + EAR_PROTRUSION - HOLE_INSET)
    for sign_x in (+1, -1):
        cx = sign_x * EAR_X_PITCH / 2
        body += Pos(cx, cy_ear, EAR_Z_OFFSET) * ear_proto
        body -= Pos(cx, cy_hole, EAR_Z_OFFSET) * hole_proto

export_stl(body, "enclosure.stl")
print(f"wrote enclosure.stl  bbox={body.bounding_box()}  volume={body.volume:.1f} mm^3")
