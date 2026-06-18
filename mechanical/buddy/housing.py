"""The printed shell: the single solid everything else mounts to.

Builds the rounded landscape body, carves the concave forearm underside, cuts
the side display pocket with its bevelled bezel, adds the four tilted bungee
ears with their through-holes, and raises the four acoustic collars with their
air-backing bores. The pockets and bores are features of this one part, so
they live here; the parts that sit in them live in display.py and acoustics.py.
"""
from build123d import (
    Align,
    Axis,
    Box,
    Color,
    Cylinder,
    Plane,
    Polyline,
    Pos,
    Rot,
    SlotOverall,
    chamfer,
    extrude,
    fillet,
    make_face,
)

from . import dimensions as d


def _under_z(y):
    """Z of the carved underside at a given Y: flat centre, hard curl at the edges."""
    return -d.THICKNESS / 2 + d.ARM_SAG * (1 - (2 * abs(y) / d.HEIGHT) ** d.ARM_EXP)


def build_housing():
    body = Box(d.WIDTH, d.HEIGHT, d.THICKNESS)
    body = fillet(body.edges().filter_by(Axis.Z), d.CORNER_RADIUS)
    body = fillet(body.faces().sort_by(Axis.Z)[-1].edges(), d.RIM_RADIUS)  # top rim only

    body = _carve_underside(body)
    body = _cut_display_pocket(body)
    body = _add_ears(body)
    body = _add_collars(body)

    body.label, body.color = "housing", Color(0.25, 0.27, 0.30)
    return body


def _carve_underside(body):
    """Subtract a super-circular hollow along Y, constant in X."""
    prof = [(-d.HEIGHT / 2 + d.HEIGHT * i / 40, _under_z(-d.HEIGHT / 2 + d.HEIGHT * i / 40))
            for i in range(41)]
    prof += [(d.HEIGHT / 2, -d.THICKNESS), (-d.HEIGHT / 2, -d.THICKNESS)]   # close below the body
    return body - extrude(Plane.YZ * make_face(Polyline(*prof, close=True)),
                          amount=d.WIDTH, both=True)


def _cut_display_pocket(body):
    """Stepped rebate into the +Y wall: window seat -> ledge -> deeper cavity, bevelled mouth."""
    body -= Pos(0, d.Y_FACE - d.WIN_GLASS_T / 2, d.DISP_Z) * Box(d.WIN_W, d.WIN_GLASS_T, d.WIN_H)
    body -= Pos(0, d.Y_FACE - d.WIN_GLASS_T - d.CAV_DEPTH / 2, d.DISP_Z) * Box(
        d.DISP_W + d.CAV_CLEAR, d.CAV_DEPTH, d.DISP_H + d.CAV_CLEAR)

    mouth = []
    for e in body.edges():
        c = e.bounding_box().center()
        if (abs(c.Y - d.Y_FACE) < 0.05
                and abs(c.X) <= d.WIN_W / 2 + 0.5
                and abs(c.Z - d.DISP_Z) <= d.WIN_H / 2 + 0.5):
            mouth.append(e)
    return chamfer(mouth, d.MOUTH_BEVEL)


def _add_ears(body):
    """Four bungee ears, each rigidly tilted down to the underside's edge tangent."""
    ear_total_len = d.EAR_PROTRUSION + d.EAR_OVERLAP
    # stadium with long axis along Y, rotated so the broad flats face +-X
    ear = Rot(0, 90, 0) * extrude(
        Plane.XY * SlotOverall(ear_total_len, d.EAR_WIDTH, rotation=90),
        amount=d.EAR_THICKNESS / 2, both=True)
    # cylinder, axis along X (Z-axis cylinder rotated 90 about Y)
    hole = Rot(0, 90, 0) * Cylinder(d.HOLE_DIAMETER / 2, d.EAR_THICKNESS * 2)

    edge_y = d.HEIGHT / 2
    ear_cy_offset = (d.EAR_PROTRUSION - d.EAR_OVERLAP) / 2
    for sign_y in (+1, -1):
        cy_ear = sign_y * (edge_y + ear_cy_offset)
        cy_hole = sign_y * (edge_y + d.EAR_PROTRUSION - d.HOLE_INSET)
        # hinge on the underside at the body edge so the ear bottom stays tangent
        tilt = (Pos(0, sign_y * edge_y, -d.THICKNESS / 2)
                * Rot(-sign_y * d.EAR_ANGLE, 0, 0)
                * Pos(0, -sign_y * edge_y, d.THICKNESS / 2))
        for sign_x in (+1, -1):
            cx = sign_x * d.EAR_X_PITCH / 2
            body += tilt * Pos(cx, cy_ear, d.EAR_Z_OFFSET) * ear
            body -= tilt * Pos(cx, cy_hole, d.EAR_Z_OFFSET) * hole
    return body


def _add_collars(body):
    """Raise a chamfered collar at each element and drill its air-backing bore."""
    base = (Align.CENTER, Align.CENTER, Align.MIN)   # base at z=0, extrude +Z
    ped_blank = Cylinder(d.R_PEDESTAL, d.PEDESTAL_H, align=base)
    pedestal = chamfer(ped_blank.edges().sort_by(Axis.Z)[-1], d.PEDESTAL_CHAMFER)
    bore = Cylinder(d.PZT_ID / 2, d.PEDESTAL_H + d.BORE_DEPTH, align=base)
    for x, y in d.ELEMENT_XY:
        body += Pos(x, y, d.FACE_Z) * pedestal               # raised collar
        body -= Pos(x, y, d.FACE_Z - d.BORE_DEPTH) * bore    # air-backing bore into housing
    return body
