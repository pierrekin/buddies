"""Broadest-form enclosure for the dive-computer-style buddy.

Landscape pillow body (X = long axis, Z = screen-facing) with four ears
protruding along Y (two top, two bottom) for bungee attachment. Each ear
has a horizontal through-hole whose axis runs along X.

The underside (-Z) is carved to a concave, super-circular curve along Y to
mimic a forearm (flat in the middle, curling hard at the edges). The ears
are rigidly tilted down so their bottoms stay tangent to that curve at the
body edge.

Four radial-mode PZT cylinders sit at the corners of the bearing array on
the front (+Z) face, axes along Z so the radial mode is omni in the X-Y
plane (the plane we bear in). Each element sits fully proud on a chamfered
collar so its whole OD couples to sea water; nothing is recessed into the
housing. Radial layup per element, inside out:

  bore air (void) | ceramic 1.2 mm | urethane window WIN_T | sea water

The only internal regions are the air-backed bore (a blind cavity into the
housing, also the lead route) and the window cap sealing the top. The
compliant mount and the bore back-seal are not modelled yet.
"""
from build123d import (
    Align,
    Axis,
    Box,
    Color,
    Compound,
    Cylinder,
    Plane,
    Polyline,
    Pos,
    Rot,
    SlotOverall,
    chamfer,
    export_gltf,
    export_stl,
    extrude,
    fillet,
    make_face,
)
import math
import os

# --- body, landscape ---
# WIDTH 81 -> 108 to seat the 80 mm baseline plus each element's full layup
# (window + decoupling gap, R_pocket below) inside the rounded rim.
WIDTH = 108              # along X
HEIGHT = 60              # along Y
THICKNESS = 24           # along Z

CORNER_RADIUS = 12       # vertical (Z-parallel) corner rounding
RIM_RADIUS = 5           # front/back rim rounding

# --- concave underside (mimics a forearm; band wraps along Y) ---
ARM_SAG = 5              # mm the underside rises at centre (carved hollow depth)
ARM_EXP = 4              # >2: flat in the middle, curls hard at the edges (not a cylinder)

# --- ears (4 total: 2 top, 2 bottom) ---
EAR_PROTRUSION = 8       # tip distance from body face, along Y
EAR_WIDTH = 12           # along X (post-rotation: ear Z dim)
EAR_THICKNESS = 16       # along Z (post-rotation: ear X dim)
EAR_OVERLAP = 10         # ear sinks back into body so the rounded inner
                         # half of the stadium is hidden; must exceed
                         # EAR_WIDTH/2 for a clean rectangular junction
EAR_SMOOTH_OFFSET = 1    # ear outer edge sits this far inside the corner-arc tangent (room to fillet)
EAR_X_PITCH = WIDTH - 2 * CORNER_RADIUS - 2 * EAR_SMOOTH_OFFSET - EAR_THICKNESS  # paired-ear pitch
EAR_Z_OFFSET = -THICKNESS / 4   # bias ears toward back half of top/bottom face

# --- bungee through-hole (axis along X) ---
HOLE_DIAMETER = 5
HOLE_INSET = 5           # from outer tip of ear, along Y (meat to tip = HOLE_INSET - HOLE_DIAMETER/2)

# --- acoustic array element (Steminc SMC1186T10410) + layup ---
PZT_OD = 11              # ceramic outer diameter
PZT_ID = 8.6             # ceramic inner diameter (hollow, air-backed bore)
PZT_LEN = 10             # ceramic length, axis along Z
ARRAY_X = 80             # resolving baseline (long axis, transverse to target)
ARRAY_Y = 32             # short axis (axial; rim-limited, not bearing-limited)

WIN_T = 0.6              # rho-c urethane window/coating over OD + top cap
PEDESTAL_H = 0.8         # raised chamfered collar each element sits proud on
PEDESTAL_EXTRA = 1.0     # collar radius beyond the window
PEDESTAL_CHAMFER = 0.4   # top-outer-edge chamfer, cosmetic
BORE_DEPTH = 4.0         # blind air-backing cavity + lead route, into the housing
R_PEDESTAL = PZT_OD / 2 + WIN_T + PEDESTAL_EXTRA

# --- side OLED + sight window, set into the +Y wall, between that side's ears ---
DISP_W = 40              # OLED module width (X)
DISP_H = 11              # OLED module height (Z)
DISP_T = 1.4             # OLED panel thickness (into the wall, -Y)
DISP_Z = -2.0            # vertical centre on the wall (clears front rim and carved bottom)
WIN_OVER = 2.5           # window overhang onto the housing ledge, per side
WIN_GLASS_T = 1.5        # sight-window thickness
WIN_BEVEL = 0.6          # window edge chamfer (visible glass bevel)
DISP_AIR = 0.4           # window-to-OLED gap
CAV_CLEAR = 1.0          # display-cavity clearance over the module (X and Z)
CAV_BACK = 1.0           # cavity depth behind the panel (FPC / bond room)
MOUTH_BEVEL = 0.7        # housing opening chamfer (beveled bezel)
WIN_W = DISP_W + 2 * WIN_OVER   # window rests on a WIN_OVER ledge around the cavity
WIN_H = DISP_H + 2 * WIN_OVER

# === body ===
body = Box(WIDTH, HEIGHT, THICKNESS)
body = fillet(body.edges().filter_by(Axis.Z), CORNER_RADIUS)
body = fillet(body.faces().sort_by(Axis.Z)[-1].edges(), RIM_RADIUS)  # top rim only

# === concave underside: carve a super-circular hollow along Y, constant in X ===
def _under_z(y):
    return -THICKNESS / 2 + ARM_SAG * (1 - (2 * abs(y) / HEIGHT) ** ARM_EXP)

_prof = [(-HEIGHT / 2 + HEIGHT * i / 40, _under_z(-HEIGHT / 2 + HEIGHT * i / 40))
         for i in range(41)]
_prof += [(HEIGHT / 2, -THICKNESS), (-HEIGHT / 2, -THICKNESS)]   # close below the body
body -= extrude(Plane.YZ * make_face(Polyline(*_prof, close=True)), amount=WIDTH, both=True)

# === side OLED + sight window: stepped layup into the +Y wall ===
# rebate (window seat) -> ledge -> deeper display cavity; mouth bevelled.
y_face = HEIGHT / 2
cav_depth = DISP_AIR + DISP_T + CAV_BACK
body -= Pos(0, y_face - WIN_GLASS_T / 2, DISP_Z) * Box(WIN_W, WIN_GLASS_T, WIN_H)
body -= Pos(0, y_face - WIN_GLASS_T - cav_depth / 2, DISP_Z) * Box(
    DISP_W + CAV_CLEAR, cav_depth, DISP_H + CAV_CLEAR)

_mouth = []
for _e in body.edges():
    _c = _e.bounding_box().center()
    if (abs(_c.Y - y_face) < 0.05
            and abs(_c.X) <= WIN_W / 2 + 0.5
            and abs(_c.Z - DISP_Z) <= WIN_H / 2 + 0.5):
        _mouth.append(_e)
body = chamfer(_mouth, MOUTH_BEVEL)        # bevelled bezel mouth

# sight window (beveled glass) and OLED panel, as separate solids in the layup
_win_blank = Box(WIN_W - 0.2, WIN_GLASS_T, WIN_H - 0.2)
sight_window = chamfer(
    [e for e in _win_blank.edges() if e.bounding_box().center().Y > WIN_GLASS_T / 2 - 0.05],
    WIN_BEVEL,
).translate((0, y_face - WIN_GLASS_T / 2, DISP_Z))
oled = Box(DISP_W, DISP_T, DISP_H).translate(
    (0, y_face - WIN_GLASS_T - DISP_AIR - DISP_T / 2, DISP_Z))

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

# === place ears, each rigidly tilted down to the underside's edge tangent ===
edge_y = HEIGHT / 2
ear_cy_offset = (EAR_PROTRUSION - EAR_OVERLAP) / 2
ear_slope = ARM_SAG * ARM_EXP * 2 / HEIGHT          # |dz/dy| of the underside at the Y edge
ear_angle = math.degrees(math.atan(ear_slope))

for sign_y in (+1, -1):
    cy_ear = sign_y * (edge_y + ear_cy_offset)
    cy_hole = sign_y * (edge_y + EAR_PROTRUSION - HOLE_INSET)
    # hinge on the underside at the body edge so the ear bottom stays tangent
    tilt = (Pos(0, sign_y * edge_y, -THICKNESS / 2)
            * Rot(-sign_y * ear_angle, 0, 0)
            * Pos(0, -sign_y * edge_y, THICKNESS / 2))
    for sign_x in (+1, -1):
        cx = sign_x * EAR_X_PITCH / 2
        body += tilt * Pos(cx, cy_ear, EAR_Z_OFFSET) * ear_proto
        body -= tilt * Pos(cx, cy_hole, EAR_Z_OFFSET) * hole_proto

# === acoustic array layup: elements sit fully proud on the collars ===
BASE = (Align.CENTER, Align.CENTER, Align.MIN)   # base at z=0, extrude +Z
face_z = THICKNESS / 2
tube_z = face_z + PEDESTAL_H                      # ceramic base on the collar top: whole OD in water

ped_blank = Cylinder(R_PEDESTAL, PEDESTAL_H, align=BASE)
pedestal = chamfer(ped_blank.edges().sort_by(Axis.Z)[-1], PEDESTAL_CHAMFER)
bore = Cylinder(PZT_ID / 2, PEDESTAL_H + BORE_DEPTH, align=BASE)  # air column through collar into body
ceramic = (Cylinder(PZT_OD / 2, PZT_LEN, align=BASE)
           - Cylinder(PZT_ID / 2, PZT_LEN + 2, align=BASE))      # hollow, open bore
window = (Cylinder(PZT_OD / 2 + WIN_T, PZT_LEN + WIN_T, align=BASE)
          - Cylinder(PZT_OD / 2, PZT_LEN, align=BASE))           # wall over OD + top cap

ceramics, windows = [], []
for sx in (+1, -1):
    for sy in (+1, -1):
        p = (sx * ARRAY_X / 2, sy * ARRAY_Y / 2)
        body += Pos(*p, face_z) * pedestal               # raised collar
        body -= Pos(*p, face_z - BORE_DEPTH) * bore       # air-backing bore into housing
        # translate (not Pos*) bakes the transform into the geometry, giving each
        # part an identity location so the glTF exporter can name it
        ceramics.append(ceramic.translate((*p, tube_z)))
        windows.append(window.translate((*p, tube_z)))

x_margin = WIDTH / 2 - (ARRAY_X / 2 + R_PEDESTAL) - RIM_RADIUS
y_margin = HEIGHT / 2 - (ARRAY_Y / 2 + R_PEDESTAL) - RIM_RADIUS

# === labels + colours for the glTF / Blender import ===
body.label, body.color = "housing", Color(0.25, 0.27, 0.30)
sight_window.label, sight_window.color = "sight_window", Color(0.40, 0.60, 0.90, 0.35)
oled.label, oled.color = "oled", Color(0.05, 0.05, 0.07)
for i, (cer, win) in enumerate(zip(ceramics, windows)):
    cer.label, cer.color = f"pzt_ceramic_{i}", Color(0.80, 0.80, 0.82)
    win.label, win.color = f"pzt_window_{i}", Color(0.30, 0.50, 0.90, 0.50)

def name_glb_nodes(path, labels):
    """OCCT names glTF meshes for assembly components generically ('SOLID');
    rewrite node + mesh names from the child labels (glTF node order = child order)."""
    import json
    import struct

    raw = bytearray(open(path, "rb").read())
    jlen = struct.unpack_from("<I", raw, 12)[0]
    js = json.loads(bytes(raw[20:20 + jlen]))
    ordered = []

    def walk(i):
        node = js["nodes"][i]
        if "mesh" in node:
            ordered.append(node)
        for child in node.get("children", []):
            walk(child)

    for root in js["scenes"][js.get("scene", 0)]["nodes"]:
        walk(root)
    for node, label in zip(ordered, labels):
        node["name"] = label
        js["meshes"][node["mesh"]]["name"] = label

    newj = json.dumps(js, separators=(",", ":")).encode()
    newj += b" " * (-len(newj) % 4)
    rest = bytes(raw[20 + jlen:])   # BIN chunk (own header + data), unchanged
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(newj) + len(rest))
    out += struct.pack("<II", len(newj), 0x4E4F534A) + newj + rest
    open(path, "wb").write(out)


export_stl(body, "enclosure.stl")
assembly = Compound(label="buddy", children=[body, *ceramics, *windows, sight_window, oled])
export_stl(assembly, "assembly.stl")
export_gltf(assembly, "assembly.glb", binary=True, linear_deflection=0.05, angular_deflection=0.2)
name_glb_nodes("assembly.glb", [c.label for c in assembly.children])
print(f"wrote enclosure.stl + assembly.stl + assembly.glb  bbox={assembly.bounding_box()}")
print(f"collar flat-face clearance to rim: x={x_margin:.1f} mm  y={y_margin:.1f} mm")
print(f"element base z={tube_z} vs face z={face_z}  (proud by {tube_z - face_z} mm, none recessed)")
print(f"layup r(mm): bore<{PZT_ID/2} | ceramic {PZT_ID/2}-{PZT_OD/2} | "
      f"window {PZT_OD/2}-{PZT_OD/2+WIN_T} | water")
print(f"underside: sag {ARM_SAG} mm at centre, exp {ARM_EXP}, ears tilted {ear_angle:.1f} deg")
print(f"ear outer edge x={EAR_X_PITCH/2 + EAR_THICKNESS/2:.1f} mm vs corner tangent x={WIDTH/2 - CORNER_RADIUS:.1f} mm")
print(f"bungee hole d={HOLE_DIAMETER} mm, {HOLE_INSET - HOLE_DIAMETER/2:.1f} mm meat to ear tip")
print(f"OLED {DISP_W}x{DISP_H} mm in +Y wall; window {WIN_W}x{WIN_H}x{WIN_GLASS_T} mm on "
      f"{WIN_OVER} mm ledge; bezel {len(_mouth)} edges bevelled {MOUTH_BEVEL} mm")

if os.environ.get("YACV"):   # browser viewer: uv add --dev yacv-server, then YACV=1 uv run ...
    try:
        from yacv_server import show
        show(body, sight_window, oled, *ceramics, *windows)
        print("yacv: serving at http://localhost:32323")
    except Exception as exc:
        print("yacv unavailable:", exc)
