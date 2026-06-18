"""Every buddy dimension, in millimetres.

Pure data. Everything geometric is derived from these, so this is the one
file to edit when tuning the part.
"""
import math

# --- body, landscape ---
# WIDTH 81 -> 108 to seat the 80 mm baseline plus each element's full layup
# (window + decoupling gap, R_PEDESTAL below) inside the rounded rim.
WIDTH = 108              # along X
HEIGHT = 60              # along Y
THICKNESS = 24           # along Z
CORNER_RADIUS = 12       # vertical (Z-parallel) corner rounding
RIM_RADIUS = 5           # front/back rim rounding

# --- concave underside (mimics a forearm; band wraps along Y) ---
ARM_SAG = 5              # mm the underside rises at centre (carved hollow depth)
ARM_EXP = 4              # >2: flat in the middle, curls hard at the edges

# --- ears (4 total: 2 top, 2 bottom) ---
EAR_PROTRUSION = 8       # tip distance from body face, along Y
EAR_WIDTH = 12           # along X (post-rotation: ear Z dim)
EAR_THICKNESS = 16       # along Z (post-rotation: ear X dim)
EAR_OVERLAP = 10         # ear sinks back into body so the rounded inner half is hidden
EAR_SMOOTH_OFFSET = 1    # ear outer edge sits this far inside the corner-arc tangent
EAR_X_PITCH = WIDTH - 2 * CORNER_RADIUS - 2 * EAR_SMOOTH_OFFSET - EAR_THICKNESS
EAR_Z_OFFSET = -THICKNESS / 4   # bias ears toward back half of top/bottom face

# ear tilt: each ear rotates down to stay tangent to the underside at the body edge
EAR_SLOPE = ARM_SAG * ARM_EXP * 2 / HEIGHT   # |dz/dy| of the underside at the Y edge
EAR_ANGLE = math.degrees(math.atan(EAR_SLOPE))

# --- bungee through-hole (axis along X) ---
HOLE_DIAMETER = 5
HOLE_INSET = 5           # from outer tip of ear, along Y

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

FACE_Z = THICKNESS / 2          # front (+Z) face
TUBE_Z = FACE_Z + PEDESTAL_H    # ceramic base on the collar top: whole OD in water

# the four element positions, in build order. This order drives the part
# labels (pzt_ceramic_0..3) and therefore the glTF node names.
ELEMENT_XY = [(sx * ARRAY_X / 2, sy * ARRAY_Y / 2)
              for sx in (+1, -1) for sy in (+1, -1)]

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
CAV_DEPTH = DISP_AIR + DISP_T + CAV_BACK
Y_FACE = HEIGHT / 2             # the +Y wall the display sits in
