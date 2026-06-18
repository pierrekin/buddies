"""Build the buddy and write the meshes:  uv run -m buddy

Set YACV=1 to also serve the model in the browser viewer (needs the dev
dependency: uv add --dev yacv-server).
"""
import os

from . import dimensions as d
from .assembly import build_buddy
from .export import export_all


def main():
    buddy = build_buddy()
    export_all(buddy)
    _print_summary(buddy)
    _maybe_serve(buddy)


def _print_summary(buddy):
    x_margin = d.WIDTH / 2 - (d.ARRAY_X / 2 + d.R_PEDESTAL) - d.RIM_RADIUS
    y_margin = d.HEIGHT / 2 - (d.ARRAY_Y / 2 + d.R_PEDESTAL) - d.RIM_RADIUS
    print(f"wrote dist/assembly.glb  bbox={buddy.bounding_box()}")
    print(f"collar flat-face clearance to rim: x={x_margin:.1f} mm  y={y_margin:.1f} mm")
    print(f"element base z={d.TUBE_Z} vs face z={d.FACE_Z}  (proud by {d.TUBE_Z - d.FACE_Z} mm, none recessed)")
    print(f"layup r(mm): bore<{d.PZT_ID/2} | ceramic {d.PZT_ID/2}-{d.PZT_OD/2} | "
          f"window {d.PZT_OD/2}-{d.PZT_OD/2+d.WIN_T} | water")
    print(f"underside: sag {d.ARM_SAG} mm at centre, exp {d.ARM_EXP}, ears tilted {d.EAR_ANGLE:.1f} deg")
    print(f"ear outer edge x={d.EAR_X_PITCH/2 + d.EAR_THICKNESS/2:.1f} mm vs "
          f"corner tangent x={d.WIDTH/2 - d.CORNER_RADIUS:.1f} mm")
    print(f"bungee hole d={d.HOLE_DIAMETER} mm, {d.HOLE_INSET - d.HOLE_DIAMETER/2:.1f} mm meat to ear tip")
    print(f"OLED {d.DISP_W}x{d.DISP_H} mm in +Y wall; window {d.WIN_W}x{d.WIN_H}x{d.WIN_GLASS_T} mm "
          f"on {d.WIN_OVER} mm ledge; bezel bevelled {d.MOUTH_BEVEL} mm")


def _maybe_serve(buddy):
    if not os.environ.get("YACV"):
        return
    try:
        from yacv_server import show
        show(*buddy.children)
        print("yacv: serving at http://localhost:32323")
    except Exception as exc:
        print("yacv unavailable:", exc)


if __name__ == "__main__":
    main()
