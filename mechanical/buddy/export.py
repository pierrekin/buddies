"""Mesh export: write the full assembly as glTF, then fix up the glTF node
names OCCT leaves generic.
"""
import json
import struct
from pathlib import Path

from build123d import export_gltf

DIST = Path(__file__).resolve().parent.parent / "dist"


def export_all(buddy):
    """Write dist/assembly.glb from the buddy Compound."""
    DIST.mkdir(exist_ok=True)
    glb = DIST / "assembly.glb"
    export_gltf(buddy, str(glb), binary=True,
                linear_deflection=0.05, angular_deflection=0.2)
    name_glb_nodes(glb, [c.label for c in buddy.children])


def name_glb_nodes(path, labels):
    """OCCT names glTF meshes for assembly components generically ('SOLID');
    rewrite node + mesh names from the child labels (glTF node order = child order)."""
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
