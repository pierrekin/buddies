"""Mesh export: write the assembly as glTF, then rebuild its node tree to
mirror the buddy Compound.

OCCT's glTF export names every node with an internal XCAF label reference
('=>[0:1:1:N]') and wraps each leaf in identity 'instance' empties. We throw
that hierarchy away and author a clean one straight from the Compound, so the
glTF tree matches the Python structure: buddy -> {housing, acoustics, display}.
"""
import json
import struct
from pathlib import Path

from build123d import export_gltf

DIST = Path(__file__).resolve().parent.parent / "dist"

# node transform fields; OCCT omits them entirely when identity
_TRS = ("translation", "rotation", "scale", "matrix")


def export_all(buddy):
    """Write dist/assembly.glb from the buddy Compound, tree mirroring the source."""
    DIST.mkdir(exist_ok=True)
    glb = DIST / "assembly.glb"
    export_gltf(buddy, str(glb), binary=True,
                linear_deflection=0.05, angular_deflection=0.2)
    _retree(glb, buddy)


def _label_tree(part):
    """A build123d part as nested (label, children) where leaves have children=None."""
    kids = list(getattr(part, "children", []) or [])
    return (part.label, [_label_tree(k) for k in kids] if kids else None)


def _leaf_labels(tree):
    label, kids = tree
    if kids is None:
        return [label]
    return [leaf for k in kids for leaf in _leaf_labels(k)]


def _retree(path, buddy):
    js, binchunk = _read_glb(path)
    nodes = js["nodes"]
    scene = js["scenes"][js.get("scene", 0)]

    # OCCT emits the leaf meshes depth-first, in the same order as the Compound
    mesh_nodes, intermediates = [], []

    def walk(i):
        (mesh_nodes if "mesh" in nodes[i] else intermediates).append(i)
        for c in nodes[i].get("children", []):
            walk(c)

    for root in scene["nodes"]:
        walk(root)

    # the scene root carries OCCT's global mm->m scale and Z-up->Y-up rotation;
    # keep it. Component placement lives on each leaf. Everything between (the
    # instance wrappers we drop) must be identity or flattening would move geometry.
    if len(scene["nodes"]) != 1:
        raise ValueError(f"expected one scene root, found {len(scene['nodes'])}")
    root_id = scene["nodes"][0]
    root_xf = {k: nodes[root_id][k] for k in _TRS if k in nodes[root_id]}
    for i in intermediates:
        if i != root_id and any(k in nodes[i] for k in _TRS):
            raise ValueError(f"node[{i}] {nodes[i].get('name')!r} carries a transform; "
                             "flattening would move geometry")

    tree = _label_tree(buddy)
    leaves = _leaf_labels(tree)
    if len(leaves) != len(mesh_nodes):
        raise ValueError(f"{len(leaves)} compound leaves but {len(mesh_nodes)} glTF meshes")

    # carry each leaf's mesh + transform across, named from the Compound; also
    # rename the mesh datablock so viewers that key off it agree
    leaf_node = {}
    for label, i in zip(leaves, mesh_nodes):
        kept = {"name": label, "mesh": nodes[i]["mesh"]}
        kept.update({k: nodes[i][k] for k in _TRS if k in nodes[i]})
        leaf_node[label] = kept
        js["meshes"][nodes[i]["mesh"]]["name"] = label

    new_nodes = []

    def emit(t):
        label, kids = t
        if kids is None:
            new_nodes.append(leaf_node[label])
            return len(new_nodes) - 1
        node = {"name": label}
        new_nodes.append(node)
        idx = len(new_nodes) - 1
        node["children"] = [emit(k) for k in kids]
        return idx

    root_idx = emit(tree)
    new_nodes[root_idx].update(root_xf)   # global scale + axis flip stays on the root
    scene["nodes"] = [root_idx]
    js["nodes"] = new_nodes
    _write_glb(path, js, binchunk)


def _read_glb(path):
    raw = bytearray(open(path, "rb").read())
    jlen = struct.unpack_from("<I", raw, 12)[0]
    js = json.loads(bytes(raw[20:20 + jlen]))
    return js, bytes(raw[20 + jlen:])   # BIN chunk (own header + data), unchanged


def _write_glb(path, js, binchunk):
    newj = json.dumps(js, separators=(",", ":")).encode()
    newj += b" " * (-len(newj) % 4)
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(newj) + len(binchunk))
    out += struct.pack("<II", len(newj), 0x4E4F534A) + newj + binchunk
    open(path, "wb").write(out)
