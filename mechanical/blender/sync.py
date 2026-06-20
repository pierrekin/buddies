"""Re-sync the CAD geometry into the current Blender scene without losing work.

Run from Blender (Text Editor -> Run Script, or bind to a key). It imports the
freshly exported dist/assembly.glb, then for every part that already exists in
the scene it swaps ONLY the mesh data onto the existing object. Materials,
modifiers, origins, parenting and lighting live on the object, so they stay put.
Parts that don't exist yet are kept as new objects; the throwaway import
hierarchy is deleted.

Workflow:  uv run -m buddy   ->   click Run here.
"""
import re
from pathlib import Path

import bpy

# dist/assembly.glb lives at the repo root. Resolve it relative to this script
# (blender/sync.py -> repo root) when run from disk, else relative to the .blend.
def _find_glb():
    candidates = []
    here = globals().get("__file__")
    if here:
        candidates.append(Path(here).resolve().parent.parent / "dist" / "assembly.glb")
    candidates.append(Path(bpy.path.abspath("//")) / "dist" / "assembly.glb")
    return next((p for p in candidates if p.exists()), None)


GLB = _find_glb()

_SUFFIX = re.compile(r"\.\d{3}$")          # Blender's ".001" de-dupe suffix


def _base(name):
    return _SUFFIX.sub("", name)


def _swap_mesh(dst, new_mesh):
    """Point dst at new_mesh, re-applying dst's existing material slots."""
    saved = [(s.link, s.material) for s in dst.material_slots]
    dst.data = new_mesh
    new_mesh.name = dst.name                # keep datablock name tidy
    while len(dst.data.materials) < len(saved):
        dst.data.materials.append(None)
    for i, (link, mat) in enumerate(saved):
        dst.material_slots[i].link = link
        dst.material_slots[i].material = mat


def _adopt_class_materials(obj):
    """Point a freshly-added part at the existing authored material for its class.

    The exporter names each glTF material after its class (urethane, housing...),
    so the import arrives carrying that name. If a material of that name already
    exists (the one you authored PBR on), re-point the slot at it; Blender's
    import-time '.001' duplicate is then orphaned and purged below.
    """
    for slot in obj.material_slots:
        if slot.material is None:
            continue
        canonical = bpy.data.materials.get(_base(slot.material.name))
        if canonical is not None and canonical is not slot.material:
            slot.material = canonical


def sync():
    if GLB is None:
        raise FileNotFoundError("assembly.glb not found; run `uv run -m buddy` first")

    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(GLB))
    imported = set(bpy.data.objects) - before

    updated, added, trash = [], [], []
    for obj in imported:
        if obj.type != "MESH":
            trash.append(obj)              # buddy/group empties from the import
            continue
        target = bpy.data.objects.get(_base(obj.name))
        if target is not None and target in before and target.type == "MESH":
            _swap_mesh(target, obj.data)
            updated.append(target.name)
            trash.append(obj)              # mesh borrowed; drop the carrier object
        else:
            obj.name = _base(obj.name)
            _adopt_class_materials(obj)    # hook new part to its class's material
            added.append(obj.name)         # genuinely new part, keep it

    for obj in trash:
        bpy.data.objects.remove(obj, do_unlink=True)

    # drop now-orphaned meshes/empties the import left behind
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)

    print(f"sync: updated {updated or 'nothing'} | added {added or 'nothing'}")


if __name__ == "__main__":
    sync()
