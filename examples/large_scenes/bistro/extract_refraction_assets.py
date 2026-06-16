# Extracts a wine bottle + wine glass (with their liquid meshes) from NVIDIA's
# BistroInterior_Wine FBX into a small GLB for the `solari_refraction` example,
# with the same KHR transmission/ior/volume patch `prepare_bistro.py` applies
# (values from NVIDIA's BistroInterior_Wine.pyscene).
#
#   blender --background --python extract_refraction_assets.py -- \
#       /path/to/Bistro_v5_2 /path/to/assets/models/refraction.glb

import bpy
import inspect
import re
import sys
from pathlib import Path
from mathutils import Vector

FBX_DIR = Path.home() / "Downloads/Bistro_v5_2/Bistro_v5_2"
OUT_PATH = Path("/mnt/code/p/aurora/assets/models/refraction.glb")
if "--" in sys.argv:
    cli = sys.argv[sys.argv.index("--") + 1 :]
    if len(cli) >= 1:
        FBX_DIR = Path(cli[0])
    if len(cli) >= 2:
        OUT_PATH = Path(cli[1])

bpy.ops.wm.read_homefile(use_empty=True)

# Blender 5.x alpha removed CyclesLightSettings.cast_shadow but the bundled
# FBX importer still assigns it; rebuild blen_read_light with that line stubbed.
from io_scene_fbx import import_fbx as _fbx_mod

_src = inspect.getsource(_fbx_mod.blen_read_light)
_patched = re.sub(r"lamp\.cycles\.cast_shadow\s*=.*", "pass", _src)
exec(compile(_patched, "<shim>", "exec"), _fbx_mod.__dict__)

bpy.ops.import_scene.fbx(filepath=str(FBX_DIR / "BistroInterior_Wine.fbx"))
# The tall table wine bottle (Paris_LiquorBottle_01_Glass_Wine) only exists in
# the exterior file.
bpy.ops.import_scene.fbx(filepath=str(FBX_DIR / "BistroExterior.fbx"))


def material_names(obj):
    return {m.name.split(".")[0] for m in obj.data.materials if m}


def world_center(obj):
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    return sum(corners, Vector()) / 8.0


meshes = [o for o in bpy.data.objects if o.type == "MESH"]

# Anchor on a red-wine liquid, then take the nearest glass and bottle so the
# trio is one physically-adjacent table setting.
liquid = next(o for o in meshes if "Red_Wine" in material_names(o))
glass = min(
    (o for o in meshes if "TransparentGlass" in material_names(o)),
    key=lambda o: (world_center(o) - world_center(liquid)).length,
)
bottle = min(
    (o for o in meshes if "TransparentGlassWine" in material_names(o)),
    key=lambda o: (world_center(o) - world_center(glass)).length,
)
# The dark-green table wine bottle from the exterior cafe tables.
wine_bottle = next(
    o for o in meshes if "Paris_LiquorBottle_01_Glass_Wine" in material_names(o)
)
keep = [bottle, wine_bottle, glass, liquid]
for obj in keep:
    print(f"keeping {obj.name} ({sorted(material_names(obj))})")

# Detach from the (about-to-be-deleted) FBX hierarchy, keeping world
# transforms — the root carries the FBX unit scale.
for obj in keep:
    world = obj.matrix_world.copy()
    obj.parent = None
    obj.matrix_world = world
bpy.context.view_layer.update()

for obj in list(bpy.data.objects):
    if obj not in keep:
        bpy.data.objects.remove(obj, do_unlink=True)

# Arrange a table setting at the origin: the glass (with its wine, which must
# keep its exact relative position) left of center, the bottle to the right,
# both standing on z = 0. They come from different spots in the bar, so each
# is re-rooted independently.
def place(objs, anchor, target_xy):
    floor = min(
        (obj.matrix_world @ Vector(c)).z for obj in objs for c in obj.bound_box
    )
    c = world_center(anchor)
    offset = Vector((target_xy[0] - c.x, target_xy[1] - c.y, -floor))
    for obj in objs:
        obj.matrix_world.translation += offset


place([glass, liquid], glass, (-0.08, 0.0))
place([bottle], bottle, (0.1, -0.04))
place([wine_bottle], wine_bottle, (0.24, 0.03))
bpy.context.view_layer.update()
for obj in keep:
    c = world_center(obj)
    print(f"  {obj.name[:40]}: center {tuple(round(v, 3) for v in c)}")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.export_scene.gltf(
    filepath=str(OUT_PATH),
    export_format="GLB",
    export_image_format="NONE",
)

# Same glass patch as prepare_bistro.py (subset of materials present here).
sys.path.insert(0, str(Path(__file__).parent))
src = open(Path(__file__).parent / "prepare_bistro.py").read()
start = src.index("# name -> (ior, roughness, absorption sigma 1/m or None)")
end = src.index("patch_glass_materials(OUT_PATH)")
exec(src[start:end])
patch_glass_materials(OUT_PATH)  # noqa: F821
print(f"exported {OUT_PATH}")
