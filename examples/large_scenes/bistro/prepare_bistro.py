# Prepares NVIDIA's Bistro_v5_2 FBX release for the bistro example: imports
# BistroExterior + BistroInterior_Wine into one scene, reconciles them, and
# exports a single glTF.
#
# Run headless:
#   blender --background --python prepare_bistro.py -- /path/to/Bistro_v5_2 /path/to/out/Bistro.glb
# or open it in Blender's Text Editor, edit the two paths below, and Run Script.
#
# What it does:
#   1. Imports both FBX files into their own collections.
#   2. Measures the scale + position mismatch between the two files using the
#      bistro building shell that ships duplicated in BistroInterior_Wine, and
#      transforms the interior to match the exterior exactly.
#   3. Deletes the interior's duplicate copy of the building shell (the
#      exterior's full-detail copy becomes the only one).
#   4. Splits the entrance door mesh (`dOORS_2`) into left/right panels
#      (full-width pieces like the frame/transom stay static) and swings both
#      panels open around their hinge edges.
#   5. Adds planar UVs to meshes that have none (tangent generation needs them).
#   6. Deletes cameras, then exports everything as one .glb.

import bpy
import bmesh
import math
import re
import sys
from pathlib import Path
from mathutils import Matrix, Vector

# ---------------------------------------------------------------------------
# Config (positional CLI args after `--` override the two paths)
# ---------------------------------------------------------------------------

FBX_DIR = Path.home() / "Downloads/Bistro_v5_2/Bistro_v5_2"
OUT_PATH = FBX_DIR / "Bistro.glb"

# The interior ships its own copy of the bistro building (walls/door/facade)
# so it works standalone; with the exterior loaded these are coplanar
# duplicates.
SHELL_PREFIX = "Bistro_Research_Exterior"
DOOR_NAME = "dOORS_2"
# Degrees each panel swings. Negative swings the other way (use it if the
# doors open into the wall).
DOOR_OPEN_DEGREES = 100.0

if "--" in sys.argv:
    cli = sys.argv[sys.argv.index("--") + 1 :]
    if len(cli) >= 1:
        FBX_DIR = Path(cli[0])
    if len(cli) >= 2:
        OUT_PATH = Path(cli[1])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def base_name(name: str) -> str:
    """Object name without Blender's duplicate suffix (`Wall.001` -> `Wall`)."""
    return re.sub(r"\.\d+$", "", name)


def world_center(obj) -> Vector:
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    return sum(corners, Vector()) / 8.0


def collection_roots(col):
    return [o for o in col.objects if o.parent is None or o.parent.name not in col.objects]


def import_fbx(filepath: Path, collection_name: str):
    col = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(col)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(filepath))
    imported = [o for o in bpy.data.objects if o not in before]
    for obj in imported:
        for other in obj.users_collection:
            other.objects.unlink(obj)
        col.objects.link(obj)
    print(f"imported {len(imported)} objects from {filepath.name}")
    return col


def median(values):
    values = sorted(values)
    return values[len(values) // 2]


# ---------------------------------------------------------------------------
# 1. Fresh scene + imports
# ---------------------------------------------------------------------------

bpy.ops.wm.read_homefile(use_empty=True)

# Blender 5.x alpha removed CyclesLightSettings.cast_shadow but the bundled
# FBX importer still assigns it, aborting the whole import on the first lamp.
# Rebuild blen_read_light with that line stubbed out.
try:
    import inspect

    from io_scene_fbx import import_fbx as _fbx_mod

    _src = inspect.getsource(_fbx_mod.blen_read_light)
    _patched = re.sub(r"lamp\.cycles\.cast_shadow\s*=.*", "pass", _src)
    if _patched != _src:
        exec(compile(_patched, "<prepare_bistro shim>", "exec"), _fbx_mod.__dict__)
        print("shimmed io_scene_fbx.blen_read_light (cycles.cast_shadow)")
except ImportError:
    pass

exterior = import_fbx(FBX_DIR / "BistroExterior.fbx", "BistroExterior")
interior = import_fbx(FBX_DIR / "BistroInterior_Wine.fbx", "BistroInterior_Wine")

# ---------------------------------------------------------------------------
# 2. Alignment check. NVIDIA's own pipeline (the .pyscene files) imports both
#    FBX files with no transforms -- they are authored aligned and in meters,
#    and Blender's unit-aware FBX import preserves that. The duplicated shell
#    has no usable name/geometry correspondence between the files (hash names
#    collide across unrelated meshes), so only a conservative residual check
#    is possible: nearest-neighbor offsets between shell pieces. A correction
#    is applied only if a strong majority agrees on the same small offset.
# ---------------------------------------------------------------------------

from collections import defaultdict


def shell_objs(col, prefix):
    return [
        o
        for o in col.objects
        if o.type == "MESH" and base_name(o.name).startswith(prefix)
    ]


ext_centers = [world_center(o) for o in shell_objs(exterior, SHELL_PREFIX)]
votes = []
for int_obj in shell_objs(interior, SHELL_PREFIX):
    center = world_center(int_obj)
    votes.append(min((c - center for c in ext_centers), key=lambda v: v.length))
bins = defaultdict(list)
for v in votes:
    if v.length < 2.0:
        bins[tuple(round(c / 0.25) for c in v)].append(v)
cluster = max(bins.values(), key=len, default=[])
if len(cluster) >= len(votes) // 2 and cluster:
    offset = sum(cluster, Vector()) / len(cluster)
    if offset.length > 0.01:
        print(f"residual offset {tuple(round(v, 4) for v in offset)} -- moving interior")
        for obj in collection_roots(interior):
            obj.matrix_world = Matrix.Translation(offset) @ obj.matrix_world
        bpy.context.view_layer.update()
    else:
        print("interior already aligned")
else:
    print("no consistent residual offset found -- trusting authored placement")

# ---------------------------------------------------------------------------
# 3. Delete the interior's duplicate shell
# ---------------------------------------------------------------------------

duplicates = [o for o in interior.objects if base_name(o.name).startswith(SHELL_PREFIX)]
print(f"removing {len(duplicates)} duplicate shell objects from the interior")
for obj in duplicates:
    bpy.data.objects.remove(obj, do_unlink=True)

# ---------------------------------------------------------------------------
# 4. Split the entrance doors and swing them open. The entrance is a pair of
#    full-height arched leaves (the lattice runs continuously up into the
#    arch -- there is no separate transom). Connectivity can't separate them
#    (welded at the seam), and the door is smeared across objects: its glass
#    and parts of its frame live in the Paris_Building street meshes. On top
#    of that the entrance sits diagonally on the building corner, so nothing
#    is world-axis-aligned: all classification happens in the DOOR'S LOCAL
#    FRAME (X = width, tight bbox, true seam plane, true hinge edge). Faces
#    go to the left/right leaf their center falls in; only the floor
#    threshold strip stays with the street (the mesh is bisected at the sill
#    line first so no face spans the boundary).
# ---------------------------------------------------------------------------

door = next((o for o in exterior.objects if base_name(o.name) == DOOR_NAME), None)
if door is None:
    raise RuntimeError(f"door object {DOOR_NAME!r} not found in the exterior")

mw = door.matrix_world
to_local = mw.inverted()
bbox_min = Vector((min(v.co[i] for v in door.data.vertices) for i in range(3)))
bbox_max = Vector((max(v.co[i] for v in door.data.vertices) for i in range(3)))
extent = bbox_max - bbox_min

# Door-local axes: vertical is whichever maps closest to world Z; width is
# the larger of the other two.
basis = mw.to_3x3()
up_axis = max(range(3), key=lambda k: abs(basis.col[k].normalized().z))
horizontal = [k for k in range(3) if k != up_axis]
width_axis = max(horizontal, key=lambda k: extent[k])
width = extent[width_axis]
height = extent[up_axis]
mid = (bbox_max[width_axis] + bbox_min[width_axis]) / 2.0
# One world meter in door-local units (the FBX root carries a cm scale).
local_per_meter = 1.0 / mw.to_scale()[width_axis]
pad = 0.05 * local_per_meter


def in_door_region(world_point):
    p = to_local @ world_point
    return all(bbox_min[i] - pad <= p[i] <= bbox_max[i] + pad for i in range(3))


def door_face(obj, world_center, material_index):
    """Whether this face is part of the entrance doors.

    The doors are smeared across several objects: `dOORS_2` plus faces baked
    into the Paris_Building street meshes, distinguished only by their
    dedicated materials (`MASTER_Bistro_Main_Door`, glass). Threshold/step
    faces (concrete) in the same region stay with the street.
    """
    if not in_door_region(world_center):
        return False
    if obj is door:
        return True
    materials = obj.data.materials
    mat = materials[material_index] if material_index < len(materials) else None
    return mat is not None and ("Door" in mat.name or "Glass" in mat.name)


# Every exterior mesh contributing faces to the doors.
door_objects = []
for obj in exterior.objects:
    if obj.type != "MESH":
        continue
    hits = sum(
        1
        for poly in obj.data.polygons
        if door_face(obj, obj.matrix_world @ poly.center, poly.material_index)
    )
    if hits:
        door_objects.append(obj)
        print(f"door faces in {obj.name[:60]}: {hits}")

sill_top = bbox_min[up_axis] + 0.01 * height


def bisect_at_sill(obj):
    """Cut this object's door faces along the sill plane so no face spans
    the moving/static boundary."""
    omw = obj.matrix_world
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="FACE")
    bm = bmesh.from_edit_mesh(obj.data)
    count = 0
    for f in bm.faces:
        f.select = False
    for f in bm.faces:
        if in_door_region(omw @ f.calc_center_median()):
            f.select = True
            count += 1
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data)
    if count:
        co = (bbox_min + bbox_max) / 2.0
        co[up_axis] = sill_top
        bpy.ops.mesh.bisect(plane_co=mw @ co, plane_no=(0.0, 0.0, 1.0))
    bpy.ops.object.mode_set(mode="OBJECT")


def split_by_side(obj, side):
    """Separate `obj`'s door faces on one side of the seam into a new
    object; returns None if no faces matched."""
    omw = obj.matrix_world
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="FACE")
    bm = bmesh.from_edit_mesh(obj.data)
    count = 0
    for f in bm.faces:
        f.select = False
    for f in bm.faces:
        world_center = omw @ f.calc_center_median()
        if not door_face(obj, world_center, f.material_index):
            continue
        local = to_local @ world_center
        if local[up_axis] < sill_top:
            continue  # floor threshold: stays with the street
        face_side = "left" if local[width_axis] < mid else "right"
        if face_side == side:
            f.select = True
            count += 1
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data)
    piece = None
    if count:
        before = set(bpy.data.objects)
        bpy.ops.mesh.separate(type="SELECTED")
        piece = next(o for o in bpy.data.objects if o not in before)
    bpy.ops.object.mode_set(mode="OBJECT")
    return piece


# Each side's moving set: pieces carved out of every contributing object.
moving = {"left": [], "right": []}
for obj in list(door_objects):
    bisect_at_sill(obj)
for side in ("left", "right"):
    for obj in door_objects:
        piece = split_by_side(obj, side)
        if piece is not None:
            piece.name = f"{DOOR_NAME}_{side}_{len(moving[side])}"
            moving[side].append(piece)
    if not moving[side]:
        raise RuntimeError(f"no door faces found on the {side} side")

# Swing each side's pieces around the vertical axis through its hinge: the
# outer edge of that side's combined bounds, in the door's local frame.
for side, sign in (("left", 1.0), ("right", -1.0)):
    locals_ = [
        to_local @ (obj.matrix_world @ Vector(c))
        for obj in moving[side]
        for c in obj.bound_box
    ]
    values = [p[width_axis] for p in locals_]
    hinge_local = sum(locals_, Vector()) / len(locals_)
    hinge_local[width_axis] = min(values) if side == "left" else max(values)
    hinge = mw @ hinge_local
    angle = sign * math.radians(DOOR_OPEN_DEGREES)
    pivot = Matrix.Translation(hinge) @ Matrix.Rotation(angle, 4, "Z") @ Matrix.Translation(-hinge)
    for obj in moving[side]:
        obj.matrix_world = pivot @ obj.matrix_world
print(f"doors split and opened {DOOR_OPEN_DEGREES} degrees "
      "(negative DOOR_OPEN_DEGREES swings the other way)")

# ---------------------------------------------------------------------------
# 5. Planar UVs for meshes without any (mikktspace tangent generation
#    requires a UV layer; these are props like the cypresses and lanterns)
# ---------------------------------------------------------------------------

fixed = 0
for mesh in bpy.data.meshes:
    if mesh.uv_layers or not mesh.polygons:
        continue
    bounds_min = Vector((min(v.co[i] for v in mesh.vertices) for i in range(3)))
    bounds_max = Vector((max(v.co[i] for v in mesh.vertices) for i in range(3)))
    extent = bounds_max - bounds_min
    # Project along the thinnest local axis.
    axes = sorted(range(3), key=lambda i: extent[i])[1:]
    scale = max(extent[axes[0]], extent[axes[1]], 1e-6)
    layer = mesh.uv_layers.new()
    for loop in mesh.loops:
        co = mesh.vertices[loop.vertex_index].co
        layer.data[loop.index].uv = (
            (co[axes[0]] - bounds_min[axes[0]]) / scale,
            (co[axes[1]] - bounds_min[axes[1]]) / scale,
        )
    fixed += 1
print(f"added planar UVs to {fixed} meshes that had none")

# ---------------------------------------------------------------------------
# 6. Cleanup + export
# ---------------------------------------------------------------------------

cameras = [o for o in bpy.data.objects if o.type == "CAMERA"]
for obj in cameras:
    bpy.data.objects.remove(obj, do_unlink=True)
print(f"removed {len(cameras)} cameras")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.export_scene.gltf(
    filepath=str(OUT_PATH), export_format="GLB", export_lights=True
)
print(f"exported {OUT_PATH}")
