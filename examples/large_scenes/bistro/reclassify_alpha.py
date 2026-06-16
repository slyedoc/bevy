# Reclassifies glTF material alpha modes for the ray-traced bistro.
#
# The FBX -> Blender -> glTF export flattens NVIDIA's authored alpha modes:
# 189 of 199 Bistro materials come out `BLEND` (incl. solid concrete/metal/
# wood). bevy_solari forces every non-opaque instance through the any-hit
# `rayQueryProceed` candidate loop (with a base-color alpha fetch per
# candidate) instead of committing in hardware -- so ~95% of the scene pays
# alpha-test traversal it doesn't need. That shows up as inflated RTCore
# traversal stalls in the GPU trace.
#
# This pass sets each material's alpha mode from its actual base-color alpha
# content, because bevy_solari only alpha-tests `AlphaMode::Mask` (a genuine
# cutout); `Opaque` and `Blend` both traverse as opaque:
#   - a real binary cutmask (significant holes AND significant solid area)
#     -> MASK (cutoff 0.5), so the ray tracer keeps testing it
#   - everything else (fully opaque, or an unused/packed alpha channel that is
#     uniformly transparent) -> OPAQUE
# Transmissive materials (glass/liquids) are left untouched -- refraction owns
# them and `prepare_bistro.py` already set them opaque.
#
# Usage:
#   python reclassify_alpha.py <alpha_ref.glb> <target.glb>
# `alpha_ref` is read for decodable base-color images (the PNG GLB); `target`
# (e.g. the KTX2 GLB, whose textures aren't easily decodable) is patched in
# place, matching materials by name. They can be the same file.

import io
import json
import struct
import sys

from PIL import Image as PImage

GLB_MAGIC = 0x46546C67
JSON_TYPE = 0x4E4F534A

# A base-color texture is a genuine binary cutmask when at least this fraction
# of its texels are cut away (alpha < 128) AND at least this fraction remain
# solid (alpha >= 200). The solid floor rejects channels that are uniformly
# transparent (unused/packed alpha), which would vanish if masked at 0.5.
CUTMASK_MIN_HOLE_FRACTION = 0.01
CUTMASK_MIN_SOLID_FRACTION = 0.05
ALPHA_HOLE = 128
ALPHA_SOLID = 200
MASK_CUTOFF = 0.5


def read_glb(path):
    data = open(path, "rb").read()
    magic, _version, _length = struct.unpack_from("<III", data, 0)
    assert magic == GLB_MAGIC, f"{path}: not a GLB"
    json_len, json_type = struct.unpack_from("<II", data, 12)
    assert json_type == JSON_TYPE
    gltf = json.loads(data[20 : 20 + json_len])
    bin_start = 20 + json_len + 8  # skip the BIN chunk header
    return gltf, data, bin_start


def base_color_is_cutmask(gltf, data, bin_start, material):
    """True if the material's base color is a genuine binary alpha cutmask."""
    pbr = material.get("pbrMetallicRoughness", {})
    info = pbr.get("baseColorTexture")
    if info is None:
        return False  # no texture: opacity is the factor's alpha, not a cutmask
    source = gltf["textures"][info["index"]].get("source")
    if source is None:
        return False
    view = gltf["bufferViews"][gltf["images"][source]["bufferView"]]
    off = bin_start + view.get("byteOffset", 0)
    image = PImage.open(io.BytesIO(data[off : off + view["byteLength"]]))
    if image.mode not in ("RGBA", "LA", "PA") and "transparency" not in image.info:
        return False  # no alpha channel at all
    alpha = image.convert("RGBA").getchannel("A")
    histogram = alpha.histogram()
    total = sum(histogram) or 1
    holes = sum(histogram[:ALPHA_HOLE]) / total
    solid = sum(histogram[ALPHA_SOLID:]) / total
    return holes >= CUTMASK_MIN_HOLE_FRACTION and solid >= CUTMASK_MIN_SOLID_FRACTION


def write_glb(path, gltf, data, bin_start):
    rest = data[bin_start - 8 :]  # BIN chunk header + payload, unchanged
    payload = json.dumps(gltf, separators=(",", ":")).encode()
    payload += b" " * (-len(payload) % 4)
    out = bytearray()
    out += struct.pack("<III", GLB_MAGIC, 2, 12 + 8 + len(payload) + len(rest))
    out += struct.pack("<II", len(payload), JSON_TYPE)
    out += payload
    out += rest
    open(path, "wb").write(out)


def has_transmission(material):
    return "KHR_materials_transmission" in material.get("extensions", {})


def main():
    ref_path, target_path = sys.argv[1], sys.argv[2]
    ref, ref_data, ref_bin = read_glb(ref_path)

    # Classify each material (by name) as a cutmask from the decodable ref.
    cutmask_names = set()
    for material in ref.get("materials", []):
        if has_transmission(material):
            continue
        if base_color_is_cutmask(ref, ref_data, ref_bin, material):
            cutmask_names.add(material.get("name"))

    target, target_data, target_bin = read_glb(target_path)
    mask, opaque = 0, 0
    for material in target.get("materials", []):
        if has_transmission(material):
            continue
        if material.get("name") in cutmask_names:
            material["alphaMode"] = "MASK"
            material["alphaCutoff"] = MASK_CUTOFF
            mask += 1
        else:
            material["alphaMode"] = "OPAQUE"
            material.pop("alphaCutoff", None)
            opaque += 1

    write_glb(target_path, target, target_data, target_bin)
    print(f"{target_path}: {mask} MASK (cutout), {opaque} OPAQUE, rest transmissive")


if __name__ == "__main__":
    main()
