// Shared octahedral normal codec + snorm16 packing (`bevy_render::utils`
// parity), so every kernel that reads or writes the packed-attribute pools
// (`vertex_normals` / `base_packed` / the deform pool) round-trips normals
// identically. Pure functions, no bindings.
module octahedral;

// For decoding normals or unit direction vectors from octahedral coordinates.
// Input is [-1, 1].
public float3 octahedral_decode_signed(float2 v) {
    var n = float3(v.x, v.y, 1.0 - abs(v.x) - abs(v.y));
    let t = saturate(-n.z);
    n.x += n.x >= 0.0 ? -t : t;
    n.y += n.y >= 0.0 ? -t : t;
    return normalize(n);
}

// https://jcgt.org/published/0003/02/01/paper.pdf
// For encoding normals or unit direction vectors as octahedral coordinates.
// Returns [0, 1]; the pools store `encode * 2 - 1` (see the callers).
public float2 octahedral_encode(float3 v) {
    let n = v / (abs(v.x) + abs(v.y) + abs(v.z));
    if (n.z >= 0.0) {
        return n.xy * 0.5 + 0.5;
    }
    let octahedral_wrap = float2(
        (1.0 - abs(n.y)) * (n.x > 0.0 ? 1.0 : -1.0),
        (1.0 - abs(n.x)) * (n.y > 0.0 ? 1.0 : -1.0));
    return octahedral_wrap * 0.5 + 0.5;
}

// WGSL `unpack2x16snorm`: two s16s, each clamp(v/32767, -1, 1).
public float2 unpack_snorm2x16(uint v) {
    let iv = int2(int(v << 16) >> 16, int(v) >> 16);
    return clamp(float2(iv) / 32767.0, -1.0, 1.0);
}

// WGSL `pack2x16snorm`: round(clamp(e, -1, 1) * 32767) per component.
public uint pack_snorm2x16(float2 v) {
    let c = clamp(v, -1.0, 1.0) * 32767.0;
    let x = uint(int(round(c.x))) & 0xffff;
    let y = uint(int(round(c.y))) & 0xffff;
    return x | (y << 16);
}
