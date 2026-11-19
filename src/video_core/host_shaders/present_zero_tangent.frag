// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#version 460 core

layout(location = 0) in vec2 frag_tex_coord;
layout(location = 0) out vec4 color;
layout(binding = 0) uniform sampler2D color_texture;

// Zero-Tangent Cardinal (B = 0, C = 0) cubic Hermite interpolation on snapped texel centers.
vec4 textureZeroTangent(sampler2D samp, vec2 uv) {
    vec2 tex_size = vec2(textureSize(samp, 0));
    vec2 inv_size = 1.0 / tex_size;
    vec2 min_uv = 0.5 * inv_size;
    vec2 max_uv = (tex_size - 0.5) * inv_size;

    vec2 coord = uv * tex_size - 0.5;
    vec2 base = floor(coord);
    vec2 f = coord - base;
    vec2 w = f * f * (3.0 - 2.0 * f);

    vec2 x_uv = clamp((base.x + vec2(0.5, 1.5)) * inv_size.x, min_uv.x, max_uv.x);
    vec2 y_uv = clamp((base.y + vec2(0.5, 1.5)) * inv_size.y, min_uv.y, max_uv.y);

    vec4 row0 = mix(texture(samp, vec2(x_uv.x, y_uv.x)), texture(samp, vec2(x_uv.y, y_uv.x)), w.x);
    vec4 row1 = mix(texture(samp, vec2(x_uv.x, y_uv.y)), texture(samp, vec2(x_uv.y, y_uv.y)), w.x);
    return mix(row0, row1, w.y);
}

void main() {
    color = textureZeroTangent(color_texture, frag_tex_coord);
}
