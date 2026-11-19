// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#version 460 core

layout(location = 0) in vec2 frag_tex_coord;
layout(location = 0) out vec4 color;
layout(binding = 0) uniform sampler2D color_texture;

// Catmull-Rom (B = 0, C = 0.5) cubic weights for fractional offset t in [0, 1).
vec4 cubic(float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    return vec4(
        -0.5 * t3 + t2 - 0.5 * t,
         1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
         0.5 * t3 - 0.5 * t2
    );
}

vec4 sampleRow(sampler2D samp, vec4 x_uv, float y_uv, vec4 wx) {
    return texture(samp, vec2(x_uv.x, y_uv)) * wx.x +
           texture(samp, vec2(x_uv.y, y_uv)) * wx.y +
           texture(samp, vec2(x_uv.z, y_uv)) * wx.z +
           texture(samp, vec2(x_uv.w, y_uv)) * wx.w;
}

vec4 textureBicubic(sampler2D samp, vec2 uv) {
    vec2 tex_size = vec2(textureSize(samp, 0));
    vec2 inv_size = 1.0 / tex_size;
    vec2 min_uv = 0.5 * inv_size;
    vec2 max_uv = (tex_size - 0.5) * inv_size;

    vec2 coord = uv * tex_size - 0.5;
    vec2 base = floor(coord);
    vec2 f = coord - base;

    vec4 wx = cubic(f.x);
    vec4 wy = cubic(f.y);

    vec4 x_uv = clamp((base.x + vec4(-0.5, 0.5, 1.5, 2.5)) * inv_size.x, min_uv.x, max_uv.x);
    vec4 y_uv = clamp((base.y + vec4(-0.5, 0.5, 1.5, 2.5)) * inv_size.y, min_uv.y, max_uv.y);

    return sampleRow(samp, x_uv, y_uv.x, wx) * wy.x +
           sampleRow(samp, x_uv, y_uv.y, wx) * wy.y +
           sampleRow(samp, x_uv, y_uv.z, wx) * wy.z +
           sampleRow(samp, x_uv, y_uv.w, wx) * wy.w;
}

void main() {
    color = textureBicubic(color_texture, frag_tex_coord);
}
