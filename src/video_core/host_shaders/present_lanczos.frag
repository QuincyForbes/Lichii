// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// https://en.wikipedia.org/wiki/Lanczos_resampling

#version 460 core

layout(location = 0) in vec2 frag_tex_coord;
layout(location = 0) out vec4 color;
layout(binding = 0) uniform sampler2D color_texture;

const float PI = 3.14159265358979323846;

float sinc(float x) {
    float px = PI * x;
    return abs(x) < 1e-5 ? 1.0 : sin(px) / px;
}

float lanczos(float x, float a) {
    if (abs(x) >= a) {
        return 0.0;
    }
    return sinc(x) * sinc(x / a);
}

vec4 textureLanczos(sampler2D textureSampler, vec2 p) {
    const float a = 3.0;
    vec2 res = vec2(textureSize(textureSampler, 0));
    vec2 inv_res = 1.0 / res;
    vec2 min_uv = 0.5 * inv_res;
    vec2 max_uv = (res - 0.5) * inv_res;

    vec2 coord = p * res - 0.5;
    vec2 center = floor(coord);

    float wx[6];
    float wy[6];
    float x_uv[6];
    float y_uv[6];
    for (int i = 0; i < 6; ++i) {
        float offset = float(i - 2);
        wx[i] = lanczos(coord.x - (center.x + offset), a);
        wy[i] = lanczos(coord.y - (center.y + offset), a);
        x_uv[i] = clamp((center.x + offset + 0.5) * inv_res.x, min_uv.x, max_uv.x);
        y_uv[i] = clamp((center.y + offset + 0.5) * inv_res.y, min_uv.y, max_uv.y);
    }

    vec4 c_sum = vec4(0.0);
    float w_sum = 0.0;
    for (int iy = 0; iy < 6; ++iy) {
        for (int ix = 0; ix < 6; ++ix) {
            float w = wx[ix] * wy[iy];
            c_sum += w * texture(textureSampler, vec2(x_uv[ix], y_uv[iy]));
            w_sum += w;
        }
    }
    return c_sum / max(w_sum, 1e-6);
}

void main() {
    color = textureLanczos(color_texture, frag_tex_coord);
}
