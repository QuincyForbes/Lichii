// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#version 460 core

layout(location = 0) in vec2 frag_tex_coord;
layout(location = 0) out vec4 color;
layout(binding = 0) uniform sampler2D color_texture;

// Area (box) pixel mixing: integrates exact overlap of each output pixel's footprint with source
// texels (sharp bilinear when upscaling, exact box filter when downscaling).
vec4 AreaSampling(sampler2D textureSampler, vec2 texCoords) {
    ivec2 tex_size = textureSize(textureSampler, 0);
    vec2 source_size = vec2(tex_size);
    ivec2 tex_max = max(tex_size - ivec2(1), ivec2(0));

    vec2 range = clamp(fwidth(texCoords) * source_size, vec2(1e-4), vec2(16.0));
    vec2 beg = texCoords * source_size - 0.5 * range;
    vec2 end = beg + range;

    int x0 = int(floor(beg.x));
    int x1 = max(x0, int(ceil(end.x)) - 1);
    int y0 = int(floor(beg.y));
    int y1 = max(y0, int(ceil(end.y)) - 1);

    vec4 sum = vec4(0.0);
    float weight_sum = 0.0;
    for (int y = y0; y <= y1; ++y) {
        float wy = max(0.0, min(end.y, float(y + 1)) - max(beg.y, float(y)));
        int cy = clamp(y, 0, tex_max.y);
        for (int x = x0; x <= x1; ++x) {
            float wx = max(0.0, min(end.x, float(x + 1)) - max(beg.x, float(x)));
            float w = wx * wy;
            sum += w * texelFetch(textureSampler, ivec2(clamp(x, 0, tex_max.x), cy), 0);
            weight_sum += w;
        }
    }

    return sum / max(weight_sum, 1e-6);
}

void main() {
    color = AreaSampling(color_texture, frag_tex_coord);
}
