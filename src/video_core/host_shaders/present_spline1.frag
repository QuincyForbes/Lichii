// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Smooth subtexel bilinear interpolation (see https://iquilezles.org/articles/texture/).

#version 460 core

layout(location = 0) in vec2 frag_tex_coord;
layout(location = 0) out vec4 color;
layout(binding = 0) uniform sampler2D color_texture;

vec4 textureSpline1(sampler2D sam, vec2 uv) {
    vec2 res = vec2(textureSize(sam, 0));
    vec2 p = uv * res - 0.5;
    vec2 i = floor(p);
    vec2 f = p - i;
    f = f * f * (3.0 - 2.0 * f);
    return texture(sam, (i + f + 0.5) / res);
}

void main() {
    color = textureSpline1(color_texture, frag_tex_coord);
}
