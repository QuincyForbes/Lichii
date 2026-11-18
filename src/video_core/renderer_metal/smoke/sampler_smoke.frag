// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-smoke: samples a texture at a point of the pixel's position (RunSamplers).

#version 450

layout(binding = 0, std140) uniform Point {
    vec4 p;
};
layout(binding = 0) uniform sampler2D tex;
layout(location = 0) out vec4 color;

void main() {
    color = texture(tex, gl_FragCoord.xy * p.xy + p.zw);
}
