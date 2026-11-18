// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-smoke: a vertex program with one attribute and one varying, for the line and
// triangle expansion (metal_raster_expand.h).

#version 460

layout(location = 0) in vec2 position;
layout(location = 1) out vec4 tint;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    tint = vec4(0.0, 1.0, 0.0, 1.0);
}
