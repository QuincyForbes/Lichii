// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-smoke: a rectangle over the left half of the target (x -1..0, y -1..1) as two
// triangles, for the raster state the translated programs are patched for (metal_raster_patch.h).

#version 460

void main() {
    const vec2 corners[6] = vec2[6](vec2(-1.0, -1.0), vec2(0.0, -1.0), vec2(0.0, 1.0),
                                    vec2(-1.0, -1.0), vec2(0.0, 1.0), vec2(-1.0, 1.0));
    gl_Position = vec4(corners[gl_VertexIndex], 0.0, 1.0);
}
