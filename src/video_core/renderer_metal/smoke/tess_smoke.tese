// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-smoke: the evaluation program of a tessellation pipeline (the post-tessellation
// vertex function).

#version 450

layout(triangles, equal_spacing, ccw) in;
layout(location = 0) in vec4 color_in[];
layout(location = 0) out vec4 color;

void main() {
    gl_Position = gl_TessCoord.x * gl_in[0].gl_Position + gl_TessCoord.y * gl_in[1].gl_Position +
                  gl_TessCoord.z * gl_in[2].gl_Position;
    color = gl_TessCoord.x * color_in[0] + gl_TessCoord.y * color_in[1] +
            gl_TessCoord.z * color_in[2];
}
