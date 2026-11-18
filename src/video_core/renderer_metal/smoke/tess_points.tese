// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-smoke: the evaluation program of a triangle pipeline in point mode: its points,
// four pixels wide.

#version 450

layout(triangles, equal_spacing, ccw, point_mode) in;
layout(location = 0) in vec4 color_in[];
layout(location = 0) out vec4 color;

void main() {
    gl_Position = gl_TessCoord.x * gl_in[0].gl_Position + gl_TessCoord.y * gl_in[1].gl_Position +
                  gl_TessCoord.z * gl_in[2].gl_Position;
    gl_PointSize = 4.0;
    color = color_in[0];
}
