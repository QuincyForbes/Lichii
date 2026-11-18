// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-smoke: the evaluation program of an isoline pipeline: the line between the two
// control points, x along it.

#version 450

layout(isolines, equal_spacing) in;
layout(location = 0) in vec4 color_in[];
layout(location = 0) out vec4 color;

void main() {
    gl_Position = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x);
    color = mix(color_in[0], color_in[1], gl_TessCoord.x);
}
