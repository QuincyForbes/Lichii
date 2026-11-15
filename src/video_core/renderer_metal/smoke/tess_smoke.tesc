// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-smoke: the control program of a tessellation pipeline (factors 4).

#version 450

layout(vertices = 3) out;
layout(location = 0) in vec4 color_in[];
layout(location = 0) out vec4 color_out[];

void main() {
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    color_out[gl_InvocationID] = color_in[gl_InvocationID];
    if (gl_InvocationID == 0) {
        gl_TessLevelOuter[0] = 4.0;
        gl_TessLevelOuter[1] = 4.0;
        gl_TessLevelOuter[2] = 4.0;
        gl_TessLevelInner[0] = 4.0;
    }
}
