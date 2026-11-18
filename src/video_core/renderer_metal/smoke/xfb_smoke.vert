// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-smoke: a vertex program of a pipeline that captures (run as a stage-in kernel; its
// outputs are named as the shader recompiler names them).

#version 450

layout(location = 0) in vec3 position;
layout(location = 0) out vec4 out_attr0;
layout(location = 3) out vec4 out_attr3;

void main() {
    gl_Position = vec4(position, 1.0);
    out_attr0 = vec4(position.xy, 7.0, 8.0);
    out_attr3 = vec4(float(gl_VertexIndex), float(gl_InstanceIndex), 2.0, 3.0);
}
