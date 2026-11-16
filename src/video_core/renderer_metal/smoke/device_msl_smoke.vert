// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-smoke: a vertex program that reads the base instance, as the recompiler's do for
// the instance index. Instance 0 is the target's left half in red, instance 1 its right half in
// green; with the base instance left in, both are off the target.

#version 460

layout(location = 0) out vec4 color;

void main() {
    const int instance = gl_InstanceIndex - gl_BaseInstance;
    const vec2 corner = vec2(float(gl_VertexIndex & 1), float(gl_VertexIndex >> 1));
    gl_Position = vec4(float(instance) - 1.0 + corner.x, corner.y * 2.0 - 1.0, 0.0, 1.0);
    color = instance == 1 ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}
