// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-smoke: the vertex program of a tessellation pipeline (run as a stage-in kernel).

#version 450

layout(location = 0) in vec2 position;
layout(location = 0) out vec4 color;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    color = vec4(0.0, 1.0, 0.0, 1.0);
}
