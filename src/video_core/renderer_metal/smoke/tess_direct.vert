// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-smoke: the vertex program of a pipeline without a control program: an output the
// evaluation program does not read sits between those it does.

#version 450

layout(location = 0) in vec2 position;
layout(location = 0) out vec4 color;
layout(location = 1) out vec4 unused;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    color = vec4(0.0, 1.0, 0.0, 1.0);
    unused = vec4(1.0, 2.0, 3.0, 4.0);
}
