// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-smoke: the pixel program after native_mesh_smoke.mesh.

#version 460

layout(location = 0) in vec4 color;
layout(location = 0) out vec4 frag_color;

void main() {
    frag_color = color;
}
