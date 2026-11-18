// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-smoke: the pixel program of raster_line.vert, writing its varying.

#version 460

layout(location = 1) in vec4 tint;
layout(location = 0) out vec4 color;

void main() {
    color = tint;
}
