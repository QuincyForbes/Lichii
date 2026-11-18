// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-smoke: a pixel program writing one integer colour to target 0.

#version 460

layout(location = 0) out uvec4 color;

void main() {
    color = uvec4(0xA5u, 0x3Cu, 0x0Fu, 0xF0u);
}
