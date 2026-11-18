// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-smoke: a pixel program writing one signed integer colour to target 0 (the bytes
// 0xA5, 0x3C, 0x0F, 0xF0 as two's complement).

#version 460

layout(location = 0) out ivec4 color;

void main() {
    color = ivec4(-91, 60, 15, -16);
}
