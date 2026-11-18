// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-smoke: a pixel program writing one colour to target 0 (the logic operation's source).

#version 460

layout(location = 0) out vec4 color;

void main() {
    color = vec4(0.2, 0.4, 0.6, 0.8);
}
