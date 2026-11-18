// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-smoke: a vertex program with sixteen uniform buffers (RunSpilledBuffers): the
// first eight sum into red, the last eight into green.

#version 450

layout(binding = 0, std140) uniform Block0 { vec4 v0; };
layout(binding = 1, std140) uniform Block1 { vec4 v1; };
layout(binding = 2, std140) uniform Block2 { vec4 v2; };
layout(binding = 3, std140) uniform Block3 { vec4 v3; };
layout(binding = 4, std140) uniform Block4 { vec4 v4; };
layout(binding = 5, std140) uniform Block5 { vec4 v5; };
layout(binding = 6, std140) uniform Block6 { vec4 v6; };
layout(binding = 7, std140) uniform Block7 { vec4 v7; };
layout(binding = 8, std140) uniform Block8 { vec4 v8; };
layout(binding = 9, std140) uniform Block9 { vec4 v9; };
layout(binding = 10, std140) uniform Block10 { vec4 v10; };
layout(binding = 11, std140) uniform Block11 { vec4 v11; };
layout(binding = 12, std140) uniform Block12 { vec4 v12; };
layout(binding = 13, std140) uniform Block13 { vec4 v13; };
layout(binding = 14, std140) uniform Block14 { vec4 v14; };
layout(binding = 15, std140) uniform Block15 { vec4 v15; };
layout(location = 0) out vec4 color;

void main() {
    const vec2 positions[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    color = vec4(v0.x + v1.x + v2.x + v3.x + v4.x + v5.x + v6.x + v7.x,
                 v8.x + v9.x + v10.x + v11.x + v12.x + v13.x + v14.x + v15.x, 0.0, 1.0);
}
