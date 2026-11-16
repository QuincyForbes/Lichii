// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-smoke: a pixel program with the subgroup operations the recompiler emits (a
// shuffle and a ballot). Each pixel reads its own colour back through its subgroup.

#version 460
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require

layout(location = 0) in vec4 color;
layout(location = 0) out vec4 frag_color;

void main() {
    const vec4 own = subgroupShuffle(color, gl_SubgroupInvocationID);
    const uvec4 live = subgroupBallot(true);
    frag_color = (live.x | live.y) != 0u ? own : vec4(0.0);
}
