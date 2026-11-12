// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

namespace Vulkan::MetalGpuTime {

/// Apple: adds the GPU time of every Metal command buffer the process commits (MoltenVK's,
/// MetalFX's) to Common::PerfReport::GpuBusyNs, from MTLCommandBuffer GPUStartTime/GPUEndTime.
/// Vulkan timestamps on MoltenVK would add a Metal pass per query. Once per process;
/// EDEN_NX2_METAL_GPU_TIME=0 leaves Metal alone.
void Start();

} // namespace Vulkan::MetalGpuTime
