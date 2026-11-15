// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <memory>

#include "common/common_types.h"
#include "common/dynamic_library.h"
#include "core/frontend/graphics_context.h"

namespace Vulkan {

std::shared_ptr<Common::DynamicLibrary> OpenLibrary(
    [[maybe_unused]] Core::Frontend::GraphicsContext* context = nullptr);

/// One stage of MoltenVK's shader and pipeline compilation, summed since the device was created.
struct MoltenVKStage {
    u32 count{};
    double total_ms{};
};

/// In the order of MoltenVK's MVKShaderCompilationPerformance.
inline constexpr std::array<const char*, 11> MOLTENVK_STAGE_NAMES{
    "hash",          "SPIR-V to MSL",     "MSL compile",    "MSL load",
    "MSL compress",  "MSL decompress",    "library cached", "function",
    "specialise",    "pipeline compile",  "GLSL",
};

/// False unless the loaded library is a MoltenVK that tracks them (EDEN_NX2_MVK_STATS) and its
/// statistics have the layout read here. `device` is the VkDevice.
bool ReadMoltenVKStages(void* device, std::array<MoltenVKStage, 11>& stages);

} // namespace Vulkan
