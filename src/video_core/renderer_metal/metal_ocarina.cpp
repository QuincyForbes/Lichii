// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <span>
#include <string>
#include <utility>

#include "common/logging.h"
#include "video_core/host_shaders/ocarina_bilinear_upscale_comp_spv.h"
#include "video_core/renderer_metal/metal_ocarina.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {

namespace {
// The passes Vulkan's TryOcarinaBilinear takes: 2560x1440 (menus, gameplay), the gameplay one
// when the scene is under half the target (above 0.4 of it, and 0.4 and below: oot_resolution
// 40 % and 33 %) and the 0xe40f5efb5480e692 one.
constexpr std::array<u64, 5> BilinearShaders{0x4c1ae2fe58f30a97ULL, 0x9302bc0d6d1d9bc1ULL,
                                             0x03bf0ffc7d2d0bedULL, 0x31c0acb7c5e87c89ULL,
                                             0xe40f5efb5480e692ULL};
constexpr u32 GroupSize = 8;
} // Anonymous namespace

OcarinaUpscale::OcarinaUpscale(const Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_} {}

OcarinaUpscale::~OcarinaUpscale() = default;

bool OcarinaUpscale::BilinearEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_OOT_BILINEAR_UPSCALE");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

bool OcarinaUpscale::IsBilinearPass(u64 shader_hash, const Shader::Info& info) {
    // As Vulkan's ComputePipeline::CanReplaceOcarinaUpscale.
    return std::ranges::find(BilinearShaders, shader_hash) != BilinearShaders.end() &&
           !info.stores_global_memory && info.texture_buffer_descriptors.empty() &&
           info.image_buffer_descriptors.empty() && info.texture_descriptors.size() == 1 &&
           info.texture_descriptors.front().count == 1 && info.image_descriptors.size() == 1 &&
           info.image_descriptors.front().count == 1 && info.image_descriptors.front().is_written &&
           !info.image_descriptors.front().is_read &&
           !info.image_descriptors.front().is_byte_addressed &&
           std::ranges::none_of(info.storage_buffers_descriptors,
                                &Shader::StorageBufferDescriptor::is_written);
}

bool OcarinaUpscale::Prepare() {
    if (state || failed) {
        return !failed;
    }
    failed = true;
    const ShaderTranslate::Result msl = ShaderTranslate::SpirvToMsl(
        std::span<const u32>(OCARINA_BILINEAR_UPSCALE_COMP_SPV), ShaderTranslate::Stage::Compute,
        {});
    std::string error = msl.error;
    if (!msl.source.empty()) {
        library = CreateLibrary(device, msl.source, error);
        function = library ? CreateFunction(library.Get(), msl.entry_point) : Object{};
        state = function ? CreateComputePipelineState(device, function.Get(), error) : Object{};
    }
    sampler = CreateLinearClampSampler(device);
    if (!state || !sampler) {
        LOG_WARNING(Render_Metal, "Metal: Ocarina bilinear upscale unavailable: {}", error);
        state = {};
        return false;
    }
    slots = msl.slots;
    failed = false;
    return true;
}

bool OcarinaUpscale::Bilinear(TextureHandle color, std::array<u32, 2> color_size,
                              std::array<u32, 2> extent, TextureHandle target,
                              std::array<u32, 2> target_size) {
    if (color == nullptr || target == nullptr || color_size[0] == 0 || color_size[1] == 0 ||
        !Prepare()) {
        return false;
    }
    // ocarina_bilinear_upscale.comp: scene (binding 0), guest_color (binding 1), scene_scale.
    const std::array<f32, 2> scene_scale{
        static_cast<f32>(extent[0]) / static_cast<f32>(color_size[0]),
        static_cast<f32>(extent[1]) / static_cast<f32>(color_size[1]),
    };
    scheduler.BeginCompute(state.Get());
    for (const ShaderTranslate::ResourceSlot& slot : slots) {
        switch (slot.kind) {
        case ShaderTranslate::ResourceKind::Buffer:
            if (slot.set == ShaderTranslate::PushConstantSet) {
                scheduler.BindBytes(ShaderStage::Compute, slot.index, scene_scale.data(),
                                    static_cast<u32>(sizeof(scene_scale)));
            }
            break;
        case ShaderTranslate::ResourceKind::Texture:
            scheduler.BindTexture(ShaderStage::Compute, slot.index,
                                  slot.binding == 0 ? color : target);
            break;
        case ShaderTranslate::ResourceKind::Sampler:
            scheduler.BindSampler(ShaderStage::Compute, slot.index, sampler.Get());
            break;
        }
    }
    scheduler.Dispatch({(target_size[0] + GroupSize - 1) / GroupSize,
                        (target_size[1] + GroupSize - 1) / GroupSize, 1},
                       {GroupSize, GroupSize, 1});
    ++passes;
    return true;
}

u64 OcarinaUpscale::TakePasses() noexcept {
    return std::exchange(passes, 0);
}

} // namespace Metal
