// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2025 lsfg-vk
// SPDX-License-Identifier: GPL-3.0-or-later

// LSFG frame generation on Metal, as Vulkan's LsfgChain (present/lsfg_chain.cpp and the alpha,
// beta, gamma, delta, generate and mipmap files): the same passes over the same images in the
// same order, with the compute programs the user's Lossless Scaling DLL holds (their SPIR-V, as
// Vulkan loads it, through the SPIR-V -> MSL translator). Metal orders the passes itself: no
// barriers or descriptor sets. Plain C++.

#pragma once

#ifdef HAS_LSFG

#include <array>
#include <map>
#include <vector>

#include "common/common_types.h"
#include "video_core/frame_gen/lossless_dll.h"
#include "video_core/renderer_metal/metal_object.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_shader_translate.h"

namespace Metal {

class Device;

class LsfgChain {
public:
    static constexpr size_t MipLevels = 7;
    static constexpr size_t HistorySlots = 3;
    static constexpr size_t DeltaInstances = 3;
    static constexpr size_t BetaOutputs = 6;
    static constexpr size_t MaxGenerations = 3;
    static constexpr size_t GenerationSlots = MaxGenerations * (MaxGenerations + 1) / 2;

    LsfgChain(const Device& device, Scheduler& scheduler,
              const VideoCore::FrameGen::ShaderModules& shaders, std::array<u32, 2> extent,
              u32 format, f32 flow_scale);
    ~LsfgChain();

    /// Whether every program was built and every image made.
    [[nodiscard]] bool Valid() const noexcept {
        return valid;
    }

    /// Where frame `frame_count`'s picture goes (the chain keeps the last two).
    [[nodiscard]] TextureHandle Input(u64 frame_count) const noexcept {
        return frames[frame_count % 2].texture.Get();
    }

    /// The flow of the last two frames (mipmaps, alpha and beta).
    void DispatchShared(u64 frame_count);

    /// Generated frame `generation` of `generation_count` into `target` (`size`).
    void DispatchGeneration(u64 frame_count, size_t generation_count, size_t generation,
                            TextureHandle target, std::array<u32, 2> size);

    [[nodiscard]] static constexpr size_t GenerationSlot(size_t generation_count,
                                                         size_t generation) {
        return (generation_count - 1) * generation_count / 2 + generation;
    }

private:
    struct Image {
        Object texture;
        std::array<u32, 2> size{};
        [[nodiscard]] TextureHandle H() const noexcept {
            return texture.Get();
        }
    };
    using Pair = std::array<Image, 2>;
    using History = std::array<Pair, HistorySlots>;

    struct Program {
        Object library;
        Object function;
        Object state;
        std::vector<ShaderTranslate::ResourceSlot> slots;
        std::array<u32, 3> group{};
    };

    /// One descriptor of a dispatch, in the shader's binding order.
    struct Bind {
        enum class Kind : u8 { Constants, Sampler, Texture } kind{};
        std::array<u8, 48> constants{};
        void* sampler{};
        TextureHandle texture{};
    };

    struct Alpha {
        Image temp1;
        Image temp2;
        Pair temp3;
        History out;
    };

    struct Gamma {
        std::array<Image, 3> temp1;
        Pair temp2;
        Image out;
    };

    struct Delta {
        std::array<Image, 3> temp1;
        Pair temp2;
        Image out1;
        Image out2;
    };

    Program* GetProgram(u32 shader_id);
    Image MakeImage(std::array<u32, 2> size, u32 format);
    [[nodiscard]] static Bind Texture(const Image* image);
    [[nodiscard]] Bind Texture(const Image& image) const {
        return Texture(&image);
    }
    [[nodiscard]] Bind Constants(f32 timestamp, bool first_iter, bool first_iter_s) const;
    [[nodiscard]] static Bind Sampler(void* sampler);
    void Run(u32 shader_id, std::vector<Bind> binds, std::array<u32, 2> size, u32 tile_shift);
    void GammaDispatch(size_t index, u64 frame_count, size_t slot);
    void DeltaDispatch(size_t index, u64 frame_count, size_t slot);

    const Device& device;
    Scheduler& scheduler;
    const VideoCore::FrameGen::ShaderModules& modules;
    std::map<u32, Program> programs;
    bool valid{true};
    f32 flow_scale{1.0f};

    Object linear;       ///< Clamp to border (transparent black), linear
    Object white_border; ///< Clamp to border (opaque white)
    Object edge;         ///< Clamp to edge
    Image dummy;         ///< Where Vulkan binds no image

    Pair frames;
    std::array<u32, 2> flow_extent{};
    std::array<Image, MipLevels> flow;
    std::array<Alpha, MipLevels> alpha;
    // Beta.
    Pair beta_temp1;
    Pair beta_temp2;
    std::array<Image, BetaOutputs> beta_out;
    std::array<Gamma, MipLevels> gamma;
    std::array<Delta, DeltaInstances> delta;
};

} // namespace Metal

#endif // HAS_LSFG
