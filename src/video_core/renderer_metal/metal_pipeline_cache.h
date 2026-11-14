// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Guest programs -> shader recompiler IR -> SPIR-V -> MSL (metal_shader_translate), cached per
// pipeline key. Plain C++; Metal objects are made by metal_resources.h.

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "common/container/unordered_map.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/runtime_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_object.h"
#include "video_core/renderer_metal/metal_shader_translate.h"
#include "video_core/shader_cache.h"

namespace Metal {

class Device;

struct ShaderPools {
    void ReleaseContents() {
        flow_block.ReleaseContents();
        block.ReleaseContents();
        inst.ReleaseContents();
    }

    Shader::ObjectPool<Shader::IR::Inst> inst{8192};
    Shader::ObjectPool<Shader::IR::Block> block{32};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_block{32};
};

struct GraphicsPipelineKey {
    std::array<u64, 6> unique_hashes{};
    /// Two bits per generic attribute (FixedPipelineState::DynamicAttributeTypes).
    u64 attribute_types{};
    u32 topology{}; ///< Maxwell PrimitiveTopology of the draw
    u32 early_z{};

    bool operator==(const GraphicsPipelineKey&) const noexcept = default;

    [[nodiscard]] size_t Hash() const noexcept;
};

struct ComputePipelineKey {
    u64 unique_hash{};
    u32 shared_memory_size{};
    std::array<u32, 3> workgroup_size{};

    bool operator==(const ComputePipelineKey&) const noexcept = default;

    [[nodiscard]] size_t Hash() const noexcept;
};

} // namespace Metal

template <>
struct std::hash<Metal::GraphicsPipelineKey> {
    size_t operator()(const Metal::GraphicsPipelineKey& key) const noexcept {
        return key.Hash();
    }
};

template <>
struct std::hash<Metal::ComputePipelineKey> {
    size_t operator()(const Metal::ComputePipelineKey& key) const noexcept {
        return key.Hash();
    }
};

namespace Metal {

/// One translated stage: its MSL, the function, the slots its resources got, and what it uses.
struct StageProgram {
    ShaderTranslate::Stage stage{};
    std::string msl;
    std::string entry_point;
    std::vector<ShaderTranslate::ResourceSlot> slots;
    Shader::Info info;
    /// The MTLLibrary and MTLFunction, made on first use (BuildFunctions).
    Object library;
    Object function;
    bool build_failed{};
};

class GraphicsPipeline {
public:
    explicit GraphicsPipeline(const GraphicsPipelineKey& key, StageProgram vertex,
                              StageProgram fragment);

    [[nodiscard]] const GraphicsPipelineKey& Key() const noexcept {
        return key;
    }

    [[nodiscard]] StageProgram& Vertex() noexcept {
        return vertex;
    }

    [[nodiscard]] StageProgram& Fragment() noexcept {
        return fragment;
    }

    /// Builds the stages' functions; false (logged once) when Metal rejects one.
    bool BuildFunctions(const Device& device);

    /// The MTLRenderPipelineState for these targets, blending and vertex layout (desc's function
    /// fields are filled in here), made on first use; null when Metal refuses it.
    [[nodiscard]] void* RenderState(const Device& device, RenderPipelineDesc& desc);

private:
    GraphicsPipelineKey key;
    StageProgram vertex;
    StageProgram fragment;
    ::Common::unordered_map<u64, Object> render_states;
};

class ComputePipeline {
public:
    explicit ComputePipeline(const ComputePipelineKey& key, StageProgram program);

    [[nodiscard]] StageProgram& Program() noexcept {
        return program;
    }

    [[nodiscard]] const ComputePipelineKey& Key() const noexcept {
        return key;
    }

    /// The MTLComputePipelineState, made on first use; null (logged once) when it fails.
    [[nodiscard]] void* State(const Device& device);

private:
    ComputePipelineKey key;
    StageProgram program;
    Object state;
    bool failed{};
};

class PipelineCache : public VideoCommon::ShaderCache {
public:
    explicit PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory, const Device& device);
    ~PipelineCache();

    /// The pipeline of the current draw, or null when it can't be made (unsupported stages,
    /// translation errors).
    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipeline();

    [[nodiscard]] ComputePipeline* CurrentComputePipeline();

    /// Pipelines made and refused since the last call (the renderer's 10 s report).
    [[nodiscard]] std::array<u64, 3> TakeCounts() noexcept;

private:
    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline();

    std::unique_ptr<ComputePipeline> CreateComputePipeline(const ComputePipelineKey& key,
                                                           const VideoCommon::ShaderInfo* shader);

    [[nodiscard]] std::optional<StageProgram> Translate(Shader::IR::Program& program,
                                                        ShaderTranslate::Stage stage,
                                                        Shader::Backend::Bindings& bindings,
                                                        const Shader::RuntimeInfo& runtime_info);

    const Device& device;

    Shader::Profile profile;
    Shader::HostTranslateInfo host_info;
    ShaderTranslate::Options translate_options;

    ShaderPools pools;
    GraphicsPipelineKey graphics_key{};
    GraphicsPipeline* current_pipeline{};
    ::Common::unordered_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_cache;
    ::Common::unordered_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_cache;

    u64 graphics_made{};
    u64 compute_made{};
    u64 refused{};
};

} // namespace Metal
