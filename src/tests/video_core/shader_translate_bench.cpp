// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "common/thread.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/shader_environment.h"

namespace {

struct BenchResult {
    size_t records{};
    size_t programs{};
    size_t failed{};
    size_t spirv_words{};
    u64 spirv_hash{0xcbf29ce484222325ULL};
    double read_ms{};
    double cfg_ms{};
    double translate_ms{};
    double emit_ms{};
};

double Since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}

// A desktop device: no lowering passes, as on the PCs the cache files come from.
Shader::HostTranslateInfo DesktopHost() {
    Shader::HostTranslateInfo host;
    host.min_ssbo_alignment = 16;
    host.support_float64 = true;
    host.support_float16 = true;
    host.support_int64 = true;
    host.support_snorm_render_buffer = true;
    host.support_viewport_index_layer = true;
    host.support_viewport_mask = true;
    host.support_geometry_shader_passthrough = true;
    host.support_conditional_barrier = true;
    host.support_byte_addressed_r16 = true;
    host.ApplyDescriptorLimitPolicy();
    return host;
}

Shader::Profile DesktopProfile() {
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.unified_descriptor_binding = true;
    profile.support_descriptor_aliasing = true;
    profile.support_int8 = true;
    profile.support_int16 = true;
    profile.support_int64 = true;
    profile.support_float_controls = true;
    profile.support_explicit_workgroup_layout = true;
    profile.support_vote = true;
    profile.support_viewport_index_layer_non_geometry = true;
    profile.support_viewport_mask = true;
    profile.support_typeless_image_loads = true;
    profile.support_demote_to_helper_invocation = true;
    profile.support_int64_atomics = true;
    profile.support_derivative_control = true;
    profile.support_geometry_shader_passthrough = true;
    profile.support_native_ndc = true;
    profile.support_scaled_attributes = true;
    profile.support_multi_viewport = true;
    profile.min_ssbo_alignment = 16;
    return profile;
}

// Reads the records as VideoCommon::LoadPipelines does, without its deletion of a file it cannot
// read, and translates every `stride`th one stage by stage.
BenchResult Run(const char* path, size_t stride, size_t rounds) {
    BenchResult result;
    const Shader::HostTranslateInfo host{DesktopHost()};
    const Shader::Profile profile{DesktopProfile()};
    Vulkan::ShaderPools pools;
    for (size_t round = 0; round < rounds; ++round) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return result;
        }
        const auto end{file.tellg()};
        file.seekg(12, std::ios::beg); // magic number and cache version
        size_t record{};
        while (file.good() && file.tellg() < end) {
            const auto read_start{std::chrono::steady_clock::now()};
            u32 num_envs{};
            file.read(reinterpret_cast<char*>(&num_envs), sizeof(num_envs));
            if (!file.good() || num_envs == 0 || num_envs > 6) {
                break;
            }
            std::vector<VideoCommon::FileEnvironment> envs(num_envs);
            for (auto& env : envs) {
                env.Deserialize(file);
            }
            const bool compute{envs.front().ShaderStage() == Shader::Stage::Compute};
            file.seekg(compute ? sizeof(Vulkan::ComputePipelineCacheKey)
                               : sizeof(Vulkan::GraphicsPipelineCacheKey),
                       std::ios::cur);
            result.read_ms += Since(read_start);
            if (record++ % stride != 0) {
                continue;
            }
            ++result.records;
            pools.ReleaseContents();
            Shader::Backend::Bindings bindings;
            for (auto& env : envs) {
                ++result.programs;
                try {
                    const bool vertex_a{env.ShaderStage() == Shader::Stage::VertexA};
                    const u32 start{compute ? env.StartAddress()
                                            : static_cast<u32>(env.StartAddress() +
                                                               sizeof(Shader::ProgramHeader))};
                    auto mark{std::chrono::steady_clock::now()};
                    Shader::Maxwell::Flow::CFG cfg{env, pools.flow_block, start, vertex_a};
                    result.cfg_ms += Since(mark);
                    mark = std::chrono::steady_clock::now();
                    auto program{
                        Shader::Maxwell::TranslateProgram(pools.inst, pools.block, env, cfg, host)};
                    result.translate_ms += Since(mark);
                    if (vertex_a) {
                        continue;
                    }
                    mark = std::chrono::steady_clock::now();
                    const auto code{
                        Shader::Backend::SPIRV::EmitSPIRV(profile, {}, program, bindings)};
                    result.emit_ms += Since(mark);
                    result.spirv_words += code.size();
                    for (const u32 word : code) {
                        result.spirv_hash = (result.spirv_hash ^ word) * 0x100000001b3ULL;
                    }
                } catch (const std::exception&) {
                    ++result.failed;
                }
            }
        }
    }
    return result;
}

} // Anonymous namespace

// EDEN_NX2_SHADER_BENCH_FILE=<pipeline cache file> (a copy of
// shader/<title>/vulkan-frizti.fork-v<n>.bin): CPU time of the stages of shader translation over
// its records, and a hash of the SPIR-V they give. EDEN_NX2_SHADER_BENCH_STRIDE (default 8) takes
// every nth record, EDEN_NX2_SHADER_BENCH_ROUNDS (default 1) repeats the file. The work runs on a
// thread named ShaderBench, so EDEN_NX2_SAMPLE_PROFILE=1000 EDEN_NX2_SAMPLE_THREADS=ShaderBench
// samples it.
TEST_CASE("Shader translation cost on a pipeline cache file", "[.][shader_translate_bench]") {
    const char* const path{std::getenv("EDEN_NX2_SHADER_BENCH_FILE")};
    if (path == nullptr) {
        WARN("EDEN_NX2_SHADER_BENCH_FILE is not set");
        return;
    }
    const auto number = [](const char* variable, size_t fallback) {
        const char* const value{std::getenv(variable)};
        const size_t parsed{value != nullptr
                                ? static_cast<size_t>(std::strtoull(value, nullptr, 10))
                                : size_t{}};
        return parsed != 0 ? parsed : fallback;
    };
    const size_t stride{number("EDEN_NX2_SHADER_BENCH_STRIDE", 8)};
    const size_t rounds{number("EDEN_NX2_SHADER_BENCH_ROUNDS", 1)};
    BenchResult result;
    std::thread worker{[&] {
        Common::SetCurrentThreadName("ShaderBench");
        result = Run(path, stride, rounds);
    }};
    worker.join();
    REQUIRE(result.records != 0);
    WARN(result.records << " records, " << result.programs << " programs (" << result.failed
                        << " failed), " << result.spirv_words << " SPIR-V words, hash "
                        << result.spirv_hash << ": read " << result.read_ms
                        << " ms, control flow " << result.cfg_ms << " ms, translate "
                        << result.translate_ms << " ms, emit " << result.emit_ms << " ms");
}
