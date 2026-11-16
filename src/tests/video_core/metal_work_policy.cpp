// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cmath>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/ocarina_upscale.h"
#include "video_core/renderer_metal/metal_bind_memo.h"
#include "video_core/renderer_metal/metal_work_policy.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"

using namespace Metal::WorkPolicy;

TEST_CASE("MetalWorkPolicy[FlushDraws]", "[video_core]") {
    REQUIRE(ParseFlushDraws(nullptr) == 0);
    REQUIRE(ParseFlushDraws("64") == 64);
    // Vulkan's legacy parser takes 0 and negative values as the smallest budget too.
    REQUIRE(ParseFlushDraws("0") == 1);
    REQUIRE(ParseFlushDraws("-5") == 1);

    DrawFlushCounter off{};
    for (int draw = 0; draw < 1000; ++draw) {
        REQUIRE_FALSE(off.Note());
    }
    DrawFlushCounter every_three{.budget = 3};
    int commits = 0;
    for (int draw = 1; draw <= 10; ++draw) {
        const bool commit = every_three.Note();
        REQUIRE(commit == (draw % 3 == 0));
        commits += commit ? 1 : 0;
    }
    REQUIRE(commits == 3);
}

TEST_CASE("MetalWorkPolicy[NoMesh]", "[video_core]") {
    MeshSkip none{};
    REQUIRE_FALSE(none.Skip([] { return true; }));

    MeshSkip always{.always = true};
    REQUIRE(always.Skip([] { return false; }));

    // The file is looked for on the first mesh draw and every 1024 after it.
    bool exists = true;
    int looked = 0;
    MeshSkip file{.has_file = true};
    const auto probe = [&] {
        ++looked;
        return exists;
    };
    REQUIRE(file.Skip(probe));
    exists = false;
    for (int draw = 1; draw < 1024; ++draw) {
        REQUIRE(file.Skip(probe));
    }
    REQUIRE(looked == 1);
    REQUIRE_FALSE(file.Skip(probe));
    REQUIRE(looked == 2);
}

TEST_CASE("MetalWorkPolicy[PressureEvictions]", "[video_core]") {
    // 600 pipelines: 300 drawn in the last 600 frames, 300 not, at frame 10,000.
    std::vector<PipelineUse> uses;
    for (size_t index = 0; index < 600; ++index) {
        const bool recent = index < 300;
        uses.push_back({.last_use = recent ? 9'900U : static_cast<u64>(index), .used = true,
                        .current = false});
    }
    uses[400].current = true;
    // Too early in the run, or too few pipelines: none.
    REQUIRE(PressureEvictions(uses, 500, false, 256).empty());
    REQUIRE(PressureEvictions(std::span(uses).first(256), 10'000, false, 256).empty());
    // A signal: down to the 256 kept, the stale ones only, least recently used first.
    const auto evicted = PressureEvictions(uses, 10'000, false, 256);
    REQUIRE(evicted.size() == 299); // 600 - 256 = 344, of 299 stale (the current one stays)
    for (const size_t index : evicted) {
        REQUIRE(index >= 300);
        REQUIRE(index != 400);
    }
    // Near the limit: unused for 3600 frames, 512 kept, at most `evict_max`.
    REQUIRE(PressureEvictions(uses, 10'000, true, 256).empty()); // 600 - 512 = 88 < 128
    std::vector<PipelineUse> many(1000, PipelineUse{.last_use = 1, .used = true, .current = false});
    const auto near_limit = PressureEvictions(many, 10'000, true, 200);
    REQUIRE(near_limit.size() == 200);
    // Near the limit boot-built pipelines no draw took stay, up to 2000.
    for (PipelineUse& use : many) {
        use.used = false;
    }
    REQUIRE(PressureEvictions(many, 10'000, true, 200).empty());
    REQUIRE(PressureEvictions(many, 10'000, false, 200).size() == 744);
}

TEST_CASE("MetalBindMemo[KindsAsVulkan]", "[video_core]") {
    // The memo is keyed by binding kind: Metal's must be Vulkan's, element for element.
    Shader::Info info{};
    info.texture_buffer_descriptors.push_back({.count = 2});
    info.image_buffer_descriptors.push_back(
        {.format = Shader::ImageFormat::R32_UINT, .is_written = true, .count = 1});
    info.texture_descriptors.push_back(
        {.type = Shader::TextureType::Color2D, .is_depth = false, .count = 3});
    info.texture_descriptors.push_back(
        {.type = Shader::TextureType::ColorCube, .is_depth = true, .count = 1});
    info.image_descriptors.push_back({.type = Shader::TextureType::Color2D,
                                      .format = Shader::ImageFormat::Typeless,
                                      .is_written = true,
                                      .count = 2,
                                      .is_byte_addressed = true});
    for (const bool compute : {false, true}) {
        std::vector<u32> vulkan;
        Vulkan::AppendBindingKinds(info, compute, vulkan);
        std::vector<u32> metal;
        Metal::BindMemo::AppendKinds(
            info, compute ? Vulkan::TextureCache::BINDING_KIND_COMPUTE : 0U, metal);
        REQUIRE(metal.size() == 9);
        REQUIRE(metal == vulkan);
    }
}

TEST_CASE("MetalWorkPolicy[PresentPacer]", "[video_core]") {
    PresentPacer pacer{};
    // Refresh unknown: at once.
    for (int frame = 0; frame < 20; ++frame) {
        pacer.NoteFrame(33.3e6);
    }
    REQUIRE(pacer.MinimumDurationNs() == 0.0);
    // 30 fps on 120 Hz: four refreshes, presented from 3.5 refreshes after the last.
    pacer.refresh_ns = 1e9 / 120.0;
    REQUIRE(std::abs(pacer.MinimumDurationNs() - 3.5 * 1e9 / 120.0) < 1e3);
    // 60 fps on 60 Hz: one refresh, nothing held.
    PresentPacer every{.refresh_ns = 1e9 / 60.0};
    for (int frame = 0; frame < 20; ++frame) {
        every.NoteFrame(16.7e6);
    }
    REQUIRE(every.MinimumDurationNs() == 0.0);
    // Uneven 30 fps frames (25 and 41.6 ms by turns) still average to four refreshes.
    PresentPacer uneven{.refresh_ns = 1e9 / 120.0};
    for (int frame = 0; frame < 40; ++frame) {
        uneven.NoteFrame(frame % 2 == 0 ? 25e6 : 41.6e6);
    }
    REQUIRE(std::abs(uneven.MinimumDurationNs() - 3.5 * 1e9 / 120.0) < 1e3);
    // A stall restarts the average: nothing held until eight frames came again.
    uneven.NoteFrame(2e9);
    REQUIRE(uneven.MinimumDurationNs() == 0.0);
}

TEST_CASE("MetalWorkPolicy[OcarinaJitter]", "[video_core]") {
    // Halton (2, 3) of frame 0 is the sequence's first point (1/2, 1/3), centred on the pixel.
    const auto first = OcarinaJitter(0, 8);
    REQUIRE(std::abs(first[0] - 0.0f) < 1e-6f);
    REQUIRE(std::abs(first[1] - (1.0f / 3.0f - 0.5f)) < 1e-6f);
    const auto second = OcarinaJitter(1, 8);
    REQUIRE(std::abs(second[0] - (0.25f - 0.5f)) < 1e-6f);
    REQUIRE(std::abs(second[1] - (2.0f / 3.0f - 0.5f)) < 1e-6f);
    // The phase count wraps the sequence, as Vulkan's render_trace_frame % phases.
    const u32 phases = VideoCommon::OcarinaJitterPhases(1280, 1920);
    REQUIRE(phases == 18);
    REQUIRE(OcarinaJitter(5, phases) == OcarinaJitter(5 + phases, phases));
    for (u64 frame = 0; frame < phases; ++frame) {
        const auto offset = OcarinaJitter(frame, phases);
        REQUIRE(std::abs(offset[0]) <= 0.5f);
        REQUIRE(std::abs(offset[1]) <= 0.5f);
    }
}
