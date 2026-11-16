// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cmath>

#include <catch2/catch_test_macros.hpp>

#include "video_core/ocarina_upscale.h"
#include "video_core/renderer_metal/metal_work_policy.h"

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
