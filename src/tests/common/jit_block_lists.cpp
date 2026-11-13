// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "common/jit_block_lists.h"

namespace {

using Common::JitBlockLists;

std::vector<JitBlockLists::Relocation> RelocationsOf(const JitBlockLists& lists, std::size_t index) {
    std::vector<JitBlockLists::Relocation> result;
    lists.ForEachRelocation(index, [&result](JitBlockLists::Relocation relocation) {
        result.push_back(relocation);
    });
    return result;
}

} // namespace

TEST_CASE("JitBlockLists: a block without links or patches is one word", "[common]") {
    const auto data = JitBlockLists::Build(176, {}, {});
    REQUIRE(data != nullptr);
    const JitBlockLists lists{data.get()};
    REQUIRE(lists.CodeSize() == 176);
    REQUIRE(lists.Targets() == 0);
    REQUIRE(lists.Patches() == 0);
    REQUIRE(lists.Bytes() == 8);
    REQUIRE(lists.FindTarget(0x1000) == 0);
    REQUIRE(!lists.FindPatch(0));
}

TEST_CASE("JitBlockLists: targets, relocations and patches read back", "[common]") {
    const std::array<JitBlockLists::Relocation, 3> first{{{0x10, 0}, {0x14, 1}, {0x7FFF'FFFF, 1}}};
    const std::array<JitBlockLists::Relocation, 1> second{{{0x40, 0}}};
    const std::array<JitBlockLists::Target, 3> targets{{
        {0x0000'0000'8000'1000ULL, first},
        {0x0000'0001'0000'0000ULL, {}},
        {0xFFFF'0000'8000'2000ULL, second},
    }};
    const std::array<JitBlockLists::Patch, 3> patches{{
        {0x20, 0x8000'1000ULL, 3, 0x1234'5678'9ABC'DEF0ULL, true},
        {0x24, 0x8000'1000ULL, 0x7FFF'FFFF, 0x11ULL, false},
        {0x7FFF'FFFF, 0xFFFF'FFFF'FFFF'FFFFULL, 0, 0xFFFF'FFFF'FFFF'FFFFULL, true},
    }};
    const auto data = JitBlockLists::Build(0xFFFF'FFFFULL, targets, patches);
    REQUIRE(data != nullptr);
    const JitBlockLists lists{data.get()};

    REQUIRE(lists.CodeSize() == 0xFFFF'FFFFU);
    REQUIRE(lists.Targets() == 3);
    REQUIRE(lists.Patches() == 3);
    // 1 header + 3 targets + 9 patch words + (3 ends + 4 relocations, rounded up) / 2.
    REQUIRE(lists.Bytes() == (1 + 3 + 9 + 4) * 8);

    for (std::size_t index = 0; index < targets.size(); ++index) {
        REQUIRE(lists.TargetLocation(index) == targets[index].location);
        REQUIRE(lists.FindTarget(targets[index].location) == index);
        const auto relocations = RelocationsOf(lists, index);
        REQUIRE(relocations.size() == targets[index].relocations.size());
        for (std::size_t i = 0; i < relocations.size(); ++i) {
            REQUIRE(relocations[i].code_offset == targets[index].relocations[i].code_offset);
            REQUIRE(relocations[i].type == targets[index].relocations[i].type);
        }
    }
    REQUIRE(lists.FindTarget(0) == lists.Targets());
    REQUIRE(lists.FindTarget(0x0000'0000'8000'1004ULL) == lists.Targets());
    REQUIRE(lists.FindTarget(0xFFFF'FFFF'FFFF'FFFFULL) == lists.Targets());

    for (const JitBlockLists::Patch& patch : patches) {
        const auto found = lists.FindPatch(patch.code_offset);
        REQUIRE(found.has_value());
        REQUIRE(found->code_offset == patch.code_offset);
        REQUIRE(found->marker_location == patch.marker_location);
        REQUIRE(found->marker_offset == patch.marker_offset);
        REQUIRE(found->call_pc == patch.call_pc);
        REQUIRE(found->recompile == patch.recompile);
    }
    REQUIRE(!lists.FindPatch(0));
    REQUIRE(!lists.FindPatch(0x22));
    REQUIRE(!lists.FindPatch(0x28));
    // An offset beyond 32 bits is not the patch its low half names.
    REQUIRE(!lists.FindPatch(0x1'0000'0020ULL));
}

TEST_CASE("JitBlockLists: many blocks of every shape read back", "[common]") {
    for (u32 target_count = 0; target_count < 6; ++target_count) {
        for (u32 patch_count = 0; patch_count < 5; ++patch_count) {
            std::vector<std::vector<JitBlockLists::Relocation>> relocations(target_count);
            std::vector<JitBlockLists::Target> targets;
            u32 offset = 0;
            std::size_t relocation_total = 0;
            for (u32 t = 0; t < target_count; ++t) {
                for (u32 r = 0; r < (t * 7 + patch_count) % 4; ++r) {
                    relocations[t].push_back({offset += 4, (t + r) & 1});
                }
                relocation_total += relocations[t].size();
                targets.push_back({0x7100'0000'00ULL + t * 0x1000ULL, relocations[t]});
            }
            std::vector<JitBlockLists::Patch> patches;
            for (u32 p = 0; p < patch_count; ++p) {
                patches.push_back({offset += 8, 0x7100'0000'00ULL + p, p, 0xABC0ULL + p, (p & 1) != 0});
            }
            const auto data = JitBlockLists::Build(offset + 4, targets, patches);
            REQUIRE(data != nullptr);
            const JitBlockLists lists{data.get()};
            REQUIRE(lists.CodeSize() == offset + 4);
            REQUIRE(lists.Targets() == target_count);
            REQUIRE(lists.Patches() == patch_count);
            REQUIRE(lists.Bytes() ==
                    (1 + target_count + patch_count * 3 + (target_count + relocation_total + 1) / 2) * 8);
            for (u32 t = 0; t < target_count; ++t) {
                REQUIRE(lists.FindTarget(targets[t].location) == t);
                const auto read = RelocationsOf(lists, t);
                REQUIRE(read.size() == relocations[t].size());
                for (std::size_t i = 0; i < read.size(); ++i) {
                    REQUIRE(read[i].code_offset == relocations[t][i].code_offset);
                    REQUIRE(read[i].type == relocations[t][i].type);
                }
            }
            for (const JitBlockLists::Patch& patch : patches) {
                const auto found = lists.FindPatch(patch.code_offset);
                REQUIRE(found.has_value());
                REQUIRE(found->marker_location == patch.marker_location);
                REQUIRE(found->marker_offset == patch.marker_offset);
                REQUIRE(found->call_pc == patch.call_pc);
                REQUIRE(found->recompile == patch.recompile);
            }
        }
    }
}

TEST_CASE("JitBlockLists: what does not fit is refused", "[common]") {
    const std::array<JitBlockLists::Relocation, 1> far{{{0x8000'0000U, 0}}};
    const std::array<JitBlockLists::Relocation, 1> odd_type{{{0x10, 2}}};
    const std::array<JitBlockLists::Target, 1> far_target{{{0x1000, far}}};
    const std::array<JitBlockLists::Target, 1> odd_target{{{0x1000, odd_type}}};
    const std::array<JitBlockLists::Patch, 1> far_patch{{{0x8000'0000U, 0, 0, 0, false}}};
    const std::array<JitBlockLists::Patch, 1> far_marker{{{0x10, 0, 0x8000'0000U, 0, false}}};

    REQUIRE(JitBlockLists::Build(16, far_target, {}) == nullptr);
    REQUIRE(JitBlockLists::Build(16, odd_target, {}) == nullptr);
    REQUIRE(JitBlockLists::Build(16, {}, far_patch) == nullptr);
    REQUIRE(JitBlockLists::Build(16, {}, far_marker) == nullptr);
    REQUIRE(JitBlockLists::Build(0x1'0000'0000ULL, {}, {}) == nullptr);

    const std::vector<JitBlockLists::Target> many_targets(65536, JitBlockLists::Target{0x1000, {}});
    REQUIRE(JitBlockLists::Build(16, many_targets, {}) == nullptr);
    const std::vector<JitBlockLists::Patch> many_patches(65536, JitBlockLists::Patch{0x10, 0, 0, 0, false});
    REQUIRE(JitBlockLists::Build(16, {}, many_patches) == nullptr);
}
