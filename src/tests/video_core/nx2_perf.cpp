// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "video_core/nx2_perf.h"

namespace {

using namespace VideoCommon::Nx2Perf;

constexpr size_t Index(Opt opt) {
    return static_cast<size_t>(opt);
}

} // namespace

TEST_CASE("Nx2Perf default mask keeps every switch on", "[nx2_perf]") {
    REQUIRE(ParseOffMask(false, "", "") == u64{0});
}

TEST_CASE("Nx2Perf bank_once_per_frame is on by default and off under SAFE", "[nx2_perf]") {
    REQUIRE(opt_names[Index(Opt::BankOncePerFrame)] == "bank_once_per_frame");
    REQUIRE(!IsOff(ParseOffMask(false, "", ""), Index(Opt::BankOncePerFrame)));
    REQUIRE(IsOff(ParseOffMask(true, "", ""), Index(Opt::BankOncePerFrame)));
    REQUIRE(ParseOffMask(false, "", "bank_once_per_frame") == Bit(Opt::BankOncePerFrame));
}

TEST_CASE("Nx2Perf OFF=bind_memo flips only its bit", "[nx2_perf]") {
    const auto mask = ParseOffMask(false, "", "bind_memo");
    REQUIRE(mask == Bit(Opt::BindMemo));
    REQUIRE(!IsOff(mask, Index(Opt::IndexRebind)));
}

TEST_CASE("Nx2Perf SAFE is the default-off set", "[nx2_perf]") {
    const auto mask = ParseOffMask(true, "", "");
    REQUIRE(mask == DefaultOffMask());
    for (size_t i = Index(Opt::Bounds); i <= Index(Opt::IndexRebind); ++i) {
        REQUIRE(!IsOff(mask, i));
    }
    for (size_t i = Index(Opt::DownloadBatch); i <= Index(Opt::BindMemo); ++i) {
        REQUIRE(IsOff(mask, i));
    }
}

TEST_CASE("Nx2Perf lists of names", "[nx2_perf]") {
    REQUIRE(ParseOffMask(false, "", "bounds,draw_preamble") ==
            (Bit(Opt::Bounds) | Bit(Opt::DrawPreamble)));
    REQUIRE(ParseOffMask(true, "front_run,download_batch", "") ==
            (DefaultOffMask() & ~(Bit(Opt::FrontRun) | Bit(Opt::DownloadBatch))));
}

TEST_CASE("geometry_consistency: named, and off at start (EDEN_NX2_PERF_ON turns it on)",
          "[nx2_perf]") {
    REQUIRE(opt_names[Index(Opt::GeometryConsistency)] == "geometry_consistency");
    REQUIRE(!IsOff(DefaultOffMask(), Index(Opt::GeometryConsistency)));
    REQUIRE(ParseOffMask(false, "", "geometry_consistency") == Bit(Opt::GeometryConsistency));
    REQUIRE(!IsOff(ParseOffMask(false, "geometry_consistency", "", Bit(Opt::GeometryConsistency)),
                   Index(Opt::GeometryConsistency)));
}
