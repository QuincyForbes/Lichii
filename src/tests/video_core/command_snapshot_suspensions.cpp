// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "core/core.h"
#include "tests/core/system_test_context.h"
#include "video_core/control/channel_state.h"
#include "video_core/dma_pusher.h"
#include "video_core/memory_manager.h"

TEST_CASE("Entries parsed without a pending acquire keep their submit-time snapshot",
          "[video_core][dma_pusher]") {
    ScopedSystemTestContext context;
    Core::System system;
    Tegra::MemoryManager memory_manager{system};
    Tegra::Control::ChannelState channel_state{0};
    Tegra::DmaPusher dma_pusher{system, memory_manager, channel_state};

    const u64 snapshot_suspensions = dma_pusher.suspensions.load();
    for (int entry = 0; entry < 8; ++entry) {
        REQUIRE_FALSE(dma_pusher.puller.HasPendingAcquire());
        REQUIRE(dma_pusher.puller.ResumeAcquire(dma_pusher));
    }
    REQUIRE(dma_pusher.suspensions.load() == snapshot_suspensions);
}

TEST_CASE("RefreshSnapshot skips already-parsed entries from first_entry onward",
          "[video_core][dma_pusher]") {
    ScopedSystemTestContext context;
    Core::System system;
    Tegra::MemoryManager memory_manager{system};
    Tegra::Control::ChannelState channel_state{0};
    Tegra::DmaPusher dma_pusher{system, memory_manager, channel_state};

    Tegra::CommandList list;
    Tegra::CommandListHeader h0{};
    h0.addr.Assign(0x1000);
    h0.size.Assign(2);
    Tegra::CommandListHeader h1{};
    h1.addr.Assign(0x2000);
    h1.size.Assign(2);
    list.command_lists = {h0, h1};
    list.submitted_offsets = {0, 2};
    list.submitted_words.resize(4);
    list.submitted_words[0].argument = 0x11111111;
    list.submitted_words[1].argument = 0x22222222;
    list.submitted_words[2].argument = 0x33333333;
    list.submitted_words[3].argument = 0x44444444;

    // Refreshing from first_entry == 2 leaves both entries untouched while updating suspension tag.
    dma_pusher.RefreshSnapshot(list, 42, 2);
    REQUIRE(list.snapshot_suspensions == 42);
    REQUIRE(list.submitted_words[0].argument == 0x11111111);
    REQUIRE(list.submitted_words[1].argument == 0x22222222);
    REQUIRE(list.submitted_words[2].argument == 0x33333333);
    REQUIRE(list.submitted_words[3].argument == 0x44444444);

    // Refreshing from first_entry == 1 leaves entry 0 untouched and reads entry 1 (0 on unmapped).
    dma_pusher.RefreshSnapshot(list, 44, 1);
    REQUIRE(list.snapshot_suspensions == 44);
    REQUIRE(list.submitted_words[0].argument == 0x11111111);
    REQUIRE(list.submitted_words[1].argument == 0x22222222);
    REQUIRE(list.submitted_words[2].argument == 0);
    REQUIRE(list.submitted_words[3].argument == 0);
}
