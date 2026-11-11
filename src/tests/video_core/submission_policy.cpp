// SPDX-License-Identifier: GPL-3.0-or-later

#include <bit>
#include <limits>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/submission_policy.h"
#include "video_core/renderer_vulkan/upload_batch.h"

namespace {

using Vulkan::SubmissionPolicy::Action;
using Vulkan::SubmissionPolicy::Advance;
using Vulkan::SubmissionPolicy::DefaultNativeBudget;
using Vulkan::SubmissionPolicy::ParseBudget;

} // namespace

TEST_CASE("Native desktop submissions use an exact default budget", "[video_core][submission_policy]") {
    u32 counter{};
    std::vector<u32> dispatches;
    std::vector<u32> submissions;
    for (u32 work = 1; work <= 384; ++work) {
        const auto action = Advance(counter, true, false);
        if (action == Action::Dispatch) {
            dispatches.push_back(work);
        } else if (action == Action::Flush) {
            submissions.push_back(work);
            REQUIRE(counter == 0);
        }
    }
    REQUIRE(DefaultNativeBudget == 128);
    REQUIRE(submissions == std::vector<u32>{128, 256, 384});
    REQUIRE(dispatches == std::vector<u32>{32, 64, 96, 160, 192, 224, 288, 320, 352});
}

TEST_CASE("Native submission overrides do not round to a dispatch boundary",
          "[video_core][submission_policy]") {
    for (const u32 budget : {1U, 2U, 31U, 32U, 33U, 127U, 128U, 129U, 4096U}) {
        CAPTURE(budget);
        u32 counter{};
        for (u32 work = 1; work <= budget * 3; ++work) {
            const auto action = Advance(counter, true, false, budget);
            if (work % budget == 0) {
                REQUIRE(action == Action::Flush);
                REQUIRE(counter == 0);
            } else {
                REQUIRE(action == ((work % budget) % 32 == 0 ? Action::Dispatch : Action::None));
                REQUIRE(counter == work % budget);
            }
        }
    }
}

TEST_CASE("Legacy and Android submission schedules remain unchanged",
          "[video_core][submission_policy]") {
    for (const bool android : {false, true}) {
        for (const bool native : {false, true}) {
            if (native && !android) {
                continue;
            }
            for (const u32 budget : {1U, 128U, 129U, 4096U}) {
                CAPTURE(android, native, budget);
                u32 counter{};
                u32 original_counter{};
                u32 submissions{};
                const u32 check_mask = android ? 3 : 31;
                const u32 threshold = android ? 512 : 4096;
                const u32 cadence = android ? 515 : 4127;
                for (u32 work = 1; work <= cadence * 3; ++work) {
                    Action expected = Action::None;
                    if ((++original_counter & check_mask) == check_mask) {
                        if (original_counter < threshold) {
                            expected = Action::Dispatch;
                        } else {
                            expected = Action::Flush;
                            original_counter = 0;
                        }
                    }
                    const auto action = Advance(counter, native, android, budget);
                    REQUIRE(action == expected);
                    REQUIRE(counter == original_counter);
                    REQUIRE((action == Action::Flush) == (work % cadence == 0));
                    submissions += action == Action::Flush;
                }
                REQUIRE(submissions == 3);
            }
        }
    }
}

TEST_CASE("Submission budget parsing accepts only positive complete u32 decimals",
          "[video_core][submission_policy]") {
    REQUIRE(ParseBudget("1") == 1U);
    REQUIRE(ParseBudget("31") == 31U);
    REQUIRE(ParseBudget("128") == 128U);
    REQUIRE(ParseBudget("000129") == 129U);
    REQUIRE(ParseBudget("4294967295") == std::numeric_limits<u32>::max());
    for (const auto value : {"", "0", "00", "-1", "+128", " 128", "128 ", "128\n", "1.5", "1e2",
                             "0x80", "128garbage", "4294967296", "999999999999999999999999999"}) {
        CAPTURE(value);
        REQUIRE_FALSE(ParseBudget(value));
    }
    REQUIRE_FALSE(ParseBudget(std::string_view{"128\0junk", 8}));
}

TEST_CASE("Resetting the work counter restarts the submission budget",
          "[video_core][submission_policy]") {
    u32 counter{};
    for (u32 work = 1; work < 128; ++work) {
        REQUIRE(Advance(counter, true, false) != Action::Flush);
    }
    counter = 0;
    for (u32 work = 1; work < 128; ++work) {
        REQUIRE(Advance(counter, true, false) != Action::Flush);
    }
    REQUIRE(Advance(counter, true, false) == Action::Flush);
    REQUIRE(counter == 0);
    REQUIRE(Advance(counter, true, false) == Action::None);
    REQUIRE(counter == 1);
}

TEST_CASE("A maximum native budget flushes before the counter wraps",
          "[video_core][submission_policy]") {
    constexpr u32 maximum = std::numeric_limits<u32>::max();
    u32 counter = maximum - 1;
    REQUIRE(Advance(counter, true, false, maximum) == Action::Flush);
    REQUIRE(counter == 0);
    REQUIRE(Advance(counter, true, false, maximum) == Action::None);
    REQUIRE(counter == 1);
}

TEST_CASE("Changing the active native policy preserves recorded work",
          "[video_core][submission_policy]") {
    u32 counter = 200;
    REQUIRE(Advance(counter, true, false) == Action::Flush);
    REQUIRE(counter == 0);
    for (u32 work = 1; work < 31; ++work) {
        REQUIRE(Advance(counter, false, false) == Action::None);
    }
    REQUIRE(Advance(counter, false, false) == Action::Dispatch);
    REQUIRE(counter == 31);
    REQUIRE(Advance(counter, true, false) == Action::Dispatch);
    REQUIRE(counter == 32);
}

TEST_CASE("UploadBatch coalesces non-overlapping copies and rejects hazards",
          "[video_core][submission_policy][upload_batch]") {
    const auto buf = [](u64 id) { return std::bit_cast<VkBuffer>(id); };
    const VkBuffer stream = buf(1);
    const VkBuffer dst_a = buf(2);
    const VkBuffer dst_b = buf(3);

    Vulkan::UploadBatch batch;
    REQUIRE(batch.Empty());

    const VkBufferCopy c0{.srcOffset = 0, .dstOffset = 0x100, .size = 0x40};
    const VkBufferCopy c1{.srcOffset = 0x40, .dstOffset = 0x140, .size = 0x40};
    REQUIRE(batch.CanAdd(stream, dst_a, std::span{&c0, 1}));
    batch.Add(stream, dst_a, std::span{&c0, 1});
    REQUIRE(batch.CanAdd(stream, dst_a, std::span{&c1, 1}));
    batch.Add(stream, dst_a, std::span{&c1, 1});
    REQUIRE(batch.groups.size() == 1);
    REQUIRE(batch.groups[0].count == 2);

    // Overlapping destination bytes are rejected so WAW order is preserved.
    const VkBufferCopy overlap{.srcOffset = 0x80, .dstOffset = 0x120, .size = 0x30};
    REQUIRE_FALSE(batch.CanAdd(stream, dst_a, std::span{&overlap, 1}));

    // Different destination buffer starts a second group in the same batch.
    REQUIRE(batch.CanAdd(stream, dst_b, std::span{&c0, 1}));
    batch.Add(stream, dst_b, std::span{&c0, 1});
    REQUIRE(batch.groups.size() == 2);

    // RAW (src == pending dst) and WAR (dst == pending src) hazards are rejected.
    REQUIRE_FALSE(batch.CanAdd(dst_a, dst_b, std::span{&c0, 1}));
    REQUIRE_FALSE(batch.CanAdd(dst_b, stream, std::span{&c0, 1}));

    // Capacity cap (MaxCopies) is enforced.
    for (size_t i = batch.copies.size(); i < Vulkan::UploadBatch::MaxCopies; ++i) {
        const VkBufferCopy non_overlap{
            .srcOffset = i * 0x10,
            .dstOffset = 0x1000 + i * 0x10,
            .size = 0x10,
        };
        REQUIRE(batch.CanAdd(stream, dst_b, std::span{&non_overlap, 1}));
        batch.Add(stream, dst_b, std::span{&non_overlap, 1});
    }
    const VkBufferCopy past_cap{.srcOffset = 0, .dstOffset = 0x9000, .size = 0x10};
    REQUIRE_FALSE(batch.CanAdd(stream, dst_b, std::span{&past_cap, 1}));
}
