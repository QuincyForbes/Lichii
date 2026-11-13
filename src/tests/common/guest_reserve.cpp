// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "common/platform_events.h"

namespace {

using Common::PlatformEvents::GuestReserveTracker;
constexpr u64 MiB = u64{1} << 20;
constexpr long long Second = 1'000'000'000LL;

GuestReserveTracker::Config PhoneConfig(bool ratchet = false) {
    return {
        .base = 1024 * MiB,
        .max = 3072 * MiB,
        .threshold = 1500,
        .sustain_seconds = 5,
        .decay_seconds = 10,
        .step = 256 * MiB,
        .loading_growth = 32 * MiB,
        .ratchet = ratchet,
    };
}

struct Sample {
    double rate;            ///< Page-ins a second
    double resident_growth; ///< MiB a second
};

/// Feeds a tracker one sample a second; returns the extra reserve after each.
class Feeder {
public:
    explicit Feeder(bool ratchet = false) : tracker{PhoneConfig(ratchet)} {
        tracker.Sample(now, pageins, static_cast<u64>(resident));
    }

    std::vector<u64> Run(const std::vector<Sample>& samples) {
        std::vector<u64> extras;
        for (const Sample& sample : samples) {
            now += Second;
            pageins += static_cast<u64>(sample.rate);
            resident += sample.resident_growth * static_cast<double>(MiB);
            tracker.Sample(now, pageins, static_cast<u64>(resident));
            extras.push_back(tracker.Extra());
        }
        return extras;
    }

private:
    GuestReserveTracker tracker;
    long long now = 100 * Second;
    u64 pageins = 1'000'000;
    double resident = 512.0 * static_cast<double>(MiB);
};

std::vector<Sample> Repeat(Sample sample, size_t count) {
    return std::vector<Sample>(count, sample);
}

// iPhone18,2, build 7707654e51 at 69-80 s: the page-in rates of boot and title loading. The log
// has no resident figure for them; the file filling as it loads (growth near the rate's MiB at
// 16 KiB pages) is assumed.
const std::vector<Sample> BootLoading{
    {20371, 300}, {21394, 300}, {5753, 90}, {8530, 130}, {12177, 190},
    {5202, 80},   {7943, 120},  {7943, 120}, {1505, 24},
};

} // namespace

TEST_CASE("The earlier rule grew the reserve to its maximum while the game loaded",
          "[guest_reserve]") {
    Feeder feeder{true};
    CHECK(feeder.Run(BootLoading).back() == 2048 * MiB);
    // And kept it at the Kokiri rate (450/s, over a quarter of 1500).
    CHECK(feeder.Run(Repeat({450, 0}, 40)).back() == 2048 * MiB);
}

TEST_CASE("Loading leaves the reserve alone", "[guest_reserve]") {
    Feeder feeder;
    CHECK(feeder.Run(BootLoading).back() == 0);
}

TEST_CASE("Sustained thrashing grows the reserve a step at a time", "[guest_reserve]") {
    Feeder feeder;
    const auto extras = feeder.Run(Repeat({9000, 0}, 12));
    CHECK(extras[3] == 0);
    CHECK(extras[4] == 256 * MiB);
    CHECK(extras[8] == 256 * MiB);
    CHECK(extras[9] == 512 * MiB);

    // A loading second in between starts the count again.
    Feeder mixed;
    std::vector<Sample> samples = Repeat({9000, 0}, 3);
    samples.push_back({9000, 200});
    for (const Sample& sample : Repeat({9000, -5}, 3)) {
        samples.push_back(sample);
    }
    CHECK(mixed.Run(samples).back() == 0);
}

TEST_CASE("The reserve decays once page-ins calm down", "[guest_reserve]") {
    Feeder feeder;
    std::vector<Sample> samples = Repeat({9000, 0}, 10);
    for (const Sample& sample : Repeat({450, 0}, 25)) {
        samples.push_back(sample);
    }
    const auto extras = feeder.Run(samples);
    CHECK(extras[9] == 512 * MiB);
    CHECK(extras[18] == 512 * MiB);
    CHECK(extras[19] == 256 * MiB);
    CHECK(extras[29] == 0);
}

TEST_CASE("The caches give the file at most half their room", "[guest_reserve]") {
    using Common::PlatformEvents::BoundedGuestFileReserve;
    constexpr u64 Own = 768 * MiB;
    // The iPhone: 3072 asked for with ~2 GiB left left the caches 1 MiB.
    CHECK(BoundedGuestFileReserve(3072 * MiB, 2048 * MiB, Own) == 640 * MiB);
    CHECK(BoundedGuestFileReserve(1024 * MiB, 4096 * MiB, Own) == 1024 * MiB);
    CHECK(BoundedGuestFileReserve(1024 * MiB, 512 * MiB, Own) == 0);
}
