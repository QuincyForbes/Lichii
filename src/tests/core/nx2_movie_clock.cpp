// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/hle/service/nx2_movie_clock.h"

namespace {
constexpr s64 Frame = 16667;
constexpr s64 Start = 5'000'000;

struct Shown {
    s64 frame;
    s64 ahead; ///< Its pts minus the clock when the client took it (us)
};

// A client with two output images that shows one movie frame per frame it draws, every `period`
// us (Ocarina of Time): each draw gives an image back, the decode thread fills it at once and
// the client takes it in the same draw, unless the frame waits for the clock. Its draws fall
// `lean` us before the movie's frame times and vary by `jitter` us either way. Returns the frames
// taken; `repeats` counts the draws that found no new frame.
std::vector<Shown> Play(s64 period, int draws, bool hold_early, int& repeats, s64 lean = 0,
                        s64 jitter = 0) {
    Service::Nx2MovieClock clock;
    std::vector<Shown> shown;
    s64 front = 0;
    std::optional<s64> held_until;
    bool late_before = false;
    repeats = 0;
    for (int draw = 0; draw < draws; ++draw) {
        const s64 now =
            Start + draw * period + (draw == 0 ? lean : draw % 2 != 0 ? jitter : -jitter);
        const auto due = clock.Due(now, 0);
        bool write = !due || (held_until && *held_until <= now);
        while (!write && !held_until) {
            const u64 pts = static_cast<u64>(front * Frame);
            switch (Service::PickMovieFrame(*due, Frame, pts, pts + Frame, true, late_before,
                                            hold_early, false)) {
            case Service::Nx2MoviePick::Drop:
                ++front;
                break;
            case Service::Nx2MoviePick::Hold:
                // Waits until it is a frame early.
                REQUIRE(Service::PickMovieFrame(pts - Frame, Frame, pts, pts + Frame, true, false,
                                                true, true) == Service::Nx2MoviePick::Write);
                held_until = now + static_cast<s64>(pts - *due) - Frame;
                break;
            default:
                write = true;
                break;
            }
        }
        if (!write) {
            ++repeats;
            continue;
        }
        held_until.reset();
        late_before = due && static_cast<u64>(front * Frame) < *due;
        clock.Acquired(now, static_cast<u64>(front * Frame), 0);
        shown.push_back({front, front * Frame - (now - Start)});
        ++front;
    }
    return shown;
}
} // namespace

TEST_CASE("NX2 movie clock ignores frames acquired ahead of it", "[nx2_movie_clock]") {
    // A player with two output images that takes each frame as soon as it is written, a frame
    // ahead of when it shows it.
    Service::Nx2MovieClock clock;
    clock.Acquired(Start, 0, 0);
    for (s64 frame = 1; frame < 600; ++frame) {
        CAPTURE(frame);
        const s64 now = Start + (frame - 1) * Frame + 500;
        // The frame is not late, and is not held back either.
        const u64 pts = static_cast<u64>(frame * Frame);
        REQUIRE(Service::PickMovieFrame(*clock.Due(now, 0), Frame, pts, pts + Frame, true, false,
                                        true, false) == Service::Nx2MoviePick::Write);
        REQUIRE_FALSE(clock.Acquired(now, pts, 0));
        REQUIRE(clock.Due(now, 0) == static_cast<u64>(now - Start));
    }
}

TEST_CASE("NX2 movie clock runs on while a slow client falls behind", "[nx2_movie_clock]") {
    // Ocarina of Time: one 60 fps movie frame per frame drawn, at 48 fps.
    Service::Nx2MovieClock clock;
    for (s64 frame = 0; frame < 120; ++frame) {
        const s64 now = Start + frame * 20833;
        clock.Acquired(now, static_cast<u64>(frame * Frame), 0);
        REQUIRE(clock.Due(now, 0) == static_cast<u64>(now - Start));
    }
}

TEST_CASE("NX2 movie clock restarts after a pause, a loop or a long lag", "[nx2_movie_clock]") {
    using Clock = Service::Nx2MovieClock;
    Clock clock;
    REQUIRE_FALSE(clock.Due(Start, 0));
    REQUIRE(clock.Acquired(Start, 0, 0));
    REQUIRE_FALSE(clock.Acquired(Start + Frame, Frame, 0));

    // No acquire for over a second: no due pts until the next acquire restarts the clock.
    s64 now = Start + Frame + Clock::PauseGap + 1;
    REQUIRE_FALSE(clock.Due(now, 0));
    REQUIRE(clock.Acquired(now, 2 * Frame, 0));
    REQUIRE(clock.Due(now + 1000, 0) == static_cast<u64>(2 * Frame + 1000));

    // Emulation paused and resumed since the last acquire.
    const s64 resumed = now + 2000;
    REQUIRE_FALSE(clock.Due(now + 3000, resumed));
    now += 4000;
    REQUIRE(clock.Acquired(now, 3 * Frame, resumed));
    REQUIRE(clock.Due(now, resumed) == static_cast<u64>(3 * Frame));

    // The movie loops back to its start.
    now += Frame;
    REQUIRE(clock.Acquired(now, 0, resumed));
    REQUIRE(clock.Due(now, resumed) == u64{0});

    // The client takes a frame every half second: once it lags by more than MaxCatchUp, the
    // clock restarts from its frame.
    const s64 loop = now;
    for (s64 step = 1; step <= 7; ++step) {
        now = loop + step * Clock::PauseGap / 2;
        clock.Acquired(now, static_cast<u64>(step * Frame), resumed);
        const s64 lag = (now - loop) - step * Frame;
        CAPTURE(step, lag);
        REQUIRE(clock.Due(now, resumed) ==
                static_cast<u64>(lag > Clock::MaxCatchUp ? step * Frame : now - loop));
    }

    clock.Reset();
    REQUIRE_FALSE(clock.Due(now, resumed));
}

TEST_CASE("NX2 movie clock waits for audio the host device starved of", "[nx2_movie_clock]") {
    using Clock = Service::Nx2MovieClock;
    // The device's totals from before the movie don't count.
    Clock clock;
    clock.Audio(Start, 900'000, 40'000);
    clock.Acquired(Start, 0, 0);
    REQUIRE(clock.Due(Start + 100'000, 0) == 100'000u);

    // 30 ms the device had nothing to play: the sound is 30 ms later from here on.
    clock.Audio(Start + 100'000, 970'000, 70'000);
    REQUIRE(clock.Due(Start + 100'000, 0) == 70'000u);
    clock.Audio(Start + 200'000, 1'070'000, 70'000);
    REQUIRE(clock.Due(Start + 200'000, 0) == 170'000u);
    // The client keeping up with that clock doesn't restart it.
    REQUIRE_FALSE(clock.Acquired(Start + 200'000, 170'000, 0));

    // A stream that stopped playing (the movie's sound isn't on it) no longer holds the clock.
    s64 now = Start + 200'000;
    for (int step = 0; step < 40; ++step) {
        now += 20'000;
        clock.Audio(now, 1'070'000, static_cast<u64>(70'000 + (step + 1) * 20'000));
        clock.Acquired(now, static_cast<u64>(now - Start - 30'000), 0);
    }
    const s64 waited = now - Start - static_cast<s64>(*clock.Due(now, 0));
    REQUIRE(waited >= 30'000);
    REQUIRE(waited <= 30'000 + Clock::AudioAlive);

    // A stream that never played is never waited for.
    Clock silent;
    silent.Audio(Start, 0, 0);
    silent.Acquired(Start, 0, 0);
    silent.Audio(Start + 100'000, 0, 100'000);
    REQUIRE(silent.Due(Start + 100'000, 0) == 100'000u);
}

TEST_CASE("NX2 movie frames are picked nearest the clock", "[nx2_movie_clock]") {
    using Pick = Service::Nx2MoviePick;
    const u64 f = Frame;
    const auto pick = [](u64 due, u64 front, std::optional<u64> next, bool more_input,
                         bool late_before, bool hold_early, bool holding) {
        return Service::PickMovieFrame(due, Frame, front, next, more_input, late_before,
                                       hold_early, holding);
    };
    // On time, a little late, a little early: written.
    REQUIRE(pick(10 * f, 10 * f, 11 * f, true, false, true, false) == Pick::Write);
    REQUIRE(pick(10 * f + 3000, 10 * f, 11 * f, true, false, true, false) == Pick::Write);
    REQUIRE(pick(10 * f - 20000, 10 * f, 11 * f, true, false, true, false) == Pick::Write);
    // Half a frame late: the pick stays on the side of the frame before.
    REQUIRE(pick(10 * f + 8000, 10 * f, 11 * f, true, true, false, false) == Pick::Write);
    REQUIRE(pick(10 * f + 8000, 10 * f, 11 * f, true, false, false, false) == Pick::Drop);
    REQUIRE(pick(10 * f + 13000, 10 * f, 11 * f, true, true, false, false) == Pick::Drop);
    // The next frame is nearer the clock: dropped, or decoded first.
    REQUIRE(pick(11 * f, 10 * f, 11 * f, true, false, false, false) == Pick::Drop);
    REQUIRE(pick(11 * f, 10 * f, std::nullopt, true, false, false, false) == Pick::DecodeMore);
    REQUIRE(pick(11 * f, 10 * f, std::nullopt, false, false, false, false) == Pick::Write);
    // Timestamps that don't rise are never skipped.
    REQUIRE(pick(11 * f, 0, 0, true, false, false, false) == Pick::Write);
    // More than a frame and a half early: held (when asked to) until a frame early.
    REQUIRE(pick(10 * f, 12 * f, 13 * f, true, false, true, false) == Pick::Hold);
    REQUIRE(pick(10 * f, 12 * f, 13 * f, true, false, false, false) == Pick::Write);
    REQUIRE(pick(11 * f - 4000, 12 * f, 13 * f, true, false, true, true) == Pick::Hold);
    REQUIRE(pick(11 * f - 4000, 12 * f, 13 * f, true, false, true, false) == Pick::Write);
    REQUIRE(pick(11 * f, 12 * f, 13 * f, true, false, true, true) == Pick::Write);
}

TEST_CASE("NX2 movie keeps time in a client slower or faster than it", "[nx2_movie_clock]") {
    int repeats = 0;

    // At its own rate nothing is skipped or repeated.
    auto shown = Play(Frame, 600, true, repeats);
    REQUIRE(repeats == 0);
    REQUIRE(shown.back().frame == 599);

    // At half its rate (Ocarina of Time's 30 FPS mode) every other frame is shown, on time.
    shown = Play(2 * Frame, 300, true, repeats);
    REQUIRE(repeats == 0);
    for (size_t i = 3; i < shown.size(); ++i) {
        CAPTURE(i);
        REQUIRE(shown[i].frame - shown[i - 1].frame == 2);
        REQUIRE(std::abs(shown[i].ahead) <= Frame / 2);
    }

    // Draws that fall between the movie's frame times and vary: still every other frame, after
    // one step to the side it settles on at most.
    for (const s64 lean : {-7500, -3000, 7500}) {
        shown = Play(2 * Frame, 300, true, repeats, lean, 2000);
        REQUIRE(repeats == 0);
        for (size_t i = 10; i < shown.size(); ++i) {
            CAPTURE(lean, i);
            REQUIRE(shown[i].frame - shown[i - 1].frame == 2);
        }
    }

    // At 48 fps it stays within three quarters of a frame.
    shown = Play(20833, 480, true, repeats);
    REQUIRE(repeats == 0);
    for (size_t i = 3; i < shown.size(); ++i) {
        CAPTURE(i);
        REQUIRE(std::abs(shown[i].ahead) <= Frame * 3 / 4 + 1);
    }

    // A loop 1% faster than the movie (a display genlocked a little above 60 Hz) runs ahead of
    // the sound, 10 ms a second, unless early frames wait.
    shown = Play(16500, 3600, false, repeats);
    REQUIRE(repeats == 0);
    REQUIRE(shown.back().ahead > 500'000);
    shown = Play(16500, 3600, true, repeats);
    REQUIRE(repeats > 30);
    REQUIRE(repeats < 40);
    for (size_t i = 0; i < shown.size(); ++i) {
        CAPTURE(i);
        REQUIRE(shown[i].ahead >= 0);
        REQUIRE(shown[i].ahead <= 2 * Frame);
    }
}
