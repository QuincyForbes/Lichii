// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>

#include "common/common_types.h"

namespace Service {

// The clock a video decoder session (nx2:vdecsvc) shows a movie's frames by: the pts the movie's
// audio has reached. Times are host microseconds; pause_end is when emulation last resumed.
class Nx2MovieClock {
public:
    // A longer gap between acquires is a pause, a longer lag a jump in the stream; both restart
    // the clock.
    static constexpr s64 PauseGap = 1'000'000;
    static constexpr s64 MaxCatchUp = 3'000'000;
    // Audio the host device waited for counts as the movie's sound running late only while the
    // stream played this recently: a stream that plays nothing isn't the movie's.
    static constexpr s64 AudioAlive = 250'000;

    void Reset() {
        offset.reset();
    }

    // The client acquired the frame at pts. The clock starts from the first acquire and runs in
    // real time from there. A frame acquired ahead of the clock doesn't move it: Kirby Air Riders
    // takes a frame before it shows it and holds it until its own clock reaches it, and following
    // those acquires ran the clock ahead of the picture, so 563 of 588 frames came out late.
    // Returns whether the clock (re)started from this frame.
    bool Acquired(s64 now, u64 pts, s64 pause_end) {
        const s64 acquired_offset = now - stalled - static_cast<s64>(pts);
        const bool restart = !offset || now - last_acquire > PauseGap || pause_end > last_acquire ||
                             pts < last_acquired_pts || acquired_offset - *offset > MaxCatchUp;
        if (restart) {
            offset = acquired_offset;
        }
        last_acquire = now;
        last_acquired_pts = pts;
        return restart;
    }

    // The host audio device's totals for the game's audio (us): what it played, and how long it
    // had nothing to play. The second delays everything the game plays after it, so the clock
    // waits with it.
    void Audio(s64 now, u64 played, u64 starved) {
        if (played != last_played) {
            last_played = played;
            alive_until = now + AudioAlive;
        }
        if (now <= alive_until && starved > last_starved) {
            stalled += static_cast<s64>(starved - last_starved);
        }
        last_starved = starved;
    }

    // The pts the movie's audio has reached, unless the client hasn't acquired a frame yet or for
    // a while (paused; the next acquire restarts the clock).
    std::optional<u64> Due(s64 now, s64 pause_end) const {
        if (!offset) {
            return std::nullopt;
        }
        const s64 due = now - stalled - *offset;
        if (due < 0 || now - last_acquire > PauseGap || pause_end > last_acquire) {
            return std::nullopt;
        }
        return static_cast<u64>(due);
    }

private:
    std::optional<s64> offset; ///< Host time minus pts of the frame the clock started from
    s64 last_acquire{};
    u64 last_acquired_pts{};
    s64 stalled{}; ///< Audio starvation the clock waited for
    s64 alive_until{};
    u64 last_played{};
    u64 last_starved{};
};

enum class Nx2MoviePick {
    Write,      ///< Put the oldest decoded frame in the free output image
    Drop,       ///< It is late and the next one is due: skip it
    DecodeMore, ///< It is late and the next one isn't decoded yet: decode on first
    Hold,       ///< It is early: wait
};

// What to do with the oldest decoded frame (pts front) when an output image is free. due is the
// pts the clock will be at when the client takes the frame, interval the time between frames,
// next the pts of the decoded frame after it and more_input whether one can be decoded.
//
// A client that shows one frame per frame it draws and never skips one (Ocarina of Time) gets
// the frame nearest the clock: at half the movie's rate that is every other frame, whichever way
// its timing leans. late_before says the frame written before this one was past its time: the
// pick leans the same way by a quarter of a frame, or a client whose draws fall half a frame
// from the movie's would get the frames in steps of 1 and 3 as its timing varies.
//
// Early frames wait only when hold_early is set, and only once they are more than a frame and a
// half early: frame times that merely vary neither repeat nor skip a frame, and a client that
// takes each frame one ahead of its own clock is left alone. A waiting frame (holding) is
// written a frame early, for the client's next draw to find it.
constexpr Nx2MoviePick PickMovieFrame(u64 due, u64 interval, u64 front, std::optional<u64> next,
                                      bool more_input, bool late_before, bool hold_early,
                                      bool holding) {
    // How far ahead of the clock the next frame may be to be shown instead of this one.
    const u64 reach = late_before ? interval / 4 : interval - interval / 4;
    if (next) {
        // Streams whose timestamps don't rise are never skipped.
        if (*next > front && *next <= due + reach) {
            return Nx2MoviePick::Drop;
        }
    } else if (more_input && front + interval <= due + reach) {
        return Nx2MoviePick::DecodeMore;
    }
    if (hold_early && front > due + interval + (holding ? 0 : interval / 2)) {
        return Nx2MoviePick::Hold;
    }
    return Nx2MoviePick::Write;
}

} // namespace Service
