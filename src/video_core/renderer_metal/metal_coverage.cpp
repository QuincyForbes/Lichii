// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <atomic>
#include <map>
#include <mutex>
#include <utility>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/renderer_metal/metal_coverage.h"

namespace Metal::Coverage {

namespace {

constexpr size_t NumWork = static_cast<size_t>(Work::Count);
constexpr size_t NumPaths = static_cast<size_t>(Path::Count);
/// Causes past this many count as "other" (a cause is a short fixed text, not a state).
constexpr size_t MaxCauses = 48;

constexpr std::array<std::string_view, NumWork> WorkNames{"draw", "dispatch", "clear"};
constexpr std::array<std::string_view, NumWork> WorkPlurals{"draws", "dispatches", "clears"};
constexpr std::array<std::string_view, NumPaths> PathNames{
    "native",        "mesh as compute", "tessellation kernels", "rewritten indices",
    "indirect loop", "replaced",        "clear as a draw",      "empty",
};

struct State {
    std::array<std::array<std::atomic<u64>, NumPaths>, NumWork> paths{};
    std::atomic<u64> total_refused{};
    std::mutex mutex;
    /// By cause: refusals since the last line.
    std::array<std::map<std::string, u64, std::less<>>, NumWork> refused;
    /// By name: guards since the last line.
    std::map<std::string, u64, std::less<>> guards;
};

State& Get() {
    static State state;
    return state;
}

} // Anonymous namespace

void Note(Work work, Path path) noexcept {
    Get().paths[static_cast<size_t>(work)][static_cast<size_t>(path)].fetch_add(
        1, std::memory_order_relaxed);
}

bool NoteRefused(Work work, std::string_view cause) {
    State& state = Get();
    state.total_refused.fetch_add(1, std::memory_order_relaxed);
    std::scoped_lock lock{state.mutex};
    auto& causes = state.refused[static_cast<size_t>(work)];
    if (const auto it = causes.find(cause); it != causes.end()) {
        ++it->second;
        return false;
    }
    if (causes.size() >= MaxCauses) {
        ++causes["other"];
        return false;
    }
    causes.emplace(std::string{cause}, 1);
    return true;
}

void LogRefused(Work work, std::string_view cause, std::string_view state) {
    if (state.empty()) {
        LOG_WARNING(Render_Metal, "Metal: {} not done: {}", WorkNames[static_cast<size_t>(work)],
                    cause);
    } else {
        LOG_WARNING(Render_Metal, "Metal: {} not done: {} ({})",
                    WorkNames[static_cast<size_t>(work)], cause, state);
    }
}

bool CountGuard(std::string_view name) {
    State& state = Get();
    std::scoped_lock lock{state.mutex};
    if (const auto it = state.guards.find(name); it != state.guards.end()) {
        ++it->second;
        return false;
    }
    if (state.guards.size() >= MaxCauses) {
        ++state.guards["other"];
        return false;
    }
    state.guards.emplace(std::string{name}, 1);
    return true;
}

void NoteGuard(std::string_view name, std::string_view detail) {
    if (CountGuard(name)) {
        LogGuard(name, detail);
    }
}

void LogGuard(std::string_view name, std::string_view detail) {
    if (detail.empty()) {
        LOG_WARNING(Render_Metal, "Metal: guard: {}", name);
    } else {
        LOG_WARNING(Render_Metal, "Metal: guard: {} ({})", name, detail);
    }
}

std::string TakeLine() {
    State& state = Get();
    std::string line;
    std::scoped_lock lock{state.mutex};
    std::string guards;
    for (auto& [name, count] : state.guards) {
        if (count != 0) {
            guards += fmt::format("{}{} {}", guards.empty() ? "" : ", ", name, count);
            count = 0;
        }
    }

    for (size_t work = 0; work < NumWork; ++work) {
        std::array<u64, NumPaths> paths;
        u64 emulated = 0;
        for (size_t path = 0; path < NumPaths; ++path) {
            paths[path] = state.paths[work][path].exchange(0, std::memory_order_relaxed);
            if (path != static_cast<size_t>(Path::Native) &&
                path != static_cast<size_t>(Path::Empty)) {
                emulated += paths[path];
            }
        }
        u64 refused = 0;
        std::string causes;
        for (auto& [cause, count] : state.refused[work]) {
            if (count == 0) {
                continue;
            }
            refused += count;
            causes += fmt::format("{}{} {}", causes.empty() ? "" : ", ", cause, count);
            // The cause stays known, so it is logged with its state only once.
            count = 0;
        }
        const u64 native = paths[static_cast<size_t>(Path::Native)];
        const u64 empty = paths[static_cast<size_t>(Path::Empty)];
        const u64 total = native + emulated + refused + empty;
        if (total == 0) {
            continue;
        }
        line += fmt::format("{}{} {} = {} native + {} emulated", line.empty() ? "" : "; ",
                            WorkPlurals[work], total, native, emulated);
        if (emulated != 0) {
            std::string kinds;
            for (size_t path = 0; path < NumPaths; ++path) {
                if (path == static_cast<size_t>(Path::Native) ||
                    path == static_cast<size_t>(Path::Empty) || paths[path] == 0) {
                    continue;
                }
                kinds += fmt::format("{}{} {}", kinds.empty() ? "" : ", ", PathNames[path],
                                     paths[path]);
            }
            line += fmt::format(" ({})", kinds);
        }
        line += fmt::format(" + {} not done", refused);
        if (refused != 0) {
            line += fmt::format(" ({})", causes);
        }
        line += fmt::format(" + {} empty", empty);
    }
    if (!guards.empty()) {
        line += fmt::format("{}guards: {}", line.empty() ? "" : "; ", guards);
    }
    return line;
}

u64 TotalRefused() noexcept {
    return Get().total_refused.load(std::memory_order_relaxed);
}

} // namespace Metal::Coverage
