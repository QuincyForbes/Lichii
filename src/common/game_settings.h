// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>
#include <string_view>

#include "common/common_types.h"

// Settings that change one game only. A per-game page lists them only for that game; the global
// pages group them under the game's name (Qt Graphics, the iOS Graphics page).
namespace Settings {

inline constexpr u64 OcarinaOfTimeProgramId = 0x0400486024DEC000ULL;

struct GameSettings {
    u64 program_id;
    std::string_view name;
    std::span<const std::string_view> keys;
    /// Settings for every game that this game's own ones take the place of (Ocarina of Time's
    /// resolution includes the resolution scale): its per-game page doesn't list them.
    std::span<const std::string_view> replaces;
};

/// Every game with settings of its own.
std::span<const GameSettings> GamesWithSettings();

/// The game a setting belongs to, or nullptr when it applies to every game.
const GameSettings* GameOfSetting(std::string_view key);

/// Whether a per-game page of `program_id` (0: the global pages) lists the setting: not another
/// game's own settings, nor what the game's own replace.
bool AppliesToGame(std::string_view key, u64 program_id);

} // namespace Settings
