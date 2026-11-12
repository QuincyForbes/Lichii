// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>

#include "common/game_presets.h"
#include "common/game_settings.h"

namespace Settings {

namespace {
constexpr std::array<std::string_view, 4> OcarinaKeys{
    "oot_resolution",
    "oot_frame_rate",
    "oot_anti_aliasing",
    "oot_movie_upscale",
};
// EffectiveResolutionSetup: oot_resolution sets the scale for Ocarina of Time.
constexpr std::array<std::string_view, 1> OcarinaReplaces{
    "resolution_setup",
};
constexpr std::array<std::string_view, 1> MarioKartKeys{
    "mkw_frame_rate",
};
constexpr std::array<GameSettings, 2> Games{{
    {OcarinaOfTimeProgramId, "The Legend of Zelda: Ocarina of Time", OcarinaKeys,
     OcarinaReplaces},
    {MarioKartWorldProgramId, "Mario Kart World", MarioKartKeys, {}},
}};
} // namespace

std::span<const GameSettings> GamesWithSettings() {
    return Games;
}

const GameSettings* GameOfSetting(std::string_view key) {
    for (const GameSettings& game : Games) {
        if (std::ranges::find(game.keys, key) != game.keys.end()) {
            return &game;
        }
    }
    return nullptr;
}

bool AppliesToGame(std::string_view key, u64 program_id) {
    if (program_id == 0) {
        return true;
    }
    if (const GameSettings* const game = GameOfSetting(key)) {
        return game->program_id == program_id;
    }
    for (const GameSettings& game : Games) {
        if (game.program_id == program_id &&
            std::ranges::find(game.replaces, key) != game.replaces.end()) {
            return false;
        }
    }
    return true;
}

} // namespace Settings
