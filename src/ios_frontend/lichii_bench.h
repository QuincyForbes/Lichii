// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Bench mode: a host (tools/nx2/ios_bench) starts the app with a request, and the app boots the
// game, drives it to the measured place, measures and writes Documents/bench/result-*.json, with
// nothing done on the screen. Off unless a launch argument, the environment or
// Documents/bench/request.json asks for it.

namespace LichiiBench {

/// SDL_AppInit, after logging starts and before the launch parameters are read: finds a request
/// ("--lichii-bench <profile|file>", LICHII_BENCH, Documents/bench/request.json), removes the bench
/// arguments from argv and applies the request's environment and experiments.
void Configure(int& argc, char** argv);
bool Active();
/// Appended to the iOS log filter.
const char* LogFilter();

/// Once the global config is loaded and the system is up (IosStartApp): the request's settings for
/// this run only, then the bench thread. Without bench mode it puts back an owner save an
/// interrupted run left aside.
void Start();
/// After a game's own config loaded over the global values (IosFrontend::ApplyGameConfig).
void ReapplySettings();
/// The config is not written while the bench runs: its settings are for this run only.
bool BlocksConfigSave();

} // namespace LichiiBench
