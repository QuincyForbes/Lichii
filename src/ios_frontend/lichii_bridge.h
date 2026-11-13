// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// C interface between the emulator core and the native iOS app (Swift). Every function is called
// on the main thread unless noted. Returned `char*` strings are heap copies the caller releases
// with lichii_free; `const char*` strings stay valid for the life of the process.
//
// Other areas add their own headers next to this one (lichii_bridge_controls.h,
// lichii_bridge_content.h, lichii_bridge_platform.h) and list them in module.modulemap.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void lichii_free(void* pointer);

/* ------------------------------------------------------------------------------------- paths */

/// The app's Documents folder (shared with the Files app).
char* lichii_documents_dir(void);
/// Emulator data: Documents/eden (config, keys, nand, sdmc, load, log, shader).
char* lichii_user_dir(void);
/// The current log file (Documents/eden/log/eden_log.txt).
char* lichii_log_file(void);
/// "<build name> | <branch>-<description>".
char* lichii_build_version(void);

/* ------------------------------------------------------------------------------------- games */

typedef enum LichiiFileType {
    LICHII_FILE_UNKNOWN = 0,
    LICHII_FILE_NSO,
    LICHII_FILE_NRO,
    LICHII_FILE_NCA,
    LICHII_FILE_NSP,
    LICHII_FILE_NSZ,
    LICHII_FILE_XCI,
    LICHII_FILE_XCZ,
    LICHII_FILE_NUP,
    LICHII_FILE_DIRECTORY, ///< a deconstructed ROM directory (exefs with main + main.npdm)
} LichiiFileType;

typedef struct LichiiGame {
    char* path;      ///< what lichii_emu_launch takes (a file, a split "00" part, or a directory)
    char* title;
    char* developer;
    char* version;
    uint64_t program_id;
    uint64_t file_size;
    int32_t file_type; ///< LichiiFileType
    bool is_homebrew;
    uint8_t* icon;     ///< encoded image (JPEG, or PNG for homebrew), NULL when the title has none
    size_t icon_size;
} LichiiGame;

/// Scans `dir` and its subfolders (two levels, skipping "eden") for games. Fills *out_games with an
/// array the caller frees with lichii_games_free. Safe to call from a background thread while no
/// game is running.
size_t lichii_games_scan(const char* dir, LichiiGame** out_games);
/// Reads one game; false when the file is not a bootable title. Background-thread safe like scan.
bool lichii_game_read(const char* path, LichiiGame* out_game);
void lichii_game_clear(LichiiGame* game);
void lichii_games_free(LichiiGame* games, size_t count);

/* ---------------------------------------------------------------------------------- settings */

typedef enum LichiiSettingKind {
    LICHII_SETTING_BOOL = 0,
    LICHII_SETTING_INT,
    LICHII_SETTING_FLOAT,
    LICHII_SETTING_ENUM,
    LICHII_SETTING_STRING,
} LichiiSettingKind;

typedef struct LichiiSettingInfo {
    const char* key;      ///< ini key, e.g. "resolution_setup"
    const char* category; ///< ini group, e.g. "Renderer"
    uint32_t category_id; ///< Settings::Category
    int32_t kind;         ///< LichiiSettingKind
    bool switchable;      ///< can be overridden per game
    bool runtime_modifiable;
    bool ranged;
    bool percentage;
    uint32_t specialization;
    double minimum;
    double maximum;
    const char* paired_key; ///< NULL when unpaired
    uint32_t enum_count;
} LichiiSettingInfo;

/// Every setting a config file saves, in declaration order.
size_t lichii_settings_count(void);
bool lichii_settings_info(size_t index, LichiiSettingInfo* out_info);
bool lichii_settings_find(const char* key, LichiiSettingInfo* out_info);
/// Enum choices: canonical name and the value lichii_settings_set takes (as a decimal string).
const char* lichii_settings_enum_name(const char* key, uint32_t index);
uint32_t lichii_settings_enum_value(const char* key, uint32_t index);

/// Values as strings ("true"/"false", decimal numbers, enum values as numbers). Inside a
/// per-game scope they read and write the game's value.
char* lichii_settings_get(const char* key);
char* lichii_settings_get_default(const char* key);
bool lichii_settings_set(const char* key, const char* value);

/// Per-game scope: settings read and written between begin and end belong to that title; end
/// saves Documents/eden/config/custom/<program id>.ini. Global settings are restored unless that
/// game is running.
bool lichii_settings_begin_game(uint64_t program_id);
void lichii_settings_end_game(void);
bool lichii_settings_in_game_scope(void);
bool lichii_settings_has_game_config(uint64_t program_id);
bool lichii_settings_uses_global(const char* key);
void lichii_settings_set_use_global(const char* key, bool use_global);

/// Writes the global config. Applies the values to a running game as well.
void lichii_settings_save(void);

/// Outside a per-game scope: the value in effect now (the running game's own value where its
/// per-game config overrides the setting, else the global one).
char* lichii_settings_get_effective(const char* key);
/// Changes the value in effect and saves it where it lives: the running game's per-game config
/// when that overrides the setting, else the global config. Applies it to a running game.
bool lichii_settings_set_effective(const char* key, const char* value);

/// Optional parts of the renderer this build has: frame generation (LSFG and FSR 3 frame
/// generation) and ReShade post-processing.
bool lichii_settings_frame_gen_built(void);
bool lichii_settings_post_shaders_built(void);

/// The game a setting belongs to (common/game_settings.h), 0 when it applies to every game. A
/// per-game page lists it only for that game.
uint64_t lichii_settings_game(const char* key);
/// Whether the per-game page of `program_id` lists the setting: not another game's own settings,
/// nor those the game's own replace (Ocarina of Time's resolution includes the scale).
bool lichii_settings_applies(const char* key, uint64_t program_id);
/// The name of a game with settings of its own, "" otherwise.
const char* lichii_settings_game_name(uint64_t program_id);
/// Inside a per-game scope: the game gets this device's recommended value of every setting that
/// applies to it, its other overrides go back to the global value. Returns how many changed, -1
/// outside a scope.
int lichii_settings_reset_game(void);
/// What lichii_settings_reset_game would change, one "key\tfrom\tto" line each ("global" for an
/// override removed). Free with lichii_free.
char* lichii_settings_reset_game_preview(void);

/* ----------------------------------------------------------------------- recommended settings */

/// One recommended value for this device (common/device_profile.h).
typedef struct LichiiRecommended {
    const char* key;     ///< settings key, or an app item ("guest_memory_file", "auto_cool")
    const char* value;   ///< as lichii_settings_set takes it
    const char* display; ///< canonical name ("Percent56", "On")
    const char* reason;
    bool app;            ///< an app item the app applies itself
} LichiiRecommended;

/// Builds the recommended settings for this device; returns how many. The strings stay valid
/// until the next call.
size_t lichii_recommended_load(void);
bool lichii_recommended_get(size_t index, LichiiRecommended* out);
/// "iPhone 17 Pro Max (A19 Pro, 12 GB)"
char* lichii_recommended_device(void);
/// Writes the recommended settings to the global config and saves it. Returns how many changed,
/// -1 while a game runs or a per-game scope is open.
int lichii_recommended_apply(void);
/// How many global settings differ from this device's recommendation (0 inside a per-game scope).
size_t lichii_recommended_differences(void);

/* ------------------------------------------------------------------------------------ emulation */

typedef enum LichiiEmuState {
    LICHII_EMU_IDLE = 0,
    LICHII_EMU_WAITING_FOR_JIT, ///< launch queued until the JIT pool is ready
    LICHII_EMU_BOOTING,
    LICHII_EMU_RUNNING,
    LICHII_EMU_PAUSED,
    LICHII_EMU_FAILED, ///< lichii_emu_error says why; the next launch clears it
} LichiiEmuState;

/// Queues a game; it boots as soon as the JIT pool is ready. False while another game runs.
bool lichii_emu_launch(const char* path);
/// Like lichii_emu_launch, but the boot builds every cached pipeline of the game (the shader
/// precache, whatever precache_shaders says) and returns to the library instead of starting it.
bool lichii_emu_build_shaders(const char* path);

/// The running (or last) shader build: Build Shaders, or the precache before a game starts.
typedef struct LichiiBuildShadersStatus {
    uint64_t total;      ///< pipelines to build this run
    uint64_t done;       ///< built, failed or given up
    uint64_t built;
    uint64_t failed;     ///< includes builds given up (that never returned)
    uint64_t skipped;    ///< already built for this driver by an earlier run
    uint64_t threads;
    uint64_t elapsed_ms;
    uint64_t slowest_ms; ///< the longest build still running
    bool running;
} LichiiBuildShadersStatus;
void lichii_emu_build_shaders_status(LichiiBuildShadersStatus* out);
/// Drops a queued launch, or stops a boot once the game has loaded (the shader cache load ends
/// early); the state returns to IDLE.
void lichii_emu_cancel_launch(void);
void lichii_emu_pause(void);
void lichii_emu_resume(void);
/// Shuts the game down and returns to the library (asynchronous: watch the state).
void lichii_emu_stop(void);
int32_t lichii_emu_state(void);
char* lichii_emu_error(void);
uint64_t lichii_emu_program_id(void);
char* lichii_emu_game_path(void);
/// UIWindow* the game renders into while a game is booted, else NULL.
void* lichii_emu_window(void);

typedef struct LichiiPerfStats {
    double fps;           ///< displayed frames per second (game + generated)
    double game_fps;      ///< frames the game rendered
    double generated_fps; ///< frame generation
    double frametime_ms;
    double speed;         ///< emulation speed, 1.0 = full speed
} LichiiPerfStats;

/// Stats since the previous call. False while no game runs.
bool lichii_emu_perf_stats(LichiiPerfStats* out_stats);

/// Called on a renderer thread with BGRA pixels (rows bottom-up when flip_y).
typedef void (*LichiiScreenshotFn)(void* context, const uint8_t* bgra, uint32_t width,
                                   uint32_t height, bool flip_y);
bool lichii_emu_screenshot(uint32_t height, LichiiScreenshotFn done, void* context);

typedef enum LichiiEvent {
    LICHII_EVENT_STATE_CHANGED = 0,
    LICHII_EVENT_GAME_WINDOW_READY,  ///< the render window exists: overlays can be attached
    LICHII_EVENT_MENU_REQUESTED,     ///< a controller or overlay asked for the in-game menu
    LICHII_EVENT_CONTROLLERS_CHANGED,
    LICHII_EVENT_OPEN_URL, ///< a URL or file was opened with the app: lichii_take_open_url
    LICHII_EVENT_JIT_CHANGED, ///< the JIT state or the prepared pool size changed
} LichiiEvent;

/// The URL (lichii://game?id=<title id>) or file path opened with the app, or "" when none.
char* lichii_take_open_url(void);

typedef void (*LichiiEventFn)(int32_t event);
/// Events arrive on the main thread.
void lichii_set_event_handler(LichiiEventFn handler);
void lichii_post_event(int32_t event);

/* ------------------------------------------------------------------------------------------ JIT */

typedef enum LichiiJitState {
    LICHII_JIT_WAITING = 0, ///< no debugger yet
    LICHII_JIT_ACQUIRING,   ///< StikDebug is preparing the pool
    LICHII_JIT_READY,
    LICHII_JIT_FAILED,
} LichiiJitState;

int32_t lichii_jit_state(void);
uint64_t lichii_jit_prepared_bytes(void);
uint64_t lichii_jit_pool_bytes(void);
char* lichii_jit_error(void);
/// CS_DEBUGGED is set (a debugger attached at least once).
bool lichii_jit_debugged(void);
/// This device needs StikDebug's script (iOS 26+ with TXM).
bool lichii_jit_needs_script(void);
bool lichii_jit_stikdebug_installed(void);
/// Asks StikDebug to enable JIT for this process (opens its URL scheme).
bool lichii_jit_request(void);
/// Opens another JIT provider's URL, keeping the app alive in the background meanwhile.
bool lichii_jit_request_url(const char* url);

/* ---------------------------------------------------------------------------------------- input */

/// Buttons are InputCommon::VirtualGamepad::VirtualButton values.
typedef enum LichiiButton {
    LICHII_BUTTON_A = 0,
    LICHII_BUTTON_B,
    LICHII_BUTTON_X,
    LICHII_BUTTON_Y,
    LICHII_BUTTON_LS,
    LICHII_BUTTON_RS,
    LICHII_BUTTON_L,
    LICHII_BUTTON_R,
    LICHII_BUTTON_ZL,
    LICHII_BUTTON_ZR,
    LICHII_BUTTON_PLUS,
    LICHII_BUTTON_MINUS,
    LICHII_BUTTON_LEFT,
    LICHII_BUTTON_UP,
    LICHII_BUTTON_RIGHT,
    LICHII_BUTTON_DOWN,
    LICHII_BUTTON_SL,
    LICHII_BUTTON_SR,
    LICHII_BUTTON_HOME,
    LICHII_BUTTON_CAPTURE,
} LichiiButton;

/// Player 1 and handheld both receive virtual input. Stick 0 is left, 1 right; y is +1 up.
void lichii_input_button(int32_t button, bool pressed);
void lichii_input_stick(int32_t stick, float x, float y);
/// Touch screen in normalized coordinates of the game image (0..1).
void lichii_input_touch_down(uint32_t finger, float x, float y);
void lichii_input_touch_move(uint32_t finger, float x, float y);
void lichii_input_touch_up(uint32_t finger);
void lichii_input_release_all(void);

#ifdef __cplusplus
}
#endif
