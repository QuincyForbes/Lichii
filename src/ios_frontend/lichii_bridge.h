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
} LichiiEvent;

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
