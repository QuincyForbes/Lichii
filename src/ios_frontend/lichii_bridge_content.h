// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Content management for the iOS app: keys, firmware, updates/DLC, mods, cheats, saves, logs,
// user profiles, system applets and guest dialogs. Plain C so Swift imports it directly.
//
// Conventions:
// - Program ids are the base title id (the game), as a uint64.
// - Functions returning `char*` hand over a malloc'd UTF-8 string (usually JSON); release it with
//   lichii_content_free. NULL means failure.
// - Long operations take a progress callback; return true from it to cancel. They block, so call
//   them off the main thread.
// - Everything is serialised by one internal lock and safe to call from any thread.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef bool (*LichiiContentProgress)(void* user, uint64_t done, uint64_t total);

void lichii_content_free(void* p);

/// {"user_dir","documents","keys_dir","load_dir","log_dir","nand_dir","sdmc_dir","save_dir",
///  "cache_dir"}
char* lichii_content_paths_json(void);

/// {"keys":{...},"firmware":{...},"profiles":N,"ready":bool}: what onboarding still needs.
char* lichii_content_setup_status_json(void);

// ---- Keys ----------------------------------------------------------------------------------

typedef enum {
    LICHII_KEYS_OK = 0,
    LICHII_KEYS_NO_FILES = 1,      // nothing recognisable was given
    LICHII_KEYS_INVALID_FORMAT = 2, // a file is not a "name = hex" key list
    LICHII_KEYS_COPY_FAILED = 3,
    LICHII_KEYS_INCOMPLETE = 4,    // copied, but the header/master keys are still missing
} LichiiKeyResult;

/// Validates and copies prod.keys / title.keys / key_retail.bin (any names: a file whose
/// entries look like title keys is stored as title.keys, other *.keys as prod.keys) into
/// Documents/eden/keys, then reloads the key manager.
/// `out_message` (optional) receives a malloc'd human-readable summary.
LichiiKeyResult lichii_content_install_keys(const char* const* paths, int count,
                                            char** out_message);

/// {"prod_keys":bool,"title_keys":bool,"header_key":bool,"master_keys":N,"title_key_count":N,
///  "usable":bool}
char* lichii_content_key_status_json(void);

/// Deletes prod.keys and title.keys from the keys folder.
bool lichii_content_remove_keys(void);

// ---- Firmware ------------------------------------------------------------------------------

typedef enum {
    LICHII_FIRMWARE_OK = 0,
    LICHII_FIRMWARE_NO_KEYS = 1,
    LICHII_FIRMWARE_NO_NCAS = 2,
    LICHII_FIRMWARE_BAD_ZIP = 3,
    LICHII_FIRMWARE_DELETE_FAILED = 4,
    LICHII_FIRMWARE_COPY_FAILED = 5,
    LICHII_FIRMWARE_VERIFY_FAILED = 6,
    LICHII_FIRMWARE_CANCELLED = 7,
    LICHII_FIRMWARE_BUSY = 8, // a game is running
} LichiiFirmwareResult;

/// Installs firmware from a .zip or a folder of .nca files (searched recursively), replacing
/// nand/system/Contents/registered, then verifies the NCAs.
LichiiFirmwareResult lichii_content_install_firmware(const char* path,
                                                     LichiiContentProgress progress, void* user,
                                                     char** out_message);

/// {"present":bool,"version":"20.1.0","title":"...","corrupted":bool}
char* lichii_content_firmware_status_json(void);

bool lichii_content_remove_firmware(void);

// ---- Updates and DLC -----------------------------------------------------------------------

typedef enum {
    LICHII_INSTALL_OK = 0,
    LICHII_INSTALL_OVERWRITE = 1,
    LICHII_INSTALL_FAILED = 2,
    LICHII_INSTALL_BASE_GAME = 3, // base games are not installed to NAND
    LICHII_INSTALL_NOT_FOUND = 4,
    LICHII_INSTALL_WRONG_TYPE = 5, // only .nsp/.nsz
    LICHII_INSTALL_NO_KEYS = 6,
    LICHII_INSTALL_WRONG_GAME = 7, // update/DLC belongs to another game
    LICHII_INSTALL_CANCELLED = 8,
} LichiiInstallResult;

/// Installs an update or DLC package (.nsp/.nsz) into the user NAND. With a non-zero
/// `program_id` the package must belong to that game.
LichiiInstallResult lichii_content_install_nsp(const char* path, uint64_t program_id,
                                               LichiiContentProgress progress, void* user);

/// Base title id of the content in an .nsp (0 when unreadable).
uint64_t lichii_content_nsp_program_id(const char* path);

/// Everything that patches a game, in PatchManager order:
/// [{"name","toggle_key","version","type":"update|dlc|mod|cheat|builtin","enabled":bool,
///   "source":"nand|sdmc|external|packed|unknown","title_id":"hex","location","removable":bool}]
/// `game_path` (optional) lets packed updates inside XCIs show up.
char* lichii_content_list_addons_json(uint64_t program_id, const char* game_path);

/// Each installed DLC: [{"title_id":"hex","index":N,"source":"nand|sdmc|external","removable"}]
char* lichii_content_list_dlc_json(uint64_t program_id);

/// Enables/disables an add-on by its toggle_key (Settings::disabled_addons) and saves the config.
bool lichii_content_set_addon_enabled(uint64_t program_id, const char* toggle_key, bool enabled);

/// A built-in patch's per-game choice: 0 follows the global setting of its kind, 1 on, 2 off.
/// Saves the config. A "builtin" entry of the add-on list carries "builtin": {"kind","game",
/// "description","effect","choice":"global|on|off","source":"game|global|default",
/// "resolved_enabled","included_in"}.
bool lichii_content_set_builtin_patch_choice(uint64_t program_id, const char* name, int choice);

/// Names of the games that ship built-in patches: ["The Legend of Zelda: Ocarina of Time"].
char* lichii_content_builtin_patch_games_json(void);

bool lichii_content_remove_update(uint64_t program_id);
bool lichii_content_remove_dlc(uint64_t dlc_title_id);
int lichii_content_remove_all_dlc(uint64_t program_id);

// ---- Mods ----------------------------------------------------------------------------------

/// Documents/eden/load/<PROGRAMID> (created): one subfolder per mod with exefs/, romfs/,
/// romfslite/, romfs_ext/ or cheats/.
char* lichii_content_mod_dir(uint64_t program_id);

typedef enum {
    LICHII_MOD_OK = 0,
    LICHII_MOD_NOT_FOUND = 1,
    LICHII_MOD_NO_MOD = 2, // no exefs/romfs/cheats folder inside
    LICHII_MOD_FAILED = 3,
    LICHII_MOD_BAD_ZIP = 4,
} LichiiModResult;

/// Installs mods from a folder or .zip. Every folder inside that holds exefs/romfs/romfslite/
/// romfs_ext/cheats becomes one mod (named after that folder, or `name` when given and only one
/// mod is found). `out_count` receives the number of mods installed.
LichiiModResult lichii_content_install_mod(uint64_t program_id, const char* path,
                                           const char* name, int* out_count);

/// Removes a mod folder (or a cheat_*.txt file) from the load directory.
bool lichii_content_remove_mod(uint64_t program_id, const char* name);

// ---- Cheats --------------------------------------------------------------------------------

/// [{"source":"<mod folder or cheat_ file>","source_enabled":bool,"file":"<build id>.txt",
///   "cheats":[{"name","enabled":bool,"opcodes":N}]}]
/// Cheats are read for every build id file, so they show before the game has run once.
char* lichii_content_list_cheats_json(uint64_t program_id);

/// Toggles one cheat (stored as "cheat:<source>/<name>" in disabled_addons; the cheat engine
/// skips it on the next boot). Use lichii_content_set_addon_enabled with the source name to
/// toggle a whole file.
bool lichii_content_set_cheat_enabled(uint64_t program_id, const char* source, const char* name,
                                      bool enabled);

/// Copies a cheat .txt (Atmosphere format) to load/<PROGRAMID>/cheat_<name>.txt. Fails when the
/// file doesn't parse.
bool lichii_content_install_cheat_file(uint64_t program_id, const char* path);

// ---- Saves ---------------------------------------------------------------------------------

/// The game's save folder for the current user (or the device save when the game declares
/// only that). `game_path` (optional) lets the NACP decide; NULL assumes an account save.
char* lichii_content_save_dir(uint64_t program_id, const char* game_path);

/// {"path","exists":bool,"size":bytes,"files":N}
char* lichii_content_save_info_json(uint64_t program_id, const char* game_path);

/// Writes the save folder to `zip_path`.
bool lichii_content_export_save(uint64_t program_id, const char* game_path, const char* zip_path,
                                char** out_message);

/// Replaces the save folder with a .zip (a single top-level folder inside is unwrapped). The old
/// save is kept as <save>.bak until the import succeeds.
bool lichii_content_import_save(uint64_t program_id, const char* game_path, const char* zip_path,
                                char** out_message);

bool lichii_content_delete_save(uint64_t program_id, const char* game_path);

// ---- Logs ----------------------------------------------------------------------------------

/// [{"name","path","size":bytes,"modified":unix seconds}] for eden_log.txt(.old),
/// lichii-startup.txt, console.txt and crash dumps.
char* lichii_content_list_logs_json(void);

/// The last `max_bytes` of a text file (whole file when 0), starting at a line boundary.
char* lichii_content_read_log(const char* path, uint64_t max_bytes);

/// Zips every log file (plus config/*.ini) into `zip_path` for the share sheet.
bool lichii_content_export_logs(const char* zip_path, char** out_message);

// ---- User profiles -------------------------------------------------------------------------

/// [{"uuid","name","current":bool,"avatar":"<jpg path or empty>","index":N}]
char* lichii_content_list_profiles_json(void);

/// Returns the new profile's uuid (malloc'd) or NULL.
char* lichii_content_create_profile(const char* name);
bool lichii_content_rename_profile(const char* uuid, const char* name);
bool lichii_content_delete_profile(const char* uuid);
bool lichii_content_select_profile(const char* uuid);

/// Where the avatar JPEG of a profile lives (the guest reads it as-is; 256x256 recommended).
char* lichii_content_profile_avatar_path(const char* uuid);

/// Copies a JPEG to the profile's avatar path.
bool lichii_content_set_profile_avatar(const char* uuid, const char* jpeg_path);

// ---- System applets ------------------------------------------------------------------------

/// Path of the Mii Maker (MiiEdit, 0x0100000000001009) program NCA in the installed firmware,
/// or NULL.
char* lichii_content_mii_edit_path(void);

/// Same path, and the next boot of exactly that path starts as the MiiEdit applet instead of an
/// application. Pass the result to lichii_emu_launch.
char* lichii_content_prepare_mii_edit(void);

// ---- Guest dialogs -------------------------------------------------------------------------

typedef enum {
    LICHII_DIALOG_DISMISS = -1, // close dialog "target" (the game closed it); don't respond
    LICHII_DIALOG_ERROR = 0,    // OK only
    LICHII_DIALOG_KEYBOARD = 1, // text field + OK/Cancel
    LICHII_DIALOG_PROFILE = 2,  // pick one of `options_json` (profile list) or cancel
    LICHII_DIALOG_CONTROLLER = 3, // information + OK
    LICHII_DIALOG_TEXT_CHECK = 4, // the game rejected/confirms the text: OK (and Cancel)
} LichiiDialogKind;

/// The engine asks the UI to show a dialog. `request_json` describes it:
/// {"kind":N,"title","message","detail","initial_text","guide_text","ok_text","max_length",
///  "min_length","password":bool,"numeric":bool,"cancel":bool,"options":[...]}
/// Present it on the main thread and answer with lichii_content_dialog_respond(id, ...).
typedef void (*LichiiDialogHandler)(void* user, uint64_t id, const char* request_json);

/// Registers the UI handler. Without one, dialogs resolve with their default answer.
void lichii_content_set_dialog_handler(LichiiDialogHandler handler, void* user);

/// Answers dialog `id`. `accepted` false means cancel; `text` is the keyboard text or the chosen
/// option index as a decimal string (profiles).
void lichii_content_dialog_respond(uint64_t id, bool accepted, const char* text);

#ifdef __cplusplus
}
#endif
