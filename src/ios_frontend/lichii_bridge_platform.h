// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// iOS platform layer for the app (C API, callable from Swift): device facts and per-device
// default settings, memory budgets, lifecycle (background, audio session, idle timer), thermal
// state and shader cache management. Implemented in lichii_bridge_platform.cpp and
// lichii_platform_ios.mm (target lichii_platform). See docs/ios-platform.md.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ------------------------------------------------------------------------------------- setup

/// Call once on the main thread at app start, before the Vulkan library is opened (MoltenVK
/// reads its options then) and before a game boots. Installs the lifecycle, memory warning,
/// thermal, power and audio session observers; sets the GPU dependent MoltenVK options; sizes
/// the cache budgets from this process's memory limit; keeps the caches out of iCloud backups;
/// applies opt-in experiments. Later calls do nothing.
void lichii_platform_init(void);

// ------------------------------------------------------------------------- device and defaults

typedef struct LichiiDeviceInfo {
    char machine[32];           ///< "iPhone16,2"
    char gpu_name[64];          ///< MTLDevice.name, "Apple A17 Pro GPU"
    char chip[16];              ///< "A17", "M2" (from the GPU name), "" when unknown
    int gpu_family;             ///< Highest MTLGPUFamilyAppleN: 7 A14/M1, 8 A15/A16/M2, 9 A17/A18/M3/M4, 10 A19/M5
    int argument_buffers_tier;  ///< MTLDevice.argumentBuffersSupport + 1 (1 or 2)
    int is_ipad;
    int is_m_series;
    int os_major;
    int os_minor;
    int max_fps;                ///< UIScreen.maximumFramesPerSecond
    uint64_t physical_memory_mib;
    uint64_t available_memory_mib;        ///< os_proc_available_memory() now
    uint64_t launch_available_memory_mib; ///< os_proc_available_memory() at lichii_platform_init
    uint64_t footprint_mib;     ///< phys_footprint now (what jetsam compares with the limit)
    int increased_memory_limit; ///< The entitlement in our signature: 1, 0, -1 unknown
    int extended_virtual_addressing; ///< Same for extended-virtual-addressing
    int va_reserve_4g;          ///< A 4 GiB PROT_NONE reservation succeeded
    int va_reserve_16g;         ///< A 16 GiB one succeeded (the guest needs large reservations)
    uint64_t va_reserve_max_gib; ///< Largest single PROT_NONE reservation at launch, in GiB
    int txm;                    ///< Trusted Execution Monitor (JIT needs StikDebug's script)
    int txm_from_device_tree;   ///< ... known from the device tree rather than the model table
    int low_power_mode;
    int thermal_state;          ///< 0 nominal, 1 fair, 2 serious, 3 critical
} LichiiDeviceInfo;

/// Fills `out`; returns 0 before lichii_platform_init.
int lichii_platform_device_info(LichiiDeviceInfo* out);

typedef enum LichiiDeviceTier {
    LICHII_TIER_UNSUPPORTED = 0, ///< under 6 GB or older than A15: games will likely be killed
    LICHII_TIER_LOW = 1,         ///< 6 GB (A15/A16 phones)
    LICHII_TIER_MID = 2,         ///< 8 GB (A17 Pro/A18 phones, 8 GB M iPads)
    LICHII_TIER_HIGH = 3,        ///< 12 GB (A19 Pro)
    LICHII_TIER_ULTRA = 4,       ///< 16 GB+ M iPads
} LichiiDeviceTier;

typedef struct LichiiDeviceDefaults {
    int tier;                  ///< LichiiDeviceTier
    char tier_name[16];
    int resolution_setup;      ///< Settings::ResolutionSetup index (0 = 1/4x ... 3 = 1x ...)
    float resolution_scale;    ///< The same as a factor
    int frame_gen;             ///< Frame generation on by default
    int frame_gen_capable;     ///< Offer frame generation in the UI
    int gpu_accuracy;          ///< 0 Low, 1 High
    int async_shaders;
    int docked;
    uint32_t texture_cache_expected_mib;
    uint32_t texture_cache_critical_mib;
    uint32_t buffer_cache_collect_mib;
    uint32_t memory_low_water_mib; ///< Caches collect below this much memory left
    char note[192];            ///< One line for the UI, may be empty
} LichiiDeviceDefaults;

/// The defaults for this device; returns 0 before lichii_platform_init.
int lichii_platform_device_defaults(LichiiDeviceDefaults* out);

typedef enum LichiiApplyDefaultsMode {
    LICHII_DEFAULTS_FIRST_LAUNCH = 0, ///< Once per device profile version (marker in the config dir)
    LICHII_DEFAULTS_FORCE = 1,        ///< Now, overwriting the user's values ("Reset to device defaults")
    LICHII_DEFAULTS_UNCHANGED = 2,    ///< Only settings still at their built-in default, every call
} LichiiApplyDefaultsMode;

/// Writes the defaults into the global settings. Returns 1 when anything was applied: save the
/// configuration afterwards.
int lichii_platform_apply_device_defaults(int mode);

// ---------------------------------------------------------------------------------- lifecycle

typedef enum LichiiPlatformEvent {
    LICHII_PLATFORM_EVENT_WILL_RESIGN_ACTIVE = 0, ///< Pause emulation now (Control Center, calls, app switcher)
    LICHII_PLATFORM_EVENT_DID_ENTER_BACKGROUND = 1, ///< Caches are being written (in a background task)
    LICHII_PLATFORM_EVENT_WILL_ENTER_FOREGROUND = 2,
    LICHII_PLATFORM_EVENT_DID_BECOME_ACTIVE = 3,  ///< Resume, unless the user paused
    LICHII_PLATFORM_EVENT_MEMORY_WARNING = 4,     ///< Caches already told to collect
    LICHII_PLATFORM_EVENT_THERMAL_STATE = 5,      ///< See lichii_platform_thermal_advice
    LICHII_PLATFORM_EVENT_LOW_POWER_MODE = 6,
    LICHII_PLATFORM_EVENT_AUDIO_INTERRUPTION_BEGAN = 7,
    LICHII_PLATFORM_EVENT_AUDIO_INTERRUPTION_ENDED = 8, ///< Session reactivated
    LICHII_PLATFORM_EVENT_AUDIO_SERVICES_RESET = 9,     ///< Session reconfigured
    LICHII_PLATFORM_EVENT_WILL_TERMINATE = 10,          ///< Caches written before this returns
} LichiiPlatformEvent;

/// Called on the main thread for each LichiiPlatformEvent. The platform layer does its own part
/// (cache writes, memory pressure, audio session) before calling it; pausing and resuming the
/// emulation is the handler's.
typedef void (*LichiiPlatformEventFn)(int event, void* user);
void lichii_platform_set_event_handler(LichiiPlatformEventFn handler, void* user);

/// Writes the pipeline caches out in a UIKit background task (asynchronous). Done automatically
/// on DID_ENTER_BACKGROUND; call it also before stopping a game.
void lichii_platform_flush_caches(void);

/// While playing: idle timer off (no auto-lock) and the audio session active.
void lichii_platform_set_playing(int playing);

/// Audio session: playback category, optionally mixing with other apps' audio, and the
/// preferred hardware buffer (milliseconds, 0 = keep; 20 by default). Call before audio starts.
void lichii_platform_configure_audio(int mix_with_others, double io_buffer_ms);

// ----------------------------------------------------------------------------- thermal, power

typedef struct LichiiThermalAdvice {
    int thermal_state;     ///< 0 nominal ... 3 critical
    int low_power_mode;
    int disable_frame_gen; ///< Turn frame generation off
    int max_fps;           ///< Cap presentation to this rate, 0 = no cap
    int lower_resolution;  ///< Suggest a lower resolution (needs a restart of the game)
    int suggest_pause;     ///< Critical: iOS throttles hard and may end the app
    char message[160];     ///< For a banner, empty when nothing to say
} LichiiThermalAdvice;

int lichii_platform_thermal_state(void);
int lichii_platform_low_power_mode(void);
void lichii_platform_thermal_advice(LichiiThermalAdvice* out);

// ------------------------------------------------------------------------------------- memory

uint64_t lichii_platform_available_memory_mib(void);
uint64_t lichii_platform_footprint_mib(void);
/// Ask the caches to collect now (as a memory warning does).
void lichii_platform_trim_memory(void);

// ------------------------------------------------------------------------------------ display

/// UIScreen.maximumFramesPerSecond (also given to the frame pacer).
int lichii_platform_max_fps(void);

// ------------------------------------------------------------------------------ shader caches

typedef enum LichiiShaderStage {
    LICHII_SHADER_IDLE = -1,
    LICHII_SHADER_PREPARE = 0,
    LICHII_SHADER_BUILD = 1,
    LICHII_SHADER_COMPLETE = 2,
} LichiiShaderStage;

/// The front end forwards the disk cache loader's progress here (VideoCore's
/// DiskResourceLoadCallback); the UI reads it or registers a handler.
void lichii_platform_note_shader_progress(int stage, uint64_t done, uint64_t total);
/// Returns the stage; done/total may be null.
int lichii_platform_shader_progress(uint64_t* done, uint64_t* total);
void lichii_platform_reset_shader_progress(void);
/// Called on the loading thread; keep it short (dispatch to the main queue).
typedef void (*LichiiShaderProgressFn)(int stage, uint64_t done, uint64_t total, void* user);
void lichii_platform_set_shader_progress_handler(LichiiShaderProgressFn handler, void* user);

/// Bytes in the shader caches of `title_id` (0: every title).
uint64_t lichii_platform_shader_cache_bytes(uint64_t title_id);
/// Deletes the shader caches of `title_id` (0: every title). Returns the files removed, -1 on
/// error. Not while that game runs.
int lichii_platform_clear_shader_cache(uint64_t title_id);

// -------------------------------------------------------------------------------- experiments

/// Opt-in switches, kept in the config dir and applied at the next launch:
/// "no_footprint": ask for guest memory out of the process footprint (iOS 26 refuses it without
/// a private entitlement: KERN_NO_ACCESS). "untagged_guest": guest memory counted per mapping, as
/// before ledger tagging, with zero trimming. "no_zero_trim": no zero page trimming.
/// "chunked_guest": guest memory in purgeable 2 MiB objects, the all-zero ones given back.
void lichii_platform_set_experiment(const char* name, int enabled);
int lichii_platform_experiment(const char* name);

#ifdef __cplusplus
}
#endif
