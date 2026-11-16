// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// The one place the apps' names and help texts are written (ui_catalog.h). Each entry:
//   Setting    {"config_key", label, help[, platforms that list it[, why the others don't]]}
//   CHOICE     (Enum, Value, label[, platforms]): one value of a setting's list, in list order
//   Text       {"id", text}: page, section and footer names, dialogs, overlay lines
//   Experiment {"id", "section", label}: a switch of the iOS app's Memory & Experiments
// A text can name another entry with {setting:key}, {choice:key.Value}, {exp:id} or {text:id}, so
// it follows a rename. tools/ui/catalog_check.py fails when an app writes one of these texts
// itself, or a setting or experiment has no entry here.

#include <algorithm>
#include <unordered_map>
#include <vector>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "common/settings_enums.h"
#include "common/ui_catalog.h"

#ifndef QT_TRANSLATE_NOOP
#define QT_TRANSLATE_NOOP(scope, text) text
#endif

namespace UiCatalog {

namespace {

#define CHOICE(ENUM, VALUE, LABEL)                                                               \
    Choice{&Settings::EnumMetadata<Settings::ENUM>::Index, static_cast<u32>(Settings::ENUM::VALUE),  \
           #VALUE, LABEL}
#define CHOICE_ON(ENUM, VALUE, LABEL, SHOWN)                                                      \
    Choice{&Settings::EnumMetadata<Settings::ENUM>::Index, static_cast<u32>(Settings::ENUM::VALUE),  \
           #VALUE, LABEL, SHOWN}

// clang-format off
const Setting kSettings[] = {
    {"cabinet_applet_mode",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Amiibo Editor"),
     nullptr,
     Desktop | IOS},
    {"controller_applet_mode",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Controller Configuration"),
     nullptr,
     Desktop | IOS},
    {"data_erase_applet_mode",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Data Erase"),
     nullptr,
     Desktop | IOS},
    {"error_applet_mode",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Error"),
     nullptr,
     Desktop | IOS},
    {"net_connect_applet_mode",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Net Connect"),
     nullptr,
     Desktop | IOS},
    {"player_select_applet_mode",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Player Select"),
     nullptr,
     Desktop | IOS},
    {"swkbd_applet_mode",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Software Keyboard"),
     nullptr,
     Desktop | IOS},
    {"mii_edit_applet_mode",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Mii Edit"),
     nullptr,
     Desktop | IOS},
    {"web_applet_mode",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Online Web"),
     nullptr,
     Desktop | IOS},
    {"shop_applet_mode",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Shop"),
     nullptr,
     Desktop | IOS},
    {"photo_viewer_applet_mode",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Photo Viewer"),
     nullptr,
     Desktop | IOS},
    {"offline_web_applet_mode",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Offline Web"),
     nullptr,
     Desktop | IOS},
    {"login_share_applet_mode",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Login Share"),
     nullptr,
     Desktop | IOS},
    {"wifi_web_auth_applet_mode",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Wi-Fi Web Auth"),
     nullptr,
     Desktop | IOS},
    {"my_page_applet_mode",
     QT_TRANSLATE_NOOP("ConfigurationShared", "My Page"),
     nullptr,
     Desktop | IOS},
    {"enable_overlay",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable Overlay Applet"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enables Horizon's built-in overlay applet. Press and "
                                              "hold the home button for 1 second to show it."),
     Desktop | IOS},
    {"output_engine",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Audio Output Engine"),
     nullptr,
     Desktop,
     QT_TRANSLATE_NOOP("ConfigurationShared", "Lichii plays audio through SDL on iOS.")},
    {"output_device",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Audio Output Device"),
     nullptr,
     Desktop},
    {"input_device",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Audio Input Device"),
     nullptr,
     Desktop},
    {"audio_muted",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Mute Audio"),
     nullptr,
     Desktop | IOS},
    {"volume",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Volume"),
     nullptr,
     Desktop | IOS},
    {"dump_audio_commands",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Dump Audio Commands To Console**"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable this to output the latest generated audio "
                                              "command list to the console. Only affects games using "
                                              "the audio renderer."),
     Desktop},
    {"muteWhenInBackground",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Mute audio when in background"),
     nullptr,
     Desktop},
    {"use_multi_core",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Multicore CPU Emulation"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "This option increases CPU emulation thread use from 1 "
                                              "to the maximum of 4.\n"
                                              "This is mainly a debug option and shouldn't be disabled."),
     Desktop | IOS},
    {"memory_layout_mode",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Memory Layout"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "The emulated RAM of Switch 1 games (Switch 2 games "
                                              "always get 12 GB). More can let HD texture mods load; "
                                              "it doesn't change performance. 6, 10 and 12 GB are "
                                              "unsafe for some games."),
     Desktop | IOS},
    {"use_speed_limit",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Limit Emulation Speed"),
     nullptr,
     IOS},
    {"current_speed_mode",
     nullptr,
     nullptr,
     Nowhere},
    {"speed_limit",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Speed Limit"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Controls the game's maximum rendering speed, but it's "
                                              "up to each game if it runs faster or not.\n"
                                              "200% for a 30 FPS game is 60 FPS, and for a 60 FPS game "
                                              "it will be 120 FPS.\n"
                                              "Disabling it means unlocking the framerate to the "
                                              "maximum your PC can reach."),
     Desktop | IOS},
    {"turbo_speed_limit",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Turbo Speed"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "When the Turbo Speed hotkey is pressed, the speed will "
                                              "be limited to this percentage."),
     Desktop | IOS},
    {"slow_speed_limit",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Slow Speed"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "When the Slow Speed hotkey is pressed, the speed will "
                                              "be limited to this percentage."),
     Desktop | IOS},
    {"sync_core_speed",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Synchronize Core Speed"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Synchronizes CPU core speed with the game's maximum "
                                              "rendering speed to boost FPS without affecting game "
                                              "speed (animations, physics, etc.).\n"
                                              "Can help reduce stuttering at lower framerates."),
     Desktop | IOS},
    {"cpu_accuracy",
     QT_TRANSLATE_NOOP("ConfigurationShared", "CPU Accuracy"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Change the accuracy of the emulated CPU (for debugging "
                                              "only)."),
     Desktop | IOS},
    {"cpu_backend",
     QT_TRANSLATE_NOOP("ConfigurationShared", "CPU Backend"),
     nullptr,
     Desktop,
     QT_TRANSLATE_NOOP("ConfigurationShared", "iOS runs games with the Dynarmic JIT only.")},
    {"fast_cpu_time",
     QT_TRANSLATE_NOOP("ConfigurationShared", "CPU Clocks"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Raises the clock the emulated CPU reports, which "
                                              "removes some FPS limiters.\n"
                                              "Weaker CPUs may see reduced performance, and certain "
                                              "games may behave improperly."),
     Desktop | IOS},
    {"use_custom_cpu_ticks",
     nullptr,
     nullptr,
     Nowhere},
    {"cpu_ticks",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Custom CPU Ticks"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Set a custom value of CPU ticks. Higher values can "
                                              "increase performance, but may cause deadlocks. A range "
                                              "of 77-21000 is recommended."),
     Desktop | IOS},
    {"guest_wait_backoff",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Sleep in Long Guest Waits"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "A game thread that only waits (yields or reads the "
                                              "clock) for 2 ms sleeps briefly instead of keeping a CPU "
                                              "core busy, leaving the cores to the rest of the game."),
     Desktop | IOS},
    {"cpuopt_compact_jit_code",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Compact JIT Code"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "About half as much translated CPU code, so the code "
                                              "cache holds twice as much: where the host grants less "
                                              "address space than the game uses (phones), memory "
                                              "accesses outside it take the slow path instead of an "
                                              "inline page table walk.\n"
                                              "Applies the next time the game starts."),
     Desktop | IOS},
    {"cpuopt_unsafe_host_mmu",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable Host MMU Emulation (fastmem)"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "This optimization speeds up memory accesses by the "
                                              "guest program.\n"
                                              "Enabling it causes guest memory reads/writes to be done "
                                              "directly into memory and make use of Host's MMU.\n"
                                              "Disabling this forces all memory accesses to use "
                                              "Software MMU Emulation."),
     Desktop | IOS},
    {"cpuopt_unsafe_unfuse_fma",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Unfuse FMA (improve performance on CPUs without FMA)"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "This option improves speed by reducing accuracy of "
                                              "fused-multiply-add instructions on CPUs without native "
                                              "FMA support."),
     Desktop | IOS},
    {"cpuopt_unsafe_reduce_fp_error",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Faster FRSQRTE and FRECPE"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "This option improves the speed of some approximate "
                                              "floating-point functions by using less accurate native "
                                              "approximations."),
     Desktop | IOS},
    {"cpuopt_unsafe_ignore_standard_fpcr",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Faster ASIMD instructions (32 bits only)"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "This option improves the speed of 32 bits ASIMD "
                                              "floating-point functions by running with incorrect "
                                              "rounding modes."),
     Desktop | IOS},
    {"cpuopt_unsafe_inaccurate_nan",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Inaccurate NaN handling"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "This option improves speed by removing NaN checking.\n"
                                              "Please note this also reduces accuracy of certain "
                                              "floating-point instructions."),
     Desktop | IOS},
    {"cpuopt_unsafe_fastmem_check",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Disable address space checks"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "This option improves speed by eliminating a safety "
                                              "check before every memory operation.\n"
                                              "Disabling it may allow arbitrary code execution."),
     Desktop | IOS},
    {"cpuopt_unsafe_ignore_global_monitor",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Ignore global monitor"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "This option improves speed by relying only on the "
                                              "semantics of cmpxchg to ensure safety of exclusive "
                                              "access instructions.\n"
                                              "Please note this may result in deadlocks and other race "
                                              "conditions."),
     Desktop | IOS},
    {"backend",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Graphics API"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "{choice:backend.Vulkan} is the renderer games are made "
                                              "to work with (through MoltenVK on Apple devices). "
                                              "{choice:backend.Metal} (Apple only, experimental) is "
                                              "Lichii's own Metal renderer, still being built: it runs "
                                              "draws, compute, occlusion queries, tessellation and "
                                              "mesh shaders, but not geometry shaders or transform "
                                              "feedback, so expect missing or wrong graphics (a "
                                              "dark-blue screen means no frame reached it). OpenGL "
                                              "(desktop) and {choice:backend.Null} are for testing.\n"
                                              "Applies the next time a game starts."),
     Desktop | IOS},
    {"vulkan_device",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Device"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "This setting selects the GPU to use (Vulkan only)."),
     Desktop},
    {"resolution_setup",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Resolution"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "The size games draw at, as a multiple of their own (1x "
                                              "is 720p handheld, 1080p docked). Games with their own "
                                              "Resolution (Ocarina of Time) use that instead. Higher "
                                              "is sharper and costs GPU time, memory and battery; "
                                              "below 1x is experimental and can show artifacts. NVIDIA "
                                              "DLSS, MetalFX and AMD FSR 3.1 as the {setting:upscaler} "
                                              "need 1x."),
     Desktop | IOS},
    {"scaling_filter",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Screen Filter"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "The last step: how the finished picture is scaled to "
                                              "the window or screen."),
     Desktop | IOS},
    {"upscaler",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Upscaler"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "What takes the place of a game's own upscaler. "
                                              "{choice:upscaler.Auto}: Ocarina of Time's own upscale "
                                              "pass, AMD FSR 3.1 in games that call DLSS (Fire Emblem: "
                                              "Fortune's Weave). {choice:upscaler.Game}: also runs "
                                              "those games' own DLSS network, translated for this GPU "
                                              "(AMD FSR 3.1 still draws while it is checked or fails "
                                              "Lichii's checks). {choice:upscaler.Fsr3}, "
                                              "{choice:upscaler.Dlss} (GeForce RTX on Windows, "
                                              "NVIDIA's files from the download button) and "
                                              "{choice:upscaler.MetalFx} (Apple) replace both and draw "
                                              "at 1x; in Ocarina of Time they need a fixed "
                                              "{setting:oot_resolution}, so "
                                              "{choice:oot_resolution.Game} draws at 75% handheld "
                                              "(100% docked) with them, and with "
                                              "{choice:upscaler.Dlss} the game's Resolution picks the "
                                              "DLSS mode.\n"
                                              "Applies the next time a game starts."),
     Desktop | IOS},
    {"nx2_frame_rate_limit",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Frame Rate Limit (Switch 2)"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Rate of the Switch 2 VSync event that games start their "
                                              "frames on (default 60 Hz).\n"
                                              "Above 60 Hz most games run too fast. Ocarina of Time "
                                              "follows it only with its frame rate set to Unlocked, "
                                              "and keeps its speed there.\n"
                                              "Takes effect at once, also while a game runs."),
     Desktop | IOS},
    {"fsr_sharpening_slider",
     QT_TRANSLATE_NOOP("ConfigurationShared", "FSR Sharpness"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Strength of RCAS (AMD's contrast-adaptive sharpening) "
                                              "on the upscaler used in place of a game's DLSS (AMD FSR "
                                              "3.1); 0% turns it off.\n"
                                              "Also sets the sharpening of the AMD FSR 1 and SGSR "
                                              "window adapting filters."),
     Desktop | IOS},
    {"nvidia_dlss_preset",
     QT_TRANSLATE_NOOP("ConfigurationShared", "DLSS Preset"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "The DLSS model. K is the tested default; L and M are "
                                              "NVIDIA's newer models, J the older one. "
                                              "{choice:nvidia_dlss_preset.Default} lets NVIDIA choose "
                                              "per quality mode."),
     Windows,
     QT_TRANSLATE_NOOP("ConfigurationShared", "NVIDIA DLSS needs an NVIDIA RTX GPU on Windows.")},
    {"moltenvk_driver",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Graphics Driver"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Which MoltenVK runs Vulkan on Metal. "
                                              "{choice:moltenvk_driver.Mesh} draws mesh shaders "
                                              "(Ocarina of Time's foliage) with Metal's own mesh "
                                              "shaders; {choice:moltenvk_driver.Legacy} emulates them "
                                              "with compute. Lichii uses the legacy one when the new "
                                              "one doesn't load.\n"
                                              "Applies the next time Lichii starts."),
     Apple,
     QT_TRANSLATE_NOOP("ConfigurationShared", "MoltenVK runs on macOS and iOS only.")},
    {"mesh_direct_outputs",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Mesh Shader Direct Outputs"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "{choice:moltenvk_driver.Mesh} only. On: mesh shaders "
                                              "write their vertices straight to Metal's outputs, which "
                                              "keeps Ocarina of Time's foliage within Metal's limits. "
                                              "Off: they copy them at the end, as builds before "
                                              "November 13 did (some foliage is then drawn with "
                                              "compute).\n"
                                              "Applies the next time the game starts."),
     Apple,
     QT_TRANSLATE_NOOP("ConfigurationShared", "MoltenVK runs on macOS and iOS only.")},
    {"anti_aliasing",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Anti-Aliasing"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Smooths jagged edges after the game has drawn its "
                                              "frame. SMAA looks best; FXAA keeps a low resolution "
                                              "steadier."),
     Desktop | IOS},
    {"oot_anti_aliasing",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Sharpening"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Game: the game's own anti-aliasing network, which "
                                              "sharpens heavily.\n"
                                              "No sharpening: the image as the game renders it, "
                                              "without the network (faster; pairs well with a higher "
                                              "resolution). Has no effect while an upscaler replaces "
                                              "the game's.\n"
                                              "Takes effect when the game starts."),
     Desktop | IOS},
    {"oot_movie_upscale",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Movies"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "How the title screen and cutscene videos are scaled "
                                              "up.\n"
                                              "Bilinear: a plain, fast stretch.\n"
                                              "Game's own upscaler: the game's video upscaler, which "
                                              "removes banding and sharpens. Fast on NVIDIA RTX, slow "
                                              "on GPUs that emulate its math.\n"
                                              "Automatic: the game's upscaler where the GPU runs it "
                                              "natively, bilinear elsewhere."),
     Desktop | IOS},
    {"oot_graphics_preset",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Graphics Preset"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Shadows, reflections and effects as the game sets them "
                                              "in its own parameter files. "
                                              "{choice:oot_graphics_preset.PhoneLow}: a small "
                                              "reflection probe, no bloom or depth of field, one small "
                                              "sun shadow, thinner grass, 50% resolution and Quarter "
                                              "textures (about a third fewer render passes). "
                                              "{choice:oot_graphics_preset.PhoneMedium}: a small "
                                              "reflection probe and two small sun shadows, 56% and "
                                              "Half. {choice:oot_graphics_preset.Console}: the game as "
                                              "shipped. {choice:oot_graphics_preset.High}: sharper sun "
                                              "shadows.\n"
                                              "Picking one sets {setting:oot_resolution} and "
                                              "{setting:texture_quality} too; applies the next time "
                                              "the game starts."),
     Desktop | IOS},
    {"oot_resolution",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Resolution"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "The size the game draws at, its one resolution setting "
                                              "(the global Resolution doesn't apply). "
                                              "{choice:oot_resolution.Game}: the game's own, lower "
                                              "(down to 480p) when it thinks the GPU is busy. A "
                                              "percentage: a fixed share of full size (100% is "
                                              "1920x1080 handheld and 2560x1440 docked; handheld 75% "
                                              "is 1440x810, 56% 1080x608, 50% 960x540, 40% 768x432, "
                                              "33% 640x360; docked, 33% draws as 40%): less GPU work "
                                              "and memory, a softer picture. This is the 3D scene "
                                              "only: the HUD and menus stay at full sharpness. "
                                              "150-300%: full size, scaled up by the emulator.\n"
                                              "With NVIDIA DLSS as the {setting:upscaler} this is "
                                              "DLSS's input to 4K: 100% Quality, 87% Balanced, 75% "
                                              "Performance, 50% Ultra Performance (above 100% counts "
                                              "as 100%). With MetalFX Temporal, AMD FSR 3.1 or NVIDIA "
                                              "DLSS as the upscaler, {choice:oot_resolution.Game} "
                                              "draws at 75% handheld (100% docked): they need a fixed "
                                              "size.\n"
                                              "Applies the next time the game starts."),
     Desktop | IOS},
    {"oot_frame_rate",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Frame Rate"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "{choice:oot_frame_rate.Fps30} saves power and heat; "
                                              "{choice:oot_frame_rate.Fps60} is the game's normal "
                                              "rate; {choice:oot_frame_rate.Unlocked} goes up to "
                                              "{setting:frame_rate_limit}. The game keeps its normal "
                                              "speed at each (its time step follows real time).\n"
                                              "Takes effect at once."),
     Desktop | IOS},
    {"mkw_frame_rate",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Frame Rate"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "{choice:mkw_frame_rate.Game}: no change. "
                                              "{choice:mkw_frame_rate.Fps30}: the game's native 30 FPS "
                                              "mode at normal speed. {choice:mkw_frame_rate.Dynamic}: "
                                              "the game moves between 30 and 60 FPS with whole-frame "
                                              "timing to keep normal speed (experimental).\n"
                                              "Applies the next time the game starts."),
     Desktop | IOS},
    {"patch_performance",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Occlusion Off"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Game patch with a trade-off (currently Ocarina of Time: "
                                              "Occlusion Off).\n"
                                              "Auto and On: the game stops issuing per-object "
                                              "occlusion queries (fewer query stalls and pop-in "
                                              "problems; hidden objects are drawn, so more GPU work). "
                                              "Off: the game's own queries.\n"
                                              "The game's fixes with no visible effect (native zstd, "
                                              "fast lookup, resource poll backoff, texture pool x2) "
                                              "are always on.\n"
                                              "A game's Add-Ons choice overrides this. Takes effect "
                                              "when the game starts."),
     Desktop | IOS},
    {"patch_image",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Disable Game Anti-Aliasing"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Game enhancement. Applies to every game that ships a "
                                              "patch removing its own anti-aliasing (currently Ocarina "
                                              "of Time: No NDSR anti-aliasing).\n"
                                              "Auto: on Apple devices, which lack the cooperative "
                                              "matrices the network needs, off elsewhere. On: the game "
                                              "skips its anti-aliasing and sharpening network (faster, "
                                              "softer image). Off: the game's own anti-aliasing.\n"
                                              "Not applied while MetalFX Temporal is on. A game's "
                                              "Add-Ons choice overrides this. Takes effect when the "
                                              "game starts."),
     Desktop | IOS},
    {"post_shader_chain",
     nullptr,
     nullptr,
     Nowhere},
    {"post_shader_preset",
     nullptr,
     nullptr,
     Nowhere},
    {"post_shader_enabled",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable post-processing effects"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Applies post-processing effects to the final image."),
     Desktop | IOS},
    {"fullscreen_mode",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Fullscreen Mode"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "The method used to render the window in fullscreen.\n"
                                              "Borderless offers the best compatibility with the "
                                              "on-screen keyboard that some games request for input.\n"
                                              "Exclusive fullscreen may offer better performance and "
                                              "better Freesync/Gsync support."),
     Desktop},
    {"aspect_ratio",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Aspect Ratio"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "The shape of the picture. Games are made for "
                                              "{choice:aspect_ratio.R16_9}; the others stretch it or "
                                              "need a mod to fill it. {choice:aspect_ratio.Stretch} "
                                              "fills the window. Also the shape of screenshots."),
     Desktop | IOS},
    {"use_disk_shader_cache",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Disk Shader Cache"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Keeps built shaders in storage, so the next start of a "
                                              "game stutters less. Turning it off is only for "
                                              "debugging."),
     Desktop | IOS},
    {"use_asynchronous_gpu_emulation",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Use asynchronous GPU emulation"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Uses an extra CPU thread for rendering.\n"
                                              "This option should always remain enabled."),
     Desktop | IOS},
    {"nvdec_emulation",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Video Decoding"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "How game videos are decoded: on the "
                                              "{choice:nvdec_emulation.Gpu} (fastest in most cases), "
                                              "the {choice:nvdec_emulation.Cpu}, or "
                                              "{choice:nvdec_emulation.Off} (videos show black)."),
     Desktop,
     QT_TRANSLATE_NOOP("ConfigurationShared", "Lichii decodes game videos on the CPU on iOS and sets "
                                              "this at every start.")},
    {"accelerate_astc",
     QT_TRANSLATE_NOOP("ConfigurationShared", "ASTC Decoding"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "How ASTC textures are decoded. "
                                              "{choice:accelerate_astc.Gpu}: the GPU's compute shaders "
                                              "(recommended). {choice:accelerate_astc.Cpu}: the CPU. "
                                              "{choice:accelerate_astc.CpuAsynchronous}: the CPU, as "
                                              "they are needed, which removes ASTC stutter but can "
                                              "show artifacts."),
     Desktop,
     QT_TRANSLATE_NOOP("ConfigurationShared", "Apple GPUs read ASTC textures directly; this is for "
                                              "GPUs without ASTC.")},
    {"astc_recompression",
     QT_TRANSLATE_NOOP("ConfigurationShared", "ASTC Recompression"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "GPUs without ASTC support get the textures decompressed "
                                              "to RGBA8 ({choice:astc_recompression.Uncompressed}, "
                                              "best quality), or recompressed to "
                                              "{choice:astc_recompression.Bc1} or "
                                              "{choice:astc_recompression.Bc3}, which saves VRAM and "
                                              "costs quality."),
     Desktop,
     QT_TRANSLATE_NOOP("ConfigurationShared", "Apple GPUs read ASTC textures directly; this is for "
                                              "GPUs without ASTC.")},
    {"frame_pacing_mode",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Frame Rate Target"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Controls how the emulator manages frame pacing to "
                                              "reduce stuttering and make the frame rate smoother and "
                                              "more consistent."),
     Desktop | IOS},
    {"vram_usage_mode",
     QT_TRANSLATE_NOOP("ConfigurationShared", "VRAM Usage Mode"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Selects whether the emulator should prefer to conserve "
                                              "memory or make maximum usage of available video memory "
                                              "for performance.\n"
                                              "Aggressive mode may impact performance of other "
                                              "applications such as recording software."),
     Desktop | IOS},
    {"skip_cpu_inner_invalidation",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Skip CPU Inner Invalidation"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Skips certain cache invalidations during memory "
                                              "updates, reducing CPU usage and improving latency. This "
                                              "may cause soft-crashes."),
     Desktop | IOS},
    {"use_vsync",
     QT_TRANSLATE_NOOP("ConfigurationShared", "VSync Mode"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "FIFO (VSync) does not drop frames or exhibit tearing "
                                              "but is limited by the screen refresh rate.\n"
                                              "FIFO Relaxed allows tearing as it recovers from a slow "
                                              "down.\n"
                                              "Mailbox can have lower latency than FIFO and does not "
                                              "tear but may drop frames.\n"
                                              "Immediate (no synchronization) presents whatever is "
                                              "available and can exhibit tearing."),
     Desktop | IOS},
    {"bg_red",
     nullptr,
     nullptr,
     Nowhere},
    {"bg_green",
     nullptr,
     nullptr,
     Nowhere},
    {"bg_blue",
     nullptr,
     nullptr,
     Nowhere},
    {"frame_gen",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Frame Generation"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Adds generated frames between the game's own: smoother "
                                              "motion on a high refresh rate screen, more GPU work and "
                                              "about one real frame more input lag. The game itself "
                                              "runs no faster; it works best when the game already "
                                              "holds a steady 60 FPS.\n"
                                              "Applies the next time a game starts."),
     Desktop | IOS},
    {"frame_gen_type",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Frame Generation Type"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "{choice:frame_gen_type.Fsr3}: Lichii's own frame "
                                              "interpolation from the motion between two frames (a "
                                              "flow shader, or NVIDIA's optical flow hardware on RTX "
                                              "20 and newer); needs nothing else. It is not AMD FSR "
                                              "3.\n"
                                              "{choice:frame_gen_type.Lsfg}: the shaders of "
                                              "Lossless.dll from your own copy of Lossless Scaling "
                                              "(Steam); without the file no frames are generated.\n"
                                              "{choice:frame_gen_type.MetalFx}: Apple's interpolation "
                                              "on the same motion, on macOS 26 and iOS 26, 2x only (3x "
                                              "and 4x use {choice:frame_gen_type.Fsr3})."),
     Desktop | IOS},
    {"frame_gen_multiplier",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Frame Generation Multiplier"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "How many frames are shown per real frame: 2x adds one "
                                              "generated frame, 3x two, 4x three.\n"
                                              "Ignored while Target Frame Rate is set."),
     Desktop | IOS},
    {"frame_gen_target_rate",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Frame Generation Target"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Off: use the multiplier.\n"
                                              "Above 60: generate as many frames (up to three per real "
                                              "frame) as it takes to reach this rate, in place of the "
                                              "multiplier. Set it to your display's refresh rate.\n"
                                              "Values from 1 to 60 count as Off."),
     Desktop | IOS},
    {"frame_gen_flow_scale_auto",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Auto Flow Scale"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Picks the resolution of the motion estimate from the "
                                              "rendering and window resolutions. Turn it off to set "
                                              "Flow Scale yourself."),
     Desktop | IOS},
    {"frame_gen_flow_scale",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Flow Scale"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Resolution of the motion estimate relative to the "
                                              "image. Lower is faster but blurs fast motion; higher is "
                                              "sharper but costs more GPU time.\n"
                                              "Only used while Auto Flow Scale is off."),
     Desktop | IOS},
    {"frame_gen_queue_target",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Queue Target"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Extra frames queued before display. 0 has the lowest "
                                              "latency; 1 or 2 smooth out uneven frame times at the "
                                              "cost of one or two frames of latency."),
     Desktop | IOS},
    {"frame_gen_dump_flow",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Dump Optical Flow"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "When checked, frame generation writes its optical flow "
                                              "fields to the dump folder. Very slow; for debugging "
                                              "frame generation only."),
     Desktop},
    {"sync_memory_operations",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Sync Memory Operations"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Ensures data consistency between compute and memory "
                                              "operations.\n"
                                              "This option fixes issues in games, but may degrade "
                                              "performance.\n"
                                              "Unreal Engine 4 games often see the most significant "
                                              "changes thereof."),
     Desktop | IOS},
    {"async_presentation",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable asynchronous presentation (Vulkan only)"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Slightly improves performance by moving presentation to "
                                              "a separate CPU thread."),
     Desktop | IOS},
    {"force_max_clock",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Force Maximum GPU Clocks"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Runs work in the background while waiting for graphics "
                                              "commands to keep the GPU from lowering its clock speed."),
     Desktop,
     QT_TRANSLATE_NOOP("ConfigurationShared", "Only works with AMD, NVIDIA, Intel and Qualcomm drivers.")},
    {"max_anisotropy",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Anisotropic Filtering"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Controls the quality of texture rendering at oblique "
                                              "angles.\n"
                                              "Safe to set at 16x on most GPUs."),
     Desktop | IOS},
    {"gpu_accuracy",
     QT_TRANSLATE_NOOP("ConfigurationShared", "GPU Accuracy"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Most games draw correctly with "
                                              "{choice:gpu_accuracy.Low}; some need "
                                              "{choice:gpu_accuracy.High} (particles often do), which "
                                              "is slower."),
     Desktop | IOS},
    {"dma_accuracy",
     QT_TRANSLATE_NOOP("ConfigurationShared", "DMA Accuracy"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Controls the DMA read mode.\n"
                                              "Unsafe is faster, while Safe is more stable and can fix "
                                              "issues in some games.\n"
                                              "Default follows the GPU Accuracy setting."),
     Desktop | IOS},
    {"gpu_fence_behavior",
     QT_TRANSLATE_NOOP("ConfigurationShared", "GPU Fence Behavior"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Controls the GPU fence synchronization behavior.\n"
                                              "Immediate is the fastest option, but can introduce some "
                                              "issues.\n"
                                              "Balanced offers better compatibility and may fix issues "
                                              "in some games.\n"
                                              "Accurate further improves compatibility at the cost of "
                                              "some performance.\n"
                                              "Default follows the GPU Mode setting."),
     Desktop | IOS},
    {"native_compute_write_mark",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Native compute write mark"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "How much of a Switch 2 compute shader's storage buffer "
                                              "is treated as written by the GPU after each dispatch.\n"
                                              "Auto marks what the shader's stores can reach and falls "
                                              "back to Full when that cannot be proven.\n"
                                              "Full marks the whole binding (up to 8 MiB), the "
                                              "original behaviour: it can delay the game's own writes "
                                              "to nearby memory (flickering effects) and copy far more "
                                              "data back from the GPU.\n"
                                              "64 KiB cap marks at most 64 KiB, which can miss data a "
                                              "large dispatch really wrote."),
     Desktop | IOS},
    {"use_asynchronous_shaders",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Asynchronous Shader Building"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Builds new shaders in the background: less stutter, but "
                                              "an object can be missing for a moment."),
     Desktop | IOS},
    {"fast_gpu_time",
     QT_TRANSLATE_NOOP("ConfigurationShared", "GPU Clocks"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Makes the game believe GPU work finishes faster than it "
                                              "does, so it stops lowering resolution and render "
                                              "distance to fit the Switch's clocks."),
     Desktop | IOS},
    {"gpu_unswizzle_enabled",
     QT_TRANSLATE_NOOP("ConfigurationShared", "GPU Unswizzle"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Accelerates BCn 3D texture decoding using GPU compute.\n"
                                              "Disable if experiencing crashes or graphical glitches."),
     Desktop | IOS},
    {"gpu_unswizzle_texture_size",
     QT_TRANSLATE_NOOP("ConfigurationShared", "GPU Unswizzle Max Texture Size"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Sets the maximum size (MiB) for GPU-based texture "
                                              "unswizzling.\n"
                                              "While the GPU is faster for medium and large textures, "
                                              "the CPU may be more efficient for very small ones.\n"
                                              "Adjust this to find the balance between GPU "
                                              "acceleration and CPU overhead."),
     Desktop | IOS},
    {"gpu_unswizzle_stream_size",
     QT_TRANSLATE_NOOP("ConfigurationShared", "GPU Unswizzle Stream Size"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Sets the maximum amount of texture data (in MiB) "
                                              "processed per frame.\n"
                                              "Higher values can reduce stutter during texture loading "
                                              "but may impact frame consistency."),
     Desktop | IOS},
    {"gpu_unswizzle_chunk_size",
     QT_TRANSLATE_NOOP("ConfigurationShared", "GPU Unswizzle Chunk Size"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Determines the number of depth slices processed in a "
                                              "single dispatch.\n"
                                              "Increasing this can improve throughput on high-end GPUs "
                                              "but may cause TDR or driver timeouts on weaker hardware."),
     Desktop | IOS},
    {"use_vulkan_driver_pipeline_cache",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Driver Pipeline Cache"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enables GPU vendor-specific pipeline cache.\n"
                                              "This option can improve shader loading time "
                                              "significantly in cases where the Vulkan driver does not "
                                              "store pipeline cache files internally."),
     Desktop | IOS},
    {"precache_shaders",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Precache shaders before game start"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Builds shaders on the loading screen, on every CPU "
                                              "thread, so they don't stutter the first time the game "
                                              "uses them: the compute shaders found in the game files, "
                                              "and the shaders saved by earlier play sessions that the "
                                              "normal shader loading leaves for later. The first "
                                              "launch reads the game files and takes a few minutes; "
                                              "later launches only build what is new, until the GPU "
                                              "driver changes. The loading screen has a Skip button.\n"
                                              "Ask asks at launch while the game has never been "
                                              "precached."),
     Desktop | IOS},
    {"enable_compute_pipelines",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable Compute Pipelines (Intel Vulkan Only)"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Required by some games.\n"
                                              "This setting only exists for Intel proprietary drivers "
                                              "and may crash if enabled.\n"
                                              "Compute pipelines are always enabled on all other "
                                              "drivers."),
     Desktop,
     QT_TRANSLATE_NOOP("ConfigurationShared", "Only matters for some Intel drivers on Windows.")},
    {"use_reactive_flushing",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable Reactive Flushing"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Uses reactive flushing instead of predictive flushing, "
                                              "allowing more accurate memory syncing."),
     Desktop | IOS},
    {"use_video_framerate",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Sync to framerate of video playback"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Run the game at normal speed during video playback, "
                                              "even when the framerate is unlocked."),
     Desktop | IOS},
    {"oot_fps_asked",
     nullptr,
     nullptr,
     Nowhere},
    {"frizti_settings_revision",
     nullptr,
     nullptr,
     Nowhere},
    {"texture_quality",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Texture Quality"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "{choice:texture_quality.Half} or "
                                              "{choice:texture_quality.Quarter} loads the game's large "
                                              "textures with one or two fewer detail levels: softer up "
                                              "close, much less memory (about 1.2 GB less in Ocarina "
                                              "of Time). Your game files are not changed.\n"
                                              "Applies the next time the game starts."),
     Desktop | IOS},
    {"barrier_feedback_loops",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Barrier feedback loops"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Improves rendering of transparency effects in specific "
                                              "games."),
     Desktop | IOS},
    {"enable_buffer_history",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable buffer history"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enables access to previous buffer states.\n"
                                              "This option may improve rendering quality and "
                                              "performance consistency in some games."),
     Desktop | IOS},
    {"fix_bloom_effects",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Fix bloom effects"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Removes bloom in Burnout."),
     Desktop | IOS},
    {"dyna_state",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Extended Dynamic State"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Controls the number of features that can be used in "
                                              "Extended Dynamic State.\n"
                                              "Higher states allow for more features and can increase "
                                              "performance, but may cause additional graphical issues."),
     Desktop | IOS},
    {"vertex_input_dynamic_state",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Vertex Input Dynamic State"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enables vertex input dynamic state feature for better "
                                              "quality and performance."),
     Desktop | IOS},
    {"sample_shading_fraction",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Sample Shading"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Allows the fragment shader to execute per sample in a "
                                              "multi-sampled fragment instead of once per fragment. "
                                              "Improves graphics quality at the cost of performance.\n"
                                              "Higher values improve quality but degrade performance."),
     Desktop | IOS},
    {"rng_seed",
     QT_TRANSLATE_NOOP("ConfigurationShared", "RNG Seed"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Controls the seed of the random number generator.\n"
                                              "Mainly used for speedrunning."),
     Desktop | IOS},
    {"rng_seed_enabled",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Fixed RNG Seed"),
     nullptr,
     IOS},
    {"device_name",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Console Name"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "The name of the console."),
     Desktop | IOS},
    {"program_args",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Homebrew Args"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Command-line arguments passed to homebrew at launch "
                                              "(e.g. -noglsl)."),
     Desktop},
    {"custom_rtc",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Custom RTC Date"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "This option allows to change the clock of the console.\n"
                                              "Can be used to manipulate time in games."),
     Desktop | IOS},
    {"custom_rtc_enabled",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Custom RTC Date"),
     nullptr,
     IOS},
    {"custom_rtc_offset",
     QT_TRANSLATE_NOOP("ConfigurationShared", "RTC Offset (Seconds)"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "The number of seconds from the current unix time"),
     Desktop},
    {"language_index",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Language"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "This option can be overridden when region setting is "
                                              "auto-select"),
     Desktop | IOS},
    {"region_index",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Region"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "The region of the console."),
     Desktop | IOS},
    {"time_zone_index",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Time Zone"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "The time zone of the console."),
     Desktop | IOS},
    {"sound_index",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Sound Output Mode"),
     nullptr,
     Desktop | IOS},
    {"use_docked_mode",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Console Mode"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Docked: games run as on a TV, at their TV resolution, "
                                              "which costs more GPU time and memory. Handheld: as in "
                                              "handheld mode, lighter. Games change their resolution, "
                                              "details and supported controllers with it."),
     Desktop | IOS},
    {"current_user",
     nullptr,
     nullptr,
     Nowhere},
    {"select_user_on_boot",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Prompt for user profile on boot"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Useful if multiple people use the same PC."),
     Desktop},
    {"pauseWhenInBackground",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Pause when not in focus"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Pauses emulation when focusing on other windows."),
     Desktop},
    {"confirmStop",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Confirm before stopping emulation"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Overrides prompts asking to confirm stopping the "
                                              "emulation.\n"
                                              "Enabling it bypasses such prompts and directly exits "
                                              "the emulation."),
     Desktop},
    {"hideInactiveMouse",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Hide mouse on inactivity"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Hides the mouse after 2.5s of inactivity."),
     Desktop},
    {"check_for_updates",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Check for updates"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Whether or not to check for updates upon startup."),
     Desktop},
    {"enable_gamemode",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable Gamemode"),
     nullptr,
     Linux,
     QT_TRANSLATE_NOOP("ConfigurationShared", "Feral GameMode exists on Linux only.")},
    {"gui_force_x11",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Force X11 as Graphics Backend"),
     nullptr,
     Desktop},
    {"gui_hide_backend_warning",
     nullptr,
     nullptr,
     Nowhere},
    {"microphone_volume",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Microphone Volume"),
     nullptr},
    {"cpuopt_page_tables",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable inline page tables"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "This optimization speeds up memory accesses by the "
                                              "guest program.\n"
                                              "Enabling it inlines accesses to PageTable::pointers "
                                              "into emitted code.\n"
                                              "Disabling this forces all memory accesses to go through "
                                              "the Memory::Read/Memory::Write functions."),
     Desktop | IOS},
    {"cpuopt_block_linking",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable block linking"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "This optimization avoids dispatcher lookups by allowing "
                                              "emitted basic blocks to jump directly to other basic "
                                              "blocks if the destination PC is static."),
     Desktop | IOS},
    {"cpuopt_return_stack_buffer",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable return stack buffer"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "This optimization avoids dispatcher lookups by keeping "
                                              "track potential return addresses of BL instructions. "
                                              "This approximates what happens with a return stack "
                                              "buffer on a real CPU."),
     Desktop | IOS},
    {"cpuopt_fast_dispatcher",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable fast dispatcher"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable a two-tiered dispatch system. A faster "
                                              "dispatcher written in assembly has a small MRU cache of "
                                              "jump destinations is used first. If that fails, "
                                              "dispatch falls back to the slower C++ dispatcher."),
     Desktop | IOS},
    {"cpuopt_context_elimination",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable context elimination"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enables an IR optimization that reduces unnecessary "
                                              "accesses to the CPU context structure."),
     Desktop | IOS},
    {"cpuopt_const_prop",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable constant propagation"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enables IR optimizations that involve constant "
                                              "propagation."),
     Desktop | IOS},
    {"cpuopt_misc_ir",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable miscellaneous optimizations"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enables miscellaneous IR optimizations."),
     Desktop | IOS},
    {"cpuopt_reduce_misalign_checks",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable misalignment check reduction"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "When enabled, a misalignment is only triggered when an "
                                              "access crosses a page boundary.\n"
                                              "When disabled, a misalignment is triggered on all "
                                              "misaligned accesses."),
     Desktop | IOS},
    {"cpuopt_fastmem",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable Host MMU Emulation (general memory instructions)"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "This optimization speeds up memory accesses by the "
                                              "guest program.\n"
                                              "Enabling it causes guest memory reads/writes to be done "
                                              "directly into memory and make use of Host's MMU.\n"
                                              "Disabling this forces all memory accesses to use "
                                              "Software MMU Emulation."),
     Desktop | IOS},
    {"cpuopt_fastmem_exclusives",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable Host MMU Emulation (exclusive memory "
                                              "instructions)"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "This optimization speeds up exclusive memory accesses "
                                              "by the guest program.\n"
                                              "Enabling it causes guest exclusive memory reads/writes "
                                              "to be done directly into memory and make use of Host's "
                                              "MMU.\n"
                                              "Disabling this forces all exclusive memory accesses to "
                                              "use Software MMU Emulation."),
     Desktop | IOS},
    {"cpuopt_recompile_exclusives",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable recompilation of exclusive memory instructions"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "This optimization speeds up exclusive memory accesses "
                                              "by the guest program.\n"
                                              "Enabling it reduces the overhead of fastmem failure of "
                                              "exclusive memory accesses."),
     Desktop | IOS},
    {"cpuopt_ignore_memory_aborts",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable fallbacks for invalid memory accesses"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "This optimization speeds up memory accesses by allowing "
                                              "invalid memory accesses to succeed.\n"
                                              "Enabling it reduces the overhead of all memory accesses "
                                              "and has no impact on programs that don't access invalid "
                                              "memory."),
     Desktop | IOS},
    {"frame_gen_motion",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Frame Generation Motion"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Where frame generation gets its motion from. "
                                              "{choice:frame_gen_motion.Auto} uses NVIDIA's "
                                              "optical flow hardware (RTX 20 series and newer) when "
                                              "available, and the flow shader otherwise."),
     Desktop},
    {"frame_gen_flow_quality",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Optical Flow Quality"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Quality of the optical flow hardware's motion "
                                              "estimate. Higher quality costs more GPU time."),
     Desktop},
    {"use_optimized_vertex_buffers",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Optimized Vertex Buffers"),
     nullptr,
     IOS},
    {"emulate_bgr565",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Emulate BGR565"),
     nullptr,
     Nowhere,
     QT_TRANSLATE_NOOP("ConfigurationShared", "Only for some GPUs on Android.")},
    {"debug",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable Graphics Debugging"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "When checked, the graphics API enters a slower "
                                              "debugging mode"),
     Desktop | IOS},
    {"shader_feedback",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable Shader Feedback"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "When checked, Lichii will log statistics about the "
                                              "compiled pipeline cache"),
     Desktop | IOS},
    {"nsight_aftermath",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable Nsight Aftermath"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "When checked, it enables Nsight Aftermath crash dumps"),
     Desktop},
    {"disable_shader_loop_safety_checks",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Disable Loop safety checks"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "When checked, it executes shaders without loop logic "
                                              "changes"),
     Desktop | IOS},
    {"renderdoc_hotkey",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable Renderdoc Hotkey"),
     nullptr,
     Desktop},
    {"patch_old_qcom_drivers",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Patch Old Qualcomm Drivers"),
     nullptr,
     Nowhere,
     QT_TRANSLATE_NOOP("ConfigurationShared", "Only for old Qualcomm drivers on Android.")},
    {"disable_buffer_reorder",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Disable Buffer Reorder"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "When checked, disables reordering of mapped memory "
                                              "uploads which allows to associate uploads with specific "
                                              "draws. May reduce performance in some cases."),
     Desktop | IOS},
    {"serial_battery",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Battery Serial"),
     nullptr,
     Desktop | IOS},
    {"serial_unit",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Unit Serial"),
     nullptr,
     Desktop | IOS},
    {"disable_wgi_xinput",
     nullptr,
     nullptr,
     Nowhere},
    {"enable_raw_input",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable XInput 8 player support (disables web applet)"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Requires restarting Lichii"),
     Desktop},
    {"controller_navigation",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Controller navigation"),
     nullptr,
     Desktop},
    {"enable_joycon_driver",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable direct JoyCon driver"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Requires restarting Lichii"),
     Desktop},
    {"enable_procon_driver",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable direct Pro Controller driver [EXPERIMENTAL]"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Requires restarting Lichii"),
     Desktop},
    {"vibration_enabled",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Vibration"),
     nullptr,
     Desktop | IOS},
    {"enable_accurate_vibrations",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable Accurate Vibration"),
     nullptr,
     Desktop | IOS},
    {"motion_enabled",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Motion"),
     nullptr,
     Desktop | IOS},
    {"udp_input_servers",
     nullptr,
     nullptr,
     Nowhere},
    {"enable_udp_controller",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable UDP controllers (not needed for motion)"),
     nullptr,
     Desktop},
    {"pause_tas_on_load",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Pause execution during loads"),
     nullptr,
     Desktop},
    {"tas_enable",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable TAS features"),
     nullptr,
     Desktop},
    {"tas_loop",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Loop script"),
     nullptr,
     Desktop},
    {"tas_show_recording_dialog",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Show recording dialog"),
     nullptr,
     Desktop},
    {"mouse_panning",
     nullptr,
     nullptr,
     Nowhere},
    {"mouse_panning_sensitivity",
     nullptr,
     nullptr,
     Nowhere},
    {"mouse_enabled",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Mouse"),
     nullptr,
     Desktop},
    {"mouse_panning_x_sensitivity",
     nullptr,
     nullptr,
     Nowhere},
    {"mouse_panning_y_sensitivity",
     nullptr,
     nullptr,
     Nowhere},
    {"mouse_panning_deadzone_counterweight",
     nullptr,
     nullptr,
     Nowhere},
    {"mouse_panning_decay_strength",
     nullptr,
     nullptr,
     Nowhere},
    {"mouse_panning_min_decay",
     nullptr,
     nullptr,
     Nowhere},
    {"emulate_analog_keyboard",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Emulate Analog with Keyboard Input"),
     nullptr,
     Desktop},
    {"keyboard_enabled",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Keyboard"),
     nullptr,
     Desktop},
    {"debug_pad_enabled",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Debug Controller"),
     nullptr,
     Desktop},
    {"touch_device",
     nullptr,
     nullptr,
     Nowhere},
    {"touch_from_button_map",
     nullptr,
     nullptr,
     Nowhere},
    {"enable_ring_controller",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Ring Controller"),
     nullptr,
     Desktop | IOS},
    {"enable_ir_sensor",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Infrared Camera"),
     nullptr,
     Desktop},
    {"ir_sensor_device",
     nullptr,
     nullptr,
     Nowhere},
    {"random_amiibo_id",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Use random Amiibo ID"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Allows unlimited uses of the same Amiibo in games that "
                                              "would otherwise limit you to one use."),
     Desktop | IOS},
    {"use_virtual_sd",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Virtual SD Card"),
     nullptr,
     IOS},
    {"gamecard_inserted",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Inserted"),
     nullptr,
     Desktop},
    {"gamecard_current_game",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Current Game"),
     nullptr,
     Desktop},
    {"gamecard_path",
     nullptr,
     nullptr,
     Nowhere},
    {"ext_content_from_game_dirs",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Updates and DLC From Game Folders"),
     nullptr,
     IOS},
    {"use_gdbstub",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable GDB Stub"),
     nullptr,
     Desktop},
    {"gdbstub_port",
     QT_TRANSLATE_NOOP("ConfigurationShared", "GDB Stub Port"),
     nullptr,
     Desktop},
    {"dump_exefs",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Dump ExeFS"),
     nullptr,
     Desktop | IOS},
    {"dump_nso",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Dump Decompressed NSOs"),
     nullptr,
     Desktop | IOS},
    {"dump_guest_shaders",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Dump Guest (Maxwell) Shaders"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "When checked, it will dump original Maxwell guest "
                                              "shader bytecode (the input to the recompiler) as .ash "
                                              "files under DumpDir/shaders/. Useful for inspection "
                                              "with nvdisasm."),
     Desktop | IOS},
    {"dump_macros",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Dump Maxwell Macros"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "When checked, it will dump all the macro programs of "
                                              "the GPU"),
     Desktop | IOS},
    {"enable_fs_access_log",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable FS Access Log"),
     nullptr,
     Desktop | IOS},
    {"reporting_services",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable Verbose Reporting Services**"),
     nullptr,
     Desktop | IOS},
    {"quest_flag",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Kiosk (Quest) Mode"),
     nullptr,
     Desktop | IOS},
    {"use_dev_keys",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Use dev.keys"),
     nullptr,
     Desktop | IOS},
    {"disable_macro_jit",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Disable Macro JIT"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "When checked, it disables the macro Just In Time "
                                              "compiler. Enabling this makes games run slower"),
     Desktop,
     QT_TRANSLATE_NOOP("ConfigurationShared", "The macro JIT exists only on x86-64 PCs.")},
    {"disable_macro_hle",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Disable Macro HLE"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "When checked, it disables the macro HLE functions. "
                                              "Enabling this makes games run slower"),
     Desktop | IOS},
    {"extended_logging",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable Extended Logging**"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "When checked, the max size of the log increases from "
                                              "100 MB to 1 GB"),
     Desktop | IOS},
    {"use_debug_asserts",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable Debug Asserts"),
     nullptr,
     Desktop | IOS},
    {"use_auto_stub",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable Auto-Stub"),
     nullptr,
     Desktop | IOS},
    {"enable_all_controllers",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable All Controller Types"),
     nullptr,
     Desktop | IOS},
    {"perform_vulkan_check",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Perform Startup Vulkan Check"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enables Lichii to check for a working Vulkan "
                                              "environment when the program starts up. Disable this if "
                                              "this is causing issues with external programs seeing "
                                              "Lichii."),
     Desktop},
    {"disable_web_applet",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Disable Web Applet"),
     nullptr,
     Desktop | IOS},
    {"gpu_log_level",
     QT_TRANSLATE_NOOP("ConfigurationShared", "GPU Log Level"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Detail level for GPU logs. Off disables logging "
                                              "entirely."),
     Desktop},
    {"gpu_log_vulkan_calls",
     nullptr,
     nullptr,
     Nowhere},
    {"gpu_log_shader_dumps",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Dump SPIR-V Shaders"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "When checked, it will dump the recompiler's output "
                                              "SPIR-V binaries (.spv) under LogDir/shaders/. Inspect "
                                              "via SPIRV-Tools (spirv-dis / spirv-cross / spirv-val) "
                                              "or RenderDoc."),
     Desktop},
    {"gpu_log_memory_tracking",
     nullptr,
     nullptr,
     Nowhere},
    {"gpu_log_driver_debug",
     nullptr,
     nullptr,
     Nowhere},
    {"gpu_log_ring_buffer_size",
     nullptr,
     nullptr,
     Nowhere},
    {"debug_knobs",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Debug Knobs"),
     QT_TRANSLATE_NOOP("ConfigurationShared", "Bitmask for quick development toggles"),
     Desktop | IOS},
    {"log_filter",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Log Filter"),
     nullptr,
     Desktop,
     QT_TRANSLATE_NOOP("ConfigurationShared", "Lichii sets its own log filter on iOS at every start.")},
    {"flush_line",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Flush log output on each line"),
     nullptr,
     Desktop,
     QT_TRANSLATE_NOOP("ConfigurationShared", "On iOS every log line is written at once, so a log "
                                              "survives iOS closing Lichii.")},
    {"censor_username",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Censor username in logs"),
     nullptr,
     Desktop},
    {"first_launch",
     nullptr,
     nullptr,
     Nowhere},
    {"network_interface",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Network Interface"),
     nullptr,
     Desktop},
    {"airplane_mode",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Enable Airplane Mode"),
     nullptr,
     Desktop},
    {"web_api_url",
     nullptr,
     nullptr,
     Nowhere},
    {"eden_username",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Web Service Username"),
     nullptr,
     Desktop},
    {"eden_token",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Web Service Token"),
     nullptr,
     Desktop},
    {"singleWindowMode",
     nullptr,
     nullptr,
     Nowhere},
    {"fullscreen",
     nullptr,
     nullptr,
     Nowhere},
    {"showFilterBar",
     nullptr,
     nullptr,
     Nowhere},
    {"showStatusBar",
     nullptr,
     nullptr,
     Nowhere},
    {"firstStart",
     nullptr,
     nullptr,
     Nowhere},
    {"skipped_update_tag",
     nullptr,
     nullptr,
     Nowhere},
    {"update_remind_at",
     nullptr,
     nullptr,
     Nowhere},
    {"enable_discord_presence",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Show Current Game in your Discord Status"),
     nullptr,
     Desktop},
    {"showConsole",
     nullptr,
     nullptr,
     Nowhere},
    {"enable_screenshot_save_as",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Ask Where To Save Screenshots (Windows Only)"),
     nullptr,
     Desktop},
    {"screenshot_height",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Screenshot Resolution"),
     nullptr,
     Desktop},
    {"language",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Interface language"),
     nullptr,
     Desktop},
    {"steamgriddb_api_key",
     nullptr,
     nullptr,
     Nowhere},
    {"calloutFlags",
     nullptr,
     nullptr,
     Nowhere},
    {"nickname",
     nullptr,
     nullptr,
     Nowhere},
    {"filter_text",
     nullptr,
     nullptr,
     Nowhere},
    {"filter_games_owned",
     nullptr,
     nullptr,
     Nowhere},
    {"filter_games_hide_empty",
     nullptr,
     nullptr,
     Nowhere},
    {"filter_games_hide_full",
     nullptr,
     nullptr,
     Nowhere},
    {"ip",
     nullptr,
     nullptr,
     Nowhere},
    {"port",
     nullptr,
     nullptr,
     Nowhere},
    {"room_nickname",
     nullptr,
     nullptr,
     Nowhere},
    {"room_name",
     nullptr,
     nullptr,
     Nowhere},
    {"max_player",
     nullptr,
     nullptr,
     Nowhere},
    {"room_port",
     nullptr,
     nullptr,
     Nowhere},
    {"host_type",
     nullptr,
     nullptr,
     Nowhere},
    {"game_id",
     nullptr,
     nullptr,
     Nowhere},
    {"room_description",
     nullptr,
     nullptr,
     Nowhere},
    {"show_add_ons",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Show Add-Ons Column"),
     nullptr,
     Desktop},
    {"game_icon_size",
     nullptr,
     nullptr,
     Nowhere},
    {"folder_icon_size",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Folder Icon Size"),
     nullptr,
     Desktop},
    {"row_1_text_id",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Row 1 Text"),
     nullptr,
     Desktop},
    {"row_2_text_id",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Row 2 Text"),
     nullptr,
     Desktop},
    {"game_list_mode",
     nullptr,
     nullptr,
     Nowhere},
    {"show_game_name",
     nullptr,
     nullptr,
     Nowhere},
    {"cache_game_list",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Cache Game List Metadata"),
     nullptr,
     Desktop},
    {"favorites_expanded",
     nullptr,
     nullptr,
     Nowhere},
    {"show_perf_overlay",
     nullptr,
     nullptr,
     Nowhere},
    {"show_size",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Show Size Column"),
     nullptr,
     Desktop},
    {"show_types",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Show File Types Column"),
     nullptr,
     Desktop},
    {"show_play_time",
     QT_TRANSLATE_NOOP("ConfigurationShared", "Show Play Time Column"),
     nullptr,
     Desktop},
    {"show_fw_warning",
     nullptr,
     nullptr,
     Nowhere},
    {"show_fw_warning",
     nullptr,
     nullptr,
     Nowhere},
};

const Choice kChoices[] = {
    CHOICE(AppletMode, HLE, QT_TRANSLATE_NOOP("ConfigurationShared", "Custom Frontend")),
    CHOICE(AppletMode, LLE, QT_TRANSLATE_NOOP("ConfigurationShared", "Real Applet")),
    CHOICE(AppletMode, Disabled, QT_TRANSLATE_NOOP("ConfigurationShared", "Disabled")),
    CHOICE(SpirvOptimizeMode, Never, QT_TRANSLATE_NOOP("ConfigurationShared", "Never")),
    CHOICE(SpirvOptimizeMode, OnLoad, QT_TRANSLATE_NOOP("ConfigurationShared", "On Load")),
    CHOICE(SpirvOptimizeMode, Always, QT_TRANSLATE_NOOP("ConfigurationShared", "Always")),
    CHOICE(AstcDecodeMode, Cpu, QT_TRANSLATE_NOOP("ConfigurationShared", "CPU")),
    CHOICE(AstcDecodeMode, Gpu, QT_TRANSLATE_NOOP("ConfigurationShared", "GPU")),
    CHOICE(AstcDecodeMode, CpuAsynchronous, QT_TRANSLATE_NOOP("ConfigurationShared", "CPU Asynchronous")),
    CHOICE(AstcRecompression, Uncompressed, QT_TRANSLATE_NOOP("ConfigurationShared", "Uncompressed")),
    CHOICE(AstcRecompression, Bc1, QT_TRANSLATE_NOOP("ConfigurationShared", "BC1")),
    CHOICE(AstcRecompression, Bc3, QT_TRANSLATE_NOOP("ConfigurationShared", "BC3")),
    CHOICE(FramePacingMode, Target_Auto, QT_TRANSLATE_NOOP("ConfigurationShared", "Auto")),
    CHOICE(FramePacingMode, Target_30, QT_TRANSLATE_NOOP("ConfigurationShared", "30 FPS")),
    CHOICE(FramePacingMode, Target_60, QT_TRANSLATE_NOOP("ConfigurationShared", "60 FPS")),
    CHOICE(FramePacingMode, Target_90, QT_TRANSLATE_NOOP("ConfigurationShared", "90 FPS")),
    CHOICE(FramePacingMode, Target_120, QT_TRANSLATE_NOOP("ConfigurationShared", "120 FPS")),
    CHOICE(VramUsageMode, Conservative, QT_TRANSLATE_NOOP("ConfigurationShared", "Conservative")),
    CHOICE(VramUsageMode, Aggressive, QT_TRANSLATE_NOOP("ConfigurationShared", "Aggressive")),
    CHOICE(RendererBackend, Vulkan, QT_TRANSLATE_NOOP("ConfigurationShared", "Vulkan")),
    CHOICE_ON(RendererBackend, OpenGL_GLSL, QT_TRANSLATE_NOOP("ConfigurationShared", "OpenGL GLSL"), Desktop),
    CHOICE_ON(RendererBackend, OpenGL_GLASM, QT_TRANSLATE_NOOP("ConfigurationShared", "OpenGL GLASM"), Desktop),
    CHOICE_ON(RendererBackend, OpenGL_SPIRV, QT_TRANSLATE_NOOP("ConfigurationShared", "OpenGL SPIR-V"), Desktop),
    CHOICE(RendererBackend, Metal, QT_TRANSLATE_NOOP("ConfigurationShared", "Metal")),
    CHOICE_ON(RendererBackend, Null, QT_TRANSLATE_NOOP("ConfigurationShared", "Null"), Desktop),
    CHOICE(GpuAccuracy, Low, QT_TRANSLATE_NOOP("ConfigurationShared", "Fast")),
    CHOICE(GpuAccuracy, High, QT_TRANSLATE_NOOP("ConfigurationShared", "Accurate")),
    CHOICE(DmaAccuracy, Default, QT_TRANSLATE_NOOP("ConfigurationShared", "Default")),
    CHOICE(DmaAccuracy, Unsafe, QT_TRANSLATE_NOOP("ConfigurationShared", "Unsafe")),
    CHOICE(DmaAccuracy, Safe, QT_TRANSLATE_NOOP("ConfigurationShared", "Safe")),
    CHOICE(NativeComputeWriteMark, Auto, QT_TRANSLATE_NOOP("ConfigurationShared", "Auto")),
    CHOICE(NativeComputeWriteMark, Full, QT_TRANSLATE_NOOP("ConfigurationShared", "Full Binding")),
    CHOICE(NativeComputeWriteMark, Cap64K, QT_TRANSLATE_NOOP("ConfigurationShared", "64 KiB Cap")),
    CHOICE(GpuFenceBehavior, Default, QT_TRANSLATE_NOOP("ConfigurationShared", "Default")),
    CHOICE(GpuFenceBehavior, Immediate, QT_TRANSLATE_NOOP("ConfigurationShared", "Immediate")),
    CHOICE(GpuFenceBehavior, Balanced, QT_TRANSLATE_NOOP("ConfigurationShared", "Balanced")),
    CHOICE(GpuFenceBehavior, Accurate, QT_TRANSLATE_NOOP("ConfigurationShared", "Accurate")),
    CHOICE(CpuAccuracy, Auto, QT_TRANSLATE_NOOP("ConfigurationShared", "Auto")),
    CHOICE(CpuAccuracy, Accurate, QT_TRANSLATE_NOOP("ConfigurationShared", "Accurate")),
    CHOICE(CpuAccuracy, Unsafe, QT_TRANSLATE_NOOP("ConfigurationShared", "Unsafe")),
    CHOICE(CpuAccuracy, Paranoid, QT_TRANSLATE_NOOP("ConfigurationShared", "Paranoid")),
    CHOICE(CpuAccuracy, Debugging, QT_TRANSLATE_NOOP("ConfigurationShared", "Debugging")),
    CHOICE(CpuBackend, Dynarmic, QT_TRANSLATE_NOOP("ConfigurationShared", "Dynarmic")),
    CHOICE(CpuBackend, Nce, QT_TRANSLATE_NOOP("ConfigurationShared", "NCE")),
    CHOICE(FullscreenMode, Borderless, QT_TRANSLATE_NOOP("ConfigurationShared", "Borderless Window")),
    CHOICE(FullscreenMode, Exclusive, QT_TRANSLATE_NOOP("ConfigurationShared", "Exclusive Fullscreen")),
    CHOICE(NvdecEmulation, Off, QT_TRANSLATE_NOOP("ConfigurationShared", "Off")),
    CHOICE(NvdecEmulation, Cpu, QT_TRANSLATE_NOOP("ConfigurationShared", "CPU")),
    CHOICE(NvdecEmulation, Gpu, QT_TRANSLATE_NOOP("ConfigurationShared", "GPU")),
    CHOICE(ResolutionSetup, Res1_4X, QT_TRANSLATE_NOOP("ConfigurationShared", "0.25x")),
    CHOICE(ResolutionSetup, Res1_2X, QT_TRANSLATE_NOOP("ConfigurationShared", "0.5x")),
    CHOICE(ResolutionSetup, Res3_4X, QT_TRANSLATE_NOOP("ConfigurationShared", "0.75x")),
    CHOICE(ResolutionSetup, Res1X, QT_TRANSLATE_NOOP("ConfigurationShared", "1x")),
    CHOICE(ResolutionSetup, Res5_4X, QT_TRANSLATE_NOOP("ConfigurationShared", "1.25x")),
    CHOICE(ResolutionSetup, Res3_2X, QT_TRANSLATE_NOOP("ConfigurationShared", "1.5x")),
    CHOICE(ResolutionSetup, Res2X, QT_TRANSLATE_NOOP("ConfigurationShared", "2x")),
    CHOICE(ResolutionSetup, Res3X, QT_TRANSLATE_NOOP("ConfigurationShared", "3x")),
    CHOICE(ResolutionSetup, Res4X, QT_TRANSLATE_NOOP("ConfigurationShared", "4x")),
    CHOICE(ResolutionSetup, Res5X, QT_TRANSLATE_NOOP("ConfigurationShared", "5x")),
    CHOICE(ResolutionSetup, Res6X, QT_TRANSLATE_NOOP("ConfigurationShared", "6x")),
    CHOICE(ResolutionSetup, Res7X, QT_TRANSLATE_NOOP("ConfigurationShared", "7x")),
    CHOICE(ResolutionSetup, Res8X, QT_TRANSLATE_NOOP("ConfigurationShared", "8x")),
    CHOICE(GpuLogLevel, Off, QT_TRANSLATE_NOOP("ConfigurationShared", "Off")),
    CHOICE(GpuLogLevel, Errors, QT_TRANSLATE_NOOP("ConfigurationShared", "Errors")),
    CHOICE(GpuLogLevel, Standard, QT_TRANSLATE_NOOP("ConfigurationShared", "Standard")),
    CHOICE(GpuLogLevel, Verbose, QT_TRANSLATE_NOOP("ConfigurationShared", "Verbose")),
    CHOICE(GpuLogLevel, All, QT_TRANSLATE_NOOP("ConfigurationShared", "All")),
    CHOICE(FrameGenMotion, Auto, QT_TRANSLATE_NOOP("ConfigurationShared", "Auto")),
    CHOICE(FrameGenMotion, Shader, QT_TRANSLATE_NOOP("ConfigurationShared", "Flow Shader")),
    CHOICE(FrameGenFlowQuality, Fast, QT_TRANSLATE_NOOP("ConfigurationShared", "Fast")),
    CHOICE(FrameGenFlowQuality, Medium, QT_TRANSLATE_NOOP("ConfigurationShared", "Medium")),
    CHOICE(FrameGenFlowQuality, Best, QT_TRANSLATE_NOOP("ConfigurationShared", "Best")),
    CHOICE(FrameGenType, Lsfg, QT_TRANSLATE_NOOP("ConfigurationShared", "Lossless Scaling")),
    CHOICE(FrameGenType, Fsr3, QT_TRANSLATE_NOOP("ConfigurationShared", "Optical Flow")),
    CHOICE(FrameGenType, MetalFx, QT_TRANSLATE_NOOP("ConfigurationShared", "MetalFX")),
    CHOICE(ScalingFilter, NearestNeighbor, QT_TRANSLATE_NOOP("ConfigurationShared", "Nearest Neighbor")),
    CHOICE(ScalingFilter, Bilinear, QT_TRANSLATE_NOOP("ConfigurationShared", "Bilinear")),
    CHOICE(ScalingFilter, Bicubic, QT_TRANSLATE_NOOP("ConfigurationShared", "Bicubic")),
    CHOICE(ScalingFilter, Gaussian, QT_TRANSLATE_NOOP("ConfigurationShared", "Gaussian")),
    CHOICE(ScalingFilter, Lanczos, QT_TRANSLATE_NOOP("ConfigurationShared", "Lanczos")),
    CHOICE(ScalingFilter, ScaleForce, QT_TRANSLATE_NOOP("ConfigurationShared", "ScaleForce")),
    CHOICE(ScalingFilter, Fsr, QT_TRANSLATE_NOOP("ConfigurationShared", "AMD FSR 1")),
    CHOICE(ScalingFilter, Area, QT_TRANSLATE_NOOP("ConfigurationShared", "Area")),
    CHOICE(ScalingFilter, Mmpx, QT_TRANSLATE_NOOP("ConfigurationShared", "MMPX")),
    CHOICE(ScalingFilter, ZeroTangent, QT_TRANSLATE_NOOP("ConfigurationShared", "Zero-Tangent")),
    CHOICE(ScalingFilter, BSpline, QT_TRANSLATE_NOOP("ConfigurationShared", "B-Spline")),
    CHOICE(ScalingFilter, Mitchell, QT_TRANSLATE_NOOP("ConfigurationShared", "Mitchell")),
    CHOICE(ScalingFilter, Spline1, QT_TRANSLATE_NOOP("ConfigurationShared", "Spline-1")),
    CHOICE(ScalingFilter, Sgsr, QT_TRANSLATE_NOOP("ConfigurationShared", "Snapdragon GSR")),
    CHOICE(ScalingFilter, SgsrEdge, QT_TRANSLATE_NOOP("ConfigurationShared", "Snapdragon GSR Edge")),
    CHOICE(ScalingFilter, Fsr3, QT_TRANSLATE_NOOP("ConfigurationShared", "AMD FSR 1 Soft")),
    CHOICE(ScalingFilter, MetalFxSpatial, QT_TRANSLATE_NOOP("ConfigurationShared", "MetalFX Spatial")),
    CHOICE(AntiAliasing, None, QT_TRANSLATE_NOOP("ConfigurationShared", "None")),
    CHOICE(AntiAliasing, Fxaa, QT_TRANSLATE_NOOP("ConfigurationShared", "FXAA")),
    CHOICE(AntiAliasing, Smaa, QT_TRANSLATE_NOOP("ConfigurationShared", "SMAA")),
    CHOICE(DlssPreset, Default, QT_TRANSLATE_NOOP("ConfigurationShared", "Default")),
    CHOICE(DlssPreset, J, QT_TRANSLATE_NOOP("ConfigurationShared", "J")),
    CHOICE(DlssPreset, K, QT_TRANSLATE_NOOP("ConfigurationShared", "K")),
    CHOICE(DlssPreset, L, QT_TRANSLATE_NOOP("ConfigurationShared", "L")),
    CHOICE(DlssPreset, M, QT_TRANSLATE_NOOP("ConfigurationShared", "M")),
    CHOICE(GamePatchMode, Auto, QT_TRANSLATE_NOOP("ConfigurationShared", "Auto")),
    CHOICE(GamePatchMode, On, QT_TRANSLATE_NOOP("ConfigurationShared", "On")),
    CHOICE(GamePatchMode, Off, QT_TRANSLATE_NOOP("ConfigurationShared", "Off")),
    CHOICE(MoltenVkDriver, Legacy, QT_TRANSLATE_NOOP("ConfigurationShared", "Legacy MoltenVK")),
    CHOICE(MoltenVkDriver, Mesh, QT_TRANSLATE_NOOP("ConfigurationShared", "New MoltenVK")),
    CHOICE(TextureQuality, Full, QT_TRANSLATE_NOOP("ConfigurationShared", "Full")),
    CHOICE(TextureQuality, Half, QT_TRANSLATE_NOOP("ConfigurationShared", "Half")),
    CHOICE(TextureQuality, Quarter, QT_TRANSLATE_NOOP("ConfigurationShared", "Quarter")),
    CHOICE(OotResolution, Game, QT_TRANSLATE_NOOP("ConfigurationShared", "Dynamic")),
    CHOICE(OotResolution, Percent33, QT_TRANSLATE_NOOP("ConfigurationShared", "33%")),
    CHOICE(OotResolution, Percent40, QT_TRANSLATE_NOOP("ConfigurationShared", "40%")),
    CHOICE(OotResolution, Percent50, QT_TRANSLATE_NOOP("ConfigurationShared", "50%")),
    CHOICE(OotResolution, Percent56, QT_TRANSLATE_NOOP("ConfigurationShared", "56%")),
    CHOICE(OotResolution, Percent75, QT_TRANSLATE_NOOP("ConfigurationShared", "75%")),
    CHOICE(OotResolution, Percent87, QT_TRANSLATE_NOOP("ConfigurationShared", "87%")),
    CHOICE(OotResolution, Full, QT_TRANSLATE_NOOP("ConfigurationShared", "100%")),
    CHOICE(OotResolution, Percent150, QT_TRANSLATE_NOOP("ConfigurationShared", "150%")),
    CHOICE(OotResolution, Percent200, QT_TRANSLATE_NOOP("ConfigurationShared", "200%")),
    CHOICE(OotResolution, Percent300, QT_TRANSLATE_NOOP("ConfigurationShared", "300%")),
    CHOICE(OotGraphicsPreset, PhoneLow, QT_TRANSLATE_NOOP("ConfigurationShared", "Phone Low")),
    CHOICE(OotGraphicsPreset, PhoneMedium, QT_TRANSLATE_NOOP("ConfigurationShared", "Phone Medium")),
    CHOICE(OotGraphicsPreset, Console, QT_TRANSLATE_NOOP("ConfigurationShared", "Console")),
    CHOICE(OotGraphicsPreset, High, QT_TRANSLATE_NOOP("ConfigurationShared", "High")),
    CHOICE(Upscaler, Auto, QT_TRANSLATE_NOOP("ConfigurationShared", "Automatic")),
    CHOICE(Upscaler, Game, QT_TRANSLATE_NOOP("ConfigurationShared", "Game's Own")),
    CHOICE(Upscaler, Fsr3, QT_TRANSLATE_NOOP("ConfigurationShared", "AMD FSR 3.1")),
    CHOICE_ON(Upscaler, Dlss, QT_TRANSLATE_NOOP("ConfigurationShared", "NVIDIA DLSS"), Desktop),
    CHOICE(Upscaler, MetalFx, QT_TRANSLATE_NOOP("ConfigurationShared", "MetalFX Temporal")),
    CHOICE(OotFrameRate, Fps30, QT_TRANSLATE_NOOP("ConfigurationShared", "30 FPS")),
    CHOICE(OotFrameRate, Fps60, QT_TRANSLATE_NOOP("ConfigurationShared", "60 FPS")),
    CHOICE(OotFrameRate, Unlocked, QT_TRANSLATE_NOOP("ConfigurationShared", "Unlocked")),
    CHOICE(MkwFrameRate, Game, QT_TRANSLATE_NOOP("ConfigurationShared", "Game's Own")),
    CHOICE(MkwFrameRate, Fps30, QT_TRANSLATE_NOOP("ConfigurationShared", "30 FPS")),
    CHOICE(MkwFrameRate, Dynamic, QT_TRANSLATE_NOOP("ConfigurationShared", "Dynamic 30-60 FPS")),
    CHOICE(OotMovieUpscale, Automatic, QT_TRANSLATE_NOOP("ConfigurationShared", "Automatic")),
    CHOICE(OotMovieUpscale, Bilinear, QT_TRANSLATE_NOOP("ConfigurationShared", "Bilinear")),
    CHOICE(OotMovieUpscale, Game, QT_TRANSLATE_NOOP("ConfigurationShared", "Game's Own")),
    CHOICE(OotAntiAliasing, Game, QT_TRANSLATE_NOOP("ConfigurationShared", "Game's Own")),
    CHOICE(OotAntiAliasing, NoSharpening, QT_TRANSLATE_NOOP("ConfigurationShared", "No Sharpening")),
    CHOICE(AspectRatio, R16_9, QT_TRANSLATE_NOOP("ConfigurationShared", "16:9")),
    CHOICE(AspectRatio, R4_3, QT_TRANSLATE_NOOP("ConfigurationShared", "4:3")),
    CHOICE(AspectRatio, R21_9, QT_TRANSLATE_NOOP("ConfigurationShared", "21:9")),
    CHOICE(AspectRatio, R16_10, QT_TRANSLATE_NOOP("ConfigurationShared", "16:10")),
    CHOICE(AspectRatio, Stretch, QT_TRANSLATE_NOOP("ConfigurationShared", "Stretch")),
    CHOICE(AnisotropyMode, Automatic, QT_TRANSLATE_NOOP("ConfigurationShared", "Automatic")),
    CHOICE(AnisotropyMode, Default, QT_TRANSLATE_NOOP("ConfigurationShared", "Default")),
    CHOICE(AnisotropyMode, X2, QT_TRANSLATE_NOOP("ConfigurationShared", "2x")),
    CHOICE(AnisotropyMode, X4, QT_TRANSLATE_NOOP("ConfigurationShared", "4x")),
    CHOICE(AnisotropyMode, X8, QT_TRANSLATE_NOOP("ConfigurationShared", "8x")),
    CHOICE(AnisotropyMode, X16, QT_TRANSLATE_NOOP("ConfigurationShared", "16x")),
    CHOICE(Language, Japanese, QT_TRANSLATE_NOOP("ConfigurationShared", "Japanese (日本語)")),
    CHOICE(Language, EnglishAmerican, QT_TRANSLATE_NOOP("ConfigurationShared", "American English")),
    CHOICE(Language, French, QT_TRANSLATE_NOOP("ConfigurationShared", "French (français)")),
    CHOICE(Language, German, QT_TRANSLATE_NOOP("ConfigurationShared", "German (Deutsch)")),
    CHOICE(Language, Italian, QT_TRANSLATE_NOOP("ConfigurationShared", "Italian (italiano)")),
    CHOICE(Language, Spanish, QT_TRANSLATE_NOOP("ConfigurationShared", "Spanish (español)")),
    CHOICE(Language, Chinese, QT_TRANSLATE_NOOP("ConfigurationShared", "Chinese")),
    CHOICE(Language, Korean, QT_TRANSLATE_NOOP("ConfigurationShared", "Korean (한국어)")),
    CHOICE(Language, Dutch, QT_TRANSLATE_NOOP("ConfigurationShared", "Dutch (Nederlands)")),
    CHOICE(Language, Portuguese, QT_TRANSLATE_NOOP("ConfigurationShared", "Portuguese (português)")),
    CHOICE(Language, Russian, QT_TRANSLATE_NOOP("ConfigurationShared", "Russian (Русский)")),
    CHOICE(Language, Taiwanese, QT_TRANSLATE_NOOP("ConfigurationShared", "Taiwanese")),
    CHOICE(Language, EnglishBritish, QT_TRANSLATE_NOOP("ConfigurationShared", "British English")),
    CHOICE(Language, FrenchCanadian, QT_TRANSLATE_NOOP("ConfigurationShared", "Canadian French")),
    CHOICE(Language, SpanishLatin, QT_TRANSLATE_NOOP("ConfigurationShared", "Latin American Spanish")),
    CHOICE(Language, ChineseSimplified, QT_TRANSLATE_NOOP("ConfigurationShared", "Simplified Chinese")),
    CHOICE(Language, ChineseTraditional, QT_TRANSLATE_NOOP("ConfigurationShared", "Traditional Chinese (正體中文)")),
    CHOICE(Language, PortugueseBrazilian, QT_TRANSLATE_NOOP("ConfigurationShared", "Brazilian Portuguese (português do Brasil)")),
    CHOICE(Language, Polish, QT_TRANSLATE_NOOP("ConfigurationShared", "Polish (polska)")),
    CHOICE(Language, Thai, QT_TRANSLATE_NOOP("ConfigurationShared", "Thai (แบบไทย)")),
    CHOICE(Region, Japan, QT_TRANSLATE_NOOP("ConfigurationShared", "Japan")),
    CHOICE(Region, Usa, QT_TRANSLATE_NOOP("ConfigurationShared", "USA")),
    CHOICE(Region, Europe, QT_TRANSLATE_NOOP("ConfigurationShared", "Europe")),
    CHOICE(Region, Australia, QT_TRANSLATE_NOOP("ConfigurationShared", "Australia")),
    CHOICE(Region, China, QT_TRANSLATE_NOOP("ConfigurationShared", "China")),
    CHOICE(Region, Korea, QT_TRANSLATE_NOOP("ConfigurationShared", "Korea")),
    CHOICE(Region, Taiwan, QT_TRANSLATE_NOOP("ConfigurationShared", "Taiwan")),
    CHOICE(TimeZone, Auto, QT_TRANSLATE_NOOP("ConfigurationShared", "Auto")),
    CHOICE(TimeZone, Default, QT_TRANSLATE_NOOP("ConfigurationShared", "Default")),
    CHOICE(TimeZone, Cet, QT_TRANSLATE_NOOP("ConfigurationShared", "CET")),
    CHOICE(TimeZone, Cst6Cdt, QT_TRANSLATE_NOOP("ConfigurationShared", "CST6CDT")),
    CHOICE(TimeZone, Cuba, QT_TRANSLATE_NOOP("ConfigurationShared", "Cuba")),
    CHOICE(TimeZone, Eet, QT_TRANSLATE_NOOP("ConfigurationShared", "EET")),
    CHOICE(TimeZone, Egypt, QT_TRANSLATE_NOOP("ConfigurationShared", "Egypt")),
    CHOICE(TimeZone, Eire, QT_TRANSLATE_NOOP("ConfigurationShared", "Eire")),
    CHOICE(TimeZone, Est, QT_TRANSLATE_NOOP("ConfigurationShared", "EST")),
    CHOICE(TimeZone, Est5Edt, QT_TRANSLATE_NOOP("ConfigurationShared", "EST5EDT")),
    CHOICE(TimeZone, Gb, QT_TRANSLATE_NOOP("ConfigurationShared", "GB")),
    CHOICE(TimeZone, GbEire, QT_TRANSLATE_NOOP("ConfigurationShared", "GB-Eire")),
    CHOICE(TimeZone, Gmt, QT_TRANSLATE_NOOP("ConfigurationShared", "GMT")),
    CHOICE(TimeZone, GmtPlusZero, QT_TRANSLATE_NOOP("ConfigurationShared", "GMT+0")),
    CHOICE(TimeZone, GmtMinusZero, QT_TRANSLATE_NOOP("ConfigurationShared", "GMT-0")),
    CHOICE(TimeZone, GmtZero, QT_TRANSLATE_NOOP("ConfigurationShared", "GMT0")),
    CHOICE(TimeZone, Greenwich, QT_TRANSLATE_NOOP("ConfigurationShared", "Greenwich")),
    CHOICE(TimeZone, Hongkong, QT_TRANSLATE_NOOP("ConfigurationShared", "Hongkong")),
    CHOICE(TimeZone, Hst, QT_TRANSLATE_NOOP("ConfigurationShared", "HST")),
    CHOICE(TimeZone, Iceland, QT_TRANSLATE_NOOP("ConfigurationShared", "Iceland")),
    CHOICE(TimeZone, Iran, QT_TRANSLATE_NOOP("ConfigurationShared", "Iran")),
    CHOICE(TimeZone, Israel, QT_TRANSLATE_NOOP("ConfigurationShared", "Israel")),
    CHOICE(TimeZone, Jamaica, QT_TRANSLATE_NOOP("ConfigurationShared", "Jamaica")),
    CHOICE(TimeZone, Japan, QT_TRANSLATE_NOOP("ConfigurationShared", "Japan")),
    CHOICE(TimeZone, Kwajalein, QT_TRANSLATE_NOOP("ConfigurationShared", "Kwajalein")),
    CHOICE(TimeZone, Libya, QT_TRANSLATE_NOOP("ConfigurationShared", "Libya")),
    CHOICE(TimeZone, Met, QT_TRANSLATE_NOOP("ConfigurationShared", "MET")),
    CHOICE(TimeZone, Mst, QT_TRANSLATE_NOOP("ConfigurationShared", "MST")),
    CHOICE(TimeZone, Mst7Mdt, QT_TRANSLATE_NOOP("ConfigurationShared", "MST7MDT")),
    CHOICE(TimeZone, Navajo, QT_TRANSLATE_NOOP("ConfigurationShared", "Navajo")),
    CHOICE(TimeZone, Nz, QT_TRANSLATE_NOOP("ConfigurationShared", "NZ")),
    CHOICE(TimeZone, NzChat, QT_TRANSLATE_NOOP("ConfigurationShared", "NZ-CHAT")),
    CHOICE(TimeZone, Poland, QT_TRANSLATE_NOOP("ConfigurationShared", "Poland")),
    CHOICE(TimeZone, Portugal, QT_TRANSLATE_NOOP("ConfigurationShared", "Portugal")),
    CHOICE(TimeZone, Prc, QT_TRANSLATE_NOOP("ConfigurationShared", "PRC")),
    CHOICE(TimeZone, Pst8Pdt, QT_TRANSLATE_NOOP("ConfigurationShared", "PST8PDT")),
    CHOICE(TimeZone, Roc, QT_TRANSLATE_NOOP("ConfigurationShared", "ROC")),
    CHOICE(TimeZone, Rok, QT_TRANSLATE_NOOP("ConfigurationShared", "ROK")),
    CHOICE(TimeZone, Singapore, QT_TRANSLATE_NOOP("ConfigurationShared", "Singapore")),
    CHOICE(TimeZone, Turkey, QT_TRANSLATE_NOOP("ConfigurationShared", "Turkey")),
    CHOICE(TimeZone, Uct, QT_TRANSLATE_NOOP("ConfigurationShared", "UCT")),
    CHOICE(TimeZone, Universal, QT_TRANSLATE_NOOP("ConfigurationShared", "Universal")),
    CHOICE(TimeZone, Utc, QT_TRANSLATE_NOOP("ConfigurationShared", "UTC")),
    CHOICE(TimeZone, WSu, QT_TRANSLATE_NOOP("ConfigurationShared", "W-SU")),
    CHOICE(TimeZone, Wet, QT_TRANSLATE_NOOP("ConfigurationShared", "WET")),
    CHOICE(TimeZone, Zulu, QT_TRANSLATE_NOOP("ConfigurationShared", "Zulu")),
    CHOICE(AudioMode, Mono, QT_TRANSLATE_NOOP("ConfigurationShared", "Mono")),
    CHOICE(AudioMode, Stereo, QT_TRANSLATE_NOOP("ConfigurationShared", "Stereo")),
    CHOICE(AudioMode, Surround, QT_TRANSLATE_NOOP("ConfigurationShared", "Surround")),
    CHOICE(MemoryLayout, Memory_4Gb, QT_TRANSLATE_NOOP("ConfigurationShared", "4 GB")),
    CHOICE(MemoryLayout, Memory_6Gb, QT_TRANSLATE_NOOP("ConfigurationShared", "6 GB")),
    CHOICE(MemoryLayout, Memory_8Gb, QT_TRANSLATE_NOOP("ConfigurationShared", "8 GB")),
    CHOICE(MemoryLayout, Memory_10Gb, QT_TRANSLATE_NOOP("ConfigurationShared", "10 GB")),
    CHOICE(MemoryLayout, Memory_12Gb, QT_TRANSLATE_NOOP("ConfigurationShared", "12 GB")),
    CHOICE(ConsoleMode, Docked, QT_TRANSLATE_NOOP("ConfigurationShared", "Docked")),
    CHOICE(ConsoleMode, Handheld, QT_TRANSLATE_NOOP("ConfigurationShared", "Handheld")),
    CHOICE(PrecacheShaders, Off, QT_TRANSLATE_NOOP("ConfigurationShared", "Off")),
    CHOICE(PrecacheShaders, Ask, QT_TRANSLATE_NOOP("ConfigurationShared", "Ask")),
    CHOICE(PrecacheShaders, Always, QT_TRANSLATE_NOOP("ConfigurationShared", "Always")),
    CHOICE(CpuClock, Normal, QT_TRANSLATE_NOOP("ConfigurationShared", "Normal")),
    CHOICE(CpuClock, Boost, QT_TRANSLATE_NOOP("ConfigurationShared", "Boost")),
    CHOICE(CpuClock, Overclock, QT_TRANSLATE_NOOP("ConfigurationShared", "Overclock")),
    CHOICE(ConfirmStop, Ask_Always, QT_TRANSLATE_NOOP("ConfigurationShared", "Always Ask")),
    CHOICE(ConfirmStop, Ask_Based_On_Game, QT_TRANSLATE_NOOP("ConfigurationShared", "When the Game Asks")),
    CHOICE(ConfirmStop, Ask_Never, QT_TRANSLATE_NOOP("ConfigurationShared", "Never Ask")),
    CHOICE(GpuClock, Normal, QT_TRANSLATE_NOOP("ConfigurationShared", "Normal")),
    CHOICE(GpuClock, Boost, QT_TRANSLATE_NOOP("ConfigurationShared", "Boost")),
    CHOICE(GpuClock, Overclock, QT_TRANSLATE_NOOP("ConfigurationShared", "Overclock")),
    CHOICE(GpuUnswizzleSize, VerySmall, QT_TRANSLATE_NOOP("ConfigurationShared", "Very Small (16 MB)")),
    CHOICE(GpuUnswizzleSize, Small, QT_TRANSLATE_NOOP("ConfigurationShared", "Small (32 MB)")),
    CHOICE(GpuUnswizzleSize, Normal, QT_TRANSLATE_NOOP("ConfigurationShared", "Normal (128 MB)")),
    CHOICE(GpuUnswizzleSize, Large, QT_TRANSLATE_NOOP("ConfigurationShared", "Large (256 MB)")),
    CHOICE(GpuUnswizzleSize, VeryLarge, QT_TRANSLATE_NOOP("ConfigurationShared", "Very Large (512 MB)")),
    CHOICE(GpuUnswizzle, VeryLow, QT_TRANSLATE_NOOP("ConfigurationShared", "Very Low (4 MB)")),
    CHOICE(GpuUnswizzle, Low, QT_TRANSLATE_NOOP("ConfigurationShared", "Low (8 MB)")),
    CHOICE(GpuUnswizzle, Normal, QT_TRANSLATE_NOOP("ConfigurationShared", "Normal (16 MB)")),
    CHOICE(GpuUnswizzle, Medium, QT_TRANSLATE_NOOP("ConfigurationShared", "Medium (32 MB)")),
    CHOICE(GpuUnswizzle, High, QT_TRANSLATE_NOOP("ConfigurationShared", "High (64 MB)")),
    CHOICE(GpuUnswizzleChunk, VeryLow, QT_TRANSLATE_NOOP("ConfigurationShared", "Very Low (32)")),
    CHOICE(GpuUnswizzleChunk, Low, QT_TRANSLATE_NOOP("ConfigurationShared", "Low (64)")),
    CHOICE(GpuUnswizzleChunk, Normal, QT_TRANSLATE_NOOP("ConfigurationShared", "Normal (128)")),
    CHOICE(GpuUnswizzleChunk, Medium, QT_TRANSLATE_NOOP("ConfigurationShared", "Medium (256)")),
    CHOICE(GpuUnswizzleChunk, High, QT_TRANSLATE_NOOP("ConfigurationShared", "High (512)")),
    CHOICE(ExtendedDynamicState, Disabled, QT_TRANSLATE_NOOP("ConfigurationShared", "Off")),
    CHOICE(ExtendedDynamicState, EDS1, QT_TRANSLATE_NOOP("ConfigurationShared", "EDS 1")),
    CHOICE(ExtendedDynamicState, EDS2, QT_TRANSLATE_NOOP("ConfigurationShared", "EDS 2")),
    CHOICE(ExtendedDynamicState, EDS3, QT_TRANSLATE_NOOP("ConfigurationShared", "EDS 3")),
    CHOICE(GameListMode, TreeView, QT_TRANSLATE_NOOP("ConfigurationShared", "Tree View")),
    CHOICE(GameListMode, GridView, QT_TRANSLATE_NOOP("ConfigurationShared", "Grid View")),
    CHOICE(VSyncMode, Immediate, QT_TRANSLATE_NOOP("ConfigurationShared", "Off (Immediate)")),
    CHOICE(VSyncMode, Mailbox, QT_TRANSLATE_NOOP("ConfigurationShared", "Mailbox")),
    CHOICE(VSyncMode, Fifo, QT_TRANSLATE_NOOP("ConfigurationShared", "On (FIFO)")),
    CHOICE(VSyncMode, FifoRelaxed, QT_TRANSLATE_NOOP("ConfigurationShared", "Relaxed FIFO")),
};

const Text kTexts[] = {
    // Pages and groups of Settings (the iOS settings root and the desktop sidebar).
    {"page.settings", QT_TRANSLATE_NOOP("UiCatalog", "Settings")},
    {"page.graphics", QT_TRANSLATE_NOOP("UiCatalog", "Graphics")},
    {"page.controls", QT_TRANSLATE_NOOP("UiCatalog", "Controls")},
    {"page.audio", QT_TRANSLATE_NOOP("UiCatalog", "Audio")},
    {"page.system", QT_TRANSLATE_NOOP("UiCatalog", "System")},
    {"page.users", QT_TRANSLATE_NOOP("UiCatalog", "Users")},
    {"page.storage", QT_TRANSLATE_NOOP("UiCatalog", "Storage")},
    {"page.general", QT_TRANSLATE_NOOP("UiCatalog", "General")},
    {"page.advanced", QT_TRANSLATE_NOOP("UiCatalog", "Advanced")},
    {"page.recommended", QT_TRANSLATE_NOOP("UiCatalog", "Recommended Settings")},
    {"page.game_settings", QT_TRANSLATE_NOOP("UiCatalog", "Game Settings")},
    {"page.experiments", QT_TRANSLATE_NOOP("UiCatalog", "Memory & Experiments")},
    {"page.library", QT_TRANSLATE_NOOP("UiCatalog", "Library")},
    {"page.keys_firmware", QT_TRANSLATE_NOOP("UiCatalog", "Keys & Firmware")},
    {"page.setup_assistant", QT_TRANSLATE_NOOP("UiCatalog", "Setup Assistant")},
    {"page.logs", QT_TRANSLATE_NOOP("UiCatalog", "Logs")},
    {"page.about_device", QT_TRANSLATE_NOOP("UiCatalog", "About This Device")},
    {"page.shader_cache", QT_TRANSLATE_NOOP("UiCatalog", "Shader Cache")},
    {"group.games_data", QT_TRANSLATE_NOOP("UiCatalog", "Games & Data")},
    {"group.help", QT_TRANSLATE_NOOP("UiCatalog", "Help")},
    {"group.app", QT_TRANSLATE_NOOP("UiCatalog", "App")},
    {"section.device", QT_TRANSLATE_NOOP("UiCatalog", "Device")},
    {"advanced.warning",
     QT_TRANSLATE_NOOP("UiCatalog", "These settings are for testing and troubleshooting. If a game "
                                    "stops starting after a change here, set it back.")},
    {"advanced.footer",
     QT_TRANSLATE_NOOP("UiCatalog", "Debug switches, CPU and GPU hacks. Changing them can stop games "
                                    "from starting.")},
    {"advanced.show", QT_TRANSLATE_NOOP("UiCatalog", "Show Advanced Settings")},
    {"advanced.show.footer",
     QT_TRANSLATE_NOOP("UiCatalog", "Debug switches, CPU and GPU hacks. Changing them can stop the "
                                    "game from starting.")},

    // Section names: a setting category ("category.<Category>"), or a group of its own.
    {"category.Renderer", QT_TRANSLATE_NOOP("UiCatalog", "Image")},
    {"category.RendererAdvanced", QT_TRANSLATE_NOOP("UiCatalog", "Performance")},
    {"category.RendererHacks", QT_TRANSLATE_NOOP("UiCatalog", "Graphics Hacks")},
    {"category.RendererExtensions", QT_TRANSLATE_NOOP("UiCatalog", "Vulkan Extensions")},
    {"category.RendererDebug", QT_TRANSLATE_NOOP("UiCatalog", "Graphics Debug")},
    {"category.Core", QT_TRANSLATE_NOOP("UiCatalog", "Emulation")},
    {"category.Cpu", QT_TRANSLATE_NOOP("UiCatalog", "CPU")},
    {"category.CpuDebug", QT_TRANSLATE_NOOP("UiCatalog", "CPU Debug")},
    {"category.CpuUnsafe", QT_TRANSLATE_NOOP("UiCatalog", "CPU Unsafe Optimizations")},
    {"category.Audio", QT_TRANSLATE_NOOP("UiCatalog", "Audio")},
    {"category.System", QT_TRANSLATE_NOOP("UiCatalog", "System")},
    {"category.SystemAudio", QT_TRANSLATE_NOOP("UiCatalog", "System Audio")},
    {"category.DataStorage", QT_TRANSLATE_NOOP("UiCatalog", "Storage")},
    {"category.Controls", QT_TRANSLATE_NOOP("UiCatalog", "Controls")},
    {"category.Debugging", QT_TRANSLATE_NOOP("UiCatalog", "Debugging")},
    {"category.DebuggingGraphics", QT_TRANSLATE_NOOP("UiCatalog", "Graphics Debugging")},
    {"category.Network", QT_TRANSLATE_NOOP("UiCatalog", "Network")},
    {"category.Miscellaneous", QT_TRANSLATE_NOOP("UiCatalog", "Miscellaneous")},
    {"category.Services", QT_TRANSLATE_NOOP("UiCatalog", "Services")},
    {"category.LibraryApplet", QT_TRANSLATE_NOOP("UiCatalog", "System Applets")},
    {"category.Overlay", QT_TRANSLATE_NOOP("UiCatalog", "Overlay")},
    {"category.GpuDriver", QT_TRANSLATE_NOOP("UiCatalog", "GPU Driver")},
    {"category.AddOns", QT_TRANSLATE_NOOP("UiCatalog", "Updates, DLC & Mods")},
    {"category.GameEnhancements", QT_TRANSLATE_NOOP("UiCatalog", "Game Enhancements")},
    {"category.Resolution", QT_TRANSLATE_NOOP("UiCatalog", "Resolution & Textures")},
    {"category.Upscaling", QT_TRANSLATE_NOOP("UiCatalog", "Upscaling")},
    {"category.FrameRate", QT_TRANSLATE_NOOP("UiCatalog", "Frame Rate")},
    {"category.Experiments", QT_TRANSLATE_NOOP("UiCatalog", "Experiments")},
    {"category.Essentials", QT_TRANSLATE_NOOP("UiCatalog", "Essentials")},
    {"category.Renderer.desktop", QT_TRANSLATE_NOOP("UiCatalog", "Renderer")},
    {"section.advanced_graphics", QT_TRANSLATE_NOOP("UiCatalog", "Advanced Graphics")},
    {"section.hacks_extensions", QT_TRANSLATE_NOOP("UiCatalog", "Graphics Hacks & Extensions")},
    {"section.hotkeys", QT_TRANSLATE_NOOP("UiCatalog", "Hotkeys")},
    {"section.interface", QT_TRANSLATE_NOOP("UiCatalog", "Interface")},
    {"section.web_service", QT_TRANSLATE_NOOP("UiCatalog", "Web Service")},
    {"section.debug", QT_TRANSLATE_NOOP("UiCatalog", "Debug")},
    {"section.network", QT_TRANSLATE_NOOP("UiCatalog", "Network")},
    {"section.cpu", QT_TRANSLATE_NOOP("UiCatalog", "CPU")},

    // Section footers.
    {"category.Resolution.footer",
     QT_TRANSLATE_NOOP("UiCatalog", "The size games draw at and their texture detail. Games with "
                                    "their own Resolution (%1) have it in their section below.")},
    {"category.Upscaling.footer",
     QT_TRANSLATE_NOOP("UiCatalog", "The game's picture goes through the upscaler, then the screen "
                                    "filter.")},
    {"category.FrameRate.footer",
     QT_TRANSLATE_NOOP("UiCatalog", "{setting:frame_rate_limit} paces every game without its own "
                                    "choice. Frame generation adds frames on top of the game's.")},
    {"category.Game.footer",
     QT_TRANSLATE_NOOP("UiCatalog", "Game-specific: these apply to this game only.")},
    {"category.GameEnhancements.footer",
     QT_TRANSLATE_NOOP("UiCatalog", "Built-in patches by kind. Each applies to every game that ships "
                                    "such a patch, currently: %1. Auto keeps each patch's default for "
                                    "this device. A game's own choice in its Updates, DLC & Mods "
                                    "overrides these. Changes apply the next time a game starts.")},
    {"category.Experiments.footer",
     QT_TRANSLATE_NOOP("UiCatalog", "To compare upscalers: {setting:upscaler} ({choice:upscaler."
                                    "MetalFx} or {choice:upscaler.Auto}) and {setting:scaling_filter} "
                                    "({choice:scaling_filter.MetalFxSpatial}). The log's \"Ocarina "
                                    "upscaler:\" lines count the frames {choice:upscaler.MetalFx} "
                                    "drew.")},
    {"list.none_yet", QT_TRANSLATE_NOOP("UiCatalog", "none yet")},

    // Notes under a setting, and why a setting isn't listed.
    {"note.lsfg_missing",
     QT_TRANSLATE_NOOP("UiCatalog", "{choice:frame_gen_type.Lsfg} needs Lossless.dll from Lossless "
                                    "Scaling in Lichii's eden/lossless folder (Files app). "
                                    "{choice:frame_gen_type.Fsr3} needs nothing.")},
    {"note.metalfx_frame_gen",
     QT_TRANSLATE_NOOP("UiCatalog", "{choice:frame_gen_type.MetalFx} makes the 2x frame on iOS 26. "
                                    "Before iOS 26, and at 3x or 4x, {choice:frame_gen_type.Fsr3} "
                                    "does.")},
    {"note.next_start", QT_TRANSLATE_NOOP("UiCatalog", "Applies the next time the game starts.")},
    {"note.stop_game", QT_TRANSLATE_NOOP("UiCatalog", "Stop the game to change this.")},
    {"note.game_overrides",
     QT_TRANSLATE_NOOP("UiCatalog", "The running game's {text:page.game_settings} override this "
                                    "setting: change it there.")},
    {"reason.frame_gen_build",
     QT_TRANSLATE_NOOP("UiCatalog", "Frame generation isn't part of this build.")},
    {"reason.frame_gen_chip",
     QT_TRANSLATE_NOOP("UiCatalog", "Frame generation needs an A17 Pro, M-series or newer chip.")},
    {"reason.reshade",
     QT_TRANSLATE_NOOP("UiCatalog", "ReShade post-processing isn't part of this build.")},
    {"reason.legacy_moltenvk",
     QT_TRANSLATE_NOOP("UiCatalog", "This build has only the legacy MoltenVK.")},
    {"reason.dlss_build",
     QT_TRANSLATE_NOOP("UiCatalog", "This build has no NVIDIA Streamline, which NVIDIA DLSS runs "
                                    "through.")},

    // Search.
    {"search.settings", QT_TRANSLATE_NOOP("UiCatalog", "Search settings")},
    {"search.games", QT_TRANSLATE_NOOP("UiCatalog", "Search games")},
    {"search.no_results", QT_TRANSLATE_NOOP("UiCatalog", "No Results for \u201c%1\u201d")},
    {"search.unavailable", QT_TRANSLATE_NOOP("UiCatalog", "Not Available on %1")},
    {"search.hidden_now", QT_TRANSLATE_NOOP("UiCatalog", "shown when it applies")},

    // A game's settings.
    {"game.note",
     QT_TRANSLATE_NOOP("UiCatalog", "Overrides apply to this game only. Only settings that apply to "
                                    "this game are listed.")},
    {"game.note.ios", QT_TRANSLATE_NOOP("UiCatalog", "Tap the globe next to a setting to override "
                                                     "it.")},
    {"game.note.desktop",
     QT_TRANSLATE_NOOP("UiCatalog", "A changed setting gets a button next to it that sets it back to "
                                    "the global value.")},
    {"game.info", QT_TRANSLATE_NOOP("UiCatalog", "Game Info")},
    {"game.addons", QT_TRANSLATE_NOOP("UiCatalog", "Updates, DLC & Mods")},

    // Recommended settings and resetting a game.
    {"recommended.question", QT_TRANSLATE_NOOP("UiCatalog", "Change these settings?")},
    {"recommended.use", QT_TRANSLATE_NOOP("UiCatalog", "Use Recommended Settings")},
    {"recommended.for", QT_TRANSLATE_NOOP("UiCatalog", "Recommended for %1:")},
    {"recommended.applies",
     QT_TRANSLATE_NOOP("UiCatalog", "Games use them from their next start; a game's own overrides "
                                    "stay until you reset that game in its settings.")},
    {"recommended.all_set", QT_TRANSLATE_NOOP("UiCatalog", "Everything is already set as "
                                                           "recommended.")},
    {"recommended.stop_game",
     QT_TRANSLATE_NOOP("UiCatalog", "Stop the game first: it has its own values loaded.")},
    {"recommended.why", QT_TRANSLATE_NOOP("UiCatalog", "Every recommended value and why:")},
    {"reset.button", QT_TRANSLATE_NOOP("UiCatalog", "Reset This Game to Recommended")},
    {"reset.footer",
     QT_TRANSLATE_NOOP("UiCatalog", "Gives this game the recommended settings for this device and "
                                    "removes its other overrides.")},
    {"reset.question", QT_TRANSLATE_NOOP("UiCatalog", "Use the recommended settings for this "
                                                      "game?")},
    {"reset.confirm", QT_TRANSLATE_NOOP("UiCatalog", "Reset to Recommended")},
    {"reset.already", QT_TRANSLATE_NOOP("UiCatalog", "This game already has the recommended "
                                                     "settings.")},
    {"reset.removes", QT_TRANSLATE_NOOP("UiCatalog", "Its other overrides are removed.")},
    {"change.line", QT_TRANSLATE_NOOP("UiCatalog", "%1: %2 \u2192 %3")},
    {"value.global", QT_TRANSLATE_NOOP("UiCatalog", "Global")},
    {"value.on", QT_TRANSLATE_NOOP("UiCatalog", "On")},
    {"value.off", QT_TRANSLATE_NOOP("UiCatalog", "Off")},

    // Library.
    {"library.empty", QT_TRANSLATE_NOOP("UiCatalog", "No games yet")},
    {"library.empty.ios",
     QT_TRANSLATE_NOOP("UiCatalog", "Import .nsp, .xci, .nup, .nca or .nro files, or copy them into "
                                    "Lichii's folder with the Files app.")},
    {"library.empty.desktop",
     QT_TRANSLATE_NOOP("UiCatalog", "Add a folder with your .nsp, .xci, .nup, .nca or .nro files: "
                                    "Lichii lists every game in it. Double-clicking here does the "
                                    "same.")},
    {"library.import", QT_TRANSLATE_NOOP("UiCatalog", "Import Games")},
    {"library.add_folder", QT_TRANSLATE_NOOP("UiCatalog", "Add Game Folder...")},
    {"library.grid", QT_TRANSLATE_NOOP("UiCatalog", "Grid")},
    {"library.list", QT_TRANSLATE_NOOP("UiCatalog", "List")},

    // The performance overlay's lines, in the order it shows them.
    {"overlay.real", QT_TRANSLATE_NOOP("UiCatalog", "Real %1 FPS")},
    {"overlay.generated", QT_TRANSLATE_NOOP("UiCatalog", "Generated +%1")},
    {"overlay.displayed", QT_TRANSLATE_NOOP("UiCatalog", "Displayed %1 FPS")},
    {"overlay.frame", QT_TRANSLATE_NOOP("UiCatalog", "Frame %1 ms")},
    {"overlay.frame_out", QT_TRANSLATE_NOOP("UiCatalog", "Frame %1 ms (%2 out)")},
    {"overlay.low10", QT_TRANSLATE_NOOP("UiCatalog", "10% low %1")},
    {"overlay.frame_gen", QT_TRANSLATE_NOOP("UiCatalog", "Frame gen %1")},
    {"overlay.frame_gen_rate", QT_TRANSLATE_NOOP("UiCatalog", "%1 %2x")},
    {"overlay.warming_up", QT_TRANSLATE_NOOP("UiCatalog", "%1 warming up")},
    {"overlay.lossless_missing", QT_TRANSLATE_NOOP("UiCatalog", "%1 needs Lossless.dll")},
    {"overlay.scale", QT_TRANSLATE_NOOP("UiCatalog", "Scale %1x")},
    {"overlay.game_dlss", QT_TRANSLATE_NOOP("UiCatalog", "Game DLSS")},
    {"overlay.dlss_idle", QT_TRANSLATE_NOOP("UiCatalog", "%1 idle")},
    {"overlay.dlss_off", QT_TRANSLATE_NOOP("UiCatalog", "%1 off")},
    {"overlay.speed", QT_TRANSLATE_NOOP("UiCatalog", "Speed %1%")},
    {"overlay.shaders", QT_TRANSLATE_NOOP("UiCatalog", "Shaders %1 building")},
    {"overlay.scene", QT_TRANSLATE_NOOP("UiCatalog", "3D %1")},
    {"overlay.game", QT_TRANSLATE_NOOP("UiCatalog", "Game %1")},
    {"overlay.screen", QT_TRANSLATE_NOOP("UiCatalog", "Screen %1")},
    {"overlay.gpu_accuracy", QT_TRANSLATE_NOOP("UiCatalog", "GPU accuracy %1")},
    {"overlay.cpu", QT_TRANSLATE_NOOP("UiCatalog", "CPU %1%")},
    {"overlay.ram", QT_TRANSLATE_NOOP("UiCatalog", "RAM %1 / %2 GB")},
    {"overlay.ram_used", QT_TRANSLATE_NOOP("UiCatalog", "RAM %1 GB")},
    {"overlay.vram", QT_TRANSLATE_NOOP("UiCatalog", "VRAM %1 / %2 GB")},
    {"overlay.vram_used", QT_TRANSLATE_NOOP("UiCatalog", "VRAM %1 GB")},
    {"overlay.textures", QT_TRANSLATE_NOOP("UiCatalog", "Textures %1 / %2 MB")},
    {"overlay.textures_used", QT_TRANSLATE_NOOP("UiCatalog", "Textures %1 MB")},
    {"overlay.fps", QT_TRANSLATE_NOOP("UiCatalog", "%1 FPS")},
    {"overlay.fps_generated", QT_TRANSLATE_NOOP("UiCatalog", "%1 \u2192 %2 FPS")},
    {"overlay.preset.off", QT_TRANSLATE_NOOP("UiCatalog", "Off")},
    {"overlay.preset.minimal", QT_TRANSLATE_NOOP("UiCatalog", "Minimal")},
    {"overlay.preset.standard", QT_TRANSLATE_NOOP("UiCatalog", "Standard")},
    {"overlay.preset.detailed", QT_TRANSLATE_NOOP("UiCatalog", "Detailed")},

    // Memory & Experiments: Keep Guest Memory in a File and the sections of switches.
    {"experiments.large_games.title", QT_TRANSLATE_NOOP("UiCatalog", "Large Games")},
    {"experiments.large_games.subtitle", QT_TRANSLATE_NOOP("UiCatalog", "Recommended for large "
                                                                        "games")},
    {"experiments.large_games.footer",
     QT_TRANSLATE_NOOP("UiCatalog", "A game's own memory is most of what Lichii uses while it runs, "
                                    "and iOS closes Lichii when it uses too much. This moves that "
                                    "memory into a temporary file in storage, which iOS doesn't "
                                    "count: large games such as Ocarina of Time need it to get past "
                                    "their first areas.\n\nThe file can grow to %1 and needs %2 of "
                                    "free storage when the game starts. Expect some stutter while a "
                                    "game loads, and a few GB written to storage per session (about "
                                    "2.5 GB in Ocarina of Time). The file is deleted when the game "
                                    "stops. Takes effect the next time a game starts%3.")},
    {"experiments.large_games.quit_first", QT_TRANSLATE_NOOP("UiCatalog", "; quit the game to change "
                                                                          "it")},
    {"experiments.audio.title", QT_TRANSLATE_NOOP("UiCatalog", "Audio")},
    {"experiments.audio.footer",
     QT_TRANSLATE_NOOP("UiCatalog", "Music or podcasts from other apps keep playing under the "
                                    "game.")},
    {"experiments.mix_audio", QT_TRANSLATE_NOOP("UiCatalog", "Mix with Other Audio")},
    {"experiments.memory_savings.title", QT_TRANSLATE_NOOP("UiCatalog", "Memory Savings")},
    {"experiments.memory_savings.footer",
     QT_TRANSLATE_NOOP("UiCatalog", "Untested on a device; applies the next time Lichii "
                                    "starts. Splits guest memory into 2 MB pieces and "
                                    "returns the ones holding only zeros to iOS (about 0.5 "
                                    "GB in Ocarina of Time). To do that safely Lichii stops "
                                    "all of its threads for about a millisecond at a time, "
                                    "which may show as a stutter. No effect while "
                                    "{exp:file_guest} or {exp:untagged_guest} is on.")},
    {"experiments.memory.title", QT_TRANSLATE_NOOP("UiCatalog", "Memory")},
    {"experiments.memory.footer",
     QT_TRANSLATE_NOOP("UiCatalog", "Guest memory normally counts once toward the limit iOS "
                                    "closes Lichii at. Per mapping is the old accounting: "
                                    "pages both the game and the emulator touch count twice, "
                                    "and all-zero pages are trimmed.\n"
                                    "\n"
                                    "Compressed files of a .nup game are normally kept in "
                                    "memory up to 96 MB once read, and files over 48 MB (a "
                                    "movie, a texture archive) are unpacked to temporary "
                                    "files in storage instead. {exp:nup_in_memory} is the "
                                    "old behaviour: up to 512 MB in memory. Turn it on if a "
                                    "game that used to start no longer does.\n"
                                    "\n"
                                    "{exp:malloc_census} writes what Lichii's own memory "
                                    "holds, by block size, to the log 60 s and 180 s into a "
                                    "game (always once when memory runs low). While it "
                                    "counts, threads that need memory wait.\n"
                                    "\n"
                                    "All apply the next time Lichii starts.")},
    {"experiments.graphics.title", QT_TRANSLATE_NOOP("UiCatalog", "Graphics")},
    {"experiments.graphics.footer",
     QT_TRANSLATE_NOOP("UiCatalog", "For tracking down crashes. Early submits hand the GPU a "
                                    "scene load's texture uploads in parts, which keeps "
                                    "graphics memory down while a level loads; off, a large "
                                    "load can run out of memory. Mesh shader emulation draws "
                                    "some of Ocarina of Time's foliage and walls; off, they "
                                    "are missing. {exp:mvk_uint8_indices} is the old way of "
                                    "drawing models stored with 8-bit indices, which crashed "
                                    "Ocarina of Time entering Kokiri Forest; Lichii now "
                                    "converts them itself. {exp:native_per_primitive} builds "
                                    "one of Ocarina of Time's mesh shaders the way it was "
                                    "built before, which the iPhone's shader compiler "
                                    "refused at every start; Lichii now draws it with the "
                                    "emulation. {exp:mvk_shader_dump} saves every shader as "
                                    "MoltenVK hands it to Metal in dump/mvk-shaders "
                                    "(thousands of files, slower start), to find out why "
                                    "that one is refused; turn both on for that. "
                                    "{exp:metal_trace_off} is for the Metal renderer "
                                    "(Graphics API): normally Lichii writes a log line when "
                                    "the app aborts or an exception goes uncaught, and asks "
                                    "Metal which step of a failed command buffer failed; on, "
                                    "it does neither, as before. {exp:metal_check_off} is "
                                    "for the Metal renderer too: Metal closes the app when a "
                                    "draw's vertex layout, targets or a texture are outside "
                                    "what it takes (Ocarina of Time's first draw was), so "
                                    "Lichii now rewrites the vertex layout to what the "
                                    "shader reads, keeps 1D textures as 2D ones a pixel high "
                                    "and skips a draw or a texture view Metal would still "
                                    "reject, with a line in the log; on, Metal gets the "
                                    "state as the game set it, as before.\n"
                                    "\n"
                                    "All apply the next time Lichii starts.")},
    {"experiments.performance.title", QT_TRANSLATE_NOOP("UiCatalog", "Performance")},
    {"experiments.performance.footer",
     QT_TRANSLATE_NOOP("UiCatalog", "For comparing builds. {exp:tc_old_eviction} reads "
                                    "textures the GPU drew back from the GPU before freeing "
                                    "them, a wait for the GPU each time, even when the game "
                                    "draws them again anyway. {exp:helper_qos_off} gives the "
                                    "timing and present threads the same priority as the "
                                    "game's. {exp:frame_deadlines} tells iOS each frame's "
                                    "deadline, which may keep the game on the performance "
                                    "cores; untested on a device. {exp:jit_old_tables} keeps "
                                    "the translated code's bookkeeping as it was (about 0.3 "
                                    "GB more memory in Ocarina of Time); turn it on if a "
                                    "game that used to run crashes. {exp:fence_submit_each} "
                                    "hands the GPU the game's work at every game fence, as "
                                    "before, instead of in batches of up to 2 ms (76-95 "
                                    "command buffers a frame in Ocarina of Time's Kokiri "
                                    "Forest on an iPhone); turn it on if the screen flashes "
                                    "white. {exp:render_passes_old} ends the GPU's render "
                                    "pass for every buffer copy and parked clear, as before, "
                                    "instead of moving copies ahead of the pass (each ended "
                                    "pass stores and loads the whole screen again); turn it "
                                    "on if parts of the picture show stale or wrong data. "
                                    "{exp:oot_game_upscale} runs Ocarina of Time's own "
                                    "upscale from its scene to the screen size, as before, "
                                    "instead of one bilinear tap whenever MetalFX Temporal "
                                    "doesn't take the frame (the game's pass was about 40 % "
                                    "of the phone's GPU time in Kokiri Forest); turn it on "
                                    "if the picture looks wrong. {exp:targetless_full} gives "
                                    "a draw that writes no image a render pass of the game's "
                                    "full surface size, as before (32768x32768 in Ocarina of "
                                    "Time, about 6-7 % of the phone's GPU time), instead of "
                                    "only the pixels it can reach; turn it on if lens flares "
                                    "or glows behave differently. {exp:ubo_zero_tails} fills "
                                    "the unused rest of each uniform buffer a shader indexes "
                                    "with zeros, as before (about 75 % of the uniform bytes "
                                    "Lichii wrote in Kokiri Forest); turn it on if models or "
                                    "effects look wrong. {exp:release_submit} hands the GPU "
                                    "its work every time the game waits on a GPU value, as "
                                    "before, even when nothing waits for that work (about 11 "
                                    "command buffers a frame in Kokiri Forest); turn it on "
                                    "if the screen flashes or stutters. "
                                    "{exp:idle_submit_grace} lets the GPU thread, once it "
                                    "runs out of commands, wait up to 0.25 ms for more "
                                    "before it hands the GPU what it has (about 5 fewer "
                                    "command buffers a frame on a PC, but it keeps a CPU "
                                    "core busy meanwhile); untested on a device. "
                                    "{exp:idle_hold_off} hands the GPU what the GPU thread "
                                    "has as soon as it runs out of commands, as before, "
                                    "instead of keeping it, asleep, up to 0.5 ms while the "
                                    "GPU still runs what it was handed earlier (about half "
                                    "of the phone's command buffers a frame in Kokiri Forest "
                                    "were such hand-overs; on a PC 2 of 3 of them went "
                                    "away); turn it on if the game stutters or runs slower "
                                    "than before. {exp:mesh_compute_in_pass} ends the render "
                                    "pass for each emulated mesh draw's compute, as before, "
                                    "instead of running that compute just before the pass; "
                                    "turn it on if foliage looks wrong. {exp:dispatch_chain} "
                                    "lets the game's consecutive compute dispatches share "
                                    "one Metal compute pass (the ~150 dispatches a frame in "
                                    "Ocarina of Time otherwise get a pass each); untested on "
                                    "a device, turn it on to compare and off if effects "
                                    "flicker. {exp:emulate_all_mesh} draws all mesh shaders "
                                    "(most foliage) through the compute emulation instead of "
                                    "Metal mesh shaders, which hold their outputs in scarce "
                                    "GPU tile memory; untested on a device, for comparing "
                                    "the two. {exp:dispatch_deps} writes a \"Dispatch deps\" "
                                    "line every 10 s naming each compute dispatch that ends "
                                    "the screen's render pass and whether it reads that "
                                    "pass's images; it changes nothing that is drawn.\n"
                                    "\n"
                                    "All apply the next time Lichii starts.")},
    {"experiments.cpu_scheduling.title", QT_TRANSLATE_NOOP("UiCatalog", "CPU Scheduling")},
    {"experiments.cpu_scheduling.footer",
     QT_TRANSLATE_NOOP("UiCatalog", "For finding out why the game's CPU threads run on the "
                                    "efficiency cores. {exp:guest_cores_first} leaves the "
                                    "top priority to two of the six emulated CPU cores and "
                                    "the graphics threads, and runs the other four one step "
                                    "lower, so iOS has fewer threads asking for its two "
                                    "performance cores (on a computer limited to four CPUs "
                                    "this gave Ocarina of Time 25 % more frames). "
                                    "{exp:oot_narrow_affinity} lets Ocarina of Time's main "
                                    "threads run only on the cores the game asks for, "
                                    "instead of on any core. {exp:oot_ideal_affinity} keeps "
                                    "each such thread on one core, so one host thread does "
                                    "its work; wins over {exp:oot_narrow_affinity}. "
                                    "{exp:guest_idle_spin} keeps an emulated core that runs "
                                    "out of work polling for 20 µs before it sleeps, as on "
                                    "a computer, which saves a wake-up per hand-off but "
                                    "keeps a CPU busy. {exp:perf_sched_off} stops the \"Perf "
                                    "threads:\" and \"Perf sched:\" log lines and the "
                                    "counting behind them (thread listing every 10 s, a 20 "
                                    "Hz thread sampler, timestamps at each guest thread "
                                    "switch). {exp:frame_time_off} stops Lichii's three "
                                    "graphics threads from telling iOS how much CPU time "
                                    "they need each frame, as before; with it iOS schedules "
                                    "them like audio threads, which on another emulator kept "
                                    "the performance cores faster and busier (\"frame "
                                    "deadlines:\" in the log's Perf line shows what each "
                                    "declared). Turn it on if the game stutters or runs "
                                    "slower than before. {exp:frame_time_guest} does the "
                                    "same for the emulated CPU cores; untested, and iOS may "
                                    "take it back from a core that never rests. "
                                    "{exp:split_std_waits} lets the two graphics threads "
                                    "poll for up to 64 µs before they sleep waiting on each "
                                    "other, as before. {exp:poll_yield} lets the graphics "
                                    "thread that waits for the game to release a GPU value, "
                                    "and threads waiting for the graphics thread to send its "
                                    "work, give up the CPU between checks instead of "
                                    "sleeping 50 µs, as before; when other threads are "
                                    "waiting for a CPU, iOS then runs the yielding thread "
                                    "last for up to 10 ms. It covers the other short waits "
                                    "too (frame pacing, the present hold, a guest write "
                                    "waiting for GPU data to land, shader handover, ring "
                                    "space, kernel thread starts).\n"
                                    "\n"
                                    "All apply the next time Lichii starts.")},
    {"experiments.guest_memory.title", QT_TRANSLATE_NOOP("UiCatalog", "Guest Memory Access")},
    {"experiments.guest_memory.footer",
     QT_TRANSLATE_NOOP("UiCatalog", "{exp:jit_inline_exclusives} runs the game's exclusive "
                                    "loads and stores (LDXR/STXR, for locks and counters) "
                                    "directly on guest memory, as on a computer, instead of "
                                    "a call into the emulator each; Ocarina of Time uses "
                                    "none on a computer, other games may. "
                                    "{exp:jit_load_faults} lets a game load that hit "
                                    "protected memory (a page the GPU wrote) keep reading "
                                    "guest memory directly, and take the slow path only "
                                    "after its eighth such fault instead of the first; on a "
                                    "computer it cut Ocarina of Time's slow loads of GPU "
                                    "pages by 70 % but saved little time. Both are off by "
                                    "default. The \"Perf memory callbacks:\" log line counts "
                                    "both.\n"
                                    "\n"
                                    "{exp:cpu_writes_old} keeps every game write to a memory "
                                    "page that also holds data the GPU wrote on the slow "
                                    "path until the next GPU sync, as before. Normally "
                                    "Lichii copies the GPU's data on such a page back first "
                                    "and then lets the game write the page directly (on a "
                                    "computer this halved Ocarina of Time's slow writes in "
                                    "Kokiri Forest); turn it on if objects flicker or show "
                                    "stale data. The \"Perf CPU writes:\" log line counts "
                                    "these pages.\n"
                                    "\n"
                                    "{exp:gpu_reads_old}: when the game reads data the GPU "
                                    "wrote and that data is still on its way back, the "
                                    "graphics thread copies that memory page again and waits "
                                    "until the GPU has finished everything, page after page, "
                                    "as before. Normally the game's thread waits only for "
                                    "the copy that is already under way and takes the data "
                                    "itself, so the graphics thread keeps working (on a "
                                    "computer this cut the time the graphics thread was held "
                                    "by such reads in Kokiri Forest by 93 %); turn it on if "
                                    "the picture's brightness jumps or objects show stale "
                                    "data. The \"Perf CPU reads:\" log line counts these "
                                    "reads and their waits.\n"
                                    "\n"
                                    "{exp:hot_pages_off} write-protects every memory page "
                                    "the game rewrote after each copy to the GPU, as before, "
                                    "so each new write takes the slow path. Normally a page "
                                    "the game rewrites over and over is left unprotected and "
                                    "copied again once per command list it is used in (on a "
                                    "computer this halved Ocarina of Time's slow writes in "
                                    "Kokiri Forest, for about 40 % more buffer copies); turn "
                                    "it on if objects jump, flicker or show stale data. The "
                                    "\"Perf CPU writes:\" log line counts these hot pages.\n"
                                    "\n"
                                    "All apply the next time Lichii starts.")},
    {"experiments.metal.title", QT_TRANSLATE_NOOP("UiCatalog", "Metal Renderer")},
    {"experiments.metal.footer",
     QT_TRANSLATE_NOOP("UiCatalog", "Only for the Metal renderer (Graphics API). "
                                    "{exp:metal_fixed_state_off} builds the game's shaders "
                                    "as before, without what the draw sets outside the "
                                    "shader: the depth range the game uses, the size of "
                                    "points, the alpha test, blending with two colours and "
                                    "whole-number colour targets. Turn it on if the picture "
                                    "got worse with a new build.\n"
                                    "\n"
                                    "{exp:metal_cache_plain}: when the game reads or writes "
                                    "memory the graphics side also uses, the Metal renderer "
                                    "now handles it the way the Vulkan renderer does (the "
                                    "game's threads wait less for the graphics thread, and "
                                    "the switches under Guest Memory Access apply to it "
                                    "too). On, it handles every such read and write the slow "
                                    "way it did before; turn it on if objects flicker, show "
                                    "stale data or the game hangs with a new build.\n"
                                    "\n"
                                    "{exp:metal_msl_plain}: the game's shaders are now "
                                    "turned into Metal shaders with what this iPhone's GPU "
                                    "can do, as the Vulkan renderer's MoltenVK does it. "
                                    "Before, every shader that reads which copy of a model "
                                    "it is drawing was refused (about a fifth of Ocarina of "
                                    "Time's, their draws skipped), and shaders that share "
                                    "values between neighbouring pixels did it in groups of "
                                    "four. On, they are translated as before; turn it on if "
                                    "shaders fail to build or the picture got worse with a "
                                    "new build.\n"
                                    "\n"
                                    "{exp:metal_copies_plain}: the Metal renderer now copies "
                                    "one depth image into another when the game asks for it, "
                                    "and copies between a compressed image and a plain one "
                                    "whose pixels hold a compressed block each (Ocarina of "
                                    "Time does both). Before, both kinds of copy were "
                                    "skipped. On, they are skipped again; turn it on if the "
                                    "game stops with a new build.\n"
                                    "\n"
                                    "{exp:metal_samplers_plain}: the Metal renderer now "
                                    "gives each texture read the sampler Vulkan would: the "
                                    "game's level-of-detail bias and minimum or maximum "
                                    "filtering (on iOS 26), border colours and mirror modes "
                                    "as close as Metal has them, and a variant for each kind "
                                    "of image read. On, every guest sampler is one plain "
                                    "sampler, as before; turn it on if textures look "
                                    "blurred, too sharp or wrongly tiled with a new build.\n"
                                    "\n"
                                    "{exp:metal_msaa_off}: images with several samples per "
                                    "pixel are now uploaded, read back, copied and resolved "
                                    "as on Vulkan (Ocarina of Time uses none). On, none of "
                                    "that is done, as before.\n"
                                    "\n"
                                    "{exp:metal_formats_plain}: images in two formats Metal "
                                    "has no texture for (three 32-bit floats, and two 4-bit "
                                    "channels) are now converted to the nearest format it "
                                    "has. On, they get no texture or the wrong one, as "
                                    "before.\n"
                                    "\n"
                                    "{exp:metal_layers_off}: drawing into images with layers "
                                    "now reaches every layer a shader picks. On, a render "
                                    "pass gets one layer of each image, as before.\n"
                                    "\n"
                                    "{exp:metal_multirange_off}: a buffer the game reads "
                                    "from several places in memory is now gathered into one. "
                                    "On, only its first part is read, as before.\n"
                                    "\n"
                                    "{exp:metal_clears_plain}: when the game clears only a "
                                    "rectangle of an image, only some of its colour channels "
                                    "or only some stencil bits, the Metal renderer now does "
                                    "exactly that (Ocarina of Time clears rectangles many "
                                    "times a frame). Before, it cleared the whole image. A "
                                    "clear of a whole image now also waits for the drawing "
                                    "that follows it and is done as that drawing starts, "
                                    "instead of as a step of its own. On, it clears the "
                                    "whole image again, each clear as its own step; turn it "
                                    "on if parts of the picture go missing with a new "
                                    "build.\n"
                                    "\n"
                                    "{exp:metal_barrier_off}: when the game draws into an "
                                    "image and then reads that same image in a later draw, "
                                    "it tells the graphics chip so first. The Metal renderer "
                                    "now finishes the drawing done so far at that point, so "
                                    "the later draw reads it. Before, such a draw read the "
                                    "image as it was before any of that drawing. On, the "
                                    "barriers are ignored again; turn it on if the game got "
                                    "slower with a new build.\n"
                                    "\n"
                                    "{exp:metal_state_plain}: the Metal renderer now tells "
                                    "the graphics chip only what changed since the draw "
                                    "before (which shader, which buffers and images, the "
                                    "drawing area and so on). On, it sets everything again "
                                    "for every draw, as before; turn it on if objects are "
                                    "drawn with the wrong image or in the wrong place with a "
                                    "new build.\n"
                                    "\n"
                                    "{exp:metal_ahead_off}: data the game changed, and the "
                                    "small programs that prepare some draws (grass and other "
                                    "models built on the graphics chip, some index lists), "
                                    "used to interrupt the drawing into an image each time; "
                                    "the image was put away and fetched again. Now they are "
                                    "sent ahead of that drawing where nothing in between "
                                    "depends on the order. On, they interrupt the drawing "
                                    "again, as before; turn it on if objects flicker, show "
                                    "stale data or parts of the picture go missing with a "
                                    "new build.\n"
                                    "\n"
                                    "{exp:metal_fences_plain}: the game marks points in its "
                                    "drawing and waits for the graphics chip to reach them. "
                                    "When nothing was drawn since the last such point, the "
                                    "Metal renderer now waits for the work already sent "
                                    "instead of sending an empty batch. On, it sends one "
                                    "every time, as before; turn it on if the game hangs or "
                                    "stutters with a new build.\n"
                                    "\n"
                                    "{exp:metal_staging_plain}: while a place loads, the "
                                    "game sends thousands of images and models at once. The "
                                    "Metal renderer now sends what it has to the graphics "
                                    "chip every 24 MiB so the memory can be used again, "
                                    "waits for it when 96 MiB are held, and gives unused "
                                    "memory back after a few frames. On, there are no such "
                                    "bounds, as before; turn it on if loading got slower or "
                                    "stutters with a new build (Lichii then needs more "
                                    "memory while loading).\n"
                                    "\n"
                                    "{exp:metal_network_runs}: with Ocarina of Time's "
                                    "anti-aliasing set to No sharpening, the Metal renderer "
                                    "now skips the game's anti-aliasing network and shows "
                                    "the picture from before it, as the Vulkan renderer "
                                    "does. Before, the setting did nothing on Metal. On, the "
                                    "network runs again whatever the setting says; turn it "
                                    "on if the picture is wrong or stays black with that "
                                    "setting and a new build.\n"
                                    "\n"
                                    "{exp:metal_prebuild_off}: the Metal renderer now notes "
                                    "every shader it builds and, when the game starts the "
                                    "next time, builds them all again while the game loads, "
                                    "on several cores, so play does not stop for them. On, "
                                    "nothing is noted or built ahead, as before: each shader "
                                    "is built when it is first drawn, and play waits for it; "
                                    "turn it on if the game stops while loading with a new "
                                    "build.\n"
                                    "\n"
                                    "{exp:metal_temporal_off}: with the upscaler set to "
                                    "MetalFX Temporal, the Metal renderer now hands Ocarina "
                                    "of Time's picture to MetalFX as the Vulkan renderer "
                                    "does. On, the game's own upscaling (or the simple one) "
                                    "runs instead, as before; turn it on if the picture is "
                                    "wrong or stutters with a new build.\n"
                                    "\n"
                                    "{exp:metal_mesh_emulated}: Ocarina of Time draws its "
                                    "grass and some other models with mesh shaders. The "
                                    "Metal renderer now hands them to the iPhone's own mesh "
                                    "shader stage. Before, each was run as a compute program "
                                    "whose output a second draw then drew. On, they go "
                                    "through that emulation again; turn it on if grass or "
                                    "other models are missing or wrong with a new build.\n"
                                    "\n"
                                    "{exp:metal_vertex_plain}: the Metal renderer now draws "
                                    "models stored in three forms Metal can't read directly "
                                    "(with neighbouring-vertex data, with 10-bit whole "
                                    "numbers, or spread over more buffers than Metal takes), "
                                    "converting them first. Ocarina of Time uses none of "
                                    "them. On, such draws are skipped or read wrongly again, "
                                    "as before; turn it on if models are wrong with a new "
                                    "build.\n"
                                    "\n"
                                    "{exp:metal_async_off}: a shader the game uses for the "
                                    "first time is now built on other cores; draws that may "
                                    "be left out meanwhile are, the others wait for it at "
                                    "full priority. On, it is built where the draw is and "
                                    "the game waits, as before; turn it on if objects go "
                                    "missing for longer with a new build.\n"
                                    "\n"
                                    "{exp:metal_provoking_first}: values the game does not "
                                    "blend across a triangle are now taken from the corner "
                                    "the game asks for, as on a PC. On, they come from the "
                                    "first corner, as before and as on MoltenVK; turn it on "
                                    "if draws got slower or look wrong with a new build.\n"
                                    "\n"
                                    "{exp:metal_condition_cpu}: draws the game makes only if "
                                    "an earlier one showed any pixels are now decided by the "
                                    "graphics chip once it has counted them. On, they are "
                                    "decided from what memory holds at the time, as before; "
                                    "turn it on if objects flicker or the game stutters with "
                                    "a new build.\n"
                                    "\n"
                                    "{exp:metal_indirect_cpu}: draws whose sizes the "
                                    "graphics chip wrote are now prepared on it (their "
                                    "count, their index lists). On, the sizes are read on "
                                    "the CPU, which waits for the chip, and some of those "
                                    "draws are skipped, as before; turn it on if objects go "
                                    "missing or flicker with a new build.\n"
                                    "\n"
                                    "{exp:metal_draw_texture_off}: when the game draws a "
                                    "picture straight onto the screen image, the Metal "
                                    "renderer now does it. On, such draws are skipped, as "
                                    "before.\n"
                                    "\n"
                                    "{exp:metal_clear_layers_off}: a clear of an image with "
                                    "layers now clears the layers the game names. On, only "
                                    "the first, as before.\n"
                                    "\n"
                                    "{exp:metal_list_copies_off}: drawing commands the "
                                    "graphics chip writes for itself are now read from "
                                    "copies made right after they are written, as the Vulkan "
                                    "renderer does. On, the CPU waits for all work first, as "
                                    "before; turn it on if the game hangs or draws the wrong "
                                    "things with a new build.\n"
                                    "\n"
                                    "{exp:metal_viewports_off}: shaders that pick one of "
                                    "several drawing areas now get all sixteen. On, only the "
                                    "first, as before.\n"
                                    "\n"
                                    "{exp:metal_fill_lines_off}: triangles the game asks to "
                                    "draw as outlines are now drawn so. On, they are filled, "
                                    "as before.\n"
                                    "\n"
                                    "{exp:metal_tess_base_off}: tessellated models now start "
                                    "at the vertex the game names. On, they start at vertex "
                                    "0, as before; turn it on if tessellated ground or water "
                                    "looks torn with a new build.\n"
                                    "\n"
                                    "{exp:metal_alpha_to_one_off}: when the game asks for "
                                    "the alpha of drawn pixels to be set to one, the Metal "
                                    "renderer now does it. On, it is left as drawn, as "
                                    "before.\n"
                                    "\n"
                                    "{exp:metal_vulkan_switches_off}: the Metal renderer now "
                                    "follows the Vulkan renderer's switches that it has a "
                                    "counterpart for (on this iPhone that changes one thing: "
                                    "fences sent together, below). On, they are ignored, as "
                                    "before.\n"
                                    "\n"
                                    "{exp:metal_fence_batch_off}: the game's fences that "
                                    "come within 2 ms of the last batch sent to the graphics "
                                    "chip now go with the next one, as on the Vulkan "
                                    "renderer, instead of each in a batch of its own. On, "
                                    "each is sent at once, as before; turn it on if the game "
                                    "hangs or flashes with a new build.\n"
                                    "\n"
                                    "{exp:metal_filters_off}: the scaling filter and "
                                    "anti-aliasing settings (FSR, SGSR, FXAA, SMAA, bicubic, "
                                    "Lanczos and the others) now apply under the Metal "
                                    "renderer as under Vulkan. Before, the picture was "
                                    "always scaled to the screen bilinearly. On, it is "
                                    "again; turn it on if the picture is wrong with one of "
                                    "these settings and a new build.\n"
                                    "\n"
                                    "{exp:metal_standin_off}: where the upscaler is not "
                                    "MetalFX Temporal, or MetalFX cannot run, the game's own "
                                    "DLSS is now replaced by the Vulkan renderer's temporal "
                                    "upscaler. On, the game's DLSS runs again, as before; "
                                    "turn it on if the picture is wrong or slower with a new "
                                    "build.\n"
                                    "\n"
                                    "{exp:metal_bindmemo_off}: the Metal renderer now "
                                    "remembers which image and sampler each of the game's "
                                    "texture slots stood for, as the Vulkan renderer does, "
                                    "and looks again only when the game changed them. On, it "
                                    "looks every draw, as before; turn it on if objects show "
                                    "the wrong image with a new build.\n"
                                    "\n"
                                    "{exp:metal_pacing_off}: a game running below the "
                                    "screen's rate now shows every frame for the same number "
                                    "of refreshes, as under the Vulkan renderer (30 fps on a "
                                    "120 Hz screen: four each, not three and five by turns). "
                                    "On, each frame is shown as soon as it is ready, as "
                                    "before; turn it on if the game feels slower to respond "
                                    "with a new build.\n"
                                    "\n"
                                    "{exp:metal_colorspace_off}: the picture is now marked "
                                    "as sRGB, as the Vulkan renderer marks it, so the screen "
                                    "shows its colours as meant. On, it is left unmarked, as "
                                    "before.\n"
                                    "\n"
                                    "{exp:metal_evict_off}: when iOS says memory is low, the "
                                    "Metal renderer now frees the drawing pipelines no draw "
                                    "used for 10 s or more, as the Vulkan renderer does. On, "
                                    "it keeps them all, as before; turn it on if the game "
                                    "stutters after a memory warning with a new build.\n"
                                    "\n"
                                    "All of them apply the next time Lichii starts.\n"
                                    "\n"
                                    "{exp:metal_actor_off}: with MetalFX Temporal in Ocarina "
                                    "of Time, the Metal renderer now tells MetalFX how Link "
                                    "himself moved (his animation from one frame to the "
                                    "next), as the Vulkan renderer does, so he trails less. "
                                    "On, his pixels get the camera's motion only, as before; "
                                    "turn it on if Link smears or flickers with a new "
                                    "build.\n"
                                    "\n"
                                    "{exp:metal_effects_off}: with MetalFX Temporal in "
                                    "Ocarina of Time (iOS 18 and later), the Metal renderer "
                                    "now marks where Navi's glow, fireflies and dust were "
                                    "drawn so MetalFX takes those pixels from this frame "
                                    "instead of blending in earlier ones, as the Vulkan "
                                    "renderer does. On, no such mask is made, as before; "
                                    "turn it on if glows flicker or the game got slower with "
                                    "a new build.")},
    {"experiments.shader_builds.title", QT_TRANSLATE_NOOP("UiCatalog", "Shader Builds")},
    {"experiments.shader_builds.footer",
     QT_TRANSLATE_NOOP("UiCatalog", "A draw whose shaders are not built yet either is left "
                                    "out for a frame or waits. Normally the build a draw "
                                    "waits for runs at once on two threads of its own at the "
                                    "waiting thread's priority, and the background builds "
                                    "pause meanwhile. {exp:pipeline_holds_old} leaves it in "
                                    "line behind the other builds on the low-priority build "
                                    "threads, as before (waits of 0.1-2.2 s each in Kokiri "
                                    "Forest on an iPhone); turn it on if the game stutters "
                                    "more than before.\n"
                                    "\n"
                                    "Lichii keeps a list of the shaders a game really drew "
                                    "and builds those on the loading screen. Normally the "
                                    "list is saved within 10 s of a new shader's first use "
                                    "and when Lichii leaves the foreground. "
                                    "{exp:pipeline_list_old} saves it only when a new shader "
                                    "is first drawn 15 s or more after the last save or the "
                                    "game is stopped from the menu, so the shaders first "
                                    "drawn since the last save are missing from it when "
                                    "Lichii is closed from the app switcher.\n"
                                    "\n"
                                    "{exp:mvk_stats_off} stops MoltenVK timing its work (a "
                                    "few timestamps per command buffer), which the \"Perf "
                                    "pipelines:\" log line reports per stage.\n"
                                    "\n"
                                    "All apply the next time Lichii starts.")},
    {"experiments.thread_cuts.title", QT_TRANSLATE_NOOP("UiCatalog", "Graphics Thread Cuts")},
    {"experiments.thread_cuts.footer",
     QT_TRANSLATE_NOOP("UiCatalog", "Small savings on the graphics threads, ported from "
                                    "another Lichii fork; none was tried on an iPhone yet. "
                                    "Each switch turns one back to how it was.\n"
                                    "\n"
                                    "{exp:query_cuts_off}: occlusion and counter reports are "
                                    "again looked up, freed and numbered one at a time under "
                                    "a lock. {exp:fence_cuts_off}: each new fence wakes the "
                                    "thread that waits for fences even when it is busy "
                                    "anyway. {exp:protect_locked}: when the game writes "
                                    "memory the GPU caches use, the page protection system "
                                    "call runs while the buffer cache stays locked for "
                                    "everyone else. {exp:words_pool_off}: the copies of the "
                                    "game's command lists go back to the system heap instead "
                                    "of being kept for reuse. {exp:resume_publish_off}: "
                                    "after the game releases a GPU wait, the graphics thread "
                                    "copies the waiting lists before it hands over what it "
                                    "already decoded. {exp:collect_split_off}: a game write "
                                    "crossing a 2 KB boundary is noted as one piece, as "
                                    "before, which can lose part of it (turn it on only to "
                                    "compare). {exp:usage_reset_full}: the per-frame record "
                                    "of which parts of a buffer were used is cleared whole "
                                    "instead of only where it was set.\n"
                                    "\n"
                                    "Turn one on if the picture shows stale data, objects "
                                    "flicker or the game stops; all apply the next time "
                                    "Lichii starts.")},
};

const Experiment kExperiments[] = {
    {"file_guest", "large_games", QT_TRANSLATE_NOOP("UiCatalog", "Keep Guest Memory in a File")},
    {"chunked_guest", "memory_savings", QT_TRANSLATE_NOOP("UiCatalog", "Give Back Zeroed Guest Memory")},
    {"untagged_guest", "memory", QT_TRANSLATE_NOOP("UiCatalog", "Count Guest Memory per Mapping")},
    {"no_zero_trim", "memory", QT_TRANSLATE_NOOP("UiCatalog", "Zero Page Trimming Off")},
    {"nup_in_memory", "memory", QT_TRANSLATE_NOOP("UiCatalog", "Keep Decompressed Game Files in Memory")},
    {"malloc_census", "memory", QT_TRANSLATE_NOOP("UiCatalog", "Count Memory Blocks in the Log")},
    {"no_early_submits", "graphics", QT_TRANSLATE_NOOP("UiCatalog", "Early Submits During Loads Off")},
    {"no_mesh_emulation", "graphics", QT_TRANSLATE_NOOP("UiCatalog", "Mesh Shader Emulation Off")},
    {"mvk_uint8_indices", "graphics", QT_TRANSLATE_NOOP("UiCatalog", "MoltenVK Converts 8-bit Indices")},
    {"native_per_primitive", "graphics", QT_TRANSLATE_NOOP("UiCatalog", "Native Per-Primitive Mesh Attributes")},
    {"mvk_shader_dump", "graphics", QT_TRANSLATE_NOOP("UiCatalog", "MoltenVK Writes Its Shaders to Storage")},
    {"metal_trace_off", "graphics", QT_TRANSLATE_NOOP("UiCatalog", "No Metal Crash Log Lines")},
    {"metal_check_off", "graphics", QT_TRANSLATE_NOOP("UiCatalog", "Metal Takes the Game's Draw State Unchecked")},
    {"tc_old_eviction", "performance", QT_TRANSLATE_NOOP("UiCatalog", "Old Texture Eviction")},
    {"helper_qos_off", "performance", QT_TRANSLATE_NOOP("UiCatalog", "Helper Threads at Full Priority")},
    {"frame_deadlines", "performance", QT_TRANSLATE_NOOP("UiCatalog", "Frame Deadlines for the CPU")},
    {"jit_old_tables", "performance", QT_TRANSLATE_NOOP("UiCatalog", "Old JIT Tables")},
    {"fence_submit_each", "performance", QT_TRANSLATE_NOOP("UiCatalog", "Submit at Every Fence")},
    {"render_passes_old", "performance", QT_TRANSLATE_NOOP("UiCatalog", "Render Passes as Before")},
    {"oot_game_upscale", "performance", QT_TRANSLATE_NOOP("UiCatalog", "Game's Own Upscale Pass")},
    {"targetless_full", "performance", QT_TRANSLATE_NOOP("UiCatalog", "Full-Size Passes Without Targets")},
    {"ubo_zero_tails", "performance", QT_TRANSLATE_NOOP("UiCatalog", "Zero Uniform Buffer Tails")},
    {"release_submit", "performance", QT_TRANSLATE_NOOP("UiCatalog", "Submit at Every Fence Release")},
    {"idle_submit_grace", "performance", QT_TRANSLATE_NOOP("UiCatalog", "GPU Thread Waits for More Work")},
    {"idle_hold_off", "performance", QT_TRANSLATE_NOOP("UiCatalog", "GPU Thread Submits When Idle")},
    {"mesh_compute_in_pass", "performance", QT_TRANSLATE_NOOP("UiCatalog", "Mesh Emulation Ends Its Pass")},
    {"dispatch_chain", "performance", QT_TRANSLATE_NOOP("UiCatalog", "Back-to-Back Dispatches Without Barriers")},
    {"emulate_all_mesh", "performance", QT_TRANSLATE_NOOP("UiCatalog", "Emulate Every Mesh Shader")},
    {"dispatch_deps", "performance", QT_TRANSLATE_NOOP("UiCatalog", "Log What Ties Dispatches to Render Passes")},
    {"guest_cores_first", "cpu_scheduling", QT_TRANSLATE_NOOP("UiCatalog", "Two Guest Cores First")},
    {"oot_narrow_affinity", "cpu_scheduling", QT_TRANSLATE_NOOP("UiCatalog", "Game's Own Thread Cores")},
    {"oot_ideal_affinity", "cpu_scheduling", QT_TRANSLATE_NOOP("UiCatalog", "One Core per Game Thread")},
    {"guest_idle_spin", "cpu_scheduling", QT_TRANSLATE_NOOP("UiCatalog", "Idle Guest Cores Wait Briefly")},
    {"perf_sched_off", "cpu_scheduling", QT_TRANSLATE_NOOP("UiCatalog", "No CPU Scheduling Log Lines")},
    {"frame_time_off", "cpu_scheduling", QT_TRANSLATE_NOOP("UiCatalog", "Graphics Threads Declare No Frame Time")},
    {"frame_time_guest", "cpu_scheduling", QT_TRANSLATE_NOOP("UiCatalog", "Guest Cores Declare Frame Time Too")},
    {"split_std_waits", "cpu_scheduling", QT_TRANSLATE_NOOP("UiCatalog", "Graphics Threads Poll Before Sleeping")},
    {"poll_yield", "cpu_scheduling", QT_TRANSLATE_NOOP("UiCatalog", "Polling Threads Yield")},
    {"jit_inline_exclusives", "guest_memory", QT_TRANSLATE_NOOP("UiCatalog", "Exclusive Accesses in Fastmem")},
    {"jit_load_faults", "guest_memory", QT_TRANSLATE_NOOP("UiCatalog", "Loads Keep Fastmem After a Fault")},
    {"cpu_writes_old", "guest_memory", QT_TRANSLATE_NOOP("UiCatalog", "Writes Next to GPU Data as Before")},
    {"gpu_reads_old", "guest_memory", QT_TRANSLATE_NOOP("UiCatalog", "Reads of GPU Data as Before")},
    {"hot_pages_off", "guest_memory", QT_TRANSLATE_NOOP("UiCatalog", "Watch Every Rewritten Page")},
    {"metal_fixed_state_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Shaders Without the Draw's State")},
    {"metal_cache_plain", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Game Memory Access as Before")},
    {"metal_msl_plain", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Shaders Translated as Before")},
    {"metal_copies_plain", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Image Copies as Before")},
    {"metal_samplers_plain", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Samplers as Before")},
    {"metal_msaa_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "No Multisampled Image Copies")},
    {"metal_formats_plain", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Formats Metal Lacks as Before")},
    {"metal_layers_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "One Layer per Render Pass")},
    {"metal_multirange_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Split Storage Buffers as Before")},
    {"metal_clears_plain", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Clears as Before")},
    {"metal_barrier_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Ignore the Game's Draw Barriers")},
    {"metal_state_plain", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Set All Draw State Every Draw")},
    {"metal_ahead_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Uploads and Kernels in Their Place")},
    {"metal_fences_plain", "metal", QT_TRANSLATE_NOOP("UiCatalog", "A Command Buffer for Every Fence")},
    {"metal_staging_plain", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Upload Buffers Without Bounds")},
    {"metal_network_runs", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Always Run the Anti-Aliasing Network")},
    {"metal_prebuild_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Build Shaders at First Use Only")},
    {"metal_temporal_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "No MetalFX Temporal in Ocarina of Time")},
    {"metal_mesh_emulated", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Mesh Shaders Through the Emulation")},
    {"metal_vertex_plain", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Vertex Input as Before")},
    {"metal_async_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Build New Shaders on the Draw's Thread")},
    {"metal_provoking_first", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Flat Colours From the First Vertex")},
    {"metal_condition_cpu", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Conditional Draws Decided as Before")},
    {"metal_indirect_cpu", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Indirect Draws as Before")},
    {"metal_draw_texture_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Skip Texture Draws")},
    {"metal_clear_layers_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Clear the Bound Layer Only")},
    {"metal_list_copies_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "GPU-Written Command Lists as Before")},
    {"metal_viewports_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "One Viewport Only")},
    {"metal_fill_lines_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Always Fill Triangles")},
    {"metal_tess_base_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Tessellation Without Base Vertex")},
    {"metal_alpha_to_one_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "No Alpha to One")},
    {"metal_actor_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Link Moves With the Camera Only")},
    {"metal_effects_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "No Effects Mask for MetalFX")},
    {"metal_vulkan_switches_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Ignore Vulkan's Switches")},
    {"metal_fence_batch_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Commit at Every Fence With Work")},
    {"metal_filters_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Plain Window Scaling")},
    {"metal_standin_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Game's DLSS Without MetalFX")},
    {"metal_bindmemo_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Resolve Every Binding Every Draw")},
    {"metal_pacing_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "No Frame Pacing")},
    {"metal_colorspace_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Untagged Screen Colours")},
    {"metal_evict_off", "metal", QT_TRANSLATE_NOOP("UiCatalog", "Keep Every Pipeline")},
    {"pipeline_holds_old", "shader_builds", QT_TRANSLATE_NOOP("UiCatalog", "Held Draws Wait in the Queue")},
    {"pipeline_list_old", "shader_builds", QT_TRANSLATE_NOOP("UiCatalog", "Used Shader List Saved as Before")},
    {"mvk_stats_off", "shader_builds", QT_TRANSLATE_NOOP("UiCatalog", "No MoltenVK Compile Times")},
    {"query_cuts_off", "thread_cuts", QT_TRANSLATE_NOOP("UiCatalog", "GPU Reports Kept as Before")},
    {"fence_cuts_off", "thread_cuts", QT_TRANSLATE_NOOP("UiCatalog", "Every Fence Wakes the Fence Thread")},
    {"protect_locked", "thread_cuts", QT_TRANSLATE_NOOP("UiCatalog", "Page Protection Inside the Lock")},
    {"words_pool_off", "thread_cuts", QT_TRANSLATE_NOOP("UiCatalog", "Command Copies Freed at Once")},
    {"resume_publish_off", "thread_cuts", QT_TRANSLATE_NOOP("UiCatalog", "Resumed Lists Sent After Their Copy")},
    {"collect_split_off", "thread_cuts", QT_TRANSLATE_NOOP("UiCatalog", "Page-Crossing Writes as One")},
    {"usage_reset_full", "thread_cuts", QT_TRANSLATE_NOOP("UiCatalog", "Buffer Use Cleared in Full")},
};

const std::string_view kExperimentSections[] = {
    "memory_savings", "memory",       "graphics", "performance",   "cpu_scheduling",
    "guest_memory",   "metal",        "shader_builds", "thread_cuts",
};
// clang-format on

#undef CHOICE
#undef CHOICE_ON

} // namespace

Platform ThisPlatform() {
#if defined(__ANDROID__)
    return Android;
#elif defined(__APPLE__)
#if TARGET_OS_IPHONE
    return IOS;
#else
    return MacOS;
#endif
#elif defined(_WIN32)
    return Windows;
#else
    return Linux;
#endif
}

std::span<const Setting> AllSettings() {
    return kSettings;
}

std::span<const Choice> AllChoices() {
    return kChoices;
}

std::span<const Text> AllTexts() {
    return kTexts;
}

std::span<const Experiment> AllExperiments() {
    return kExperiments;
}

std::span<const std::string_view> ExperimentSections() {
    return kExperimentSections;
}

const Setting* FindSetting(std::string_view key) {
    static const auto index = [] {
        std::unordered_map<std::string_view, const Setting*> map;
        for (const Setting& setting : kSettings) {
            map.emplace(setting.key, &setting);
        }
        return map;
    }();
    const auto it = index.find(key);
    return it == index.end() ? nullptr : it->second;
}

std::span<const Choice> ChoicesOf(u32 enum_index) {
    static const auto index = [] {
        std::unordered_map<u32, std::vector<Choice>> map;
        for (const Choice& choice : AllChoices()) {
            map[choice.enum_index()].push_back(choice);
        }
        return map;
    }();
    const auto it = index.find(enum_index);
    return it == index.end() ? std::span<const Choice>{} : std::span<const Choice>{it->second};
}

const Choice* FindChoice(u32 enum_index, u32 value) {
    for (const Choice& choice : ChoicesOf(enum_index)) {
        if (choice.value == value) {
            return &choice;
        }
    }
    return nullptr;
}

const Experiment* FindExperiment(std::string_view id) {
    for (const Experiment& experiment : kExperiments) {
        if (experiment.id == id) {
            return &experiment;
        }
    }
    return nullptr;
}

const char* GetText(std::string_view id) {
    static const auto index = [] {
        std::unordered_map<std::string_view, const char*> map;
        for (const Text& text : kTexts) {
            map.emplace(text.id, text.text);
        }
        return map;
    }();
    const auto it = index.find(id);
    return it == index.end() ? id.data() : it->second;
}

bool ShownOn(const Setting& setting, Platform platform) {
    return setting.label != nullptr && (setting.shown & platform) != 0;
}

std::string Resolve(std::string_view text,
                    const std::function<std::string(std::string_view, std::string_view)>& name) {
    std::string out;
    out.reserve(text.size());
    std::size_t pos = 0;
    while (pos < text.size()) {
        const std::size_t open = text.find('{', pos);
        const std::size_t colon = open == std::string_view::npos ? open : text.find(':', open);
        const std::size_t close = colon == std::string_view::npos ? colon : text.find('}', colon);
        if (close == std::string_view::npos) {
            out.append(text.substr(pos));
            break;
        }
        const std::string_view kind = text.substr(open + 1, colon - open - 1);
        const bool known = kind == "setting" || kind == "choice" || kind == "exp" || kind == "text";
        out.append(text.substr(pos, open - pos));
        if (known) {
            out.append(name(kind, text.substr(colon + 1, close - colon - 1)));
        } else {
            out.append(text.substr(open, close - open + 1));
        }
        pos = close + 1;
    }
    return out;
}

} // namespace UiCatalog
