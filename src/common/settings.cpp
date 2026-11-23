// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <version>
#if __cpp_lib_chrono >= 201907L
#include <chrono>
#include <exception>
#include <stdexcept>
#endif
#include <atomic>
#include <compare>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <string_view>
#include <type_traits>
#include <deque>
#include <fmt/core.h>

#include "common/settings_enums.h"
#include "common/assert.h"
#include "common/fs/fs_util.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/settings.h"
#include "common/time_zone.h"

#if defined(__linux__ ) && defined(ARCHITECTURE_arm64)
#include <unistd.h>
#endif

namespace Settings {

// Clang 14 and earlier have errors when explicitly instantiating these classes
#ifndef CANNOT_EXPLICITLY_INSTANTIATE
#define SETTING(TYPE, RANGED) template class Setting<TYPE, RANGED>
#define SWITCHABLE(TYPE, RANGED) template class SwitchableSetting<TYPE, RANGED>

SETTING(AppletMode, false);
SETTING(AudioEngine, false);
SETTING(bool, false);
SETTING(int, false);
SETTING(std::string, false);
SETTING(u16, false);
SWITCHABLE(AnisotropyMode, true);
SWITCHABLE(AntiAliasing, false);
SWITCHABLE(OotAntiAliasing, false);
SWITCHABLE(Upscaler, false);
SWITCHABLE(DlssQuality, false);
SWITCHABLE(DlssPreset, false);
SWITCHABLE(OotMovieUpscale, false);
SWITCHABLE(TextureQuality, false);
SWITCHABLE(OotResolution, false);
SWITCHABLE(OotGraphicsPreset, false);
SWITCHABLE(OotFrameRate, false);
SWITCHABLE(MkwFrameRate, false);
SWITCHABLE(AspectRatio, true);
SWITCHABLE(AstcDecodeMode, true);
SWITCHABLE(AstcRecompression, true);
SWITCHABLE(AudioMode, true);
SWITCHABLE(CpuBackend, true);
SWITCHABLE(CpuAccuracy, true);
SWITCHABLE(FullscreenMode, true);
SWITCHABLE(GpuAccuracy, true);
SWITCHABLE(Language, true);
SWITCHABLE(MemoryLayout, true);
SWITCHABLE(NvdecEmulation, false);
SWITCHABLE(Region, true);
SWITCHABLE(RendererBackend, true);
SWITCHABLE(ScalingFilter, false);
SWITCHABLE(FrameGenType, false);
SWITCHABLE(FrameGenMotion, false);
SWITCHABLE(FrameGenFlowQuality, false);
SWITCHABLE(SpirvOptimizeMode, true);
SWITCHABLE(TimeZone, true);
SETTING(VSyncMode, true);
SWITCHABLE(bool, false);
SWITCHABLE(int, false);
SWITCHABLE(int, true);
SWITCHABLE(s64, false);
SWITCHABLE(u16, true);
SWITCHABLE(u32, false);
SWITCHABLE(u8, false);
SWITCHABLE(u8, true);

// Used in UISettings
// TODO see if we can move this to uisettings.cpp
SWITCHABLE(ConfirmStop, true);

#undef SETTING
#undef SWITCHABLE
#endif

Values values;

std::string GetTimeZoneString(TimeZone time_zone) {
    const auto time_zone_index = static_cast<std::size_t>(time_zone);
    ASSERT(time_zone_index < Common::TimeZone::GetTimeZoneStrings().size());

    std::string location_name;
    if (time_zone_index == 0) { // Auto
#if __cpp_lib_chrono >= 201907L && !defined(MINGW)
        // Disabled for MinGW -- tzdb always returns Etc/UTC
        try {
            const struct std::chrono::tzdb& time_zone_data = std::chrono::get_tzdb();
            const std::chrono::time_zone* current_zone = time_zone_data.current_zone();
            std::string_view current_zone_name = current_zone->name();
            location_name = current_zone_name;
        } catch (std::runtime_error& runtime_error) {
            // VCRUNTIME will throw a runtime_error if the operating system's selected time zone
            // cannot be found
            location_name = Common::TimeZone::FindSystemTimeZone();
            LOG_WARNING(Common,
                        "Error occurred when trying to determine system time zone:\n{}\nFalling "
                        "back to hour offset \"{}\"",
                        runtime_error.what(), location_name);
        }
#else
        location_name = Common::TimeZone::FindSystemTimeZone();
#endif
    } else {
        location_name = Common::TimeZone::GetTimeZoneStrings()[time_zone_index];
    }
    return location_name;
}

/// Logs every setting's value, modified ones first, hiding the token secrets.
void LogSettings() {
    std::deque<std::string> settings_list;
    for (auto& [category, settings] : values.linkage.by_category) {
        for (const auto& setting : settings) {
            // Hide the token secrets, for security reasons.
            if (setting->Id() != values.eden_token.Id() && setting->Id() != values.mcp_token.Id()) {
                auto const is_default = setting->ToString() == setting->DefaultToString();
                auto const name = fmt::format(
                    "{:c}{:c} {}.{}",
                    is_default ? '-' : 'M',
                    setting->UsingGlobal() ? '-' : 'C', TranslateCategory(category),
                    setting->GetLabel());
                if (is_default)
                    settings_list.push_back(fmt::format("{}: {}", name, setting->Canonicalize()));
                else
                    settings_list.push_front(fmt::format("{}: {}", name, setting->Canonicalize()));
            }
        }
    }
    LOG_INFO(Config, "Eden Configuration:");
    for (auto const& e : settings_list)
        LOG_INFO(Config, "{}", e);
#define LOG_PATH(NAME) \
    LOG_INFO(Config, #NAME ": {}", Common::FS::PathToUTF8String(Common::FS::GetEdenPath(Common::FS::EdenPath::NAME)))
    LOG_PATH(CacheDir);
    LOG_PATH(ConfigDir);
    LOG_PATH(LoadDir);
    LOG_PATH(NANDDir);
    LOG_PATH(SaveDir);
    LOG_PATH(SDMCDir);
#undef LOG_PATH
}

bool GetDebugKnobAt(u8 i) {
    return (values.debug_knobs.GetValue() & (1 << (i & 0xF))) != 0;
}

void UpdateGPUAccuracy() {
    values.current_gpu_accuracy = values.gpu_accuracy.GetValue();
}

bool IsGPULevelHigh() {
    return values.current_gpu_accuracy == GpuAccuracy::High;
}

bool IsDMALevelDefault() {
    return values.dma_accuracy.GetValue() == DmaAccuracy::Default;
}

bool IsDMALevelSafe() {
    return values.dma_accuracy.GetValue() == DmaAccuracy::Safe;
}

bool IsGPUFenceBehaviorDefault() {
    return values.gpu_fence_behavior.GetValue() == GpuFenceBehavior::Default;
}

bool IsGPUFenceBehaviorBalanced() {
    return values.gpu_fence_behavior.GetValue() == GpuFenceBehavior::Balanced;
}

bool IsGPUFenceBehaviorAccurate() {
    return values.gpu_fence_behavior.GetValue() == GpuFenceBehavior::Accurate;
}

bool IsFastmemEnabled() {
    if (values.cpu_accuracy.GetValue() == Settings::CpuAccuracy::Debugging)
        return bool(values.cpuopt_fastmem);
    else if (values.cpu_accuracy.GetValue() == CpuAccuracy::Unsafe)
        return bool(values.cpuopt_unsafe_host_mmu);
#if defined(__linux__) && defined(ARCHITECTURE_arm64)
    // Only 4kb systems support host MMU right now
    // TODO: Support this
    return getpagesize() == 4096;
#elif !defined(__APPLE__) && !defined(__ANDROID__) && !defined(_WIN32) && !defined(__linux__) && !defined(__FreeBSD__)
    return false;
#else
    return true;
#endif
}

static bool is_nce_enabled = false;

void SetNceEnabled(bool is_39bit) {
    const bool is_nce_selected = values.cpu_backend.GetValue() == CpuBackend::Nce;
    if (is_nce_selected && !IsFastmemEnabled()) {
        LOG_WARNING(Common, "Fastmem is required to natively execute code in a performant manner, "
                            "falling back to Dynarmic");
    }
    if (is_nce_selected && !is_39bit) {
        LOG_WARNING(
            Common,
            "Program does not utilize 39-bit address space, unable to natively execute code");
    }
    is_nce_enabled = IsFastmemEnabled() && is_nce_selected && is_39bit;
}

bool IsNceEnabled() {
    return is_nce_enabled;
}

namespace {
enum class CpuBackendEnvOverride : u8 {
    None,
    CachedInterpreter,
    Hypervisor,
    Other,
};

CpuBackendEnvOverride GetCpuBackendEnvOverride() noexcept {
    static const CpuBackendEnvOverride override_val = [] {
        const char* env = std::getenv("EDEN_NX2_CPU_BACKEND");
        if (env == nullptr || env[0] == '\0') {
            return CpuBackendEnvOverride::None;
        }
        const std::string_view sv{env};
        if (sv == "cached" || sv == "interpreter") {
            return CpuBackendEnvOverride::CachedInterpreter;
        }
        if (sv == "hypervisor" || sv == "hv") {
            return CpuBackendEnvOverride::Hypervisor;
        }
        return CpuBackendEnvOverride::Other;
    }();
    return override_val;
}
} // Anonymous namespace

bool IsCachedInterpreterEnabled() {
    switch (GetCpuBackendEnvOverride()) {
    case CpuBackendEnvOverride::None:
        return values.cpu_backend.GetValue() == CpuBackend::CachedInterpreter;
    case CpuBackendEnvOverride::CachedInterpreter:
        return true;
    default:
        return false;
    }
}

bool IsHypervisorEnabled() {
#ifdef HAS_HYPERVISOR
    switch (GetCpuBackendEnvOverride()) {
    case CpuBackendEnvOverride::None:
        return values.cpu_backend.GetValue() == CpuBackend::Hypervisor;
    case CpuBackendEnvOverride::Hypervisor:
        return true;
    default:
        return false;
    }
#else
    return false;
#endif
}

static u64 current_program_id = 0;

void SetCurrentProgramID(u64 program_id) {
    current_program_id = program_id;
}

u64 GetCurrentProgramID() {
    return current_program_id;
}

static bool current_program_is_switch2 = false;

void SetCurrentProgramIsSwitch2(bool is_switch2) {
    current_program_is_switch2 = is_switch2;
}

bool CurrentProgramIsSwitch2() {
    return current_program_is_switch2;
}

static std::optional<AstcRecompression> astc_recompression_override;

AstcRecompression GetAstcRecompression() {
    return astc_recompression_override.value_or(values.astc_recompression.GetValue());
}

void SetAstcRecompressionOverride(std::optional<AstcRecompression> recompression) {
    astc_recompression_override = recompression;
}

bool IsDockedMode() {
    return values.use_docked_mode.GetValue() == Settings::ConsoleMode::Docked;
}

float Volume() {
    if (values.audio_muted) {
        return 0.0f;
    }
    return values.volume.GetValue() / static_cast<f32>(values.volume.GetDefault());
}

const char* TranslateCategory(Category category) {
    switch (category) {
    case Category::Android:
        return "Android";
    case Category::Audio:
        return "Audio";
    case Category::Core:
        return "Core";
    case Category::Cpu:
    case Category::CpuDebug:
    case Category::CpuUnsafe:
        return "Cpu";
    case Category::Overlay:
        return "Overlay";
    case Category::Renderer:
    case Category::RendererAdvanced:
    case Category::RendererHacks:
    case Category::RendererDebug:
    case Category::RendererExtensions:
    case Category::RendererMemory:
        return "Renderer";
    case Category::System:
    case Category::SystemAudio:
        return "System";
    case Category::DataStorage:
        return "Data Storage";
    case Category::Debugging:
    case Category::DebuggingGraphics:
        return "Debugging";
    case Category::GpuDriver:
        return "GpuDriver";
    case Category::LibraryApplet:
        return "LibraryApplet";
    case Category::Miscellaneous:
        return "Miscellaneous";
    case Category::Network:
        return "Network";
    case Category::WebService:
        return "WebService";
    case Category::AddOns:
        return "DisabledAddOns";
    case Category::Controls:
        return "Controls";
    case Category::Ui:
    case Category::UiGeneral:
        return "UI";
    case Category::UiAudio:
        return "UiAudio";
    case Category::UiLayout:
        return "UILayout";
    case Category::UiGameList:
        return "UIGameList";
    case Category::Screenshots:
        return "Screenshots";
    case Category::Shortcuts:
        return "Shortcuts";
    case Category::Multiplayer:
        return "Multiplayer";
    case Category::Services:
        return "Services";
    case Category::Paths:
        return "Paths";
    case Category::MaxEnum:
        break;
    }
    return "Miscellaneous";
}

void TranslateResolutionInfo(ResolutionSetup setup, ResolutionScalingInfo& info) {
    info.downscale = false;
    switch (setup) {
    case ResolutionSetup::Res1_4X:
        info.up_scale = 1;
        info.down_shift = 2;
        info.downscale = true;
        break;
    case ResolutionSetup::Res1_2X:
        info.up_scale = 1;
        info.down_shift = 1;
        info.downscale = true;
        break;
    case ResolutionSetup::Res3_4X:
        info.up_scale = 3;
        info.down_shift = 2;
        info.downscale = true;
        break;
    case ResolutionSetup::Res1X:
        info.up_scale = 1;
        info.down_shift = 0;
        break;
    case ResolutionSetup::Res3_2X:
        info.up_scale = 3;
        info.down_shift = 1;
        break;
    case ResolutionSetup::Res5_4X:
        info.up_scale = 5;
        info.down_shift = 2;
        break;
    case ResolutionSetup::Res2X:
        info.up_scale = 2;
        info.down_shift = 0;
        break;
    case ResolutionSetup::Res3X:
        info.up_scale = 3;
        info.down_shift = 0;
        break;
    case ResolutionSetup::Res4X:
        info.up_scale = 4;
        info.down_shift = 0;
        break;
    case ResolutionSetup::Res5X:
        info.up_scale = 5;
        info.down_shift = 0;
        break;
    case ResolutionSetup::Res6X:
        info.up_scale = 6;
        info.down_shift = 0;
        break;
    case ResolutionSetup::Res7X:
        info.up_scale = 7;
        info.down_shift = 0;
        break;
    case ResolutionSetup::Res8X:
        info.up_scale = 8;
        info.down_shift = 0;
        break;
    default:
        UNREACHABLE();
    }
    info.up_factor = static_cast<f32>(info.up_scale) / (1U << info.down_shift);
    info.down_factor = static_cast<f32>(1U << info.down_shift) / info.up_scale;
    info.active = info.up_scale != 1 || info.down_shift != 0;
}

void UpdateRescalingInfo() {
    const auto setup = EffectiveResolutionSetup();
    auto& info = values.resolution_info;
    TranslateResolutionInfo(setup, info);
}

u32 FrameGenMultiplier() {
    return std::clamp(values.frame_gen_multiplier.GetValue(), MIN_FRAME_GEN_MULTIPLIER,
                      MAX_FRAME_GEN_MULTIPLIER);
}

size_t FrameGenGenerations() {
    if (!values.frame_gen.GetValue()) {
        return 0;
    }
    return FrameGenMultiplier() - 1;
}

size_t FrameGenMaxGenerations() {
    if (!values.frame_gen.GetValue()) {
        return 0;
    }
    if (values.frame_gen_target_rate.GetValue() != 0) {
        return MAX_FRAME_GEN_MULTIPLIER - 1;
    }
    return FrameGenMultiplier() - 1;
}

void RestoreGlobalState(bool is_powered_on) {
    // If a game is running, DO NOT restore the global settings state
    if (is_powered_on) {
        return;
    }

    for (const auto& reset : values.linkage.restore_functions) {
        reset();
    }

    // Reset per-game flags
    values.use_squashed_iterated_blend = false;
    SetAppliedOcarinaRenderScale(std::nullopt);
}

static bool configuring_global = true;

bool IsConfiguringGlobal() {
    return configuring_global;
}

void SetConfiguringGlobal(bool is_global) {
    configuring_global = is_global;
}

static u64 configuring_program_id = 0;

void SetConfiguringProgramId(u64 program_id) {
    configuring_program_id = program_id;
}

u64 ConfiguringProgramId() {
    return configuring_program_id;
}

u16 SpeedLimit() {
    switch (SpeedMode(values.current_speed_mode)) {
    case SpeedMode::Standard:
        return values.speed_limit.GetValue();
    case SpeedMode::Turbo:
        return values.turbo_speed_limit.GetValue();
    case SpeedMode::Slow:
        return values.slow_speed_limit.GetValue();
    default:
        UNIMPLEMENTED();
    }

    return 100;
}

namespace {
constexpr u64 OcarinaOfTimeId = 0x0400486024DEC000ULL;
}

namespace {
// Ocarina of Time follows the shared frame target (frame_pacing_mode); only Auto falls back to its
// own stored oot_frame_rate (the device profile default).
OotFrameRate OcarinaRate() {
    switch (values.frame_pacing_mode.GetValue()) {
    case FramePacingMode::Target_30:
        return OotFrameRate::Fps30;
    case FramePacingMode::Target_60:
        return OotFrameRate::Fps60;
    case FramePacingMode::Target_90:
    case FramePacingMode::Target_120:
    case FramePacingMode::Unlocked:
        return OotFrameRate::Unlocked;
    case FramePacingMode::Target_Auto:
        break;
    }
    return values.oot_frame_rate.GetValue();
}
} // namespace

// The vsync rate every game starts frames on: the shared frame target (frame_pacing_mode), or the
// frame_rate_limit setting with Auto. Ocarina of Time stays at 60 unless unlocked (OcarinaHalfRate
// covers 30); when unlocked with frame_rate_limit left at or below its 60 Hz default, lift the cap.
u16 FrameRateLimit() {
    constexpr u16 kUnlockedVsyncHz = 1000;
    const bool is_ocarina = GetCurrentProgramID() == OcarinaOfTimeId;
    if (is_ocarina && OcarinaRate() != OotFrameRate::Unlocked) {
        return 60;
    }
    switch (values.frame_pacing_mode.GetValue()) {
    case FramePacingMode::Target_30:
        return 30;
    case FramePacingMode::Target_60:
        return 60;
    case FramePacingMode::Target_90:
        return 90;
    case FramePacingMode::Target_120:
        return 120;
    case FramePacingMode::Unlocked:
        return kUnlockedVsyncHz;
    case FramePacingMode::Target_Auto:
        break;
    }
    const u16 limit = values.frame_rate_limit.GetValue();
    if (is_ocarina && limit <= values.frame_rate_limit.GetDefault()) {
        return kUnlockedVsyncHz;
    }
    return limit;
}

bool OcarinaHalfRate(u64 program_id) {
    return program_id == OcarinaOfTimeId && OcarinaRate() == OotFrameRate::Fps30;
}

OcarinaDlssMode OcarinaDlssModeFor(OotResolution resolution) {
    switch (resolution) {
    case OotResolution::Percent87:
        return OcarinaDlssMode::Balanced;
    case OotResolution::Percent75:
        return OcarinaDlssMode::Performance;
    case OotResolution::Percent50:
    case OotResolution::Percent40:
    case OotResolution::Percent33:
        return OcarinaDlssMode::UltraPerformance;
    default:
        return OcarinaDlssMode::Quality;
    }
}

OotResolution OcarinaGameResolution(OotResolution resolution) {
    switch (resolution) {
    case OotResolution::Percent150:
    case OotResolution::Percent200:
    case OotResolution::Percent300:
        return OotResolution::Full;
    default:
        return resolution;
    }
}

OotResolution OcarinaUpscaledResolution(OotResolution resolution, Upscaler upscaler, bool docked) {
    if (resolution != OotResolution::Game) {
        return resolution;
    }
    switch (upscaler) {
#if defined(__APPLE__)
    case Upscaler::Auto: {
        if (values.renderer_backend.GetValue() != RendererBackend::Metal) {
            return resolution;
        }
        if (const char* const env = std::getenv("EDEN_NX2_METAL_OOT_TEMPORAL");
            env != nullptr && env[0] == '0') {
            return resolution;
        }
        return docked ? OotResolution::Full : OotResolution::Percent75;
    }
#endif
    case Upscaler::Fsr3:
    case Upscaler::Fsr4:
    case Upscaler::Dlss:
    case Upscaler::MetalFx:
        return docked ? OotResolution::Full : OotResolution::Percent75;
    default:
        return resolution;
    }
}

OotPresetCompanions OotGraphicsPresetCompanions(OotGraphicsPreset preset) {
    switch (preset) {
    case OotGraphicsPreset::UltraLow:
        return {OotResolution::Percent40, TextureQuality::Quarter, OotAntiAliasing::NoSharpening};
    case OotGraphicsPreset::PhoneLow:
        return {OotResolution::Percent50, TextureQuality::Quarter, std::nullopt};
    case OotGraphicsPreset::PhoneMedium:
        return {OotResolution::Percent56, TextureQuality::Half, std::nullopt};
    default:
        return {};
    }
}

void ApplyOotGraphicsPreset(OotGraphicsPreset preset) {
    const bool global = IsConfiguringGlobal();
    const auto set = [global](auto& setting, auto value) {
        if (!global) {
            setting.SetGlobal(false);
            setting.SetValue(value);
            return;
        }
        const bool custom = !setting.UsingGlobal();
        if (custom) {
            setting.SetGlobal(true);
        }
        setting.SetValue(value);
        if (custom) {
            setting.SetGlobal(false);
        }
    };
    set(values.oot_graphics_preset, preset);
    const auto companions = OotGraphicsPresetCompanions(preset);
    if (companions.resolution) {
        set(values.oot_resolution, *companions.resolution);
    }
    if (!global && companions.textures) {
        set(values.texture_quality, *companions.textures);
    }
    if (companions.sharpening) {
        set(values.oot_anti_aliasing, *companions.sharpening);
    }
}

ResolutionSetup EffectiveResolutionSetup() {
    if (GetCurrentProgramID() != OcarinaOfTimeId) {
        return values.resolution_setup.GetValue();
    }
    switch (values.upscaler.GetValue()) {
    case Upscaler::Fsr3:
    case Upscaler::Fsr4:
    case Upscaler::Dlss:
    case Upscaler::MetalFx:
        return ResolutionSetup::Res1X;
    default:
        break;
    }
    switch (values.oot_resolution.GetValue()) {
    case OotResolution::Percent150:
        return ResolutionSetup::Res3_2X;
    case OotResolution::Percent200:
        return ResolutionSetup::Res2X;
    case OotResolution::Percent300:
        return ResolutionSetup::Res3X;
    default:
        return ResolutionSetup::Res1X;
    }
}

bool NvidiaDlssChosen() {
    return values.upscaler.GetValue() == Upscaler::Dlss;
}

bool AmdFsr4Chosen() {
    return values.upscaler.GetValue() == Upscaler::Fsr4;
}

bool MetalFxTemporalChosen() {
    return values.upscaler.GetValue() == Upscaler::MetalFx;
}

bool GameDlssNetworkChosen() {
    return values.upscaler.GetValue() == Upscaler::Game;
}

bool OcarinaFsrChosen() {
    const auto upscaler = values.upscaler.GetValue();
    return upscaler == Upscaler::Fsr3 || upscaler == Upscaler::Fsr4;
}

namespace {
std::atomic<u32> applied_ocarina_render_scale{};
} // namespace

std::optional<NvidiaDlssQuality> OcarinaDlssRenderScale() {
    const auto quality = values.nvidia_dlss_quality.GetValue();
    if (!NvidiaDlssChosen() || !IsDockedMode() ||
        (quality != NvidiaDlssQuality::Balanced && quality != NvidiaDlssQuality::Performance &&
         quality != NvidiaDlssQuality::UltraPerformance)) {
        return std::nullopt;
    }
    return quality;
}

void SetAppliedOcarinaRenderScale(std::optional<NvidiaDlssQuality> quality) {
    applied_ocarina_render_scale.store(quality ? static_cast<u32>(*quality) + 1 : 0,
                                       std::memory_order_relaxed);
}

std::optional<NvidiaDlssQuality> AppliedOcarinaRenderScale() {
    const u32 value = applied_ocarina_render_scale.load(std::memory_order_relaxed);
    if (value == 0) {
        return std::nullopt;
    }
    return static_cast<NvidiaDlssQuality>(value - 1);
}

void SetSpeedMode(const SpeedMode& mode) {
    values.current_speed_mode.SetValue(mode);

    switch (mode) {
    case SpeedMode::Turbo:
    case SpeedMode::Slow:
        values.use_speed_limit.SetValue(true);
        break;
    case SpeedMode::Standard:
    default:
        break;
    }
}

void ToggleStandardMode() {
    values.use_speed_limit.SetValue(!values.use_speed_limit.GetValue());
    SetSpeedMode(SpeedMode::Standard);
}

void ToggleTurboMode() {
    if (values.current_speed_mode.GetValue() != SpeedMode::Turbo)
        SetSpeedMode(SpeedMode::Turbo);
    else
        SetSpeedMode(SpeedMode::Standard);
}

void ToggleSlowMode() {
    if (values.current_speed_mode.GetValue() != SpeedMode::Slow)
        SetSpeedMode(SpeedMode::Slow);
    else
        SetSpeedMode(SpeedMode::Standard);
}

bool IsOpenGL() {
    const auto backend = Settings::values.renderer_backend.GetValue();
    switch (backend) {
    case RendererBackend::OpenGL_GLSL:
    case RendererBackend::OpenGL_GLASM:
    case RendererBackend::OpenGL_SPIRV:
        return true;
    default:
        return false;
    }
}

} // namespace Settings
