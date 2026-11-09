// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <iterator>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <string>
#include "common/settings_enums.h"
#include "core/launch_params.h"
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif
#define SDL_MAIN_USE_CALLBACKS 1
#include <SDL3/SDL_main.h>

#include <fmt/ostream.h>

#include "common/fs/fs_util.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/scm_rev.h"
#include "common/settings.h"
#include "common/string_util.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/cpu_manager.h"
#include "core/crypto/key_manager.h"
#include "core/file_sys/common_funcs.h"
#include "core/file_sys/nup_writer.h"
#include "core/file_sys/registered_cache.h"
#include "core/file_sys/vfs/vfs_real.h"
#include "core/hle/service/am/applet_manager.h"
#include "core/hle/service/filesystem/filesystem.h"
#include "core/loader/loader.h"
#include "frontend_common/config.h"
#include "frontend_common/nx2_switches.h"
#include "input_common/drivers/keyboard.h"
#include "input_common/main.h"
#include "network/network.h"
#include "sdl_config.h"
#include "video_core/gpu.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_base.h"
#include "yuzu_cmd/emu_window/emu_window_sdl3.h"
#ifdef HAS_OPENGL
#include "yuzu_cmd/emu_window/emu_window_sdl3_gl.h"
#endif
#include "yuzu_cmd/emu_window/emu_window_sdl3_null.h"
#include "yuzu_cmd/emu_window/emu_window_sdl3_vk.h"

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif
#if defined(__APPLE__) && TARGET_OS_IPHONE
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <vector>
#include <os/log.h>
#include "common/ios_jit.h"
#include "ios_frontend/lichii_bridge_platform.h"
#include "yuzu_cmd/ios_support.h"
#include "hid_core/frontend/emulated_controller.h"
#include "hid_core/hid_core.h"
#include "hid_core/hid_types.h"
#include "input_common/drivers/virtual_gamepad.h"
#include <os/proc.h>
#include <sys/sysctl.h>
#include <unistd.h>
#include <CoreFoundation/CoreFoundation.h>
#ifdef LICHII_IOS_UI
#include "common/fs/fs.h"
#include "ios_frontend/bridge_host.h"
#include "ios_frontend/lichii_bridge_content_host.h"
#include "ios_frontend/controls_host.h"
#endif

// Code-signing status: dynarmic's code pages are only executable while the process is being
// debugged (that is what StikDebug/SideStore "JIT" means on iOS 17+).
extern "C" int csops(pid_t pid, unsigned int ops, void* useraddr, size_t usersize);
static constexpr unsigned int kCsOpsStatus = 0;
static constexpr int kCsDebugged = 0x10000000;
static bool IosJitEnabled() {
    int flags = 0;
    if (csops(getpid(), kCsOpsStatus, &flags, sizeof(flags)) != 0) {
        return false;
    }
    return (flags & kCsDebugged) != 0;
}
// dynarmic: one code cache per guest core (six), plus two chunks for smaller blocks. Acquired once
// through StikDebug's universal script (iOS 26+), then the debugger is released. The script
// prepares every page over the debug connection while this app waits in the background, so the
// pool is kept small (800 MiB took long enough for iOS to kill the app).
static constexpr std::size_t kIosJitPoolBytes = 8 * Common::IosJit::ChunkBytes;
static std::string IosBundleIdentifier() {
    std::string out;
    if (CFStringRef id = CFBundleGetIdentifier(CFBundleGetMainBundle())) {
        char buf[256];
        if (CFStringGetCString(id, buf, sizeof(buf), kCFStringEncodingUTF8)) {
            out = buf;
        }
    }
    return out;
}

// Startup breadcrumbs for a device without a debugger: os_log (visible in the device syslog) and
// Documents/lichii-startup.txt. stdout/stderr are also redirected into Documents/console.txt so the
// console log backend and any printf survive an early exit.
static std::string IosDocumentsDir() {
    const char* home = std::getenv("HOME");
    return std::string(home ? home : ".") + "/Documents";
}
static void IosMark(const char* what) {
    os_log(OS_LOG_DEFAULT, "[lichii] %{public}s", what);
    static FILE* marks = nullptr;
    if (marks == nullptr) {
        marks = std::fopen((IosDocumentsDir() + "/lichii-startup.txt").c_str(), "a");
    }
    if (marks != nullptr) {
        std::fprintf(marks, "%s\n", what);
        std::fflush(marks);
    }
}
static void IosCaptureConsole() {
    const std::string path = IosDocumentsDir() + "/console.txt";
    std::freopen(path.c_str(), "a", stdout);
    std::freopen(path.c_str(), "a", stderr);
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
}
namespace {
struct IosStaticInitMark {
    IosStaticInitMark() {
        IosMark("yuzu.cpp static init");
    }
} ios_static_init_mark;
} // namespace
#define IOS_MARK(what) IosMark(what)
#else
#define IOS_MARK(what) ((void)0)
#endif

#ifdef _WIN32
// windows.h needs to be included before shellapi.h
#include <windows.h>
#include <shellapi.h>
#include "common/windows/timer_resolution.h"
#endif

#ifdef _WIN32
extern "C" {
// tells Nvidia and AMD drivers to use the dedicated GPU by default on laptops with switchable
// graphics
__declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}

// EDEN_NX2_CRASH_DUMP=<dir>: on an unhandled exception (a real crash; fastmem faults are handled
// before this), write <dir>/eden-crash-<pid>.dmp. A debugger changes the timing enough to hide
// some crashes, so this catches them at full speed. dbghelp.dll is loaded only then.
// Prints a code address as module+offset (resolve with llvm-symbolizer and the module's PDB).
static void PrintCodeAddress(const char* label, const void* address) {
    HMODULE module{};
    char name[MAX_PATH] = "?";
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCSTR>(address), &module)) {
        GetModuleFileNameA(module, name, sizeof(name));
    }
    const char* base_name = std::strrchr(name, '\\');
    std::fprintf(stderr, "  %s %p %s+%#llx\n", label, address, base_name ? base_name + 1 : name,
                 static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(address) -
                                                 reinterpret_cast<uintptr_t>(module)));
}

static LONG WINAPI WriteCrashDump(EXCEPTION_POINTERS* exception) {
    static std::atomic<bool> written{};
    const char* directory = std::getenv("EDEN_NX2_CRASH_DUMP");
    if (directory == nullptr || written.exchange(true)) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // The faulting address and stack first: they survive even if the dump fails.
    const EXCEPTION_RECORD& record = *exception->ExceptionRecord;
    std::fprintf(stderr, "Unhandled exception %#lx on thread %lu, info %#llx %#llx\n",
                 record.ExceptionCode, GetCurrentThreadId(),
                 static_cast<unsigned long long>(record.NumberParameters > 0 ? record.ExceptionInformation[0] : 0),
                 static_cast<unsigned long long>(record.NumberParameters > 1 ? record.ExceptionInformation[1] : 0));
    PrintCodeAddress("at", record.ExceptionAddress);
    void* frames[48];
    const USHORT count = RtlCaptureStackBackTrace(0, 48, frames, nullptr);
    for (USHORT i = 0; i < count; ++i) {
        PrintCodeAddress("frame", frames[i]);
    }
    std::fflush(stderr);
    struct ExceptionInformation {
        DWORD thread_id;
        EXCEPTION_POINTERS* exception_pointers;
        BOOL client_pointers;
    };
    using MiniDumpWriteDumpFn = BOOL(WINAPI*)(HANDLE, DWORD, HANDLE, int, const ExceptionInformation*,
                                              const void*, const void*);
    const HMODULE dbghelp = LoadLibraryA("dbghelp.dll");
    const auto write = dbghelp ? reinterpret_cast<MiniDumpWriteDumpFn>(
                                     GetProcAddress(dbghelp, "MiniDumpWriteDump"))
                               : nullptr;
    if (write == nullptr) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    char path[MAX_PATH];
    std::snprintf(path, sizeof(path), "%s\\eden-crash-%lu.dmp", directory, GetCurrentProcessId());
    const HANDLE file = CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                                    FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file != INVALID_HANDLE_VALUE) {
        const ExceptionInformation information{GetCurrentThreadId(), exception, FALSE};
        // MiniDumpWithThreadInfo; the larger MiniDumpWithIndirectlyReferencedMemory failed on
        // Eden's address space and left an empty file.
        const BOOL ok = write(GetCurrentProcess(), GetCurrentProcessId(), file, 0x1000,
                              &information, nullptr, nullptr);
        const DWORD error = ok ? 0 : GetLastError();
        CloseHandle(file);
        std::fprintf(stderr, ok ? "Crash dump written to %s\n" : "Crash dump to %s failed (%#lx)\n",
                     path, error);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif

static void OnStateChanged(const Network::RoomMember::State& state) {
    switch (state) {
    case Network::RoomMember::State::Idle:
        LOG_DEBUG(Network, "Network is idle");
        break;
    case Network::RoomMember::State::Joining:
        LOG_DEBUG(Network, "Connection sequence to room started");
        break;
    case Network::RoomMember::State::Joined:
        LOG_DEBUG(Network, "Successfully joined to the room");
        break;
    case Network::RoomMember::State::Moderator:
        LOG_DEBUG(Network, "Successfully joined the room as a moderator");
        break;
    default:
        break;
    }
}

static void OnNetworkError(const Network::RoomMember::Error& error) {
    switch (error) {
    case Network::RoomMember::Error::LostConnection:
        LOG_DEBUG(Network, "Lost connection to the room");
        break;
    case Network::RoomMember::Error::CouldNotConnect:
        LOG_ERROR(Network, "Error: Could not connect");
        exit(1);
        break;
    case Network::RoomMember::Error::NameCollision:
        LOG_ERROR(
            Network,
            "You tried to use the same nickname as another user that is connected to the Room");
        exit(1);
        break;
    case Network::RoomMember::Error::IpCollision:
        LOG_ERROR(Network, "You tried to use the same fake IP-Address as another user that is "
                           "connected to the Room");
        exit(1);
        break;
    case Network::RoomMember::Error::WrongPassword:
        LOG_ERROR(Network, "Room replied with: Wrong password");
        exit(1);
        break;
    case Network::RoomMember::Error::WrongVersion:
        LOG_ERROR(Network,
                  "You are using a different version than the room you are trying to connect to");
        exit(1);
        break;
    case Network::RoomMember::Error::RoomIsFull:
        LOG_ERROR(Network, "The room is full");
        exit(1);
        break;
    case Network::RoomMember::Error::HostKicked:
        LOG_ERROR(Network, "You have been kicked by the host");
        break;
    case Network::RoomMember::Error::HostBanned:
        LOG_ERROR(Network, "You have been banned by the host");
        break;
    case Network::RoomMember::Error::UnknownError:
        LOG_ERROR(Network, "UnknownError");
        break;
    case Network::RoomMember::Error::PermissionDenied:
        LOG_ERROR(Network, "PermissionDenied");
        break;
    case Network::RoomMember::Error::NoSuchUser:
        LOG_ERROR(Network, "NoSuchUser");
        break;
    }
}

static void OnMessageReceived(const Network::ChatEntry& msg) {
    std::cout << std::endl << msg.nickname << ": " << msg.message << std::endl << std::endl;
}

static void OnStatusMessageReceived(const Network::StatusMessageEntry& msg) {
    std::string message = [&]() {
        switch (msg.type) {
        case Network::IdMemberJoin:
            return fmt::format("{} has joined", msg.nickname);
        case Network::IdMemberLeave:
            return fmt::format("{} has left", msg.nickname);
        case Network::IdMemberKicked:
            return fmt::format("{} has been kicked", msg.nickname);
        case Network::IdMemberBanned:
            return fmt::format("{} has been banned", msg.nickname);
        case Network::IdAddressUnbanned:
            return fmt::format("{} has been unbanned", msg.nickname);
        default:
            return std::string{};
        }
    }();
    if (!message.empty())
        std::cout << std::endl << "* " << message << std::endl << std::endl;
}

static std::atomic_bool restart_requested{false};

struct SdlState {
    Core::System system{};
    // Must outlive emu_window, which keeps a pointer to it.
    InputCommon::InputSubsystem input_subsystem{};
    std::unique_ptr<EmuWindow_SDL3> emu_window;
    bool network_initialized{};
    // iOS: launch parameters parked until the JIT (debugged) state is on; finished in SDL_AppIterate.
    std::optional<Core::LaunchParams> pending_launch;
    Uint64 jit_wait_start{};
    Uint64 jit_failed_at{};
    bool jit_have_stikdebug{};
    bool jit_url_opened{};
    std::uint64_t jit_bg_task{};
    bool ios_controllers_dirty{};
    bool ios_paused_for_background{};
    // Plain SDL window shown while waiting for the JIT so the screen is not just black.
    SDL_Window* wait_window{};
    SDL_Renderer* wait_renderer{};
    bool system_initialized{};
    bool content_initialized{};
#ifdef LICHII_IOS_UI
    // Set while the native app drives launches; the global config stays open to save edits.
    std::unique_ptr<SdlConfig> ios_config;
    Core::LaunchParams ios_launch_template;
    Core::SystemResultStatus ios_load_result{Core::SystemResultStatus::Success};
#endif
};

static SDL_AppResult SDL_AppInitRest(SdlState* state, Core::LaunchParams lp);

// The app keeps one system for several games: these run once per process.
static bool InitializeSystem(SdlState* state) {
    if (state->system_initialized) {
        return true;
    }
    state->network_initialized = Network::Init();
    if (!state->network_initialized) {
        LOG_CRITICAL(Network, "Failed to initialize room networking");
        return false;
    }

    IOS_MARK("system.Initialize");
    state->system.Initialize();
    IOS_MARK("system initialized");
    state->system_initialized = true;
    return true;
}

static void InitializeContent(SdlState* state) {
    if (state->content_initialized) {
        return;
    }
    state->system.SetContentProvider(std::make_unique<FileSys::ContentProviderUnion>());
    state->system.SetFilesystem(std::make_shared<FileSys::RealVfsFilesystem>());
    state->system.GetFileSystemController().CreateFactories(*state->system.GetFilesystem());
    state->content_initialized = true;
}

#if defined(__APPLE__) && TARGET_OS_IPHONE
// Touch overlay -> virtual gamepad. Player 1 (port 0) and Handheld (port 8) both listen to it,
// so the game sees the input whichever controller style it reads.
static InputCommon::InputSubsystem* g_ios_input = nullptr;
static constexpr std::size_t kIosVirtualPorts[] = {0, 8};

static void IosTouchButton(int button, bool pressed) {
    if (g_ios_input == nullptr) {
        return;
    }
    if (auto* pad = g_ios_input->GetVirtualGamepad()) {
        for (const std::size_t port : kIosVirtualPorts) {
            pad->SetButtonState(port, button, pressed);
        }
    }
}

static void IosTouchStick(int stick, float x, float y) {
    if (g_ios_input == nullptr) {
        return;
    }
    if (auto* pad = g_ios_input->GetVirtualGamepad()) {
        for (const std::size_t port : kIosVirtualPorts) {
            pad->SetStickPosition(port, stick, x, y);
        }
    }
}

// Maps the first physical controller (SDL: MFi, Xbox, DualSense, Switch Pro, Backbone, ...)
// onto Player 1 and Handheld with its default layout, and shows the touch overlay only when no
// controller is connected.
static void IosApplyControllers(SdlState* state) {
    auto& input = state->input_subsystem;
    std::optional<Common::ParamPackage> pad;
    for (const auto& device : input.GetInputDevices()) {
        if (device.Get("engine", "") == "sdl") {
            pad = device;
            break;
        }
    }
    if (pad) {
        const auto buttons = input.GetButtonMappingForDevice(*pad);
        const auto analogs = input.GetAnalogMappingForDevice(*pad);
        for (const auto id : {Core::HID::NpadIdType::Player1, Core::HID::NpadIdType::Handheld}) {
            auto* controller = state->system.HIDCore().GetEmulatedController(id);
            if (controller == nullptr) {
                continue;
            }
            for (const auto& [index, param] : buttons) {
                controller->SetButtonParam(static_cast<std::size_t>(index), param);
            }
            for (const auto& [index, param] : analogs) {
                controller->SetStickParam(static_cast<std::size_t>(index), param);
            }
        }
        LOG_INFO(Frontend, "iOS: controller \"{}\" mapped to player 1, touch controls hidden",
                 pad->Get("display", "gamepad"));
    } else {
        LOG_INFO(Frontend, "iOS: no controller connected, touch controls shown");
    }
    IosSupport::SetTouchControlsVisible(!pad.has_value());
#ifdef LICHII_IOS_UI
    IosFrontend::PostEvent(LICHII_EVENT_CONTROLLERS_CHANGED);
#endif
}

static bool IosNativeApp([[maybe_unused]] SdlState* state) {
#ifdef LICHII_IOS_UI
    return state->ios_config != nullptr;
#else
    return false;
#endif
}

static void IosInstallTouchControls(SdlState* state) {
    g_ios_input = &state->input_subsystem;
    int count = 0;
    SDL_Window** windows = SDL_GetWindows(&count);
    void* ui_window = nullptr;
    if (windows != nullptr && count > 0) {
        ui_window = SDL_GetPointerProperty(SDL_GetWindowProperties(windows[0]),
                                           SDL_PROP_WINDOW_UIKIT_WINDOW_POINTER, nullptr);
    }
    SDL_free(windows);
    if (ui_window == nullptr) {
        LOG_ERROR(Frontend, "iOS: no UIWindow for the touch controls");
        return;
    }
    IosSupport::InstallTouchControls(ui_window, &IosTouchButton, &IosTouchStick);
    IosApplyControllers(state);
}

#ifdef LICHII_IOS_UI
// Defaults the SDL launcher forced on every boot, applied once so the app's settings can change
// them afterwards.
static void IosApplyFirstRunDefaults(SdlState* state) {
    // Per-device defaults (resolution, frame generation...) once per device profile version.
    bool changed = lichii_platform_apply_device_defaults(LICHII_DEFAULTS_FIRST_LAUNCH) != 0;
    const auto marker = Common::FS::GetEdenPath(Common::FS::EdenPath::ConfigDir) / "ios-defaults-v1";
    if (!Common::FS::Exists(marker)) {
        Settings::values.use_docked_mode.SetValue(Settings::ConsoleMode::Handheld);
        Settings::values.use_asynchronous_shaders.SetValue(true);
        std::ofstream{marker} << "1\n";
        changed = true;
    }
    if (changed) {
        state->ios_config->SaveAllValues();
    }
}

static SDL_AppResult IosStartApp(SdlState* state, Core::LaunchParams lp) {
    state->ios_config = std::make_unique<SdlConfig>(lp.config_path);
    IosApplyFirstRunDefaults(state);
    if (!InitializeSystem(state)) {
        return SDL_APP_FAILURE;
    }
    InitializeContent(state);
    lp.fullscreen = true;
    state->ios_launch_template = std::move(lp);
    IosFrontend::Attach({
        .system = &state->system,
        .input = &state->input_subsystem,
        .save_config = [state] { state->ios_config->SaveAllValues(); },
    });
    LichiiControls::Attach(state->system, state->input_subsystem);
    IosFrontend::SetJitPoolBytes(kIosJitPoolBytes);
    SDL_SetHint(SDL_HINT_ORIENTATIONS, "LandscapeLeft LandscapeRight");
    IOS_MARK("showing the app");
    lichii_ui_main();
    return SDL_APP_CONTINUE;
}

static void* IosGameWindow(SdlState* state) {
    if (!state->emu_window || state->emu_window->GetSdlWindow() == nullptr) {
        return nullptr;
    }
    return SDL_GetPointerProperty(SDL_GetWindowProperties(state->emu_window->GetSdlWindow()),
                                  SDL_PROP_WINDOW_UIKIT_WINDOW_POINTER, nullptr);
}

static void IosStopGame(SdlState* state) {
    LOG_INFO(Frontend, "iOS: stopping the game");
    IosSupport::SetTouchControlsVisible(false);
    if (state->system.IsPoweredOn()) {
        void(state->system.Pause());
        state->system.ShutdownMainProcess();
    }
    if (state->emu_window) {
        SDL_HideWindow(state->emu_window->GetSdlWindow());
    }
    // Every dynarmic instance went with the process, so the prepared pool can be handed out again.
    Common::IosJit::Reset();
    IosFrontend::RestoreGlobal();
    IosFrontend::ClearRunningGame();
    IosFrontend::SetState(LICHII_EMU_IDLE);
}

static void IosBootFailed(SdlState* state) {
    std::string error = "The game could not be started. The log has the details.";
    if (state->ios_load_result != Core::SystemResultStatus::Success) {
        const u16 loader_id = u16(Core::SystemResultStatus::ErrorLoader);
        const u16 code = u16(state->ios_load_result);
        if (code > loader_id) {
            error = fmt::format("Loader error {:04X}-{:04X}: {}", loader_id, code - loader_id,
                                Loader::ResultStatus(code - loader_id));
        } else if (state->ios_load_result == Core::SystemResultStatus::ErrorVideoCore) {
            error = "The renderer (Vulkan/MoltenVK) failed to start.";
        } else if (state->ios_load_result == Core::SystemResultStatus::ErrorGetLoader) {
            error = "This file is not a supported game format.";
        }
    }
    IosStopGame(state);
    IosFrontend::SetState(LICHII_EMU_FAILED, std::move(error));
}

// Launches, stops and JIT waits requested by the app. Returns a result when this frame is done.
static std::optional<SDL_AppResult> IosAppIterate(SdlState* state) {
    using Common::IosJit::State;
    if (IosFrontend::TakeCancelRequest() && state->pending_launch) {
        state->pending_launch.reset();
        IosFrontend::SetState(LICHII_EMU_IDLE);
    }
    // The game asked to run another program: boot the same file again with the restart handoff,
    // in this process (the pool is already prepared).
    if (restart_requested.exchange(false) && state->system.IsPoweredOn()) {
        Core::LaunchParams lp = state->ios_launch_template;
        lp.filepath = IosFrontend::RunningGamePath();
        IosStopGame(state);
        setenv("EDEN_CLI_RESTART_HANDOFF", "1", 1);
        state->pending_launch = std::move(lp);
        IosFrontend::SetState(LICHII_EMU_BOOTING);
    }
    if (!state->pending_launch && !state->system.IsPoweredOn()) {
        if (auto path = IosFrontend::TakeLaunchRequest()) {
            Core::LaunchParams lp = state->ios_launch_template;
            lp.filepath = std::move(*path);
            state->pending_launch = std::move(lp);
            state->jit_wait_start = SDL_GetTicks();
            IosFrontend::SetState(LICHII_EMU_WAITING_FOR_JIT);
        }
    }
    if (state->pending_launch) {
        const State jit = Common::IosJit::GetState();
        if (jit == State::Ready) {
            LOG_INFO(Frontend, "iOS: JIT pool ready, booting {}", state->pending_launch->filepath);
            IosFrontend::EndJitBackgroundTask();
            IosFrontend::SetState(LICHII_EMU_BOOTING);
            Core::LaunchParams lp = std::move(*state->pending_launch);
            state->pending_launch.reset();
            state->ios_load_result = Core::SystemResultStatus::Success;
            const SDL_AppResult result = SDL_AppInitRest(state, std::move(lp));
            unsetenv("EDEN_CLI_RESTART_HANDOFF");
            if (result != SDL_APP_CONTINUE) {
                IosBootFailed(state);
            }
        } else if (jit == State::Failed) {
            IosFrontend::EndJitBackgroundTask();
            state->pending_launch.reset();
            IosFrontend::SetState(LICHII_EMU_FAILED,
                                  "JIT unavailable: " + Common::IosJit::GetError());
        }
        return SDL_APP_CONTINUE;
    }
    if (IosFrontend::TakeStopRequest()) {
        IosStopGame(state);
    }
    if (!state->system.IsPoweredOn()) {
        return SDL_APP_CONTINUE;
    }
    return std::nullopt;
}
#endif
#endif

constexpr int kRestartExitCode = 75;

// ExecuteProgram state for the restarted process: what the Qt frontend keeps in-process when it
// reboots a game. Only read when the launcher sets EDEN_CLI_RESTART_HANDOFF, so a leftover file
// never changes a normal launch.
struct RestartHandoff {
    u64 program_id{};
    s32 program_index{};
    s32 previous_program_index{-1};
    std::vector<std::vector<u8>> user_channel;
};

static std::filesystem::path RestartHandoffPath() {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::EdenDir) / "restart-handoff.bin";
}

template <typename T>
static void PutRaw(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <typename T>
static void GetRaw(std::istream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
}

// Bounds keep a damaged handoff from requesting huge allocations; the writer refuses anything the
// reader would reject.
constexpr u32 kHandoffMaxEntries = 256;
constexpr u64 kHandoffMaxTotalSize = 64u << 20;

static bool WriteRestartHandoff(const RestartHandoff& handoff) {
    u64 total = 0;
    for (const auto& entry : handoff.user_channel) {
        total += entry.size();
    }
    if (handoff.user_channel.size() > kHandoffMaxEntries || total > kHandoffMaxTotalSize) {
        return false;
    }
    std::ofstream out{RestartHandoffPath(), std::ios::binary | std::ios::trunc};
    PutRaw(out, handoff.program_id);
    PutRaw(out, handoff.program_index);
    PutRaw(out, handoff.previous_program_index);
    PutRaw(out, static_cast<u32>(handoff.user_channel.size()));
    for (const auto& entry : handoff.user_channel) {
        PutRaw(out, static_cast<u32>(entry.size()));
        out.write(reinterpret_cast<const char*>(entry.data()), std::streamsize(entry.size()));
    }
    out.close();
    return !out.fail();
}

// Leaves the file in place; the caller removes it once the game has loaded.
static std::optional<RestartHandoff> ReadRestartHandoff() {
    std::ifstream in{RestartHandoffPath(), std::ios::binary};
    RestartHandoff handoff;
    u32 count{};
    GetRaw(in, handoff.program_id);
    GetRaw(in, handoff.program_index);
    GetRaw(in, handoff.previous_program_index);
    GetRaw(in, count);
    if (in.fail() || count > kHandoffMaxEntries) {
        return std::nullopt;
    }
    u64 total = 0;
    for (u32 i = 0; i < count; ++i) {
        u32 size{};
        GetRaw(in, size);
        total += size;
        if (in.fail() || total > kHandoffMaxTotalSize) {
            return std::nullopt;
        }
        auto& entry = handoff.user_channel.emplace_back(size);
        in.read(reinterpret_cast<char*>(entry.data()), std::streamsize(size));
        if (in.fail()) {
            return std::nullopt;
        }
    }
    return handoff;
}

static std::atomic_bool pack_nup_cancel{false};
static std::atomic_bool unpack_nup_cancel{false};
static std::atomic_bool edit_nup_cancel{false};

// eden-cli --pack-nup <dump> [<out.nup>]; the output defaults to <dump>.nup next to the dump.
static bool PackNup(const std::string& source, const std::string& output) {
    const auto to_path = [](const std::string& utf8) {
        return std::filesystem::path{std::u8string{utf8.begin(), utf8.end()}};
    };
    const auto dump_root = to_path(source);
    const auto archive =
        output.empty() ? FileSys::Nup::DefaultArchivePath(dump_root) : to_path(output);
    std::signal(SIGINT, [](int) { pack_nup_cancel = true; });

    const auto start = std::chrono::steady_clock::now();
    auto last_report = start;
    std::string error;
    const auto result = FileSys::Nup::PackDump(
        dump_root, archive, {},
        [&](const FileSys::Nup::PackProgress& p) {
            const auto now = std::chrono::steady_clock::now();
            if (now - last_report < std::chrono::seconds(1) && p.files_done != p.files_total) {
                return;
            }
            last_report = now;
            fmt::print(stderr, "\r{} / {} MiB, {} / {} files", p.bytes_done >> 20,
                       p.bytes_total >> 20, p.files_done, p.files_total);
        },
        &pack_nup_cancel, &error);
    const auto seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    fmt::print(stderr, "\n");
    switch (result) {
    case FileSys::Nup::PackResult::Success:
        fmt::print(stderr, "Wrote {} in {:.1f} s\n", archive.string(), seconds);
        return true;
    case FileSys::Nup::PackResult::Cancelled:
        fmt::print(stderr, "Cancelled\n");
        return false;
    default:
        fmt::print(stderr, "Packing failed: {}\n", error);
        return false;
    }
}

// eden-cli --unpack-nup <file.nup> [<out_dir>]; the output defaults to <name>/ next to the NUP.
static bool UnpackNup(const std::string& source, const std::string& output) {
    const auto to_path = [](const std::string& utf8) {
        return std::filesystem::path{std::u8string{utf8.begin(), utf8.end()}};
    };
    const auto archive = to_path(source);
    const auto out_dir =
        output.empty() ? FileSys::Nup::DefaultExtractPath(archive) : to_path(output);
    std::signal(SIGINT, [](int) { unpack_nup_cancel = true; });

    const auto start = std::chrono::steady_clock::now();
    auto last_report = start;
    std::string error;
    const auto result = FileSys::Nup::ExtractNup(
        archive, out_dir,
        [&](const FileSys::Nup::PackProgress& p) {
            const auto now = std::chrono::steady_clock::now();
            if (now - last_report < std::chrono::seconds(1) && p.files_done != p.files_total) {
                return;
            }
            last_report = now;
            fmt::print(stderr, "\r{} / {} MiB, {} / {} files", p.bytes_done >> 20,
                       p.bytes_total >> 20, p.files_done, p.files_total);
        },
        &unpack_nup_cancel, &error);
    const auto seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    fmt::print(stderr, "\n");
    switch (result) {
    case FileSys::Nup::ExtractResult::Success:
        fmt::print(stderr, "Extracted to {} in {:.1f} s\n", out_dir.string(), seconds);
        return true;
    case FileSys::Nup::ExtractResult::Cancelled:
        fmt::print(stderr, "Cancelled\n");
        return false;
    default:
        fmt::print(stderr, "Extraction failed: {}\n", error);
        return false;
    }
}

static bool EditNup(const Core::LaunchParams& lp) {
    const auto to_path = [](const std::string& utf8) {
        return std::filesystem::path{std::u8string{utf8.begin(), utf8.end()}};
    };
    const auto archive = to_path(*lp.edit_nup_source);
    FileSys::Nup::MetadataEdit edit;
    if (lp.edit_nup_name) {
        edit.title_name = *lp.edit_nup_name;
    }
    if (lp.edit_nup_publisher) {
        edit.publisher = *lp.edit_nup_publisher;
    }
    if (lp.edit_nup_version) {
        edit.display_version = *lp.edit_nup_version;
    }
    if (lp.edit_nup_icon) {
        if (lp.edit_nup_icon->empty()) {
            edit.icon = std::vector<u8>{};
        } else {
            const auto icon_path = to_path(*lp.edit_nup_icon);
            std::ifstream in(icon_path, std::ios::binary);
            if (!in) {
                fmt::print(stderr, "Cannot read icon {}\n", icon_path.string());
                return false;
            }
            edit.icon = std::vector<u8>((std::istreambuf_iterator<char>(in)), {});
            if (edit.icon->empty()) {
                fmt::print(stderr, "Icon file is empty\n");
                return false;
            }
        }
    }
    if (!edit.title_name && !edit.icon && !edit.display_version && !edit.publisher) {
        fmt::print(stderr, "Nothing to edit: pass --nup-name, --nup-icon, --nup-version or "
                           "--nup-publisher (or a --clear-* flag)\n");
        return false;
    }
    std::signal(SIGINT, [](int) { edit_nup_cancel = true; });
    std::string error;
    switch (FileSys::Nup::EditNupMetadata(archive, edit, &error, &edit_nup_cancel)) {
    case FileSys::Nup::EditResult::Success:
        fmt::print(stderr, "Updated {}\n", archive.string());
        return true;
    case FileSys::Nup::EditResult::NoChange:
        fmt::print(stderr, "No changes to apply\n");
        return true;
    case FileSys::Nup::EditResult::Cancelled:
        fmt::print(stderr, "Cancelled\n");
        return false;
    default:
        fmt::print(stderr, "Edit failed: {}\n", error);
        return false;
    }
}

extern "C" SDL_AppResult SDL_AppInit(void **appstate, int argc, char **argv) {
#ifdef _WIN32
    SetUnhandledExceptionFilter(WriteCrashDump);
#endif
    // SDL calls SDL_AppIterate back to back by default, which kept a host core at 100% polling
    // input; 240 Hz is plenty for input and the key script.
    SDL_SetHint(SDL_HINT_MAIN_CALLBACK_RATE, "240");
#if defined(__APPLE__) && TARGET_OS_IPHONE
    IosCaptureConsole();
    IOS_MARK("SDL_AppInit enter");
#endif
    FrontendCommon::LoadNx2Switches(argc > 0 ? argv[0] : nullptr);
    SdlState* state = new SdlState();
    IOS_MARK("SdlState constructed (Core::System + input)");
    *appstate = state;

#ifdef _WIN32
    if (AttachConsole(ATTACH_PARENT_PROCESS)) {
        freopen("CONOUT$", "wb", stdout);
        freopen("CONOUT$", "wb", stderr);
    }
    // Raise the CRT's open-file limit like the Qt frontend: loose RomFS folders keep thousands of
    // files open through the VFS cache.
    _setmaxstdio(8192);
#elif defined(__unix__) || defined(__APPLE__)
    // Same for the open file limit (macOS shells default to 256).
    if (struct rlimit rl; getrlimit(RLIMIT_NOFILE, &rl) == 0) {
        rl.rlim_cur = std::min<rlim_t>(8192, rl.rlim_max);
        setrlimit(RLIMIT_NOFILE, &rl);
    }
#endif

    Common::Log::Initialize();
    Common::Log::SetColorConsoleBackendEnabled(true);
    Common::Log::Start();
    IOS_MARK("logging started");

    int option_index = 0;
#ifdef _WIN32
    int argc_w;
    auto argv_w = CommandLineToArgvW(GetCommandLineW(), &argc_w);
    if (argv_w == nullptr) {
        LOG_CRITICAL(Frontend, "Failed to get command line arguments");
        return SDL_APP_FAILURE;
    }
#else
    wchar_t **argv_w = nullptr;
#endif
    Core::LaunchParams lp = Core::ParseLaunchParams(state->system, argc, argv, argv_w);
#ifdef _WIN32
    LocalFree(argv_w);
#endif
    SdlConfig config{lp.config_path};
    IOS_MARK("config loaded");
#if defined(__APPLE__) && TARGET_OS_IPHONE
    // Device facts, cache budgets, MoltenVK options and lifecycle observers (before Vulkan loads).
    lichii_platform_init();
    // Audio: playback session with a 20 ms hardware buffer and a larger SDL device buffer; the
    // default small buffer underran whenever a frame took longer than usual (audible skips).
    lichii_platform_configure_audio(0, 20.0);
    SDL_SetHint(SDL_HINT_AUDIO_CATEGORY, "playback");
    SDL_SetHint(SDL_HINT_AUDIO_DEVICE_SAMPLE_FRAMES, "1024");
    // Controllers come through Apple's GameController framework. SDL's HIDAPI layer on iOS only
    // adds a Bluetooth LE scan for Steam Controllers that ran every few seconds for the whole
    // session (battery, and Bluetooth prompts).
    SDL_SetHint(SDL_HINT_JOYSTICK_HIDAPI, "0");
#ifdef LICHII_IOS_UI
    // The app reads GameController devices itself (controls bridge), so SDL leaves them alone.
    SDL_SetHint(SDL_HINT_JOYSTICK_MFI, "0");
#endif
    // Per-draw GPU and service info logging costs frame time on the phone; keep warnings.
    Settings::values.log_filter.SetValue(
        "*:Info HW.GPU:Warning Render.Vulkan:Warning Service:Warning Service.NVDRV:Error "
        "Service.AM:Error Debug:Error");
    {
        Common::Log::Filter filter;
        filter.ParseFilterString(Settings::values.log_filter.GetValue());
        Common::Log::SetGlobalFilter(filter);
    }
#endif

    Core::ApplyLaunchParams(lp);

    // Informational options already printed by ApplyLaunchParams; handle them before any
    // archive operation so --help/--version never packs, unpacks or edits.
    if (lp.print_help || lp.print_version) {
        return SDL_APP_SUCCESS;
    }

    if (lp.pack_nup_source) {
        return PackNup(*lp.pack_nup_source, lp.filepath) ? SDL_APP_SUCCESS : SDL_APP_FAILURE;
    }

    if (lp.unpack_nup_source) {
        return UnpackNup(*lp.unpack_nup_source, lp.filepath) ? SDL_APP_SUCCESS : SDL_APP_FAILURE;
    }

    if (lp.edit_nup_source) {
        return EditNup(lp) ? SDL_APP_SUCCESS : SDL_APP_FAILURE;
    }

#if defined(__APPLE__) && TARGET_OS_IPHONE
    // The JIT must be on before the first guest instruction runs, or the kernel kills the process
    // with CODESIGNING "Invalid Page". Waiting here would block the main thread and trip the 10 s
    // scene-update watchdog, so park the launch and let SDL_AppIterate finish it.
    // dynarmic needs executable memory before the first guest instruction. On iOS 26+ (TXM)
    // only a debugger running StikDebug's universal script can provide it, so the launch is
    // parked while a background thread acquires the code pool; SDL_AppIterate shows progress,
    // asks StikDebug through its URL scheme and finishes the launch once the pool is ready
    // (blocking here would trip the 10 s scene-update watchdog).
    const bool txm = IosSupport::HasTxm();
    state->jit_have_stikdebug = IosSupport::CanOpenScheme("stikdebug");
    LOG_INFO(Frontend, "iOS {}: TXM {}, debugged {}, traced {}, StikDebug {}",
             IosSupport::MajorVersion(), txm, IosJitEnabled(), Common::IosJit::DebuggerAttached(),
             state->jit_have_stikdebug ? "installed" : "not installed");
    IOS_MARK(IosJitEnabled() ? "JIT enabled at launch, acquiring pool" : "JIT not enabled, waiting");
    Common::IosJit::StartAcquire(kIosJitPoolBytes, txm);
#ifdef LICHII_IOS_UI
    // The pool is acquired while the library is open; a game launches from the app.
    if (lp.filepath.empty()) {
        return IosStartApp(state, std::move(lp));
    }
#endif
    state->pending_launch = std::move(lp);
    state->jit_wait_start = SDL_GetTicks();
    return SDL_APP_CONTINUE;
#else
    return SDL_AppInitRest(state, std::move(lp));
#endif
}

static SDL_AppResult SDL_AppInitRest(SdlState* state, Core::LaunchParams lp) {
#if defined(__APPLE__) && TARGET_OS_IPHONE
    // Movies: software decoding (dav1d for AV1). The VideoToolbox AV1 hwaccel failed every
    // frame on iOS 27 and the game died as soon as the New Game movie started.
    Settings::values.nvdec_emulation.SetValue(Settings::NvdecEmulation::Cpu);
#ifdef LICHII_IOS_UI
    const bool ios_app = state->ios_config != nullptr;
#else
    constexpr bool ios_app = false;
#endif
    // Speed: a phone is a handheld (the game renders at its handheld resolution, not 4K docked),
    // and shaders build in the background instead of stalling frames (and audio) while they
    // compile. The app applies these once as defaults instead.
    if (!ios_app) {
        Settings::values.use_docked_mode.SetValue(Settings::ConsoleMode::Handheld);
        Settings::values.use_asynchronous_shaders.SetValue(true);
        // Resolution and the rest follow the device's defaults where the configuration keeps the
        // built-in value.
        lichii_platform_apply_device_defaults(LICHII_DEFAULTS_UNCHANGED);
    }
    // Memory: CPU and GPU share the app's ~6.6 GB budget (about 5.3 GB free at launch); the cache
    // budgets come from it (lichii_platform_init).
    // iOS has no command line: run the first game in the app's Documents folder, which the user
    // fills through the Files app (UIFileSharingEnabled in Info.plist).
    if (lp.filepath.empty()) {
        lp.fullscreen = true;
        // The per-process memory budget decides whether a Switch 2 title fits on this device.
        {
            u64 physical = 0;
            size_t physical_size = sizeof(physical);
            sysctlbyname("hw.memsize", &physical, &physical_size, nullptr, 0);
            LOG_INFO(Frontend, "iOS: physical memory {} MiB, available to this process {} MiB",
                     physical >> 20, static_cast<u64>(os_proc_available_memory()) >> 20);
        }
        const char* home = std::getenv("HOME");
        const std::filesystem::path documents = std::filesystem::path(home ? home : ".") / "Documents";
        std::error_code ec;
        std::filesystem::create_directories(documents, ec);
        std::vector<std::filesystem::path> candidates;
        // Documents and one level of subfolders (Files app copies and USB pushes both land there).
        for (auto it = std::filesystem::recursive_directory_iterator(documents, ec);
             it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
            if (ec) {
                break;
            }
            const auto& entry = *it;
            if (entry.is_directory(ec)) {
                if (it.depth() >= 1 || entry.path().filename() == "eden") {
                    it.disable_recursion_pending();
                }
                continue;
            }
            if (!entry.is_regular_file(ec)) {
                continue;
            }
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (ext == ".nup" || ext == ".nsp" || ext == ".xci" || ext == ".nca" || ext == ".nso" ||
                ext == ".nro") {
                candidates.push_back(entry.path());
            }
        }
        std::sort(candidates.begin(), candidates.end());
        if (!candidates.empty()) {
            lp.filepath = candidates.front().string();
            LOG_INFO(Frontend, "iOS: launching {}", lp.filepath);
        } else {
            LOG_CRITICAL(Frontend, "iOS: no game found in {}", documents.string());
            SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_INFORMATION, "Lichii",
                                     "No game found. Copy a .nup, .nsp or .xci into this app's folder "
                                     "(Files app > On My iPhone > Lichii), then launch again.",
                                     nullptr);
        }
    }
#endif
    if (lp.filepath.empty()) {
        LOG_CRITICAL(Frontend, "Failed to load ROM: No ROM specified");
        return SDL_APP_FAILURE;
    }

    if (!InitializeSystem(state)) {
        return SDL_APP_FAILURE;
    }

#ifdef LICHII_IOS_UI
    if (ios_app) {
        IosFrontend::ApplyGameConfig(IosFrontend::ReadProgramId(lp.filepath));
    }
#endif

    // Apply the command line arguments
    state->system.ApplySettings();

    // A second game in the same app session renders into the window of the first.
    if (state->emu_window) {
        SDL_ShowWindow(state->emu_window->GetSdlWindow());
    } else {
        switch (Settings::values.renderer_backend.GetValue()) {
#ifdef HAS_OPENGL
        case Settings::RendererBackend::OpenGL_GLSL:
        case Settings::RendererBackend::OpenGL_GLASM:
        case Settings::RendererBackend::OpenGL_SPIRV:
            state->emu_window = std::make_unique<EmuWindow_SDL3_GL>(&state->input_subsystem, state->system, lp.fullscreen);
            break;
#endif
        case Settings::RendererBackend::Vulkan:
            state->emu_window = std::make_unique<EmuWindow_SDL3_VK>(&state->input_subsystem, state->system, lp.fullscreen);
            break;
        case Settings::RendererBackend::Null:
            state->emu_window = std::make_unique<EmuWindow_SDL3_Null>(&state->input_subsystem, state->system, lp.fullscreen);
            break;
        default:
            LOG_CRITICAL(Frontend, "Invalid renderer backend");
            return SDL_APP_FAILURE;
        }
    }

#ifdef _WIN32
    Common::Windows::SetCurrentTimerResolutionToMaximum();
    state->system.CoreTiming().SetTimerResolutionNs(Common::Windows::GetCurrentTimerResolution());
#endif

    InitializeContent(state);
    state->system.GetUserChannel().clear();

    Service::AM::FrontendAppletParameters load_parameters{
        .applet_id = Service::AM::AppletId::Application,
    };
    const char* const handoff_env = std::getenv("EDEN_CLI_RESTART_HANDOFF");
    const bool restarted = handoff_env != nullptr && handoff_env[0] != '\0';
    if (restarted) {
        auto handoff = ReadRestartHandoff();
        if (!handoff) {
            // Booting as a frontend launch would clear the game's temp storage.
            LOG_CRITICAL(Frontend, "Restart handoff {} is missing or damaged, not booting",
                         Common::FS::PathToUTF8String(RestartHandoffPath()));
            return SDL_APP_FAILURE;
        }
        state->system.GetUserChannel() = std::move(handoff->user_channel);
        load_parameters.program_id = handoff->program_id;
        load_parameters.launch_type = Service::AM::LaunchType::ApplicationInitiated;
        load_parameters.program_index = handoff->program_index;
        load_parameters.previous_program_index = handoff->previous_program_index;
        LOG_INFO(Frontend, "Restarted by the game: program index {}, {} user channel entries",
                 handoff->program_index, state->system.GetUserChannel().size());
    }
#ifdef LICHII_IOS_UI
    if (ios_app) {
        LichiiContent::InstallFrontendApplets(state->system);
        load_parameters.applet_id =
            static_cast<Service::AM::AppletId>(LichiiContent::TakePendingAppletId(lp.filepath));
    }
#endif
    IOS_MARK("system.Load");
    const Core::SystemResultStatus load_result = state->system.Load(*state->emu_window, lp.filepath, load_parameters);
    IOS_MARK("system.Load returned");
#ifdef LICHII_IOS_UI
    state->ios_load_result = load_result;
#endif
    switch (load_result) {
    case Core::SystemResultStatus::Success:
        if (restarted) {
            std::error_code ec;
            std::filesystem::remove(RestartHandoffPath(), ec);
        }
        break; // Expected case
    case Core::SystemResultStatus::ErrorGetLoader:
        LOG_CRITICAL(Frontend, "Failed to obtain loader for {}!", lp.filepath);
        return SDL_APP_FAILURE;
    case Core::SystemResultStatus::ErrorLoader:
        LOG_CRITICAL(Frontend, "Failed to load ROM!");
        return SDL_APP_FAILURE;
    case Core::SystemResultStatus::ErrorNotInitialized:
        LOG_CRITICAL(Frontend, "CPUCore not initialized");
        return SDL_APP_FAILURE;
    case Core::SystemResultStatus::ErrorVideoCore:
        LOG_CRITICAL(Frontend, "Failed to initialize VideoCore!");
        return SDL_APP_FAILURE;
    default:
        const u16 loader_id = u16(Core::SystemResultStatus::ErrorLoader);
        const u16 error_id = u16(load_result) - loader_id;
        LOG_CRITICAL(Frontend,
            "While attempting to load the ROM requested, an error occurred. Please "
            "refer to the Eden wiki for more information or the Eden discord for "
            "additional help.\n\nError Code: {:04X}-{:04X}\nError Description: {}",
            loader_id, error_id, Loader::ResultStatus(error_id));
        return SDL_APP_FAILURE;
    }

    if (lp.use_multiplayer) {
        if (auto member = Network::GetRoomMember().lock()) {
            member->BindOnChatMessageReceived(OnMessageReceived);
            member->BindOnStatusMessageReceived(OnStatusMessageReceived);
            member->BindOnStateChanged(OnStateChanged);
            member->BindOnError(OnNetworkError);
            LOG_DEBUG(Network, "Start connection to {}:{} with nickname {}", lp.address, lp.port, lp.nickname);
            member->Join(lp.nickname, lp.address.c_str(), lp.port, 0, Network::NoPreferredIP, lp.password);
        } else {
            LOG_ERROR(Network, "Could not access RoomMember");
            return SDL_APP_FAILURE;
        }
    }

    // Core is loaded, start the GPU (makes the GPU contexts current to this thread)
    IOS_MARK("GPU start");
    state->system.GPU().Start();
    state->system.GetCpuManager().OnGpuReady();

    if (Settings::values.use_disk_shader_cache.GetValue()) {
        state->system.Renderer().ReadRasterizer()->LoadDiskResources(
            state->system.GetApplicationProcessProgramID(), std::stop_token{},
            []([[maybe_unused]] VideoCore::LoadCallbackStage stage,
               [[maybe_unused]] size_t value, [[maybe_unused]] size_t total) {
#if defined(__APPLE__) && TARGET_OS_IPHONE
                lichii_platform_note_shader_progress(static_cast<int>(stage), value, total);
#endif
            });
    }

    // don't do anything, SDL3 already exists for us :D
    state->system.RegisterExitCallback([] {});
#if defined(__APPLE__) && TARGET_OS_IPHONE
    // The app brings its own on-screen controller.
    if (!ios_app) {
        IosInstallTouchControls(state);
    }
#endif
#ifdef LICHII_IOS_UI
    if (ios_app) {
        // A game that exits by itself returns to the library.
        state->system.RegisterExitCallback([] { lichii_emu_stop(); });
        IosFrontend::SetRunningGame(lp.filepath, state->system.GetApplicationProcessProgramID(),
                                    IosGameWindow(state));
    }
#endif

    // A game executing a program (e.g. restarting after a settings change) writes its launch state
    // to the restart handoff and quits with kRestartExitCode; the launcher boots the same file again
    // with EDEN_CLI_RESTART_HANDOFF set. Mirrors MainWindow::OnExecuteProgram.
    state->system.RegisterExecuteProgramCallback([system = &state->system](std::size_t program_index) {
        RestartHandoff handoff{
            .program_index = static_cast<s32>(program_index),
            .user_channel = system->GetUserChannel(),
        };
        const u64 previous_program_id = system->GetApplicationProcessProgramID();
        if (previous_program_id > static_cast<u64>(Service::AM::AppletProgramId::MaxProgramId)) {
            handoff.previous_program_index =
                static_cast<s32>(previous_program_id - FileSys::GetBaseTitleID(previous_program_id));
            handoff.program_id = previous_program_id;
        }
        // Without the handoff, a relaunch would boot as a frontend launch and clear temp storage.
        if (!WriteRestartHandoff(handoff)) {
            LOG_CRITICAL(Frontend, "ExecuteProgram: could not write {}, not restarting",
                         Common::FS::PathToUTF8String(RestartHandoffPath()));
            return;
        }
        LOG_INFO(Frontend, "ExecuteProgram: program index {}, exiting with code {} to restart",
                 program_index, kRestartExitCode);
        restart_requested = true;
    });

    // QLaunch launched applications (should) now reload shader cache.
    static std::mutex shader_cache_reload_mutex;
    state->system.RegisterApplicationChangedCallback([&state](u64 changed_program_id) {
        if (!Settings::values.use_disk_shader_cache.GetValue()) {
            return;
        }

        std::scoped_lock lk{shader_cache_reload_mutex};

        LOG_INFO(Frontend, "Reloading disk shader cache for {:016X}", changed_program_id);

        state->system.Pause();
        state->system.GPU().WaitForIdle();
        state->system.GPU().ObtainContext();

        state->system.Renderer().ReadRasterizer()->LoadDiskResources(
            changed_program_id, std::stop_token{},
            [](VideoCore::LoadCallbackStage, size_t, size_t) {});

        state->system.GPU().ReleaseContext();
        state->system.Run();
    });

    IOS_MARK("system.Run");
    void(state->system.Run());
#if defined(__APPLE__) && TARGET_OS_IPHONE
    lichii_platform_set_playing(1);
#endif
    if (state->system.DebuggerEnabled())
        state->system.InitializeDebugger();
#ifdef LICHII_IOS_UI
    if (ios_app) {
        IosFrontend::SetState(LICHII_EMU_RUNNING);
    }
#endif
    return SDL_APP_CONTINUE;
}
// EDEN_NX2_KEY_SCRIPT="seconds:code,seconds:-code,..." (or "@<file>"): press (code) or release (-code) keyboard
// keys, by SDL scancode, that many seconds after start. Lets hidden test runs drive menus.
double Nx2KeyClockSeconds(); // emu_window_sdl3.cpp

static void RunKeyScript(InputCommon::InputSubsystem& input_subsystem) {
    struct KeyEvent {
        double seconds; // when frame is 0
        u64 frame;      // "f<frame>:code": after that many presents
        int code;
    };
    static const std::vector<KeyEvent> script = [] {
        std::vector<KeyEvent> events;
        if (const char* text = std::getenv("EDEN_NX2_KEY_SCRIPT")) {
            // "@<file>" reads the list from a file: long scripts in the environment slow down
            // every getenv in the process.
            std::string list{text};
            if (!list.empty() && list[0] == '@') {
                std::ifstream file{list.substr(1)};
                list.assign(std::istreambuf_iterator<char>{file}, {});
            }
            size_t start = 0;
            while (start < list.size()) {
                const size_t end = (std::min)(list.find(',', start), list.size());
                const std::string item = list.substr(start, end - start);
                if (const size_t colon = item.find(':'); colon != std::string::npos) {
                    const std::string when = item.substr(0, colon);
                    const int code = std::stoi(item.substr(colon + 1));
                    if (!when.empty() && when[0] == 'f') {
                        events.push_back({0.0, std::stoull(when.substr(1)), code});
                    } else {
                        events.push_back({std::stod(when), 0, code});
                    }
                }
                start = end + 1;
            }
        }
        return events;
    }();
    static size_t next = 0;
    const double elapsed = Nx2KeyClockSeconds();
    const u64 frame = Core::g_nx2_present_count.load();
    while (next < script.size() && (script[next].frame != 0 ? script[next].frame <= frame
                                                            : script[next].seconds <= elapsed)) {
        const int code = script[next].code;
        LOG_INFO(Frontend, "Key script: {} key {} at {:.1f} s (frame {})",
                 code < 0 ? "release" : "press", code < 0 ? -code : code, elapsed, frame);
        if (code < 0) {
            input_subsystem.GetKeyboard()->ReleaseKey(static_cast<std::size_t>(-code));
        } else {
            input_subsystem.GetKeyboard()->PressKey(static_cast<std::size_t>(code));
        }
        ++next;
    }
}

extern "C" SDL_AppResult SDL_AppIterate(void *appstate) {
    SdlState *state = (SdlState *)appstate;
#ifdef LICHII_IOS_UI
    if (state->ios_config) {
        if (const auto result = IosAppIterate(state)) {
            return *result;
        }
    }
#endif
#if defined(__APPLE__) && TARGET_OS_IPHONE
    if (state->pending_launch) {
        using Common::IosJit::State;
        const State jit = Common::IosJit::GetState();
        if (jit == State::Ready) {
            LOG_INFO(Frontend, "iOS: JIT pool ready after {} ms", SDL_GetTicks() - state->jit_wait_start);
            IOS_MARK("JIT pool ready");
            IosSupport::EndBackgroundTask(state->jit_bg_task);
            state->jit_bg_task = 0;
            if (state->wait_renderer) {
                SDL_DestroyRenderer(state->wait_renderer);
                state->wait_renderer = nullptr;
            }
            if (state->wait_window) {
                SDL_DestroyWindow(state->wait_window);
                state->wait_window = nullptr;
            }
            Core::LaunchParams lp = std::move(*state->pending_launch);
            state->pending_launch.reset();
            return SDL_AppInitRest(state, std::move(lp));
        }
        if (!state->wait_window) {
            state->wait_window = SDL_CreateWindow("Lichii", 0, 0, SDL_WINDOW_FULLSCREEN | SDL_WINDOW_HIGH_PIXEL_DENSITY);
            if (state->wait_window) {
                state->wait_renderer = SDL_CreateRenderer(state->wait_window, nullptr);
            }
        }
        // Hand off to StikDebug once (after the first frame, so there is something to come back
        // to): it attaches to this PID, runs universal.js, serves the pool request and returns.
        // With TXM also when the process is already debugged: JIT enabled from StikDebug's app
        // list attaches without the script, and only this request brings it.
        const bool txm = IosSupport::HasTxm();
        if (!state->jit_url_opened && state->wait_window && state->jit_have_stikdebug &&
            jit == State::Waiting && (txm || !IosJitEnabled())) {
            state->jit_url_opened = true;
            std::string url = "stikdebug://enable-jit?bundle-id=" + IosBundleIdentifier() +
                              "&pid=" + std::to_string(getpid());
            if (txm) {
                url += "&script-name=universal.js";
            }
            state->jit_bg_task = IosSupport::BeginBackgroundTask("lichii-jit");
            IOS_MARK("asking StikDebug for JIT");
            LOG_INFO(Frontend, "iOS: opening {}", url);
            IosSupport::OpenUrl(url.c_str());
        }
        if (jit == State::Failed && state->jit_failed_at == 0) {
            state->jit_failed_at = SDL_GetTicks();
            IOS_MARK("JIT pool failed");
            IosSupport::EndBackgroundTask(state->jit_bg_task);
            state->jit_bg_task = 0;
        }
        if (state->wait_renderer) {
            const Uint64 waited = (SDL_GetTicks() - state->jit_wait_start) / 1000;
            std::vector<std::string> lines;
            switch (jit) {
            case State::Waiting:
                if (txm && IosJitEnabled()) {
                    lines = {"JIT is on, but without its script.", "",
                             "Let StikDebug enable JIT from this",
                             "app's request (script universal.js);",
                             "this screen continues by itself."};
                } else if (state->jit_have_stikdebug) {
                    lines = {"Requesting JIT from StikDebug...", "",
                             "Tap \"Enable JIT\" in StikDebug if it asks;",
                             "this screen continues by itself."};
                } else {
                    lines = {"Waiting for JIT...", "",
                             "Install StikDebug 3.2 or newer, then",
                             "enable JIT for this app (universal.js)."};
                }
                break;
            case State::Acquiring:
                lines = {"Preparing JIT memory...", "",
                         "StikDebug has mapped " +
                             std::to_string(Common::IosJit::PreparedBytes() >> 20) + " of " +
                             std::to_string(kIosJitPoolBytes >> 20) + " MiB."};
                break;
            case State::Failed: {
                lines = {"JIT unavailable", ""};
                std::string err = Common::IosJit::GetError();
                while (!err.empty()) {
                    std::size_t cut = std::min<std::size_t>(err.size(), 44);
                    if (cut < err.size()) {
                        const auto sp = err.rfind(' ', cut);
                        if (sp != std::string::npos && sp > 20) {
                            cut = sp;
                        }
                    }
                    lines.push_back(err.substr(0, cut));
                    err.erase(0, cut);
                    while (!err.empty() && err.front() == ' ') {
                        err.erase(0, 1);
                    }
                }
                lines.push_back("");
                lines.push_back("Closing in a moment; launch again.");
                break;
            }
            case State::Ready:
                break;
            }
            SDL_SetRenderDrawColor(state->wait_renderer, 12, 12, 16, 255);
            SDL_RenderClear(state->wait_renderer);
            SDL_SetRenderScale(state->wait_renderer, 3.0f, 3.0f);
            SDL_SetRenderDrawColor(state->wait_renderer, 235, 235, 235, 255);
            float y = 40.0f;
            for (const auto& line : lines) {
                SDL_RenderDebugText(state->wait_renderer, 16, y, line.c_str());
                y += 10.0f;
            }
            const std::string elapsed = "waited " + std::to_string(waited) + " s";
            SDL_RenderDebugText(state->wait_renderer, 16, y + 10.0f, elapsed.c_str());
            SDL_SetRenderScale(state->wait_renderer, 1.0f, 1.0f);
            SDL_RenderPresent(state->wait_renderer);
        }
        // No message boxes here: SDL's UIKit alert spins the run loop, which re-enters this
        // callback and recursed until the stack overflowed.
        if (jit == State::Failed && SDL_GetTicks() - state->jit_failed_at > 20 * 1000) {
            LOG_CRITICAL(Frontend, "iOS: exiting, JIT pool failed: {}", Common::IosJit::GetError());
            return SDL_APP_FAILURE;
        }
        if (SDL_GetTicks() - state->jit_wait_start > 10 * 60 * 1000) {
            LOG_CRITICAL(Frontend, "iOS: JIT still not available after 10 minutes, exiting");
            return SDL_APP_FAILURE;
        }
        SDL_Delay(20);
        return SDL_APP_CONTINUE;
    }
#endif
#if defined(__APPLE__) && TARGET_OS_IPHONE
    // The app maps controllers itself (controls bridge).
    if (state->ios_controllers_dirty && state->emu_window && !IosNativeApp(state)) {
        state->ios_controllers_dirty = false;
        IosApplyControllers(state);
    }
#endif
    RunKeyScript(state->input_subsystem);
    if (state->emu_window) {
        state->emu_window->ApplyPendingTitle();
    }
    if (restart_requested) {
        return SDL_APP_SUCCESS;
    }
    return state->emu_window && state->emu_window->IsOpen() ? SDL_APP_CONTINUE : SDL_APP_SUCCESS;
}
extern "C" SDL_AppResult SDL_AppEvent(void *appstate, SDL_Event *event) {
    SdlState *state = (SdlState *)appstate;
#if defined(__APPLE__) && TARGET_OS_IPHONE
    switch (event->type) {
    case SDL_EVENT_GAMEPAD_ADDED:
    case SDL_EVENT_GAMEPAD_REMOVED:
    case SDL_EVENT_JOYSTICK_ADDED:
    case SDL_EVENT_JOYSTICK_REMOVED:
        // The input driver handles the device in its event watch; remap on the next frame.
        state->ios_controllers_dirty = true;
        break;
    case SDL_EVENT_WILL_ENTER_BACKGROUND:
        // iOS forbids GPU work in the background (Metal fails the command buffers and the
        // Vulkan scheduler aborted on the lost device): stop emulating until we are back.
        if (state->system.IsPoweredOn() && !state->pending_launch) {
            LOG_INFO(Frontend, "iOS: entering background, pausing");
            void(state->system.Pause());
            state->ios_paused_for_background = true;
        }
        break;
#ifdef LICHII_IOS_UI
    case SDL_EVENT_DROP_FILE:
        // SDL reports URLs and files opened with the app this way.
        if (state->ios_config && event->drop.data != nullptr) {
            IosFrontend::OpenUrl(event->drop.data);
        }
        break;
#endif
    case SDL_EVENT_LOW_MEMORY:
        LOG_WARNING(Frontend, "iOS: low memory warning ({} MiB left to this process)",
                    os_proc_available_memory() >> 20);
        break;
    case SDL_EVENT_TERMINATING:
        LOG_INFO(Frontend, "iOS: terminating");
        Common::Log::Stop();
        break;
    case SDL_EVENT_DID_ENTER_FOREGROUND:
        if (state->ios_paused_for_background) {
#ifdef LICHII_IOS_UI
            // Paused from the in-game menu: stays paused.
            if (IosFrontend::GetState() == LICHII_EMU_PAUSED) {
                state->ios_paused_for_background = false;
                break;
            }
#endif
            LOG_INFO(Frontend, "iOS: back in the foreground, resuming");
            state->ios_paused_for_background = false;
            void(state->system.Run());
        }
        break;
    default:
        break;
    }
#endif
    if (state->emu_window)
        state->emu_window->OnEvent(*event);
    return SDL_APP_CONTINUE;
}
extern "C" void SDL_AppQuit(void *appstate, SDL_AppResult result) {
    SdlState *state = (SdlState *)appstate;
    if (!state) {
        return;
    }
    state->system.DetachDebugger();
    // SDL calls this even when SDL_AppInit never powered the system on (no ROM given, --help or
    // --version): the kernel is uninitialised then, and Pause()/ShutdownMainProcess() would
    // dereference it.
    if (state->system.IsPoweredOn()) {
        void(state->system.Pause());
        state->system.ShutdownMainProcess();
    }
    const bool network_initialized = state->network_initialized;
    delete state;
    if (network_initialized) {
        Network::Shutdown();
    }
    if (restart_requested) {
        // SDL3 maps SDL_APP_SUCCESS to exit code 0; quit SDL ourselves to return the restart code.
        SDL_Quit();
        std::exit(kRestartExitCode);
    }
}

#define VMA_IMPLEMENTATION
#include "video_core/vulkan_common/vma.h"
