// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "ios_frontend/lichii_bridge_controls.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

#include "common/settings.h"
#include "core/core.h"
#include "hid_core/frontend/emulated_controller.h"
#include "hid_core/hid_core.h"
#include "hid_core/hid_types.h"
#include "input_common/drivers/virtual_gamepad.h"
#include "input_common/main.h"
#include "ios_frontend/controls_host.h"
#include "ios_frontend/controls_router.h"

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/task_info.h>
#if __has_include(<os/proc.h>)
#include <os/proc.h>
#define LICHII_HAVE_OS_PROC 1
#endif
#endif

namespace LichiiControls {
namespace {

constexpr std::size_t HandheldPort = 8;

std::mutex attach_mutex;
Core::System* attached_system{};
InputCommon::InputSubsystem* attached_input{};

InputCommon::VirtualGamepad* Gamepad() {
    std::scoped_lock lock{attach_mutex};
    return attached_input ? attached_input->GetVirtualGamepad() : nullptr;
}

void SendButton(int player, int button, bool pressed) {
    if (auto* pad = Gamepad()) {
        pad->SetButtonState(static_cast<std::size_t>(player), button, pressed);
        if (player == 0) {
            pad->SetButtonState(HandheldPort, button, pressed);
        }
    }
}

void SendStick(int player, int stick, float x, float y) {
    if (auto* pad = Gamepad()) {
        pad->SetStickPosition(static_cast<std::size_t>(player), stick, x, y);
        if (player == 0) {
            pad->SetStickPosition(HandheldPort, stick, x, y);
        }
    }
}

void SendMotion(int player, u64 delta_us, float gx, float gy, float gz, float ax, float ay,
                float az) {
    if (player < 0 || player >= PlayerCount) {
        return;
    }
    if (auto* pad = Gamepad()) {
        pad->SetMotionState(static_cast<std::size_t>(player), delta_us, gx, gy, gz, ax, ay, az);
        if (player == 0) {
            pad->SetMotionState(HandheldPort, delta_us, gx, gy, gz, ax, ay, az);
        }
    }
}

Router& GetRouter() {
    static Router router{&SendButton, &SendStick};
    return router;
}

template <typename Fn>
void EditConfig(int32_t slot, Fn&& edit) {
    Router& router = GetRouter();
    if (!router.IsConnected(slot)) {
        return;
    }
    PadConfig config = router.Config(slot);
    edit(config);
    router.SetConfig(slot, config);
}

bool ValidStick(int32_t stick) {
    return stick == 0 || stick == 1;
}

} // Anonymous namespace

void Attach(Core::System& system, InputCommon::InputSubsystem& input) {
    {
        std::scoped_lock lock{attach_mutex};
        attached_system = &system;
        attached_input = &input;
    }
    GetRouter().Resend();
}

void Detach() {
    std::scoped_lock lock{attach_mutex};
    attached_system = nullptr;
    attached_input = nullptr;
}

} // namespace LichiiControls

using namespace LichiiControls;

extern "C" {

void lichii_controls_touch_button(int32_t player, int32_t button, bool pressed) {
    GetRouter().TouchButton(player, button, pressed);
}

void lichii_controls_touch_stick(int32_t player, int32_t stick, float x, float y) {
    GetRouter().TouchStick(player, stick, x, y);
}

void lichii_controls_release_touch(void) {
    GetRouter().ReleaseTouch();
}

void lichii_controls_motion(int32_t player, uint64_t delta_us, float gyro_x, float gyro_y,
                            float gyro_z, float accel_x, float accel_y, float accel_z) {
    SendMotion(player, delta_us, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z);
}

int32_t lichii_pad_connect(int32_t kind) {
    return GetRouter().Connect(kind);
}

void lichii_pad_disconnect(int32_t slot) {
    GetRouter().Disconnect(slot);
}

void lichii_pad_set_player(int32_t slot, int32_t player) {
    GetRouter().SetPlayer(slot, player);
}

int32_t lichii_pad_get_player(int32_t slot) {
    return GetRouter().IsConnected(slot) ? GetRouter().Config(slot).player : -1;
}

void lichii_pad_element(int32_t slot, int32_t element, bool pressed) {
    GetRouter().Element(slot, element, pressed);
}

void lichii_pad_axis(int32_t slot, int32_t axis, float value) {
    GetRouter().Axis(slot, axis, value);
}

void lichii_pad_motion(int32_t slot, uint64_t delta_us, float gyro_x, float gyro_y, float gyro_z,
                       float accel_x, float accel_y, float accel_z) {
    const int32_t player = lichii_pad_get_player(slot);
    SendMotion(player, delta_us, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z);
}

void lichii_pad_set_mapping(int32_t slot, int32_t element, int32_t target) {
    if (element < 0 || element >= ElementCount || !IsValidTarget(target)) {
        return;
    }
    EditConfig(slot, [&](PadConfig& config) { config.map[element] = target; });
}

int32_t lichii_pad_get_mapping(int32_t slot, int32_t element) {
    if (element < 0 || element >= ElementCount || !GetRouter().IsConnected(slot)) {
        return LICHII_ACTION_NONE;
    }
    return GetRouter().Config(slot).map[element];
}

void lichii_pad_reset_mapping(int32_t slot, bool by_label) {
    const int kind = GetRouter().Kind(slot);
    EditConfig(slot, [&](PadConfig& config) { config.map = DefaultMapping(kind, by_label); });
}

void lichii_pad_set_deadzone(int32_t slot, int32_t stick, float inner, float outer) {
    if (!ValidStick(stick)) {
        return;
    }
    EditConfig(slot, [&](PadConfig& config) {
        config.sticks[stick].inner = inner;
        config.sticks[stick].outer = outer;
    });
}

void lichii_pad_get_deadzone(int32_t slot, int32_t stick, float* inner, float* outer) {
    const StickConfig config =
        ValidStick(stick) ? GetRouter().Config(slot).sticks[stick] : StickConfig{};
    if (inner) {
        *inner = config.inner;
    }
    if (outer) {
        *outer = config.outer;
    }
}

void lichii_pad_set_trigger_threshold(int32_t slot, float threshold) {
    EditConfig(slot, [&](PadConfig& config) { config.trigger_threshold = threshold; });
}

float lichii_pad_get_trigger_threshold(int32_t slot) {
    return GetRouter().Config(slot).trigger_threshold;
}

void lichii_pad_set_invert_y(int32_t slot, int32_t stick, bool invert) {
    if (!ValidStick(stick)) {
        return;
    }
    EditConfig(slot, [&](PadConfig& config) { config.sticks[stick].invert_y = invert; });
}

bool lichii_pad_get_invert_y(int32_t slot, int32_t stick) {
    return ValidStick(stick) && GetRouter().Config(slot).sticks[stick].invert_y;
}

void lichii_pad_set_swap_sticks(int32_t slot, bool swap) {
    EditConfig(slot, [&](PadConfig& config) { config.swap_sticks = swap; });
}

bool lichii_pad_get_swap_sticks(int32_t slot) {
    return GetRouter().Config(slot).swap_sticks;
}

void lichii_pad_set_menu_chord(bool enabled) {
    GetRouter().SetMenuChord(enabled);
}

bool lichii_pad_get_menu_chord(void) {
    return GetRouter().MenuChord();
}

uint32_t lichii_controls_take_actions(void) {
    return GetRouter().TakeActions();
}

bool lichii_controls_rumble(int32_t player, LichiiRumble* out) {
    if (out == nullptr || player < 0 || player >= PlayerCount) {
        return false;
    }
    *out = {};
    std::scoped_lock lock{attach_mutex};
    if (attached_system == nullptr || !attached_system->IsPoweredOn()) {
        return false;
    }
    if (!Settings::values.vibration_enabled.GetValue()) {
        return true;
    }
    auto& hid = attached_system->HIDCore();
    const auto merge = [out](const Core::HID::EmulatedController* controller) {
        if (controller == nullptr || !controller->IsConnected()) {
            return;
        }
        const auto left = controller->GetActualVibrationValue(Core::HID::DeviceIndex::Left);
        const auto right = controller->GetActualVibrationValue(Core::HID::DeviceIndex::Right);
        out->left_low = (std::max)(out->left_low, left.low_amplitude);
        out->left_high = (std::max)(out->left_high, left.high_amplitude);
        out->right_low = (std::max)(out->right_low, right.low_amplitude);
        out->right_high = (std::max)(out->right_high, right.high_amplitude);
    };
    merge(hid.GetEmulatedControllerByIndex(static_cast<std::size_t>(player)));
    if (player == 0) {
        merge(hid.GetEmulatedController(Core::HID::NpadIdType::Handheld));
    }
    const auto& players = Settings::values.players.GetValue();
    if (static_cast<std::size_t>(player) < players.size()) {
        const auto& config = players[static_cast<std::size_t>(player)];
        if (!config.vibration_enabled) {
            *out = {};
        } else {
            const float strength = static_cast<float>(config.vibration_strength) / 100.0f;
            out->left_low = (std::min)(out->left_low * strength, 1.0f);
            out->left_high = (std::min)(out->left_high * strength, 1.0f);
            out->right_low = (std::min)(out->right_low * strength, 1.0f);
            out->right_high = (std::min)(out->right_high * strength, 1.0f);
        }
    }
    return true;
}

bool lichii_controls_system_stats(LichiiSystemStats* out) {
    if (out == nullptr) {
        return false;
    }
    *out = {};
    out->cpu_percent = -1.0;
    out->cpu_cores = std::thread::hardware_concurrency();
#ifdef __APPLE__
    static std::mutex stats_mutex;
    static double last_cpu_seconds = -1.0;
    static std::chrono::steady_clock::time_point last_wall{};

    task_vm_info_data_t vm{};
    mach_msg_type_number_t vm_count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&vm),
                  &vm_count) == KERN_SUCCESS) {
        out->memory_footprint = vm.phys_footprint;
    }
#ifdef LICHII_HAVE_OS_PROC
    if (__builtin_available(iOS 13.0, *)) {
        out->memory_available = os_proc_available_memory();
    }
#endif

    double cpu_seconds = 0.0;
    thread_act_array_t threads{};
    mach_msg_type_number_t thread_count = 0;
    if (task_threads(mach_task_self(), &threads, &thread_count) == KERN_SUCCESS) {
        out->threads = thread_count;
        for (mach_msg_type_number_t i = 0; i < thread_count; ++i) {
            thread_basic_info_data_t info{};
            mach_msg_type_number_t count = THREAD_BASIC_INFO_COUNT;
            if (thread_info(threads[i], THREAD_BASIC_INFO, reinterpret_cast<thread_info_t>(&info),
                            &count) == KERN_SUCCESS) {
                cpu_seconds += info.user_time.seconds + info.user_time.microseconds * 1e-6 +
                               info.system_time.seconds + info.system_time.microseconds * 1e-6;
            }
            mach_port_deallocate(mach_task_self(), threads[i]);
        }
        vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(threads),
                      thread_count * sizeof(thread_act_t));
    }
    // The task's own times cover the threads that already exited.
    mach_task_basic_info_data_t basic{};
    mach_msg_type_number_t basic_count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&basic),
                  &basic_count) == KERN_SUCCESS) {
        cpu_seconds += basic.user_time.seconds + basic.user_time.microseconds * 1e-6 +
                       basic.system_time.seconds + basic.system_time.microseconds * 1e-6;
    }

    std::scoped_lock lock{stats_mutex};
    const auto now = std::chrono::steady_clock::now();
    if (last_cpu_seconds >= 0.0) {
        const double wall = std::chrono::duration<double>(now - last_wall).count();
        if (wall > 0.0) {
            out->cpu_percent = (std::max)(0.0, (cpu_seconds - last_cpu_seconds) / wall * 100.0);
        }
    }
    last_cpu_seconds = cpu_seconds;
    last_wall = now;
#endif
    return true;
}

} // extern "C"
