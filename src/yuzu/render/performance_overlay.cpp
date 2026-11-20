// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "common/game_settings.h"
#include "common/settings.h"
#include "qt_common/config/catalog_text.h"
#include "core/perf_stats.h"
#include "video_core/render_probe.h"
#include "performance_overlay.h"
#include "ui_performance_overlay.h"
#include "video_core/shader_notify.h"
#ifdef EGLANTINE_STREAMLINE
#include "video_core/vulkan_common/streamline.h"
#endif
#if defined(_WIN32) || (defined(__linux__) && !defined(__ANDROID__))
#include "video_core/vulkan_common/ffx_runtime.h"
#endif

#include "main_window.h"

#include <QChart>
#include <QCoreApplication>
#include <QFontDatabase>
#include <QFrame>
#include <QLabel>
#include <QChartView>
#include <QGraphicsLayout>
#include <QGuiApplication>
#include <QLineSeries>
#include <QMouseEvent>
#include <QPainter>
#include <QValueAxis>
#include <QWindow>

#ifdef _WIN32
// clang-format off
#include <windows.h>
#include <dxgi1_4.h>
#include <psapi.h>
// clang-format on
#elif defined(__linux__)
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include "common/memory_detect.h"
#endif

#ifdef HAS_LSFG
#include <filesystem>
#include "video_core/frame_gen/lossless_dll.h"
#endif

namespace {

constexpr double GiB = 1024.0 * 1024.0 * 1024.0;

// LSFG without its Lossless.dll never generates a frame.
bool LosslessMissing() {
#ifdef HAS_LSFG
    std::error_code ec;
    return Settings::values.frame_gen_type.GetValue() == Settings::FrameGenType::Lsfg &&
           !std::filesystem::exists(VideoCore::FrameGen::GetLosslessDllPath(), ec);
#else
    return false;
#endif
}

bool QueryProcessRam(double& used_gb, double& free_gb) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    counters.cb = sizeof(counters);
    if (!K32GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        return false;
    }
    used_gb = static_cast<double>(counters.WorkingSetSize) / GiB;
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    free_gb = GlobalMemoryStatusEx(&status) ? static_cast<double>(status.ullAvailPhys) / GiB : -1.0;
    return true;
#elif defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    u64 total_pages = 0;
    u64 rss_pages = 0;
    const long page_size = sysconf(_SC_PAGESIZE);
    if (!(statm >> total_pages >> rss_pages) || page_size <= 0) {
        return false;
    }
    used_gb = static_cast<double>(rss_pages) * static_cast<double>(page_size) / GiB;
    free_gb = -1.0;
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    u64 value = 0;
    std::string unit;
    while (meminfo >> key >> value >> unit) {
        if (key == "MemAvailable:") {
            free_gb = static_cast<double>(value * 1024ULL) / GiB;
            break;
        }
    }
    return true;
#elif defined(__APPLE__)
    task_vm_info_data_t vm_info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&vm_info),
                  &count) != KERN_SUCCESS) {
        return false;
    }
    used_gb = static_cast<double>(vm_info.phys_footprint) / GiB;
    const u64 total_phys = Common::GetMemInfo().TotalPhysicalMemory;
    free_gb = total_phys > vm_info.phys_footprint
                  ? static_cast<double>(total_phys - vm_info.phys_footprint) / GiB
                  : -1.0;
    return true;
#else
    used_gb = -1.0;
    free_gb = -1.0;
    return false;
#endif
}

bool QueryProcessVram(double& used_gb, double& free_gb) {
#ifdef _WIN32
    using CreateFactoryFn = HRESULT(WINAPI*)(REFIID, void**);
    static const CreateFactoryFn create_factory = []() -> CreateFactoryFn {
        const HMODULE dxgi = LoadLibraryW(L"dxgi.dll");
        if (!dxgi) {
            return nullptr;
        }
        return reinterpret_cast<CreateFactoryFn>(
            reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory1")));
    }();
    if (!create_factory) {
        return false;
    }
    static IDXGIFactory1* const factory = []() -> IDXGIFactory1* {
        IDXGIFactory1* created = nullptr;
        return SUCCEEDED(
                   create_factory(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&created)))
                   ? created
                   : nullptr;
    }();
    if (!factory) {
        return false;
    }
    UINT64 best_usage = 0;
    UINT64 best_budget = 0;
    IDXGIAdapter1* adapter = nullptr;
    for (UINT index = 0; factory->EnumAdapters1(index, &adapter) != DXGI_ERROR_NOT_FOUND; ++index) {
        DXGI_ADAPTER_DESC1 desc{};
        const bool hardware =
            SUCCEEDED(adapter->GetDesc1(&desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE);
        IDXGIAdapter3* adapter3 = nullptr;
        if (hardware && SUCCEEDED(adapter->QueryInterface(__uuidof(IDXGIAdapter3),
                                                          reinterpret_cast<void**>(&adapter3)))) {
            DXGI_QUERY_VIDEO_MEMORY_INFO info{};
            if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
                                                         &info)) &&
                info.CurrentUsage > best_usage) {
                best_usage = info.CurrentUsage;
                best_budget = info.Budget;
            }
            adapter3->Release();
        }
        adapter->Release();
    }
    if (best_usage == 0) {
        return false;
    }
    used_gb = static_cast<double>(best_usage) / GiB;
    free_gb = best_budget > best_usage ? static_cast<double>(best_budget - best_usage) / GiB : 0.0;
    return true;
#elif defined(__linux__)
    static const std::filesystem::path gpu_device = []() -> std::filesystem::path {
        std::error_code ec;
        u64 best_total = 0;
        std::filesystem::path best;
        for (const auto& entry : std::filesystem::directory_iterator("/sys/class/drm", ec)) {
            const auto dev = entry.path() / "device";
            std::ifstream total_file(dev / "mem_info_vram_total");
            std::ifstream used_file(dev / "mem_info_vram_used");
            u64 total = 0;
            u64 used = 0;
            if ((total_file >> total) && (used_file >> used) && total > best_total) {
                best_total = total;
                best = dev;
            }
        }
        return best;
    }();
    if (gpu_device.empty()) {
        return false;
    }
    std::ifstream used_file(gpu_device / "mem_info_vram_used");
    std::ifstream total_file(gpu_device / "mem_info_vram_total");
    u64 used = 0;
    u64 total = 0;
    if (!(used_file >> used) || !(total_file >> total) || total == 0) {
        return false;
    }
    used_gb = static_cast<double>(used) / GiB;
    free_gb = total > used ? static_cast<double>(total - used) / GiB : 0.0;
    return true;
#else
    used_gb = -1.0;
    free_gb = -1.0;
    return false;
#endif
}

// The iOS overlay's colours: real and displayed frame rates, a slow speed, shaders building.
const QColor RealColor{56, 189, 248};
const QColor DisplayedColor{74, 222, 128};
const QColor WarningColor{255, 159, 10};
const QColor BuildingColor{255, 214, 10};

// Live upscaler, scene scale, and window scaling filter labels shared by MainWindow::UpdateStatusBar
// and PerformanceOverlay, cached across frames so unchanged state causes zero QString churn.
struct ScalingStatusCache {
    struct Key {
        Settings::RendererBackend backend{};
        Settings::Upscaler upscaler{};
        Settings::ScalingFilter filter{};
        Settings::OotResolution oot_resolution{};
        std::optional<Settings::NvidiaDlssQuality> applied_oot_scale{};
        u64 program_id{};
        bool docked{};
        f32 res_scale{1.0f};
        u32 scene_w{};
        u32 scene_h{};
        u32 game_w{};
        u32 game_h{};
        const char* probed_scene{};
        bool dlss_available{};
        bool dlss_active{};
        bool fsr4_available{};
        bool fsr4_active{};
        std::string mode_name;
        std::string preset_name;
        std::string version;
        bool operator==(const Key&) const = default;
    };

    std::optional<Key> last_key;
    ScalingStatusTelemetry data;
};

ScalingStatusCache& GetScalingStatusCache() {
    static ScalingStatusCache cache;
    return cache;
}

} // namespace

QString FormatFrameGenMethodLabel(bool nvidia_fg_active, bool generating) {
    auto fg_type = Settings::values.frame_gen_type.GetValue();
    if (nvidia_fg_active) {
        fg_type = Settings::FrameGenType::NvidiaDlssG;
    } else if (fg_type == Settings::FrameGenType::NvidiaDlssG && generating) {
        fg_type = Settings::FrameGenType::Fsr3;
    }
    return UiText::ChoiceLabel(Settings::EnumMetadata<Settings::FrameGenType>::Index(),
                               static_cast<u32>(fg_type));
}

void ResetScalingStatusTelemetry() {
    GetScalingStatusCache().last_key.reset();
}

const ScalingStatusTelemetry& QueryScalingStatusTelemetry() {
    auto& cache = GetScalingStatusCache();
    const auto backend = Settings::values.renderer_backend.GetValue();
    const auto upscaler_setting = Settings::values.upscaler.GetValue();
    const auto filter_setting = Settings::values.scaling_filter.GetValue();
    const auto& scaling = Settings::values.resolution_info;
    const f32 res_scale = scaling.up_factor;
    const u64 program_id = Settings::GetCurrentProgramID();
    const bool docked = Settings::IsDockedMode();
    const auto oot_res = Settings::values.oot_resolution.GetValue();
    const auto applied_oot_scale = Settings::AppliedOcarinaRenderScale();
    const u64 scene_extents = VideoCore::Probe::scene_extents.load(std::memory_order_relaxed);
    const u64 frame_extents = VideoCore::Probe::frame_extents.load(std::memory_order_relaxed);
    const char* const probed_scene =
        VideoCore::Probe::scene_upscaler.load(std::memory_order_relaxed);

    ScalingStatusCache::Key key{
        .backend = backend,
        .upscaler = upscaler_setting,
        .filter = filter_setting,
        .oot_resolution = oot_res,
        .applied_oot_scale = applied_oot_scale,
        .program_id = program_id,
        .docked = docked,
        .res_scale = res_scale,
        .scene_w = static_cast<u32>(scene_extents & 0xFFFF),
        .scene_h = static_cast<u32>((scene_extents >> 16) & 0xFFFF),
        .game_w = scaling.ScaleUp(static_cast<u32>(frame_extents & 0xFFFF)),
        .game_h = scaling.ScaleUp(static_cast<u32>((frame_extents >> 16) & 0xFFFF)),
        .probed_scene = probed_scene,
    };
#ifdef EGLANTINE_STREAMLINE
    if (backend == Settings::RendererBackend::Vulkan &&
        upscaler_setting == Settings::Upscaler::Dlss) {
        const auto dlss = Vulkan::Streamline::GetSessionStatus();
        key.dlss_available = dlss.available;
        key.dlss_active = dlss.available && dlss.frames > 0;
        key.mode_name = dlss.mode_name;
        key.preset_name = dlss.preset_name;
        key.version = dlss.version;
    }
#endif
#if defined(_WIN32) || (defined(__linux__) && !defined(__ANDROID__))
    if (backend == Settings::RendererBackend::Vulkan &&
        upscaler_setting == Settings::Upscaler::Fsr4) {
        const auto fsr4 = Vulkan::FfxRuntime::GetSessionStatus();
        key.fsr4_available = fsr4.available;
        key.fsr4_active = fsr4.available && fsr4.frames > 0;
        key.version = fsr4.version;
    }
#endif

    if (cache.last_key == key) {
        return cache.data;
    }
    cache.last_key = key;
    auto& out = cache.data;
    out = {};

    auto effective_filter = filter_setting;
#ifndef LICHII_METALFX
    if (effective_filter == Settings::ScalingFilter::MetalFxSpatial) {
        effective_filter = Settings::ScalingFilter::Fsr;
    }
#endif
    out.filter_label = UiText::ChoiceLabel(
        Settings::EnumMetadata<Settings::ScalingFilter>::Index(),
        static_cast<u32>(effective_filter));
    const QString raw_filter_label = UiText::ChoiceLabel(
        Settings::EnumMetadata<Settings::ScalingFilter>::Index(),
        static_cast<u32>(filter_setting));

    const QString metalfx_name =
        UiText::ChoiceLabel(Settings::EnumMetadata<Settings::FrameGenType>::Index(),
                            static_cast<u32>(Settings::FrameGenType::MetalFx));
    const std::string_view scene_sv = probed_scene != nullptr ? probed_scene : "";
    if (scene_sv == "DLSS") {
        out.scene_upscaler_label = QStringLiteral("DLSS");
    } else if (scene_sv == "FSR 4") {
        out.scene_upscaler_label = QStringLiteral("FSR 4");
    } else if (scene_sv == "FSR 3.1") {
        out.scene_upscaler_label = QStringLiteral("FSR 3.1");
    } else if (scene_sv.starts_with("Metal")) {
        out.scene_upscaler_label = metalfx_name;
    }

    // Status-bar upscaler badge & tooltip.
    if (key.dlss_active) {
        out.status_bar_upscaler_text = QStringLiteral("DLSS");
        if (!key.mode_name.empty()) {
            out.status_bar_upscaler_text +=
                QLatin1Char(' ') + QString::fromStdString(key.mode_name).toUpper();
        }
        if (!key.preset_name.empty()) {
            out.status_bar_upscaler_text +=
                QLatin1Char(' ') + QString::fromStdString(key.preset_name).toUpper();
        }
        out.status_bar_upscaler_tooltip =
            QCoreApplication::translate("MainWindow", "%1 (%2) active — window filter: %3.")
                .arg(UiText::ChoiceLabel("upscaler", "Dlss"),
                     key.version.empty() ? QStringLiteral("Streamline")
                                         : QString::fromStdString(key.version),
                     raw_filter_label);
    } else if (key.fsr4_active) {
        const QString ver = QString::fromStdString(key.version);
        out.status_bar_upscaler_text = ver.startsWith(QStringLiteral("4."))
                                           ? QStringLiteral("FSR %1").arg(ver)
                                           : QStringLiteral("FSR 4");
        out.status_bar_upscaler_tooltip =
            QCoreApplication::translate("MainWindow", "%1 active — window filter: %2.")
                .arg(UiText::ChoiceLabel("upscaler", "Fsr4"), raw_filter_label);
    } else if (backend == Settings::RendererBackend::Vulkan ||
               backend == Settings::RendererBackend::Metal) {
        if (!out.scene_upscaler_label.isEmpty()) {
            out.status_bar_upscaler_text = out.scene_upscaler_label;
        } else if (upscaler_setting == Settings::Upscaler::MetalFx) {
#if defined(__APPLE__)
            out.status_bar_upscaler_text = metalfx_name;
#endif
        } else if (upscaler_setting == Settings::Upscaler::Fsr3) {
            out.status_bar_upscaler_text = QStringLiteral("FSR 3.1");
        }
        if (!out.status_bar_upscaler_text.isEmpty()) {
            out.status_bar_upscaler_tooltip =
                QCoreApplication::translate("MainWindow", "%1 active — window filter: %2.")
                    .arg(out.status_bar_upscaler_text, raw_filter_label);
        }
    }

    // Performance-overlay upscaler summary.
    [[maybe_unused]] const u32 upscaler_enum = Settings::EnumMetadata<Settings::Upscaler>::Index();
    QString upscaler_label;
    switch (upscaler_setting) {
    case Settings::Upscaler::MetalFx:
#if defined(__APPLE__)
        if (backend == Settings::RendererBackend::Metal ||
            backend == Settings::RendererBackend::Vulkan) {
            upscaler_label = !out.scene_upscaler_label.isEmpty() ? out.scene_upscaler_label
                                                                 : metalfx_name;
            break;
        }
#endif
        upscaler_label = UiText::Text("overlay.dlss_off")
                             .arg(UiText::ChoiceLabel(
                                 upscaler_enum, static_cast<u32>(Settings::Upscaler::MetalFx)));
        break;
    case Settings::Upscaler::Game:
        upscaler_label = UiText::Text("overlay.game_dlss");
        break;
    case Settings::Upscaler::Dlss: {
        const QString dlss_name = QStringLiteral("DLSS");
#ifdef EGLANTINE_STREAMLINE
        if (backend == Settings::RendererBackend::Vulkan) {
            if (key.dlss_active) {
                upscaler_label = dlss_name;
                if (!key.mode_name.empty()) {
                    upscaler_label += QLatin1Char(' ') + QString::fromStdString(key.mode_name);
                }
                if (!key.preset_name.empty()) {
                    upscaler_label += QLatin1Char(' ') + QString::fromStdString(key.preset_name);
                }
            } else if (!out.scene_upscaler_label.isEmpty()) {
                upscaler_label = out.scene_upscaler_label;
            } else {
                upscaler_label = key.dlss_available
                                     ? UiText::Text("overlay.dlss_idle").arg(dlss_name)
                                     : UiText::Text("overlay.dlss_off").arg(dlss_name);
            }
            break;
        }
#endif
        upscaler_label = !out.scene_upscaler_label.isEmpty()
                             ? out.scene_upscaler_label
                             : UiText::Text("overlay.dlss_off").arg(dlss_name);
        break;
    }
    case Settings::Upscaler::Fsr4: {
        const QString fsr4_name = QStringLiteral("FSR 4");
#if defined(_WIN32) || (defined(__linux__) && !defined(__ANDROID__))
        if (backend == Settings::RendererBackend::Vulkan) {
            if (key.fsr4_active) {
                upscaler_label = fsr4_name;
            } else if (!out.scene_upscaler_label.isEmpty()) {
                upscaler_label = out.scene_upscaler_label;
            } else {
                upscaler_label = key.fsr4_available
                                     ? UiText::Text("overlay.dlss_idle").arg(fsr4_name)
                                     : UiText::Text("overlay.dlss_off").arg(fsr4_name);
            }
            break;
        }
#endif
        upscaler_label = !out.scene_upscaler_label.isEmpty()
                             ? out.scene_upscaler_label
                             : UiText::Text("overlay.dlss_off").arg(fsr4_name);
        break;
    }
    case Settings::Upscaler::Fsr3:
        upscaler_label = QStringLiteral("FSR 3.1");
        break;
    case Settings::Upscaler::Auto:
        upscaler_label = out.scene_upscaler_label;
        break;
    }

    // Effective resolution / scene scale for status bar & overlay.
    QString oot_overlay_pct;
    if (program_id == Settings::OcarinaOfTimeProgramId) {
        const u32 oot_enum = Settings::EnumMetadata<Settings::OotResolution>::Index();
        auto eff_oot = Settings::OcarinaUpscaledResolution(
            Settings::OcarinaGameResolution(oot_res), upscaler_setting, docked);
#ifdef __APPLE__
        if (eff_oot == Settings::OotResolution::Game &&
            upscaler_setting == Settings::Upscaler::Auto) {
            eff_oot = docked ? Settings::OotResolution::Full : Settings::OotResolution::Percent75;
        }
#endif
        if (docked && eff_oot == Settings::OotResolution::Percent33) {
            eff_oot = Settings::OotResolution::Percent40;
        }
        if (eff_oot == Settings::OotResolution::Full && applied_oot_scale) {
            switch (*applied_oot_scale) {
            case Settings::DlssQuality::Balanced:
                eff_oot = Settings::OotResolution::Percent87;
                break;
            case Settings::DlssQuality::Performance:
                eff_oot = Settings::OotResolution::Percent75;
                break;
            case Settings::DlssQuality::UltraPerformance:
                eff_oot = Settings::OotResolution::Percent50;
                break;
            default:
                break;
            }
        }

        if (res_scale > 1.01f) {
            const QString pct_label = UiText::ChoiceLabel(oot_enum, static_cast<u32>(oot_res));
            out.status_bar_scale_text =
                QCoreApplication::translate("MainWindow", "Scale: %1 (%2x)")
                    .arg(pct_label)
                    .arg(res_scale);
        } else if (eff_oot == Settings::OotResolution::Game) {
            if (key.scene_h > 0) {
                const u32 full_h = docked ? 1440u : 1080u;
                const int pct = static_cast<int>(
                    std::lround(100.0 * static_cast<double>(key.scene_h) /
                                static_cast<double>(full_h)));
                out.status_bar_scale_text =
                    QCoreApplication::translate("MainWindow", "Scale: %1% (Dyn)").arg(pct);
                oot_overlay_pct = QStringLiteral("%1%").arg(pct);
            } else {
                const QString dyn_label =
                    UiText::ChoiceLabel(oot_enum, static_cast<u32>(Settings::OotResolution::Game));
                out.status_bar_scale_text =
                    QCoreApplication::translate("MainWindow", "Scale: %1").arg(dyn_label);
            }
        } else {
            const QString pct_label = UiText::ChoiceLabel(oot_enum, static_cast<u32>(eff_oot));
            out.status_bar_scale_text =
                QCoreApplication::translate("MainWindow", "Scale: %1").arg(pct_label);
            if (eff_oot != Settings::OotResolution::Full) {
                oot_overlay_pct = pct_label;
            }
        }
    } else {
        out.status_bar_scale_text =
            QCoreApplication::translate("MainWindow", "Scale: %1x",
                                        "%1 is the resolution scaling factor")
                .arg(res_scale);
    }

    if (key.scene_w > 0 && key.scene_h > 0 && key.game_w > 0 && key.game_h > 0) {
        out.status_bar_scale_tooltip =
            QCoreApplication::translate("MainWindow", "3D scene: %1×%2 → Output: %3×%4")
                .arg(key.scene_w)
                .arg(key.scene_h)
                .arg(key.game_w)
                .arg(key.game_h);
    } else if (key.game_w > 0 && key.game_h > 0) {
        out.status_bar_scale_tooltip =
            QCoreApplication::translate("MainWindow", "Output: %1×%2")
                .arg(key.game_w)
                .arg(key.game_h);
    }

    QString line = UiText::Text("overlay.scale").arg(res_scale, 0, 'g', 2);
    if (!oot_overlay_pct.isEmpty()) {
        line += QStringLiteral(" (%1)").arg(oot_overlay_pct);
    }
    if (!upscaler_label.isEmpty()) {
        line += QStringLiteral(" · ") + upscaler_label;
    }
    if (!out.filter_label.isEmpty()) {
        line += QStringLiteral(" · ") + out.filter_label;
    }
    out.overlay_scale_line = std::move(line);
    return out;
}

PerformanceOverlay::PerformanceOverlay(MainWindow* parent)
    : QWidget(parent), m_mainWindow{parent}, ui(new Ui::PerformanceOverlay) {
    ui->setupUi(this);

    setAttribute(Qt::WA_TranslucentBackground);
    setWindowFlags(Qt::FramelessWindowHint | Qt::Tool);
    raise();
    // The app's style sheet stays out: the cards keep the iOS overlay's look in every theme.
    setStyleSheet(QStringLiteral("QFrame#overlayCard { background: rgba(0, 0, 0, 140); border: 1px "
                                 "solid rgba(255, 255, 255, 31); border-radius: 10px; }"
                                 "QLabel { color: #FFFFFF; background: transparent; }"));

    QFrame* const frames = addCard();
    m_real = addLine(frames);
    m_generated = addLine(frames);
    m_displayed = addLine(frames);
    m_frametime = addLine(frames);
    m_low10 = addLine(frames);
    QFrame* const info = addCard();
    m_frameGen = addLine(info);
    m_scale = addLine(info);
    m_speed = addLine(info);
    m_shaders = addLine(info);
    QFrame* const detail = addCard();
    m_scene = addLine(detail);
    m_game = addLine(detail);
    m_screen = addLine(detail);
    m_gpuAccuracy = addLine(detail);
    m_ram = addLine(detail);
    m_vram = addLine(detail);
    m_textures = addLine(detail);
    for (QLabel* const line : {m_generated, m_displayed, m_low10, m_shaders, m_scene, m_game,
                               m_screen, m_gpuAccuracy, m_ram, m_vram, m_textures}) {
        line->hide();
    }

    // Real FPS (sky blue) and displayed FPS (green), the iOS overlay's colours.
    m_fpsSeries = new QLineSeries(this);
    QPen realPen(RealColor);
    realPen.setWidth(2);
    m_fpsSeries->setPen(realPen);

    m_totalFpsSeries = new QLineSeries(this);
    QPen totalPen(DisplayedColor);
    totalPen.setWidth(2);
    m_totalFpsSeries->setPen(totalPen);

    m_fpsChart = new QChart;
    m_fpsChart->addSeries(m_totalFpsSeries);
    m_fpsChart->addSeries(m_fpsSeries);
    m_fpsChart->legend()->hide();
    m_fpsChart->setBackgroundBrush(m_background);
    m_fpsChart->setBackgroundRoundness(10);
    m_fpsChart->setBackgroundVisible(true);
    m_fpsChart->layout()->setContentsMargins(2, 2, 2, 2);
    m_fpsChart->setMargins(QMargins{4, 4, 4, 4});

    // axes
    m_fpsX = new QValueAxis(this);
    m_fpsX->setRange(0, NUM_FPS_SAMPLES);
    m_fpsX->setVisible(false);

    m_fpsY = new QValueAxis(this);
    m_fpsY->setRange(0, 120);
    m_fpsY->setLabelFormat(QStringLiteral("%d"));
    m_fpsY->setLabelsColor(QColor(255, 255, 255, 170));

    QFont axisFont = m_fpsY->labelsFont();
    axisFont.setPixelSize(10);
    m_fpsY->setLabelsFont(axisFont);
    m_fpsY->setTickCount(5);

    m_fpsY->setLabelsVisible(true);
    m_fpsY->setGridLineColor(QColor(255, 255, 255, 28));
    m_fpsY->setLinePenColor(QColor(255, 255, 255, 60));

    m_fpsChart->addAxis(m_fpsX, Qt::AlignBottom);
    m_fpsChart->addAxis(m_fpsY, Qt::AlignLeft);
    m_totalFpsSeries->attachAxis(m_fpsX);
    m_totalFpsSeries->attachAxis(m_fpsY);
    m_fpsSeries->attachAxis(m_fpsX);
    m_fpsSeries->attachAxis(m_fpsY);

    // chart view
    m_fpsChartView = new QChartView(m_fpsChart, this);
    m_fpsChartView->setRenderHint(QPainter::Antialiasing);
    m_fpsChartView->setMinimumHeight(90);
    m_fpsChartView->setStyleSheet(QStringLiteral("background: transparent;"));
    m_fpsChartView->setAttribute(Qt::WA_TranslucentBackground);

    ui->verticalLayout->addWidget(m_fpsChartView, 1);

    // pos/stats
    resetPosition(m_mainWindow->pos());
    connect(parent, &MainWindow::positionChanged, this, &PerformanceOverlay::resetPosition);
    connect(m_mainWindow, &MainWindow::statsUpdated, this, &PerformanceOverlay::updateStats);
}

PerformanceOverlay::~PerformanceOverlay() {
    delete ui;
}

QFrame* PerformanceOverlay::addCard() {
    QFrame* const card = new QFrame(this);
    card->setObjectName(QStringLiteral("overlayCard"));
    QVBoxLayout* const layout = new QVBoxLayout(card);
    layout->setContentsMargins(9, 6, 9, 6);
    layout->setSpacing(1);
    ui->verticalLayout->addWidget(card);
    return card;
}

QLabel* PerformanceOverlay::addLine(QFrame* card) {
    QLabel* const line = new QLabel(card);
    QFont font = QFontDatabase::systemFont(QFontDatabase::FixedFont);
    font.setPixelSize(12);
    font.setWeight(QFont::DemiBold);
    line->setFont(font);
    card->layout()->addWidget(line);
    return line;
}

void PerformanceOverlay::setLine(QLabel* line, const QString& text, const QColor& color) {
    const bool visible = !text.isEmpty();
    if (line->isVisible() != visible) {
        line->setVisible(visible);
    }
    if (!visible) {
        return;
    }
    if (line->text() != text) {
        line->setText(text);
    }
    const QString style = color.isValid() ? QStringLiteral("color: %1;").arg(color.name())
                                          : QString();
    if (line->styleSheet() != style) {
        line->setStyleSheet(style);
    }
}

void PerformanceOverlay::resetPosition(const QPoint& _) {
    auto pos = m_mainWindow->pos();
    move(pos.x() + m_offset.x(), pos.y() + m_offset.y());
}

void PerformanceOverlay::updateStats(const Core::PerfStatsResults& results,
                                     const VideoCore::ShaderNotify& shaders) {
    if (!isVisible()) {
        return;
    }

    const double fps = std::isnan(results.average_game_fps) ? 0.0 : results.average_game_fps;
    const double gen_fps = std::isnan(results.generated_fps) ? 0.0 : results.generated_fps;
    const double total_fps =
        !std::isnan(results.total_fps) && results.total_fps > 0.0 ? results.total_fps : fps + gen_fps;
    const bool generating = gen_fps > 0.5 && fps > 0.5;

    // Frames: the same lines as the iOS overlay.
    setLine(m_real, UiText::Text("overlay.real").arg(fps, 0, 'f', 1), RealColor);
    setLine(m_generated, generating ? UiText::Text("overlay.generated").arg(gen_fps, 0, 'f', 1) : QString());
    setLine(m_displayed, generating ? UiText::Text("overlay.displayed").arg(total_fps, 0, 'f', 1) : QString(),
            DisplayedColor);

    static constexpr double FPS_SAMPLE_THRESHOLD = 3.0;
    if (fps > FPS_SAMPLE_THRESHOLD) {
        const qreal x = m_xPos++;
        m_fpsSamples.push_back(fps);
        m_totalFpsSamples.push_back(total_fps);
        m_fpsPoints.push_back(QPointF{x, fps});
        m_totalFpsPoints.push_back(QPointF{x, total_fps});
    }
    while (m_fpsSamples.size() > NUM_FPS_SAMPLES) {
        m_fpsSamples.pop_front();
        m_fpsPoints.pop_front();
    }
    while (m_totalFpsSamples.size() > NUM_FPS_SAMPLES) {
        m_totalFpsSamples.pop_front();
        m_totalFpsPoints.pop_front();
    }

    const double ft = results.frametime;
    const double real_ft_ms =
        (!std::isnan(ft) && ft > 0.0) ? ft * 1000.0 : (fps > 0.1 ? 1000.0 / fps : 0.0);
    setLine(m_frametime, generating && total_fps > 0.5
                             ? UiText::Text("overlay.frame_out")
                                   .arg(real_ft_ms, 0, 'f', 1)
                                   .arg(1000.0 / total_fps, 0, 'f', 1)
                             : UiText::Text("overlay.frame").arg(real_ft_ms, 0, 'f', 1));

    if (!m_fpsSamples.empty()) {
        // Average of the lowest 10% of the real FPS samples
        std::vector<double> sorted(m_fpsSamples.begin(), m_fpsSamples.end());
        std::sort(sorted.begin(), sorted.end());
        const size_t low_count = std::max<size_t>(1, sorted.size() / 10);
        const double low10_fps = std::accumulate(sorted.begin(), sorted.begin() + low_count, 0.0) /
                                 static_cast<double>(low_count);
        setLine(m_low10, UiText::Text("overlay.low10").arg(low10_fps, 0, 'f', 0));

        double peak_chart_fps = *std::max_element(m_fpsSamples.begin(), m_fpsSamples.end());
        if (!m_totalFpsSamples.empty()) {
            peak_chart_fps =
                std::max(peak_chart_fps,
                         *std::max_element(m_totalFpsSamples.begin(), m_totalFpsSamples.end()));
        }
        const double axis_top = std::max(60.0, std::ceil((peak_chart_fps + 5.0) / 30.0) * 30.0);
        m_fpsSeries->replace(QList<QPointF>(m_fpsPoints.begin(), m_fpsPoints.end()));
        m_totalFpsSeries->replace(QList<QPointF>(m_totalFpsPoints.begin(), m_totalFpsPoints.end()));
        m_fpsX->setRange(std::max(0.0, m_xPos - NUM_FPS_SAMPLES), std::max(qreal(10), m_xPos));
        m_fpsY->setRange(0.0, axis_top);
    }

    // What draws the frames.
    const bool nvidia_fg_active = m_mainWindow != nullptr && m_mainWindow->IsNvidiaFrameGenActive();
    QString frame_gen = UiText::Text("value.off");
    if (Settings::values.frame_gen.GetValue() || nvidia_fg_active) {
        const QString method = FormatFrameGenMethodLabel(nvidia_fg_active, generating);
        if (!nvidia_fg_active && LosslessMissing()) {
            frame_gen = UiText::Text("overlay.lossless_missing").arg(method);
        } else if (generating) {
            frame_gen = UiText::Text("overlay.frame_gen_rate").arg(method).arg(total_fps / fps, 0, 'f', 1);
        } else {
            frame_gen = UiText::Text("overlay.warming_up").arg(method);
        }
    }
    setLine(m_frameGen, UiText::Text("overlay.frame_gen").arg(frame_gen));

    const auto& scaling_telemetry = QueryScalingStatusTelemetry();
    setLine(m_scale, scaling_telemetry.overlay_scale_line);

    const double speed_pct =
        std::isnan(results.emulation_speed) ? 100.0 : results.emulation_speed * 100.0;
    setLine(m_speed, UiText::Text("overlay.speed").arg(speed_pct, 0, 'f', 0),
            speed_pct > 0.0 && speed_pct < 90.0 ? WarningColor : QColor{});
    const int building = const_cast<VideoCore::ShaderNotify&>(shaders).ShadersBuilding();
    setLine(m_shaders, building > 0 ? UiText::Text("overlay.shaders").arg(building) : QString(),
            BuildingColor);

    // The 3D scene, the game's finished picture (HUD and menus) and the screen.
    const auto size = [](u32 width, u32 height) {
        return width > 0 && height > 0 ? QStringLiteral("%1×%2").arg(width).arg(height)
                                       : QString();
    };
    const u64 scene = VideoCore::Probe::scene_extents.load(std::memory_order_relaxed);
    const u64 extents = VideoCore::Probe::frame_extents.load(std::memory_order_relaxed);
    const auto& scaling = Settings::values.resolution_info;
    const QString scene_size =
        size(static_cast<u32>(scene & 0xFFFF), static_cast<u32>((scene >> 16) & 0xFFFF));
    const QString game_size = size(scaling.ScaleUp(static_cast<u32>(extents & 0xFFFF)),
                                   scaling.ScaleUp(static_cast<u32>((extents >> 16) & 0xFFFF)));
    const QString screen_size = size(static_cast<u32>((extents >> 32) & 0xFFFF),
                                     static_cast<u32>((extents >> 48) & 0xFFFF));
    setLine(m_scene, scene_size.isEmpty()
                         ? QString()
                         : (scaling_telemetry.scene_upscaler_label.isEmpty()
                                ? UiText::Text("overlay.scene").arg(scene_size)
                                : UiText::Text("overlay.scene")
                                          .arg(scene_size) +
                                      QStringLiteral(" · ") +
                                      scaling_telemetry.scene_upscaler_label));
    setLine(m_game, game_size.isEmpty() ? QString() : UiText::Text("overlay.game").arg(game_size));
    setLine(m_screen, game_size.isEmpty() || screen_size.isEmpty()
                          ? QString()
                          : UiText::Text("overlay.screen").arg(screen_size) +
                                QStringLiteral(" · ") + scaling_telemetry.filter_label);
    setLine(m_gpuAccuracy,
            UiText::Text("overlay.gpu_accuracy")
                .arg(UiText::ChoiceLabel(Settings::EnumMetadata<Settings::GpuAccuracy>::Index(),
                                         static_cast<u32>(Settings::values.current_gpu_accuracy))));

    double ram_used = -1.0;
    double ram_free = -1.0;
    if (QueryProcessRam(ram_used, ram_free) && ram_used >= 0.0) {
        setLine(m_ram,
                ram_free >= 0.0 ? UiText::Text("overlay.ram")
                                      .arg(ram_used, 0, 'f', 1)
                                      .arg(ram_used + ram_free, 0, 'f', 1)
                                : UiText::Text("overlay.ram_used").arg(ram_used, 0, 'f', 1),
                ram_free >= 0.0 && ram_free < 2.0 ? WarningColor : QColor{});
    } else {
        setLine(m_ram, QString());
    }
    double vram_used = -1.0;
    double vram_free = -1.0;
    if (QueryProcessVram(vram_used, vram_free) && vram_used >= 0.0) {
        setLine(m_vram,
                vram_free >= 0.0 ? UiText::Text("overlay.vram")
                                       .arg(vram_used, 0, 'f', 1)
                                       .arg(vram_used + vram_free, 0, 'f', 1)
                                 : UiText::Text("overlay.vram_used").arg(vram_used, 0, 'f', 1),
                vram_free >= 0.0 && vram_free < 1.0 ? WarningColor : QColor{});
    } else {
        setLine(m_vram, QString());
    }
    const u64 texture_bytes = VideoCore::Probe::texture_cache_bytes.load(std::memory_order_relaxed);
    const u64 texture_expected =
        VideoCore::Probe::texture_cache_expected.load(std::memory_order_relaxed);
    if (texture_bytes == 0) {
        setLine(m_textures, QString());
    } else if (texture_expected > 0) {
        setLine(m_textures,
                UiText::Text("overlay.textures").arg(texture_bytes >> 20).arg(texture_expected >> 20));
    } else {
        setLine(m_textures, UiText::Text("overlay.textures_used").arg(texture_bytes >> 20));
    }
    adjustSize();
}

void PerformanceOverlay::paintEvent(QPaintEvent* event) {
    // The cards draw themselves; the window stays clear between them.
    QWidget::paintEvent(event);
}

void PerformanceOverlay::mousePressEvent(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton) {
        return;
    }
    if (QGuiApplication::platformName().startsWith(QStringLiteral("wayland")) && windowHandle() &&
        windowHandle()->startSystemMove()) {
        return;
    }
    m_drag_start_pos = event->pos();
}

void PerformanceOverlay::mouseMoveEvent(QMouseEvent* event) {
    // drag
    if (event->buttons() & Qt::LeftButton) {
        QPoint new_global_pos = event->globalPosition().toPoint() - m_drag_start_pos;
        m_offset = new_global_pos - m_mainWindow->pos();
        move(new_global_pos);
    }
}

void PerformanceOverlay::closeEvent(QCloseEvent* event) {
    emit closed();
}
