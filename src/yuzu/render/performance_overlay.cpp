// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "common/settings.h"
#include "core/perf_stats.h"
#include "performance_overlay.h"
#include "ui_performance_overlay.h"
#include "video_core/shader_notify.h"
#ifdef EGLANTINE_STREAMLINE
#include "video_core/vulkan_common/streamline.h"
#endif

#include "main_window.h"

#include <QChart>
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

QString FrameGenMethodLabel() {
    switch (Settings::values.frame_gen_type.GetValue()) {
    case Settings::FrameGenType::Lsfg:
        return QStringLiteral("LSFG");
    case Settings::FrameGenType::MetalFx:
        return QStringLiteral("MetalFX");
    case Settings::FrameGenType::Fsr3:
    default:
        return QStringLiteral("FSR3");
    }
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

} // namespace

PerformanceOverlay::PerformanceOverlay(MainWindow* parent)
    : QWidget(parent), m_mainWindow{parent}, ui(new Ui::PerformanceOverlay) {
    ui->setupUi(this);

    setAttribute(Qt::WA_TranslucentBackground);
    setWindowFlags(Qt::FramelessWindowHint | Qt::Tool);
    raise();

    // Actual Game FPS series (sky blue)
    m_fpsSeries = new QLineSeries(this);
    QPen realPen(QColor(56, 189, 248));
    realPen.setWidth(2);
    m_fpsSeries->setPen(realPen);

    // Total Displayed FPS series (emerald green)
    m_totalFpsSeries = new QLineSeries(this);
    QPen totalPen(QColor(74, 222, 128));
    totalPen.setWidth(2);
    m_totalFpsSeries->setPen(totalPen);

    m_fpsChart = new QChart;
    m_fpsChart->addSeries(m_totalFpsSeries);
    m_fpsChart->addSeries(m_fpsSeries);
    m_fpsChart->legend()->hide();
    m_fpsChart->setBackgroundBrush(QColor(9, 14, 26, 230));
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
    m_fpsY->setLabelsColor(QColor(203, 213, 225));

    QFont axisFont = m_fpsY->labelsFont();
    axisFont.setPixelSize(10);
    m_fpsY->setLabelsFont(axisFont);
    m_fpsY->setTickCount(5);

    m_fpsY->setLabelsVisible(true);
    m_fpsY->setGridLineColor(QColor(45, 55, 72));
    m_fpsY->setLinePenColor(QColor(100, 116, 139));

    m_fpsChart->addAxis(m_fpsX, Qt::AlignBottom);
    m_fpsChart->addAxis(m_fpsY, Qt::AlignLeft);
    m_totalFpsSeries->attachAxis(m_fpsX);
    m_totalFpsSeries->attachAxis(m_fpsY);
    m_fpsSeries->attachAxis(m_fpsX);
    m_fpsSeries->attachAxis(m_fpsY);

    // chart view
    m_fpsChartView = new QChartView(m_fpsChart, this);
    m_fpsChartView->setRenderHint(QPainter::Antialiasing);
    m_fpsChartView->setMinimumHeight(105);

    ui->verticalLayout->addWidget(m_fpsChartView, 1);

    QFont font = ui->fps->font();
    font.setWeight(QFont::DemiBold);

    ui->total_fps->setFont(font);
    ui->fps->setFont(font);
    ui->fps_gen->setFont(font);
    ui->frametime->setFont(font);
    ui->ram_widget->setVisible(false);
    ui->vram_widget->setVisible(false);

    // pos/stats
    resetPosition(m_mainWindow->pos());
    connect(parent, &MainWindow::positionChanged, this, &PerformanceOverlay::resetPosition);
    connect(m_mainWindow, &MainWindow::statsUpdated, this, &PerformanceOverlay::updateStats);
}

PerformanceOverlay::~PerformanceOverlay() {
    delete ui;
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

    const double fps = results.average_game_fps;
    const double gen_fps = std::isnan(results.generated_fps) ? 0.0 : results.generated_fps;
    const double total_fps =
        !std::isnan(results.total_fps) && results.total_fps > 0.0 ? results.total_fps : fps + gen_fps;

    if (!std::isnan(fps)) {
        static constexpr double FPS_SAMPLE_THRESHOLD = 3.0;

        ui->fps->setText(tr("%1 FPS").arg(fps, 0, 'f', 1));

        if (gen_fps > 0.5 && fps > 0.5) {
            const double mult = total_fps / fps;
            const QString method = FrameGenMethodLabel();
            ui->fps_gen->setText(
                tr("+%1 FPS (%2x %3)").arg(gen_fps, 0, 'f', 1).arg(mult, 0, 'f', 1).arg(method));
            ui->total_fps->setText(
                tr("%1 FPS (%2x %3)").arg(total_fps, 0, 'f', 1).arg(mult, 0, 'f', 1).arg(method));
        } else if (Settings::values.frame_gen.GetValue()) {
            ui->fps_gen->setText(LosslessMissing()
                                     ? tr("Unavailable (no Lossless.dll, see Graphics)")
                                     : tr("+0.0 FPS (%1 Warmup)").arg(FrameGenMethodLabel()));
            ui->total_fps->setText(tr("%1 FPS").arg(fps, 0, 'f', 1));
        } else {
            ui->fps_gen->setText(tr("Off (0 FPS)"));
            ui->total_fps->setText(tr("%1 FPS (Real only)").arg(fps, 0, 'f', 1));
        }

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

        if (m_fpsSamples.size() >= 2) {
            const size_t back_search = std::min(size_t(10), m_fpsSamples.size());
            const double sum =
                std::accumulate(m_fpsSamples.end() - back_search, m_fpsSamples.end(), 0.0);
            const double avg = sum / static_cast<double>(back_search);
            ui->fps_avg->setText(tr("Avg: %1").arg(avg, 0, 'f', 0));
        }

        if (!m_fpsPoints.empty()) {
            auto [min_it, max_it] = std::minmax_element(m_fpsSamples.begin(), m_fpsSamples.end());
            const double min_fps = *min_it;
            const double max_fps = *max_it;

            // Average of the lowest 10% of the real FPS samples
            std::vector<double> sorted(m_fpsSamples.begin(), m_fpsSamples.end());
            std::sort(sorted.begin(), sorted.end());
            const size_t low_count = std::max<size_t>(1, sorted.size() / 10);
            const double low10_fps =
                std::accumulate(sorted.begin(), sorted.begin() + low_count, 0.0) /
                static_cast<double>(low_count);

            ui->fps_min->setText(tr("Min: %1").arg(min_fps, 0, 'f', 0));
            ui->fps_low1->setText(tr("10% low: %1").arg(low10_fps, 0, 'f', 0));
            ui->fps_max->setText(tr("Max: %1").arg(max_fps, 0, 'f', 0));

            double peak_chart_fps = max_fps;
            if (!m_totalFpsSamples.empty()) {
                peak_chart_fps = std::max(
                    peak_chart_fps,
                    *std::max_element(m_totalFpsSamples.begin(), m_totalFpsSamples.end()));
            }
            const double axis_top = std::max(60.0, std::ceil((peak_chart_fps + 5.0) / 30.0) * 30.0);

            m_fpsSeries->replace(QList<QPointF>(m_fpsPoints.begin(), m_fpsPoints.end()));
            m_totalFpsSeries->replace(
                QList<QPointF>(m_totalFpsPoints.begin(), m_totalFpsPoints.end()));

            const qreal x_min = std::max(0.0, m_xPos - NUM_FPS_SAMPLES);
            const qreal x_max = std::max(qreal(10), m_xPos);
            m_fpsX->setRange(x_min, x_max);
            m_fpsY->setRange(0.0, axis_top);
        }
    }

    const double ft = results.frametime;
    const double real_ft_ms =
        (!std::isnan(ft) && ft > 0.0) ? ft * 1000.0 : (fps > 0.1 ? 1000.0 / fps : 0.0);
    if (real_ft_ms > 0.0) {
        static constexpr double FT_SAMPLE_THRESHOLD = 500.0;

        if (gen_fps > 0.5 && total_fps > 0.5) {
            const double out_ft_ms = 1000.0 / total_fps;
            ui->frametime->setText(
                tr("%1 ms (%2 ms out)").arg(real_ft_ms, 0, 'f', 2).arg(out_ft_ms, 0, 'f', 2));
        } else {
            ui->frametime->setText(tr("%1 ms").arg(real_ft_ms, 0, 'f', 2));
        }

        if (real_ft_ms <= FT_SAMPLE_THRESHOLD) {
            m_frametimeSamples.push_back(real_ft_ms);
        }

        if (m_frametimeSamples.size() > NUM_FRAMETIME_SAMPLES) {
            m_frametimeSamples.pop_front();
        }

        if (!m_frametimeSamples.empty()) {
            auto [min_it, max_it] =
                std::minmax_element(m_frametimeSamples.begin(), m_frametimeSamples.end());
            ui->ft_min->setText(tr("Min: %1").arg(*min_it, 0, 'f', 1));
            ui->ft_max->setText(tr("Max: %1").arg(*max_it, 0, 'f', 1));
        }

        if (m_frametimeSamples.size() >= 2) {
            const size_t back_search = std::min(size_t(10), m_frametimeSamples.size());
            const double sum = std::accumulate(m_frametimeSamples.end() - back_search,
                                               m_frametimeSamples.end(), 0.0);
            const double avg = sum / static_cast<double>(back_search);
            ui->ft_avg->setText(tr("Avg: %1").arg(avg, 0, 'f', 1));
        }
    }

    double ram_used = -1.0;
    double ram_free = -1.0;
    if (QueryProcessRam(ram_used, ram_free) && ram_used >= 0.0) {
        ui->ram_widget->setVisible(true);
        ui->ram_usage->setText(ram_free >= 0.0
                                   ? tr("%1 GB (%2 GB free)")
                                         .arg(ram_used, 0, 'f', 1)
                                         .arg(ram_free, 0, 'f', 1)
                                   : tr("%1 GB").arg(ram_used, 0, 'f', 1));
        ui->ram_usage->setStyleSheet(ram_free >= 0.0 && ram_free < 2.0
                                         ? QStringLiteral("color: #fbbf24;")
                                         : QStringLiteral("color: #e2e8f0;"));
    } else {
        ui->ram_widget->setVisible(false);
    }

    double vram_used = -1.0;
    double vram_free = -1.0;
    if (QueryProcessVram(vram_used, vram_free) && vram_used >= 0.0) {
        ui->vram_widget->setVisible(true);
        ui->vram_usage->setText(vram_free >= 0.0
                                    ? tr("%1 GB (%2 GB free)")
                                          .arg(vram_used, 0, 'f', 1)
                                          .arg(vram_free, 0, 'f', 1)
                                    : tr("%1 GB").arg(vram_used, 0, 'f', 1));
        ui->vram_usage->setStyleSheet(vram_free >= 0.0 && vram_free < 1.0
                                          ? QStringLiteral("color: #fbbf24;")
                                          : QStringLiteral("color: #e2e8f0;"));
    } else {
        ui->vram_widget->setVisible(false);
    }

    const double speed_pct =
        std::isnan(results.emulation_speed) ? 100.0 : results.emulation_speed * 100.0;
    const f32 res_scale = Settings::values.resolution_info.up_factor;
    const int building = const_cast<VideoCore::ShaderNotify&>(shaders).ShadersBuilding();
    // What NVIDIA DLSS does in this game, not just the setting.
    QString dlss_tag;
#ifdef EGLANTINE_STREAMLINE
    if (Settings::values.nvidia_dlss.GetValue()) {
        const auto dlss = Vulkan::Streamline::GetSessionStatus();
        dlss_tag = dlss.available && dlss.frames > 0 ? QStringLiteral(" | DLSS")
                   : dlss.available                  ? tr(" | DLSS idle")
                                                     : tr(" | DLSS off");
    }
#endif
    if (building > 0) {
        ui->status_info->setText(tr("Speed: %1% | Scale: %2x%3 | Shaders: %4")
                                     .arg(speed_pct, 0, 'f', 0)
                                     .arg(res_scale)
                                     .arg(dlss_tag)
                                     .arg(building));
    } else {
        ui->status_info->setText(tr("Speed: %1% | Scale: %2x%3")
                                     .arg(speed_pct, 0, 'f', 0)
                                     .arg(res_scale)
                                     .arg(dlss_tag));
    }
}

void PerformanceOverlay::paintEvent(QPaintEvent* event) {
    QPainter painter(this);

    painter.setRenderHint(QPainter::Antialiasing);

    painter.setBrush(m_background);
    QPen borderPen(m_border);
    borderPen.setWidth(1);
    painter.setPen(borderPen);

    painter.drawRoundedRect(rect().adjusted(1, 1, -1, -1), 10.0, 10.0);
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
