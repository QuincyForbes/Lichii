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

#include "main_window.h"

#include <QChart>
#include <QChartView>
#include <QGraphicsLayout>
#include <QLineSeries>
#include <QMouseEvent>
#include <QPainter>
#include <QValueAxis>

#ifdef HAS_LSFG
#include <filesystem>
#include "video_core/frame_gen/lossless_dll.h"
#endif

namespace {

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
    const double fps = results.average_game_fps;
    const double gen_fps = std::isnan(results.generated_fps) ? 0.0 : results.generated_fps;
    const double total_fps =
        !std::isnan(results.total_fps) && results.total_fps > 0.0 ? results.total_fps : fps + gen_fps;

    if (!std::isnan(fps)) {
        static constexpr double FPS_SAMPLE_THRESHOLD = 3.0;

        ui->fps->setText(tr("%1 FPS").arg(fps, 0, 'f', 1));

        if (gen_fps > 0.5 && fps > 0.5) {
            const double mult = total_fps / fps;
            ui->fps_gen->setText(tr("+%1 FPS (%2x)").arg(gen_fps, 0, 'f', 1).arg(mult, 0, 'f', 1));
            ui->total_fps->setText(tr("%1 FPS (%2x FG)").arg(total_fps, 0, 'f', 1).arg(mult, 0, 'f', 1));
        } else if (Settings::values.frame_gen.GetValue()) {
            ui->fps_gen->setText(LosslessMissing() ? tr("Unavailable (no Lossless.dll, see Graphics)")
                                                   : tr("+0.0 FPS (Warmup)"));
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

    const double speed_pct =
        std::isnan(results.emulation_speed) ? 100.0 : results.emulation_speed * 100.0;
    const f32 res_scale = Settings::values.resolution_info.up_factor;
    const int building = const_cast<VideoCore::ShaderNotify&>(shaders).ShadersBuilding();
    const QString dlss_tag =
        Settings::values.nvidia_dlss.GetValue() ? QStringLiteral(" | DLSS") : QString{};
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
    if (event->button() == Qt::LeftButton) {
        m_drag_start_pos = event->pos();
    }
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
