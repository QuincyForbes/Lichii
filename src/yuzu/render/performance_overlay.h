// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <deque>
#include <QWidget>

namespace VideoCore {
class ShaderNotify;
}
namespace Core {
struct PerfStatsResults;
}
namespace Ui {
class PerformanceOverlay;
}

class QFrame;
class QLabel;
class QLineSeries;
class QChart;
class QChartView;
class QValueAxis;
class MainWindow;

class PerformanceOverlay : public QWidget {
    Q_OBJECT

public:
    explicit PerformanceOverlay(MainWindow* parent = nullptr);
    ~PerformanceOverlay();

protected:
    void paintEvent(QPaintEvent* event) override;

    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void closeEvent(QCloseEvent* event) override;

private:
    void resetPosition(const QPoint& pos);
    void updateStats(const Core::PerfStatsResults& results, const VideoCore::ShaderNotify& shaders);
    /// A small translucent card of lines, as the iOS overlay's.
    QFrame* addCard();
    QLabel* addLine(QFrame* card);
    static void setLine(QLabel* line, const QString& text, const QColor& color = {});

    MainWindow* m_mainWindow = nullptr;
    Ui::PerformanceOverlay* ui;

    // The iOS overlay's Detailed preset, in its order: frames, then what draws them, then sizes and
    // memory. VRAM and the chart are the desktop's own.
    QLabel* m_real = nullptr;
    QLabel* m_generated = nullptr;
    QLabel* m_displayed = nullptr;
    QLabel* m_frametime = nullptr;
    QLabel* m_low10 = nullptr;
    QLabel* m_frameGen = nullptr;
    QLabel* m_scale = nullptr;
    QLabel* m_speed = nullptr;
    QLabel* m_shaders = nullptr;
    QLabel* m_scene = nullptr;
    QLabel* m_game = nullptr;
    QLabel* m_screen = nullptr;
    QLabel* m_gpuAccuracy = nullptr;
    QLabel* m_ram = nullptr;
    QLabel* m_vram = nullptr;
    QLabel* m_textures = nullptr;

    // colors
    QColor m_background{0, 0, 0, 140};
    QColor m_border{255, 255, 255, 31};

    QPoint m_offset{25, 75};

    // frametime
    const size_t NUM_FRAMETIME_SAMPLES = 300;
    std::deque<double> m_frametimeSamples;

    // fps
    const size_t NUM_FPS_SAMPLES = 120;
    qreal m_xPos = 0;
    std::deque<double> m_fpsSamples;
    std::deque<double> m_totalFpsSamples;
    std::deque<QPointF> m_fpsPoints;
    std::deque<QPointF> m_totalFpsPoints;

    // drag
    QPoint m_drag_start_pos;

    // fps chart
    QLineSeries* m_fpsSeries = nullptr;
    QLineSeries* m_totalFpsSeries = nullptr;
    QChart* m_fpsChart = nullptr;
    QChartView* m_fpsChartView = nullptr;
    QValueAxis* m_fpsX = nullptr;
    QValueAxis* m_fpsY = nullptr;

signals:
    void closed();
};
