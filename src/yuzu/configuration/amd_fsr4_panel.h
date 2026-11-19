// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Shown under the Upscaler setting on Windows and Linux: whether AMD's FidelityFX runtime (Windows)
// or the fsr4-vk community Vulkan port (Linux) is present, what FSR 4 did in the running game, and
// a button on Windows that downloads AMD's signed runtime from GitHub.

#pragma once

#if defined(_WIN32) || defined(__linux__)

#include <QString>
#include <QWidget>

class QLabel;
class QPushButton;
class QTimer;

class AmdFsr4Panel : public QWidget {
public:
    explicit AmdFsr4Panel(QWidget* parent);

    void Refresh();

    void SetDevice(const QString& name, bool capable = true);
    void SetUpscalerSelected(bool selected);

protected:
    void showEvent(QShowEvent* event) override;
    void hideEvent(QHideEvent* event) override;

private:
    void Download();

    QLabel* icon_label;
    QLabel* status_label;
    QLabel* session_label;
    QPushButton* download_button;
    QPushButton* folder_button;
    QTimer* refresh_timer;
    QString device_name;
    bool device_capable{true};
    bool upscaler_selected{true};
};

#endif
