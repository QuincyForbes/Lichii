// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Shown under the NVIDIA DLSS setting: the GPU, which of NVIDIA's files were found, what DLSS
// does in the running game, and a button that fetches missing files from NVIDIA's own Streamline
// release.

#pragma once

#ifdef EGLANTINE_STREAMLINE

#include <QString>
#include <QWidget>

class QLabel;
class QPushButton;
class QTimer;

class NvidiaDlssPanel : public QWidget {
public:
    explicit NvidiaDlssPanel(QWidget* parent);

    void Refresh();

    /// The GPU the renderer uses (the Graphics tab's device).
    void SetDevice(const QString& name);

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
};

#endif
