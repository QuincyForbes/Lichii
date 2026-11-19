// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Shown under the NVIDIA DLSS setting: the GPU, which of NVIDIA's files were found, what DLSS
// does in the running game, and a button that fetches missing files from NVIDIA's own Streamline
// release.

#pragma once

#ifdef EGLANTINE_STREAMLINE

#include <QString>
#include <QWidget>

class QHideEvent;
class QLabel;
class QPushButton;
class QShowEvent;
class QTimer;

class NvidiaDlssPanel : public QWidget {
public:
    explicit NvidiaDlssPanel(QWidget* parent);

    void Refresh();

    /// The GPU the renderer uses (the Graphics tab's device).
    void SetDevice(const QString& name, bool capable = true);

    /// Whether DLSS Frame Generation is the chosen frame generation type: the panel then lists
    /// its files and downloads them with the rest.
    void SetFrameGenWanted(bool wanted);

    /// Whether NVIDIA DLSS is currently chosen in the Upscaler combobox.
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
    bool frame_gen_wanted{};
    bool upscaler_selected{true};
};

#endif
