// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <QWidget>

class QCheckBox;
class QLabel;
class QLineEdit;
class QSpinBox;
class QWidget;

/// Top-level MCP Server settings page: built-in MCP server on/off, port, Allow Writes, token,
/// and quick copy buttons. The main window applies changes to a running server when the dialog closes.
class ConfigureMcp : public QWidget {
    Q_OBJECT

public:
    explicit ConfigureMcp(QWidget* parent = nullptr);
    ~ConfigureMcp() override;

    /// Writes the fields to the settings.
    void ApplyConfiguration();
    /// Fills the fields from the settings.
    void SetConfiguration();

private:
    /// Enables the dependent fields only while the server is enabled.
    void UpdateEnabledState();

    QCheckBox* enable = nullptr;
    QSpinBox* port = nullptr;
    QCheckBox* writes = nullptr;
    QLineEdit* token = nullptr;
    QLabel* endpoints_label = nullptr;
    QWidget* details = nullptr;
};
