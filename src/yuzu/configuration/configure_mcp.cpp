// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <QCheckBox>
#include <QClipboard>
#include <QFormLayout>
#include <QGroupBox>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>

#include "common/settings.h"
#include "yuzu/configuration/configure_mcp.h"

ConfigureMcp::ConfigureMcp(QWidget* parent) : QWidget(parent) {
    auto* layout = new QVBoxLayout(this);
    auto* group = new QGroupBox(tr("MCP & AI Control"), this);
    auto* group_layout = new QVBoxLayout(group);

    auto* intro = new QLabel(
        tr("Expose a local Model Context Protocol (MCP) server and NX2 GPU Debug Server so AI "
           "agents and scripts can read live logs, inspect performance/frames, and control the "
           "emulator. Bound strictly to 127.0.0.1 (loopback only)."),
        group);
    intro->setWordWrap(true);
    group_layout->addWidget(intro);

    enable = new QCheckBox(tr("Enable MCP Server"), group);
    group_layout->addWidget(enable);

    details = new QWidget(group);
    auto* form = new QFormLayout(details);
    form->setContentsMargins(0, 0, 0, 0);
    writes = new QCheckBox(tr("Allow Writes (input, memory writes, settings)"), details);
    form->addRow(writes);
    port = new QSpinBox(details);
    port->setRange(1, 65535);
    form->addRow(tr("Port:"), port);
    token = new QLineEdit(details);
    token->setEchoMode(QLineEdit::Password);
    token->setPlaceholderText(tr("Optional: clients must send it as a bearer token"));
    form->addRow(tr("Token:"), token);

    endpoints_label = new QLabel(details);
    endpoints_label->setTextInteractionFlags(Qt::TextSelectableByMouse);
    endpoints_label->setWordWrap(true);
    form->addRow(tr("Endpoints:"), endpoints_label);

    auto* button_row = new QWidget(details);
    auto* buttons = new QHBoxLayout(button_row);
    buttons->setContentsMargins(0, 4, 0, 0);
    auto* copy_url = new QPushButton(tr("Copy MCP URL"), button_row);
    auto* copy_cmd = new QPushButton(tr("Copy Claude / Agent Command"), button_row);
    buttons->addWidget(copy_url);
    buttons->addWidget(copy_cmd);
    buttons->addStretch();
    form->addRow(button_row);

    connect(copy_url, &QPushButton::clicked, this, [this] {
        QGuiApplication::clipboard()->setText(
            QStringLiteral("http://127.0.0.1:%1/mcp").arg(port->value()));
    });
    connect(copy_cmd, &QPushButton::clicked, this, [this] {
        QGuiApplication::clipboard()->setText(
            QStringLiteral("claude mcp add --transport http lichii http://127.0.0.1:%1/mcp")
                .arg(port->value()));
    });
    group_layout->addWidget(details);

    auto* note = new QLabel(
        tr("You can also toggle MCP live anytime from the bottom status bar (MCP button) or "
           "Tools > MCP Server. Without Allow Writes, agents can read logs, inspect state, and "
           "boot/pause/stop games, but cannot inject input or modify memory/settings."),
        group);
    note->setWordWrap(true);
    group_layout->addWidget(note);

    layout->addWidget(group);
    layout->addStretch();

    connect(enable, &QCheckBox::toggled, this, [this] { UpdateEnabledState(); });
    connect(port, &QSpinBox::valueChanged, this, [this] { UpdateEnabledState(); });
    SetConfiguration();
}

ConfigureMcp::~ConfigureMcp() = default;

void ConfigureMcp::SetConfiguration() {
    enable->setChecked(Settings::values.enable_mcp_server.GetValue());
    port->setValue(Settings::values.mcp_port.GetValue());
    writes->setChecked(Settings::values.mcp_allow_writes.GetValue());
    token->setText(QString::fromStdString(Settings::values.mcp_token.GetValue()));
    UpdateEnabledState();
}

void ConfigureMcp::UpdateEnabledState() {
    details->setEnabled(enable->isChecked());
    if (endpoints_label != nullptr) {
        const int mcp_p = port->value();
        const int dbg_p = mcp_p < 65535 ? (mcp_p + 1) : 6545;
        endpoints_label->setText(
            tr("MCP HTTP: http://127.0.0.1:%1/mcp  ·  GPU Debug Server: 127.0.0.1:%2")
                .arg(mcp_p)
                .arg(dbg_p));
    }
}

void ConfigureMcp::ApplyConfiguration() {
    Settings::values.enable_mcp_server = enable->isChecked();
    Settings::values.mcp_port = static_cast<u16>(port->value());
    Settings::values.mcp_allow_writes = writes->isChecked();
    Settings::values.mcp_token = token->text().toStdString();
}
