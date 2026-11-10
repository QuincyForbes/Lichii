// SPDX-FileCopyrightText: 2016 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <vector>
#include <QString>
#include <QWidget>
#include <qobjectdefs.h>

class QObject;

namespace ConfigurationShared {

class Tab : public QWidget {
    Q_OBJECT

public:
    explicit Tab(std::shared_ptr<std::vector<Tab*>> group, QWidget* parent = nullptr);
    ~Tab();

    virtual void ApplyConfiguration() = 0;
    virtual void SetConfiguration() = 0;
};

/// Greys out a setting this platform, GPU or build can't use and puts the reason first in its
/// tooltip; an empty reason restores the state and tooltip it had before the first call. The value
/// itself is left alone.
void SetUnavailable(QWidget* widget, const QString& reason);

} // namespace ConfigurationShared
