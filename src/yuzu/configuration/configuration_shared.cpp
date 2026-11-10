// SPDX-FileCopyrightText: 2016 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <memory>
#include <type_traits>
#include <vector>
#include <QVariant>
#include "yuzu/configuration/configuration_shared.h"

namespace ConfigurationShared {

Tab::Tab(std::shared_ptr<std::vector<Tab*>> group, QWidget* parent) : QWidget(parent) {
    if (group != nullptr) {
        group->push_back(this);
    }
}

Tab::~Tab() = default;

void SetUnavailable(QWidget* widget, const QString& reason) {
    if (widget == nullptr) {
        return;
    }
    constexpr char enabled_key[] = "lichii_base_enabled";
    constexpr char tooltip_key[] = "lichii_base_tooltip";
    if (!widget->property(enabled_key).isValid()) {
        widget->setProperty(enabled_key, widget->isEnabled());
        widget->setProperty(tooltip_key, widget->toolTip());
    }
    const QString tooltip = widget->property(tooltip_key).toString();
    widget->setEnabled(reason.isEmpty() && widget->property(enabled_key).toBool());
    widget->setToolTip(reason.isEmpty()     ? tooltip
                       : tooltip.isEmpty() ? reason
                                           : reason + QStringLiteral("\n\n") + tooltip);
}

} // namespace ConfigurationShared
