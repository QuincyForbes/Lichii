// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <QColor>
#include <QIcon>
#include <QString>

class QApplication;
class QFrame;
class QLabel;
class QWidget;

/// The Lichii theme: the iOS app's colours, type and grouped lists on the desktop
/// (docs/design/lichii-design.md). The widgets made here also read well in the other themes.
namespace LichiiStyle {

struct Colors {
    QColor window;     ///< grouped background (iOS systemGroupedBackground)
    QColor card;       ///< a group's rows (secondarySystemGroupedBackground)
    QColor fill;       ///< text fields, pickers, buttons (tertiarySystemFill)
    QColor fill_hover;
    QColor fill_pressed;
    QColor text;       ///< label
    QColor secondary;  ///< secondaryLabel: section headers, footers, captions
    QColor tertiary;   ///< tertiaryLabel: disabled text
    QColor separator;
    QColor accent;     ///< Theme.accent of the iOS app
    QColor accent_hover;
    QColor accent_pressed;
    QColor accent_soft; ///< the accent behind a selected or found row
    QColor destructive;
    QColor warning;
    QColor success;
    QColor scroll;
};

/// The iOS app's accent (Theme.accent: red 0.55, green 0.36, blue 0.96).
inline constexpr QRgb AccentRgb = 0xFF8C5CF5;

[[nodiscard]] bool Active();
[[nodiscard]] bool Dark();
[[nodiscard]] Colors ColorsFor(bool dark);
[[nodiscard]] Colors Current();

/// Fusion with the Lichii palette and style sheet, light or dark.
void Apply(QApplication& app, bool dark);
/// The style and palette the app started with, for the other themes.
void Restore(QApplication& app);

/// A Lucide icon (the shapes of the SF Symbols the iOS app uses) in `color`, the text colour when
/// none is given. Selected rows draw it white, disabled ones in the tertiary colour.
[[nodiscard]] QIcon Icon(const QString& name, const QColor& color = {});

/// iOS grouped-list parts: the header above a group, the rounded group, the footer under it.
[[nodiscard]] QLabel* SectionHeader(const QString& text, QWidget* parent);
[[nodiscard]] QLabel* Footer(const QString& text, QWidget* parent);
[[nodiscard]] QFrame* Card(QWidget* parent);
/// Appends a row to a Card, with a separator between rows.
void AddRow(QFrame* card, QWidget* row);
[[nodiscard]] QLabel* PageTitle(const QString& text, QWidget* parent);

/// The name of this desktop platform as the settings search says it ("Windows").
[[nodiscard]] QString PlatformName();

} // namespace LichiiStyle
