// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <map>
#include <optional>
#include <tuple>

#include <QApplication>
#include <QFile>
#include <QFont>
#include <QFrame>
#include <QImage>
#include <QLabel>
#include <QPainter>
#include <QPaintEvent>
#include <QPalette>
#include <QPen>
#include <QPixmap>
#include <QStyle>
#include <QStyleFactory>
#include <QVBoxLayout>

#include "qt_common/config/uisettings.h"
#include "yuzu/util/lichii_style.h"

namespace LichiiStyle {

namespace {

struct Original {
    QString style;
    QPalette palette;
};

std::optional<Original>& Saved() {
    static std::optional<Original> saved;
    return saved;
}

QString Css(const QColor& color) {
    if (color.alpha() == 255) {
        return color.name(QColor::HexRgb);
    }
    return QStringLiteral("rgba(%1, %2, %3, %4)")
        .arg(color.red())
        .arg(color.green())
        .arg(color.blue())
        .arg(color.alpha());
}

QPalette PaletteFor(const Colors& c) {
    QPalette palette;
    for (const auto group : {QPalette::Active, QPalette::Inactive, QPalette::Disabled}) {
        const bool disabled = group == QPalette::Disabled;
        const QColor text = disabled ? c.tertiary : c.text;
        palette.setColor(group, QPalette::Window, c.window);
        palette.setColor(group, QPalette::WindowText, text);
        palette.setColor(group, QPalette::Base, c.card);
        palette.setColor(group, QPalette::AlternateBase, c.card);
        palette.setColor(group, QPalette::ToolTipBase, c.card);
        palette.setColor(group, QPalette::ToolTipText, c.text);
        palette.setColor(group, QPalette::PlaceholderText, c.secondary);
        palette.setColor(group, QPalette::Text, text);
        palette.setColor(group, QPalette::Button, c.fill);
        palette.setColor(group, QPalette::ButtonText, text);
        palette.setColor(group, QPalette::BrightText, c.destructive);
        palette.setColor(group, QPalette::Light, c.card.lighter(110));
        palette.setColor(group, QPalette::Midlight, c.fill_hover);
        palette.setColor(group, QPalette::Mid, c.separator);
        palette.setColor(group, QPalette::Dark, c.separator.darker(120));
        palette.setColor(group, QPalette::Shadow, QColor(0, 0, 0, 90));
        palette.setColor(group, QPalette::Highlight, disabled ? c.fill_pressed : c.accent);
        palette.setColor(group, QPalette::HighlightedText, QColor(Qt::white));
        palette.setColor(group, QPalette::Link, c.accent);
        palette.setColor(group, QPalette::LinkVisited, c.accent_pressed);
#if QT_VERSION >= QT_VERSION_CHECK(6, 6, 0)
        palette.setColor(group, QPalette::Accent, c.accent);
#endif
    }
    return palette;
}

QString SheetFor(const Colors& c, bool dark) {
    QFile file(QStringLiteral(":/lichii/style.qss"));
    if (!file.open(QFile::ReadOnly | QFile::Text)) {
        return {};
    }
    QString sheet = QString::fromUtf8(file.readAll());
    const std::pair<const char*, QString> tokens[] = {
        {"window", Css(c.window)},
        {"card", Css(c.card)},
        {"fill_hover", Css(c.fill_hover)},
        {"fill_pressed", Css(c.fill_pressed)},
        {"fill", Css(c.fill)},
        {"text", Css(c.text)},
        {"secondary", Css(c.secondary)},
        {"tertiary", Css(c.tertiary)},
        {"separator", Css(c.separator)},
        {"accent_hover", Css(c.accent_hover)},
        {"accent_pressed", Css(c.accent_pressed)},
        {"accent_soft", Css(c.accent_soft)},
        {"accent", Css(c.accent)},
        {"destructive", Css(c.destructive)},
        {"scroll", Css(c.scroll)},
        {"scheme", dark ? QStringLiteral("dark") : QStringLiteral("light")},
    };
    for (const auto& [name, value] : tokens) {
        sheet.replace(QLatin1Char('@') + QLatin1String(name) + QLatin1Char('@'), value);
    }
    return sheet;
}

QPixmap Tinted(const QImage& mask, const QColor& color) {
    QImage image = mask.convertToFormat(QImage::Format_ARGB32_Premultiplied);
    QPainter painter(&image);
    painter.setCompositionMode(QPainter::CompositionMode_SourceIn);
    painter.fillRect(image.rect(), color);
    painter.end();
    return QPixmap::fromImage(image);
}

} // namespace

bool Active() {
    return UISettings::IsLichiiTheme();
}

bool Dark() {
    return UISettings::IsDarkTheme();
}

Colors ColorsFor(bool dark) {
    if (dark) {
        return Colors{
            .window = QColor(0x1C, 0x1C, 0x1E),
            .card = QColor(0x2C, 0x2C, 0x2E),
            .fill = QColor(0x3A, 0x3A, 0x3C),
            .fill_hover = QColor(0x44, 0x44, 0x46),
            .fill_pressed = QColor(0x50, 0x50, 0x53),
            .text = QColor(0xFF, 0xFF, 0xFF),
            .secondary = QColor(0x98, 0x98, 0x9F),
            .tertiary = QColor(0x5E, 0x5E, 0x63),
            .separator = QColor(0x3F, 0x3F, 0x43),
            .accent = QColor(AccentRgb),
            .accent_hover = QColor(0x9D, 0x74, 0xF7),
            .accent_pressed = QColor(0x7B, 0x4A, 0xE6),
            .accent_soft = QColor(0x8C, 0x5C, 0xF5, 72),
            .destructive = QColor(0xFF, 0x45, 0x3A),
            .warning = QColor(0xFF, 0x9F, 0x0A),
            .success = QColor(0x30, 0xD1, 0x58),
            .scroll = QColor(0x5E, 0x5E, 0x63),
        };
    }
    return Colors{
        .window = QColor(0xF2, 0xF2, 0xF7),
        .card = QColor(0xFF, 0xFF, 0xFF),
        .fill = QColor(0xEE, 0xEE, 0xF0),
        .fill_hover = QColor(0xE5, 0xE5, 0xEA),
        .fill_pressed = QColor(0xD8, 0xD8, 0xDE),
        .text = QColor(0x00, 0x00, 0x00),
        .secondary = QColor(0x6C, 0x6C, 0x70),
        .tertiary = QColor(0xAE, 0xAE, 0xB2),
        .separator = QColor(0xC6, 0xC6, 0xC8),
        .accent = QColor(AccentRgb),
        .accent_hover = QColor(0x9A, 0x6E, 0xF7),
        .accent_pressed = QColor(0x7B, 0x4A, 0xE6),
        .accent_soft = QColor(0x8C, 0x5C, 0xF5, 41),
        .destructive = QColor(0xFF, 0x3B, 0x30),
        .warning = QColor(0xFF, 0x95, 0x00),
        .success = QColor(0x34, 0xC7, 0x59),
        .scroll = QColor(0xC7, 0xC7, 0xCC),
    };
}

Colors Current() {
    return ColorsFor(Dark());
}

void Apply(QApplication& app, bool dark) {
    if (!Saved()) {
        Saved() = Original{app.style()->name(), app.palette()};
    }
    const Colors colors = ColorsFor(dark);
    if (app.style()->name().compare(QStringLiteral("fusion"), Qt::CaseInsensitive) != 0) {
        app.setStyle(QStyleFactory::create(QStringLiteral("Fusion")));
    }
    app.setPalette(PaletteFor(colors));
    app.setStyleSheet(SheetFor(colors, dark));
}

void Restore(QApplication& app) {
    if (!Saved()) {
        return;
    }
    if (app.style()->name().compare(Saved()->style, Qt::CaseInsensitive) != 0) {
        if (QStyle* const style = QStyleFactory::create(Saved()->style)) {
            app.setStyle(style);
        }
    }
    app.setPalette(Saved()->palette);
}

QIcon Icon(const QString& name, const QColor& color) {
    static std::map<std::tuple<QString, QRgb, bool>, QIcon> cache;
    const bool dark = Dark();
    const Colors colors = ColorsFor(dark);
    const QColor tint = color.isValid() ? color : colors.text;
    const auto key = std::make_tuple(name, tint.rgba(), dark);
    if (const auto it = cache.find(key); it != cache.end()) {
        return it->second;
    }
    const QImage mask(QStringLiteral(":/lichii/icons/%1.png").arg(name));
    if (mask.isNull()) {
        return {};
    }
    QIcon icon;
    icon.addPixmap(Tinted(mask, tint), QIcon::Normal);
    icon.addPixmap(Tinted(mask, QColor(Qt::white)), QIcon::Selected);
    icon.addPixmap(Tinted(mask, colors.tertiary), QIcon::Disabled);
    cache.emplace(key, icon);
    return icon;
}

QLabel* SectionHeader(const QString& text, QWidget* parent) {
    auto* const label = new QLabel(text.toUpper(), parent);
    label->setObjectName(QStringLiteral("lichiiSectionHeader"));
    QFont font = label->font();
    font.setPointSizeF(font.pointSizeF() * 0.85);
    font.setLetterSpacing(QFont::PercentageSpacing, 104);
    label->setFont(font);
    return label;
}

QLabel* Footer(const QString& text, QWidget* parent) {
    auto* const label = new QLabel(text, parent);
    label->setObjectName(QStringLiteral("lichiiFooter"));
    label->setWordWrap(true);
    label->setTextInteractionFlags(Qt::TextBrowserInteraction);
    label->setOpenExternalLinks(true);
    QFont font = label->font();
    font.setPointSizeF(font.pointSizeF() * 0.92);
    label->setFont(font);
    return label;
}

namespace {

/// A rounded group of rows with a hairline between each two shown rows (rows hide and show as
/// other settings change, so the lines are drawn, not widgets).
class CardFrame final : public QFrame {
public:
    explicit CardFrame(QWidget* parent) : QFrame(parent) {}

protected:
    void paintEvent(QPaintEvent* event) override {
        QFrame::paintEvent(event);
        const auto* const box = layout();
        if (box == nullptr) {
            return;
        }
        QPainter painter(this);
        const QColor line = Active() ? Current().separator : palette().color(QPalette::Mid);
        painter.setPen(QPen(line, 1));
        const QWidget* previous = nullptr;
        for (int i = 0; i < box->count(); ++i) {
            const QWidget* const row = box->itemAt(i)->widget();
            if (row == nullptr || row->isHidden()) {
                continue;
            }
            if (previous != nullptr) {
                const int y = (previous->geometry().bottom() + row->geometry().top()) / 2;
                const int right = width() - box->contentsMargins().right();
                painter.drawLine(row->geometry().left(), y, right, y);
            }
            previous = row;
        }
    }
};

} // namespace

QFrame* Card(QWidget* parent) {
    auto* const card = new CardFrame(parent);
    card->setObjectName(QStringLiteral("lichiiCard"));
    auto* const layout = new QVBoxLayout(card);
    layout->setContentsMargins(14, 8, 14, 8);
    layout->setSpacing(13);
    return card;
}

void AddRow(QFrame* card, QWidget* row) {
    if (auto* const layout = qobject_cast<QVBoxLayout*>(card->layout())) {
        layout->addWidget(row);
    }
}

QLabel* PageTitle(const QString& text, QWidget* parent) {
    auto* const label = new QLabel(text, parent);
    label->setObjectName(QStringLiteral("lichiiPageTitle"));
    QFont font = label->font();
    font.setPointSizeF(font.pointSizeF() * 1.6);
    font.setBold(true);
    label->setFont(font);
    return label;
}

QString PlatformName() {
#if defined(_WIN32)
    return QStringLiteral("Windows");
#elif defined(__APPLE__)
    return QStringLiteral("macOS");
#else
    return QStringLiteral("Linux");
#endif
}

} // namespace LichiiStyle
