// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <QFontMetrics>
#include <QLinearGradient>
#include <QPainter>
#include <QPainterPath>
#include "game_card.h"
#include "qt_common/config/uisettings.h"
#include "qt_common/game_list/game_list_p.h"
#include "yuzu/util/lichii_style.h"

GameCard::GameCard(QObject* parent) : QStyledItemDelegate{parent} {
    setObjectName("GameCard");
}

void GameCard::paint(QPainter* painter, const QStyleOptionViewItem& option,
                     const QModelIndex& index) const {
    if (!index.isValid())
        return;

    // The iOS library's card: the cover with rounded corners and a soft shadow (a play badge while
    // the pointer is on it), the title on up to two lines and the format under it.
    painter->save();
    painter->setRenderHint(QPainter::Antialiasing);
    painter->setRenderHint(QPainter::SmoothPixmapTransform, true);

    constexpr int cardCornerRadius = 12;

    const QRect cardRect = getCardRect(option, index);
    const bool lichii = LichiiStyle::Active();
    const LichiiStyle::Colors colors = LichiiStyle::Current();
    const QPalette& palette = option.palette;
    const QColor textColor = lichii ? colors.text : palette.text().color();
    const QColor secondaryColor = lichii ? colors.secondary : palette.placeholderText().color();
    const bool selected = option.state & QStyle::State_Selected;
    const bool hovered = option.state & QStyle::State_MouseOver;

    if (selected || hovered) {
        QColor background = selected ? (lichii ? colors.accent_soft : palette.highlight().color())
                                     : (lichii ? colors.fill : palette.midlight().color());
        if (!lichii && !selected) {
            background.setAlpha(120);
        }
        painter->setPen(Qt::NoPen);
        painter->setBrush(background);
        painter->drawRoundedRect(cardRect, cardCornerRadius, cardCornerRadius);
    }

    const int icon_size = static_cast<int>(UISettings::values.game_icon_size.GetValue());
    QPixmap icon_pixmap = index.data(Qt::DecorationRole).value<QPixmap>();
    // A game without an icon gets a transparent one (GetDefaultIcon).
    if (!icon_pixmap.isNull() && icon_pixmap.hasAlphaChannel() &&
        icon_pixmap.copy(icon_pixmap.width() / 2, icon_pixmap.height() / 2, 1, 1)
                .toImage()
                .pixelColor(0, 0)
                .alpha() == 0) {
        icon_pixmap = QPixmap();
    }
    const int side = std::max(icon_size, 32);
    const QRect iconRect{cardRect.left() + (cardRect.width() - side) / 2,
                         cardRect.top() + cardMargin - 1, side, side};
    const qreal radius = side * 0.12;
    QString title = index.data(GameListItemPath::TitleRole).toString();
    if (title.isEmpty()) {
        title = index.data(Qt::DisplayRole).toString().split(QLatin1Char('\n')).first().trimmed();
    }

    if (icon_size > 0) {
        painter->setPen(Qt::NoPen);
        painter->setBrush(QColor(0, 0, 0, 46));
        painter->drawRoundedRect(QRectF(iconRect).translated(0, 3).adjusted(1, 1, -1, 0), radius,
                                 radius);
        QPainterPath clip_path;
        clip_path.addRoundedRect(iconRect, radius, radius);
        painter->save();
        painter->setClipPath(clip_path);
        if (!icon_pixmap.isNull()) {
            painter->drawPixmap(iconRect, icon_pixmap);
        } else {
            // No icon: the initials on the accent gradient, as the iOS app's GameIcon.
            QLinearGradient gradient(iconRect.topLeft(), iconRect.bottomRight());
            QColor accent{LichiiStyle::AccentRgb};
            accent.setAlphaF(0.85f);
            gradient.setColorAt(0, accent);
            gradient.setColorAt(1, QColor(0x58, 0x56, 0xD6));
            painter->fillRect(iconRect, gradient);
            QString initials;
            const QStringList words = title.split(QLatin1Char(' '), Qt::SkipEmptyParts);
            for (const QString& word : words.mid(0, 2)) {
                initials += word.at(0).toUpper();
            }
            QFont font = option.font;
            font.setBold(true);
            font.setPixelSize(std::max(10, side * 3 / 10));
            painter->setFont(font);
            painter->setPen(Qt::white);
            painter->drawText(iconRect, Qt::AlignCenter, initials);
        }
        painter->restore();

        if (hovered && side >= 64) {
            const int badge = std::max(24, side / 5);
            const QRect badgeRect{iconRect.right() - badge - side / 16,
                                  iconRect.bottom() - badge - side / 16, badge, badge};
            painter->setPen(Qt::NoPen);
            painter->setBrush(QColor(LichiiStyle::AccentRgb));
            painter->drawEllipse(badgeRect);
            const QPointF c = QRectF(badgeRect).center();
            const qreal t = badge * 0.2;
            const QPointF play[] = {{c.x() - t * 0.7, c.y() - t}, {c.x() - t * 0.7, c.y() + t},
                                    {c.x() + t, c.y()}};
            painter->setBrush(Qt::white);
            painter->drawPolygon(play, 3);
        }
    }

    if (UISettings::values.show_game_name.GetValue()) {
        const int text_left = icon_size > 0
                                  ? std::min(iconRect.left(), cardRect.left() + cardMargin)
                                  : cardRect.left() + cardMargin;
        const int text_width = cardRect.right() - cardMargin - text_left;
        const int top = (icon_size > 0 ? iconRect.bottom() : cardRect.top()) + cardMargin;

        QFont title_font = option.font;
        title_font.setWeight(QFont::DemiBold);
        const QFontMetrics title_metrics(title_font);
        painter->setFont(title_font);
        painter->setPen(selected && !lichii ? palette.highlightedText().color() : textColor);
        const QRect titleBox{text_left, top, text_width, title_metrics.lineSpacing() * 2};
        QRect titleRect;
        painter->drawText(titleBox, Qt::AlignLeft | Qt::AlignTop | Qt::TextWordWrap,
                          title_metrics.elidedText(title, Qt::ElideRight, text_width * 2 - 8),
                          &titleRect);
        titleRect.setBottom(std::min(titleRect.bottom(), titleBox.bottom()));

        QFont detail_font = option.font;
        detail_font.setPointSizeF(detail_font.pointSizeF() * 0.88);
        const QFontMetrics detail_metrics(detail_font);
        const QString detail = index.data(GameListItemPath::FileTypeRole).toString();
        painter->setFont(detail_font);
        painter->setPen(secondaryColor);
        const QRect detailRect{text_left, titleRect.bottom() + 1, text_width,
                               detail_metrics.lineSpacing()};
        if (detailRect.bottom() <= cardRect.bottom() + cardMargin) {
            painter->drawText(detailRect, Qt::AlignLeft | Qt::AlignTop,
                              detail_metrics.elidedText(detail, Qt::ElideRight, text_width));
        }
    }

    painter->restore();
}

QRect GameCard::getCardRect(const QStyleOptionViewItem& option, const QModelIndex& index) const {
    const int cell_width = option.rect.width();

    const int card_width = cell_width - m_padding;

    int card_left, card_top, card_height;

    if (m_columns >= 1) {
        // grid mode

        // center everything in-line, such that the leftmost and rightmost cards
        // have ~ equal padding to the edge of the viewport
        // spacing between each card is larger, but equal to each other
        const int column = index.row() % m_columns;

        const int row_width = m_columns * cell_width;
        const int total_gap = row_width - cardMargin * 2 - m_columns * card_width;
        const int gap = (m_columns > 1) ? (total_gap / (m_columns - 1)) : 0;
        card_left =
            option.rect.left() - column * cell_width + cardMargin + column * (card_width + gap) + 4;

        // fill cell vertically
        card_top = option.rect.top() + cardMargin;
        card_height = option.rect.height() - cardMargin - 1;
    } else {
        // carousel mode
        card_left = option.rect.left() + cardMargin + 4;

        // the delegate itself takes up the full height, but the card itself
        // gets centered
        const int content_height = m_contentSize.height() - cardMargin;
        const int cell_height = option.rect.height();

        card_height = std::min(content_height, cell_height - cardMargin * 2) - 1;
        card_top = option.rect.top() + (cell_height - card_height) / 2;
    }

    return QRect(card_left, card_top, card_width - cardMargin, card_height);
}

bool GameCard::hitTest(const QPoint& point, const QModelIndex& index, const QWidget* widget,
                       const QRect& cellRect) const {
    QStyleOptionViewItem option;
    option.initFrom(widget);
    option.rect = cellRect;
    return getCardRect(option, index).contains(point);
}

QSize GameCard::sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const {
    return m_size;
}

void GameCard::setSize(const QSize& newSize, const QSize& contentSize, const int padding,
                       const int columns) {
    m_size = newSize;
    m_contentSize = contentSize;
    m_padding = padding;
    m_columns = columns;
}
