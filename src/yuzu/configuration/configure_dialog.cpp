// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2016 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <memory>
#include <QComboBox>
#include <QFrame>
#include <QGroupBox>
#include <QGuiApplication>
#include <QLabel>
#include <QLineEdit>
#include <QListView>
#include <QListWidget>
#include <QScreen>
#include <QScrollArea>
#include <QShowEvent>
#include <QStyle>
#include <QTimer>
#include <QVBoxLayout>
#include "common/logging.h"
#include "common/settings.h"
#include "common/settings_enums.h"
#include "core/core.h"
#include "qt_common/config/uisettings.h"
#include "qt_common/util/vk.h"
#include "ui_configure.h"
#include "yuzu/configuration/configure_applets.h"
#include "yuzu/configuration/configure_audio.h"
#include "yuzu/configuration/configure_cpu.h"
#include "yuzu/configuration/configure_debug_tab.h"
#include "yuzu/configuration/configure_dialog.h"
#include "yuzu/configuration/configure_filesystem.h"
#include "yuzu/configuration/configure_general.h"
#include "yuzu/configuration/configure_graphics.h"
#include "yuzu/configuration/configure_graphics_advanced.h"
#include "yuzu/configuration/configure_graphics_extensions.h"
#include "yuzu/configuration/configure_hotkeys.h"
#include "yuzu/configuration/configure_input.h"
#include "yuzu/configuration/configure_input_player.h"
#include "yuzu/configuration/configure_network.h"
#include "yuzu/configuration/configure_performance.h"
#include "yuzu/configuration/configure_profile_manager.h"
#include "yuzu/configuration/configure_system.h"
#include "yuzu/configuration/configure_ui.h"
#include "yuzu/configuration/configure_web.h"
#ifdef YUZU_ENABLE_MCP
#include "yuzu/configuration/configure_mcp.h"
#endif
#include "yuzu/configuration/recommended_settings.h"
#include "yuzu/hotkeys.h"
#include "qt_common/config/catalog_text.h"
#include "yuzu/util/lichii_style.h"

Q_DECLARE_METATYPE(QList<QWidget*>);

ConfigureDialog::ConfigureDialog(QWidget* parent, HotkeyRegistry& registry_,
                                 InputCommon::InputSubsystem* input_subsystem_,
                                 std::vector<VkDeviceInfo::Record>& vk_device_records,
                                 Core::System& system_, bool enable_web_config)
    : QDialog(parent), ui{std::make_unique<Ui::ConfigureDialog>()}, registry(registry_),
      system{system_},
      builder{std::make_unique<ConfigurationShared::Builder>(this, !system_.IsPoweredOn())},
      applets_tab{std::make_unique<ConfigureApplets>(system_, nullptr, *builder, this)},
      audio_tab{std::make_unique<ConfigureAudio>(system_, nullptr, *builder, this)},
      cpu_tab{std::make_unique<ConfigureCpu>(system_, nullptr, *builder, this)},
      debug_tab_tab{std::make_unique<ConfigureDebugTab>(system_, this)},
      filesystem_tab{std::make_unique<ConfigureFilesystem>(this)},
      general_tab{std::make_unique<ConfigureGeneral>(system_, nullptr, *builder, this)},
      graphics_advanced_tab{
          std::make_unique<ConfigureGraphicsAdvanced>(system_, nullptr, *builder, this)},
      graphics_extensions_tab{
          std::make_unique<ConfigureGraphicsExtensions>(system_, nullptr, *builder, this)},
      ui_tab{std::make_unique<ConfigureUi>(system_, this)},
      graphics_tab{std::make_unique<ConfigureGraphics>(
          system_, vk_device_records, [&]() { graphics_advanced_tab->ExposeComputeOption(); },
          [this](Settings::AspectRatio ratio, Settings::ResolutionSetup setup) {
              ui_tab->UpdateScreenshotInfo(ratio, setup);
          },
          nullptr, *builder, this)},
      hotkeys_tab{std::make_unique<ConfigureHotkeys>(system_.HIDCore(), this)},
      network_tab{std::make_unique<ConfigureNetwork>(system_, this)},
      performance_state{std::make_unique<Performance::State>()},
      performance_gpu_tab{
          std::make_unique<ConfigureGpu>(*performance_state, vk_device_records, this)},
      performance_shaders_tab{std::make_unique<ConfigureShaders>(system_, *builder, this)},
      performance_ram_tab{std::make_unique<ConfigureRam>(system_, *builder, this)},
      profile_tab{std::make_unique<ConfigureProfileManager>(system_, this)},
      system_tab{std::make_unique<ConfigureSystem>(system_, nullptr, *builder, this)},
      web_tab{std::make_unique<ConfigureWeb>(this)},
#ifdef YUZU_ENABLE_MCP
      mcp_tab{std::make_unique<ConfigureMcp>(this)},
#endif
      input_subsystem{input_subsystem_} {
    Settings::SetConfiguringGlobal(true);

    ui->setupUi(this);

    web_tab->SetWebServiceConfigEnabled(enable_web_config);
    hotkeys_tab->Populate(registry);

    general_tab->SetResetCallback([&] { this->close(); });

    SetConfiguration();

    // The sidebar as the iOS settings root: a search field over the list of pages.
    {
        QWidget* const sidebar = new QWidget(this);
        QVBoxLayout* const sidebar_layout = new QVBoxLayout(sidebar);
        sidebar_layout->setContentsMargins(0, 0, 0, 0);
        sidebar_layout->setSpacing(6);
        search_field = new QLineEdit(sidebar);
        search_field->setObjectName(QStringLiteral("settingsSearch"));
        search_field->setPlaceholderText(UiText::Text("search.settings"));
        search_field->setClearButtonEnabled(true);
        search_field->addAction(LichiiStyle::Icon(QStringLiteral("search"),
                                                  LichiiStyle::Current().secondary),
                                QLineEdit::LeadingPosition);
        ui->horizontalLayout->replaceWidget(ui->selectorList, sidebar);
        sidebar_layout->addWidget(search_field);
        sidebar_layout->addWidget(ui->selectorList);
        ui->selectorList->setMinimumWidth(176);
        ui->selectorList->setMaximumWidth(176);
        ui->selectorList->setIconSize(QSize(18, 18));
        ui->selectorList->setSpacing(0);
        sidebar->setMaximumWidth(176);

        search_results = new QListWidget(this);
        search_results->setObjectName(QStringLiteral("searchResults"));
        search_results->setWordWrap(true);
        search_results->setIconSize(QSize(18, 18));
        search_results->hide();
        connect(search_field, &QLineEdit::textChanged, this, &ConfigureDialog::OnSearchChanged);
        connect(search_field, &QLineEdit::returnPressed, this, [this] {
            if (search_results->count() > 0) {
                search_results->setFocus();
                search_results->setCurrentRow(0);
            }
        });
        connect(search_results, &QListWidget::itemActivated, this,
                &ConfigureDialog::OpenSearchResult);
        connect(search_results, &QListWidget::itemClicked, this,
                &ConfigureDialog::OpenSearchResult);
    }
    ui->tabWidget->setTabBarAutoHide(true);
    ui->tabWidget->setDocumentMode(true);
    // Only true while a game runs; the iOS app says it under each setting instead.
    ui->label->setVisible(system.IsPoweredOn());

    PopulateSelectionList();

    // Every tab is the dialog's child until its page first shows it; hidden until then, or it
    // draws at the dialog's top left behind the list and the buttons.
    for (const Page& page : pages) {
        for (const auto& [title, tab] : page.sections) {
            tab->hide();
        }
    }

    connect(ui->tabWidget, &QTabWidget::currentChanged, this, [this](int index) {
        if (index != -1) {
            debug_tab_tab->SetCurrentIndex(0);
        }
    });
    connect(ui_tab.get(), &ConfigureUi::LanguageChanged, this, &ConfigureDialog::OnLanguageChanged);
    connect(general_tab.get(), &ConfigureGeneral::ExternalContentDirsChanged, this,
            &ConfigureDialog::ExternalContentDirsChanged);
    connect(ui->selectorList, &QListWidget::itemSelectionChanged, this,
            &ConfigureDialog::UpdateVisibleTabs);

    if (system.IsPoweredOn()) {
        QPushButton* apply_button = ui->buttonBox->addButton(QDialogButtonBox::Apply);
        connect(apply_button, &QAbstractButton::clicked, this,
                &ConfigureDialog::HandleApplyButtonClicked);
    }

    // The recommended settings for this PC replace the global values, then the dialog closes so
    // its pages don't write their old values back.
    QPushButton* recommended = ui->buttonBox->addButton(
        UiText::Text("recommended.use") + QStringLiteral("..."), QDialogButtonBox::ResetRole);
    recommended->setEnabled(!system.IsPoweredOn());
    recommended->setToolTip(system.IsPoweredOn()
                                ? UiText::Text("recommended.stop_game")
                                : tr("Shows what changes, and why, before anything is set."));
    connect(recommended, &QAbstractButton::clicked, this, [this, &vk_device_records] {
        if (RecommendedSettings::ConfirmAndApply(this, *builder, vk_device_records)) {
            UISettings::values.configuration_applied = true;
            close();
        }
    });

    ui->selectorList->setCurrentRow(0);
    FitToAvailableScreen();

    // Selects the leftmost button on the bottom bar (Cancel as of writing)
    ui->buttonBox->setFocus();
}

ConfigureDialog::~ConfigureDialog() {
    // Deleting the tab pages makes the tab widget switch tabs; don't run the handler on a
    // half-destroyed dialog.
    ui->tabWidget->blockSignals(true);
}

void ConfigureDialog::SetConfiguration() {}

void ConfigureDialog::ApplyConfiguration() {
    general_tab->ApplyConfiguration();
    ui_tab->ApplyConfiguration();
    system_tab->ApplyConfiguration();
    profile_tab->ApplyConfiguration();
    filesystem_tab->ApplyConfiguration();
    if (input_tab) {
        input_tab->ApplyConfiguration();
    }
    hotkeys_tab->ApplyConfiguration(registry);
    cpu_tab->ApplyConfiguration();
    graphics_tab->ApplyConfiguration();
    graphics_advanced_tab->ApplyConfiguration();
    graphics_extensions_tab->ApplyConfiguration();
    performance_state->Apply();
    performance_shaders_tab->ApplyConfiguration();
    performance_ram_tab->ApplyConfiguration();
    audio_tab->ApplyConfiguration();
    debug_tab_tab->ApplyConfiguration();
    web_tab->ApplyConfiguration();
#ifdef YUZU_ENABLE_MCP
    if (mcp_tab) {
        mcp_tab->ApplyConfiguration();
    }
#endif
    network_tab->ApplyConfiguration();
    applets_tab->ApplyConfiguration();
    system.ApplySettings();
    Settings::LogSettings();
}

void ConfigureDialog::changeEvent(QEvent* event) {
    if (event->type() == QEvent::LanguageChange) {
        RetranslateUI();
    }

    QDialog::changeEvent(event);
}

void ConfigureDialog::showEvent(QShowEvent* event) {
    QDialog::showEvent(event);
    if (!initial_geometry_clamped) {
        initial_geometry_clamped = true;
        FitToAvailableScreen();
    }
}

void ConfigureDialog::FitToAvailableScreen() {
    const QScreen* active_screen =
        parentWidget() && parentWidget()->screen() ? parentWidget()->screen() : screen();
    if (!active_screen) {
        active_screen = QGuiApplication::primaryScreen();
    }
    if (!active_screen) {
        return;
    }
    const QRect avail = active_screen->availableGeometry();
    if (!avail.isValid()) {
        return;
    }
    const QRect frame = frameGeometry();
    const int extra_w = std::max(16, frame.width() - width());
    const int extra_h = std::max(38, frame.height() - height());
    const int max_client_w = std::max(minimumWidth(), avail.width() - extra_w - 16);
    const int max_client_h = std::max(minimumHeight(), avail.height() - extra_h - 16);
    const QSize desired = size().expandedTo(QSize(960, 760));
    const int new_w = std::min(desired.width(), max_client_w);
    const int new_h = std::min(desired.height(), max_client_h);
    resize(new_w, new_h);

    QPoint target = pos();
    if (parentWidget() && parentWidget()->window()) {
        const QRect p = parentWidget()->window()->frameGeometry();
        target = QPoint(p.center().x() - (new_w + extra_w) / 2,
                        p.center().y() - (new_h + extra_h) / 2);
    } else {
        target = QPoint(avail.center().x() - (new_w + extra_w) / 2,
                        avail.center().y() - (new_h + extra_h) / 2);
    }
    const int max_x = std::max(avail.left(), avail.right() - new_w - extra_w + 1);
    const int max_y = std::max(avail.top(), avail.bottom() - new_h - extra_h + 1);
    move(std::clamp(target.x(), avail.left(), max_x),
         std::clamp(target.y(), avail.top(), max_y));
}

QWidget* ConfigureDialog::WrapTab(QWidget* tab) {
    if (!tab) {
        return nullptr;
    }
    if (const auto it = tab_scroll_areas.find(tab); it != tab_scroll_areas.end()) {
        it->second->setAccessibleName(tab->accessibleName());
        return it->second;
    }
    auto* area = new QScrollArea(this);
    area->setFrameShape(QFrame::NoFrame);
    area->setWidgetResizable(true);
    area->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    area->setAccessibleName(tab->accessibleName());
    area->setWidget(tab);
    tab->show();
    tab_scroll_areas.emplace(tab, area);
    return area;
}

void ConfigureDialog::RetranslateUI() {
    const int old_row = ui->selectorList->currentRow();
    const int old_index = ui->tabWidget->currentIndex();

    ui->retranslateUi(this);

    PopulateSelectionList();
    ui->selectorList->setCurrentRow(old_row);

    UpdateVisibleTabs();
    ui->tabWidget->setCurrentIndex(old_index);
}

void ConfigureDialog::HandleApplyButtonClicked() {
    UISettings::values.configuration_applied = true;
    ApplyConfiguration();
}

void ConfigureDialog::PopulateSelectionList() {
    // The iOS app's settings root: Graphics, Controls, Audio, System, then Games & Data, then the
    // desktop app's own pages, and Advanced last. Each page shows its tabs one under the other.
    using UiText::Text;
    std::vector<Page> fresh{
        {Text("page.graphics"),
         QStringLiteral("monitor"),
         {},
         {},
         {{Text("page.graphics"), graphics_tab.get()},
          {Text("section.advanced_graphics"), graphics_advanced_tab.get()}}},
        {Text("page.controls"),
         QStringLiteral("gamepad-2"),
         {},
         {},
         {{Text("section.hotkeys"), hotkeys_tab.get()}},
         true},
        {Text("page.audio"), QStringLiteral("volume-2"), {}, {},
         {{Text("page.audio"), audio_tab.get()}}},
        {tr("Performance"),
         QStringLiteral("gauge"),
         {},
         {},
         {{tr("GPU"), performance_gpu_tab.get()},
          {tr("Shaders"), performance_shaders_tab.get()},
          {tr("RAM"), performance_ram_tab.get()}}},
        {Text("page.system"),
         QStringLiteral("cpu"),
         {},
         {},
         {{Text("page.system"), system_tab.get()},
          {Text("section.cpu"), cpu_tab.get()},
          {Text("category.LibraryApplet"), applets_tab.get()},
          {Text("section.network"), network_tab.get()}}},
        {Text("page.users"), QStringLiteral("users"), Text("group.games_data"), {},
         {{Text("page.users"), profile_tab.get()}}},
        {Text("page.storage"), QStringLiteral("hard-drive"), {}, {},
         {{Text("page.storage"), filesystem_tab.get()}}},
        {Text("page.general"),
         QStringLiteral("settings"),
         Text("group.app"),
         {},
         {{Text("page.general"), general_tab.get()},
          {Text("section.interface"), ui_tab.get()},
          {Text("section.web_service"), web_tab.get()}}},
#ifdef YUZU_ENABLE_MCP
        {tr("MCP Server"),
         QStringLiteral("wand-sparkles"),
         {},
         {},
         {{tr("MCP Server"), mcp_tab.get()}}},
#endif
        {Text("page.advanced"),
         QStringLiteral("wrench"),
         {},
         Text("advanced.warning"),
         {{Text("section.hacks_extensions"), graphics_extensions_tab.get()},
          {Text("section.debug"), debug_tab_tab.get()}}},
    };
    if (pages.empty()) {
        pages = std::move(fresh);
    } else {
        // Retranslated: the same pages and scroll areas, new names.
        for (std::size_t i = 0; i < pages.size(); ++i) {
            Page& page = pages[i];
            page.title = fresh[i].title;
            page.group = fresh[i].group;
            page.footer = fresh[i].footer;
            for (std::size_t s = 0; s < page.sections.size(); ++s) {
                page.sections[s].first = fresh[i].sections[s].first;
            }
            if (page.title_label != nullptr) {
                page.title_label->setText(page.title);
            }
            if (page.footer_label != nullptr) {
                page.footer_label->setText(page.footer);
            }
            for (const auto& [section, label] : page.section_labels) {
                label->setText(page.sections[section].first);
            }
        }
    }

    [[maybe_unused]] const QSignalBlocker blocker(ui->selectorList);

    ui->selectorList->clear();
    for (std::size_t i = 0; i < pages.size(); ++i) {
        const Page& page = pages[i];
        if (!page.group.isEmpty()) {
            auto* const heading = new QListWidgetItem(page.group);
            heading->setFlags(Qt::NoItemFlags);
            heading->setData(Qt::UserRole, -1);
            QFont font = heading->font();
            font.setBold(true);
            font.setPointSizeF(font.pointSizeF() * 0.85);
            heading->setFont(font);
            heading->setSizeHint(QSize(0, 34));
            ui->selectorList->addItem(heading);
        }
        auto* const item = new QListWidgetItem(LichiiStyle::Icon(page.icon), page.title);
        item->setData(Qt::UserRole, static_cast<int>(i));
        item->setSizeHint(QSize(0, 32));
        ui->selectorList->addItem(item);
    }
}

QScrollArea* ConfigureDialog::PageArea(Page& page) {
    if (page.area != nullptr) {
        return page.area;
    }
    QWidget* const content = new QWidget;
    QVBoxLayout* const layout = new QVBoxLayout(content);
    layout->setContentsMargins(10, 4, 14, 14);
    layout->setSpacing(6);
    page.title_label = LichiiStyle::PageTitle(page.title, content);
    layout->addWidget(page.title_label);
    if (!page.footer.isEmpty()) {
        page.footer_label = LichiiStyle::Footer(page.footer, content);
        page.footer_label->setObjectName(QStringLiteral("lichiiPageSubtitle"));
        layout->addWidget(page.footer_label);
    }
    for (std::size_t section = 0; section < page.sections.size(); ++section) {
        const auto& [title, tab] = page.sections[section];
        // A heading per tab where a page shows several; none for the first one when it carries
        // the page's own name.
        if (page.sections.size() > 1 && !(section == 0 && title == page.title)) {
            QLabel* const heading = new QLabel(title, content);
            heading->setObjectName(QStringLiteral("lichiiTabHeading"));
            QFont font = heading->font();
            font.setPointSizeF(font.pointSizeF() * 1.2);
            font.setBold(true);
            heading->setFont(font);
            heading->setContentsMargins(4, 14, 0, 0);
            layout->addWidget(heading);
            page.section_labels.emplace_back(section, heading);
        }
        layout->addWidget(tab);
        tab->show();
    }
    layout->addStretch(1);

    auto* const area = new QScrollArea(this);
    area->setFrameShape(QFrame::NoFrame);
    area->setWidgetResizable(true);
    area->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    area->setAccessibleName(page.title);
    area->setWidget(content);
    page.area = area;
    return area;
}

void ConfigureDialog::SelectPage(int index) {
    for (int row = 0; row < ui->selectorList->count(); ++row) {
        if (ui->selectorList->item(row)->data(Qt::UserRole).toInt() == index) {
            ui->selectorList->setCurrentRow(row);
            return;
        }
    }
}

void ConfigureDialog::OnSearchChanged(const QString& text) {
    const QString query = text.trimmed();
    if (query.isEmpty()) {
        search_results->hide();
        if (ui->selectorList->selectedItems().isEmpty()) {
            SelectPage(last_page);
        }
        return;
    }
    const auto matches = [&query](const QString& name) {
        return name.contains(query, Qt::CaseInsensitive);
    };
    const LichiiStyle::Colors colors = LichiiStyle::Current();
    search_results->clear();
    for (std::size_t i = 0; i < pages.size(); ++i) {
        Page& page = pages[i];
        if (matches(page.title)) {
            auto* const item = new QListWidgetItem(LichiiStyle::Icon(page.icon), page.title);
            item->setData(Qt::UserRole, static_cast<int>(i));
            search_results->addItem(item);
        }
        if (page.controls) {
            for (const auto& [title, tab] : page.sections) {
                if (matches(title)) {
                    auto* const item = new QListWidgetItem(LichiiStyle::Icon(page.icon),
                                                           page.title + QStringLiteral(" \u203A ") +
                                                               title);
                    item->setData(Qt::UserRole, static_cast<int>(i));
                    search_results->addItem(item);
                }
            }
            continue;
        }
        QWidget* const content = PageArea(page)->widget();
        for (const auto& [title, tab] : page.sections) {
            for (auto* const widget : tab->findChildren<ConfigurationShared::Widget*>()) {
                if (widget->Label().isEmpty()) {
                    continue;
                }
                const QString key = QString::fromStdString(widget->GetSetting().GetLabel());
                const bool direct_match = matches(widget->Label()) || matches(key);
                QString matched_choice;
                if (!direct_match && widget->combobox != nullptr) {
                    auto* const view = qobject_cast<QListView*>(widget->combobox->view());
                    for (int item_idx = 0; item_idx < widget->combobox->count(); ++item_idx) {
                        if (view != nullptr && view->isRowHidden(item_idx)) {
                            continue;
                        }
                        const QString choice_text = widget->combobox->itemText(item_idx);
                        if (matches(choice_text)) {
                            matched_choice = choice_text;
                            break;
                        }
                    }
                }
                if (!direct_match && matched_choice.isEmpty()) {
                    continue;
                }
                // Where it is: the page, then the group it sits in (a section or a group box).
                QString place = page.title;
                for (QWidget* parent = widget->parentWidget(); parent != nullptr && parent != tab;
                     parent = parent->parentWidget()) {
                    QString group = parent->property("lichiiSection").toString();
                    if (const auto* const box = qobject_cast<QGroupBox*>(parent);
                        group.isEmpty() && box != nullptr) {
                        group = box->title();
                    }
                    if (!group.isEmpty()) {
                        place += QStringLiteral(" \u203A ") + group;
                        break;
                    }
                }
                if (place == page.title && page.sections.size() > 1) {
                    place += QStringLiteral(" \u203A ") + title;
                }
                if (!widget->isVisibleTo(content)) {
                    place += QStringLiteral(" (%1)").arg(UiText::Text("search.hidden_now"));
                }
                const QString title_line =
                    matched_choice.isEmpty()
                        ? widget->Label()
                        : QStringLiteral("%1 (%2)").arg(widget->Label(), matched_choice);
                auto* const item = new QListWidgetItem(LichiiStyle::Icon(page.icon),
                                                       title_line + QLatin1Char('\n') + place);
                item->setData(Qt::UserRole, static_cast<int>(i));
                item->setData(Qt::UserRole + 1, QVariant::fromValue(static_cast<QObject*>(widget)));
                item->setToolTip(widget->toolTip());
                search_results->addItem(item);
            }
        }
    }
    // What a search finds that does nothing here, with the reason (iOS: "Not Available on iOS").
    QStringList unavailable;
    const auto& translations = builder->Translations();
    for (const auto& [category, settings] : Settings::values.linkage.by_category) {
        for (const auto* const setting : settings) {
            const QString reason = ConfigurationShared::PlatformUnavailableReason(*setting);
            const auto it = translations.find(setting->Id());
            if (reason.isEmpty() || it == translations.end() || it->second.first.isEmpty()) {
                continue;
            }
            QString name = it->second.first;
            while (name.endsWith(QLatin1Char(':'))) {
                name.chop(1);
            }
            if (matches(name) || matches(QString::fromStdString(setting->GetLabel()))) {
                unavailable << name + QStringLiteral(": ") + reason;
            }
        }
    }
    if (search_results->count() == 0) {
        auto* const none = new QListWidgetItem(UiText::Text("search.no_results").arg(query));
        none->setFlags(Qt::NoItemFlags);
        search_results->addItem(none);
    }
    if (!unavailable.isEmpty()) {
        auto* const heading =
            new QListWidgetItem(UiText::Text("search.unavailable").arg(LichiiStyle::PlatformName()));
        heading->setFlags(Qt::NoItemFlags);
        QFont font = heading->font();
        font.setBold(true);
        heading->setFont(font);
        search_results->addItem(heading);
        for (const QString& line : unavailable) {
            auto* const item = new QListWidgetItem(line);
            item->setFlags(Qt::NoItemFlags);
            item->setForeground(colors.secondary);
            search_results->addItem(item);
        }
    }

    {
        [[maybe_unused]] const QSignalBlocker blocker(ui->tabWidget);
        ui->tabWidget->clear();
        ui->tabWidget->addTab(search_results, tr("Search"));
    }
    search_results->show();
    ui->selectorList->clearSelection();
}

void ConfigureDialog::OpenSearchResult(QListWidgetItem* item) {
    if (item == nullptr || !(item->flags() & Qt::ItemIsEnabled)) {
        return;
    }
    const int index = item->data(Qt::UserRole).toInt();
    auto* const widget = qobject_cast<QWidget*>(item->data(Qt::UserRole + 1).value<QObject*>());
    {
        [[maybe_unused]] const QSignalBlocker blocker(search_field);
        search_field->clear();
    }
    search_results->hide();
    SelectPage(index);
    UpdateVisibleTabs();
    if (widget == nullptr || index < 0 || pages[index].area == nullptr) {
        return;
    }
    QScrollArea* const area = pages[index].area;
    QTimer::singleShot(0, widget, [area, widget] {
        area->ensureWidgetVisible(widget, 40, 120);
        widget->setProperty("lichiiSearchHit", true);
        widget->style()->unpolish(widget);
        widget->style()->polish(widget);
        widget->setFocus(Qt::OtherFocusReason);
        QTimer::singleShot(1800, widget, [widget] {
            widget->setProperty("lichiiSearchHit", false);
            widget->style()->unpolish(widget);
            widget->style()->polish(widget);
        });
    });
}

void ConfigureDialog::SelectProfilesTab() {
    for (std::size_t i = 0; i < pages.size(); ++i) {
        for (const auto& [title, tab] : pages[i].sections) {
            if (tab == profile_tab.get()) {
                SelectPage(static_cast<int>(i));
                return;
            }
        }
    }
}

void ConfigureDialog::OnLanguageChanged(const QString& locale) {
    emit LanguageChanged(locale);
    //  Reloading the game list is needed to force retranslation.
    UISettings::values.is_game_list_reload_pending = true;
    // first apply the configuration, and then restore the display
    ApplyConfiguration();
    RetranslateUI();
    SetConfiguration();
}

void ConfigureDialog::UpdateVisibleTabs() {
    const auto items = ui->selectorList->selectedItems();
    if (items.isEmpty()) {
        return;
    }
    const int index = items[0]->data(Qt::UserRole).toInt();
    if (index < 0 || static_cast<std::size_t>(index) >= pages.size()) {
        return;
    }
    if (search_field != nullptr && !search_field->text().isEmpty()) {
        [[maybe_unused]] const QSignalBlocker search_blocker(search_field);
        search_field->clear();
        search_results->hide();
    }

    [[maybe_unused]] const QSignalBlocker blocker(ui->tabWidget);

    ui->tabWidget->clear();

    last_page = index;
    Page& page = pages[index];
    if (page.controls) {
        for (auto* const tab : ControlsTabs()) {
            ui->tabWidget->addTab(WrapTab(tab), tab->accessibleName());
        }
        for (const auto& [title, tab] : page.sections) {
            ui->tabWidget->addTab(WrapTab(tab), title);
        }
        return;
    }
    ui->tabWidget->addTab(PageArea(page), page.title);
}

QList<QWidget*> ConfigureDialog::ControlsTabs() {
    if (!input_tab) {
        input_tab = std::make_unique<ConfigureInput>(system, this);
        input_tab->hide();
        input_tab->Initialize(input_subsystem);
    }
    return input_tab->GetSubTabs();
}
