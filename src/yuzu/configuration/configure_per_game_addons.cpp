// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2016 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <memory>
#include <utility>

#include <fmt/format.h>

#include <QDesktopServices>
#include <QHeaderView>
#include <QMenu>
#include <QStandardItemModel>
#include <QStandardPaths>
#include <QString>
#include <QTimer>
#include <QTreeView>

#include "common/common_types.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "configuration/addon/mod_select_dialog.h"
#include "core/core.h"
#include "core/file_sys/patch_manager.h"
#include "core/loader/loader.h"
#include "frontend_common/mod_manager.h"
#include "qt_common/abstract/frontend.h"
#include "qt_common/config/uisettings.h"
#include "qt_common/util/mod.h"
#include "ui_configure_per_game_addons.h"
#include "yuzu/configuration/configure_input.h"
#include "yuzu/configuration/configure_per_game_addons.h"

namespace {

QString BuiltinVersionText(const QString& kind, FileSys::BuiltinPatchChoice choice,
                           FileSys::BuiltinPatchSource source, const QString& included_in) {
    if (!included_in.isEmpty()) {
        return QObject::tr("%1 · included in %2").arg(kind, included_in);
    }
    using Choice = FileSys::BuiltinPatchChoice;
    if (choice != Choice::FollowGlobal) {
        return QObject::tr("%1 · set for this game").arg(kind);
    }
    return source == FileSys::BuiltinPatchSource::Global
               ? QObject::tr("%1 · follows the global setting").arg(kind)
               : QObject::tr("%1 · follows the default").arg(kind);
}

QString StripBuiltinSuffix(QString name) {
    return name.remove(QStringLiteral(" (built-in)"));
}

} // Anonymous namespace

ConfigurePerGameAddons::ConfigurePerGameAddons(Core::System& system_, QWidget* parent)
    : QWidget(parent), ui{std::make_unique<Ui::ConfigurePerGameAddons>()}, system{system_} {
    ui->setupUi(this);

    layout = new QVBoxLayout;
    tree_view = new QTreeView;
    item_model = new QStandardItemModel(tree_view);
    tree_view->setModel(item_model);
    tree_view->setAlternatingRowColors(true);
    tree_view->setSelectionMode(QHeaderView::ExtendedSelection);
    tree_view->setSelectionBehavior(QHeaderView::SelectRows);
    tree_view->setVerticalScrollMode(QHeaderView::ScrollPerPixel);
    tree_view->setHorizontalScrollMode(QHeaderView::ScrollPerPixel);
    tree_view->setSortingEnabled(true);
    tree_view->setEditTriggers(QHeaderView::NoEditTriggers);
    tree_view->setUniformRowHeights(true);
    tree_view->setContextMenuPolicy(Qt::CustomContextMenu);

    item_model->insertColumns(0, 2);
    item_model->setHeaderData(0, Qt::Horizontal, tr("Patch Name"));
    item_model->setHeaderData(1, Qt::Horizontal, tr("Version"));

    tree_view->header()->setStretchLastSection(false);
    tree_view->header()->setSectionResizeMode(0, QHeaderView::ResizeMode::Stretch);
    tree_view->header()->setMinimumSectionSize(150);

    // We must register all custom types with the Qt Automoc system so that we are able to use it
    // with signals/slots. In this case, QList falls under the umbrella of custom types.
    qRegisterMetaType<QList<QStandardItem*>>("QList<QStandardItem*>");

    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    layout->addWidget(tree_view);

    ui->scrollArea->setLayout(layout);

    ui->scrollArea->setEnabled(!system.IsPoweredOn());

    connect(item_model, &QStandardItemModel::itemChanged, this,
            &ConfigurePerGameAddons::OnItemChanged);
    connect(item_model, &QStandardItemModel::itemChanged,
            [] { UISettings::values.is_game_list_reload_pending.exchange(true); });

    connect(ui->folder, &QAbstractButton::clicked, this, &ConfigurePerGameAddons::InstallModFolder);
    connect(ui->zip, &QAbstractButton::clicked, this, &ConfigurePerGameAddons::InstallModZip);

    connect(tree_view, &QTreeView::customContextMenuRequested, this,
            &ConfigurePerGameAddons::showContextMenu);
}

ConfigurePerGameAddons::~ConfigurePerGameAddons() = default;

void ConfigurePerGameAddons::OnItemChanged(QStandardItem* item) {
    if (item->column() == 0 && item->data(BUILTIN_CHOICE).isValid() &&
        item->data(BUILTIN_CHECK).toInt() != static_cast<int>(item->checkState())) {
        // A click on a built-in patch is this game's own choice.
        const QSignalBlocker blocker{item_model};
        item->setData(static_cast<int>(item->checkState()), BUILTIN_CHECK);
        item->setData(static_cast<int>(item->checkState() == Qt::Checked
                                           ? FileSys::BuiltinPatchChoice::On
                                           : FileSys::BuiltinPatchChoice::Off),
                      BUILTIN_CHOICE);
        if (auto* version = item_model->item(item->row(), 1)) {
            version->setText(BuiltinVersionText(
                version->text().section(QStringLiteral(" · "), 0, 0),
                FileSys::BuiltinPatchChoice::On, FileSys::BuiltinPatchSource::Game, {}));
        }
        return;
    }
    if (update_items.size() > 1 && item->checkState() == Qt::Checked) {
        auto it = std::find(update_items.begin(), update_items.end(), item);
        if (it != update_items.end()) {
            for (auto* update_item : update_items) {
                if (update_item != item && update_item->checkState() == Qt::Checked) {
                    disconnect(item_model, &QStandardItemModel::itemChanged, this,
                               &ConfigurePerGameAddons::OnItemChanged);
                    update_item->setCheckState(Qt::Unchecked);
                    connect(item_model, &QStandardItemModel::itemChanged, this,
                            &ConfigurePerGameAddons::OnItemChanged);
                }
            }
        }
    }
}

void ConfigurePerGameAddons::ApplyConfiguration() {
    std::vector<std::string> disabled_addons;

    std::vector<std::pair<std::string, FileSys::BuiltinPatchChoice>> builtin_choices;

    for (const auto& item : list_items) {
        if (item.front()->data(BUILTIN_CHOICE).isValid()) {
            builtin_choices.emplace_back(item.front()->text().toStdString(),
                                         static_cast<FileSys::BuiltinPatchChoice>(
                                             item.front()->data(BUILTIN_CHOICE).toInt()));
            continue;
        }
        const auto disabled = item.front()->checkState() == Qt::Unchecked;
        if (disabled) {
            QVariant userData = item.front()->data(Qt::UserRole);
            if (userData.isValid() && userData.canConvert<quint32>() &&
                item.front()->text() == QStringLiteral("Update")) {
                quint32 numeric_version = userData.toUInt();
                disabled_addons.push_back(fmt::format("Update@{}", numeric_version));
            } else {
                disabled_addons.push_back(item.front()->text().toStdString());
            }
        }
    }

    for (const auto& [name, choice] : builtin_choices) {
        FileSys::ApplyBuiltinPatchChoice(disabled_addons, name, choice);
    }

    auto current = Settings::values.disabled_addons[title_id];
    std::sort(disabled_addons.begin(), disabled_addons.end());
    std::sort(current.begin(), current.end());
    if (disabled_addons != current) {
        Common::FS::RemoveFile(Common::FS::GetEdenPath(Common::FS::EdenPath::CacheDir) /
                               "game_list" / fmt::format("{:016X}.pv.txt", title_id));
    }

    Settings::values.disabled_addons[title_id] = disabled_addons;
}

void ConfigurePerGameAddons::LoadFromFile(FileSys::VirtualFile file_) {
    SetFile(std::move(file_));
    LoadConfiguration();
}

void ConfigurePerGameAddons::SetFile(FileSys::VirtualFile file_) {
    file = std::move(file_);
}

void ConfigurePerGameAddons::SetTitleId(u64 id) {
    this->title_id = id;
}

void ConfigurePerGameAddons::InstallMods(const QStringList& mods) {
    QStringList failed;
    for (const auto& mod : mods) {
        if (FrontendCommon::InstallMod(mod.toStdString(), title_id, true) ==
            FrontendCommon::Failed) {
            failed << QFileInfo(mod).baseName();
        }
    }

    if (failed.empty()) {
        QtCommon::Frontend::Information(tr("Mod Install Succeeded"),
                                        tr("Successfully installed all mods."));

        item_model->removeRows(0, item_model->rowCount());
        list_items.clear();
        LoadConfiguration();

        UISettings::values.is_game_list_reload_pending.exchange(true);
    } else {
        QtCommon::Frontend::Critical(
            tr("Mod Install Failed"),
            tr("Failed to install the following mods:\n\t%1\nCheck the log for details.")
                .arg(failed.join(QStringLiteral("\n\t"))));
    }
}

void ConfigurePerGameAddons::InstallModPath(const QString& path, const QString& fallbackName) {
    const auto mods = QtCommon::Mod::GetModFolders(path, fallbackName);

    if (mods.size() > 1) {
        ModSelectDialog* dialog = new ModSelectDialog(mods, this);
        connect(dialog, &ModSelectDialog::modsSelected, this, &ConfigurePerGameAddons::InstallMods);
        dialog->show();
    } else if (!mods.empty()) {
        InstallMods(mods);
    }
}

void ConfigurePerGameAddons::InstallModFolder() {
    const auto path = QtCommon::Frontend::GetExistingDirectory(
        tr("Mod Folder"), QStandardPaths::writableLocation(QStandardPaths::DownloadLocation));
    if (path.isEmpty()) {
        return;
    }

    InstallModPath(path);
}

void ConfigurePerGameAddons::InstallModZip() {
    // TODO(crueter): use GetOpenFileName to allow select multiple ZIPs
    const auto path = QtCommon::Frontend::GetOpenFileName(
        tr("Zipped Mod Location"),
        QStandardPaths::writableLocation(QStandardPaths::DownloadLocation),
        tr("Zipped Archives (*.zip)"));
    if (path.isEmpty()) {
        return;
    }

    const QString extracted = QtCommon::Mod::ExtractMod(path);
    if (!extracted.isEmpty())
        InstallModPath(extracted, QFileInfo(path).baseName());
}

void ConfigurePerGameAddons::AddonDeleteRequested(QList<QModelIndex> selected) {
    QList<QModelIndex> filtered;
    for (const QModelIndex& index : selected) {
        if (!index.data(PATCH_LOCATION).toString().isEmpty())
            filtered << index;
    }

    if (filtered.empty()) {
        QtCommon::Frontend::Critical(tr("Invalid Selection"),
                                     tr("Only mods, cheats, and patches can be deleted.\nTo delete "
                                        "NAND-installed updates, right-click the game in the game "
                                        "list and click Remove -> Remove Installed Update."));
        return;
    }

    const auto header = tr("You are about to delete the following installed mods:\n");
    QString selected_str;
    for (const QModelIndex& index : filtered) {
        selected_str = selected_str % index.data().toString() % QStringLiteral("\n");
    }

    const auto footer = tr("\nOnce deleted, these can NOT be recovered. Are you 100% sure "
                           "you want to delete them?");

    QString caption = header % selected_str % footer;

    auto choice = QtCommon::Frontend::Warning(tr("Delete add-on(s)?"), caption,
                                              QtCommon::Frontend::StandardButton::Yes |
                                                  QtCommon::Frontend::StandardButton::No);

    if (choice == QtCommon::Frontend::StandardButton::No)
        return;

    for (const QModelIndex& index : filtered) {
        std::filesystem::remove_all(index.data(PATCH_LOCATION).toString().toStdString());
    }

    QtCommon::Frontend::Information(tr("Successfully deleted"),
                                    tr("Successfully deleted all selected mods."));

    item_model->removeRows(0, item_model->rowCount());
    list_items.clear();
    LoadConfiguration();

    UISettings::values.is_game_list_reload_pending.exchange(true);
}

void ConfigurePerGameAddons::showContextMenu(const QPoint& pos) {
    const QModelIndex index = tree_view->indexAt(pos);
    auto selected = tree_view->selectionModel()->selectedRows();
    if (index.isValid() && selected.empty()) {
        QModelIndex idx = item_model->index(index.row(), 0);
        if (idx.isValid())
            selected << idx;
    }

    if (selected.empty())
        return;

    QMenu menu(this);

    const bool all_builtin = std::ranges::all_of(selected, [](const QModelIndex& index) {
        return index.data(BUILTIN_CHOICE).isValid();
    });
    if (all_builtin) {
        QAction* follow = menu.addAction(tr("&Follow the global setting"));
        connect(follow, &QAction::triggered, this, [this, selected]() {
            const QSignalBlocker blocker{item_model};
            for (const QModelIndex& index : selected) {
                auto* item = item_model->itemFromIndex(index.sibling(index.row(), 0));
                const bool on = item->data(BUILTIN_FOLLOW_STATE).toBool();
                item->setCheckState(on ? Qt::Checked : Qt::Unchecked);
                item->setData(static_cast<int>(item->checkState()), BUILTIN_CHECK);
                item->setData(static_cast<int>(FileSys::BuiltinPatchChoice::FollowGlobal),
                              BUILTIN_CHOICE);
                if (auto* version = item_model->item(item->row(), 1)) {
                    version->setText(BuiltinVersionText(
                        version->text().section(QStringLiteral(" · "), 0, 0),
                        FileSys::BuiltinPatchChoice::FollowGlobal,
                        FileSys::BuiltinPatchSource::Global, {}));
                }
            }
            UISettings::values.is_game_list_reload_pending.exchange(true);
        });
        menu.addSeparator();
    }

    QAction* remove = menu.addAction(tr("&Delete"));
    connect(remove, &QAction::triggered, this,
            [this, selected]() { AddonDeleteRequested(selected); });

    if (selected.length() == 1) {
        auto loc = selected.at(0).data(PATCH_LOCATION).toString();
        if (QFileInfo::exists(loc)) {
            QAction* open = menu.addAction(tr("&Open in File Manager"));
            connect(open, &QAction::triggered, this,
                    [selected, loc]() { QDesktopServices::openUrl(QUrl::fromLocalFile(loc)); });
        }
    }

    menu.exec(tree_view->viewport()->mapToGlobal(pos));
}

void ConfigurePerGameAddons::changeEvent(QEvent* event) {
    if (event->type() == QEvent::LanguageChange) {
        RetranslateUI();
    }

    QWidget::changeEvent(event);
}

void ConfigurePerGameAddons::RetranslateUI() {
    ui->retranslateUi(this);
}

void ConfigurePerGameAddons::LoadConfiguration() {
    if (file == nullptr) {
        return;
    }

    const FileSys::PatchManager pm{title_id, system.GetFileSystemController(),
                                   system.GetContentProvider()};
    const auto loader = Loader::GetLoader(system, file);

    FileSys::VirtualFile update_raw;
    loader->ReadUpdateRaw(update_raw);

    const auto& disabled = Settings::values.disabled_addons[title_id];

    update_items.clear();
    list_items.clear();
    item_model->removeRows(0, item_model->rowCount());

    std::vector<FileSys::Patch> patches = pm.GetPatches(update_raw);

    bool has_enabled_update = false;

    for (const auto& patch : patches) {
        const auto name = QString::fromStdString(patch.name);

        auto* const first_item = new QStandardItem;
        first_item->setText(name);
        first_item->setCheckable(true);

        const bool is_external_update = patch.type == FileSys::PatchType::Update &&
                                        patch.source == FileSys::PatchSource::External &&
                                        patch.numeric_version != 0;

        const bool is_mod = patch.type == FileSys::PatchType::Mod;

        if (is_external_update) {
            first_item->setData(static_cast<quint32>(patch.numeric_version), NUMERIC_VERSION);
        } else if (is_mod) {
            first_item->setData(QString::fromStdString(patch.location), PATCH_LOCATION);
        }

        bool patch_disabled = false;
        if (is_external_update) {
            std::string disabled_key = fmt::format("Update@{}", patch.numeric_version);
            patch_disabled =
                std::find(disabled.begin(), disabled.end(), disabled_key) != disabled.end();
        } else {
            patch_disabled =
                std::find(disabled.begin(), disabled.end(), name.toStdString()) != disabled.end();
        }

        bool should_enable = !patch_disabled;

        QString version_text = QString::fromStdString(patch.version);
        QString tooltip;
        if (patch.builtin) {
            const auto& info = *patch.builtin;
            const auto included_in = StripBuiltinSuffix(QString::fromStdString(info.included_in));
            should_enable = patch.enabled;
            first_item->setData(static_cast<int>(info.choice), BUILTIN_CHOICE);
            first_item->setData(FileSys::BuiltinPatchEnabledWhenFollowing(patch.name),
                                BUILTIN_FOLLOW_STATE);
            first_item->setData(should_enable ? static_cast<int>(Qt::Checked)
                                              : static_cast<int>(Qt::Unchecked),
                                BUILTIN_CHECK);
            version_text = BuiltinVersionText(
                QString::fromUtf8(FileSys::BuiltinPatchKindName(info.kind)), info.choice,
                info.source, included_in);
            tooltip = tr("%1\n\n%2\n\n%3")
                          .arg(QString::fromStdString(info.description),
                               QString::fromStdString(info.effect),
                               included_in.isEmpty()
                                   ? tr("Right-click to follow the global setting (Configure > "
                                        "Graphics).")
                                   : tr("Included in %1, which is on; this one does nothing "
                                        "while it is.")
                                         .arg(included_in));
            if (!included_in.isEmpty()) {
                first_item->setEnabled(false);
            }
        }

        if (patch.type == FileSys::PatchType::Update) {
            if (should_enable) {
                if (has_enabled_update) {
                    should_enable = false;
                } else {
                    has_enabled_update = true;
                }
            }
            update_items.push_back(first_item);
        }

        first_item->setCheckState(should_enable ? Qt::Checked : Qt::Unchecked);

        auto* const version_item = new QStandardItem{version_text};
        if (!tooltip.isEmpty()) {
            first_item->setToolTip(tooltip);
            version_item->setToolTip(tooltip);
            version_item->setEnabled(first_item->isEnabled());
        }
        list_items.push_back(QList<QStandardItem*>{first_item, version_item});
        item_model->appendRow(list_items.back());
    }

    tree_view->resizeColumnToContents(1);
}
