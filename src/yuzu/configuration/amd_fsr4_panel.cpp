// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#if defined(_WIN32) || defined(__linux__)

#ifdef _WIN32
#include "common/httplib.h"
#include "common/net/ca_store.h"
#endif

#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <QApplication>
#include <QCryptographicHash>
#include <QDesktopServices>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProgressDialog>
#include <QPushButton>
#include <QStringList>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>
#include <fmt/format.h>

#include "common/common_types.h"
#include "common/fs/fs_util.h"
#include "common/logging.h"
#include "qt_common/config/catalog_text.h"
#include "video_core/vulkan_common/ffx_files.h"
#include "video_core/vulkan_common/ffx_runtime.h"
#include "yuzu/configuration/amd_fsr4_panel.h"

namespace {

namespace FfxRuntime = Vulkan::FfxRuntime;

QString Tr(const char* text) {
    return QCoreApplication::translate("AmdFsr4Panel", text);
}

QString Fsr4Name() {
    return UiText::ChoiceLabel("upscaler", "Fsr4");
}

QString Fsr3Name() {
    return UiText::ChoiceLabel("upscaler", "Fsr3");
}

QString PathText(const std::filesystem::path& path) {
    return QDir::toNativeSeparators(QString::fromStdU16String(path.u16string()));
}

#ifdef _WIN32
u64 DownloadSize() {
    u64 total = 0;
    for (const FfxRuntime::File& file : FfxRuntime::FILES) {
        total += file.size;
    }
    return total;
}

/// GETs one file of the SDK repository at the pinned commit. Empty with `error` set on failure.
std::optional<std::string> Fetch(std::string_view entry, std::string& error,
                                 const std::function<bool(u64)>& progress = {}) {
    httplib::Client client{std::string{FfxRuntime::DOWNLOAD_HOST}};
    client.set_connection_timeout(30);
    client.set_read_timeout(30);
    client.set_write_timeout(30);
    Common::Net::ConfigureCaStore(client);
    std::string body;
    int status = 0;
    const auto result = client.Get(
        fmt::format("{}{}", FfxRuntime::DOWNLOAD_PATH, entry),
        [&status](const httplib::Response& response) {
            status = response.status;
            return response.status == 200;
        },
        [&](const char* data, size_t length) {
            body.append(data, length);
            return !progress || progress(body.size());
        });
    if (!result || status != 200) {
        error = result ? fmt::format("HTTP {}", status) : httplib::to_string(result.error());
        return std::nullopt;
    }
    return body;
}
#endif

} // namespace

AmdFsr4Panel::AmdFsr4Panel(QWidget* parent) : QWidget(parent) {
    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->setContentsMargins(24, 0, 0, 8);
    layout->setSpacing(6);

    QHBoxLayout* status_row = new QHBoxLayout;
    status_row->setSpacing(8);
    icon_label = new QLabel(this);
    icon_label->setTextFormat(Qt::RichText);
    icon_label->setAlignment(Qt::AlignTop | Qt::AlignHCenter);
    icon_label->setFixedWidth(16);
    QVBoxLayout* text_column = new QVBoxLayout;
    text_column->setSpacing(4);
    status_label = new QLabel(this);
    status_label->setTextFormat(Qt::RichText);
    status_label->setWordWrap(true);
    session_label = new QLabel(this);
    session_label->setTextFormat(Qt::RichText);
    session_label->setWordWrap(true);
    text_column->addWidget(status_label);
    text_column->addWidget(session_label);
    status_row->addWidget(icon_label);
    status_row->addLayout(text_column, 1);

    QHBoxLayout* button_row = new QHBoxLayout;
    button_row->setContentsMargins(24, 0, 0, 0);
    download_button = new QPushButton(this);
    folder_button = new QPushButton(Tr("Open Folder"), this);
#if defined(__linux__)
    QLabel* about =
        new QLabel(QStringLiteral("<a href=\"https://github.com/Astyyyyy/fsr4-vk\">%1</a>")
                       .arg(Tr("About fsr4-vk")),
                   this);
    about->setOpenExternalLinks(true);
    about->setToolTip(Tr("fsr4-vk on GitHub (GPL-3.0). Its licence and AMD's notice ship in the "
                         "package's share/licenses/fsr4-vk folder."));
    download_button->setVisible(false);
#else
    QLabel* about = new QLabel(
        QStringLiteral("<a href=\"%1\">%2</a>")
            .arg(QString::fromUtf8(FfxRuntime::RELEASE_PAGE.data(),
                                   static_cast<qsizetype>(FfxRuntime::RELEASE_PAGE.size())),
                 Tr("Where do these files come from?")),
        this);
    about->setOpenExternalLinks(true);
    about->setToolTip(Tr("AMD's FidelityFX SDK release on GitHub. The files are in "
                         "Kits\\FidelityFX\\signedbin."));
#endif
    button_row->addWidget(download_button);
    button_row->addWidget(folder_button);
    button_row->addStretch(1);
    button_row->addWidget(about);

    layout->addLayout(status_row);
    layout->addLayout(button_row);

    refresh_timer = new QTimer(this);
    refresh_timer->setInterval(1000);
    connect(refresh_timer, &QTimer::timeout, this, [this] { Refresh(); });

    connect(download_button, &QPushButton::clicked, this, [this] { Download(); });
    connect(folder_button, &QPushButton::clicked, this, [] {
        const std::filesystem::path folder = FfxRuntime::Folder();
        std::error_code ec;
        std::filesystem::create_directories(folder, ec);
        QDesktopServices::openUrl(QUrl::fromLocalFile(PathText(folder)));
    });
    Refresh();
}

void AmdFsr4Panel::showEvent(QShowEvent* event) {
    Refresh();
    refresh_timer->start();
    QWidget::showEvent(event);
}

void AmdFsr4Panel::hideEvent(QHideEvent* event) {
    refresh_timer->stop();
    QWidget::hideEvent(event);
}

void AmdFsr4Panel::SetDevice(const QString& name, bool capable) {
    device_name = name;
    device_capable = capable;
    Refresh();
}

void AmdFsr4Panel::SetUpscalerSelected(bool selected) {
    if (upscaler_selected == selected) {
        return;
    }
    upscaler_selected = selected;
    Refresh();
}

void AmdFsr4Panel::Refresh() {
    QStringList lines;
    if (!device_name.isEmpty()) {
        lines << Tr("GPU: %1").arg(device_name.toHtmlEscaped());
    }
    if (!device_capable) {
        lines << Tr("<b>GPU not supported:</b> %1 requires an AMD Radeon RX 6000/7000/9000 or "
                    "Ryzen RDNA 3+ GPU on Vulkan (%2 runs instead).")
                     .arg(Fsr4Name().toHtmlEscaped(), Fsr3Name().toHtmlEscaped());
    }

#if defined(__linux__)
    const std::filesystem::path library = FfxRuntime::PortLibraryPath();
    const bool ready = !library.empty();
    lines << (ready
                  ? Tr("<b>Files ready:</b> found fsr4-vk, the community Vulkan port of %1 (not "
                       "AMD's official runtime). %2 runs if the GPU or driver lacks a required "
                       "feature.")
                        .arg(Fsr4Name().toHtmlEscaped(), Fsr3Name().toHtmlEscaped())
                  : Tr("<b>Not in this build:</b> Lichii's x86-64 Linux packages include fsr4-vk. "
                       "For a build without it, place amd_fidelityfx_upscaler_vk.so from fsr4-vk "
                       "in the folder. Until then, %1 is used.")
                        .arg(Fsr3Name().toHtmlEscaped()));
    status_label->setToolTip(ready ? PathText(library)
                                   : Tr("Folder: %1").arg(PathText(FfxRuntime::Folder())));
    download_button->setVisible(false);
#else
    const bool ready = FfxRuntime::FilesPresent();
    const QString sdk_version =
        QString::fromUtf8(FfxRuntime::SDK_VERSION.data(),
                          static_cast<qsizetype>(FfxRuntime::SDK_VERSION.size()));
    lines << (ready
                  ? Tr("<b>Files ready:</b> AMD's FidelityFX runtime from FidelityFX SDK %1. %2 "
                       "runs if AMD's driver offers it for this GPU; otherwise %3 does.")
                        .arg(sdk_version, Fsr4Name().toHtmlEscaped(), Fsr3Name().toHtmlEscaped())
                  : Tr("<b>AMD's files needed:</b> the FidelityFX runtime from AMD's FidelityFX "
                       "SDK %1 (amd_fidelityfx_loader_vk.dll and amd_fidelityfx_upscaler_vk.dll). "
                       "Until then, %2 is used.")
                        .arg(sdk_version, Fsr3Name().toHtmlEscaped()));
    status_label->setToolTip(Tr("Folder: %1").arg(PathText(FfxRuntime::Folder())));
    download_button->setText(
        Tr("Download from AMD (%1 MB)").arg((DownloadSize() + (1 << 20) - 1) >> 20));
    download_button->setVisible(!ready && device_capable);
#endif
    status_label->setText(lines.join(QStringLiteral("<br>")));

    const FfxRuntime::SessionStatus session = FfxRuntime::GetSessionStatus();
    const QString runtime_name =
        session.version.empty()
            ? Fsr4Name()
            : QStringLiteral("%1 %2").arg(Fsr4Name(), QString::fromStdString(session.version));
    const QString covered =
        Tr("Games it covers: The Legend of Zelda: Ocarina of Time (at 1x resolution scale) and "
           "Fire Emblem: Fortune's Weave.");
    QString state;
    bool active = false;
    if (session.available && session.frames > 0) {
        active = true;
        state = Tr("<b>Active:</b> %1 has upscaled %2 frames in this game.")
                    .arg(runtime_name.toHtmlEscaped())
                    .arg(session.frames);
    } else if (session.available) {
        state = Tr("<b>Running</b>, waiting for a pass %1 can replace. %2")
                    .arg(Fsr4Name().toHtmlEscaped(), covered);
    } else if (!session.problem.empty()) {
        state = Tr("<b>Not used in the last game:</b> %1 (%2 ran instead).")
                    .arg(QString::fromStdString(session.problem).toHtmlEscaped(),
                         Fsr3Name().toHtmlEscaped());
    } else if (session.frames > 0) {
        state = Tr("Last game: %1 upscaled %2 frames. %3")
                    .arg(runtime_name.toHtmlEscaped())
                    .arg(session.frames)
                    .arg(covered);
    } else if (!upscaler_selected) {
        state = Tr("Select <b>%1</b> in %2 above to enable it for the next game. %3")
                    .arg(Fsr4Name().toHtmlEscaped(),
                         UiText::SettingLabel("upscaler").toHtmlEscaped(), covered);
    } else {
        state = Tr("Starts with the next game. %1").arg(covered);
    }
    session_label->setText(state);

    const bool problem =
        !device_capable || !ready || (!session.available && !session.problem.empty());
    icon_label->setText(
        problem ? QStringLiteral("<span style=\"color:#d29922\">&#9888;</span>")
        : active ? QStringLiteral("<span style=\"color:#2ea043\">&#9679;</span>")
                 : QStringLiteral("<span style=\"color:#2ea043\">&#10004;</span>"));
}

void AmdFsr4Panel::Download() {
#ifdef _WIN32
    const std::filesystem::path folder = FfxRuntime::Folder();
    const auto fail = [this](const std::string& error) {
        LOG_ERROR(Frontend, "AMD FSR 4 download failed: {}", error);
        QMessageBox::warning(this, Fsr4Name(),
                             Tr("Could not download AMD's files: %1. %2 is used.")
                                 .arg(QString::fromStdString(error), Fsr3Name()));
        Refresh();
    };

    std::string error;
    std::optional<std::string> licence;
    {
        QProgressDialog busy(Tr("Contacting AMD's download..."), QString{}, 0, 0, this);
        busy.setWindowTitle(Fsr4Name());
        busy.setWindowModality(Qt::WindowModal);
        busy.setMinimumDuration(0);
        busy.show();
        QApplication::processEvents();
        licence = Fetch(FfxRuntime::LICENSE_ENTRY, error);
    }
    if (!licence || licence->empty()) {
        fail(error.empty() ? "could not retrieve AMD's licence text" : error);
        return;
    }

    QDialog dialog(this);
    dialog.setWindowTitle(Tr("Download %1").arg(Fsr4Name()));
    QVBoxLayout* layout = new QVBoxLayout(&dialog);
    QLabel* intro = new QLabel(
        Tr("Lichii will download amd_fidelityfx_loader_vk.dll and amd_fidelityfx_upscaler_vk.dll "
           "from AMD's FidelityFX SDK %1 release on GitHub into %2.<br>These files are AMD's, "
           "under AMD's licence terms below. Lichii checks AMD's signature and SHA-256 checksum "
           "before using them.")
            .arg(QString::fromUtf8(FfxRuntime::SDK_VERSION.data(),
                                   static_cast<qsizetype>(FfxRuntime::SDK_VERSION.size())),
                 PathText(folder)),
        &dialog);
    intro->setWordWrap(true);
    QPlainTextEdit* licence_text = new QPlainTextEdit(QString::fromStdString(*licence), &dialog);
    licence_text->setReadOnly(true);
    licence_text->setMinimumSize(640, 380);
    QDialogButtonBox* buttons = new QDialogButtonBox(&dialog);
    buttons->addButton(Tr("Accept and Download"), QDialogButtonBox::AcceptRole);
    buttons->addButton(QDialogButtonBox::Cancel);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    layout->addWidget(intro);
    layout->addWidget(licence_text, 1);
    layout->addWidget(buttons);
    if (dialog.exec() != QDialog::Accepted) {
        return;
    }

    QProgressDialog progress(Tr("Downloading AMD's files..."), Tr("Cancel"), 0, 1000, this);
    progress.setWindowTitle(Fsr4Name());
    progress.setWindowModality(Qt::WindowModal);
    progress.setMinimumDuration(0);
    progress.setValue(0);
    QApplication::processEvents();
    std::error_code ec;
    std::filesystem::create_directories(folder, ec);
    const u64 total = DownloadSize();
    u64 done = 0;
    std::vector<std::pair<std::filesystem::path, std::filesystem::path>> verified;
    const auto discard = [&verified, &ec] {
        for (const auto& [partial, target] : verified) {
            std::filesystem::remove(partial, ec);
        }
    };

    for (const FfxRuntime::File& file : FfxRuntime::FILES) {
        const std::optional<std::string> contents = Fetch(file.entry, error, [&](u64 received) {
            progress.setValue(
                static_cast<int>((done + received) * 1000 / std::max<u64>(total, 1)));
            QApplication::processEvents();
            return !progress.wasCanceled();
        });
        if (progress.wasCanceled()) {
            discard();
            return;
        }
        if (!contents) {
            discard();
            progress.close();
            fail(error);
            return;
        }
        done += file.size;
        const QByteArray sha256 =
            QCryptographicHash::hash(QByteArrayView{contents->data(),
                                                    static_cast<qsizetype>(contents->size())},
                                     QCryptographicHash::Sha256)
                .toHex();
        if (contents->size() != file.size ||
            sha256 != QByteArray{file.sha256.data(), static_cast<qsizetype>(file.sha256.size())}) {
            discard();
            progress.close();
            fail(fmt::format("{} doesn't match the FidelityFX SDK {} file", file.name,
                             FfxRuntime::SDK_VERSION));
            return;
        }
        const std::filesystem::path target = folder / file.name;
        std::filesystem::path partial = target;
        partial += L".download";
        {
            std::ofstream stream(partial, std::ios::binary | std::ios::trunc);
            stream.write(contents->data(), static_cast<std::streamsize>(contents->size()));
            if (!stream) {
                discard();
                progress.close();
                fail("could not write " + Common::FS::PathToUTF8String(partial));
                return;
            }
        }
        if (!FfxRuntime::HasSha256(partial, file.sha256) || !FfxRuntime::IsSignedByAmd(partial)) {
            std::filesystem::remove(partial, ec);
            discard();
            progress.close();
            fail(fmt::format("{} is not signed by AMD", file.name));
            return;
        }
        verified.emplace_back(partial, target);
    }

    std::vector<std::pair<std::filesystem::path, std::filesystem::path>> replaced;
    const auto restore = [&replaced, &ec] {
        for (const auto& [backup, target] : replaced) {
            std::filesystem::remove(target, ec);
            if (!backup.empty()) {
                std::filesystem::rename(backup, target, ec);
            }
        }
    };
    for (const auto& [partial, target] : verified) {
        std::filesystem::path backup;
        if (std::filesystem::exists(target, ec)) {
            backup = target;
            backup += L".old";
            std::filesystem::remove(backup, ec);
            std::filesystem::rename(target, backup, ec);
        }
        bool failed = static_cast<bool>(ec);
        if (!failed) {
            std::filesystem::rename(partial, target, ec);
            failed = static_cast<bool>(ec);
            if (failed && !backup.empty()) {
                std::filesystem::rename(backup, target, ec);
            }
        }
        if (failed) {
            restore();
            discard();
            progress.close();
            fail("could not save " + Common::FS::PathToUTF8String(target));
            return;
        }
        replaced.emplace_back(backup, target);
    }
    for (const auto& [backup, target] : replaced) {
        if (!backup.empty()) {
            std::filesystem::remove(backup, ec);
        }
        LOG_INFO(Frontend, "Downloaded {} from AMD's FidelityFX SDK {}",
                 Common::FS::PathToUTF8String(target), FfxRuntime::SDK_VERSION);
    }
    {
        std::ofstream stream(folder / FfxRuntime::LICENSE_FILE, std::ios::binary | std::ios::trunc);
        stream.write(licence->data(), static_cast<std::streamsize>(licence->size()));
    }
    progress.setValue(1000);
    progress.close();
    Refresh();
    QMessageBox::information(
        this, Fsr4Name(),
        Tr("AMD's %1 runtime is ready. It is used the next time a game starts.").arg(Fsr4Name()));
#endif
}

#endif
