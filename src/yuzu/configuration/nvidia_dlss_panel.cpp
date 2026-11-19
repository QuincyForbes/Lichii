// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#ifdef EGLANTINE_STREAMLINE

#include "common/httplib.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <QApplication>
#include <QDesktopServices>
#include <QDialog>
#include <QDialogButtonBox>
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
#include <zlib.h>

#include "common/common_types.h"
#include "common/fs/fs_util.h"
#include "common/logging.h"
#include "video_core/vulkan_common/streamline.h"
#include "qt_common/config/catalog_text.h"
#include "yuzu/configuration/nvidia_dlss_panel.h"

#include "common/net/ca_store.h"

namespace {

namespace Streamline = Vulkan::Streamline;

constexpr const char* RELEASE_PAGE = "https://github.com/NVIDIA-RTX/Streamline/releases";
constexpr const char* RELEASE_VERSION = "2.14.16";
constexpr const char* ZIP_HOST = "https://github.com";
constexpr const char* ZIP_PATH =
    "/NVIDIA-RTX/Streamline/releases/download/v2.14.16/streamline-sdk-v2.14.16.zip";
constexpr const char* INTERPOSER_ENTRY = "bin/x64/sl.interposer.dll";
constexpr const char* MODEL_ENTRY = "bin/x64/nvngx_dlss.dll";
constexpr const char* STREAMLINE_LICENSE_ENTRY = "license.txt";
constexpr const char* DLSS_LICENSE_ENTRY = "bin/x64/nvngx_dlss.license.txt";
constexpr const char* REFLEX_LICENSE_ENTRY = "bin/x64/reflex.license.txt";
constexpr const char* BIN_DIR = "bin/x64/";

std::string FileName(const std::string& entry) {
    return Common::FS::PathToUTF8String(std::filesystem::path{entry}.filename());
}

QString Tr(const char* text) {
    return QCoreApplication::translate("NvidiaDlssPanel", text);
}

QString DlssName() {
    return UiText::ChoiceLabel("upscaler", "Dlss");
}

QString Fsr3Name() {
    return UiText::ChoiceLabel("upscaler", "Fsr3");
}

u16 U16(const std::string& data, size_t at) {
    return static_cast<u16>(static_cast<u8>(data[at]) | static_cast<u8>(data[at + 1]) << 8);
}

u32 U32(const std::string& data, size_t at) {
    return static_cast<u32>(U16(data, at)) | static_cast<u32>(U16(data, at + 2)) << 16;
}

u64 U64(const std::string& data, size_t at) {
    return static_cast<u64>(U32(data, at)) | static_cast<u64>(U32(data, at + 4)) << 32;
}

struct ZipEntry {
    u16 method{};
    u32 crc{};
    u64 compressed{};
    u64 size{};
    u64 offset{};
};

/// Reads single files out of a zip on a web server with range requests, so only the files
/// needed are downloaded (about 32 MB of the 276 MB release).
class RemoteZip {
public:
    /// Follows GitHub's redirect to the file's (short-lived) download address.
    bool Resolve() {
        host = ZIP_HOST;
        path = ZIP_PATH;
        for (int redirect = 0; redirect < 5; ++redirect) {
            const auto result = MakeClient()->Get(path, httplib::Headers{{"Range", "bytes=0-0"}});
            if (!result) {
                error = httplib::to_string(result.error());
                return false;
            }
            if (result->status == 206) {
                // "bytes 0-0/<size>"; the server doesn't take "last N bytes" ranges.
                const std::string range = result->get_header_value("Content-Range");
                const size_t slash = range.rfind('/');
                zip_size = slash == std::string::npos
                               ? 0
                               : std::strtoull(range.c_str() + slash + 1, nullptr, 10);
                if (zip_size == 0) {
                    error = "no file size";
                    return false;
                }
                return true;
            }
            const std::string location = result->get_header_value("Location");
            const size_t scheme = location.find("://");
            const size_t slash =
                scheme == std::string::npos ? std::string::npos : location.find('/', scheme + 3);
            if (result->status < 300 || result->status >= 400 || slash == std::string::npos) {
                error = fmt::format("HTTP {}", result->status);
                return false;
            }
            host = location.substr(0, slash);
            path = location.substr(slash);
        }
        error = "too many redirects";
        return false;
    }

    bool Open() {
        if (!Resolve()) {
            return false;
        }
        const u64 tail_start = zip_size > 65558 ? zip_size - 65558 : 0;
        const std::optional<std::string> tail =
            Fetch(fmt::format("bytes={}-{}", tail_start, zip_size - 1));
        if (!tail) {
            return false;
        }
        const size_t end = tail->rfind(std::string{"PK\x05\x06", 4});
        if (end == std::string::npos || end + 22 > tail->size()) {
            error = "no zip directory";
            return false;
        }
        u64 directory_size = U32(*tail, end + 12);
        u64 directory_offset = U32(*tail, end + 16);
        if (directory_offset == 0xFFFFFFFF && end >= 20 &&
            tail->compare(end - 20, 4, std::string{"PK\x06\x07", 4}) == 0) {
            const u64 record = U64(*tail, end - 20 + 8);
            const std::optional<std::string> zip64 =
                Fetch(fmt::format("bytes={}-{}", record, record + 55));
            if (!zip64 || zip64->size() < 56) {
                return false;
            }
            directory_size = U64(*zip64, 40);
            directory_offset = U64(*zip64, 48);
        }
        const std::optional<std::string> directory = Fetch(
            fmt::format("bytes={}-{}", directory_offset, directory_offset + directory_size - 1));
        if (!directory) {
            return false;
        }
        for (size_t at = 0; at + 46 <= directory->size() && U32(*directory, at) == 0x02014b50;) {
            const u16 name_length = U16(*directory, at + 28);
            const u16 extra_length = U16(*directory, at + 30);
            const u16 comment_length = U16(*directory, at + 32);
            ZipEntry entry{
                .method = U16(*directory, at + 10),
                .crc = U32(*directory, at + 16),
                .compressed = U32(*directory, at + 20),
                .size = U32(*directory, at + 24),
                .offset = U32(*directory, at + 42),
            };
            const size_t extra_end = at + 46 + name_length + extra_length;
            if (extra_end + comment_length > directory->size()) {
                error = "bad zip directory";
                return false;
            }
            const std::string name = directory->substr(at + 46, name_length);
            // Zip64 sizes and offset, in this order, for the fields that overflowed.
            for (size_t extra = at + 46 + name_length; extra + 4 <= extra_end;) {
                const u16 id = U16(*directory, extra);
                const u16 length = U16(*directory, extra + 2);
                const size_t record_end = std::min<size_t>(extra + 4 + length, extra_end);
                if (id == 0x0001) {
                    size_t field = extra + 4;
                    for (u64* value : {&entry.size, &entry.compressed, &entry.offset}) {
                        if (*value == 0xFFFFFFFF && field + 8 <= record_end) {
                            *value = U64(*directory, field);
                            field += 8;
                        }
                    }
                }
                extra += 4 + length;
            }
            entries.emplace(name, entry);
            at += 46 + name_length + extra_length + comment_length;
        }
        return true;
    }

    const ZipEntry* Find(const std::string& name) const {
        const auto it = entries.find(name);
        return it == entries.end() ? nullptr : &it->second;
    }

    std::optional<std::string> Read(const ZipEntry& entry,
                                    const std::function<bool(u64)>& progress = {}) {
        const std::optional<std::string> header =
            Fetch(fmt::format("bytes={}-{}", entry.offset, entry.offset + 29));
        if (!header) {
            return std::nullopt;
        }
        if (header->size() < 30 || U32(*header, 0) != 0x04034b50) {
            error = "bad zip entry";
            return std::nullopt;
        }
        const u64 start = entry.offset + 30 + U16(*header, 26) + U16(*header, 28);
        std::string data;
        if (entry.compressed != 0) {
            std::optional<std::string> fetched =
                Fetch(fmt::format("bytes={}-{}", start, start + entry.compressed - 1), progress);
            if (!fetched) {
                return std::nullopt;
            }
            data = std::move(*fetched);
        }
        std::string out;
        if (entry.method == 0) {
            out = std::move(data);
        } else if (entry.method == 8) {
            out.resize(entry.size);
            z_stream stream{};
            if (inflateInit2(&stream, -MAX_WBITS) != Z_OK) {
                error = "zlib";
                return std::nullopt;
            }
            stream.next_in = reinterpret_cast<Bytef*>(data.data());
            stream.avail_in = static_cast<uInt>(data.size());
            stream.next_out = reinterpret_cast<Bytef*>(out.data());
            stream.avail_out = static_cast<uInt>(out.size());
            const int result = inflate(&stream, Z_FINISH);
            inflateEnd(&stream);
            if (result != Z_STREAM_END || stream.total_out != entry.size) {
                error = "could not unpack";
                return std::nullopt;
            }
        } else {
            error = "unknown zip compression";
            return std::nullopt;
        }
        const uLong crc = crc32(0L, reinterpret_cast<const Bytef*>(out.data()),
                                static_cast<uInt>(out.size()));
        if (crc != entry.crc) {
            error = "damaged download (checksum)";
            return std::nullopt;
        }
        return out;
    }

    bool canceled{};
    std::string error;

private:
    std::unique_ptr<httplib::Client> MakeClient() const {
        auto client = std::make_unique<httplib::Client>(host);
        client->set_connection_timeout(30);
        client->set_read_timeout(30);
        client->set_write_timeout(30);
        Common::Net::ConfigureCaStore(*client);
        return client;
    }

    std::optional<std::string> Fetch(const std::string& range,
                                     const std::function<bool(u64)>& progress = {}) {
        std::string body;
        int status = 0;
        const auto result = MakeClient()->Get(
            path, httplib::Headers{{"Range", range}},
            [&status](const httplib::Response& response) {
                status = response.status;
                return response.status == 206;
            },
            [&](const char* data, size_t length) {
                body.append(data, length);
                if (progress && !progress(body.size())) {
                    canceled = true;
                    return false;
                }
                return true;
            });
        if (!result || status != 206) {
            error = result ? fmt::format("HTTP {}", status) : httplib::to_string(result.error());
            return std::nullopt;
        }
        return body;
    }

    std::string host;
    std::string path;
    u64 zip_size{};
    std::map<std::string, ZipEntry> entries;
};

QString Named(const QString& name, const std::string& version) {
    return version.empty() ? name : QStringLiteral("%1 %2").arg(name, QString::fromStdString(version));
}

QString PathText(const std::filesystem::path& path) {
    return QString::fromStdWString(path.wstring());
}

} // Anonymous namespace

NvidiaDlssPanel::NvidiaDlssPanel(QWidget* parent) : QWidget(parent) {
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
    download_button = new QPushButton(Tr("Download from NVIDIA..."), this);
    folder_button = new QPushButton(Tr("Open Folder"), this);
    QLabel* about = new QLabel(
        QStringLiteral("<a href=\"%1\">%2</a>").arg(QString::fromLatin1(RELEASE_PAGE),
                                                   Tr("Where do these files come from?")),
        this);
    about->setOpenExternalLinks(true);
    about->setToolTip(Tr("NVIDIA's Streamline releases on GitHub. The files are in bin\\x64 of "
                         "the release zip."));
    button_row->addWidget(download_button);
    button_row->addWidget(folder_button);
    button_row->addStretch(1);
    button_row->addWidget(about);

    layout->addLayout(status_row);
    layout->addLayout(button_row);

    // While a game runs, the frame count and any failure show up without reopening the settings.
    refresh_timer = new QTimer(this);
    refresh_timer->setInterval(1000);
    connect(refresh_timer, &QTimer::timeout, this, [this] { Refresh(); });

    connect(download_button, &QPushButton::clicked, this, [this] { Download(); });
    connect(folder_button, &QPushButton::clicked, this, [] {
        const std::filesystem::path folder = Streamline::FindDlssFiles().folder;
        std::error_code ec;
        std::filesystem::create_directories(folder, ec);
        QDesktopServices::openUrl(QUrl::fromLocalFile(PathText(folder)));
    });
    Refresh();
}

void NvidiaDlssPanel::showEvent(QShowEvent* event) {
    Refresh();
    refresh_timer->start();
    QWidget::showEvent(event);
}

void NvidiaDlssPanel::hideEvent(QHideEvent* event) {
    refresh_timer->stop();
    QWidget::hideEvent(event);
}

void NvidiaDlssPanel::SetDevice(const QString& name, bool capable) {
    device_name = name;
    device_capable = capable;
    Refresh();
}

void NvidiaDlssPanel::SetFrameGenWanted(bool wanted) {
    if (frame_gen_wanted == wanted) {
        return;
    }
    frame_gen_wanted = wanted;
    Refresh();
}

void NvidiaDlssPanel::SetUpscalerSelected(bool selected) {
    if (upscaler_selected == selected) {
        return;
    }
    upscaler_selected = selected;
    Refresh();
}

void NvidiaDlssPanel::Refresh() {
    const Streamline::DlssFiles files = Streamline::FindDlssFiles();
    const bool has_loader = !files.interposer.empty();
    const bool has_model = !files.model.empty() || !files.app_model.empty();
    const bool has_plugins = files.missing_plugins.empty();
    const bool sr_ready = has_loader && has_model && has_plugins;
    const bool has_frame_gen =
        !frame_gen_wanted || (files.frame_gen_missing.empty() && !files.streamline_mixed);
    const bool ready = sr_ready && has_frame_gen;
    const QString dlss = !files.model.empty()
                             ? Named(QStringLiteral("DLSS"), files.model_version)
                             : Tr("%1 from the NVIDIA app")
                                   .arg(Named(QStringLiteral("DLSS"), files.app_model_version));

    QStringList lines;
    if (!device_name.isEmpty()) {
        lines << Tr("GPU: %1").arg(device_name.toHtmlEscaped());
    }
    if (!device_capable) {
        lines << Tr("<b>GPU not supported:</b> %1 requires an NVIDIA GeForce RTX GPU on Vulkan "
                    "(%2 runs instead).")
                     .arg(DlssName().toHtmlEscaped(), Fsr3Name().toHtmlEscaped());
    }
    if (sr_ready) {
        lines << Tr("<b>Files ready:</b> %1 and %2.")
                     .arg(Named(QStringLiteral("NVIDIA Streamline"), files.interposer_version),
                          dlss);
    } else {
        QStringList missing;
        if (!has_loader) {
            missing << QStringLiteral("sl.interposer.dll");
        }
        for (const std::filesystem::path& plugin : files.missing_plugins) {
            missing << PathText(plugin.filename());
        }
        if (!has_model) {
            missing << QStringLiteral("nvngx_dlss.dll");
        }
        lines << Tr("<b>NVIDIA's files needed:</b> %1. Until then games use %2.")
                     .arg(missing.join(QStringLiteral(", ")), Fsr3Name().toHtmlEscaped());
    }
    if (frame_gen_wanted) {
        QStringList fg_missing;
        for (const std::string& name : files.frame_gen_missing) {
            fg_missing << QString::fromStdString(name);
        }
        if (!fg_missing.isEmpty()) {
            lines << Tr("<b>DLSS Frame Generation needed:</b> %1.")
                         .arg(fg_missing.join(QStringLiteral(", ")));
        } else if (files.streamline_mixed) {
            lines << Tr("<b>DLSS Frame Generation:</b> NVIDIA's Streamline files (sl.*) in the "
                        "folder come from different releases; download a matching set.");
        } else {
            lines << Tr("<b>DLSS Frame Generation:</b> files ready.");
        }
    }
    status_label->setText(lines.join(QStringLiteral("<br>")));
    status_label->setToolTip(Tr("Folder: %1").arg(PathText(files.folder)));

    // What DLSS and DLSS Frame Generation did in the running (or last) game.
    const Streamline::SessionStatus session = Streamline::GetSessionStatus();
    const QString version = Named(QStringLiteral("DLSS"), session.version);
    QString mode_detail;
    if (!session.mode_name.empty() && !session.preset_name.empty()) {
        mode_detail = QStringLiteral(" (%1, %2)")
                          .arg(QString::fromStdString(session.mode_name).toHtmlEscaped(),
                               Tr("preset %1").arg(
                                   QString::fromStdString(session.preset_name).toHtmlEscaped()));
    } else if (!session.mode_name.empty()) {
        mode_detail = QStringLiteral(" (%1)").arg(
            QString::fromStdString(session.mode_name).toHtmlEscaped());
    }
    const QString covered =
        Tr("Games it covers: The Legend of Zelda: Ocarina of Time (at 1x resolution scale) and "
           "Fire Emblem: Fortune's Weave. Other games keep their own upscaling.");
    QStringList session_lines;
    bool active = false;
    if (session.available && session.frames > 0) {
        active = true;
        session_lines << Tr("<b>Active:</b> %1%2 has upscaled %3 frames in this game.")
                             .arg(version, mode_detail)
                             .arg(session.frames);
    } else if (session.available) {
        session_lines << Tr("<b>Running</b>, but this game hasn't drawn a pass DLSS can replace. "
                            "%1")
                             .arg(covered);
    } else if (!session.problem.empty()) {
        session_lines << Tr("<b>Not used in the last game:</b> %1")
                             .arg(QString::fromStdString(session.problem).toHtmlEscaped());
    } else if (session.frames > 0) {
        session_lines << Tr("Last game: %1%2 upscaled %3 frames. %4")
                             .arg(version, mode_detail)
                             .arg(session.frames)
                             .arg(covered);
    } else if (!upscaler_selected) {
        session_lines << Tr("Select <b>%1</b> in %2 above to enable it for the next game. %3")
                             .arg(DlssName().toHtmlEscaped(),
                                  UiText::SettingLabel("upscaler").toHtmlEscaped(), covered);
    } else {
        session_lines << Tr("Starts with the next game. %1").arg(covered);
    }

    if (session.frame_gen_active && session.generated_frames > 0) {
        active = true;
        session_lines << Tr("<b>Frame Generation active:</b> +%1 generated frames.")
                             .arg(session.generated_frames);
    } else if (session.frame_gen_active) {
        session_lines << Tr("<b>Frame Generation ready</b> (pauses on menus and static scenes).");
    } else if (frame_gen_wanted && !session.frame_gen_problem.empty()) {
        session_lines << Tr("<b>Frame Generation not used:</b> %1")
                             .arg(QString::fromStdString(session.frame_gen_problem).toHtmlEscaped());
    }
    session_label->setText(session_lines.join(QStringLiteral("<br>")));

    const bool problem = !device_capable || !ready ||
                         (!session.available && !session.problem.empty()) ||
                         (frame_gen_wanted && !session.frame_gen_active &&
                          !session.frame_gen_problem.empty());
    icon_label->setText(
        problem ? QStringLiteral("<span style=\"color:#d29922\">&#9888;</span>")
        : active ? QStringLiteral("<span style=\"color:#2ea043\">&#9679;</span>")
                 : QStringLiteral("<span style=\"color:#2ea043\">&#10004;</span>"));
    download_button->setText(sr_ready && !has_frame_gen
                                 ? Tr("Download Frame Generation files from NVIDIA...")
                                 : Tr("Download from NVIDIA..."));
    download_button->setVisible(!ready && device_capable);
}

void NvidiaDlssPanel::Download() {
    const Streamline::DlssFiles files = Streamline::FindDlssFiles();
    std::vector<std::string> wanted;
    // Frame generation's files come as one set with the loader, all from this release: Streamline
    // loads a module only with the sl.common.dll and loader of its own release.
    std::vector<std::string> frame_gen_set;
    if (frame_gen_wanted && (!files.frame_gen_missing.empty() || files.streamline_mixed)) {
        frame_gen_set.emplace_back(INTERPOSER_ENTRY);
        for (const char* const name : Streamline::FrameGenFiles()) {
            frame_gen_set.push_back(BIN_DIR + std::string{name});
        }
    }
    const auto in_frame_gen_set = [&frame_gen_set](const std::string& entry) {
        return std::ranges::find(frame_gen_set, entry) != frame_gen_set.end();
    };
    const auto add_unique = [&wanted](std::string entry) {
        if (std::ranges::find(wanted, entry) == wanted.end()) {
            wanted.push_back(std::move(entry));
        }
    };
    if (files.interposer.empty() && !in_frame_gen_set(INTERPOSER_ENTRY)) {
        add_unique(INTERPOSER_ENTRY);
    }
    for (const std::filesystem::path& plugin : files.missing_plugins) {
        const std::string entry =
            fmt::format("{}{}", BIN_DIR, Common::FS::PathToUTF8String(plugin.filename()));
        if (!in_frame_gen_set(entry)) {
            add_unique(entry);
        }
    }
    if (files.model.empty() && files.app_model.empty()) {
        add_unique(MODEL_ENTRY);
    }
    for (const std::string& entry : frame_gen_set) {
        add_unique(entry);
    }
    if (wanted.empty()) {
        return;
    }
    const std::filesystem::path folder = files.folder;
    const auto fail = [this](const std::string& error) {
        LOG_ERROR(Frontend, "NVIDIA DLSS download failed: {}", error);
        QMessageBox::warning(this, DlssName(),
                             Tr("Could not download NVIDIA's files: %1")
                                 .arg(QString::fromStdString(error)));
        Refresh();
    };

    RemoteZip zip;
    std::string licences;
    {
        QProgressDialog busy(Tr("Contacting NVIDIA's download..."), QString{}, 0, 0, this);
        busy.setWindowTitle(DlssName());
        busy.setWindowModality(Qt::WindowModal);
        busy.setMinimumDuration(0);
        busy.show();
        QApplication::processEvents();
        if (!zip.Open()) {
            busy.close();
            fail(zip.error);
            return;
        }
        const auto wants = [&wanted](const std::string& name) {
            return std::ranges::find(wanted, BIN_DIR + name) != wanted.end();
        };
        std::vector<std::pair<QString, const char*>> texts{
            {Tr("NVIDIA Streamline"), STREAMLINE_LICENSE_ENTRY}};
        if (std::ranges::find(wanted, MODEL_ENTRY) != wanted.end() || wants("nvngx_dlssg.dll")) {
            texts.emplace_back(DlssName(), DLSS_LICENSE_ENTRY);
        }
        if (wants("sl.reflex.dll") || wants("NvLowLatencyVk.dll")) {
            texts.emplace_back(Tr("NVIDIA Reflex"), REFLEX_LICENSE_ENTRY);
        }
        for (const auto& [title, entry_name] : texts) {
            const ZipEntry* const entry = zip.Find(entry_name);
            const std::optional<std::string> text = entry ? zip.Read(*entry) : std::nullopt;
            if ((!text || text->empty()) && entry_name == REFLEX_LICENSE_ENTRY) {
                LOG_WARNING(Frontend, "NVIDIA's release has no {}", entry_name);
                continue;
            }
            if (!text || text->empty()) {
                busy.close();
                fail("could not retrieve NVIDIA's licence text");
                return;
            }
            licences += fmt::format("==== {} ====\n\n{}\n\n", title.toStdString(), *text);
        }
    }

    // Without every file of the frame generation set in the release, none of it is taken.
    if (const auto missing = std::ranges::find_if(
            frame_gen_set, [&zip](const std::string& name) { return !zip.Find(name); });
        missing != frame_gen_set.end()) {
        LOG_WARNING(Frontend,
                    "NVIDIA's Streamline {} release has no {}; skipping DLSS Frame Generation's "
                    "files",
                    RELEASE_VERSION, *missing);
        std::erase_if(wanted, in_frame_gen_set);
        frame_gen_set.clear();
    }
    if (wanted.empty()) {
        fail("the release has none of the missing files");
        return;
    }

    u64 total = 0;
    std::vector<const ZipEntry*> entries;
    QStringList names;
    for (const std::string& name : wanted) {
        const ZipEntry* const entry = zip.Find(name);
        if (!entry) {
            fail(name + " is not in the release");
            return;
        }
        entries.push_back(entry);
        total += entry->compressed;
        names << QString::fromStdString(FileName(name));
    }
    QDialog dialog(this);
    dialog.setWindowTitle(Tr("Download %1").arg(DlssName()));
    QVBoxLayout* layout = new QVBoxLayout(&dialog);
    QLabel* intro = new QLabel(
        Tr("Lichii will download %1 (%2 MB) from NVIDIA's Streamline %3 release on GitHub into "
           "%4, and checks NVIDIA's signature on each file before using it.<br>"
           "These files are NVIDIA's, under NVIDIA's licence terms below.")
            .arg(names.join(QStringLiteral(", ")))
            .arg(std::max<u64>(total / (1024 * 1024), 1))
            .arg(QString::fromLatin1(RELEASE_VERSION), PathText(folder)),
        &dialog);
    intro->setWordWrap(true);
    QPlainTextEdit* licence_text = new QPlainTextEdit(QString::fromStdString(licences), &dialog);
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

    QProgressDialog progress(Tr("Downloading NVIDIA's DLSS files..."), Tr("Cancel"), 0, 1000,
                             this);
    progress.setWindowTitle(DlssName());
    progress.setWindowModality(Qt::WindowModal);
    progress.setMinimumDuration(0);
    progress.setValue(0);
    QApplication::processEvents();
    // The download address from the first step only lasts a few minutes.
    if (!zip.Resolve()) {
        progress.close();
        fail(zip.error);
        return;
    }
    std::error_code ec;
    std::filesystem::create_directories(folder, ec);
    u64 done = 0;
    // Every file is checked before any is replaced; frame generation's set is saved whole or not
    // at all, so the loader's folder never mixes releases.
    std::vector<std::pair<std::filesystem::path, std::filesystem::path>> verified;
    const auto discard = [&verified, &ec] {
        for (const auto& [partial, target] : verified) {
            std::filesystem::remove(partial, ec);
        }
    };
    QStringList skipped;
    for (size_t i = 0; i < wanted.size(); ++i) {
        progress.setLabelText(Tr("Downloading %1 (%2 of %3) from NVIDIA...")
                                  .arg(names[static_cast<qsizetype>(i)])
                                  .arg(i + 1)
                                  .arg(wanted.size()));
        const std::optional<std::string> contents = zip.Read(*entries[i], [&](u64 received) {
            progress.setValue(
                static_cast<int>((done + received) * 1000 / std::max<u64>(total, 1)));
            QApplication::processEvents();
            return !progress.wasCanceled();
        });
        if (!contents) {
            discard();
            progress.close();
            if (!zip.canceled) {
                fail(zip.error);
            }
            return;
        }
        done += entries[i]->compressed;
        const std::filesystem::path target = folder / std::filesystem::path{wanted[i]}.filename();
        std::filesystem::path partial = target;
        partial += L".download";
        {
            std::ofstream file(partial, std::ios::binary | std::ios::trunc);
            file.write(contents->data(), static_cast<std::streamsize>(contents->size()));
            if (!file) {
                discard();
                progress.close();
                fail("could not write " + Common::FS::PathToUTF8String(partial));
                return;
            }
        }
        // Streamline's own modules (sl.*) carry Streamline's signature; NVIDIA's other DLLs an
        // Authenticode one.
        const bool streamline_module = FileName(wanted[i]).starts_with("sl.");
        const bool signed_ok = streamline_module ? Streamline::IsStreamlineModule(partial)
                                                 : Streamline::IsSignedByNvidia(partial);
        if (!signed_ok && in_frame_gen_set(wanted[i])) {
            std::filesystem::remove(partial, ec);
            LOG_WARNING(Frontend, "{} from NVIDIA's release is not signed by NVIDIA; skipping it",
                        FileName(wanted[i]));
            skipped << QString::fromStdString(FileName(wanted[i]));
            continue;
        }
        if (!signed_ok) {
            std::filesystem::remove(partial, ec);
            discard();
            progress.close();
            fail(Common::FS::PathToUTF8String(target.filename()) + " is not signed by NVIDIA");
            return;
        }
        verified.emplace_back(partial, target);
    }
    if (!skipped.isEmpty()) {
        std::erase_if(verified, [&](const auto& file) {
            if (!in_frame_gen_set(BIN_DIR + Common::FS::PathToUTF8String(file.second.filename()))) {
                return false;
            }
            std::filesystem::remove(file.first, ec);
            return true;
        });
    }
    // The old files move aside first, so a failed save puts every one of them back.
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
        LOG_INFO(Frontend, "Downloaded {} from NVIDIA's Streamline {} release",
                 Common::FS::PathToUTF8String(target), RELEASE_VERSION);
    }
    progress.setValue(1000);
    progress.close();
    Refresh();
    if (!skipped.isEmpty()) {
        QMessageBox::warning(this, DlssName(),
                             Tr("These files failed NVIDIA's signature check, so none of DLSS "
                                "Frame Generation's files were saved and it stays off: %1")
                                 .arg(skipped.join(QStringLiteral(", "))));
        return;
    }
    QMessageBox::information(
        this, DlssName(),
        Tr("%1 is ready. It is used the next time a game starts.").arg(DlssName()));
}

#endif
