// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "yuzu/util/export_log.h"

#include <filesystem>
#include <thread>
#include <utility>
#include <vector>

#include <QApplication>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QGuiApplication>
#include <QMap>
#include <QMessageBox>
#include <QProcess>
#include <QProcessEnvironment>
#include <QRegularExpression>
#include <QScreen>
#include <QSet>
#include <QStandardPaths>
#include <QSysInfo>
#include <QUrl>
#include <quazip.h>
#include <quazipfile.h>
#include <quazipnewinfo.h>
#include <zlib.h>

#include "common/cpu_features.h"
#include "common/frizti_settings.h"
#include "common/fs/fs_paths.h"
#include "common/fs/fs_util.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/memory_detect.h"
#include "common/scm_rev.h"
#include "common/settings.h"
#include "qt_common/qt_common.h"
#include "qt_common/util/vk.h"

namespace ExportLog {
namespace {

// Logs can grow up to 100 MiB (or 1 GiB with extended logging). Above MaxLogBytes, keep the
// startup header and the tail where the failure occurred.
constexpr qint64 MaxLogBytes = 64LL * 1024 * 1024;
constexpr qint64 HeadBytes = 8LL * 1024 * 1024;

/// Replaces the OS user name and machine name wherever they appear as whole words so file paths
/// are scrubbed without garbling unrelated substrings.
class Scrubber {
public:
    Scrubber() {
        const auto env = QProcessEnvironment::systemEnvironment();
        Add(env.value(QStringLiteral("USERNAME")), QStringLiteral("user"));
        Add(env.value(QStringLiteral("USER")), QStringLiteral("user"));
        Add(env.value(QStringLiteral("LOGNAME")), QStringLiteral("user"));
        Add(QDir(QDir::homePath()).dirName(), QStringLiteral("user"));
        Add(env.value(QStringLiteral("COMPUTERNAME")), QStringLiteral("PC"));
        Add(QSysInfo::machineHostName(), QStringLiteral("PC"));
    }

    QString Apply(QString text) const {
        for (const auto& [pattern, replacement] : rules) {
            text.replace(pattern, replacement);
        }
        return text;
    }

private:
    void Add(const QString& word, const QString& replacement) {
        if (word.size() < 2) {
            return;
        }
        for (const auto& rule : rules) {
            if (rule.first.match(word).hasMatch()) {
                return;
            }
        }
        rules.emplace_back(QRegularExpression(QStringLiteral("(?<![A-Za-z0-9])%1(?![A-Za-z0-9])")
                                                  .arg(QRegularExpression::escape(word)),
                                              QRegularExpression::CaseInsensitiveOption),
                           replacement);
    }

    std::vector<std::pair<QRegularExpression, QString>> rules;
};

bool IsSensitiveIniKey(const QString& raw_key) {
    const QString lower = raw_key.toLower();
    return lower.contains(QStringLiteral("token")) || lower.contains(QStringLiteral("password")) ||
           lower.contains(QStringLiteral("secret")) || lower.contains(QStringLiteral("api_key")) ||
           lower.contains(QStringLiteral("eden_username")) ||
           lower.contains(QStringLiteral("nickname"));
}

QString SanitizeIni(const QString& content, const Scrubber& scrubber) {
    QStringList out_lines;
    const QStringList lines = content.split(QLatin1Char('\n'));
    out_lines.reserve(lines.size());
    for (QString line : lines) {
        const bool had_cr = line.endsWith(QLatin1Char('\r'));
        if (had_cr) {
            line.chop(1);
        }
        const QString trimmed = line.trimmed();
        if (!trimmed.isEmpty() && !trimmed.startsWith(QLatin1Char('#')) &&
            !trimmed.startsWith(QLatin1Char(';')) && !trimmed.startsWith(QLatin1Char('['))) {
            const qsizetype eq = line.indexOf(QLatin1Char('='));
            if (eq > 0) {
                const QString key = line.left(eq).trimmed();
                const QString val = line.mid(eq + 1).trimmed();
                if (!val.isEmpty() && IsSensitiveIniKey(key)) {
                    line = line.left(eq + 1) + QStringLiteral("<redacted>");
                }
            }
        }
        out_lines.push_back(scrubber.Apply(line));
    }
    return out_lines.join(QLatin1Char('\n'));
}

QString ReadLog(const QString& path, qint64* size_out) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        return {};
    }
    const qint64 size = file.size();
    *size_out = size;
    if (size <= 0) {
        return {};
    }
    QByteArray bytes;
    if (size <= MaxLogBytes) {
        bytes = file.readAll();
    } else {
        bytes = file.read(HeadBytes);
        bytes += QStringLiteral("\n\n[... %1 MB cut from the middle of this log to keep the zip "
                                "small ...]\n\n")
                     .arg((size - MaxLogBytes) / (1024 * 1024))
                     .toUtf8();
        file.seek(size - (MaxLogBytes - HeadBytes));
        bytes += file.readAll();
    }
    return QString::fromUtf8(bytes);
}

QString Line(const QString& name, const QString& value) {
    return QStringLiteral("%1: %2\n").arg(name, value);
}

QString Gigabytes(u64 bytes) {
    return QStringLiteral("%1 GB").arg(static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0), 0,
                                       'f', 1);
}

QString SystemSummary(const std::vector<VkDeviceInfo::Record>& vk_device_records,
                      const QStringList& included) {
    QString text;
    text += QStringLiteral("Lichii log export, made %1\n\n")
                .arg(QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd HH:mm")));

    text += QStringLiteral("== Lichii ==\n");
    text += Line(QStringLiteral("Version"), QString::fromStdString(QtCommon::BuildVersionString()));
    text += Line(QStringLiteral("Build"), QStringLiteral("%1, %2, %3")
                                              .arg(QString::fromUtf8(Common::g_build_fullname),
                                                   QString::fromUtf8(Common::g_build_date),
                                                   QString::fromUtf8(Common::g_compiler_id)));
    text += Line(QStringLiteral("Tuned settings revision"),
                 QStringLiteral("%1 (applied: %2)")
                     .arg(Settings::Frizti::Revision)
                     .arg(Settings::values.frizti_settings_revision.GetValue()));
    auto data_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::EdenDir);
    if (!data_dir.has_filename()) {
        data_dir = data_dir.parent_path();
    }
    text += Line(QStringLiteral("Portable"),
                 data_dir.filename() == "user" ? QStringLiteral("yes") : QStringLiteral("no"));

    text += QStringLiteral("\n== System ==\n");
    text += Line(QStringLiteral("OS"), QtCommon::PrettyProductName());
#ifdef ARCHITECTURE_x86_64
    const auto& caps = Common::g_cpu_caps;
    QString cpu = QString::fromUtf8(caps.cpu_string).trimmed();
    cpu += caps.avx512f ? QStringLiteral(" | AVX-512")
           : caps.avx2  ? QStringLiteral(" | AVX2")
           : caps.avx   ? QStringLiteral(" | AVX")
                        : QString{};
    text += Line(QStringLiteral("CPU"), cpu);
#endif
#ifdef ARCHITECTURE_x86_64
    const auto cores = Common::GetProcessorCount();
#else
    const std::optional<int> cores{};
#endif
    text += Line(QStringLiteral("CPU cores / threads"),
                 QStringLiteral("%1 / %2")
                     .arg(cores ? QString::number(*cores) : QStringLiteral("?"))
                     .arg(std::thread::hardware_concurrency()));
    const auto& memory = Common::GetMemInfo();
    text += Line(QStringLiteral("RAM"), QStringLiteral("%1 (swap %2)")
                                            .arg(Gigabytes(memory.TotalPhysicalMemory),
                                                 Gigabytes(memory.TotalSwapMemory)));
    const u32 selected_gpu = Settings::values.vulkan_device.GetValue();
    for (size_t i = 0; i < vk_device_records.size(); ++i) {
        const auto& record = vk_device_records[i];
        QString gpu = QString::fromStdString(record.name);
        if (!record.driver_version.empty()) {
            gpu += QStringLiteral(", driver %1").arg(QString::fromStdString(record.driver_version));
        }
        text += Line(i == selected_gpu ? QStringLiteral("GPU (used)") : QStringLiteral("GPU"), gpu);
    }
    if (vk_device_records.empty()) {
        text += Line(QStringLiteral("GPU"), QStringLiteral("not enumerated yet or unavailable"));
    }
    const auto screens = QGuiApplication::screens();
    for (const QScreen* screen : screens) {
        const QSize pixels = screen->size() * screen->devicePixelRatio();
        text += Line(screen == QGuiApplication::primaryScreen() ? QStringLiteral("Screen (main)")
                                                                : QStringLiteral("Screen"),
                     QStringLiteral("%1x%2 at %3 Hz, scale %4%")
                         .arg(pixels.width())
                         .arg(pixels.height())
                         .arg(screen->refreshRate(), 0, 'f', 0)
                         .arg(qRound(screen->devicePixelRatio() * 100)));
    }

    text += QStringLiteral("\n== Settings ==\n");
    auto& values = Settings::values;
    std::vector<Settings::BasicSetting*> settings{
        &values.renderer_backend,  &values.vulkan_device,     &values.oot_frame_rate,
        &values.frame_rate_limit,  &values.use_speed_limit,   &values.speed_limit,
        &values.nvidia_dlss,       &values.oot_resolution,    &values.oot_anti_aliasing,
        &values.oot_movie_upscale, &values.frame_gen,         &values.frame_gen_type,
        &values.precache_shaders,  &values.accelerate_astc,   &values.astc_recompression,
        &values.texture_quality,   &values.mkw_frame_rate,
    };
#ifdef __APPLE__
    settings.push_back(&values.moltenvk_driver);
#endif
    for (auto* setting : Settings::Frizti::TunedSettings(values)) {
        settings.push_back(setting);
    }
    for (const auto* setting : settings) {
        text += Line(QString::fromStdString(setting->GetLabel()),
                     QString::fromStdString(setting->Canonicalize()));
    }

    // Ocarina of Time per-game overrides (Configure Game values with use_global=false).
    text += QStringLiteral("\n== OoT game-specific overrides (custom/0400486024DEC000.ini) ==\n");
    {
        const auto game_ini = Common::FS::GetEdenPath(Common::FS::EdenPath::ConfigDir) / "custom" /
                              "0400486024DEC000.ini";
        QFile file(QString::fromStdString(Common::FS::PathToUTF8String(game_ini)));
        QStringList overridden;
        if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            QString section;
            QSet<QString> own;
            QMap<QString, QString> values_by_key;
            while (!file.atEnd()) {
                const QString line = QString::fromUtf8(file.readLine()).trimmed();
                if (line.startsWith(QLatin1Char('[')) && line.endsWith(QLatin1Char(']'))) {
                    section = line.mid(1, line.size() - 2);
                    continue;
                }
                const qsizetype eq = line.indexOf(QLatin1Char('='));
                if (eq <= 0) {
                    continue;
                }
                const QString key = line.left(eq);
                const QString value = line.mid(eq + 1);
                if (key.endsWith(QStringLiteral("\\use_global"))) {
                    if (value == QStringLiteral("false")) {
                        own.insert(section + QLatin1Char('/') + key.chopped(11));
                    }
                } else if (!key.contains(QLatin1Char('\\'))) {
                    values_by_key.insert(section + QLatin1Char('/') + key, value);
                }
            }
            for (const QString& key : own) {
                overridden << QStringLiteral("%1: %2").arg(key, values_by_key.value(key));
            }
            overridden.sort();
        }
        text += overridden.isEmpty() ? QStringLiteral("none") : overridden.join(QLatin1Char('\n'));
        text += QLatin1Char('\n');
    }

    QStringList switches;
    const auto env = QProcessEnvironment::systemEnvironment();
    for (const QString& key : env.keys()) {
        if (key.startsWith(QStringLiteral("EDEN_")) || key.startsWith(QStringLiteral("LICHII_"))) {
            switches << QStringLiteral("%1=%2").arg(key, env.value(key));
        }
    }
    switches.sort();
    text += QStringLiteral("\n== Switches (environment) ==\n");
    text += switches.isEmpty() ? QStringLiteral("none") : switches.join(QLatin1Char('\n'));
    text += QLatin1Char('\n');

    text += QStringLiteral("\n== In this zip ==\n");
    text += included.join(QLatin1Char('\n'));
    text += QLatin1Char('\n');
    text += QStringLiteral("\nThe user name, machine name, and account tokens were redacted "
                           "everywhere in this zip.\n");
    return text;
}

bool AddFile(QuaZip& zip, const QString& name, const QByteArray& data) {
    QuaZipFile out(&zip);
    if (!out.open(QIODevice::WriteOnly, QuaZipNewInfo(name), nullptr, 0, Z_DEFLATED,
                  Z_DEFAULT_COMPRESSION)) {
        return false;
    }
    const bool written = out.write(data) == static_cast<qint64>(data.size());
    out.close();
    return written && out.getZipError() == ZIP_OK;
}

void ShowInFileManager(const QString& path) {
#ifdef _WIN32
    if (QProcess::startDetached(QStringLiteral("explorer.exe"),
                                {QStringLiteral("/select,"), QDir::toNativeSeparators(path)})) {
        return;
    }
#endif
    QDesktopServices::openUrl(QUrl::fromLocalFile(QFileInfo(path).absolutePath()));
}

} // namespace

void Export(QWidget* parent, const std::vector<VkDeviceInfo::Record>& vk_device_records) {
    QString default_dir = QStandardPaths::writableLocation(QStandardPaths::DesktopLocation);
    if (default_dir.isEmpty()) {
        default_dir = QDir::homePath();
    }
    const QString default_name =
        QStringLiteral("Lichii-log-%1.zip")
            .arg(QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd-HHmm")));
    QString path = QFileDialog::getSaveFileName(
        parent, QObject::tr("Export Log"), QDir(default_dir).filePath(default_name),
        QObject::tr("Zip file (*.zip)"));
    if (path.isEmpty()) {
        return;
    }
    if (!path.endsWith(QStringLiteral(".zip"), Qt::CaseInsensitive)) {
        path += QStringLiteral(".zip");
    }

    QApplication::setOverrideCursor(Qt::WaitCursor);

    // Flush buffered log output before reading eden_log.txt.
    Common::Log::Stop();

    const QString log_dir = QString::fromStdString(
        Common::FS::PathToUTF8String(Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir)));
    const QString current_log = QDir(log_dir).filePath(QStringLiteral(LOG_FILE));
    QString previous_log = current_log + QStringLiteral(".old.txt");
    if (!QFileInfo::exists(previous_log) &&
        QFileInfo::exists(current_log + QStringLiteral(".old"))) {
        previous_log = current_log + QStringLiteral(".old");
    }

    const Scrubber scrubber;
    QStringList included;
    std::vector<std::pair<QString, QByteArray>> files;

    const auto add_log = [&](const QString& source, const QString& zip_name, const QString& what) {
        qint64 size = 0;
        const QString text = ReadLog(source, &size);
        if (size <= 0) {
            return;
        }
        files.emplace_back(zip_name, scrubber.Apply(text).toUtf8());
        included << QStringLiteral("%1: %2 (%3 KB)").arg(zip_name, what).arg((size + 1023) / 1024);
    };
    add_log(current_log, QStringLiteral(LOG_FILE), QObject::tr("current session log"));
    add_log(previous_log, QStringLiteral(LOG_FILE ".old"), QObject::tr("previous session log"));

    const auto add_ini = [&](const std::filesystem::path& source_path, const QString& zip_name,
                             const QString& what) {
        QFile file(QString::fromStdString(Common::FS::PathToUTF8String(source_path)));
        if (!file.open(QIODevice::ReadOnly)) {
            return;
        }
        const QByteArray raw = file.readAll();
        if (raw.isEmpty()) {
            return;
        }
        const QByteArray sanitized = SanitizeIni(QString::fromUtf8(raw), scrubber).toUtf8();
        files.emplace_back(zip_name, sanitized);
        included << QStringLiteral("%1: %2 (%3 KB)")
                        .arg(zip_name, what)
                        .arg((raw.size() + 1023) / 1024);
    };

    const auto config_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::ConfigDir);
    add_ini(config_dir / "qt-config.ini", QStringLiteral("config/qt-config.ini"),
            QObject::tr("global configuration (sanitized)"));
    add_ini(config_dir / "frizti.ini", QStringLiteral("config/frizti.ini"),
            QObject::tr("tuned settings record"));

    std::error_code ec;
    const auto custom_dir = config_dir / "custom";
    if (std::filesystem::is_directory(custom_dir, ec)) {
        std::vector<std::filesystem::path> custom_inis;
        for (const auto& entry : std::filesystem::directory_iterator(custom_dir, ec)) {
            if (entry.is_regular_file(ec) && entry.path().extension() == ".ini") {
                custom_inis.push_back(entry.path());
            }
        }
        std::sort(custom_inis.begin(), custom_inis.end());
        for (const auto& ini_path : custom_inis) {
            const QString fname =
                QString::fromStdString(Common::FS::PathToUTF8String(ini_path.filename()));
            add_ini(ini_path, QStringLiteral("config/custom/") + fname,
                    QObject::tr("per-game configuration (sanitized)"));
        }
    }

    included.prepend(QStringLiteral("system-info.txt: system and settings summary"));
    files.emplace(files.begin(), QStringLiteral("system-info.txt"),
                  scrubber.Apply(SystemSummary(vk_device_records, included)).toUtf8());

    bool ok = false;
    {
        QuaZip zip(path);
        if (zip.open(QuaZip::mdCreate)) {
            ok = true;
            for (const auto& [name, data] : files) {
                ok = ok && AddFile(zip, name, data);
            }
            zip.close();
            ok = ok && zip.getZipError() == ZIP_OK;
        }
    }

    QApplication::restoreOverrideCursor();

    if (!ok) {
        QFile::remove(path);
        QMessageBox::warning(parent, QObject::tr("Export Log"),
                             QObject::tr("Could not save the log archive to %1.\n"
                                         "Please choose another folder.")
                                 .arg(QDir::toNativeSeparators(path)));
        return;
    }

    ShowInFileManager(path);

    QMessageBox box(parent);
    box.setIcon(QMessageBox::Information);
    box.setWindowTitle(QObject::tr("Log Exported"));
    box.setText(QObject::tr("<b>Saved %1</b>").arg(QFileInfo(path).fileName().toHtmlEscaped()));
    box.setInformativeText(QObject::tr(
        "Attach this zip file when reporting an issue."
        "<br><br>Included in the archive:"
        "<ul style='margin-left:-20px'>"
        "<li><code>eden_log.txt</code> and <code>eden_log.txt.old</code></li>"
        "<li>Sanitized <code>qt-config.ini</code> and per-game <code>custom/*.ini</code></li>"
        "<li><code>system-info.txt</code> (Lichii version, OS, CPU, GPU/driver, RAM, display)</li>"
        "</ul>"
        "Your OS user name, machine name, and account tokens were redacted."));
    box.setStandardButtons(QMessageBox::Ok);
    box.exec();
}

} // namespace ExportLog
