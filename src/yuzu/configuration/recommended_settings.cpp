// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <QMessageBox>
#include <QString>
#include <QStringList>

#include "common/game_settings.h"
#include "common/logging.h"
#include "common/settings.h"
#include "yuzu/configuration/recommended_settings.h"
#include "yuzu/configuration/shared_widget.h"

namespace RecommendedSettings {

namespace {

namespace Profile = Settings::DeviceProfile;

Settings::BasicSetting* Find(const std::string& key) {
    const auto& by_key = Settings::values.linkage.by_key;
    const auto it = by_key.find(key);
    return it != by_key.end() ? it->second : nullptr;
}

QString Label(const ConfigurationShared::Builder& builder, const Settings::BasicSetting& setting) {
    const auto& translations = builder.Translations();
    QString label = QString::fromStdString(setting.GetLabel());
    if (const auto it = translations.find(setting.Id());
        it != translations.end() && !it->second.first.isEmpty()) {
        label = it->second.first;
    }
    if (label.endsWith(QLatin1Char(':'))) {
        label.chop(1);
    }
    if (const auto* game = Settings::GameOfSetting(setting.GetLabel())) {
        label = QStringLiteral("%1: %2").arg(
            QString::fromUtf8(game->name.data(), static_cast<qsizetype>(game->name.size())),
            label);
    }
    return label;
}

// A stored value ("3", "true") in the words of the settings pages; `fallback` (the enum's name)
// where they have none (VSync's list is built from the GPU).
QString ValueText(const ConfigurationShared::Builder& builder,
                  const Settings::BasicSetting& setting, const std::string& value,
                  const std::string& fallback) {
    if (setting.TypeId() == "bool") {
        return value == "true" ? QObject::tr("On") : QObject::tr("Off");
    }
    if (setting.IsEnum()) {
        const auto& names = builder.ComboboxTranslations();
        if (const auto it = names.find(setting.EnumIndex()); it != names.end()) {
            for (const auto& [number, name] : it->second) {
                if (std::to_string(number) == value) {
                    return name;
                }
            }
        }
    }
    return QString::fromStdString(fallback.empty() ? value : fallback);
}

} // namespace

Settings::DeviceProfile::Device HostDevice(const std::vector<VkDeviceInfo::Record>& records) {
    auto device = Profile::DetectHost();
    const u32 index = Settings::values.vulkan_device.GetValue(true);
    if (Settings::values.renderer_backend.GetValue(true) == Settings::RendererBackend::Vulkan &&
        index < records.size()) {
        const auto& record = records[index];
        device.gpu_known = true;
        device.gpu_name = record.name;
        device.gpu_integrated = record.integrated;
        device.vram_mib = record.vram_mib;
#ifdef EGLANTINE_STREAMLINE
        device.dlss_capable = record.dlss_capable;
#endif
        switch (record.driver_id) {
        case VK_DRIVER_ID_NVIDIA_PROPRIETARY:
            device.gpu_vendor = Profile::GpuVendor::Nvidia;
            break;
        case VK_DRIVER_ID_AMD_PROPRIETARY:
        case VK_DRIVER_ID_AMD_OPEN_SOURCE:
        case VK_DRIVER_ID_MESA_RADV:
            device.gpu_vendor = Profile::GpuVendor::Amd;
            break;
        case VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS:
        case VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA:
            device.gpu_vendor = Profile::GpuVendor::Intel;
            break;
        case VK_DRIVER_ID_MOLTENVK:
            device.gpu_vendor = Profile::GpuVendor::Apple;
            break;
        default:
            device.gpu_vendor = Profile::GpuVendor::Other;
            break;
        }
    }
    return device;
}

bool ConfirmAndApply(QWidget* parent, const ConfigurationShared::Builder& builder,
                     const std::vector<VkDeviceInfo::Record>& records) {
    const auto profile = Profile::Recommend(HostDevice(records));
    QStringList changes;
    QStringList all;
    for (const auto& value : profile.settings) {
        const auto* setting = Find(value.key);
        if (setting == nullptr || !ConfigurationShared::AppliesToThisPlatform(*setting)) {
            continue;
        }
        const QString label = Label(builder, *setting);
        const QString to = ValueText(builder, *setting, value.value, value.display);
        all << QStringLiteral("%1: %2\n    %3")
                   .arg(label, to, QString::fromStdString(value.reason));
        if (setting->ToStringGlobal() != value.value) {
            changes << QStringLiteral("%1: %2 -> %3")
                           .arg(label,
                                ValueText(builder, *setting, setting->ToStringGlobal(),
                                          setting->Canonicalize()),
                                to);
        }
    }
    const QString device = QString::fromStdString(profile.device_name);
    if (changes.isEmpty()) {
        QMessageBox::information(parent, QObject::tr("Recommended Settings"),
                                 QObject::tr("Every setting already has the recommended value for "
                                             "%1.")
                                     .arg(device));
        return false;
    }
    QMessageBox box(QMessageBox::Question, QObject::tr("Use Recommended Settings"),
                    QObject::tr("Recommended for %1. These settings change:\n\n%2\n\nA game's own "
                                "overrides stay (reset them in its Properties).")
                        .arg(device, changes.join(QLatin1Char('\n'))),
                    QMessageBox::Yes | QMessageBox::Cancel, parent);
    box.setDetailedText(QObject::tr("Every recommended value and why:\n\n%1")
                            .arg(all.join(QLatin1Char('\n'))));
    box.setDefaultButton(QMessageBox::Yes);
    if (box.exec() != QMessageBox::Yes) {
        return false;
    }
    const auto applied = Profile::Apply(profile);
    LOG_INFO(Frontend, "Recommended settings for {} applied: {}", profile.device_name,
             Profile::FormatChanges(applied));
    return true;
}

bool ConfirmAndResetGame(QWidget* parent, const ConfigurationShared::Builder& builder) {
    QStringList overrides;
    for (const auto& [key, setting] : Settings::values.linkage.by_key) {
        if (setting->Switchable() && !setting->UsingGlobal() && setting->Save()) {
            overrides << QStringLiteral("%1: %2").arg(
                Label(builder, *setting), ValueText(builder, *setting, setting->ToString(),
                                                     setting->Canonicalize()));
        }
    }
    if (overrides.isEmpty()) {
        QMessageBox::information(parent, QObject::tr("Reset to Recommended"),
                                 QObject::tr("This game has no overrides: it already uses the "
                                             "global settings."));
        return false;
    }
    if (QMessageBox::question(
            parent, QObject::tr("Reset to Recommended"),
            QObject::tr("Remove this game's overrides, so it uses the global settings (Configure "
                        "> Use Recommended Settings sets those for this device)?\n\n%1")
                .arg(overrides.join(QLatin1Char('\n')))) != QMessageBox::Yes) {
        return false;
    }
    const auto changes = Profile::ResetGameOverrides();
    LOG_INFO(Frontend, "{:016X} reset to the global settings: {}",
             Settings::ConfiguringProgramId(), Profile::FormatChanges(changes));
    return true;
}

} // namespace RecommendedSettings
