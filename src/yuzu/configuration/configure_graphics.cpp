// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2016 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <filesystem>
#include <functional>
#include <map>
#include <tuple>
#include <vector>
#include <QBoxLayout>
#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QDesktopServices>
#include <QDir>
#include <QFileDialog>
#include <QFrame>
#include <QIcon>
#include <QLabel>
#include <QLineEdit>
#include <QListView>
#include <QMessageBox>
#include <QPixmap>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QStandardItemModel>
#include <QUrl>
#include <QStringLiteral>
#include <QtCore/qobjectdefs.h>
#include <qabstractbutton.h>
#include <qboxlayout.h>
#include <qcombobox.h>
#include <qcoreevent.h>
#include <qglobal.h>
#include <qgridlayout.h>
#include <vulkan/vulkan_core.h>

#include "common/common_types.h"
#include "common/game_settings.h"
#include "common/settings.h"
#include "common/settings_enums.h"
#include "core/core.h"
#include "qt_common/config/uisettings.h"
#include "qt_common/util/vk.h"
#include "ui_configure_graphics.h"
#include "yuzu/configuration/amd_fsr4_panel.h"
#include "yuzu/configuration/configuration_shared.h"
#include "yuzu/configuration/configure_graphics.h"
#include "yuzu/configuration/nvidia_dlss_panel.h"
#include "yuzu/configuration/shared_widget.h"
#include "core/file_sys/patch_manager.h"
#include "qt_common/config/catalog_text.h"
#include "yuzu/util/lichii_style.h"

#ifdef HAS_LSFG
#include "video_core/frame_gen/lossless_dll.h"
#endif

#ifdef HAS_RESHADE
#include "yuzu/configuration/configure_post_processing.h"
#endif

namespace {

template <typename E>
std::vector<std::pair<u32, QString>> EnumItems(const std::map<E, QString>& names) {
    std::vector<std::pair<u32, QString>> items;
    for (const auto& [value, name] : names) {
        items.emplace_back(static_cast<u32>(value), name);
    }
    return items;
}

QString JoinOrNone(const QStringList& names) {
    return names.isEmpty() ? UiText::Text("list.none_yet") : names.join(QStringLiteral(", "));
}

// The games with a Resolution of their own, as the iOS app's Resolution & Textures footer.
QString GamesWithResolution() {
    QStringList names;
    for (const auto& game : Settings::GamesWithSettings()) {
        for (const auto key : game.keys) {
            if (key.ends_with("_resolution")) {
                names << QString::fromUtf8(game.name.data(), static_cast<qsizetype>(game.name.size()));
                break;
            }
        }
    }
    return JoinOrNone(names);
}

QString BuiltinPatchGames() {
    QStringList names;
    for (const auto& name : FileSys::BuiltinPatchGames()) {
        names << QString::fromStdString(name);
    }
    return JoinOrNone(names);
}

void SetComboItemAvailability(QComboBox* combobox, u32 value, bool visible, bool enabled,
                              const QString& disabled_tooltip = {}) {
    if (combobox == nullptr) {
        return;
    }
    const int row = combobox->findData(value);
    if (row < 0) {
        return;
    }
    const bool is_current = combobox->currentIndex() == row;
    const bool keep_visible = visible || is_current;
    if (auto* const view = qobject_cast<QListView*>(combobox->view())) {
        view->setRowHidden(row, !keep_visible);
    }
    if (auto* const model = qobject_cast<QStandardItemModel*>(combobox->model())) {
        if (QStandardItem* const item = model->item(row)) {
            item->setEnabled((visible && enabled) || is_current);
            item->setToolTip(enabled ? QString() : disabled_tooltip);
        }
    }
}

template <typename E>
void GateComboItem(QComboBox* combobox, E value, bool offered) {
    SetComboItemAvailability(combobox, static_cast<u32>(value), offered, offered);
}

template <typename E>
void EnableComboItem(QComboBox* combobox, E value, bool enabled,
                     const QString& disabled_tooltip = {}) {
    SetComboItemAvailability(combobox, static_cast<u32>(value), true, enabled, disabled_tooltip);
}

} // namespace

ConfigureGraphics::ConfigureGraphics(
    const Core::System& system_, std::vector<VkDeviceInfo::Record>& records_,
    const std::function<void()>& expose_compute_option_,
    const std::function<void(Settings::AspectRatio, Settings::ResolutionSetup)>&
        update_aspect_ratio_,
    std::shared_ptr<std::vector<ConfigurationShared::Tab*>> group_,
    const ConfigurationShared::Builder& builder, QWidget* parent)
    : ConfigurationShared::Tab(group_, parent), ui{std::make_unique<Ui::ConfigureGraphics>()},
      records{records_}, expose_compute_option{expose_compute_option_},
      update_aspect_ratio{update_aspect_ratio_}, system{system_},
      combobox_translations{builder.ComboboxTranslations()} {
    vulkan_device = Settings::values.vulkan_device.GetValue();
    RetrieveVulkanDevices();

    ui->setupUi(this);
    UiText::ApplyCatalogue(this);

    Setup(builder);

    for (const auto& device : vulkan_devices) {
        vulkan_device_combobox->addItem(device);
    }

    UpdateBackgroundColorButton(QColor::fromRgb(Settings::values.bg_red.GetValue(),
                                                Settings::values.bg_green.GetValue(),
                                                Settings::values.bg_blue.GetValue()));
    UpdateAPILayout();
    PopulateVSyncModeSelection(false); //< must happen after UpdateAPILayout

    // VSync setting needs to be determined after populating the VSync combobox
    const auto vsync_mode_setting = Settings::values.vsync_mode.GetValue();
    const auto vsync_mode = VSyncSettingToMode(vsync_mode_setting);
    int index{};
    for (const auto mode : vsync_mode_combobox_enum_map) {
        if (mode == vsync_mode) {
            break;
        }
        index++;
    }
    if (static_cast<unsigned long>(index) < vsync_mode_combobox_enum_map.size()) {
        vsync_mode_combobox->setCurrentIndex(index);
    }

    connect(api_combobox, qOverload<int>(&QComboBox::activated), this, [this] {
        UpdateAPILayout();
        PopulateVSyncModeSelection(false);
        UpdateFeatureAvailability();
    });
    connect(vulkan_device_combobox, qOverload<int>(&QComboBox::activated), this,
            [this](int device) {
                UpdateDeviceSelection(device);
                PopulateVSyncModeSelection(false);
                UpdateFeatureAvailability();
            });
    connect(ui->bg_button, &QPushButton::clicked, this, [this] {
        const QColor new_bg_color = QColorDialog::getColor(bg_color);
        if (!new_bg_color.isValid()) {
            return;
        }
        UpdateBackgroundColorButton(new_bg_color);
    });

    const auto& update_screenshot_info = [this, &builder]() {
        const auto& combobox_enumerations = builder.ComboboxTranslations().at(
            Settings::EnumMetadata<Settings::AspectRatio>::Index());
        const auto ratio_index = aspect_ratio_combobox->currentIndex();
        const auto ratio =
            static_cast<Settings::AspectRatio>(combobox_enumerations[ratio_index].first);

        const auto& combobox_enumerations_resolution = builder.ComboboxTranslations().at(
            Settings::EnumMetadata<Settings::ResolutionSetup>::Index());
        const auto setup =
            resolution_combobox != nullptr
                ? static_cast<Settings::ResolutionSetup>(
                      combobox_enumerations_resolution[resolution_combobox->currentIndex()].first)
                : Settings::values.resolution_setup.GetValue();

        update_aspect_ratio(ratio, setup);
    };

    connect(aspect_ratio_combobox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            update_screenshot_info);
    if (resolution_combobox != nullptr) {
        connect(resolution_combobox, QOverload<int>::of(&QComboBox::currentIndexChanged),
                update_screenshot_info);
    }

    api_combobox->setEnabled(!UISettings::values.has_broken_vulkan && api_combobox->isEnabled());
    ui->api_widget->setEnabled(
        (!UISettings::values.has_broken_vulkan || Settings::IsConfiguringGlobal()) &&
        ui->api_widget->isEnabled());

    if (Settings::IsConfiguringGlobal()) {
        ui->bg_widget->setEnabled(Settings::values.bg_red.UsingGlobal());
    }

    UpdateFeatureAvailability();
}

void ConfigureGraphics::PopulateVSyncModeSelection(bool use_setting) {
    const Settings::RendererBackend backend{GetCurrentGraphicsBackend()};
    if (backend == Settings::RendererBackend::Null) {
        vsync_mode_combobox->setEnabled(false);
        return;
    }
    vsync_mode_combobox->setEnabled(true);

    const int current_index = //< current selected vsync mode from combobox
        vsync_mode_combobox->currentIndex();
    const auto current_mode = //< current selected vsync mode as a VkPresentModeKHR
        current_index == -1 || use_setting
            ? VSyncSettingToMode(Settings::values.vsync_mode.GetValue())
            : vsync_mode_combobox_enum_map[current_index];
    int index{};
    const int device{vulkan_device_combobox->currentIndex()}; //< current selected Vulkan device

    const auto& present_modes = //< relevant vector of present modes for the selected device or API
        backend == Settings::RendererBackend::Vulkan && device > -1 ? device_present_modes[device]
                                                                    : default_present_modes;

    vsync_mode_combobox->clear();
    vsync_mode_combobox_enum_map.clear();
    vsync_mode_combobox_enum_map.reserve(present_modes.size());
    for (const auto present_mode : present_modes) {
        const auto mode_name = TranslateVSyncMode(present_mode, backend);
        if (mode_name.isEmpty()) {
            continue;
        }

        vsync_mode_combobox->insertItem(index, mode_name);
        vsync_mode_combobox_enum_map.push_back(present_mode);
        if (present_mode == current_mode) {
            vsync_mode_combobox->setCurrentIndex(index);
        }
        index++;
    }

    if (!Settings::IsConfiguringGlobal()) {
        vsync_restore_global_button->setVisible(!Settings::values.vsync_mode.UsingGlobal());

        const Settings::VSyncMode global_vsync_mode = Settings::values.vsync_mode.GetValue(true);
        vsync_restore_global_button->setEnabled(
            ((backend == Settings::RendererBackend::OpenGL_GLSL ||
              backend == Settings::RendererBackend::OpenGL_GLASM ||
              backend == Settings::RendererBackend::OpenGL_SPIRV) &&
             (global_vsync_mode == Settings::VSyncMode::Immediate ||
              global_vsync_mode == Settings::VSyncMode::Fifo)) ||
            backend == Settings::RendererBackend::Vulkan ||
            backend == Settings::RendererBackend::Metal);
    }
}

void ConfigureGraphics::UpdateVsyncSetting() const {
    const Settings::RendererBackend backend{GetCurrentGraphicsBackend()};
    if (backend == Settings::RendererBackend::Null) {
        return;
    }

    const auto mode = vsync_mode_combobox_enum_map[vsync_mode_combobox->currentIndex()];
    const auto vsync_mode = PresentModeToSetting(mode);
    Settings::values.vsync_mode.SetValue(vsync_mode);
}

void ConfigureGraphics::UpdateDeviceSelection(int device) {
    if (device == -1) {
        return;
    }
    if (GetCurrentGraphicsBackend() == Settings::RendererBackend::Vulkan) {
        vulkan_device = device;
    }
}

void ConfigureGraphics::SelectVulkanDevice(int device) {
    if (device < 0 || vulkan_device_combobox == nullptr ||
        device >= vulkan_device_combobox->count()) {
        return;
    }
    vulkan_device_combobox->setCurrentIndex(device);
    UpdateDeviceSelection(device);
    PopulateVSyncModeSelection(false);
    UpdateFeatureAvailability();
}

ConfigureGraphics::~ConfigureGraphics() = default;

void ConfigureGraphics::SetConfiguration() {}

void ConfigureGraphics::Setup(const ConfigurationShared::Builder& builder) {
    QLayout* api_layout = ui->api_widget->layout();
    QWidget* api_grid_widget = new QWidget(this);
    QVBoxLayout* api_grid_layout = new QVBoxLayout(api_grid_widget);
    api_grid_layout->setContentsMargins(0, 0, 0, 0);
    api_layout->addWidget(api_grid_widget);

    QLayout& graphics_layout = *ui->graphics_widget->layout();

    std::map<u32, QWidget*> hold_graphics;
    std::vector<QWidget*> hold_api;
    QCheckBox* flow_scale_auto_checkbox = nullptr;
    QComboBox* oot_preset_combobox = nullptr;
    QComboBox* texture_quality_combobox = nullptr;
    QComboBox* frame_gen_motion_combobox = nullptr;
    const bool runtime_lock = !system.IsPoweredOn();
    const auto make_enum_row = [&](Settings::BasicSetting* s, const QString& label_text,
                                   const QString& tooltip_text,
                                   const std::vector<std::pair<u32, QString>>& items,
                                   QComboBox*& out_combobox) -> QWidget* {
        if (!Settings::IsConfiguringGlobal() && !s->Switchable()) {
            return nullptr;
        }
        QWidget* row = new QWidget(this);
        QHBoxLayout* layout = new QHBoxLayout(row);
        layout->setContentsMargins(0, 0, 0, 0);
        QLabel* label = new QLabel(label_text, row);
        label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
        QComboBox* cb = new QComboBox(row);
        cb->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
        const u32 current_val = static_cast<u32>(std::strtoul(s->ToString().c_str(), nullptr, 0));
        int current_idx = 0;
        int idx = 0;
        for (const auto& [val, text] : items) {
            cb->addItem(text, val);
            if (val == current_val) {
                current_idx = idx;
            }
            ++idx;
        }
        cb->setCurrentIndex(current_idx);
        layout->addWidget(label);
        layout->addWidget(cb);

        if (!Settings::IsConfiguringGlobal()) {
            QPushButton* restore_btn =
                ConfigurationShared::Widget::CreateRestoreGlobalButton(s->UsingGlobal(), row);
            layout->addWidget(restore_btn);
            connect(cb, QOverload<int>::of(&QComboBox::activated), row, [restore_btn] {
                restore_btn->setEnabled(true);
                restore_btn->setVisible(true);
            });
            connect(restore_btn, &QAbstractButton::clicked, row, [s, cb, restore_btn](bool) {
                restore_btn->setEnabled(false);
                restore_btn->setVisible(false);
                const u32 global_val =
                    static_cast<u32>(std::strtoul(s->ToStringGlobal().c_str(), nullptr, 0));
                const int found = cb->findData(global_val);
                if (found >= 0) {
                    cb->setCurrentIndex(found);
                }
            });
            apply_funcs.push_back([s, cb, restore_btn](bool powered_on) {
                if (s->RuntimeModifiable() || !powered_on) {
                    const bool using_global = !restore_btn->isEnabled();
                    s->SetGlobal(using_global);
                    if (!using_global) {
                        s->LoadString(std::to_string(cb->currentData().toUInt()));
                    }
                }
            });
        } else {
            apply_funcs.push_back([s, cb](bool powered_on) {
                if (s->RuntimeModifiable() || !powered_on) {
                    s->LoadString(std::to_string(cb->currentData().toUInt()));
                }
            });
        }

        bool enable = runtime_lock || s->RuntimeModifiable();
        QString reason;
        if (!enable) {
            reason = UiText::Text("note.stop_game");
        }
        if (s->Switchable() && Settings::IsConfiguringGlobal() && !runtime_lock &&
            !s->UsingGlobal()) {
            enable = false;
            reason = UiText::Text("note.game_overrides");
        }
        row->setEnabled(enable);
        row->setToolTip(reason.isEmpty() ? tooltip_text
                                         : tooltip_text + QStringLiteral("\n\n") + reason);
        out_combobox = cb;
        return row;
    };

    for (const auto setting : Settings::values.linkage.by_category[Settings::Category::Renderer]) {
        if (setting->Id() == Settings::values.frame_gen_motion.Id()) {
            if (QWidget* row = make_enum_row(
                    setting, UiText::SettingLabel("frame_gen_motion"),
                    UiText::SettingHelp("frame_gen_motion"),
                    EnumItems(UiText::ChoiceNames<Settings::FrameGenMotion>()),
                    frame_gen_motion_combobox)) {
                hold_graphics.emplace(setting->Id(), row);
            }
            continue;
        }
        if (setting->Id() == Settings::values.frame_gen_flow_quality.Id()) {
            QComboBox* quality_cb = nullptr;
            if (QWidget* row = make_enum_row(
                    setting, UiText::SettingLabel("frame_gen_flow_quality"),
                    UiText::SettingHelp("frame_gen_flow_quality"),
                    EnumItems(UiText::ChoiceNames<Settings::FrameGenFlowQuality>()), quality_cb)) {
                hold_graphics.emplace(setting->Id(), row);
            }
            continue;
        }
#ifndef HAS_RESHADE
        if (setting->Id() == Settings::values.post_shader_enabled.Id()) {
            continue;
        }
#endif
        ConfigurationShared::Widget* widget = [&]() {
            if (setting->Id() == Settings::values.fsr_sharpening_slider.Id()) {
                // FSR needs a reversed slider and a 0.5 multiplier
                return builder.BuildWidget(
                    setting, apply_funcs, ConfigurationShared::RequestType::ReverseSlider, true,
                    0.5f, nullptr, tr("%", "FSR/SGSR sharpening percentage (e.g. 50%)"));
            } else if (setting->Id() == Settings::values.frame_rate_limit.Id()) {
                return builder.BuildWidget(setting, apply_funcs,
                                           ConfigurationShared::RequestType::Slider, true, 1.0f,
                                           nullptr, tr(" Hz", "Frame rate limit (e.g. 120 Hz)"));
            } else if (setting->Id() == Settings::values.frame_gen_multiplier.Id()) {
                return builder.BuildWidget(setting, apply_funcs, nullptr,
                                           ConfigurationShared::RequestType::Default,
                                           tr("x", "Frame generation multiplier (e.g. 2x)"));
            } else if (setting->Id() == Settings::values.frame_gen_target_rate.Id()) {
                return builder.BuildWidget(setting, apply_funcs, nullptr,
                                           ConfigurationShared::RequestType::Default,
                                           tr(" fps", "Frame generation target (e.g. 120 fps)"));
            } else {
                return builder.BuildWidget(setting, apply_funcs);
            }
        }();

        if (widget == nullptr) {
            continue;
        }
        if (!widget->Valid()) {
            widget->deleteLater();
            continue;
        }

        if (setting->Id() == Settings::values.frame_gen_type.Id()) {
            frame_gen_type_combobox = widget->combobox;
            if (frame_gen_type_combobox != nullptr) {
                connect(frame_gen_type_combobox, QOverload<int>::of(&QComboBox::currentIndexChanged),
                        this, [this] { UpdateFeatureAvailability(); });
                connect(frame_gen_type_combobox, QOverload<int>::of(&QComboBox::activated), this,
                        [this] {
                            if (frame_gen_type_combobox->currentData().toUInt() !=
                                static_cast<u32>(Settings::FrameGenType::NvidiaDlssG)) {
                                return;
                            }
                            if (vulkan_device_combobox == nullptr || vulkan_device_widget == nullptr ||
                                !vulkan_device_widget->isEnabled()) {
                                return;
                            }
                            const int cur = vulkan_device_combobox->currentIndex();
                            if (cur >= 0 && size_t(cur) < records.size() &&
                                records[cur].dlss_capable) {
                                return;
                            }
                            SelectVulkanDevice(VkDeviceInfo::FindBestDevice(
                                records, [](const auto& r) { return r.dlss_capable; }));
                        });
            }
        } else if (setting->Id() == Settings::values.nvidia_dlss_quality.Id()) {
            nvidia_dlss_quality_combobox = widget->combobox;
            if (nvidia_dlss_quality_combobox != nullptr) {
                connect(nvidia_dlss_quality_combobox,
                        QOverload<int>::of(&QComboBox::currentIndexChanged), this,
                        [this] { UpdateFeatureAvailability(); });
            }
        } else if (setting->Id() == Settings::values.scaling_filter.Id()) {
            scaling_filter_combobox = widget->combobox;
        } else if (setting->Id() == Settings::values.frame_gen_target_rate.Id() &&
                   widget->spinbox != nullptr) {
            widget->spinbox->setSpecialValueText(tr("Off (use the multiplier)"));
        } else if (setting->Id() == Settings::values.frame_gen_flow_scale_auto.Id()) {
            flow_scale_auto_checkbox = widget->checkbox;
        } else if (setting->Id() == Settings::values.oot_resolution.Id()) {
            oot_resolution_combobox = widget->combobox;
        } else if (setting->Id() == Settings::values.oot_graphics_preset.Id()) {
            oot_preset_combobox = widget->combobox;
        } else if (setting->Id() == Settings::values.texture_quality.Id()) {
            texture_quality_combobox = widget->combobox;
        }

        if (setting->Id() == Settings::values.renderer_backend.Id()) {
            // Add the renderer combobox now so it's at the top
            api_grid_layout->addWidget(widget);
            api_combobox = widget->combobox;
            api_restore_global_button = widget->restore_button;
            if (!Settings::IsConfiguringGlobal()) {
                api_restore_global_button->connect(api_restore_global_button,
                                                   &QAbstractButton::clicked, [this](bool) {
                                                       UpdateAPILayout();
                                                       PopulateVSyncModeSelection(false);
                                                       UpdateFeatureAvailability();
                                                   });
                // Detach API's restore button and place it where we want
                // Lets us put it on the side, and it will automatically scale if there's a
                // second combobox (vulkan_device)
                widget->RowLayout()->removeWidget(api_restore_global_button);
                api_layout->addWidget(api_restore_global_button);
            }
        } else if (setting->Id() == Settings::values.vulkan_device.Id()) {
            // Keep track of vulkan_device's combobox so we can populate it
            hold_api.push_back(widget);
            vulkan_device_combobox = widget->combobox;
            vulkan_device_widget = widget;
        } else if (setting->Id() == Settings::values.vsync_mode.Id()) {
            // Keep track of vsync_mode's combobox so we can populate it
            vsync_mode_combobox = widget->combobox;

            // Since vsync is populated at runtime, we have to manually set up the button for
            // restoring the global setting.
            if (!Settings::IsConfiguringGlobal()) {
                QPushButton* restore_button =
                    ConfigurationShared::Widget::CreateRestoreGlobalButton(
                        Settings::values.vsync_mode.UsingGlobal(), widget);
                restore_button->setEnabled(true);
                widget->RowLayout()->addWidget(restore_button);

                restore_button->connect(restore_button, &QAbstractButton::clicked,
                                        [restore_button, this](bool) {
                                            Settings::values.vsync_mode.SetGlobal(true);
                                            PopulateVSyncModeSelection(true);

                                            restore_button->setVisible(false);
                                        });

                std::function<void()> set_non_global = [restore_button, this]() {
                    Settings::values.vsync_mode.SetGlobal(false);
                    UpdateVsyncSetting();
                    restore_button->setVisible(true);
                };
                widget->combobox->connect(widget->combobox,
                                          QOverload<int>::of(&QComboBox::activated),
                                          [set_non_global]() { set_non_global(); });
                vsync_restore_global_button = restore_button;
            }
            hold_graphics.emplace(setting->Id(), widget);
        } else if (setting->Id() == Settings::values.aspect_ratio.Id()) {
            // Keep track of the aspect ratio combobox to update other UI tabs that need it
            aspect_ratio_combobox = widget->combobox;
            hold_graphics.emplace(setting->Id(), widget);
        } else if (setting->Id() == Settings::values.resolution_setup.Id()) {
            // Keep track of the resolution combobox to update other UI tabs that need it
            resolution_combobox = widget->combobox;
            hold_graphics.emplace(setting->Id(), widget);
#ifdef HAS_RESHADE
        } else if (setting->Id() == Settings::values.post_shader_enabled.Id()) {
            QPushButton* post_shader_button = new QPushButton(tr("Configure Effects..."), widget);
            post_shader_button->setVisible(widget->checkbox->isChecked());

            connect(post_shader_button, &QAbstractButton::clicked, this, [this]() {
                ConfigurePostProcessing dialog(this);
                dialog.exec();
            });
            connect(widget->checkbox, &QCheckBox::toggled, post_shader_button,
                    &QWidget::setVisible);

            QBoxLayout* row = widget->RowLayout();
            row->insertWidget(1, post_shader_button);

            hold_graphics.emplace(setting->Id(), widget);
#endif
        } else if (setting->Id() == Settings::values.upscaler.Id()) {
            upscaler_combobox = widget->combobox;
            connect(widget->combobox, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
                    [this] { UpdateFeatureAvailability(); });
            connect(widget->combobox, QOverload<int>::of(&QComboBox::activated), this, [this] {
                if (upscaler_combobox == nullptr || vulkan_device_combobox == nullptr ||
                    vulkan_device_widget == nullptr || !vulkan_device_widget->isEnabled()) {
                    return;
                }
                const auto chosen =
                    static_cast<Settings::Upscaler>(upscaler_combobox->currentData().toUInt());
                const int cur = vulkan_device_combobox->currentIndex();
                const auto* cur_rec =
                    cur >= 0 && size_t(cur) < records.size() ? &records[cur] : nullptr;
                if (chosen == Settings::Upscaler::Dlss &&
                    (cur_rec == nullptr || !cur_rec->dlss_capable)) {
                    SelectVulkanDevice(VkDeviceInfo::FindBestDevice(
                        records, [](const auto& r) { return r.dlss_capable; }));
                } else if (chosen == Settings::Upscaler::Fsr4 &&
                           (cur_rec == nullptr || !cur_rec->fsr4_capable)) {
                    SelectVulkanDevice(VkDeviceInfo::FindBestDevice(
                        records, [](const auto& r) { return r.fsr4_capable; }));
                }
            });
#if defined(EGLANTINE_STREAMLINE) || defined(_WIN32) || (defined(__linux__) && !defined(__ANDROID__))
            QWidget* group = new QWidget(this);
            QVBoxLayout* group_layout = new QVBoxLayout(group);
            group_layout->setContentsMargins(0, 0, 0, 0);
            group_layout->setSpacing(2);
            group_layout->addWidget(widget);
#ifdef EGLANTINE_STREAMLINE
            nvidia_dlss_panel = new NvidiaDlssPanel(group);
            group_layout->addWidget(nvidia_dlss_panel);
#endif
#if defined(_WIN32) || (defined(__linux__) && !defined(__ANDROID__))
            amd_fsr4_panel = new AmdFsr4Panel(group);
            group_layout->addWidget(amd_fsr4_panel);
#endif
            hold_graphics.emplace(setting->Id(), group);
#else
            hold_graphics.emplace(setting->Id(), widget);
#endif
        } else if (setting->Id() == Settings::values.frame_gen.Id()) {
            frame_gen_checkbox = widget->checkbox;
            if (frame_gen_checkbox != nullptr) {
                connect(frame_gen_checkbox, &QCheckBox::toggled, this,
                        [this] { UpdateFeatureAvailability(); });
            }
            hold_graphics.emplace(setting->Id(), widget);
        } else {
            hold_graphics.emplace(setting->Id(), widget);
        }
    }

    // Sections, each under a heading that says what its settings do and how they combine; the
    // rest follows in the settings' order.
    {
        const auto& v = Settings::values;
        // The section names and footers of the iOS app's Graphics page.
        std::vector<std::tuple<QString, QString, std::vector<u32>>> sections{
            {UiText::Text("category.Resolution"),
             Settings::ConfiguringProgramId() == 0
                 ? UiText::Text("category.Resolution.footer").arg(GamesWithResolution())
                 : QString(),
             {v.resolution_setup.Id(), v.texture_quality.Id(), v.aspect_ratio.Id()}},
            {UiText::Text("category.Upscaling"), UiText::Text("category.Upscaling.footer"),
             {v.upscaler.Id(), v.nvidia_dlss_quality.Id(), v.nvidia_dlss_preset.Id(),
              v.scaling_filter.Id(), v.fsr_sharpening_slider.Id(), v.anti_aliasing.Id()}},
            {UiText::Text("category.FrameRate"), UiText::Text("category.FrameRate.footer"),
             {v.frame_rate_limit.Id(), v.vsync_mode.Id(), v.frame_gen.Id(), v.frame_gen_type.Id(),
              v.frame_gen_multiplier.Id(), v.frame_gen_target_rate.Id(),
              v.frame_gen_flow_scale_auto.Id(), v.frame_gen_flow_scale.Id(),
              v.frame_gen_queue_target.Id(), v.frame_gen_motion.Id(),
              v.frame_gen_flow_quality.Id()}},
        };
        // Each game's own settings under its name (on a per-game page only that game's).
        for (const auto& game : Settings::GamesWithSettings()) {
            std::vector<u32> ids;
            for (const auto key : game.keys) {
                if (const auto it = v.linkage.by_key.find(std::string{key});
                    it != v.linkage.by_key.end()) {
                    ids.push_back(it->second->Id());
                }
            }
            sections.emplace_back(QString::fromUtf8(game.name.data(),
                                                    static_cast<qsizetype>(game.name.size())),
                                  Settings::ConfiguringProgramId() == 0
                                      ? UiText::Text("category.Game.footer")
                                      : QString(),
                                  std::move(ids));
        }
        sections.emplace_back(UiText::Text("category.GameEnhancements"),
                              UiText::Text("category.GameEnhancements.footer")
                                  .arg(BuiltinPatchGames()),
                              std::vector<u32>{v.patch_performance.Id(), v.patch_image.Id()});
        sections.emplace_back(UiText::Text("category.Renderer"), QString(),
                              std::vector<u32>{});
        std::map<u32, QWidget*> rest = hold_graphics;
        // The group box only holds the sections: each is a header, a rounded group of rows and a
        // footer, as the iOS app's grouped lists.
        ui->groupBox->setTitle(QString());
        ui->groupBox->setProperty("lichiiPlain", true);
        ui->groupBox->setFlat(true);
        for (const auto& [title, text, ids] : sections) {
            std::vector<QWidget*> widgets;
            for (const u32 id : ids) {
                if (const auto it = rest.find(id); it != rest.end()) {
                    widgets.push_back(it->second);
                    rest.erase(it);
                }
            }
            if (ids.empty()) {
                for (const auto& [id, widget] : rest) {
                    widgets.push_back(widget);
                }
            }
            if (widgets.empty()) {
                continue;
            }
            graphics_layout.addWidget(LichiiStyle::SectionHeader(title, this));
            QFrame* const card = LichiiStyle::Card(this);
            card->setProperty("lichiiSection", title);
            for (QWidget* const widget : widgets) {
                LichiiStyle::AddRow(card, widget);
            }
            graphics_layout.addWidget(card);
            if (!text.isEmpty()) {
                graphics_layout.addWidget(LichiiStyle::Footer(text, this));
            }
        }
    }

    // A phone preset of Ocarina of Time's graphics also picks its resolution and texture quality.
    if (oot_preset_combobox != nullptr) {
        const auto pick = [](QComboBox* box, u32 value) {
            if (box == nullptr) {
                return;
            }
            if (const int index = box->findData(value); index >= 0) {
                box->setCurrentIndex(index);
            }
        };
        QComboBox* const preset_box = oot_preset_combobox;
        connect(preset_box, &QComboBox::activated, this,
                [this, pick, preset_box, texture_quality_combobox](int index) {
                    const auto companions = Settings::OotGraphicsPresetCompanions(
                        static_cast<Settings::OotGraphicsPreset>(
                            preset_box->itemData(index).toUInt()));
                    if (companions.resolution) {
                        pick(oot_resolution_combobox, static_cast<u32>(*companions.resolution));
                    }
                    if (companions.textures) {
                        pick(texture_quality_combobox, static_cast<u32>(*companions.textures));
                    }
                });
    }

    // The manual flow scale is only read while Auto Flow Scale is off.
    if (const auto it = hold_graphics.find(Settings::values.frame_gen_flow_scale.Id());
        flow_scale_auto_checkbox != nullptr && it != hold_graphics.end()) {
        QWidget* flow_scale = it->second;
        const auto update = [flow_scale](bool automatic) {
            ConfigurationShared::SetUnavailable(
                flow_scale, automatic ? tr("Turn Auto Flow Scale off to set this.") : QString());
        };
        update(flow_scale_auto_checkbox->isChecked());
        connect(flow_scale_auto_checkbox, &QCheckBox::toggled, flow_scale, update);
    }

    const auto held = [&hold_graphics](const Settings::BasicSetting& setting) -> QWidget* {
        const auto it = hold_graphics.find(setting.Id());
        return it != hold_graphics.end() ? it->second : nullptr;
    };
    nvidia_dlss_quality_widget = held(Settings::values.nvidia_dlss_quality);
    nvidia_dlss_preset_widget = held(Settings::values.nvidia_dlss_preset);
    frame_gen_widget = held(Settings::values.frame_gen);

#ifdef HAS_LSFG
    // Where LSFG's Lossless.dll comes from, under the type combobox.
    if (QWidget* type_widget = held(Settings::values.frame_gen_type);
        type_widget != nullptr && frame_gen_type_combobox != nullptr) {
        QWidget* row = new QWidget(type_widget);
        QHBoxLayout* row_layout = new QHBoxLayout(row);
        row_layout->setContentsMargins(0, 0, 0, 0);
        lossless_status_label = new QLabel(row);
        lossless_status_label->setWordWrap(true);
        lossless_status_label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
        QPushButton* choose = new QPushButton(tr("Choose Lossless.dll..."), row);
        QPushButton* open_folder = new QPushButton(tr("Open Folder"), row);
        row_layout->addWidget(lossless_status_label);
        row_layout->addWidget(choose);
        row_layout->addWidget(open_folder);

        QWidget* group = new QWidget(this);
        QVBoxLayout* group_layout = new QVBoxLayout(group);
        group_layout->setContentsMargins(0, 0, 0, 0);
        group_layout->setSpacing(2);
        type_widget->parentWidget()->layout()->replaceWidget(type_widget, group);
        group_layout->addWidget(type_widget);
        group_layout->addWidget(row);
        type_widget->setVisible(true);
        hold_graphics[Settings::values.frame_gen_type.Id()] = group;

        const auto show_row = [this, row] {
            row->setVisible(frame_gen_type_combobox->currentData().toUInt() ==
                            static_cast<u32>(Settings::FrameGenType::Lsfg));
        };
        show_row();
        connect(frame_gen_type_combobox, qOverload<int>(&QComboBox::currentIndexChanged), row,
                show_row);
        connect(choose, &QPushButton::clicked, this, [this] {
            const QString file = QFileDialog::getOpenFileName(
                this, tr("Choose Lossless.dll"), QString(), QStringLiteral("Lossless.dll"));
            if (file.isEmpty()) {
                return;
            }
            const std::filesystem::path source{file.toStdU16String()};
            if (VideoCore::FrameGen::ValidateLosslessDll(source) !=
                VideoCore::FrameGen::LosslessStatus::Ok) {
                QMessageBox::warning(this, tr("Lossless.dll"),
                                     tr("This file is not a Lossless.dll with the frame "
                                        "generation shaders Lichii needs."));
                return;
            }
            const auto destination = VideoCore::FrameGen::GetLosslessDllPath();
            std::error_code ec;
            std::filesystem::create_directories(destination.parent_path(), ec);
            std::filesystem::copy_file(source, destination,
                                       std::filesystem::copy_options::overwrite_existing, ec);
            if (ec) {
                QMessageBox::warning(this, tr("Lossless.dll"),
                                     tr("Could not copy the file: %1")
                                         .arg(QString::fromStdString(ec.message())));
            }
            UpdateLosslessStatus();
        });
        connect(open_folder, &QPushButton::clicked, this, [] {
            const auto folder = VideoCore::FrameGen::GetLosslessDllPath().parent_path();
            std::error_code ec;
            std::filesystem::create_directories(folder, ec);
            QDesktopServices::openUrl(QUrl::fromLocalFile(
                QString::fromStdU16String(folder.u16string())));
        });
    }
    UpdateLosslessStatus();
#endif

    // Hide rather than disable, a disabled widget would read as using the global value. Each
    // frame generation setting shows only while it applies: the flow scale to LSFG, the motion
    // source to the built-in type, and the optical flow quality to its hardware.
    if (frame_gen_checkbox != nullptr) {
        const auto selected = [](QComboBox* combobox, auto value) {
            return combobox == nullptr ||
                   combobox->currentData().toUInt() == static_cast<u32>(value);
        };
        QCheckBox* const checkbox = frame_gen_checkbox;
        QComboBox* const type = frame_gen_type_combobox;
        QComboBox* const motion = frame_gen_motion_combobox;
        std::vector<std::pair<QWidget*, std::function<bool()>>> dependents;
        const auto add = [&](u32 id, std::function<bool()> applies) {
            if (const auto it = hold_graphics.find(id); it != hold_graphics.end()) {
                dependents.emplace_back(it->second, std::move(applies));
            }
        };
        const auto always = [] { return true; };
        const auto lsfg = [=] { return selected(type, Settings::FrameGenType::Lsfg); };
        const auto built_in = [=] { return selected(type, Settings::FrameGenType::Fsr3); };
        add(Settings::values.frame_gen_type.Id(), always);
        add(Settings::values.frame_gen_multiplier.Id(), always);
        add(Settings::values.frame_gen_target_rate.Id(), always);
        add(Settings::values.frame_gen_flow_scale_auto.Id(), lsfg);
        add(Settings::values.frame_gen_flow_scale.Id(), lsfg);
        add(Settings::values.frame_gen_queue_target.Id(), always);
        add(Settings::values.frame_gen_motion.Id(), built_in);
        add(Settings::values.frame_gen_flow_quality.Id(), [=] {
            return built_in() && selected(motion, Settings::FrameGenMotion::Auto);
        });
        const auto update = [checkbox, dependents] {
            for (const auto& [widget, applies] : dependents) {
                widget->setVisible(checkbox->isChecked() && applies());
            }
        };
        update();
        connect(checkbox, &QCheckBox::toggled, this, update);
        for (QComboBox* const combobox : {type, motion}) {
            if (combobox != nullptr) {
                connect(combobox, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
                        update);
            }
        }
    }

    for (auto widget : hold_api) {
        api_grid_layout->addWidget(widget);
    }

    discrete_gpu_hint_widget = new QWidget(api_grid_widget);
    QHBoxLayout* hint_layout = new QHBoxLayout(discrete_gpu_hint_widget);
    hint_layout->setContentsMargins(24, 2, 0, 4);
    hint_layout->setSpacing(8);
    discrete_gpu_hint_label = new QLabel(discrete_gpu_hint_widget);
    discrete_gpu_hint_label->setTextFormat(Qt::RichText);
    discrete_gpu_hint_label->setWordWrap(true);
    discrete_gpu_hint_label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    QPushButton* use_discrete_btn =
        new QPushButton(tr("Use Discrete GPU"), discrete_gpu_hint_widget);
    hint_layout->addWidget(discrete_gpu_hint_label, 1);
    hint_layout->addWidget(use_discrete_btn);
    discrete_gpu_hint_widget->setVisible(false);
    api_grid_layout->addWidget(discrete_gpu_hint_widget);
    connect(use_discrete_btn, &QPushButton::clicked, this, [this] {
        SelectVulkanDevice(VkDeviceInfo::FindBestDiscreteDevice(records));
    });

    // Background color is too specific to build into the new system, so we manage it here
    // (3 settings, all collected into a single widget with a QColor to manage on top)
    if (Settings::IsConfiguringGlobal()) {
        apply_funcs.push_back([this](bool powered_on) {
            Settings::values.bg_red.SetValue(static_cast<u8>(bg_color.red()));
            Settings::values.bg_green.SetValue(static_cast<u8>(bg_color.green()));
            Settings::values.bg_blue.SetValue(static_cast<u8>(bg_color.blue()));
        });
    } else {
        QPushButton* bg_restore_button = ConfigurationShared::Widget::CreateRestoreGlobalButton(
            Settings::values.bg_red.UsingGlobal(), ui->bg_widget);
        ui->bg_widget->layout()->addWidget(bg_restore_button);

        bg_restore_button->connect(bg_restore_button, &QAbstractButton::clicked,
                                   [bg_restore_button, this](bool) {
                                       const int r = Settings::values.bg_red.GetValue(true);
                                       const int g = Settings::values.bg_green.GetValue(true);
                                       const int b = Settings::values.bg_blue.GetValue(true);
                                       UpdateBackgroundColorButton(QColor::fromRgb(r, g, b));

                                       bg_restore_button->setVisible(false);
                                       bg_restore_button->setEnabled(false);
                                   });

        ui->bg_button->connect(ui->bg_button, &QAbstractButton::clicked, [bg_restore_button](bool) {
            bg_restore_button->setVisible(true);
            bg_restore_button->setEnabled(true);
        });

        apply_funcs.push_back([bg_restore_button, this](bool powered_on) {
            const bool using_global = !bg_restore_button->isEnabled();
            Settings::values.bg_red.SetGlobal(using_global);
            Settings::values.bg_green.SetGlobal(using_global);
            Settings::values.bg_blue.SetGlobal(using_global);
            if (!using_global) {
                Settings::values.bg_red.SetValue(static_cast<u8>(bg_color.red()));
                Settings::values.bg_green.SetValue(static_cast<u8>(bg_color.green()));
                Settings::values.bg_blue.SetValue(static_cast<u8>(bg_color.blue()));
            }
        });
    }
}

const QString ConfigureGraphics::TranslateVSyncMode(VkPresentModeKHR mode,
                                                    Settings::RendererBackend backend) const {
    switch (mode) {
    case VK_PRESENT_MODE_IMMEDIATE_KHR:
        return (backend == Settings::RendererBackend::OpenGL_GLSL ||
                backend == Settings::RendererBackend::OpenGL_GLASM ||
                backend == Settings::RendererBackend::OpenGL_SPIRV)
                   ? tr("Off")
                   : QStringLiteral("Immediate (%1)").arg(tr("VSync Off"));
    case VK_PRESENT_MODE_MAILBOX_KHR:
        return QStringLiteral("Mailbox (%1)").arg(tr("Recommended"));
    case VK_PRESENT_MODE_FIFO_KHR:
        return (backend == Settings::RendererBackend::OpenGL_GLSL ||
                backend == Settings::RendererBackend::OpenGL_GLASM ||
                backend == Settings::RendererBackend::OpenGL_SPIRV)
                   ? tr("On")
                   : QStringLiteral("FIFO (%1)").arg(tr("VSync On"));
    case VK_PRESENT_MODE_FIFO_RELAXED_KHR:
        return QStringLiteral("FIFO Relaxed");
    default:
        return {};
    }
}

void ConfigureGraphics::ApplyConfiguration() {
    const bool powered_on = system.IsPoweredOn();
    for (const auto& func : apply_funcs) {
        func(powered_on);
    }

    UpdateVsyncSetting();

    Settings::values.vulkan_device.SetGlobal(true);
    auto const index = Settings::EnumMetadata<Settings::RendererBackend>::Index();
    if (Settings::IsConfiguringGlobal() ||
        (!Settings::IsConfiguringGlobal() && api_restore_global_button->isEnabled())) {
        auto backend =
            index >= combobox_translations.size() ||
                    size_t(api_combobox->currentIndex()) >= combobox_translations.at(index).size()
                ? Settings::values.renderer_backend.GetValue()
                : Settings::RendererBackend(
                      combobox_translations.at(index)[api_combobox->currentIndex()].first);
        switch (backend) {
        case Settings::RendererBackend::Vulkan:
            Settings::values.vulkan_device.SetGlobal(Settings::IsConfiguringGlobal());
            Settings::values.vulkan_device.SetValue(vulkan_device_combobox->currentIndex());
            break;
        case Settings::RendererBackend::OpenGL_GLSL:
        case Settings::RendererBackend::OpenGL_SPIRV:
        case Settings::RendererBackend::OpenGL_GLASM:
        case Settings::RendererBackend::Null:
        case Settings::RendererBackend::Metal:
            break;
        }
    }
}

void ConfigureGraphics::changeEvent(QEvent* event) {
    if (event->type() == QEvent::LanguageChange) {
        RetranslateUI();
    }

    QWidget::changeEvent(event);
}

void ConfigureGraphics::RetranslateUI() {
    ui->retranslateUi(this);
    UiText::ApplyCatalogue(this);
}

void ConfigureGraphics::UpdateBackgroundColorButton(QColor color) {
    bg_color = color;

    QPixmap pixmap(ui->bg_button->size());
    pixmap.fill(bg_color);

    const QIcon color_icon(pixmap);
    ui->bg_button->setIcon(color_icon);
}

void ConfigureGraphics::UpdateAPILayout() {
    bool runtime_lock = !system.IsPoweredOn();
    bool need_global = !(Settings::IsConfiguringGlobal() || api_restore_global_button->isEnabled());
    vulkan_device = Settings::values.vulkan_device.GetValue(need_global);
    vulkan_device_widget->setEnabled(!need_global && runtime_lock);

    const auto current_backend = GetCurrentGraphicsBackend();
    const bool is_vulkan = current_backend == Settings::RendererBackend::Vulkan;
    vulkan_device_widget->setVisible(is_vulkan);
    if (is_vulkan && int(vulkan_device) < vulkan_device_combobox->count()) {
        vulkan_device_combobox->setCurrentIndex(vulkan_device);
    }
}

void ConfigureGraphics::RetrieveVulkanDevices() {
    vulkan_devices.clear();
    vulkan_devices.reserve(records.size());
    device_present_modes.clear();
    device_present_modes.reserve(records.size());
    for (const auto& record : records) {
        vulkan_devices.push_back(QString::fromStdString(record.name));
        device_present_modes.push_back(record.vsync_support);

        if (record.has_broken_compute) {
            expose_compute_option();
        }
    }
}

void ConfigureGraphics::UpdateFeatureAvailability() {
    [[maybe_unused]] const bool vulkan =
        GetCurrentGraphicsBackend() == Settings::RendererBackend::Vulkan;
    const int device = vulkan_device_combobox != nullptr ? vulkan_device_combobox->currentIndex()
                                                         : -1;
    [[maybe_unused]] const VkDeviceInfo::Record* record =
        device >= 0 && size_t(device) < records.size() ? &records[device] : nullptr;

    const int best_discrete_index = VkDeviceInfo::FindBestDiscreteDevice(records);
    [[maybe_unused]] const int best_rtx_index =
        VkDeviceInfo::FindBestDevice(records, [](const auto& r) { return r.dlss_capable; });
    [[maybe_unused]] const int best_fsr4_index =
        VkDeviceInfo::FindBestDevice(records, [](const auto& r) { return r.fsr4_capable; });

    if (discrete_gpu_hint_widget != nullptr && discrete_gpu_hint_label != nullptr) {
        const bool show_hint = vulkan && record != nullptr && record->integrated &&
                               best_discrete_index >= 0 && vulkan_device_widget != nullptr &&
                               vulkan_device_widget->isEnabled();
        if (show_hint) {
            discrete_gpu_hint_label->setText(
                tr("<span style=\"color:#d29922\">&#9888;</span> Integrated GPU selected — "
                   "<b>%1</b> is available.")
                    .arg(QString::fromStdString(records[best_discrete_index].name).toHtmlEscaped()));
        }
        discrete_gpu_hint_widget->setVisible(show_hint);
    }

#ifdef EGLANTINE_STREAMLINE
    const bool dlss_selected_capable = vulkan && record != nullptr && record->dlss_capable;
    const bool dlss_offered = vulkan && best_rtx_index >= 0;
#else
    [[maybe_unused]] constexpr bool dlss_selected_capable = false;
    constexpr bool dlss_offered = false;
#endif
    [[maybe_unused]] const bool fsr4_selected_capable =
        vulkan && record != nullptr && record->fsr4_capable;
    [[maybe_unused]] const bool fsr4_offered = vulkan && best_fsr4_index >= 0;

    const auto upscaler_is = [this](Settings::Upscaler value) {
        return upscaler_combobox != nullptr &&
               upscaler_combobox->currentData().toUInt() == static_cast<u32>(value);
    };
    [[maybe_unused]] const bool frame_gen_dlssg_chosen =
        frame_gen_checkbox != nullptr && frame_gen_checkbox->isChecked() &&
        frame_gen_type_combobox != nullptr &&
        frame_gen_type_combobox->currentData().toUInt() ==
            static_cast<u32>(Settings::FrameGenType::NvidiaDlssG);

    [[maybe_unused]] const bool dlss_on = dlss_offered && upscaler_is(Settings::Upscaler::Dlss);
    if (nvidia_dlss_quality_widget != nullptr) {
        nvidia_dlss_quality_widget->setVisible(dlss_on);
    }
    if (nvidia_dlss_preset_widget != nullptr) {
        nvidia_dlss_preset_widget->setVisible(dlss_on);
    }
#ifdef EGLANTINE_STREAMLINE
    if (nvidia_dlss_panel != nullptr) {
        const VkDeviceInfo::Record* dlss_record =
            dlss_selected_capable ? record
                                  : (best_rtx_index >= 0 ? &records[best_rtx_index] : record);
        nvidia_dlss_panel->SetDevice(
            dlss_record != nullptr ? QString::fromStdString(dlss_record->name) : QString(),
            dlss_selected_capable || (dlss_offered && !upscaler_is(Settings::Upscaler::Dlss)));
        nvidia_dlss_panel->SetFrameGenWanted(frame_gen_dlssg_chosen);
        nvidia_dlss_panel->SetUpscalerSelected(upscaler_is(Settings::Upscaler::Dlss));
        nvidia_dlss_panel->setVisible(upscaler_is(Settings::Upscaler::Dlss) ||
                                      frame_gen_dlssg_chosen ||
                                      (dlss_offered && upscaler_is(Settings::Upscaler::Auto)));
    }
    // Ocarina of Time's fixed sizes are DLSS's input when DLSS Quality is Automatic: name the mode
    // each one corresponds to.
    if (oot_resolution_combobox != nullptr) {
        const bool auto_dlss_quality =
            nvidia_dlss_quality_combobox == nullptr ||
            nvidia_dlss_quality_combobox->currentData().toUInt() ==
                static_cast<u32>(Settings::DlssQuality::Auto);
        const auto& names =
            combobox_translations.at(Settings::EnumMetadata<Settings::OotResolution>::Index());
        for (int i = 0; i < oot_resolution_combobox->count(); ++i) {
            const u32 value = oot_resolution_combobox->itemData(i).toUInt();
            const auto name = std::ranges::find(names, value, &std::pair<u32, QString>::first);
            if (name == names.end()) {
                continue;
            }
            QString mode;
            if (dlss_on && auto_dlss_quality) {
                const u32 dlss_quality_enum =
                    Settings::EnumMetadata<Settings::DlssQuality>::Index();
                switch (static_cast<Settings::OotResolution>(value)) {
                case Settings::OotResolution::Full:
                    mode = QStringLiteral("DLSS %1").arg(UiText::ChoiceLabel(
                        dlss_quality_enum, static_cast<u32>(Settings::DlssQuality::Quality)));
                    break;
                case Settings::OotResolution::Percent87:
                    mode = QStringLiteral("DLSS %1").arg(UiText::ChoiceLabel(
                        dlss_quality_enum, static_cast<u32>(Settings::DlssQuality::Balanced)));
                    break;
                case Settings::OotResolution::Percent75:
                    mode = QStringLiteral("DLSS %1").arg(UiText::ChoiceLabel(
                        dlss_quality_enum, static_cast<u32>(Settings::DlssQuality::Performance)));
                    break;
                case Settings::OotResolution::Percent50:
                case Settings::OotResolution::Percent40:
                case Settings::OotResolution::Percent33:
                    mode = QStringLiteral("DLSS %1").arg(UiText::ChoiceLabel(
                        dlss_quality_enum,
                        static_cast<u32>(Settings::DlssQuality::UltraPerformance)));
                    break;
                default:
                    break;
                }
            }
            oot_resolution_combobox->setItemText(
                i, mode.isEmpty() ? name->second : tr("%1 - %2 to 4K").arg(name->second, mode));
        }
    }
#endif
#if defined(_WIN32) || (defined(__linux__) && !defined(__ANDROID__))
    if (amd_fsr4_panel != nullptr) {
        const VkDeviceInfo::Record* fsr4_record =
            fsr4_selected_capable ? record
                                  : (best_fsr4_index >= 0 ? &records[best_fsr4_index] : record);
        amd_fsr4_panel->SetDevice(
            fsr4_record != nullptr ? QString::fromStdString(fsr4_record->name) : QString(),
            fsr4_selected_capable || (fsr4_offered && !upscaler_is(Settings::Upscaler::Fsr4)));
        amd_fsr4_panel->SetUpscalerSelected(upscaler_is(Settings::Upscaler::Fsr4));
        amd_fsr4_panel->setVisible(upscaler_is(Settings::Upscaler::Fsr4) ||
                                   (fsr4_offered && upscaler_is(Settings::Upscaler::Auto)));
    }
#endif

    // The native Metal renderer runs FSR 3, MetalFX and LSFG frame generation.
#ifdef LICHII_METAL_RENDERER
    [[maybe_unused]] const bool metal =
        GetCurrentGraphicsBackend() == Settings::RendererBackend::Metal;
#else
    [[maybe_unused]] constexpr bool metal = false;
#endif
#ifdef HAS_LSFG
    ConfigurationShared::SetUnavailable(
        frame_gen_widget,
        vulkan || metal ? QString() : tr("Needs the Vulkan or Metal renderer."));
#else
    // Not in this build.
    if (frame_gen_widget != nullptr) {
        frame_gen_widget->setVisible(false);
    }
#endif

#ifdef LICHII_METALFX
    // Spatial, temporal and frame generation MetalFX run on the native Metal renderer too
    // (docs/nx2/metal-coverage.md N06, O03, N09).
    const bool metalfx = vulkan || metal;
    const bool metalfx_frame_gen = vulkan || metal;
#else
    constexpr bool metalfx = false;
    constexpr bool metalfx_frame_gen = false;
#endif
    // Platform-unrelated entries hide unless currently chosen; desktop hardware-gated entries stay
    // visible on Windows/Linux with an explanatory disabled tooltip.
    GateComboItem(scaling_filter_combobox, Settings::ScalingFilter::MetalFxSpatial, metalfx);
    GateComboItem(frame_gen_type_combobox, Settings::FrameGenType::MetalFx, metalfx_frame_gen);
    GateComboItem(upscaler_combobox, Settings::Upscaler::MetalFx, metalfx);
#ifdef __APPLE__
    GateComboItem(upscaler_combobox, Settings::Upscaler::Dlss, false);
    GateComboItem(upscaler_combobox, Settings::Upscaler::Fsr4, false);
    GateComboItem(frame_gen_type_combobox, Settings::FrameGenType::NvidiaDlssG, false);
#else
    EnableComboItem(upscaler_combobox, Settings::Upscaler::Dlss, dlss_offered,
                    tr("Requires Windows, the Vulkan renderer, and an NVIDIA GeForce RTX GPU."));
    EnableComboItem(
        upscaler_combobox, Settings::Upscaler::Fsr4, fsr4_offered,
        tr("Requires an AMD Radeon RX 6000/7000/9000 or Ryzen RDNA 3+ GPU on Vulkan (falls back "
           "to %1).")
            .arg(UiText::ChoiceLabel("upscaler", "Fsr3")));
    EnableComboItem(frame_gen_type_combobox, Settings::FrameGenType::NvidiaDlssG, dlss_offered,
                    tr("Requires %1 on an NVIDIA GeForce RTX 40/50 GPU.")
                        .arg(UiText::ChoiceLabel("upscaler", "Dlss")));
#endif
}

void ConfigureGraphics::UpdateLosslessStatus() {
#ifdef HAS_LSFG
    using VideoCore::FrameGen::LosslessStatus;
    const auto status = VideoCore::FrameGen::GetInstalledLosslessStatus();
    const QString where = QDir::toNativeSeparators(
        QString::fromStdU16String(VideoCore::FrameGen::GetLosslessDllPath().u16string()));
    QString problem;
    if (status == LosslessStatus::NotInstalled) {
        problem = tr("Lossless.dll not found in %1. Copy it from your own Lossless Scaling "
                     "install (Steam\\steamapps\\common\\Lossless Scaling) or choose it here.")
                      .arg(where);
    } else if (status != LosslessStatus::Ok) {
        problem = tr("%1 can't be used: it lacks the frame generation shaders (another Lossless "
                     "Scaling version?).")
                      .arg(where);
    }
    if (lossless_status_label != nullptr) {
        lossless_status_label->setText(problem.isEmpty() ? tr("Lossless.dll found.") : problem);
    }
    // Without the file LSFG generates nothing; the item stays selectable if it is already chosen.
    EnableComboItem(frame_gen_type_combobox, Settings::FrameGenType::Lsfg, problem.isEmpty(),
                    problem);
#endif
}

Settings::RendererBackend ConfigureGraphics::GetCurrentGraphicsBackend() const {
    const auto selected_backend = [&]() {
        auto const index = Settings::EnumMetadata<Settings::RendererBackend>::Index();
        if (!Settings::IsConfiguringGlobal() && !api_restore_global_button->isEnabled())
            return Settings::values.renderer_backend.GetValue(true);
        return index >= combobox_translations.size() || size_t(api_combobox->currentIndex()) >=
                                                            combobox_translations.at(index).size()
                   ? Settings::values.renderer_backend.GetValue()
                   : Settings::RendererBackend(
                         combobox_translations.at(index).at(api_combobox->currentIndex()).first);
    }();

    if (selected_backend == Settings::RendererBackend::Vulkan &&
        UISettings::values.has_broken_vulkan)
        return Settings::RendererBackend::OpenGL_GLSL;
    return selected_backend;
}
