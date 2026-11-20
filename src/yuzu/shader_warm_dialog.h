// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <string>
#include <thread>

#include <QDialog>

#include "common/common_types.h"

namespace Core {
class System;
}

namespace VideoCommon::ShaderWarm {
struct GpuWarmSummary;
struct Progress;
struct Result;
} // namespace VideoCommon::ShaderWarm

class QCloseEvent;
class QLabel;
class QProgressBar;
class QPushButton;

/// True when the title has a shader precache file.
bool ShaderWarmFilesExist(u64 title_id);

/// Removes the title's precache files. Returns false and fills `error` on failure.
bool DeleteShaderWarmFiles(u64 title_id, std::string& error);

/// Window that builds a game's shader precache file on a worker thread and shows the stage, the
/// progress and the result. Closing it while it works stops the build first.
class ShaderWarmDialog : public QDialog {
    Q_OBJECT

public:
    explicit ShaderWarmDialog(Core::System& system, std::filesystem::path game_path, u64 title_id,
                              const QString& game_name, QWidget* parent = nullptr,
                              bool compile_gpu = true, bool extract_files = true);
    ~ShaderWarmDialog() override;

    void Start();
    bool IsRunning() const {
        return running;
    }
    /// True when offline GPU compilation is unsupported (e.g. Vulkan) and a BuildOnly boot is
    /// needed to compile GPU pipelines into the driver/pipeline cache.
    bool NeedsBootGpuCompile() const {
        return needs_boot_gpu_compile;
    }
    /// The title whose precache file this dialog builds.
    u64 TitleId() const {
        return title_id;
    }

signals:
    void BuildFinished(bool ok);

protected:
    void closeEvent(QCloseEvent* event) override;
    void reject() override;

private:
    void RequestStop();
    void OnProgress(const QString& stage, qulonglong done, qulonglong total);
    void OnFinished(const VideoCommon::ShaderWarm::Result& result,
                    const VideoCommon::ShaderWarm::GpuWarmSummary& gpu_summary, bool stopped);

    Core::System& system;
    std::filesystem::path game_path;
    u64 title_id;
    bool compile_gpu{true};
    bool extract_files{true};
    bool needs_boot_gpu_compile{};

    QLabel* stage_label{};
    QProgressBar* progress_bar{};
    QLabel* result_label{};
    QPushButton* button{};

    bool running{};
    bool stopping{};
    std::jthread worker;
};
