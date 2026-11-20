// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <memory>
#include <system_error>
#include <utility>

#include <QCloseEvent>
#include <QHBoxLayout>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QVBoxLayout>

#include "common/settings.h"
#include "video_core/shader_warm/shader_warm_builder.h"
#include "yuzu/shader_warm_dialog.h"

namespace ShaderWarm = VideoCommon::ShaderWarm;

bool ShaderWarmFilesExist(u64 title_id) {
    return title_id != 0 && ShaderWarm::HasWarmFile(title_id);
}

bool DeleteShaderWarmFiles(u64 title_id, std::string& error) {
    return ShaderWarm::DeleteWarmFiles(title_id, &error);
}

namespace {

// The builder reports its stage in English; show the known ones translated.
QString StageText(const std::string& stage) {
    if (stage == "Reading game files") {
        return ShaderWarmDialog::tr("Reading game files");
    }
    if (stage == "Extracting shaders") {
        return ShaderWarmDialog::tr("Extracting shaders");
    }
    if (stage == "Converting") {
        return ShaderWarmDialog::tr("Converting shaders");
    }
    if (stage == "Writing") {
        return ShaderWarmDialog::tr("Writing the precache file");
    }
    if (stage == "Compiling Metal compute shaders") {
        return ShaderWarmDialog::tr("Compiling Metal compute shaders");
    }
    if (stage == "Compiling Metal graphics stages") {
        return ShaderWarmDialog::tr("Compiling Metal graphics stages");
    }
    return QString::fromStdString(stage);
}

} // namespace

/// Builds the stage label, progress bar, result text and the Cancel/Close button.
ShaderWarmDialog::ShaderWarmDialog(Core::System& system_, std::filesystem::path game_path_,
                                   u64 title_id_, const QString& game_name, QWidget* parent,
                                   bool compile_gpu_, bool extract_files_)
    : QDialog(parent), system{system_}, game_path{std::move(game_path_)}, title_id{title_id_},
      compile_gpu{compile_gpu_}, extract_files{extract_files_} {
    setWindowTitle(compile_gpu ? tr("Build Shader Cache: %1").arg(game_name)
                               : tr("Precache shaders: %1").arg(game_name));
    setWindowFlag(Qt::WindowContextHelpButtonHint, false);
    setModal(false);
    setMinimumWidth(420);

    stage_label = new QLabel(tr("Starting"), this);
    progress_bar = new QProgressBar(this);
    progress_bar->setRange(0, 0);
    result_label = new QLabel(this);
    result_label->setWordWrap(true);
    result_label->setTextInteractionFlags(Qt::TextSelectableByMouse);
    result_label->setVisible(false);
    button = new QPushButton(tr("Cancel"), this);
    connect(button, &QPushButton::clicked, this, [this] {
        if (running) {
            RequestStop();
        } else {
            reject();
        }
    });

    auto* buttons = new QHBoxLayout;
    buttons->addStretch(1);
    buttons->addWidget(button);

    auto* layout = new QVBoxLayout(this);
    layout->addWidget(stage_label);
    layout->addWidget(progress_bar);
    layout->addWidget(result_label);
    layout->addLayout(buttons);
}

/// Stops and joins a running build.
ShaderWarmDialog::~ShaderWarmDialog() {
    if (worker.joinable()) {
        worker.request_stop();
        worker.join();
    }
}

/// Starts the build on a worker thread. Progress (at most every 100 ms unless the stage changes)
/// and the result, failures included, come back on the UI thread.
void ShaderWarmDialog::Start() {
    if (running) {
        return;
    }
    running = true;
    stopping = false;
    needs_boot_gpu_compile = false;
    worker = std::jthread([this](std::stop_token stop) {
        auto last_update = std::chrono::steady_clock::time_point{};
        std::string last_stage;
        const auto on_progress = [this, &last_update, &last_stage](const ShaderWarm::Progress& p) {
            const auto now = std::chrono::steady_clock::now();
            const bool stage_changed = p.stage != last_stage;
            const bool stage_done = p.total != 0 && p.done >= p.total;
            if (!stage_changed && !stage_done &&
                now - last_update < std::chrono::milliseconds(100)) {
                return;
            }
            last_update = now;
            last_stage = p.stage;
            QMetaObject::invokeMethod(
                this,
                [this, stage = StageText(p.stage), done = static_cast<qulonglong>(p.done),
                 total = static_cast<qulonglong>(p.total)] { OnProgress(stage, done, total); },
                Qt::QueuedConnection);
        };
        // An exception must not leave the thread (std::terminate would close the emulator):
        // report it as a failed build instead.
        auto result = std::make_shared<ShaderWarm::Result>();
        auto gpu_summary = std::make_shared<ShaderWarm::GpuWarmSummary>();
        try {
            if (extract_files) {
                *result = ShaderWarm::BuildWarmFiles(system, game_path, title_id, on_progress, stop);
            } else {
                result->ok = true;
            }
            if (compile_gpu && result->ok && !stop.stop_requested()) {
                *gpu_summary = ShaderWarm::CompileWarmShadersOffline(title_id, stop, on_progress);
            }
        } catch (const std::exception& e) {
            result->ok = false;
            result->error = e.what();
        } catch (...) {
            result->ok = false;
            result->error = "unknown error";
        }
        const bool stopped = stop.stop_requested();
        QMetaObject::invokeMethod(
            this, [this, result, gpu_summary, stopped] { OnFinished(*result, *gpu_summary, stopped); },
            Qt::QueuedConnection);
    });
}

/// Asks a running build to stop and disables the button until it has.
void ShaderWarmDialog::RequestStop() {
    if (!running || stopping) {
        return;
    }
    stopping = true;
    worker.request_stop();
    stage_label->setText(tr("Stopping"));
    button->setEnabled(false);
}

/// Shows the stage and its progress; a busy bar when the total is unknown.
void ShaderWarmDialog::OnProgress(const QString& stage, qulonglong done, qulonglong total) {
    if (stopping) {
        return;
    }
    if (total > 0) {
        stage_label->setText(tr("%1: %L2 / %L3").arg(stage).arg(done).arg(total));
        progress_bar->setRange(0, 1000);
        progress_bar->setValue(static_cast<int>((std::min<qulonglong>)(done * 1000 / total, 1000)));
    } else {
        stage_label->setText(done > 0 ? tr("%1: %L2").arg(stage).arg(done) : stage);
        progress_bar->setRange(0, 0);
    }
}

/// Joins the worker and shows the result: program counts, stopped, or the error text.
void ShaderWarmDialog::OnFinished(const ShaderWarm::Result& result,
                                  const ShaderWarm::GpuWarmSummary& gpu_summary, bool stopped) {
    running = false;
    if (worker.joinable()) {
        worker.join();
    }
    const bool gpu_failed = compile_gpu && gpu_summary.supported && !gpu_summary.ok;
    const bool overall_ok = result.ok && !stopped && !gpu_failed;
    needs_boot_gpu_compile = overall_ok && compile_gpu && !gpu_summary.supported;

    progress_bar->setRange(0, 1);
    progress_bar->setValue(overall_ok ? 1 : 0);

    QString text;
    if (overall_ok) {
        const size_t extracted =
            result.compute_programs + result.vertex_programs + result.fragment_programs;
        if (compile_gpu && gpu_summary.supported) {
            stage_label->setText(tr("Shader cache is ready."));
            if (extract_files) {
                text = tr("Ready: %1 programs extracted and %2 GPU shaders precompiled (%3 already "
                          "cached).")
                           .arg(static_cast<qulonglong>(extracted))
                           .arg(static_cast<qulonglong>(gpu_summary.built))
                           .arg(static_cast<qulonglong>(gpu_summary.skipped));
            } else {
                text = tr("Ready: %1 GPU shaders precompiled (%2 already cached).")
                           .arg(static_cast<qulonglong>(gpu_summary.built))
                           .arg(static_cast<qulonglong>(gpu_summary.skipped));
            }
            if (result.skipped > 0 || gpu_summary.failed > 0) {
                text += QLatin1Char('\n') +
                        tr("Skipped or unsupported: %L1")
                            .arg(static_cast<qulonglong>(result.skipped + gpu_summary.failed));
            }
        } else if (extract_files) {
            stage_label->setText(tr("Shader precache is ready."));
            if (extracted == 0) {
                text = result.archives == 0
                           ? tr("This game keeps its shaders in a format Lichii can't read, so "
                                "none are prepared from the game files.")
                           : tr("The game files have no shaders to prepare.");
                text += QLatin1Char(' ') +
                        tr("The shaders your earlier play sessions saved are still precached.");
            } else {
                text = tr("Extracted programs: %L1 (compute: %L2, vertex: %L3, fragment: %L4)")
                           .arg(static_cast<qulonglong>(extracted))
                           .arg(static_cast<qulonglong>(result.compute_programs))
                           .arg(static_cast<qulonglong>(result.vertex_programs))
                           .arg(static_cast<qulonglong>(result.fragment_programs));
            }
            if (result.skipped > 0) {
                text += QLatin1Char('\n') +
                        tr("Skipped (could not be read or converted): %L1")
                            .arg(static_cast<qulonglong>(result.skipped));
            }
            text += QStringLiteral("\n\n") +
                    tr("It is built on the loading screen when Precache shaders before game start "
                       "is on.");
        } else {
            stage_label->setText(tr("Shader precache is ready."));
        }
    } else if (stopped) {
        stage_label->setText(tr("Shader precache stopped."));
    } else {
        stage_label->setText(tr("Shader precache could not be built."));
        const std::string& err = !result.ok ? result.error : gpu_summary.error;
        text = err.empty() ? tr("No details were given.") : QString::fromStdString(err);
    }
    result_label->setText(text);
    result_label->setVisible(!text.isEmpty());
    button->setText(tr("Close"));
    button->setEnabled(true);
    adjustSize();
    emit BuildFinished(overall_ok);
}

/// Closing while a build runs only asks it to stop.
void ShaderWarmDialog::closeEvent(QCloseEvent* event) {
    if (running) {
        RequestStop();
        event->ignore();
        return;
    }
    QDialog::closeEvent(event);
}

/// Escape or Cancel while a build runs only asks it to stop.
void ShaderWarmDialog::reject() {
    if (running) {
        RequestStop();
        return;
    }
    QDialog::reject();
}
