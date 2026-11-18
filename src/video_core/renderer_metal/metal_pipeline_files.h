// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// A game's Metal pipeline files, next to its pipeline archive (EDEN_NX2_METAL_PIPELINE_FILE):
// the programs of every pipeline a run made (VideoCommon::SerializePipeline records, as Vulkan's
// cache but in a file of its own), the render states made from them, and which pipelines recent
// runs used. PipelineCache::LoadDiskResources builds them before play. Plain C++.

#pragma once

#include <filesystem>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "common/thread_worker.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_pipeline_translate.h"

namespace Metal {

class PipelineFiles {
public:
    /// Versions of the records and the states file; a file of another version is started again.
    /// The states version changes with RenderPipelineDesc's meaning (its size is checked too).
    /// The records hold GraphicsPipelineKey's bytes: 2 added the transform feedback layout.
    static constexpr u32 RecordsVersion = 2;
    static constexpr u32 StatesVersion = 2;
    /// Runs a pipeline stays on the used list without being used, and is still built at boot
    /// (Vulkan on MoltenVK builds what the last 8 runs used).
    static constexpr u8 MaxListedAge = 16;
    static constexpr u8 MaxBootAge = 8;
    /// A run that used fewer pipelines than this leaves the used list as it is.
    static constexpr size_t MinimumUsed = 64;

    /// A render state of a pipeline; the descriptor's function fields are zero.
    struct StateRecord {
        GraphicsPipelineKey key;
        RenderPipelineDesc desc;
    };
    static_assert(std::is_trivially_copyable_v<StateRecord>);

    /// EDEN_NX2_METAL_PIPELINE_FILE: 0 no files, 1 boot builds the pipelines on the used list,
    /// 2 boot builds every recorded pipeline.
    [[nodiscard]] static u32 Mode();

    PipelineFiles();
    ~PipelineFiles();

    PipelineFiles(const PipelineFiles&) = delete;
    PipelineFiles& operator=(const PipelineFiles&) = delete;

    /// Names the files in the game's shader directory, reads the used list and starts the
    /// thread that writes them.
    void Open(const std::filesystem::path& directory);

    [[nodiscard]] bool IsOpen() const noexcept {
        return writer.has_value();
    }

    /// The records file (VideoCommon::LoadPipelines and SerializePipeline).
    [[nodiscard]] const std::filesystem::path& Records() const noexcept {
        return records_path;
    }

    /// Marks the boot build as running; false when the last one never finished (the process
    /// died or was stopped in it), and then this boot should build nothing.
    [[nodiscard]] bool BeginBoot();

    void EndBoot();

    /// The render states of earlier runs; a states file of another layout is removed.
    [[nodiscard]] std::vector<StateRecord> LoadStates();

    /// Appends a render state Metal made, once per state.
    void RecordState(const GraphicsPipelineKey& key, const RenderPipelineDesc& desc);

    /// Whether boot builds the pipeline with this key hash: all without a used list.
    [[nodiscard]] bool BuildAtBoot(u64 key_hash) const;

    void NoteUsed(u64 key_hash);

    /// Pipelines were first used since the used list was last written.
    [[nodiscard]] bool Dirty() const noexcept {
        return used_changed;
    }

    /// Writes the used list when pipelines were used since the last call.
    void SaveUsed();

    /// Runs work on the writing thread, in order.
    template <typename Work>
    void Queue(Work&& work) {
        writer->QueueWork(std::forward<Work>(work));
    }

    /// Waits for everything queued to be written.
    void Wait();

private:
    std::filesystem::path records_path;
    std::filesystem::path states_path;
    std::filesystem::path used_path;
    std::filesystem::path boot_path;
    /// Hashes of the states the states file holds.
    std::unordered_set<u64> known_states;
    /// Ages by key hash, as the used list held them when it was opened.
    std::unordered_map<u64, u8> listed;
    std::unordered_set<u64> used;
    bool has_list{};
    bool used_changed{};
    std::optional<Common::ThreadWorker> writer;
};

} // namespace Metal
