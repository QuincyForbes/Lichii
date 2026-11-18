// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// EDEN_NX2_FRAME_DUMP=<dir> on Metal, as Vulkan's FrameDump (vk_frame_dump.cpp) reads one frame
// back: the colour targets of every render pass, the images every dispatch writes and the
// presented image, each as <name>.raw.zst, a PNG preview and a line of manifest.jsonl with its
// min / max / mean, NaN and infinity counts (tools/nx2/frame_dump_index.py reads it). Also the
// debug server's capture_frame (Arm). Not ported: the sampled textures, bound buffers and draw
// probes of EDEN_NX2_FRAME_DUMP_SAMPLED. Plain C++.

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "video_core/renderer_metal/metal_object.h"
#include "video_core/surface.h"

namespace Metal {

class Device;
class Scheduler;

class FrameDump {
public:
    /// One image to read back: a texture of the renderer, as its guest image names it.
    struct Source {
        TextureHandle texture{};
        VideoCore::Surface::PixelFormat format{};
        u32 width{};
        u32 height{};
        GPUVAddr gpu_addr{};
        u32 slot{};
        u32 level{};
        u32 layer{};
        bool msaa{};
        bool rescaled{};
        /// MTLPixelFormat of the texture: its texel size is the host's, not the guest format's.
        u32 host_format{};
    };

    /// A texture a draw or dispatch samples (a null one has no image behind its TIC).
    struct Sampled {
        u32 stage{};
        u32 tic{};
        bool null{};
        std::optional<Source> source;
        /// The sampler's state as JSON (`,"sampler":{...}`), empty without one.
        std::string sampler;
        /// What identifies the image seen: its address, then format << 16 | level << 8 | layer.
        GPUVAddr key_addr{};
        u32 key_extra{};
        bool block_compressed{};
        /// For the record of an image that can't be read back.
        u32 format{};
        u32 width{};
        u32 height{};
    };

    /// A uniform or storage buffer binding. Its bytes come from guest memory, or from the host
    /// buffer when the GPU wrote the range and guest memory may be stale.
    struct BoundBuffer {
        u32 stage{};
        bool storage{};
        u32 index{};
        bool written{};
        GPUVAddr gpu_addr{};
        u64 device_addr{};
        u32 bound_size{};
        u32 size{};
        bool gpu_modified{};
        BufferHandle host{};
        u32 host_offset{};
        std::vector<u8> guest;
        /// EDEN_NX2_FRAME_DUMP_HOST: also read the host buffer the draw reads ("host_copy").
        bool also_host{};
    };

    /// A draw's state as the CPU knows it (Vulkan also counts its primitives and fragments).
    struct DrawRecord {
        bool mesh{};
        bool indirect{};
        u32 first{};
        u32 elements{};
        u32 instances{};
        u32 alpha_test{};
        f32 alpha_ref{};
        bool alpha_to_coverage{};
    };

    explicit FrameDump(const Device& device, Scheduler& scheduler);
    ~FrameDump();

    /// The debug server's capture_frame: captures the next frame into `directory` and reports
    /// `id` finished (Nx2Debug::FinishCapture) once its manifest is written.
    void Arm(u64 id, std::filesystem::path directory, bool sampled = false, bool buffers = false);

    /// Whether any trigger (environment or debug server) can start a capture.
    [[nodiscard]] bool Enabled() const noexcept {
        return enabled || requested;
    }

    [[nodiscard]] bool Capturing() const noexcept {
        return capturing;
    }

    /// Whether the capture reads back sampled textures (EDEN_NX2_FRAME_DUMP_SAMPLED, or the
    /// server's `sampled`) and the bound buffers.
    [[nodiscard]] bool CapturingSampled() const noexcept {
        return capturing && (sampled || buffers);
    }
    [[nodiscard]] bool CapturingTextures() const noexcept {
        return capturing && sampled;
    }
    [[nodiscard]] bool CapturingBuffers() const noexcept {
        return capturing && buffers;
    }

    /// Reads back the textures of the current draw not yet seen in this pass. True when copies
    /// were recorded.
    bool NoteSampled(std::span<const Sampled> textures);
    /// The textures of the dispatch about to run not yet seen for this shader.
    void NoteDispatchSampled(u64 shader_hash, std::span<const Sampled> textures);
    /// The buffers of the current draw (`shader_hashes` by program) or dispatch not yet seen.
    bool NoteBuffers(bool compute, std::span<const u64> shader_hashes,
                     std::span<BoundBuffer> buffers);
    /// The group counts of the dispatch about to run.
    void NoteLaunch(u64 shader_hash, const std::array<u32, 3>& grid,
                    const std::array<u32, 3>& block, std::optional<GPUVAddr> indirect,
                    std::optional<std::array<u32, 3>> indirect_grid);
    /// The draw just counted by NoteDraw.
    void NoteDrawRecord(const DrawRecord& draw);

    /// Starts a pass when `key` (the framebuffer) changes and reads back the previous one.
    /// True when copies were recorded, which ended the open render encoder.
    bool BeginPass(const void* key, std::span<const Source> colors);
    void NoteDraw(std::span<const u64> shader_hashes);
    bool EndPass();
    void CaptureDispatch(u64 shader_hash, std::span<const Source> written);
    void CapturePresent(const Source& source);
    /// A draw or dispatch dropped because its program or pipeline is unavailable.
    void NoteSkipped(bool compute, u64 program, GPUVAddr rt0, std::string_view reason);

    /// Frame end: finishes a capture (reads back, writes the summary), or starts one.
    void TickFrame();

    /// Texel bytes of an uncompressed colour MTLPixelFormat, 0 for the others (not dumped).
    [[nodiscard]] static u32 HostTexelBytes(u32 format) noexcept;

private:
    struct Readback {
        bool raw{};
        Object buffer;
        u8* contents{};
        Source source;
        std::string name;
        std::string tags;
        size_t bytes{};
    };

    struct Job {
        Source source{};
        std::string name;
        std::string tags;
        std::vector<u8> data;
        bool summary{};
        bool truncated{};
        /// Capture the debug server asked for, answered once the summary is written.
        u64 finish_id{};
        bool raw{};
    };

    bool Record(const Source& source, std::string name, std::string tags);
    bool RecordSampled(const Sampled& texture, u32 index, std::string name, std::string tags);
    bool RecordBuffer(BufferHandle source, u32 offset, u32 size, std::string tags);
    void WriteBuffer(const Job& job);
    void Drain();
    void Enqueue(Job job);
    void WriterLoop();
    void Write(const Job& job);
    void WriteFile(const std::string& name, std::span<const u8> data);
    void StopWriter();
    void ResetCapture();
    void StartCapture(double elapsed);

    const Device& device;
    Scheduler& scheduler;

    bool enabled{};
    bool sampled{};
    bool buffers{};
    bool env_sampled{};
    /// EDEN_NX2_FRAME_DUMP_SAMPLED_FROM: passes before this one do not read back textures.
    u32 sampled_from_pass{};
    size_t size_cap{};
    bool capturing{};
    bool done{};
    bool truncated{};
    std::filesystem::path directory;
    std::filesystem::path trigger_file;
    std::filesystem::path base_directory;
    u32 captures{};
    u64 trigger_frame{~0ULL};
    bool pair{};
    bool pair_next{};
    double trigger_seconds{};
    std::chrono::steady_clock::time_point start;
    u64 frame{};

    /// The debug server's request: taken at the next frame end, answered after the summary.
    std::mutex request_mutex;
    std::atomic<bool> requested{};
    u64 request_id{};
    bool request_sampled{};
    bool request_buffers{};
    std::filesystem::path request_directory;
    std::filesystem::path env_directory;
    u64 active_id{};

    std::ofstream manifest;
    std::vector<Readback> readbacks;
    size_t pending_bytes{};
    size_t recorded_bytes{};
    size_t written_bytes{};
    u32 sequence{};
    u32 images{};

    const void* pass_key{};
    std::vector<Source> pass_sources;
    std::vector<u64> pass_hashes;
    std::vector<std::pair<GPUVAddr, u32>> pass_sampled;
    std::vector<std::pair<u32, u32>> pass_null_sampled;
    std::vector<std::tuple<u64, GPUVAddr, u32>> dispatch_sampled;
    std::vector<std::tuple<GPUVAddr, u32>> recorded_blocks;
    std::vector<std::tuple<u64, bool, u32, u64, u32>> seen_buffers;
    std::vector<u64> written_buffers;
    u32 pass_index{};
    u32 pass_first_draw{};
    u32 draws{};
    u32 dispatches{};

    std::thread writer;
    std::mutex queue_mutex;
    std::condition_variable queue_ready;
    std::deque<Job> queue;
    bool stop_writer{};
};

} // namespace Metal
