// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// EDEN_NX2_FLASH_WATCH=<dir> on Metal, as Vulkan's FlashWatch (vk_flash_watch.cpp): every frame a
// compute pass (vulkan_flash_watch.comp through the SPIR-V -> MSL translator) reduces the colour
// targets of every pass, the images every dispatch writes and the presented image to tiles (mean
// and largest channel magnitude, NaN or infinite texels, texels at or above
// EDEN_NX2_FLASH_WATCH_HOT), read back a few frames later; a presented tile brighter than in the
// frames around it is a flash, reported as flash_<frame>.jsonl and .tiles. Plain C++.

#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "video_core/renderer_metal/metal_object.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_shader_translate.h"
#include "video_core/surface.h"

namespace Metal {

class Device;

class FlashWatch {
public:
    explicit FlashWatch(const Device& device, Scheduler& scheduler);
    ~FlashWatch();

    [[nodiscard]] bool Enabled() const noexcept {
        return enabled;
    }

    [[nodiscard]] bool WatchesPasses() const noexcept {
        return enabled && passes;
    }

    enum class Kind : u8 { Pass, Dispatch, Present };

    /// A colour image to reduce: a texture sampled as 2D.
    struct Source {
        TextureHandle texture{};
        VideoCore::Surface::PixelFormat format{};
        u32 width{};
        u32 height{};
        GPUVAddr gpu_addr{};
        u32 slot{};
    };

    /// Starts a pass when `key` changes and reduces the previous one.
    bool BeginPass(const void* key, std::span<const Source> colors);
    void NoteDraw(std::span<const u64> shader_hashes);
    bool EndPass();
    void NoteDispatch(u64 shader_hash, std::span<const Source> written);
    void NotePresent(const Source& source);
    void TickFrame();

    /// One reduced output: its tiles start at first_tile in its frame's tiles.
    struct Output {
        Kind kind;
        u32 index; ///< Pass, dispatch or present number in the frame
        u32 slot;
        u32 draws;
        VideoCore::Surface::PixelFormat format;
        u32 width;
        u32 height;
        u32 grid_x;
        u32 grid_y;
        u32 first_tile;
        GPUVAddr gpu_addr;
        std::vector<u64> shaders;
        /// Matches the same output across frames: kind, format, size and slot (shader for
        /// dispatches), then the number of earlier outputs of the frame with the same.
        std::string key;
    };

    /// Sum of the mean channel magnitudes (float bits), largest channel magnitude (float bits),
    /// NaN or infinite texels, hot texels.
    using Tile = std::array<u32, 4>;

private:
    struct Slot {
        Object buffer;
        u8* contents{};
        u64 tick{};
        u64 frame{};
        double seconds{};
        bool pending{};
        std::vector<Output> outputs;
        u32 tiles{};
    };

    struct Frame {
        u64 frame{};
        double seconds{};
        std::vector<Output> outputs;
        std::vector<Tile> tiles;
        std::unordered_map<std::string, u32> keys;
    };

    bool Reduce(Kind kind, u32 index, std::span<const Source> sources,
                std::span<const u64> shaders, u32 draw_count);
    void Collect(Slot& slot);
    void WorkerLoop();
    void Analyze();

    const Device& device;
    Scheduler& scheduler;

    bool enabled{};
    bool passes{true};
    std::filesystem::path directory;
    float hot{};
    double ratio{};
    double delta{};
    double pass_ratio{};
    u32 max_reports{};
    std::chrono::steady_clock::time_point start;

    Object library;
    Object function;
    Object state;
    Object nearest;
    std::vector<ShaderTranslate::ResourceSlot> program_slots;

    std::vector<Slot> slots;
    u32 current{};
    u64 frame{};
    bool warned_full{};

    const void* pass_key{};
    std::vector<Source> pass_sources;
    std::vector<u64> pass_hashes;
    u32 pass_index{};
    u32 pass_first_draw{};
    u32 draws{};
    u32 dispatches{};
    u32 presents{};
    std::unordered_map<std::string, u32> key_counts;

    // Worker thread.
    std::deque<Frame> history;
    u64 checked{};
    u64 flashes{};
    /// Frames brighter than both neighbours in a presented tile (flicker, every other frame).
    u64 alternating{};
    u64 alternating_window{};
    u32 reports{};

    std::thread worker;
    std::mutex queue_mutex;
    std::condition_variable queue_ready;
    std::deque<Frame> queue;
    bool stop_worker{};
};

} // namespace Metal
