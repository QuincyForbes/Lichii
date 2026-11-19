// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <boost/container/small_vector.hpp>
#include <catch2/catch_test_macros.hpp>

#include "common/page_table.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/kernel.h"
#include "core/memory.h"
#include "tests/core/system_test_context.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker_base.h"
#include "video_core/control/channel_state.h"
#include "video_core/gpu.h"
#include "video_core/renderer_null/null_rasterizer.h"

namespace {

struct Runtime;

struct Buffer : VideoCommon::BufferBase {
    Buffer(Runtime&, VideoCommon::NullBufferParams params) : BufferBase{params} {}
    Buffer(Runtime&, DAddr address, u64 size, bool mapped = false)
        : BufferBase{address, size}, bytes(static_cast<size_t>(size)), host_mapped{mapped} {}

    void MarkUsage(u64 offset, u64 size) {
        REQUIRE(offset + size <= bytes.size());
    }
    std::span<u8> Mapped() {
        return host_mapped ? std::span{bytes} : std::span<u8>{};
    }
    std::span<const u8> Mapped() const {
        return host_mapped ? std::span<const u8>{bytes} : std::span<const u8>{};
    }
    bool IsSparseCompatible() const noexcept {
        return false;
    }
    u64 LastUsageTick() const noexcept {
        return tick;
    }

    std::vector<u8> bytes;
    bool host_mapped{};
    u64 tick{};
};

struct Staging {
    Buffer* buffer{};
    u64 offset{};
    std::span<u8> mapped_span;
};

struct Runtime {
    bool CanReportMemoryUsage() const {
        return false;
    }
    u64 GetDeviceMemoryUsage() const {
        return 0;
    }
    u64 GetBufferAllocationTail() const {
        return 0;
    }
    u64 GetDeviceLocalMemory() const {
        return 0;
    }
    u64 GetMaxStorageBufferRange() const {
        return 1ULL << 30;
    }
    u32 GetStorageBufferAlignment() const {
        return 256;
    }
    u32 GetUniformBufferAlignment() const {
        return 1;
    }
    bool IsFree(u64 tick) const {
        return tick <= completed_tick;
    }
    u64 CurrentTick() const {
        return submitted_tick + 1;
    }
    void Wait(u64) {}
    void TickFrame(Common::SlotVector<Buffer>&) {}
    void ClearBuffer(Buffer& buffer, u32 offset, size_t size, u32 value) {
        std::memset(buffer.bytes.data() + offset, static_cast<int>(value & 0xff), size);
    }
    bool CanReorderUpload(const Buffer&, std::span<const VideoCommon::BufferCopy>) const {
        return false;
    }
    Staging DownloadStagingBuffer(u64 size, bool = false) {
        stagings.push_back(std::make_unique<Buffer>(*this, 0, size + 128, false));
        Buffer* const staging = stagings.back().get();
        return {staging, 128, std::span{staging->bytes}.subspan(128)};
    }
    Staging UploadStagingBuffer(u64 size) {
        return DownloadStagingBuffer(size);
    }
    void FreeDeferredStagingBuffer(Staging&) {}
    void CopyBuffer(Buffer* dst, Buffer& src, std::span<const VideoCommon::BufferCopy> copies,
                    bool, bool = false) {
        for (const auto& copy : copies) {
            REQUIRE(copy.src_offset + copy.size <= src.bytes.size());
            REQUIRE(copy.dst_offset + copy.size <= dst->bytes.size());
            pending.push_back([dst, &src, copy] {
                std::memcpy(dst->bytes.data() + copy.dst_offset,
                            src.bytes.data() + copy.src_offset, copy.size);
            });
        }
    }
    void CopyBuffer(Buffer& dst, Buffer& src, std::span<const VideoCommon::BufferCopy> copies,
                    bool, bool = false) {
        CopyBuffer(dst, &src, copies, true, false);
    }
    void CopyBuffer(Buffer& dst, Buffer* src, std::span<const VideoCommon::BufferCopy> copies,
                    bool, bool = false) {
        for (const auto& copy : copies) {
            REQUIRE(copy.src_offset + copy.size <= src->bytes.size());
            REQUIRE(copy.dst_offset + copy.size <= dst.bytes.size());
            std::memcpy(dst.bytes.data() + copy.dst_offset, src->bytes.data() + copy.src_offset,
                        copy.size);
        }
    }
    void Finish() {
        for (auto& copy : pending) {
            copy();
        }
        pending.clear();
        completed_tick = ++submitted_tick;
    }
    void PreCopyBarrier() {}
    void PostCopyBarrier() {}

    u64 submitted_tick{};
    u64 completed_tick{};
    std::vector<std::unique_ptr<Buffer>> stagings;
    std::vector<std::function<void()>> pending;
};

struct Tracker : VideoCommon::MemoryTrackerBase<Tegra::MaxwellDeviceMemoryManager> {
    using MemoryTrackerBase::MemoryTrackerBase;
};

struct Policy {
    using Runtime = ::Runtime;
    using Buffer = ::Buffer;
    using Async_Buffer = Staging;
    using MemoryTracker = Tracker;
    static constexpr bool IS_OPENGL = false;
    static constexpr bool HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS = false;
    static constexpr bool HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT = false;
    static constexpr bool NEEDS_BIND_UNIFORM_INDEX = false;
    static constexpr bool NEEDS_BIND_STORAGE_INDEX = false;
    static constexpr bool USE_MEMORY_MAPS = true;
    static constexpr bool SEPARATE_IMAGE_BUFFER_BINDINGS = false;
    static constexpr bool USE_MEMORY_MAPS_FOR_UPLOADS = true;
};

using Cache = VideoCommon::BufferCache<Policy>;

struct DmaFixture {
    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device{physical};
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    std::optional<Tegra::MemoryManager> memory;
    std::unique_ptr<Tegra::Control::ChannelState> channel;
    Runtime runtime;
    std::unique_ptr<Cache> cache;
    std::unique_ptr<Kernel::KProcess> process;
    Core::Asid asid{};
    u8* bytes{};
    Buffer* initial_buffer{};

    DmaFixture() {
        system.Initialize();
        process = std::make_unique<Kernel::KProcess>(system.Kernel());
        auto& table = process->GetPageTable().GetImpl();
        table.Resize(24, 12);
        bytes = physical.GetPointerFromRaw<u8>(0x10000);
        std::memset(bytes, 0, 0x40000);
        for (u64 page = 0; page < 64; ++page) {
            table.entries.GetAndFault(0x10 + page)
                .Store(false, Common::PageType::Memory, 1,
                       reinterpret_cast<uintptr_t>(bytes) - 0x10000);
        }
        process->GetMemory().SetCurrentPageTable(*process);
        asid = device.RegisterProcess(&process->GetMemory());
        device.BindInterface(&rasterizer);
        device.Map(0x10000, 0x10000, 0x40000, asid);

        memory.emplace(system, device, 32, 1ULL << 31, 16);
        memory->BindRasterizer(&rasterizer);
        memory->Map(0x100000, 0x10000, 0x10000, Tegra::PTEKind::PITCH, true);

        cache = std::make_unique<Cache>(device, runtime);
        channel = std::make_unique<Tegra::Control::ChannelState>(1);
        channel->memory_manager = &*memory;
        channel->Init(system, 0);
        channel->BindRasterizer(&rasterizer);
        cache->CreateChannel(*channel);
        cache->BindToChannel(channel->bind_id);

        auto [buf, offset] = cache->ObtainBuffer(
            0x100000, 0x10000, VideoCommon::ObtainBufferSynchronize::FullSynchronize,
            VideoCommon::ObtainBufferOperation::DoNothing);
        REQUIRE(offset == 0);
        initial_buffer = buf;
    }

    ~DmaFixture() {
        cache.reset();
        channel.reset();
        memory.reset();
        device.Unmap(0x10000, 0x40000);
        device.UnregisterProcess(asid);
        system.Kernel().UnregisterKernelObject(process.get());
    }
};

template <class C>
bool InvokeDmaCopy(C& cache, GPUVAddr src, GPUVAddr dst, u64 amount,
                   boost::container::small_vector<std::pair<DAddr, size_t>, 4>& gpu_written) {
    gpu_written.clear();
    if constexpr (requires { cache.DMACopy(src, dst, amount, false, &gpu_written); }) {
        return cache.DMACopy(src, dst, amount, false, &gpu_written);
    } else if constexpr (requires { cache.DMACopy(src, dst, amount, &gpu_written); }) {
        return cache.DMACopy(src, dst, amount, &gpu_written);
    } else {
        return cache.DMACopy(src, dst, amount, false);
    }
}

template <class C>
constexpr bool HasDmaCopyGpuWrittenOut() {
    boost::container::small_vector<std::pair<DAddr, size_t>, 4>* ptr = nullptr;
    return requires(C& c) { c.DMACopy(GPUVAddr{}, GPUVAddr{}, u64{}, false, ptr); } ||
           requires(C& c) { c.DMACopy(GPUVAddr{}, GPUVAddr{}, u64{}, ptr); };
}

} // namespace

TEST_CASE("A DMA clear finds buffers past the first quarter of its range", "[buffer_ports]") {
    DmaFixture f;
    f.memory->Map(0x200000, 0x20000, 0x20000, Tegra::PTEKind::PITCH, true);
    (void)f.cache->ObtainBuffer(0x210000, 0x100,
                                VideoCommon::ObtainBufferSynchronize::FullSynchronize,
                                VideoCommon::ObtainBufferOperation::DoNothing);

    // DMAClear counts words: 0x2000 of them (0x8000 bytes) end where the 64 KiB-aligned buffer at
    // 0x210000 starts, 0x2001 reach into it (past the first quarter of the clear range).
    REQUIRE_FALSE(f.cache->DMAClear(0x208000, 0x2000, 0x09090909));
    REQUIRE(f.cache->DMAClear(0x208000, 0x2001, 0x09090909));
}

TEST_CASE("A DMA copy names the destination ranges whose GPU-written bytes guest memory lacks",
          "[buffer_ports]") {
    DmaFixture f;
    // The GPU wrote the first half of the source; guest memory holds older bytes there.
    std::memset(f.bytes + 0x4000, 1, 0x40);
    std::fill_n(f.initial_buffer->bytes.begin() + 0x4000, 0x20, u8{9});
    (void)f.cache->ObtainBuffer(0x104000, 0x20,
                                VideoCommon::ObtainBufferSynchronize::NoSynchronize,
                                VideoCommon::ObtainBufferOperation::MarkAsWritten);

    boost::container::small_vector<std::pair<DAddr, size_t>, 4> gpu_written;
    REQUIRE(InvokeDmaCopy(*f.cache, 0x104000, 0x106000, 0x40, gpu_written));
    if constexpr (HasDmaCopyGpuWrittenOut<Cache>()) {
        REQUIRE(gpu_written.size() == 1);
        REQUIRE(gpu_written[0] == std::pair<DAddr, size_t>{0x16000, 0x20});
    }
    // An image over the first half uploading now would take the older bytes.
    REQUIRE(std::all_of(f.bytes + 0x6000, f.bytes + 0x6040, [](u8 b) { return b == 1; }));
    f.cache->DownloadMemory(0x16000, 0x40);
    REQUIRE(std::all_of(f.bytes + 0x6000, f.bytes + 0x6020, [](u8 b) { return b == 9; }));
    REQUIRE(std::all_of(f.bytes + 0x6020, f.bytes + 0x6040, [](u8 b) { return b == 1; }));

    // A source only the CPU wrote: the destination's guest memory has its bytes at once.
    std::memset(f.bytes + 0x5000, 7, 0x20);
    f.cache->WriteMemory(0x15000, 0x20);
    REQUIRE(InvokeDmaCopy(*f.cache, 0x105000, 0x107000, 0x20, gpu_written));
    if constexpr (HasDmaCopyGpuWrittenOut<Cache>()) {
        REQUIRE(gpu_written.empty());
    }
    REQUIRE(std::all_of(f.bytes + 0x7000, f.bytes + 0x7020, [](u8 b) { return b == 7; }));
}
