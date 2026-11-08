// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>

#include "common/assert.h"
#include "common/logging.h"
#include "core/core.h"
#include "video_core/engines/kepler_memory.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"

namespace Tegra::Engines {

KeplerMemory::KeplerMemory(MemoryManager& memory_manager_)
    : memory_manager{memory_manager_}, upload_state{memory_manager_, regs.upload}
{}

KeplerMemory::~KeplerMemory() = default;

void KeplerMemory::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    upload_state.BindRasterizer(rasterizer_);

    execution_mask.reset();
    execution_mask[KEPLERMEMORY_REG_INDEX(exec)] = true;
    execution_mask[KEPLERMEMORY_REG_INDEX(data)] = true;
}

void KeplerMemory::ConsumeSinkImpl(Core::System& system) {
    for (auto [method, value] : method_sink) {
        regs.reg_array[method] = value;
    }
    method_sink.clear();
}

void KeplerMemory::CallMethod(Core::System& system, u32 method, u32 method_argument, bool is_last_call) {
    ASSERT_MSG(method < Regs::NUM_REGS, "Invalid KeplerMemory register, increase the size of the Regs structure");

    regs.reg_array[method] = method_argument;

    switch (method) {
    case KEPLERMEMORY_REG_INDEX(exec): {
        upload_state.ProcessExec(regs.exec.linear != 0);
        break;
    }
    case KEPLERMEMORY_REG_INDEX(data): {
        upload_state.ProcessData(method_argument, is_last_call);
        break;
    }
    }
}

void KeplerMemory::CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) {
    switch (method) {
    case KEPLERMEMORY_REG_INDEX(data):
        upload_state.ProcessData(base_start, amount);
        return;
    default:
        for (u32 i = 0; i < amount; i++) {
            CallMethod(system, method, base_start[i], methods_pending - i <= 1);
        }
        break;
    }
}

void KeplerMemory::FrontMethod(VideoCommon::GpuSplit& split, u32 method, const u32* words,
                               u32 count, bool non_incrementing) {
    constexpr u32 base = KEPLERMEMORY_REG_INDEX(upload);
    if (method == KEPLERMEMORY_REG_INDEX(data)) {
        split.NoteMemoryWriter();
        if (!front_upload_known) {
            const auto device = memory_manager.GpuToCpuAddress(
                (GPUVAddr{front_regs[base + 2]} << 32) | front_regs[base + 3]);
            if (!device) {
                split.NoteMemoryWrite(0, ~u64{0}, "kepler upload data unmapped");
                return;
            }
            split.NoteMemoryWrite(*device,
                                  front_regs[base] * u64{(std::max)(front_regs[base + 1], 1U)} +
                                      u64{front_regs[base + 4]} * front_regs[base + 1],
                                  "kepler upload data");
            return;
        }
        const u32 remaining_bytes =
            front_upload_offset < front_upload_size ? front_upload_size - front_upload_offset : 0;
        const u32 whole = (std::min<u32>)(count, remaining_bytes / 4);
        split.NoteMemoryWriteData(front_upload_device + front_upload_offset, words, whole);
        if (whole < count && remaining_bytes % 4 != 0) {
            split.NoteMemoryWrite(front_upload_device + front_upload_offset + whole * 4,
                                  remaining_bytes % 4);
        }
        front_upload_offset += (std::min<u32>)(count * 4, remaining_bytes);
        return;
    }
    for (u32 i = 0; i < count; ++i) {
        const u32 m = non_incrementing ? method : method + i;
        if (m < front_regs.size()) {
            front_regs[m] = words[i];
        }
        if (m == KEPLERMEMORY_REG_INDEX(exec)) {
            split.NoteMemoryWriter();
            const u64 line_length = front_regs[base];
            const u64 lines = (std::max)(front_regs[base + 1], 1U);
            const u64 pitch = (std::max<u64>)(front_regs[base + 4], line_length);
            const GPUVAddr address = (GPUVAddr{front_regs[base + 2]} << 32) | front_regs[base + 3];
            const u64 size = pitch * (lines - 1) + line_length;
            const auto device = memory_manager.GpuToCpuAddress(address);
            const bool linear = (words[i] & 1) != 0;
            front_upload_known = device && linear && lines == 1 && (*device & 3) == 0 &&
                                 memory_manager.GpuToCpuAddress(address + size - 1) ==
                                     *device + size - 1;
            front_upload_device = device.value_or(0);
            front_upload_size = static_cast<u32>((std::min<u64>)(size, ~0U));
            front_upload_offset = 0;
            if (!front_upload_known) {
                if (!device || size == 0 || size > (u64{64} << 20)) {
                    split.NoteMemoryWrite(0, ~u64{0}, "kepler upload unmapped");
                } else {
                    split.NoteMemoryWrite(*device, size, "kepler upload");
                }
            }
        }
    }
}

} // namespace Tegra::Engines
