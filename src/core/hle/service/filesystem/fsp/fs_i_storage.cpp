// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <thread>

#include "common/settings.h"
#include "core/file_sys/errors.h"
#include "core/hle/service/cmif_serialization.h"
#include "core/hle/service/filesystem/fsp/fs_i_storage.h"
#include "core/hle/service/filesystem/fsp/fs_read_trace.h"

namespace Service::FileSystem {

IStorage::IStorage(Core::System& system_, FileSys::VirtualFile backend_)
    : ServiceFramework{system_, "IStorage"}, backend(std::move(backend_)) {
    static const FunctionInfo functions[] = {
        {0, D<&IStorage::Read>, "Read"},
        {1, nullptr, "Write"},
        {2, nullptr, "Flush"},
        {3, nullptr, "SetSize"},
        {4, D<&IStorage::GetSize>, "GetSize"},
        {5, nullptr, "OperateRange"},
    };
    RegisterHandlers(functions);
}

Result IStorage::Read(
    OutBuffer<BufferAttr_HipcMapAlias | BufferAttr_HipcMapTransferAllowsNonSecure> out_bytes,
    s64 offset, s64 length) {
    LOG_DEBUG(Service_FS, "called, offset={:#x}, length={}", offset, length);

    R_UNLESS(length >= 0, FileSys::ResultInvalidSize);
    R_UNLESS(offset >= 0, FileSys::ResultInvalidOffset);
    R_UNLESS(static_cast<u64>(length) <= out_bytes.size(), FileSys::ResultInvalidSize);
    if (length == 0) {
        R_SUCCEED();
    }

    // The 40 MB/s throttle emulates a Switch 1 card (Eden cbd4cf2d8c, Mario Strikers' intro).
    // Switch 2 storage is many times faster, and a Switch 2 title that streams a movie with
    // hundreds of reads a second (Ocarina of Time's opening: 300-460) sleeps most of each second
    // on the thread that also hands decoder images back, so its video falls seconds behind.
    if (!Settings::CurrentProgramIsSwitch2()) {
        const s64 now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now().time_since_epoch())
                               .count();
        const s64 target_ns = next_read_tick_ns.load(std::memory_order_relaxed);
        if (now_ns < target_ns) {
            std::this_thread::sleep_for(std::chrono::nanoseconds{target_ns - now_ns});
        }
        const s64 after_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now().time_since_epoch())
                                 .count();
        const s64 cost_ns = std::min(length, MaxChargedReadSize) * 1'000'000'000 / ReadBytesPerSecond;
        next_read_tick_ns.store(after_ns + cost_ns, std::memory_order_relaxed);
    }

    // Read the data from the Storage backend directly into the output span.
    const ReadTraceScope trace{ReadTraceSource::Storage, length};
    backend->Read(out_bytes.data(), static_cast<size_t>(length), static_cast<size_t>(offset));

    R_SUCCEED();
}

Result IStorage::GetSize(Out<u64> out_size) {
    *out_size = backend->GetSize();

    LOG_DEBUG(Service_FS, "called, size={}", *out_size);

    R_SUCCEED();
}

} // namespace Service::FileSystem
