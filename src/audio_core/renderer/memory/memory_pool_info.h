// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>

#include "audio_core/common/common.h"
#include "common/common_types.h"

namespace AudioCore::Renderer {
/**
 * CPU pools are mapped in user memory with the supplied process_handle (see PoolMapper).
 */
class MemoryPoolInfo {
public:
    /**
     * The location of this pool.
     * CPU pools are mapped in user memory with the supplied process_handle (see PoolMapper).
     * DSP pools are mapped in the current process sysmodule.
     */
    enum class Location {
        CPU = 1,
        DSP = 2,
    };

    /**
     * Current state of the pool
     */
    enum class State {
        Invalid,
        Acquired,
        RequestDetach,
        Detached,
        RequestAttach,
        Attached,
        Released,
    };

    /**
     * Result code for updating the pool (See InfoUpdater::Update)
     */
    enum class ResultState {
        Success,
        BadParam,
        MapFailed,
        InUse,
    };

    /**
     * Input parameters coming from the game which are used to update current pools
     * (See InfoUpdater::Update)
     */
    struct InParameter {
        /* 0x00 */ u64 address;
        /* 0x08 */ u64 size;
        /* 0x10 */ State state;
        /* 0x14 */ bool in_use;
        /* 0x18 */ char unk18[0x8];
    };
    static_assert(sizeof(InParameter) == 0x20, "MemoryPoolInfo::InParameter has the wrong size!");

    /**
     * Output status sent back to the game on update (See InfoUpdater::Update)
     */
    struct OutStatus {
        /* 0x00 */ State state;
        /* 0x04 */ char unk04[0xC];
    };
    static_assert(sizeof(OutStatus) == 0x10, "MemoryPoolInfo::OutStatus has the wrong size!");

    MemoryPoolInfo() = default;
    MemoryPoolInfo(Location location_) : location{location_} {}

    CpuAddr GetCpuAddress() const noexcept {
        return cpu_address;
    }

    CpuAddr GetDspAddress() const noexcept {
        return dsp_address;
    }

    u64 GetSize() const noexcept {
        return size;
    }

    Location GetLocation() const noexcept {
        return location;
    }

    void SetCpuAddress(const CpuAddr address, const u64 size_) noexcept {
        cpu_address = address;
        size = size_;
    }

    void SetDspAddress(const CpuAddr address) noexcept {
        dsp_address = address;
    }

    bool Contains(const CpuAddr address_, const u64 size_) const noexcept {
        return cpu_address <= address_ && (address_ + size_) <= (cpu_address + size);
    }

    bool IsMapped() const noexcept {
        return dsp_address != 0;
    }

    CpuAddr Translate(const CpuAddr address, const u64 size_) const noexcept {
        if (!IsMapped() || !Contains(address, size_)) {
            return 0;
        }
        return dsp_address + (address - cpu_address);
    }

    void SetUsed(const bool used) noexcept {
        in_use = used;
    }

    bool IsUsed() const noexcept {
        return in_use;
    }

private:
    /// Base address for the CPU-side memory
    CpuAddr cpu_address{};
    /// Base address for the DSP-side memory
    CpuAddr dsp_address{};
    /// Size of this pool
    u64 size{};
    /// Location of this pool, either CPU or DSP
    Location location{Location::DSP};
    /// If this pool is in use
    bool in_use{};
};

} // namespace AudioCore::Renderer
