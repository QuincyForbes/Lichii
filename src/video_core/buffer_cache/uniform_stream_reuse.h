// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <type_traits>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

#include "common/common_types.h"
#include "video_core/nx2_profile.h"

namespace VideoCommon {

/// What a buffer cache runtime offers for stream uniform reuse.
///  - StreamUniformRef: a small trivially copyable description of one stream allocation (the
///    host buffer, its offset and whatever else the runtime needs to add it to a descriptor
///    payload again, e.g. the device address of the descriptor buffer path).
///  - LastStreamUniformRef(): the allocation made by the latest BindMappedUniformBuffer call.
///  - RebindStreamUniform(ref, size): adds `ref` to the descriptor payload exactly as
///    BindMappedUniformBuffer added the allocation the first time.
template <class R>
concept StreamUniformReusable = requires(R& runtime, u32 size) {
    typename R::StreamUniformRef;
    requires std::is_trivially_copyable_v<typename R::StreamUniformRef>;
    { runtime.LastStreamUniformRef() } -> std::same_as<typename R::StreamUniformRef>;
    runtime.RebindStreamUniform(runtime.LastStreamUniformRef(), size);
    { runtime.CurrentTick() } -> std::convertible_to<u64>;
};

/// EDEN_NX2_UBOREUSE=0 turns the reuse off (every small uniform buffer is copied, as before).
[[nodiscard]] inline bool UniformStreamReuseEnabled() noexcept {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_UBOREUSE");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

/// EDEN_NX2_UBO_INPLACE=1 (default off): BindSmallUniformReusing compares a buffer whose device
/// pages are one host run (always so within one 4 KiB page) with its slot's bytes straight in
/// guest memory, instead of copying it into a scratch buffer first. Unchanged bytes are then not
/// copied at all; changed ones are copied once into the slot and the stream copy is made from the
/// slot, so the two never differ (as the scratch snapshot guaranteed before).
[[nodiscard]] inline bool UniformInPlaceCompareEnabled() noexcept {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_UBO_INPLACE");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

template <class Ref>
class UniformStreamReuse {
public:
    static constexpr u32 MaxBytes = 4096; // BufferCache's DEFAULT_SKIP_CACHE_SIZE
    static constexpr size_t NumStages = 6; // the five graphics stages and compute
    static constexpr size_t NumSlots = 18; // Maxwell constant buffers per stage
    /// One private copy per slot, then the copies keyed by guest range (range reuse): NumSets
    /// sets of NumWays.
    static constexpr size_t NumPrivate = NumStages * NumSlots;
    static constexpr size_t NumSets = 256;
    static constexpr size_t NumWays = 2;
    static constexpr size_t NumCopies = NumPrivate + NumSets * NumWays;

    /// Bytes copied to a stream allocation, and that allocation: `ref` was filled from exactly
    /// these bytes on scheduler tick `tick`. The bytes change only together with the serial.
    struct Copy {
        Ref ref{};
        u64 tick{};
        u64 address{}; ///< the guest range copied (a lookup hint: hits compare the bytes)
        u64 verified_serial{};
        u64 page_gen{};
        u32 size{};    ///< 0: nothing copied
        u32 serial{};
    };

    /// The copy a (stage, constant buffer) slot bound last.
    struct Slot {
        u32 copy{};
        u32 serial{}; ///< the copy's serial then; 0: none
    };

    UniformStreamReuse() : bytes{new u8[NumCopies * MaxBytes]} {}

    void Reset() noexcept {
        for (Copy& copy : copies) {
            copy.size = 0;
            copy.tick = 0;
            copy.address = 0;
            copy.verified_serial = 0;
            copy.page_gen = 0;
            NextSerial(copy);
        }
        for (auto& stage : slots) {
            stage.fill(Slot{});
        }
        mru.fill(0);
    }

    [[nodiscard]] u8* Bytes(size_t copy) noexcept {
        return bytes.get() + copy * MaxBytes;
    }

    static void NextSerial(Copy& copy) noexcept {
        if (++copy.serial == 0) {
            copy.serial = 1;
        }
    }

    [[nodiscard]] static size_t Set(u64 address, u32 size) noexcept {
        return static_cast<size_t>(((address ^ (u64{size} << 40)) * 0x9E3779B97F4A7C15ULL) >> 56) %
               NumSets;
    }

    std::array<Copy, NumCopies> copies{};
    std::array<std::array<Slot, NumSlots>, NumStages> slots{};
    std::array<u8, NumSets> mru{}; ///< per set: the way used last
    std::unique_ptr<u8[]> bytes;   ///< MaxBytes per copy
    std::array<u8, MaxBytes> scratch{};
};

template <class Ref>
[[nodiscard]] inline UniformStreamReuse<Ref>& UniformStreamReuseState() noexcept {
    static UniformStreamReuse<Ref> state;
    return state;
}

/// Forgets every slot. Call it when a buffer cache (and so its runtime, scheduler and stream
/// buffer) is created: ticks restart with a new scheduler and the old references are dead.
template <class Runtime>
inline void ResetUniformStreamReuse() noexcept {
    if constexpr (StreamUniformReusable<Runtime>) {
        UniformStreamReuseState<typename Runtime::StreamUniformRef>().Reset();
    }
}

/// Binds the small uniform buffer at `device_addr` for (stage, cbuf `index`) through the stream
/// buffer, binding an earlier allocation again when it holds the same bytes and was made in the
/// current command buffer: the slot's previous copy or, with `range_reuse` (EDEN_NX2_EXP17,
/// from zortos293/Lichii#641), a copy of the same guest range made by any stage or slot. Every
/// reuse compares the guest bytes with the copy's. Returns false (nothing bound) when the buffer
/// does not qualify, so the caller binds it the classic way.
[[nodiscard]] inline bool UniformBytesEqual(const u8* copy_bytes, const u8* guest,
                                            u32 size) noexcept {
    if (size >= sizeof(u64)) {
        u64 a0, b0, a1, b1;
        std::memcpy(&a0, copy_bytes, sizeof(u64));
        std::memcpy(&b0, guest, sizeof(u64));
        if (a0 != b0) {
            return false;
        }
        std::memcpy(&a1, copy_bytes + size - sizeof(u64), sizeof(u64));
        std::memcpy(&b1, guest + size - sizeof(u64), sizeof(u64));
        if (a1 != b1) {
            return false;
        }
    }
#ifdef __AVX2__
    size_t off = 0;
    for (; off + 128 <= size; off += 128) {
        const __m256i d0 = _mm256_xor_si256(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(copy_bytes + off)),
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(guest + off)));
        const __m256i d1 = _mm256_xor_si256(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(copy_bytes + off + 32)),
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(guest + off + 32)));
        const __m256i d2 = _mm256_xor_si256(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(copy_bytes + off + 64)),
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(guest + off + 64)));
        const __m256i d3 = _mm256_xor_si256(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(copy_bytes + off + 96)),
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(guest + off + 96)));
        const __m256i acc = _mm256_or_si256(_mm256_or_si256(d0, d1), _mm256_or_si256(d2, d3));
        if (!_mm256_testz_si256(acc, acc)) {
            return false;
        }
    }
    for (; off + 32 <= size; off += 32) {
        const __m256i d = _mm256_xor_si256(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(copy_bytes + off)),
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(guest + off)));
        if (!_mm256_testz_si256(d, d)) {
            return false;
        }
    }
    return size == off || std::memcmp(copy_bytes + off, guest + off, size - off) == 0;
#else
    return std::memcmp(copy_bytes, guest, size) == 0;
#endif
}

struct NoUboRangeClean {
    [[nodiscard]] constexpr bool operator()(u64&) const noexcept {
        return false;
    }
};

template <class Runtime, class DeviceMemory, class RangeCleanPred = NoUboRangeClean>
[[nodiscard]] bool BindSmallUniformReusing(Runtime& runtime, DeviceMemory& device_memory,
                                           size_t stage, u32 index, u32 binding_index,
                                           u64 device_addr, u32 size, bool range_reuse = false,
                                           u64 write_serial = 0, u64 page_gen = 0,
                                           RangeCleanPred&& range_clean = {}) {
    using Reuse = UniformStreamReuse<typename Runtime::StreamUniformRef>;
    if (size == 0 || size > Reuse::MaxBytes || stage >= Reuse::NumStages ||
        index >= Reuse::NumSlots) {
        return false;
    }
    Reuse& reuse = UniformStreamReuseState<typename Runtime::StreamUniformRef>();
    typename Reuse::Slot& slot = reuse.slots[stage][index];
    const u64 tick = runtime.CurrentTick();
    const auto rebind = [&](size_t copy_index) {
        const typename Reuse::Copy& copy = reuse.copies[copy_index];
        runtime.RebindStreamUniform(copy.ref, size);
        slot = typename Reuse::Slot{static_cast<u32>(copy_index), copy.serial};
        return true;
    };
    const auto is_verified_clean = [&](typename Reuse::Copy& copy) {
        if (write_serial == 0 || copy.verified_serial == 0) {
            return false;
        }
        if (copy.verified_serial == write_serial) [[likely]] {
            return true;
        }
        if (range_clean(copy.page_gen)) [[likely]] {
            copy.verified_serial = write_serial;
            return true;
        }
        return false;
    };
    constexpr size_t NO_COPY = ~size_t{0};
    size_t slot_copy = NO_COPY;
    if (slot.serial != 0 && reuse.copies[slot.copy].serial == slot.serial) {
        slot_copy = slot.copy;
        typename Reuse::Copy& copy = reuse.copies[slot_copy];
        if (copy.tick == tick && copy.address == device_addr && copy.size == size &&
            is_verified_clean(copy)) [[likely]] {
            if constexpr (requires { device_memory.GetSpan(device_addr, size); }) {
                if (UniformInPlaceCompareEnabled()) {
                    Nx2Profile::Add(Nx2Profile::Counter::UboInPlaceSame, 1);
                }
            }
            return rebind(slot_copy);
        }
    }
    if (range_reuse && write_serial != 0) {
        const size_t set = Reuse::Set(device_addr, size);
        const size_t base = Reuse::NumPrivate + set * Reuse::NumWays;
        for (size_t way = 0; way < Reuse::NumWays; ++way) {
            const size_t copy_index = base + way;
            typename Reuse::Copy& copy = reuse.copies[copy_index];
            if (copy_index != slot_copy && copy.address == device_addr && copy.size == size &&
                copy.tick == tick && is_verified_clean(copy)) {
                reuse.mru[set] = static_cast<u8>(way);
                if constexpr (requires { device_memory.GetSpan(device_addr, size); }) {
                    if (UniformInPlaceCompareEnabled()) {
                        Nx2Profile::Add(Nx2Profile::Counter::UboInPlaceSame, 1);
                    }
                }
                return rebind(copy_index);
            }
        }
    }
    // The bytes are compared in guest memory with EDEN_NX2_UBO_INPLACE, else in a scratch copy.
    // The stream allocation is filled from the copy's bytes, never from guest memory, so a later
    // reuse finds exactly the bytes that were uploaded. Test memories have no GetSpan.
    const u8* guest = nullptr;
    if constexpr (requires { device_memory.GetSpan(device_addr, size); }) {
        if (UniformInPlaceCompareEnabled()) {
            guest = device_memory.GetSpan(device_addr, size);
        }
    }
    const bool in_place = guest != nullptr;
    if (!in_place) {
        device_memory.ReadBlockUnsafe(device_addr, reuse.scratch.data(), size);
        guest = reuse.scratch.data();
    }
    const auto same_bytes = [&](size_t copy_index) {
        typename Reuse::Copy& copy = reuse.copies[copy_index];
        if (copy.size != size || !UniformBytesEqual(reuse.Bytes(copy_index), guest, size)) {
            return false;
        }
        if (copy.address == device_addr) {
            copy.verified_serial = write_serial;
            copy.page_gen = page_gen;
        }
        return true;
    };
    bool slot_same = false;
    if (slot_copy != NO_COPY &&
        (reuse.copies[slot_copy].tick == tick || range_reuse) &&
        (write_serial == 0 || reuse.copies[slot_copy].address == device_addr)) {
        slot_same = same_bytes(slot_copy);
    }
    if (in_place) {
        namespace Profile = Nx2Profile;
        Profile::Add(slot_same ? Profile::Counter::UboInPlaceSame
                               : Profile::Counter::UboInPlaceChanged,
                     1);
    }
    if (slot_same && reuse.copies[slot_copy].tick == tick) {
        return rebind(slot_copy);
    }
    size_t target = stage * Reuse::NumSlots + index;
    if (range_reuse) {
        const size_t set = Reuse::Set(device_addr, size);
        const size_t base = Reuse::NumPrivate + set * Reuse::NumWays;
        for (size_t way = 0; way < Reuse::NumWays; ++way) {
            const size_t copy_index = base + way;
            const typename Reuse::Copy& copy = reuse.copies[copy_index];
            if (copy_index != slot_copy && copy.address == device_addr && copy.size == size &&
                copy.tick == tick && same_bytes(copy_index)) {
                reuse.mru[set] = static_cast<u8>(way);
                return rebind(copy_index);
            }
        }
        // Replace the way holding an older copy of this range, else one from an older command
        // buffer, else the one used least recently.
        size_t victim = NO_COPY;
        for (size_t way = 0; way < Reuse::NumWays && victim == NO_COPY; ++way) {
            const typename Reuse::Copy& copy = reuse.copies[base + way];
            if (copy.address == device_addr && copy.size == size) {
                victim = way;
            }
        }
        for (size_t way = 0; way < Reuse::NumWays && victim == NO_COPY; ++way) {
            if (reuse.copies[base + way].tick != tick) {
                victim = way;
            }
        }
        if (victim == NO_COPY) {
            victim = (reuse.mru[set] + 1) % Reuse::NumWays;
        }
        reuse.mru[set] = static_cast<u8>(victim);
        target = base + victim;
    }
    if (slot_same) {
        // The slot's copy from an older command buffer holds these bytes: fill the new
        // allocation from it.
        target = slot_copy;
    } else {
        typename Reuse::Copy& fresh = reuse.copies[target];
        std::memcpy(reuse.Bytes(target), guest, size);
        fresh.size = size;
        fresh.address = device_addr;
        fresh.verified_serial = write_serial;
        fresh.page_gen = page_gen;
        Reuse::NextSerial(fresh);
    }
    typename Reuse::Copy& copy = reuse.copies[target];
    const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
    std::memcpy(span.data(), reuse.Bytes(target), size);
    copy.ref = runtime.LastStreamUniformRef();
    // After the allocation: the tick the stream region is tagged with is the one current now.
    copy.tick = runtime.CurrentTick();
    slot = typename Reuse::Slot{static_cast<u32>(target), copy.serial};
    return true;
}

} // namespace VideoCommon
