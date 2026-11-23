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
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#include "common/common_types.h"
#include "video_core/nx2_draw_stats.h"
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

[[nodiscard]] inline bool UniformSwitchOn(const char* name) noexcept {
    const char* const value = std::getenv(name);
    return value == nullptr || value[0] != '0';
}

/// Plain flags, not function statics: they are checked for every uniform buffer of every draw,
/// and MSVC reads a thread-local to check a static's initialization (zortos293/Lichii#1059).
inline bool g_uniform_stream_reuse = UniformSwitchOn("EDEN_NX2_UBOREUSE");
inline bool g_uniform_in_place_compare = UniformSwitchOn("EDEN_NX2_UBO_INPLACE");

/// Called by VideoCore::CreateGPU: static initialization runs before the frontend loads
/// nx2_switches.txt, so a switch set only there takes effect from that point on.
inline void ReadUniformStreamReuseSwitches() noexcept {
    g_uniform_stream_reuse = UniformSwitchOn("EDEN_NX2_UBOREUSE");
    g_uniform_in_place_compare = UniformSwitchOn("EDEN_NX2_UBO_INPLACE");
}

/// EDEN_NX2_UBOREUSE=0 turns the reuse off (every small uniform buffer is copied, as before).
[[nodiscard]] inline bool UniformStreamReuseEnabled() noexcept {
    return g_uniform_stream_reuse;
}

/// EDEN_NX2_UBO_INPLACE=0 (default on) turns this off: BindSmallUniformReusing compares a buffer
/// whose device pages are one host run (always so within one 4 KiB page) with its slot's bytes
/// straight in guest memory, instead of copying it into a scratch buffer first. Unchanged bytes
/// are then not copied at all; changed ones are copied once into the slot and the stream copy is
/// made from the slot, so the two never differ (as the scratch snapshot guaranteed before).
[[nodiscard]] inline bool UniformInPlaceCompareEnabled() noexcept {
    return g_uniform_in_place_compare;
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
    static constexpr size_t NumSpanCacheEntries = 32;

    struct SpanCacheEntry {
        const void* memory{};
        u64 translation_gen{};
        u64 base_addr{};
        u64 end_addr{};
        const u8* host_ptr{};
    };

    /// Bytes copied to a stream allocation, and that allocation: `ref` was filled from exactly
    /// these bytes on scheduler tick `tick`. The bytes change only together with the serial.
    struct Copy {
        Ref ref{};
        u64 tick{};
        u64 address{}; ///< the guest range copied (a lookup hint: hits compare the bytes)
        u64 verified_serial{};
        u64 page_gen{};
        u64 cpu_epoch{};
        u32 size{};    ///< 0: nothing copied
        u32 serial{};
        /// Bytes allocated (the size padded to what the binding shader could read): a shader
        /// that may read further needs a new allocation, the padding past this is not zero.
        u32 allocated{};
        u16 diff_offset{};
    };

    /// The copy a (stage, constant buffer) slot bound last.
    struct Slot {
        u32 copy{};
        u32 serial{}; ///< the copy's serial then; 0: none
    };

    /// What the compute stage bound last for a constant buffer (compute binds are not tracked by
    /// the copy table): `ref` was filled from `bytes` on scheduler tick `tick`.
    struct ComputeSlot {
        Ref ref{};
        u64 tick{};
        u64 address{};
        u64 verified_serial{};
        u64 page_gen{};
        u64 cpu_epoch{};
        u32 size{}; ///< 0: nothing copied
        u32 allocated{};
        u16 diff_offset{};
        std::array<u8, MaxBytes> bytes{};
    };

    constexpr UniformStreamReuse() noexcept = default;

    void EnsureAllocated() {
        if (!bytes) {
            bytes = std::make_unique_for_overwrite<u8[]>(NumCopies * MaxBytes);
        }
    }

    void Reset() {
        EnsureAllocated();
        for (Copy& copy : copies) {
            copy.size = 0;
            copy.allocated = 0;
            copy.diff_offset = 0;
            copy.tick = 0;
            copy.address = 0;
            copy.verified_serial = 0;
            copy.page_gen = 0;
            copy.cpu_epoch = 0;
            NextSerial(copy);
        }
        for (auto& stage : slots) {
            stage.fill(Slot{});
        }
        mru.fill(0);
        for (ComputeSlot& slot : compute_slots) {
            slot.size = 0;
            slot.allocated = 0;
            slot.diff_offset = 0;
            slot.tick = 0;
            slot.address = 0;
            slot.verified_serial = 0;
            slot.page_gen = 0;
            slot.cpu_epoch = 0;
        }
        last_span = {};
        block_span_cache.fill({});
        page_span_cache.fill({});
    }

    [[nodiscard]] u8* Bytes(size_t copy) noexcept {
        return bytes.get() + copy * MaxBytes;
    }

    template <class DeviceMemory>
    [[nodiscard]] const u8* ResolveGuestSpan(DeviceMemory& device_memory, u64 device_addr,
                                             u32 size) noexcept {
        if constexpr (requires {
                          device_memory.GetSpan(device_addr, size);
                          device_memory.TranslationGeneration();
                      }) {
            const void* const mem_id = static_cast<const void*>(&device_memory);
            const u64 gen = device_memory.TranslationGeneration();
            const u64 end_addr = device_addr + size;
            if (last_span.memory == mem_id && last_span.translation_gen == gen &&
                device_addr >= last_span.base_addr && end_addr <= last_span.end_addr) {
                return last_span.host_ptr + (device_addr - last_span.base_addr);
            }
            constexpr u64 BlockSize = 0x10000;
            constexpr u64 BlockMask = BlockSize - 1;
            const u64 block_base = device_addr & ~BlockMask;
            const bool in_block = end_addr <= block_base + BlockSize;
            SpanCacheEntry* block_entry = nullptr;
            if (in_block) {
                block_entry = &block_span_cache[static_cast<size_t>(
                    ((device_addr >> 16) ^ (device_addr >> 21)) & (NumSpanCacheEntries - 1))];
                if (block_entry->memory == mem_id && block_entry->translation_gen == gen &&
                    block_entry->base_addr == block_base && end_addr <= block_entry->end_addr) {
                    last_span = *block_entry;
                    return block_entry->host_ptr + (device_addr - block_base);
                }
            }
            constexpr u64 PageSize = 0x1000;
            constexpr u64 PageMask = PageSize - 1;
            const u64 page_base = device_addr & ~PageMask;
            const bool in_page = end_addr <= page_base + PageSize;
            SpanCacheEntry* page_entry = nullptr;
            if (in_page) {
                page_entry = &page_span_cache[static_cast<size_t>(
                    ((device_addr >> 12) ^ (device_addr >> 17)) & (NumSpanCacheEntries - 1))];
                if (page_entry->memory == mem_id && page_entry->translation_gen == gen &&
                    page_entry->base_addr == page_base && end_addr <= page_entry->end_addr) {
                    last_span = *page_entry;
                    return page_entry->host_ptr + (device_addr - page_base);
                }
            }
            if (in_block) {
                if (const u8* const block_ptr = device_memory.GetSpan(block_base, BlockSize);
                    block_ptr != nullptr) {
                    *block_entry = SpanCacheEntry{
                        .memory = mem_id,
                        .translation_gen = gen,
                        .base_addr = block_base,
                        .end_addr = block_base + BlockSize,
                        .host_ptr = block_ptr,
                    };
                    last_span = *block_entry;
                    return block_ptr + (device_addr - block_base);
                }
            }
            if (in_page) {
                if (const u8* const page_ptr = device_memory.GetSpan(page_base, PageSize);
                    page_ptr != nullptr) {
                    *page_entry = SpanCacheEntry{
                        .memory = mem_id,
                        .translation_gen = gen,
                        .base_addr = page_base,
                        .end_addr = page_base + PageSize,
                        .host_ptr = page_ptr,
                    };
                    last_span = *page_entry;
                    return page_ptr + (device_addr - page_base);
                }
            }
            return device_memory.GetSpan(device_addr, size);
        } else if constexpr (requires { device_memory.GetSpan(device_addr, size); }) {
            return device_memory.GetSpan(device_addr, size);
        } else {
            return nullptr;
        }
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
    SpanCacheEntry last_span{};
    std::array<SpanCacheEntry, NumSpanCacheEntries> block_span_cache{};
    std::array<SpanCacheEntry, NumSpanCacheEntries> page_span_cache{};
    std::array<ComputeSlot, NumSlots> compute_slots{};
    std::array<u8, MaxBytes> scratch{};
};

template <class Ref>
inline UniformStreamReuse<Ref> g_uniform_stream_reuse_state{};

template <class Ref>
[[nodiscard]] inline UniformStreamReuse<Ref>& UniformStreamReuseState() noexcept {
    return g_uniform_stream_reuse_state<Ref>;
}

/// Forgets every slot. Call it when a buffer cache (and so its runtime, scheduler and stream
/// buffer) is created: ticks restart with a new scheduler and the old references are dead.
template <class Runtime>
inline void ResetUniformStreamReuse() {
    if constexpr (StreamUniformReusable<Runtime>) {
        UniformStreamReuseState<typename Runtime::StreamUniformRef>().Reset();
    }
}

[[nodiscard]] inline bool UniformBytes16Equal(const u8* a, const u8* b) noexcept {
#if defined(__aarch64__) || defined(_M_ARM64)
    const uint8x16_t d = veorq_u8(vld1q_u8(a), vld1q_u8(b));
    return vmaxvq_u32(vreinterpretq_u32_u8(d)) == 0;
#else
    u64 a0, a1, b0, b1;
    std::memcpy(&a0, a, sizeof(u64));
    std::memcpy(&a1, a + sizeof(u64), sizeof(u64));
    std::memcpy(&b0, b, sizeof(u64));
    std::memcpy(&b1, b + sizeof(u64), sizeof(u64));
    return ((a0 ^ b0) | (a1 ^ b1)) == 0;
#endif
}

/// Copies `size` bytes from `src` to BOTH `dst_cache` and `dst_stream` reading `src` strictly
/// once into registers, guaranteeing both outputs hold bit-identical bytes even under racing guest
/// CPU writes while halving source cache reads compared to two sequential `std::memcpy` calls.
inline void UniformDualCopy(u8* __restrict dst_cache, u8* __restrict dst_stream,
                            const u8* __restrict src, u32 size) noexcept {
    size_t off = 0;
#ifdef __AVX2__
    for (; off + 128 <= size; off += 128) {
        const __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + off));
        const __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + off + 32));
        const __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + off + 64));
        const __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + off + 96));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_cache + off), v0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_cache + off + 32), v1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_cache + off + 64), v2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_cache + off + 96), v3);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_stream + off), v0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_stream + off + 32), v1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_stream + off + 64), v2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_stream + off + 96), v3);
    }
    for (; off + 32 <= size; off += 32) {
        const __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + off));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_cache + off), v0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_stream + off), v0);
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    for (; off + 64 <= size; off += 64) {
        const uint8x16_t v0 = vld1q_u8(src + off);
        const uint8x16_t v1 = vld1q_u8(src + off + 16);
        const uint8x16_t v2 = vld1q_u8(src + off + 32);
        const uint8x16_t v3 = vld1q_u8(src + off + 48);
        vst1q_u8(dst_cache + off, v0);
        vst1q_u8(dst_cache + off + 16, v1);
        vst1q_u8(dst_cache + off + 32, v2);
        vst1q_u8(dst_cache + off + 48, v3);
        vst1q_u8(dst_stream + off, v0);
        vst1q_u8(dst_stream + off + 16, v1);
        vst1q_u8(dst_stream + off + 32, v2);
        vst1q_u8(dst_stream + off + 48, v3);
    }
    for (; off + 16 <= size; off += 16) {
        const uint8x16_t v0 = vld1q_u8(src + off);
        vst1q_u8(dst_cache + off, v0);
        vst1q_u8(dst_stream + off, v0);
    }
#endif
    for (; off + sizeof(u64) <= size; off += sizeof(u64)) {
        u64 word;
        std::memcpy(&word, src + off, sizeof(u64));
        std::memcpy(dst_cache + off, &word, sizeof(u64));
        std::memcpy(dst_stream + off, &word, sizeof(u64));
    }
    for (; off < size; ++off) {
        const u8 b = src[off];
        dst_cache[off] = b;
        dst_stream[off] = b;
    }
}

/// Binds the small uniform buffer at `device_addr` for (stage, cbuf `index`) through the stream
/// buffer, binding an earlier allocation again when it holds the same bytes and was made in the
/// current command buffer: the slot's previous copy or, with `range_reuse` (EDEN_NX2_EXP17,
/// from zortos293/Lichii#641), a copy of the same guest range made by any stage or slot. Every
/// reuse compares the guest bytes with the copy's. Returns false (nothing bound) when the buffer
/// does not qualify, so the caller binds it the classic way.
[[nodiscard]] inline bool UniformBytesEqual(const u8* copy_bytes, const u8* guest, u32 size,
                                            u16& diff_offset) noexcept {
    if (size <= 32) {
        if (size >= 16) {
            return UniformBytes16Equal(copy_bytes, guest) &&
                   (size == 16 || UniformBytes16Equal(copy_bytes + size - 16, guest + size - 16));
        }
        if (size >= sizeof(u64)) {
            u64 a0, b0, a1, b1;
            std::memcpy(&a0, copy_bytes, sizeof(u64));
            std::memcpy(&b0, guest, sizeof(u64));
            std::memcpy(&a1, copy_bytes + size - sizeof(u64), sizeof(u64));
            std::memcpy(&b1, guest + size - sizeof(u64), sizeof(u64));
            return ((a0 ^ b0) | (a1 ^ b1)) == 0;
        }
        if (size >= sizeof(u32)) {
            u32 a0, b0, a1, b1;
            std::memcpy(&a0, copy_bytes, sizeof(u32));
            std::memcpy(&b0, guest, sizeof(u32));
            std::memcpy(&a1, copy_bytes + size - sizeof(u32), sizeof(u32));
            std::memcpy(&b1, guest + size - sizeof(u32), sizeof(u32));
            return ((a0 ^ b0) | (a1 ^ b1)) == 0;
        }
        if (size == 0) {
            return true;
        }
        return copy_bytes[0] == guest[0] && copy_bytes[size - 1] == guest[size - 1] &&
               copy_bytes[size >> 1] == guest[size >> 1];
    }
    if (diff_offset != 0 && static_cast<u32>(diff_offset) + 16U <= size) {
        if (!UniformBytes16Equal(copy_bytes + diff_offset, guest + diff_offset)) {
            return false;
        }
    }
    if (!UniformBytes16Equal(copy_bytes, guest)) {
        diff_offset = 0;
        return false;
    }
    if (!UniformBytes16Equal(copy_bytes + 16, guest + 16)) {
        diff_offset = 16;
        return false;
    }
    if (size <= 64) {
        if (size > 48 && !UniformBytes16Equal(copy_bytes + 32, guest + 32)) {
            diff_offset = 32;
            return false;
        }
        if (!UniformBytes16Equal(copy_bytes + size - 16, guest + size - 16)) {
            diff_offset = static_cast<u16>(size - 16);
            return false;
        }
        return true;
    }
    if (!UniformBytes16Equal(copy_bytes + size - 16, guest + size - 16)) {
        diff_offset = static_cast<u16>(size - 16);
        return false;
    }
#ifdef __AVX2__
    size_t off = 32;
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
            for (size_t k = 0; k < 128; k += 16) {
                if (!UniformBytes16Equal(copy_bytes + off + k, guest + off + k)) {
                    diff_offset = static_cast<u16>(off + k);
                    break;
                }
            }
            return false;
        }
    }
    for (; off + 32 <= size; off += 32) {
        const __m256i d = _mm256_xor_si256(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(copy_bytes + off)),
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(guest + off)));
        if (!_mm256_testz_si256(d, d)) {
            diff_offset = static_cast<u16>(
                UniformBytes16Equal(copy_bytes + off, guest + off) ? off + 16 : off);
            return false;
        }
    }
    if (off + 16 < size && !UniformBytes16Equal(copy_bytes + off, guest + off)) {
        diff_offset = static_cast<u16>(off);
        return false;
    }
    return true;
#elif defined(__aarch64__) || defined(_M_ARM64)
    size_t off = 32;
    for (; off + 128 <= size; off += 128) {
        const uint8x16_t d0 = veorq_u8(vld1q_u8(copy_bytes + off), vld1q_u8(guest + off));
        const uint8x16_t d1 = veorq_u8(vld1q_u8(copy_bytes + off + 16), vld1q_u8(guest + off + 16));
        const uint8x16_t d2 = veorq_u8(vld1q_u8(copy_bytes + off + 32), vld1q_u8(guest + off + 32));
        const uint8x16_t d3 = veorq_u8(vld1q_u8(copy_bytes + off + 48), vld1q_u8(guest + off + 48));
        const uint8x16_t d4 = veorq_u8(vld1q_u8(copy_bytes + off + 64), vld1q_u8(guest + off + 64));
        const uint8x16_t d5 = veorq_u8(vld1q_u8(copy_bytes + off + 80), vld1q_u8(guest + off + 80));
        const uint8x16_t d6 = veorq_u8(vld1q_u8(copy_bytes + off + 96), vld1q_u8(guest + off + 96));
        const uint8x16_t d7 = veorq_u8(vld1q_u8(copy_bytes + off + 112), vld1q_u8(guest + off + 112));
        const uint8x16_t a0 = vorrq_u8(vorrq_u8(d0, d1), vorrq_u8(d2, d3));
        const uint8x16_t a1 = vorrq_u8(vorrq_u8(d4, d5), vorrq_u8(d6, d7));
        if (vmaxvq_u32(vreinterpretq_u32_u8(vorrq_u8(a0, a1))) != 0) {
            for (size_t k = 0; k < 128; k += 16) {
                if (!UniformBytes16Equal(copy_bytes + off + k, guest + off + k)) {
                    diff_offset = static_cast<u16>(off + k);
                    break;
                }
            }
            return false;
        }
    }
    for (; off + 32 <= size; off += 32) {
        const uint8x16_t d0 = veorq_u8(vld1q_u8(copy_bytes + off), vld1q_u8(guest + off));
        const uint8x16_t d1 = veorq_u8(vld1q_u8(copy_bytes + off + 16), vld1q_u8(guest + off + 16));
        if (vmaxvq_u32(vreinterpretq_u32_u8(vorrq_u8(d0, d1))) != 0) {
            diff_offset =
                static_cast<u16>(vmaxvq_u32(vreinterpretq_u32_u8(d0)) != 0 ? off : off + 16);
            return false;
        }
    }
    if (off + 16 < size && !UniformBytes16Equal(copy_bytes + off, guest + off)) {
        diff_offset = static_cast<u16>(off);
        return false;
    }
    return true;
#else
    if (std::memcmp(copy_bytes + 32, guest + 32, size - 48) != 0) {
        for (u32 off = 32; off + 16 <= size; off += 16) {
            if (!UniformBytes16Equal(copy_bytes + off, guest + off)) {
                diff_offset = static_cast<u16>(off);
                break;
            }
        }
        return false;
    }
    return true;
#endif
}

[[nodiscard]] inline bool UniformBytesEqual(const u8* copy_bytes, const u8* guest,
                                            u32 size) noexcept {
    u16 diff_offset = 0;
    return UniformBytesEqual(copy_bytes, guest, size, diff_offset);
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
                                           RangeCleanPred&& range_clean = {},
                                           u64 cpu_epoch = ~u64{0}) {
    using Reuse = UniformStreamReuse<typename Runtime::StreamUniformRef>;
    if (size == 0 || size > Reuse::MaxBytes || stage >= Reuse::NumStages ||
        index >= Reuse::NumSlots) {
        return false;
    }
    Reuse& reuse = UniformStreamReuseState<typename Runtime::StreamUniformRef>();
    reuse.EnsureAllocated();
    typename Reuse::Slot& slot = reuse.slots[stage][index];
    const u64 tick = runtime.CurrentTick();
    u32 needed = size;
    if constexpr (requires { runtime.StreamUniformAllocationSize(size); }) {
        needed = runtime.StreamUniformAllocationSize(size);
    }
    const auto rebind = [&](size_t copy_index) {
        const typename Reuse::Copy& copy = reuse.copies[copy_index];
        runtime.RebindStreamUniform(copy.ref, size);
        slot = typename Reuse::Slot{static_cast<u32>(copy_index), copy.serial};
        return true;
    };
    const auto is_verified_clean = [&](typename Reuse::Copy& copy) {
        if (write_serial == 0 || copy.verified_serial == 0 || cpu_epoch == 0) {
            return false;
        }
        if (copy.cpu_epoch != ~u64{0} && copy.cpu_epoch != cpu_epoch) {
            return false;
        }
        if (copy.verified_serial == write_serial) {
            return true;
        }
        if (range_clean(copy.page_gen)) {
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
            copy.allocated >= needed && is_verified_clean(copy)) [[likely]] {
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
                copy.tick == tick && copy.allocated >= needed && is_verified_clean(copy)) {
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
            guest = reuse.ResolveGuestSpan(device_memory, device_addr, size);
        }
    }
    const bool in_place = guest != nullptr;
    if (!in_place) {
        device_memory.ReadBlockUnsafe(device_addr, reuse.scratch.data(), size);
        guest = reuse.scratch.data();
    }
    const auto same_bytes = [&](size_t copy_index) {
        typename Reuse::Copy& copy = reuse.copies[copy_index];
        if (copy.size != size ||
            !UniformBytesEqual(reuse.Bytes(copy_index), guest, size, copy.diff_offset)) {
            return false;
        }
        if (copy.address == device_addr) {
            copy.verified_serial = write_serial;
            copy.page_gen = page_gen;
            copy.cpu_epoch = cpu_epoch;
        }
        return true;
    };
    bool slot_same = false;
    const bool slot_current_tick = slot_copy != NO_COPY && reuse.copies[slot_copy].tick == tick;
    const bool slot_can_check =
        slot_copy != NO_COPY && (slot_current_tick || range_reuse) &&
        (write_serial == 0 || reuse.copies[slot_copy].address == device_addr);
    if (slot_can_check && slot_current_tick) {
        slot_same = same_bytes(slot_copy);
        if (in_place) {
            namespace Profile = Nx2Profile;
            Profile::Add(slot_same ? Profile::Counter::UboInPlaceSame
                                   : Profile::Counter::UboInPlaceChanged,
                         1);
        }
        if (slot_same && reuse.copies[slot_copy].allocated >= needed) {
            return rebind(slot_copy);
        }
    }
    size_t target = stage * Reuse::NumSlots + index;
    if (range_reuse) {
        const size_t set = Reuse::Set(device_addr, size);
        const size_t base = Reuse::NumPrivate + set * Reuse::NumWays;
        for (size_t way = 0; way < Reuse::NumWays; ++way) {
            const size_t copy_index = base + way;
            const typename Reuse::Copy& copy = reuse.copies[copy_index];
            if (copy_index != slot_copy && copy.address == device_addr && copy.size == size &&
                copy.tick == tick && copy.allocated >= needed && same_bytes(copy_index)) {
                reuse.mru[set] = static_cast<u8>(way);
                if (in_place && !(slot_can_check && slot_current_tick)) {
                    Nx2Profile::Add(Nx2Profile::Counter::UboInPlaceSame, 1);
                }
                return rebind(copy_index);
            }
        }
        if (slot_can_check && !slot_current_tick) {
            slot_same = same_bytes(slot_copy);
            if (in_place) {
                namespace Profile = Nx2Profile;
                Profile::Add(slot_same ? Profile::Counter::UboInPlaceSame
                                       : Profile::Counter::UboInPlaceChanged,
                             1);
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
    } else if (in_place && !(slot_can_check && slot_current_tick)) {
        Nx2Profile::Add(Nx2Profile::Counter::UboInPlaceChanged, 1);
    }
    if (slot_same) {
        // The slot's copy from an older command buffer holds these bytes: fill the new
        // allocation from it.
        target = slot_copy;
        typename Reuse::Copy& copy = reuse.copies[target];
        const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
        std::memcpy(span.data(), reuse.Bytes(target), size);
        copy.ref = runtime.LastStreamUniformRef();
        copy.allocated = needed;
        // After the allocation: the tick the stream region is tagged with is the one current now.
        copy.tick = runtime.CurrentTick();
        slot = typename Reuse::Slot{static_cast<u32>(target), copy.serial};
        return true;
    }
    typename Reuse::Copy& fresh = reuse.copies[target];
    if (fresh.size != size) {
        fresh.diff_offset = 0;
    } else if (target != slot_copy && slot_copy != NO_COPY &&
               reuse.copies[slot_copy].size == size) {
        fresh.diff_offset = reuse.copies[slot_copy].diff_offset;
    }
    const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
    UniformDualCopy(reuse.Bytes(target), span.data(), guest, size);
    fresh.size = size;
    fresh.address = device_addr;
    fresh.verified_serial = write_serial;
    fresh.page_gen = page_gen;
    fresh.cpu_epoch = cpu_epoch;
    Reuse::NextSerial(fresh);
    fresh.ref = runtime.LastStreamUniformRef();
    fresh.allocated = needed;
    // After the allocation: the tick the stream region is tagged with is the one current now.
    fresh.tick = runtime.CurrentTick();
    slot = typename Reuse::Slot{static_cast<u32>(target), fresh.serial};
    return true;
}

template <class Runtime, class DeviceMemory, class RangeCleanPred = NoUboRangeClean>
[[nodiscard]] bool BindSmallComputeUniformReusing(Runtime& runtime, DeviceMemory& device_memory,
                                                  u32 index, u32 binding_index, u64 device_addr,
                                                  u32 size, u64 write_serial = 0,
                                                  u64 page_gen = 0,
                                                  RangeCleanPred&& range_clean = {},
                                                  u64 cpu_epoch = ~u64{0}) {
    using Reuse = UniformStreamReuse<typename Runtime::StreamUniformRef>;
    if (size == 0 || size > Reuse::MaxBytes || index >= Reuse::NumSlots) {
        return false;
    }
    Reuse& reuse = UniformStreamReuseState<typename Runtime::StreamUniformRef>();
    typename Reuse::ComputeSlot& slot = reuse.compute_slots[index];
    const u64 tick = runtime.CurrentTick();
    u32 needed = size;
    if constexpr (requires { runtime.StreamUniformAllocationSize(size); }) {
        needed = runtime.StreamUniformAllocationSize(size);
    }
    if (slot.tick == tick && slot.address == device_addr && slot.size == size &&
        slot.allocated >= needed && write_serial != 0 && slot.verified_serial != 0 &&
        cpu_epoch != 0 && (slot.cpu_epoch == ~u64{0} || slot.cpu_epoch == cpu_epoch)) {
        if (slot.verified_serial == write_serial || range_clean(slot.page_gen)) {
            slot.verified_serial = write_serial;
            runtime.RebindStreamUniform(slot.ref, size);
            return true;
        }
    }
    const u8* guest = nullptr;
    if constexpr (requires { device_memory.GetSpan(device_addr, size); }) {
        if (UniformInPlaceCompareEnabled()) {
            guest = reuse.ResolveGuestSpan(device_memory, device_addr, size);
        }
    }
    if (guest == nullptr) {
        device_memory.ReadBlockUnsafe(device_addr, reuse.scratch.data(), size);
        guest = reuse.scratch.data();
    }
    const bool same_bytes =
        slot.size == size &&
        UniformBytesEqual(slot.bytes.data(), guest, size, slot.diff_offset);
    if (same_bytes) {
        slot.address = device_addr;
        slot.verified_serial = write_serial;
        slot.page_gen = page_gen;
        slot.cpu_epoch = cpu_epoch;
        if (slot.tick == tick && slot.allocated >= needed) {
            runtime.RebindStreamUniform(slot.ref, size);
            return true;
        }
    }
    const std::span<u8> span = runtime.BindMappedUniformBuffer(0, binding_index, size);
    if (!same_bytes) {
        if (slot.size != size) {
            slot.diff_offset = 0;
        }
        UniformDualCopy(slot.bytes.data(), span.data(), guest, size);
        slot.size = size;
        slot.address = device_addr;
        slot.verified_serial = write_serial;
        slot.page_gen = page_gen;
        slot.cpu_epoch = cpu_epoch;
    } else {
        std::memcpy(span.data(), slot.bytes.data(), size);
    }
    slot.ref = runtime.LastStreamUniformRef();
    slot.allocated = needed;
    slot.tick = runtime.CurrentTick();
    return true;
}

} // namespace VideoCommon
