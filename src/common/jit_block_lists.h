// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>

#include "common/common_types.h"

namespace Common {

/// What the JIT keeps of a translated block after it is linked, in one heap block: the block's
/// code size, the blocks it branches to with the places to patch for each, and its fastmem
/// patch points. As a map of vectors plus a map this was four to six small heap blocks a block
/// (iPhone malloc census in Kokiri: 48 B x 2.0 M + 64 B x 1.6 M + 16 B x 3.0 M = 237 MiB).
///
/// Layout, in u64 words: header {u32 code size, u16 targets, u16 patches}, the targets
/// (ascending), the patches (ascending code offset, 3 words each), then per target the end
/// index of its relocations and last the relocations, both u32.
class JitBlockLists {
public:
    struct Relocation {
        u32 code_offset;
        /// Which of the JIT's two link patches (BlockRelocationType).
        u32 type;
    };

    struct Target {
        u64 location;
        std::span<const Relocation> relocations;
    };

    struct Patch {
        u32 code_offset;
        u64 marker_location;
        u32 marker_offset;
        u64 call_pc;
        bool recompile;
    };

    /// nullptr when a count or an offset does not fit the layout; the caller then keeps the
    /// lists as they are. targets and patches must be in ascending order.
    static std::unique_ptr<u64[]> Build(u64 code_size, std::span<const Target> targets,
                                        std::span<const Patch> patches) {
        std::size_t relocations = 0;
        for (const Target& target : targets) {
            for (const Relocation& relocation : target.relocations) {
                if (relocation.code_offset > MaxOffset || relocation.type > 1) {
                    return nullptr;
                }
            }
            relocations += target.relocations.size();
        }
        for (const Patch& patch : patches) {
            if (patch.code_offset > MaxOffset || patch.marker_offset > MaxOffset) {
                return nullptr;
            }
        }
        if (code_size > std::numeric_limits<u32>::max() ||
            targets.size() > std::numeric_limits<u16>::max() ||
            patches.size() > std::numeric_limits<u16>::max() || relocations > MaxOffset) {
            return nullptr;
        }
        const std::size_t words = Words(targets.size(), patches.size(), relocations);
        std::unique_ptr<u64[]> data{new u64[words]};
        data[0] = code_size | static_cast<u64>(targets.size()) << 32 |
                  static_cast<u64>(patches.size()) << 48;
        u64* out = data.get() + 1;
        for (const Target& target : targets) {
            *out++ = target.location;
        }
        for (const Patch& patch : patches) {
            *out++ = patch.marker_location;
            *out++ = patch.call_pc;
            *out++ = static_cast<u64>(patch.code_offset) |
                     static_cast<u64>(patch.marker_offset << 1 | (patch.recompile ? 1U : 0U)) << 32;
        }
        // The u32 tail; an odd count leaves half a word unused.
        if (out != data.get() + words) {
            data[words - 1] = 0;
        }
        u8* tail = reinterpret_cast<u8*>(out);
        const auto put = [&tail](u32 value) {
            std::memcpy(tail, &value, sizeof(value));
            tail += sizeof(value);
        };
        u32 end = 0;
        for (const Target& target : targets) {
            end += static_cast<u32>(target.relocations.size());
            put(end);
        }
        for (const Target& target : targets) {
            for (const Relocation& relocation : target.relocations) {
                put(relocation.code_offset << 1 | relocation.type);
            }
        }
        return data;
    }

    explicit JitBlockLists(const u64* data_) : data{data_} {}

    u32 CodeSize() const {
        return static_cast<u32>(data[0]);
    }

    std::size_t Targets() const {
        return static_cast<u16>(data[0] >> 32);
    }

    std::size_t Patches() const {
        return static_cast<u16>(data[0] >> 48);
    }

    u64 TargetLocation(std::size_t index) const {
        return data[1 + index];
    }

    /// The index of a target, or Targets() when the block does not branch there.
    std::size_t FindTarget(u64 location) const {
        const u64* const begin = data + 1;
        const u64* const end = begin + Targets();
        const u64* const found = std::lower_bound(begin, end, location);
        return found != end && *found == location ? static_cast<std::size_t>(found - begin)
                                                  : Targets();
    }

    /// func(Relocation) for each place that branches to the target at `index`.
    template <typename Func>
    void ForEachRelocation(std::size_t index, Func&& func) const {
        const std::size_t targets = Targets();
        const u32 begin = index == 0 ? 0 : Tail(index - 1);
        const u32 end = Tail(index);
        for (u32 i = begin; i < end; ++i) {
            const u32 packed = Tail(targets + i);
            func(Relocation{packed >> 1, packed & 1});
        }
    }

    std::optional<Patch> FindPatch(u64 code_offset) const {
        std::size_t low = 0;
        std::size_t high = Patches();
        while (low < high) {
            const std::size_t middle = low + (high - low) / 2;
            if (PatchOffset(middle) < code_offset) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        if (low == Patches() || PatchOffset(low) != code_offset) {
            return std::nullopt;
        }
        const u64* const patch = PatchWords(low);
        const u32 marker = static_cast<u32>(patch[2] >> 32);
        return Patch{
            .code_offset = static_cast<u32>(patch[2]),
            .marker_location = patch[0],
            .marker_offset = marker >> 1,
            .call_pc = patch[1],
            .recompile = (marker & 1) != 0,
        };
    }

    /// Heap bytes of the block.
    std::size_t Bytes() const {
        const std::size_t targets = Targets();
        const std::size_t relocations = targets == 0 ? 0 : Tail(targets - 1);
        return Words(targets, Patches(), relocations) * sizeof(u64);
    }

private:
    static constexpr u32 MaxOffset = std::numeric_limits<u32>::max() >> 1;
    static constexpr std::size_t PatchWordCount = 3;

    static std::size_t Words(std::size_t targets, std::size_t patches, std::size_t relocations) {
        return 1 + targets + patches * PatchWordCount + (targets + relocations + 1) / 2;
    }

    const u64* PatchWords(std::size_t index) const {
        return data + 1 + Targets() + index * PatchWordCount;
    }

    u32 PatchOffset(std::size_t index) const {
        return static_cast<u32>(PatchWords(index)[2]);
    }

    u32 Tail(std::size_t index) const {
        u32 value;
        std::memcpy(&value,
                    reinterpret_cast<const u8*>(data + 1 + Targets() + Patches() * PatchWordCount) +
                        index * sizeof(u32),
                    sizeof(u32));
        return value;
    }

    const u64* data;
};

} // namespace Common
