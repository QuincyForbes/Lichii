// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// EDEN_NX2_FEATURE_CENSUS=1: what a game asks the renderer for, as distinct records with counts
// (draw states, clears, dispatches, queries, programs, images, views, samplers, copies), written
// to nx2_feature_census.json in the log directory every 10 s. Both renderers feed it; it only
// reads state. tools/nx2/feature_census.py summarises the file, the Metal replay tool reads it.

#pragma once

#include <span>

#include "common/common_types.h"

namespace Tegra::Engines {
class Maxwell3D;
}

namespace Shader::IR {
struct Program;
}

namespace VideoCommon::FeatureCensus {

namespace Detail {
extern const bool enabled;
}

[[nodiscard]] inline bool Enabled() noexcept {
    return Detail::enabled;
}

/// DrawFlags of a draw record.
enum : u32 {
    DrawIndexed = 1U << 0,
    DrawInstanced = 1U << 1,
    DrawIndirect = 1U << 2,
    DrawIndirectCount = 1U << 3, ///< The draw count comes from a buffer
    DrawByteCount = 1U << 4,     ///< Transform feedback draw (draw auto)
    DrawMesh = 1U << 5,
    DrawTask = 1U << 6,
    DrawTessellated = 1U << 7,
    DrawGeometry = 1U << 8,
    DrawInlineIndex = 1U << 9,
    DrawFromGpuArguments = 1U << 10, ///< Mesh task draw whose group count the GPU reads
};

enum class Copy : u32 {
    CopyImage,
    EmulatedCopy,
    Reinterpret,
    Convert,
    OverlapCopy,
    OverlapCopyMsaa,
    Blit,
    BlitFramebuffer,
};

enum class Misc : u32 {
    DrawTexture,
    InlineToMemory,
    ConditionalRendering,
    TransformFeedbackBinding,
    SurfaceCopy,
    TemporalUpscale,
    DmaCopy,
    DownloadMemory,
    FragmentBarrier,
    TiledCacheBarrier,
    CounterReset,
    LayerFromGuestMemory, ///< A display layer read from guest memory (no cached image)
    AppletCaptureLayer,   ///< A frame with layers for the applet capture (LastFrame)
};

/// Registers a draw record holds after its fixed columns: `count` words from `first`, `step`
/// words apart (the column is `name`, or `name[index]` when there are several).
struct DrawRange {
    const char* name;
    u32 first;
    u32 count{1};
    u32 step{1};
};

[[nodiscard]] std::span<const DrawRange> DrawRanges() noexcept;

/// The draw the engine is about to hand the rasterizer (Draw* flags), for the next NoteDraw of
/// this thread.
void SetDrawKind(u32 flags, u32 instance_count);
/// A draw with the registers as they are: unique_hashes are the six programs of its pipeline
/// key. The rasterizer calls it once it has looked the pipeline up.
void NoteDraw(const Tegra::Engines::Maxwell3D& maxwell3d, std::span<const u64, 6> unique_hashes);
void NoteClear(const Tegra::Engines::Maxwell3D& maxwell3d, u32 layer_count);
void NoteDispatch(u64 unique_hash, bool is_indirect);
/// A report semaphore query: Maxwell's report type, and its raw query word.
void NoteQuery(u32 type, u32 raw_query, bool has_address);
/// A translated program: Shader::Stage as a number, and what its Info says it uses.
void NoteProgram(u64 unique_hash, const Shader::IR::Program& program);
void NoteImage(u32 format, u32 type, u32 samples, u32 levels, u32 layers, u32 flags);
void NoteImageView(u32 type, u32 format, u32 image_format, u32 image_type, u32 swizzle,
                   u32 levels, u32 layers);
[[nodiscard]] constexpr u32 PackSwizzle(u8 x, u8 y, u8 z, u8 w) noexcept {
    return u32{x} | u32{y} << 8 | u32{z} << 16 | u32{w} << 24;
}
void NoteSampler(std::span<const u64> raw);
void NoteCopy(Copy kind, u32 src_format, u32 dst_format, u32 src_samples, u32 dst_samples,
              u32 flags = 0);
void NoteMisc(Misc kind, u32 value = 0);

/// Once per frame: writes the file when 10 s have passed.
void Tick();

} // namespace VideoCommon::FeatureCensus
