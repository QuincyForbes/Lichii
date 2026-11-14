// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <memory>

#include "common/common_types.h"
#include "video_core/renderer_metal/metal_object.h"

namespace Metal {

class Device;

struct Offset3D {
    u32 x{};
    u32 y{};
    u32 z{};
};

struct Extent3D {
    u32 width{1};
    u32 height{1};
    u32 depth{1};
};

/// One attachment of a render pass: a level and slice (or 3D depth plane) of a texture.
struct Attachment {
    TextureHandle texture{};
    u32 level{};
    u32 slice{};
    u32 depth_plane{};
};

/// A pass whose only work is clearing attachments (Maxwell clears; M3 folds them into the next
/// draw's pass as load actions).
struct ClearPass {
    std::array<Attachment, 8> color{};
    std::array<bool, 8> clear_color{};
    std::array<std::array<f64, 4>, 8> color_value{};
    Attachment depth{};
    Attachment stencil{};
    bool clear_depth{};
    bool clear_stencil{};
    f64 depth_value{};
    u32 stencil_value{};
    u32 width{};
    u32 height{};
};

/// A scaled copy of a rectangle of a 2D texture view into a render target, drawn with a
/// triangle (Fermi2D blits, presentation). A mirrored rectangle (end before start) flips.
struct BlitPass {
    TextureHandle src{};
    std::array<s32, 4> src_rect{}; ///< x0, y0, x1, y1 in texels of src
    Attachment dst{};
    u32 dst_format{};              ///< MTLPixelFormat of dst
    std::array<s32, 4> dst_rect{}; ///< x0, y0, x1, y1
    u32 dst_width{};
    u32 dst_height{};
    bool linear{};
    /// Keep what dst holds outside dst_rect; else it is cleared to black first.
    bool keep_dst{true};
};

/// Records Metal work in submission order on one command queue. Copies go to one blit encoder
/// until a render pass needs the command buffer, instead of an encoder per command; a command
/// buffer is committed by Flush. Ticks number the command buffers: a resource used in tick N is
/// free once IsFree(N).
class Scheduler {
public:
    explicit Scheduler(const Device& device);
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    /// The tick of the command buffer being recorded.
    [[nodiscard]] u64 CurrentTick() const noexcept;

    /// Whether the GPU finished every command buffer up to `tick`.
    [[nodiscard]] bool IsFree(u64 tick) const noexcept;

    /// Waits for `tick` on the CPU, committing it first when it is the one being recorded.
    void Wait(u64 tick);

    /// Whether anything was recorded since the last Flush.
    [[nodiscard]] bool HasPendingWork() const noexcept;

    /// Commits the command buffer being recorded (even when empty) and returns its tick.
    u64 Flush();

    /// Flush, then wait for the GPU to finish everything.
    void Finish();

    void CopyBuffer(BufferHandle src, u64 src_offset, BufferHandle dst, u64 dst_offset, u64 size);

    void FillBuffer(BufferHandle dst, u64 offset, u64 size, u8 value);

    void CopyBufferToTexture(BufferHandle src, u64 offset, u32 bytes_per_row, u32 bytes_per_image,
                             TextureHandle dst, u32 level, u32 slice, Offset3D origin,
                             Extent3D extent);

    void CopyTextureToBuffer(TextureHandle src, u32 level, u32 slice, Offset3D origin,
                             Extent3D extent, BufferHandle dst, u64 offset, u32 bytes_per_row,
                             u32 bytes_per_image);

    void CopyTexture(TextureHandle src, u32 src_level, u32 src_slice, Offset3D src_origin,
                     Extent3D extent, TextureHandle dst, u32 dst_level, u32 dst_slice,
                     Offset3D dst_origin);

    /// Ends the blit encoder, so the next copy sees the writes of the ones before it.
    void BreakBlits();

    void Clear(const ClearPass& pass);

    /// False when no pipeline can draw into dst's format (the blit is dropped).
    bool Blit(const BlitPass& pass);

    /// Ends the open encoder and returns the command buffer being recorded (an
    /// id<MTLCommandBuffer>, for presentDrawable and MetalFX).
    [[nodiscard]] void* CommandBuffer();

    /// Encoders and command buffers since the last call (the renderer's 10 s report).
    [[nodiscard]] std::array<u64, 4> TakeCounts() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Metal
