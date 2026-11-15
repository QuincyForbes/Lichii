// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include <optional>

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

    bool operator==(const Attachment&) const noexcept = default;
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

enum class BlitBlend : u8 {
    None,
    Premultiplied, ///< src + dst * (1 - src alpha)
    Coverage,      ///< src * src alpha + dst * (1 - src alpha)
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
    /// How the source is put over dst (presented layers above the first).
    BlitBlend blend{BlitBlend::None};
};

/// The attachments draws render into. Equal sets share one render encoder.
struct RenderTargetSet {
    std::array<Attachment, 8> color{};
    Attachment depth{};
    bool has_depth{};
    bool has_stencil{};
    u32 width{};
    u32 height{};

    bool operator==(const RenderTargetSet&) const noexcept = default;
};

struct Viewport {
    f32 x{};
    f32 y{};
    f32 width{};
    f32 height{};
    f32 znear{};
    f32 zfar{1.0f};
};

struct Scissor {
    u32 x{};
    u32 y{};
    u32 width{};
    u32 height{};
};

enum class ShaderStage : u8 {
    Vertex,
    Fragment,
    Compute,
};

/// Encoder state of a draw besides its bindings.
struct DrawState {
    void* pipeline{};      ///< id<MTLRenderPipelineState>
    void* depth_stencil{}; ///< id<MTLDepthStencilState>
    Viewport viewport{};
    Scissor scissor{};
    u8 cull_mode{};        ///< MTLCullMode
    bool front_ccw{};
    bool depth_clamp{};
    f32 depth_bias{};
    f32 slope_scale{};
    f32 bias_clamp{};
    u32 stencil_front_ref{};
    u32 stencil_back_ref{};
    std::array<f32, 4> blend_color{};
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

    /// Whether `tick` was committed. From any thread.
    [[nodiscard]] bool IsSubmitted(u64 tick) const noexcept;

    /// Waits for a committed `tick` on the CPU. From any thread (Wait is the recording thread's).
    void WaitSubmitted(u64 tick);

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

    /// Opens a render encoder on the targets (loading their contents), or keeps the open one
    /// when they are the same.
    void BeginRendering(const RenderTargetSet& targets);

    /// Slots of the visibility result buffer every draw render pass carries (occlusion queries).
    static constexpr u32 VisibilitySlots = 16384;

    /// Counts the next draws' passing samples into `slot` (adding to it within one render pass),
    /// or stops counting with nullopt.
    void SetVisibility(std::optional<u32> slot);

    /// The visibility results (one u64 per slot), readable once the GPU is done with them.
    [[nodiscard]] u64* VisibilityResults() noexcept;

    /// Changes whenever BeginRendering opens a new render encoder.
    [[nodiscard]] u64 RenderPassSerial() const noexcept;

    /// The state of the next draws on the open render encoder.
    void SetDrawState(const DrawState& state);

    void BindVertexBuffer(u32 slot, BufferHandle buffer, u32 offset);
    void BindBuffer(ShaderStage stage, u32 slot, BufferHandle buffer, u32 offset);
    void BindBytes(ShaderStage stage, u32 slot, const void* data, u32 size);
    void BindTexture(ShaderStage stage, u32 slot, TextureHandle texture);
    void BindSampler(ShaderStage stage, u32 slot, void* sampler);

    /// primitive: MTLPrimitiveType.
    void Draw(u32 primitive, u32 first_vertex, u32 vertex_count, u32 instance_count,
              u32 base_instance);

    /// One draw whose arguments (MTLDrawPrimitivesIndirectArguments, the guest's layout) are in
    /// `buffer` at `offset`.
    void DrawIndirect(u32 primitive, BufferHandle buffer, u64 offset);

    /// The same with MTLDrawIndexedPrimitivesIndirectArguments; indexStart counts from
    /// `index_offset`.
    void DrawIndexedIndirect(u32 primitive, u32 index_size, BufferHandle index_buffer,
                             u64 index_offset, BufferHandle buffer, u64 offset);

    /// Opens a compute encoder (or keeps the open one) running pipeline
    /// (an id<MTLComputePipelineState>).
    void BeginCompute(void* pipeline);

    void Dispatch(std::array<u32, 3> groups, std::array<u32, 3> group_size);

    /// The group counts are three u32 at `offset` of `buffer`.
    void DispatchIndirect(BufferHandle buffer, u64 offset, std::array<u32, 3> group_size);

    /// One thread per vertex (x, from `first`) and instance (y, from `first_instance`) of the
    /// open compute encoder's stage-in kernel (a tessellation pipeline's vertex program).
    void DispatchStageIn(u32 first, u32 count, u32 first_instance, u32 instances);

    /// The tessellation factors (one MTL*TessellationFactorsHalf per patch) of the next patches.
    void SetTessellationFactorBuffer(BufferHandle buffer, u64 offset);

    /// Tessellated patches of `control_points` control points each, read by the
    /// post-tessellation vertex function from its buffers.
    void DrawPatches(u32 control_points, u32 patch_count);

    /// index_size: 2 or 4 bytes.
    void DrawIndexed(u32 primitive, u32 index_count, u32 index_size, BufferHandle index_buffer,
                     u64 index_offset, u32 instance_count, s32 base_vertex, u32 base_instance);

    /// Ends the open encoder and returns the command buffer being recorded (an
    /// id<MTLCommandBuffer>, for presentDrawable and MetalFX).
    [[nodiscard]] void* CommandBuffer();

    /// GPU time (MTLCommandBuffer GPUStartTime..GPUEndTime) of the command buffers completed
    /// since the last call.
    [[nodiscard]] u64 TakeGpuNanoseconds() noexcept;

    /// Encoders and command buffers since the last call (the renderer's 10 s report).
    [[nodiscard]] std::array<u64, 4> TakeCounts() noexcept;

    /// A drawable was presented, with a guest layer on it or not (the progress trace: the first
    /// of each, and the heartbeat that runs until the first guest frame).
    void NotePresent(bool guest_picture);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Metal
