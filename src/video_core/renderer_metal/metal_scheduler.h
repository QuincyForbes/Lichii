// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>

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

/// A clear of whole attachments (Maxwell clears): the load actions of the next render pass on
/// them, or a pass of its own when other work comes first (EDEN_NX2_METAL_CLEAR_FOLD).
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
    u32 min_width{};
    u32 min_height{};
    bool from_bound_mismatch{};
};

/// A clear drawn as a triangle: what a load action can't do (a rectangle of the target, some
/// colour channels, some stencil bits).
struct ClearRect {
    Attachment color{};
    u8 color_index{};                ///< Colour attachment index (0..7)
    u8 color_mask{};                 ///< R 1, G 2, B 4, A 8; 0: colour stays
    std::array<f64, 4> color_value{}; ///< Integers for integer formats
    Attachment depth{};              ///< The depth (and stencil) texture, if either is cleared
    bool clear_depth{};
    bool clear_stencil{};
    f32 depth_value{};
    u32 stencil_value{};
    u32 stencil_mask{0xFF};
    std::array<u32, 4> rect{}; ///< x, y, width, height
    u32 width{};
    u32 height{};
};

enum class BlitBlend : u8 {
    None,
    Premultiplied, ///< src + dst * (1 - src alpha)
    Coverage,      ///< src * src alpha + dst * (1 - src alpha)
};

enum class BlitColorOp : u8 {
    Copy,
    SrgbToLinear,
    Square,
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
    BlitColorOp color_op{BlitColorOp::Copy};
};

/// A copy between a multisampled texture and a single-sampled one that holds each sample as a
/// texel, as the guest sees a multisampled image (sample s of pixel p at texel
/// (p << shift) | (s & (2^shift.x - 1), s >> shift.x)). Colour and depth formats.
struct SampleCopy {
    TextureHandle src{};    ///< The multisampled texture (2D multisample array), or the 2D one
    u32 src_layer{};        ///< Layer of a multisampled source
    Attachment dst{};       ///< The single-sampled texel grid, or the multisampled texture
    u32 dst_format{};       ///< MTLPixelFormat of dst
    u32 dst_width{};        ///< dst's size at its level
    u32 dst_height{};
    u32 samples{};          ///< Samples of the multisampled side
    std::array<u32, 2> shift{}; ///< log2 of the guest's samples across and down
};

/// A nearest scaling of a whole layer that Blit can't draw: a multisampled source into a
/// texture of its sample count (sample for sample), or a stencil-only one (the fragment exports
/// the stencil). Resolution scaling of such images (Image::CopyScaled).
struct ScalePass {
    TextureHandle src{};  ///< A 2D multisample array, or a Stencil8 2D array view of one level
    u32 src_layer{};
    u32 src_width{};
    u32 src_height{};
    Attachment dst{};
    u32 dst_format{};
    u32 dst_width{};
    u32 dst_height{};
};

/// A conversion between a depth and a colour image (Vulkan's BlitImageHelper::Convert*).
enum class ConvertKind : u8 {
    DepthToColor, ///< D32F / D16 / X8D24 -> R32F, R16, ABGR8 (depth in every channel, alpha 1)
    ColorToDepth, ///< R32F / R16 -> D32F / D16 (red as the depth)
    Abgr8ToD32f,
    X8d24ToAbgr8,
    Abgr8ToX8d24,
    D24s8ToAbgr8, ///< S8_UINT_D24_UNORM's bytes
    S8d24ToAbgr8, ///< D24_UNORM_S8_UINT's bytes
    Abgr8ToD24s8, ///< Depth and stencil (stencil exported by the fragment)
};

struct ConvertPass {
    TextureHandle src{}; ///< A 2D view
    Attachment dst{};
    u32 dst_format{};
    u32 width{};
    u32 height{};
    ConvertKind kind{};
};

/// How the guest lays out depth and stencil in memory (OpenGL's GL_UNSIGNED_INT_24_8 and
/// GL_FLOAT_32_UNSIGNED_INT_24_8_REV, which take the guest's bytes as they are).
enum class DepthLayout : u8 {
    Packed24_8,   ///< u32: depth in the high 24 bits, stencil in the low 8 (S8_UINT_D24_UNORM)
    Float32_X24S8, ///< f32 depth, then u32 with the stencil in the low 8 bits
    Packed8_24,   ///< u32: stencil in the high 8 bits, depth in the low 24 (D24_UNORM_S8_UINT, X8_D24_UNORM)
};

/// The attachments draws render into. Equal sets share one render encoder.
struct RenderTargetSet {
    std::array<Attachment, 8> color{};
    Attachment depth{};
    bool has_depth{};
    bool has_stencil{};
    u32 width{};
    u32 height{};
    /// Layers a program may choose between (its layer output); 1: the bound slice alone.
    u32 layers{1};

    bool operator==(const RenderTargetSet&) const noexcept = default;
};

struct Viewport {
    f32 x{};
    f32 y{};
    f32 width{};
    f32 height{};
    f32 znear{};
    f32 zfar{1.0f};

    bool operator==(const Viewport&) const noexcept = default;
};

struct Scissor {
    u32 x{};
    u32 y{};
    u32 width{};
    u32 height{};

    bool operator==(const Scissor&) const noexcept = default;
};

enum class ShaderStage : u8 {
    Vertex,
    Fragment,
    Compute,
    /// The object and mesh functions of a mesh pipeline (on the render encoder).
    Object,
    Mesh,
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
    /// Viewports and scissors 1-15 follow 0 when a program writes the viewport index: 16, else 1.
    u8 viewport_count{1};
    std::array<Viewport, 15> more_viewports{};
    std::array<Scissor, 15> more_scissors{};
    /// Triangles drawn as their edges (MTLTriangleFillModeLines).
    bool fill_lines{};
    /// The depth bounds test (Apple10 GPUs: setDepthTestMinBound:maxBound:); 0 and 1 are off.
    f32 depth_bounds_min{0.0f};
    f32 depth_bounds_max{1.0f};
    /// What the draw can write of the pass's attachments (colour targets by index, depth,
    /// stencil); what no draw of a pass writes and the pass did not clear is not stored again
    /// (P07, EDEN_NX2_METAL_UNWRITTEN_STORE).
    u8 color_written{0xFF};
    bool depth_written{true};
    bool stencil_written{true};
};

/// A region of a sparse texture's tiles to map to its heap or unmap (G18); in tiles. At a mip
/// tail's level the region is the tail's tiles in x, one row.
struct SparseTileOp {
    u32 x{};
    u32 y{};
    u32 width{1};
    u32 height{1};
    u32 level{};
    bool map{true};
};

/// Why a command buffer was committed: Flush's callers, and the command buffers a tick commits
/// before its last one (Split, Ahead). Counted for the renderer's "Metal commits:" line.
enum class CommitReason : u8 {
    Other,
    Present,
    ListEnd,       ///< FlushCommands: a guest command list ended
    Fence,         ///< FenceTick: a fence with downloads or queries
    FenceDeferred, ///< FlushDeferred: work a fence or a list end left pending, before a wait
    Idle,          ///< FlushDeferredFromIdle: the split's back ran out of work
    Wait,          ///< Wait on the tick being recorded
    Finish,
    Staging,
    DrawBudget,
    ListCopy,
    Query,
    Condition,
    Split,
    Ahead,
    Count,
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
    void Wait(u64 tick, CommitReason reason = CommitReason::Wait);

    /// Whether `tick` was committed. From any thread.
    [[nodiscard]] bool IsSubmitted(u64 tick) const noexcept;

    /// Waits for a committed `tick` on the CPU. From any thread (Wait is the recording thread's).
    void WaitSubmitted(u64 tick);

    /// Waits for `tick` on the CPU from any thread, never recording or committing (the fencing
    /// thread, EDEN_NX2_METAL_ASYNC_FENCES). A tick not committed yet must be one the owner
    /// commits on its own: a fence's (FenceTick) or one marked with RequestCommit. The split's
    /// back is asked to run its idle hook, which commits such work between entries.
    void WaitCompleted(u64 tick, bool allow_batch_grace = false);

    /// Wait on the thread that owns the scheduler (GpuSplit::OwnsScheduler), else WaitCompleted.
    void WaitAnyThread(u64 tick, CommitReason reason = CommitReason::Wait,
                       bool allow_batch_grace = false);

    /// Owner: another thread will wait for the tick being recorded (WaitCompleted). It is left
    /// pending as a fence's work (FenceTick): the next commit, FlushDeferred or the back's idle
    /// hook commits it.
    void RequestCommit();

    /// The GPU thread split, seen through functions the fence manager installs so the scheduler
    /// itself doesn't link the split (the smoke test links the scheduler alone). Defaults: no
    /// split, the caller owns the scheduler.
    struct SplitHooks {
        bool (*back_running)() = nullptr;
        bool (*on_back_thread)() = nullptr;
        bool (*owns_scheduler)() = nullptr;
        void (*request_idle_hook)() = nullptr;
    };
    static void SetSplitHooks(const SplitHooks& hooks) noexcept;

    /// Shutdown: WaitCompleted returns at once from now on (nothing will commit any more).
    void AbandonWaits();

    /// Whether anything was recorded since the last Flush.
    [[nodiscard]] bool HasPendingWork() const noexcept;

    /// Commits the command buffer being recorded (even when empty) and returns its tick.
    /// EDEN_NX2_METAL_EMPTY_SKIP: with nothing recorded and every earlier tick done on the GPU,
    /// the tick is retired without a command buffer.
    u64 Flush(CommitReason reason = CommitReason::Other);

    /// Flush, then wait for the GPU to finish everything.
    void Finish(CommitReason reason = CommitReason::Finish);

    /// A guest command list ended (FlushCommands) with work recorded: whether it may stay
    /// pending for a later commit (EDEN_NX2_METAL_LIST_END_DEFER, on the split's back while the
    /// GPU still has that many command buffers of earlier work). The back's idle hook, a fence,
    /// a wait or the next list end once the GPU caught up commits it.
    bool DeferListEnd();

    /// Coalesces non-urgent flushes (such as speculative command-list copies) on GPUBack within
    /// EDEN_NX2_FENCE_FLUSH_US while the GPU already has work in flight.
    bool DeferFlush();

    /// The tick a guest fence queued now waits for. Pending work is committed for it; with none
    /// it is the last tick committed, and no command buffer is made (EDEN_NX2_METAL_FENCE_IDLE).
    /// On the split's back within EDEN_NX2_FENCE_FLUSH_US of the last commit the work stays
    /// pending (EDEN_NX2_METAL_FENCE_BATCH), as Vulkan's DeferFlush: the next commit, a fence
    /// past the window or the back going idle (FlushDeferredFromIdle) sends it.
    u64 FenceTick();

    /// The back's idle hook: commits work a fence left pending; whether it did.
    bool FlushDeferredFromIdle(bool requested);

    /// Commits work a fence left pending (FenceManager::SubmitDeferredWork).
    void FlushDeferred();

    /// Commits deferred work once its coalescing window has expired.
    void FlushExpiredDeferred();

    /// The back's idle check: whether pending fence work may wait a little longer for more.
    [[nodiscard]] bool DeferredFlushMayWait() const;

    /// Whether the GPU has more than one command buffer of earlier work left to run.
    [[nodiscard]] bool GpuBehind() const noexcept;

    /// Whether the GPU has finished every committed command buffer so far.
    [[nodiscard]] bool GpuCaughtUp() const noexcept;

    /// Draws recorded since the last Flush.
    [[nodiscard]] u64 DrawsSinceCommit() const noexcept;

    /// Since the last call: fences whose work was left pending, and the commits of such work by
    /// the idle hook.
    [[nodiscard]] std::array<u64, 2> TakeDeferredFences() noexcept;

    /// ahead: recorded before everything else of the segment (see CanUploadAhead), leaving the
    /// open encoder as it is. `range_tracked`: the caller tracks exact byte ranges in
    /// Buffer::tracker so whole-buffer untracked serialization is not needed.
    void CopyBuffer(BufferHandle src, u64 src_offset, BufferHandle dst, u64 dst_offset, u64 size,
                    bool ahead = false, bool range_tracked = false,
                    bool src_range_tracked = false);

    /// Staging readback copy whose destination is only read on the host after CurrentTick()
    /// completes: batched to segment end (or flushed before an overlapping GPU write to `src`)
    /// so fence downloads never break an open render pass or drain a concurrent compute burst.
    void DownloadBuffer(BufferHandle src, u64 src_offset, BufferHandle dst, u64 dst_offset,
                        u64 size);

    /// Flushes any deferred staging download whose source overlaps `[offset, offset + size)`.
    void FlushDeferredDownloadsFor(BufferHandle buffer, u64 offset = 0, u64 size = ~u64{0});

    /// EDEN_NX2_METAL_AHEAD: a segment is what one command buffer of a tick records in order.
    /// Uploads and kernels that do not depend on it go into one command buffer of their own in
    /// recording order (EDEN_NX2_METAL_MERGE_AHEAD=0: two, uploads then kernels), committed
    /// before it, so they do not end its render encoder.
    [[nodiscard]] u64 SegmentSerial() const noexcept;

    /// Whether a copy may go ahead of the segment. The caller answers for the destination not
    /// being read or written in the segment so far.
    [[nodiscard]] bool CanUploadAhead() const noexcept;

    /// Whether CopyBuffer(..., ahead = true) will actually record ahead of the segment.
    [[nodiscard]] bool CanCopyBufferAhead() const noexcept;

    /// Whether `[offset, offset + size)` of `dst` can be written ahead of the segment without
    /// conflicting with in-order or adopted spans.
    [[nodiscard]] bool CanWriteBufferAhead(BufferHandle dst, u64 offset = 0,
                                           u64 size = ~u64{0}) const noexcept;

    /// Whether `[offset, offset + size)` of `src` can be read ahead of the segment without
    /// missing earlier in-order or adopted writes.
    [[nodiscard]] bool CanReadBufferAhead(BufferHandle src, u64 offset = 0,
                                          u64 size = ~u64{0}) const noexcept;

    /// Whether all ahead copies and kernels share one ordered ahead command buffer.
    [[nodiscard]] bool CanOrderAheadCopies() const noexcept;

    /// Monotonic count of ahead buffer copies/fills recorded so far.
    [[nodiscard]] u64 AheadCopyEpoch() const noexcept;

    /// Latest ahead copy/fill serial recorded into `buffer` (0 if none).
    [[nodiscard]] u64 BufferAheadWriteSerial(BufferHandle buffer) const noexcept;

    /// Whether the segment's in-order compute encoder is currently open.
    [[nodiscard]] bool HasOpenInOrderCompute() const noexcept;

    /// Whether `buffer` is referenced by any uncommitted command buffer of the current tick.
    [[nodiscard]] bool IsBufferActiveInOpenCommands(BufferHandle buffer) const;

    /// Whether a texture upload or copy into `texture` may go ahead of the segment.
    [[nodiscard]] bool CanUploadTextureAhead(TextureHandle texture) const noexcept;

    /// Whether a copy, fill or texture copy of the segment reads or writes `buffer` in order.
    [[nodiscard]] bool UsedInOrder(BufferHandle buffer, u64 offset = 0,
                                   u64 size = ~u64{0}) const noexcept;

    /// Whether in-order GPU work of the segment writes `[offset, offset + size)` of `buffer`.
    [[nodiscard]] bool WrittenInOrder(BufferHandle buffer, u64 offset = 0,
                                      u64 size = ~u64{0}) const noexcept;

    /// Whether an in-order operation without per-range Buffer::tracker coverage touched `buffer`.
    [[nodiscard]] bool UsedUntrackedInOrder(BufferHandle buffer) const noexcept;

    /// Whether a kernel of a draw into `targets` may go ahead of the segment: nothing but the
    /// render pass on them was recorded in it, so the kernel sees what it would see in order.
    /// EDEN_NX2_METAL_AHEAD_ANY: also after other work; each buffer the kernel then binds is
    /// checked against the segment's in-order work, and a conflict moves it back in order.
    [[nodiscard]] bool CanRecordAhead(const RenderTargetSet& targets) const noexcept;

    /// Nothing more goes ahead in this segment (its record of what was used is incomplete).
    void BlockAhead() noexcept;

    [[nodiscard]] bool IsRenderingTo(const RenderTargetSet& targets) const noexcept;

    /// A render pass on `targets` was ended by a kernel that could have gone ahead of it, had
    /// the pass begun its segment: SplitBefore gives the next one a segment of its own.
    void WantSplitBefore(const RenderTargetSet& targets);
    void WantSplitBeforeCurrentPass();

    /// Before the first draw of a render pass on `targets`: when such a pass was wanted at the
    /// start of a segment, commits what was recorded (same tick) and starts the next segment.
    void SplitBefore(const RenderTargetSet& targets);

    void FillBuffer(BufferHandle dst, u64 offset, u64 size, u8 value, bool ahead = false,
                    bool range_tracked = false);
    void FillBufferPattern(BufferHandle dst, u64 offset, u64 size, u32 pattern, bool ahead = false,
                           bool range_tracked = false);

    void CopyBufferToTexture(BufferHandle src, u64 offset, u32 bytes_per_row, u32 bytes_per_image,
                             TextureHandle dst, u32 level, u32 slice, Offset3D origin,
                             Extent3D extent, bool ahead = false,
                             bool src_range_tracked = false);

    void CopyTextureToBuffer(TextureHandle src, u32 level, u32 slice, Offset3D origin,
                             Extent3D extent, BufferHandle dst, u64 offset, u32 bytes_per_row,
                             u32 bytes_per_image, bool dst_range_tracked = false);

    void CopyTexture(TextureHandle src, u32 src_level, u32 src_slice, Offset3D src_origin,
                     Extent3D extent, TextureHandle dst, u32 dst_level, u32 dst_slice,
                     Offset3D dst_origin, bool ahead = false);

    /// Ends the blit encoder, so the next copy sees the writes of the ones before it.
    void BreakBlits();

    void Clear(const ClearPass& pass);

    /// False when no pipeline can draw the clear (nothing is cleared).
    bool ClearRectangle(const ClearRect& clear);

    /// False when no pipeline can draw into dst's format (the blit is dropped).
    bool Blit(const BlitPass& pass);

    /// Multisampled src into its texel grid in dst (`expand`), or a texel grid into multisampled
    /// dst. False when no pipeline draws it (stencil, or a format of no colour or depth class).
    bool CopySamples(const SampleCopy& copy, bool expand);

    /// False when no pipeline draws the conversion into dst's format (nothing is drawn).
    bool Convert(const ConvertPass& pass);

    /// False when no pipeline scales into dst's format (a depth-stencil multisampled one).
    bool ScaleNearest(const ScalePass& pass);

    /// Guest depth (and stencil) bytes from `src` into a region of a depth or depth-stencil
    /// texture: drawn, the fragment writing depth and stencil (Metal's blit copies take depth and
    /// stencil as separate planes, and there is no 24-bit depth).
    bool DepthFromBuffer(BufferHandle src, u64 offset, u32 bytes_per_row, TextureHandle dst,
                         u32 level, u32 slice, Offset3D origin, Extent3D extent,
                         DepthLayout layout);

    /// The other way: a kernel reads depth (and stencil) and writes the guest's bytes.
    bool DepthToBuffer(TextureHandle src, u32 level, u32 slice, Offset3D origin, Extent3D extent,
                       BufferHandle dst, u64 offset, u32 bytes_per_row, DepthLayout layout);

    /// Q27: `shadow` (single-sampled, 2D array, the same format and size) is `msaa`'s resolve
    /// shadow: passes drawing into the multisampled texture resolve into it, and a pass that
    /// begins with a clear does not store the samples (EDEN_NX2_MSAA_DISCARD); whatever needs
    /// the samples later gets them restored from the shadow. A null shadow removes it.
    void SetResolveShadow(TextureHandle msaa, TextureHandle shadow);

    /// Drops any unconsumed deferred clear referencing `texture` before it is pooled or freed.
    void DropPendingClears(TextureHandle texture);

    /// The shadow of `msaa` while it holds the resolve of its samples (nothing wrote them since
    /// its last pass), else null.
    [[nodiscard]] TextureHandle ResolveShadow(TextureHandle msaa) const noexcept;

    /// Passes that did not store their samples, and the restores that followed.
    [[nodiscard]] u64 SamplesDropped() const noexcept;
    [[nodiscard]] u64 SamplesRestored() const noexcept;

    /// Store actions dropped so far for attachments a pass loaded and no draw of it wrote (P07).
    [[nodiscard]] u64 StoresDropped() const noexcept;

    /// Maps or unmaps tiles of a sparse texture (CreateSparseTexture), in order with the work
    /// recorded before and after it.
    bool UpdateSparseTiles(TextureHandle texture, std::span<const SparseTileOp> ops);

    /// A 16-bit packed format (Pack16Mode `mode`, an RGBA8 texture on the host) from or to the
    /// guest's words in `buffer`: a region of one level and slice. `to_buffer` reads the texture.
    bool Pack16(u32 mode, bool to_buffer, TextureHandle texture, u32 level, u32 slice, Offset3D origin,
                  Extent3D extent, BufferHandle buffer, u64 offset, u32 bytes_per_row);

    /// Exchanges red and blue (or reverses the nibbles, shift 4) of 16-bit texels in place in
    /// `buffer`: `images` images of `rows` rows of `width` texels (PackedSwapShift), on the GPU
    /// timeline: after the copies recorded before it, before those recorded after it.
    bool SwapPacked(BufferHandle buffer, u64 offset, u32 width, u32 rows, u32 images,
                    u32 bytes_per_row, u32 bytes_per_image, u32 shift);

    /// Resolves layer `src_slice` of multisampled `src` into dst with a render pass's resolve
    /// store action (the average of colour samples; sample 0 of depth, stencil kept). The
    /// whole texture: Metal resolves the render area.
    bool Resolve(TextureHandle src, u32 src_slice, const Attachment& dst, u32 width, u32 height);

    /// Opens a render encoder on the targets (loading their contents), or keeps the open one
    /// when they are the same.
    void BeginRendering(const RenderTargetSet& targets);

    /// Whether a render encoder for draws is open.
    [[nodiscard]] bool IsRendering() const noexcept;

    [[nodiscard]] const RenderTargetSet& OpenRenderTargets() const noexcept;
    [[nodiscard]] bool HasPendingClearFor(const RenderTargetSet& targets) const noexcept;
    [[nodiscard]] bool HasPendingClears() const noexcept;
    void PrepareSampledTextureForPass(TextureHandle texture);
    void FlushPendingClearsForTargets(const RenderTargetSet& targets);

    /// Ends the open render encoder: the next BeginRendering loads what it stored.
    void EndRendering();

    /// Whether the open render encoder may have written its depth or stencil attachment (a
    /// load-action clear, or a draw with depth writes or a writing stencil test); true when no
    /// encoder is open.
    [[nodiscard]] bool RenderPassWroteDepthStencil() const noexcept;
    [[nodiscard]] u8 RenderPassWroteColorMask() const noexcept;

    /// Slots of the visibility result buffer every draw render pass carries (occlusion queries).
    static constexpr u32 VisibilitySlots = 16384;

    /// Counts the next draws' passing samples into `slot` (adding to it within one render pass),
    /// or stops counting with nullopt.
    void SetVisibility(std::optional<u32> slot);

    /// The visibility results (one u64 per slot), readable once the GPU is done with them.
    [[nodiscard]] u64* VisibilityResults() noexcept;

    /// The visibility result buffer itself, for kernels reading the results in GPU order.
    [[nodiscard]] BufferHandle VisibilityBuffer() const noexcept;

    /// Changes whenever BeginRendering opens a new render encoder.
    [[nodiscard]] u64 RenderPassSerial() const noexcept;

    /// Changes whenever a render encoder opens or stops counting into its current visibility slot,
    /// so later draws never re-bind and zero a slot that already accumulated samples.
    [[nodiscard]] u64 VisibilitySerial() const noexcept;

    /// The state of the next draws on the open render encoder.
    void SetDrawState(const DrawState& state);

    void BindVertexBuffer(u32 slot, BufferHandle buffer, u32 offset);
    void BindBuffer(ShaderStage stage, u32 slot, BufferHandle buffer, u32 offset) {
        BindBuffer(stage, slot, buffer, offset, stage == ShaderStage::Compute);
    }
    void BindBuffer(ShaderStage stage, u32 slot, BufferHandle buffer, u32 offset, bool write,
                    u64 hazard_read_size = ~u64{0}, u64 hazard_write_size = ~u64{0});
    void BindBytes(ShaderStage stage, u32 slot, const void* data, u32 size);
    /// Makes `buffer` resident for `stage`'s next draws or dispatches without binding it (a
    /// program reads it through an address: ShaderTranslate::Options::spill_buffers); returns the
    /// address `offset` bytes into it.
    [[nodiscard]] u64 UseBuffer(ShaderStage stage, BufferHandle buffer, u32 offset) {
        return UseBuffer(stage, buffer, offset, stage == ShaderStage::Compute);
    }
    [[nodiscard]] u64 UseBuffer(ShaderStage stage, BufferHandle buffer, u32 offset, bool write,
                                u64 known_gpu_address = 0, u64 hazard_read_size = ~u64{0},
                                u64 hazard_write_size = ~u64{0});
    /// on_buffer=false: `texture` is known not to be a view of a buffer (a texture cache image
    /// view or a placeholder): render stages skip noting a buffer under it for the segment
    /// (EDEN_NX2_METAL_IMAGE_NOTE_SKIP).
    void BindTexture(ShaderStage stage, u32 slot, TextureHandle texture) {
        BindTexture(stage, slot, texture, true, stage == ShaderStage::Compute);
    }
    void BindTexture(ShaderStage stage, u32 slot, TextureHandle texture, bool on_buffer) {
        BindTexture(stage, slot, texture, on_buffer, stage == ShaderStage::Compute);
    }
    void BindTexture(ShaderStage stage, u32 slot, TextureHandle texture, bool on_buffer,
                     bool write, bool track_hazard = true);
    void BindSampler(ShaderStage stage, u32 slot, void* sampler);
    /// Flushes contiguous dirty binding runs and batched read-only residency declarations for
    /// `stage` onto the active encoder. Also flushed automatically before draws and dispatches.
    void FlushStageBindings(ShaderStage stage);
    /// Cached Tier-2 GPU address / MTLResourceID queries (0 for null).
    [[nodiscard]] static u64 BufferGpuAddress(BufferHandle buffer) noexcept;
    [[nodiscard]] static u64 TextureResourceId(TextureHandle texture) noexcept;
    /// The sampler's MTLResourceID, what a program reads from a table of spilled samplers
    /// (ShaderTranslate::Result::sampler_spill_slot). 0 for none.
    [[nodiscard]] static u64 SamplerResourceId(void* sampler) noexcept;
    /// A texture read (or written) through a table of resource IDs: made resident for the stage
    /// on the current encoder, its MTLResourceID returned (0 for none).
    [[nodiscard]] u64 UseTexture(ShaderStage stage, TextureHandle texture, bool write,
                                 bool on_buffer = true, u64 known_gpu_resource_id = 0,
                                 bool track_hazard = true);

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

    /// Opens the in-order compute encoder ahead of a dispatch's buffer uploads when compute-copy
    /// is enabled so those uploads stay on one compute encoder instead of opening a blit encoder.
    void PrepareCompute();

    /// Opens a compute encoder (or keeps the open one) running pipeline
    /// (an id<MTLComputePipelineState>). ahead: the encoder of the kernels recorded ahead of
    /// the segment (see CanRecordAhead); the binds and dispatches up to the next BeginCompute
    /// go to it.
    void BeginCompute(void* pipeline, bool ahead = false);

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

    /// A mesh pipeline's draw: `groups` object threadgroups (mesh threadgroups without an object
    /// function) along x.
    void DrawMeshThreadgroups(u32 groups, u32 object_threads, u32 mesh_threads);

    /// The same with the threadgroup counts (MTLDispatchThreadgroupsIndirectArguments: three u32)
    /// at `offset` of `buffer`.
    void DrawMeshThreadgroupsIndirect(BufferHandle buffer, u64 offset, u32 object_threads,
                                      u32 mesh_threads);

    /// index_size: 2 or 4 bytes.
    void DrawIndexed(u32 primitive, u32 index_count, u32 index_size, BufferHandle index_buffer,
                     u64 index_offset, u32 instance_count, s32 base_vertex, u32 base_instance);

    /// Ends the open encoder and returns the command buffer being recorded (an
    /// id<MTLCommandBuffer>, for presentDrawable and MetalFX). `allow_proxy`: keeps the recorder
    /// proxy when presentDrawable can be recorded on it without draining.
    [[nodiscard]] void* CommandBuffer(bool allow_proxy = false);

    /// Like CommandBuffer(false), for external encoders (such as MTLFXTemporalScaler) whose
    /// accessed textures are explicitly known and touch no guest MTLBuffers, keeping ahead
    /// uploads, clears, and checked kernels enabled for the rest of the segment.
    [[nodiscard]] void* CommandBufferForTextures(std::span<const TextureHandle> read_textures,
                                                 TextureHandle write_texture);

    /// The shared event signalled with each committed tick's number (id<MTLSharedEvent>).
    [[nodiscard]] void* SharedEvent() const noexcept;

    /// GPU time (MTLCommandBuffer GPUStartTime..GPUEndTime) of the command buffers completed
    /// since the last call.
    [[nodiscard]] u64 TakeGpuNanoseconds() noexcept;

    /// Since the last call (the renderer's 10 s report): encoders, command buffers, blit
    /// commands, render passes, clears made the load actions of a pass, state and bindings set,
    /// those left as they were (EDEN_NX2_METAL_STATE_CACHE), segments split off, copies and
    /// kernels recorded ahead (EDEN_NX2_METAL_AHEAD), fences that took no command buffer
    /// (EDEN_NX2_METAL_FENCE_IDLE).
    [[nodiscard]] std::array<u64, 11> TakeCounts() noexcept;

    /// Since the last call: multi-slot range bind calls (`set*Buffers/Textures/SamplerStates:`)
    /// and multi-resource read-only residency batches (`useResources:count:`).
    [[nodiscard]] std::array<u64, 2> TakeBatchCounts() noexcept;

    /// Since the last call, per frame (`frames`): command buffers committed by reason
    /// ("Metal commits:"), and render passes ended by what ended them plus those begun by kind
    /// ("Metal pass ends:").
    [[nodiscard]] std::string TakeCommitLine(double frames);
    [[nodiscard]] std::string TakePassEndLine(double frames);

    /// A drawable was presented, with a guest layer on it or not (the progress trace: the first
    /// of each, and the heartbeat that runs until the first guest frame).
    void NotePresent(bool guest_picture);

    /// Vulkan's EDEN_NX2_PASS_TIMES=<s> (read under EDEN_NX2_METAL_VULKAN_SWITCHES): one frame
    /// in that many seconds has each render pass timed at its stage boundaries (Metal's counter
    /// sample buffers), resolved once the GPU is done with it ("Perf Metal passes:" line).
    /// Called at each guest frame's present.
    void PassTimesFrameEnd();

    /// The last "Perf Metal passes:" text, empty when there was none since the last call.
    [[nodiscard]] std::string TakePassTimesLine();

private:
    void ResolvePassTimes();

    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Metal
