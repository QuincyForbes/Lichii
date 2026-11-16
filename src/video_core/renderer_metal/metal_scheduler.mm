// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#import <Metal/Metal.h>
#include <TargetConditionals.h>
#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/renderer_metal/metal_bridge.h"
#include "video_core/renderer_metal/metal_device.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {

namespace {

/// Apple GPUs' largest render target side.
constexpr u32 MaxRenderTargetSize = 16384;

/// A stage's argument tables as Metal's validation layer names them on Apple's own GPUs (a Mac's
/// other GPUs take 128 textures).
constexpr u32 BufferSlots = 31;
constexpr u32 TextureSlots = 96;
constexpr u32 SamplerSlots = 16;

/// Metal checks a slot past a stage's tables in its validation layer only.
bool SlotFits(u32 slot, u32 slots, const char* kind) {
    if (slot < slots || !PipelineCheckEnabled()) {
        return true;
    }
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_WARNING(Render_Metal, "Metal: {} slot {} not bound, a stage has {}", kind, slot,
                    slots);
    }
    return false;
}

constexpr const char* BlitSource = R"(
#include <metal_stdlib>
using namespace metal;

struct BlitOut {
    float4 position [[position]];
    float2 uv;
};

// One triangle over the viewport; uv_rect is the source rectangle (normalized) at its corners.
vertex BlitOut blit_vertex(uint id [[vertex_id]], constant float4& uv_rect [[buffer(0)]]) {
    const float2 corner = float2(float((id << 1) & 2), float(id & 2));
    BlitOut out;
    out.position = float4(corner.x * 2.0 - 1.0, 1.0 - corner.y * 2.0, 0.0, 1.0);
    out.uv = mix(uv_rect.xy, uv_rect.zw, corner);
    return out;
}

fragment float4 blit_float(BlitOut in [[stage_in]], texture2d<float> src [[texture(0)]],
                           sampler smp [[sampler(0)]]) {
    return src.sample(smp, in.uv);
}

fragment uint4 blit_uint(BlitOut in [[stage_in]], texture2d<uint> src [[texture(0)]]) {
    const float2 size = float2(src.get_width(), src.get_height());
    return src.read(uint2(clamp(in.uv * size, float2(0.0), size - 1.0)));
}

fragment int4 blit_sint(BlitOut in [[stage_in]], texture2d<int> src [[texture(0)]]) {
    const float2 size = float2(src.get_width(), src.get_height());
    return src.read(uint2(clamp(in.uv * size, float2(0.0), size - 1.0)));
}

struct DepthOut {
    float depth [[depth(any)]];
};

fragment DepthOut blit_depth(BlitOut in [[stage_in]], depth2d<float> src [[texture(0)]]) {
    const float2 size = float2(src.get_width(), src.get_height());
    DepthOut out;
    out.depth = src.read(uint2(clamp(in.uv * size, float2(0.0), size - 1.0)));
    return out;
}

struct DepthStencilOut {
    float depth [[depth(any)]];
    uint stencil [[stencil]];
};

// The stencil goes out as the reference value of a Replace stencil operation.
fragment DepthStencilOut blit_depth_stencil(BlitOut in [[stage_in]],
                                            depth2d<float> src [[texture(0)]],
                                            texture2d<uint> src_stencil [[texture(1)]]) {
    const float2 size = float2(src.get_width(), src.get_height());
    const uint2 texel = uint2(clamp(in.uv * size, float2(0.0), size - 1.0));
    DepthStencilOut out;
    out.depth = src.read(texel);
    out.stencil = src_stencil.read(texel).x;
    return out;
}

struct ClearOut {
    float4 position [[position]];
};

// One triangle over the viewport at the depth a clear writes.
vertex ClearOut clear_vertex(uint id [[vertex_id]], constant float& depth [[buffer(0)]]) {
    const float2 corner = float2(float((id << 1) & 2), float(id & 2));
    ClearOut out;
    out.position = float4(corner.x * 2.0 - 1.0, 1.0 - corner.y * 2.0, depth, 1.0);
    return out;
}

fragment float4 clear_float(constant float4& color [[buffer(0)]]) {
    return color;
}

fragment uint4 clear_uint(constant uint4& color [[buffer(0)]]) {
    return color;
}

fragment int4 clear_sint(constant int4& color [[buffer(0)]]) {
    return color;
}
)";

/// The stencil-only view format of a depth-stencil format (invalid when it has none).
MTLPixelFormat StencilViewFormat(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatDepth32Float_Stencil8:
        return MTLPixelFormatX32_Stencil8;
#if !TARGET_OS_IPHONE
    case MTLPixelFormatDepth24Unorm_Stencil8:
        return MTLPixelFormatX24_Stencil8;
#endif
    default:
        return MTLPixelFormatInvalid;
    }
}

void ApplyAttachment(MTLRenderPassAttachmentDescriptor* descriptor, const Attachment& attachment) {
    descriptor.texture = AsTexture(attachment.texture);
    descriptor.level = attachment.level;
    descriptor.slice = attachment.slice;
    descriptor.depthPlane = attachment.depth_plane;
}

id<MTLTexture> RootOf(id<MTLTexture> texture) {
    while (texture.parentTexture != nil) {
        texture = texture.parentTexture;
    }
    return texture;
}

// EDEN_NX2_METAL_CLEAR_FOLD=0: every clear is a render pass of its own, as this backend started.
bool ClearFoldEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_CLEAR_FOLD");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

// EDEN_NX2_METAL_STATE_CACHE=0: every draw and dispatch sets all its state and bindings again.
bool StateCacheEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_STATE_CACHE");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

// EDEN_NX2_METAL_AHEAD=0: uploads and kernels are recorded where they come, ending the open
// render encoder.
bool AheadEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_AHEAD");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

// EDEN_NX2_METAL_AHEAD_SPLIT=0: no command buffer is committed early to let a render pass begin
// a segment.
bool AheadSplitEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_AHEAD_SPLIT");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

// EDEN_NX2_METAL_FENCE_IDLE=0: a guest fence commits a command buffer even when nothing was
// recorded since the last one.
bool FenceIdleEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_FENCE_IDLE");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

/// What one stage of an encoder has bound; a new encoder has nothing bound, as this starts.
struct StageBindings {
    std::array<const void*, BufferSlots> buffers{};
    std::array<u32, BufferSlots> offsets{};
    std::array<const void*, TextureSlots> textures{};
    std::array<const void*, SamplerSlots> samplers{};
};

/// A buffer slot holding bytes (setBytes): its address stands for a buffer no other equals.
const char InlineBytes{};

enum class Bound {
    Same,   ///< Nothing to set
    Offset, ///< The same buffer at another offset
    Other,
};

void ReportFailure(id<MTLCommandBuffer> finished, u64 tick, std::atomic<u32>* errors_logged) {
    if (finished.status != MTLCommandBufferStatusError || errors_logged->fetch_add(1) >= 16) {
        return;
    }
    NSError* const error = finished.error;
    std::string encoders;
    NSArray* const infos = error.userInfo[MTLCommandBufferEncoderInfoErrorKey];
    for (id<MTLCommandBufferEncoderInfo> info in infos) {
        // MTLCommandEncoderErrorState: 0 unknown, 1 completed, 2 affected, 3 pending, 4 faulted.
        if (info.errorState != MTLCommandEncoderErrorStateCompleted) {
            encoders += fmt::format("{}{} (state {})", encoders.empty() ? "" : ", ",
                                    info.label.UTF8String != nullptr ? info.label.UTF8String : "?",
                                    static_cast<int>(info.errorState));
        }
    }
    LOG_ERROR(Render_Metal, "Metal: command buffer {} ({}) failed: code {}: {}; encoders: {}", tick,
              finished.label.UTF8String != nullptr ? finished.label.UTF8String : "?",
              static_cast<long>(error.code),
              error.localizedDescription.UTF8String != nullptr
                  ? error.localizedDescription.UTF8String
                  : "no description",
              encoders.empty() ? "none reported" : encoders);
}

/// Adds `clear` to the waiting `pending` when one render pass can do both.
bool MergeClear(ClearPass& pending, const ClearPass& clear) {
    if (pending.width != clear.width || pending.height != clear.height) {
        return false;
    }
    for (size_t index = 0; index < clear.color.size(); ++index) {
        if (clear.clear_color[index] && pending.clear_color[index] &&
            pending.color[index] != clear.color[index]) {
            return false;
        }
    }
    // One depth-stencil texture a pass.
    const Attachment* texture = nullptr;
    const std::array<const Attachment*, 4> depth_stencil{
        pending.clear_depth ? &pending.depth : nullptr,
        pending.clear_stencil ? &pending.stencil : nullptr,
        clear.clear_depth ? &clear.depth : nullptr,
        clear.clear_stencil ? &clear.stencil : nullptr,
    };
    for (const Attachment* const attachment : depth_stencil) {
        if (attachment == nullptr) {
            continue;
        }
        if (texture != nullptr && *texture != *attachment) {
            return false;
        }
        texture = attachment;
    }
    for (size_t index = 0; index < clear.color.size(); ++index) {
        if (clear.clear_color[index]) {
            pending.color[index] = clear.color[index];
            pending.clear_color[index] = true;
            pending.color_value[index] = clear.color_value[index];
        }
    }
    if (clear.clear_depth) {
        pending.depth = clear.depth;
        pending.clear_depth = true;
        pending.depth_value = clear.depth_value;
    }
    if (clear.clear_stencil) {
        pending.stencil = clear.stencil;
        pending.clear_stencil = true;
        pending.stencil_value = clear.stencil_value;
    }
    return true;
}

/// Whether a render pass on `targets` can clear what `clear` does as its load actions.
bool ClearFitsTargets(const ClearPass& clear, const RenderTargetSet& targets) {
    if (clear.width != targets.width || clear.height != targets.height) {
        return false;
    }
    for (size_t index = 0; index < clear.color.size(); ++index) {
        if (clear.clear_color[index] && clear.color[index] != targets.color[index]) {
            return false;
        }
    }
    if (clear.clear_depth && (!targets.has_depth || clear.depth != targets.depth)) {
        return false;
    }
    return !clear.clear_stencil || (targets.has_stencil && clear.stencil == targets.depth);
}

} // Anonymous namespace

struct Scheduler::Impl {
    explicit Impl(const Device& device_) : device{device_}, queue{device_.GetQueue()} {
        NSError* error = nil;
        library = [device.GetDevice() newLibraryWithSource:@(BlitSource) options:nil error:&error];
        if (library == nil) {
            LOG_ERROR(Render_Metal, "Metal: blit shaders failed to build: {}",
                      error.localizedDescription.UTF8String);
        }
        MTLSamplerDescriptor* sampler_desc = [MTLSamplerDescriptor new];
        sampler_desc.minFilter = MTLSamplerMinMagFilterNearest;
        sampler_desc.magFilter = MTLSamplerMinMagFilterNearest;
        nearest = [device.GetDevice() newSamplerStateWithDescriptor:sampler_desc];
        sampler_desc.minFilter = MTLSamplerMinMagFilterLinear;
        sampler_desc.magFilter = MTLSamplerMinMagFilterLinear;
        linear = [device.GetDevice() newSamplerStateWithDescriptor:sampler_desc];
        visibility = [device.GetDevice() newBufferWithLength:VisibilitySlots * sizeof(u64)
                                                     options:MTLResourceStorageModeShared];
        std::memset(visibility.contents, 0, VisibilitySlots * sizeof(u64));
    }

    id<MTLCommandBuffer> NewCommandBuffer(NSString* label) {
        // EDEN_NX2_METAL_ENCODER_STATUS=0: no per-encoder status in command buffer errors.
        static const bool encoder_status = [] {
            const char* const value = std::getenv("EDEN_NX2_METAL_ENCODER_STATUS");
            return value == nullptr || value[0] != '0';
        }();
        id<MTLCommandBuffer> buffer = nil;
        if (encoder_status) {
            MTLCommandBufferDescriptor* descriptor = [MTLCommandBufferDescriptor new];
            descriptor.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
            buffer = [queue commandBufferWithDescriptor:descriptor];
        } else {
            buffer = [queue commandBuffer];
        }
        buffer.label = label;
        return buffer;
    }

    id<MTLCommandBuffer> Commands() {
        if (command_buffer == nil) {
            command_buffer =
                NewCommandBuffer([NSString stringWithFormat:@"Lichii %llu", current_tick]);
        }
        return command_buffer;
    }

    /// The blit encoder of the copies recorded ahead of the segment, in a command buffer
    /// committed before the kernels' and the segment's.
    id<MTLBlitCommandEncoder> UploadEncoder() {
        if (upload_blit == nil) {
            upload_buffer =
                NewCommandBuffer([NSString stringWithFormat:@"Lichii %llu uploads", current_tick]);
            upload_blit = [upload_buffer blitCommandEncoder];
            upload_blit.label = @"Uploads";
            ++encoders;
        }
        ++blit_commands;
        ++ahead_copies;
        return upload_blit;
    }

    /// The compute encoder of the kernels recorded ahead of the segment.
    id<MTLComputeCommandEncoder> KernelEncoder() {
        if (kernel_compute == nil) {
            kernel_buffer =
                NewCommandBuffer([NSString stringWithFormat:@"Lichii %llu kernels", current_tick]);
            kernel_compute = [kernel_buffer computeCommandEncoder];
            kernel_compute.label = @"Kernels";
            kernel_bound = {};
            kernel_pipeline = nullptr;
            ++encoders;
        }
        return kernel_compute;
    }

    /// Commits a command buffer of the tick being recorded that is not its last one: the tick
    /// is done when the one Flush commits is.
    void CommitEarly(id<MTLCommandBuffer> buffer, u64 tick) {
        if (buffer == nil) {
            return;
        }
        std::atomic<u64>* const gpu_time = &gpu_ns;
        std::atomic<u32>* const errors = &errors_logged;
        [buffer addCompletedHandler:^(id<MTLCommandBuffer> finished) {
            const CFTimeInterval busy = finished.GPUEndTime - finished.GPUStartTime;
            if (busy > 0.0) {
                gpu_time->fetch_add(static_cast<u64>(busy * 1e9), std::memory_order_relaxed);
            }
            ReportFailure(finished, tick, errors);
        }];
        [buffer commit];
        ++command_buffers;
        in_flight.emplace_back(tick, buffer);
    }

    /// The uploads and kernels recorded ahead, before the segment they belong to.
    void CommitAhead(u64 tick) {
        if (upload_blit != nil) {
            [upload_blit endEncoding];
            upload_blit = nil;
        }
        if (kernel_compute != nil) {
            if (active_compute == kernel_compute) {
                active_compute = nil;
                active_bound = nullptr;
            }
            [kernel_compute endEncoding];
            kernel_compute = nil;
        }
        const id<MTLCommandBuffer> uploads = upload_buffer;
        const id<MTLCommandBuffer> kernels = kernel_buffer;
        upload_buffer = nil;
        kernel_buffer = nil;
        CommitEarly(uploads, tick);
        CommitEarly(kernels, tick);
    }

    void NextSegment() {
        ++segment_serial;
        segment_encoders = 0;
        ahead_blocked = false;
        in_order_buffers.clear();
    }

    /// A copy recorded in order reads or writes `buffer`: no copy into it goes ahead any more.
    void NoteInOrder(BufferHandle buffer) {
        if (std::find(in_order_buffers.rbegin(), in_order_buffers.rend(), buffer) ==
            in_order_buffers.rend()) {
            in_order_buffers.push_back(buffer);
        }
    }

    /// Commits what the tick recorded so far; the tick goes on in a new segment.
    void Split() {
        EndEncoder();
        CommitAhead(current_tick);
        const id<MTLCommandBuffer> segment = command_buffer;
        command_buffer = nil;
        CommitEarly(segment, current_tick);
        NextSegment();
        ++splits;
        ++tick_splits;
    }

    bool Wanted(const RenderTargetSet& targets) const {
        const auto end = wanted.begin() + (std::min)(wanted_count, wanted.size());
        return std::find(wanted.begin(), end, targets) != end;
    }

    /// What binding `buffer` at `offset` takes, with `bound` as the encoder has it.
    Bound NoteBuffer(StageBindings& bound, u32 slot, const void* buffer, u32 offset) {
        if (!StateCacheEnabled() || slot >= bound.buffers.size()) {
            ++state_sets;
            return Bound::Other;
        }
        const bool same_buffer = bound.buffers[slot] == buffer;
        if (same_buffer && (buffer == nullptr || bound.offsets[slot] == offset)) {
            ++state_skips;
            return Bound::Same;
        }
        bound.buffers[slot] = buffer;
        bound.offsets[slot] = offset;
        ++state_sets;
        return same_buffer ? Bound::Offset : Bound::Other;
    }

    /// Whether `object` has to be set where `bound` is.
    template <size_t N>
    bool NoteObject(std::array<const void*, N>& bound, u32 slot, const void* object) {
        if (!StateCacheEnabled() || slot >= N) {
            ++state_sets;
            return true;
        }
        if (bound[slot] == object) {
            ++state_skips;
            return false;
        }
        bound[slot] = object;
        ++state_sets;
        return true;
    }

    void SetBuffer(ShaderStage stage, u32 slot, BufferHandle buffer, u32 offset) {
        switch (stage) {
        case ShaderStage::Vertex:
            if (render == nil) {
                return;
            }
            switch (NoteBuffer(vertex_bound, slot, buffer, offset)) {
            case Bound::Same:
                break;
            case Bound::Offset:
                [render setVertexBufferOffset:offset atIndex:slot];
                break;
            case Bound::Other:
                [render setVertexBuffer:AsBuffer(buffer) offset:offset atIndex:slot];
                break;
            }
            break;
        case ShaderStage::Fragment:
            if (render == nil) {
                return;
            }
            switch (NoteBuffer(fragment_bound, slot, buffer, offset)) {
            case Bound::Same:
                break;
            case Bound::Offset:
                [render setFragmentBufferOffset:offset atIndex:slot];
                break;
            case Bound::Other:
                [render setFragmentBuffer:AsBuffer(buffer) offset:offset atIndex:slot];
                break;
            }
            break;
        case ShaderStage::Compute:
            if (active_compute == nil) {
                return;
            }
            switch (NoteBuffer(*active_bound, slot, buffer, offset)) {
            case Bound::Same:
                break;
            case Bound::Offset:
                [active_compute setBufferOffset:offset atIndex:slot];
                break;
            case Bound::Other:
                [active_compute setBuffer:AsBuffer(buffer) offset:offset atIndex:slot];
                break;
            }
            break;
        }
    }

    /// The progress trace: the first of each kind of work at Info, so a log that stops shows how
    /// far the renderer got.
    void CountDraw() {
        if (draws++ == 0) {
            LOG_INFO(Render_Metal, "Metal: first draw encoded (command buffer {}, render pass {})",
                     current_tick, render_pass_serial);
        }
    }

    void CountDispatch() {
        if (dispatches++ == 0) {
            LOG_INFO(Render_Metal, "Metal: first dispatch encoded (command buffer {})",
                     current_tick);
        }
    }

    void EndEncoder() {
        if (blit != nil) {
            [blit endEncoding];
            blit = nil;
        }
        if (render != nil) {
            [render endEncoding];
            render = nil;
        }
        if (compute != nil) {
            if (active_compute == compute) {
                active_compute = nil;
                active_bound = nullptr;
            }
            [compute endEncoding];
            compute = nil;
        }
    }

    id<MTLBlitCommandEncoder> BlitEncoder() {
        if (blit == nil) {
            EndEncoder();
            blit = [Commands() blitCommandEncoder];
            blit.label = @"Blit";
            ++encoders;
            ++segment_encoders;
        }
        ++blit_commands;
        return blit;
    }

    id<MTLRenderCommandEncoder> RenderEncoder(MTLRenderPassDescriptor* pass) {
        EndEncoder();
        ++encoders;
        ++segment_encoders;
        ++render_passes;
        const id<MTLRenderCommandEncoder> encoder =
            [Commands() renderCommandEncoderWithDescriptor:pass];
        if (encoder == nil) {
            // Metal refused the pass (draws into it are dropped): name it once.
            if (refused_passes++ < 4) {
                LOG_ERROR(Render_Metal, "Metal: render pass refused ({}x{}, colour 0 {}, depth {})",
                          pass.renderTargetWidth, pass.renderTargetHeight,
                          pass.colorAttachments[0].texture != nil
                              ? static_cast<u32>(pass.colorAttachments[0].texture.pixelFormat)
                              : 0U,
                          pass.depthAttachment.texture != nil
                              ? static_cast<u32>(pass.depthAttachment.texture.pixelFormat)
                              : 0U);
            }
            return nil;
        }
        encoder.label = [NSString stringWithFormat:@"Pass %llu", render_pass_serial + 1];
        return encoder;
    }

    /// A render pass that only clears.
    void RecordClear(const ClearPass& clear) {
        MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
        bool any = false;
        for (size_t index = 0; index < clear.color.size(); ++index) {
            if (clear.color[index].texture == nullptr || !clear.clear_color[index]) {
                continue;
            }
            MTLRenderPassColorAttachmentDescriptor* color = pass.colorAttachments[index];
            ApplyAttachment(color, clear.color[index]);
            const auto& value = clear.color_value[index];
            color.loadAction = MTLLoadActionClear;
            color.storeAction = MTLStoreActionStore;
            color.clearColor = MTLClearColorMake(value[0], value[1], value[2], value[3]);
            any = true;
        }
        if (clear.depth.texture != nullptr && clear.clear_depth) {
            ApplyAttachment(pass.depthAttachment, clear.depth);
            pass.depthAttachment.loadAction = MTLLoadActionClear;
            pass.depthAttachment.storeAction = MTLStoreActionStore;
            pass.depthAttachment.clearDepth = clear.depth_value;
            any = true;
        }
        if (clear.stencil.texture != nullptr && clear.clear_stencil) {
            ApplyAttachment(pass.stencilAttachment, clear.stencil);
            pass.stencilAttachment.loadAction = MTLLoadActionClear;
            pass.stencilAttachment.storeAction = MTLStoreActionStore;
            pass.stencilAttachment.clearStencil = clear.stencil_value;
            any = true;
        }
        if (!any) {
            return;
        }
        pass.renderTargetWidth = std::clamp(clear.width, 1U, MaxRenderTargetSize);
        pass.renderTargetHeight = std::clamp(clear.height, 1U, MaxRenderTargetSize);
        [RenderEncoder(pass) endEncoding];
    }

    /// The clear that waits for a render pass gets one of its own: work that may read or write
    /// its attachments comes next.
    void FlushClear() {
        if (pending_clear) {
            const ClearPass clear = *pending_clear;
            pending_clear.reset();
            RecordClear(clear);
            pending_clear_textures.clear();
        }
    }

    void HoldClear(const ClearPass& clear) {
        pending_clear_textures.clear();
        for (const Attachment& attachment : clear.color) {
            if (attachment.texture != nullptr) {
                pending_clear_textures.push_back(AsTexture(attachment.texture));
            }
        }
        for (const Attachment* const attachment : {&clear.depth, &clear.stencil}) {
            if (attachment->texture != nullptr) {
                pending_clear_textures.push_back(AsTexture(attachment->texture));
            }
        }
    }

    /// A copy reads or writes `texture`: the waiting clear of it (or of what it is a view of)
    /// goes first.
    void Touch(TextureHandle texture) {
        if (!pending_clear) {
            return;
        }
        const id<MTLTexture> root = RootOf(AsTexture(texture));
        const auto same = [&](const Attachment& attachment) {
            return attachment.texture != nullptr && RootOf(AsTexture(attachment.texture)) == root;
        };
        if (std::ranges::any_of(pending_clear->color, same) || same(pending_clear->depth) ||
            same(pending_clear->stencil)) {
            FlushClear();
        }
    }

    id<MTLRenderPipelineState> BlitPipeline(MTLPixelFormat format, BlitBlend blend) {
        const u64 key = (static_cast<u64>(format) << 8) | static_cast<u64>(blend);
        if (const auto it = blit_pipelines.find(key); it != blit_pipelines.end()) {
            return it->second;
        }
        id<MTLRenderPipelineState> pipeline = nil;
        const FormatClass format_class = ClassOf(static_cast<u32>(format));
        NSString* fragment = nil;
        switch (format_class) {
        case FormatClass::Float: fragment = @"blit_float"; break;
        case FormatClass::Uint: fragment = @"blit_uint"; break;
        case FormatClass::Sint: fragment = @"blit_sint"; break;
        case FormatClass::Depth: fragment = @"blit_depth"; break;
        case FormatClass::DepthStencil:
            fragment = StencilViewFormat(format) != MTLPixelFormatInvalid ? @"blit_depth_stencil"
                                                                          : @"blit_depth";
            break;
        default: break;
        }
        if (library != nil && fragment != nil) {
            MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
            desc.label = @"Blit";
            desc.vertexFunction = [library newFunctionWithName:@"blit_vertex"];
            desc.fragmentFunction = [library newFunctionWithName:fragment];
            if (format_class == FormatClass::Depth || format_class == FormatClass::DepthStencil) {
                desc.depthAttachmentPixelFormat = format;
                if (format_class == FormatClass::DepthStencil) {
                    desc.stencilAttachmentPixelFormat = format;
                }
            } else {
                desc.colorAttachments[0].pixelFormat = format;
            }
            if (blend != BlitBlend::None && format_class == FormatClass::Float) {
                MTLRenderPipelineColorAttachmentDescriptor* color = desc.colorAttachments[0];
                color.blendingEnabled = YES;
                color.sourceRGBBlendFactor = blend == BlitBlend::Premultiplied
                                                 ? MTLBlendFactorOne
                                                 : MTLBlendFactorSourceAlpha;
                color.sourceAlphaBlendFactor = color.sourceRGBBlendFactor;
                color.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
                color.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            }
            NSError* error = nil;
            pipeline = [device.GetDevice() newRenderPipelineStateWithDescriptor:desc error:&error];
            if (pipeline == nil) {
                LOG_ERROR(Render_Metal, "Metal: blit pipeline for format {} failed: {}",
                          static_cast<u32>(format), error.localizedDescription.UTF8String);
            }
        }
        blit_pipelines.emplace(key, pipeline);
        return pipeline;
    }

    /// The depth blits' state: every fragment writes its depth, and the stencil it outputs.
    id<MTLDepthStencilState> DepthWriteAlways() {
        if (depth_write_always == nil) {
            MTLDepthStencilDescriptor* desc = [MTLDepthStencilDescriptor new];
            desc.depthCompareFunction = MTLCompareFunctionAlways;
            desc.depthWriteEnabled = YES;
            MTLStencilDescriptor* stencil = [MTLStencilDescriptor new];
            stencil.stencilCompareFunction = MTLCompareFunctionAlways;
            stencil.depthStencilPassOperation = MTLStencilOperationReplace;
            stencil.readMask = 0xFF;
            stencil.writeMask = 0xFF;
            desc.frontFaceStencil = stencil;
            desc.backFaceStencil = stencil;
            depth_write_always = [device.GetDevice() newDepthStencilStateWithDescriptor:desc];
        }
        return depth_write_always;
    }

    /// The pipeline of a drawn clear: colour 0 with a write mask, or depth and stencil alone.
    id<MTLRenderPipelineState> ClearPipeline(MTLPixelFormat color, u8 mask, MTLPixelFormat depth,
                                             bool stencil, NSUInteger samples) {
        const u64 key = (static_cast<u64>(color) << 32) | (static_cast<u64>(depth) << 12) |
                        (static_cast<u64>(samples) << 5) | (static_cast<u64>(mask) << 1) |
                        (stencil ? 1U : 0U);
        if (const auto it = clear_pipelines.find(key); it != clear_pipelines.end()) {
            return it->second;
        }
        id<MTLRenderPipelineState> pipeline = nil;
        NSString* fragment = nil;
        if (color != MTLPixelFormatInvalid) {
            switch (ClassOf(static_cast<u32>(color))) {
            case FormatClass::Float: fragment = @"clear_float"; break;
            case FormatClass::Uint: fragment = @"clear_uint"; break;
            case FormatClass::Sint: fragment = @"clear_sint"; break;
            default: break;
            }
        }
        if (library != nil && (fragment != nil || color == MTLPixelFormatInvalid)) {
            MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
            desc.label = @"Clear";
            desc.vertexFunction = [library newFunctionWithName:@"clear_vertex"];
            desc.rasterSampleCount = samples;
            if (fragment != nil) {
                desc.fragmentFunction = [library newFunctionWithName:fragment];
                desc.colorAttachments[0].pixelFormat = color;
                MTLColorWriteMask write = MTLColorWriteMaskNone;
                write |= (mask & 1) != 0 ? MTLColorWriteMaskRed : MTLColorWriteMaskNone;
                write |= (mask & 2) != 0 ? MTLColorWriteMaskGreen : MTLColorWriteMaskNone;
                write |= (mask & 4) != 0 ? MTLColorWriteMaskBlue : MTLColorWriteMaskNone;
                write |= (mask & 8) != 0 ? MTLColorWriteMaskAlpha : MTLColorWriteMaskNone;
                desc.colorAttachments[0].writeMask = write;
            }
            if (depth != MTLPixelFormatInvalid && !stencil) {
                desc.depthAttachmentPixelFormat = depth;
            } else if (depth != MTLPixelFormatInvalid) {
                // A stencil-only format has no depth to attach.
                if (depth != MTLPixelFormatStencil8) {
                    desc.depthAttachmentPixelFormat = depth;
                }
                desc.stencilAttachmentPixelFormat = depth;
            }
            NSError* error = nil;
            pipeline = [device.GetDevice() newRenderPipelineStateWithDescriptor:desc error:&error];
            if (pipeline == nil) {
                LOG_ERROR(Render_Metal, "Metal: clear pipeline for formats {} and {} failed: {}",
                          static_cast<u32>(color), static_cast<u32>(depth),
                          error.localizedDescription.UTF8String);
            }
        }
        clear_pipelines.emplace(key, pipeline);
        return pipeline;
    }

    /// The state of a drawn clear: depth written or kept, the stencil bits of `mask` replaced.
    id<MTLDepthStencilState> ClearDepthState(bool depth, bool stencil, u32 mask) {
        const u32 key = (depth ? 0x200U : 0U) | (stencil ? 0x100U | (mask & 0xFFU) : 0U);
        if (const auto it = clear_depth_states.find(key); it != clear_depth_states.end()) {
            return it->second;
        }
        MTLDepthStencilDescriptor* desc = [MTLDepthStencilDescriptor new];
        desc.depthCompareFunction = MTLCompareFunctionAlways;
        desc.depthWriteEnabled = depth ? YES : NO;
        if (stencil) {
            MTLStencilDescriptor* face = [MTLStencilDescriptor new];
            face.stencilCompareFunction = MTLCompareFunctionAlways;
            face.depthStencilPassOperation = MTLStencilOperationReplace;
            face.readMask = 0xFF;
            face.writeMask = mask & 0xFFU;
            desc.frontFaceStencil = face;
            desc.backFaceStencil = face;
        }
        const id<MTLDepthStencilState> state =
            [device.GetDevice() newDepthStencilStateWithDescriptor:desc];
        clear_depth_states.emplace(key, state);
        return state;
    }

    const Device& device;
    id<MTLCommandQueue> queue;
    id<MTLDepthStencilState> depth_write_always = nil;
    /// By ClearPipeline's formats, mask and sample count.
    std::unordered_map<u64, id<MTLRenderPipelineState>> clear_pipelines;
    std::unordered_map<u32, id<MTLDepthStencilState>> clear_depth_states;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLBlitCommandEncoder> blit = nil;
    /// The render encoder draws go to, open until other work needs the command buffer.
    id<MTLRenderCommandEncoder> render = nil;
    RenderTargetSet render_targets{};
    /// A clear not recorded yet: the load actions of the next render pass on its attachments
    /// (EDEN_NX2_METAL_CLEAR_FOLD), with its textures held until then.
    std::optional<ClearPass> pending_clear;
    std::vector<id<MTLTexture>> pending_clear_textures;
    u64 clears_folded{};
    id<MTLComputeCommandEncoder> compute = nil;
    /// EDEN_NX2_METAL_STATE_CACHE: what the open encoders have set, so a draw or dispatch sets
    /// only what differs from the one before it.
    StageBindings vertex_bound;
    StageBindings fragment_bound;
    DrawState render_state{};
    bool render_state_valid{};
    StageBindings compute_bound;
    const void* compute_pipeline{};
    StageBindings kernel_bound;
    const void* kernel_pipeline{};
    /// The compute encoder binds and dispatches go to (the segment's or the kernels'), and what
    /// it has bound.
    id<MTLComputeCommandEncoder> active_compute = nil;
    StageBindings* active_bound{};
    u64 state_sets{};
    u64 state_skips{};
    /// EDEN_NX2_METAL_AHEAD: the command buffers committed before the segment's, with the one
    /// encoder each has.
    id<MTLCommandBuffer> upload_buffer = nil;
    id<MTLBlitCommandEncoder> upload_blit = nil;
    id<MTLCommandBuffer> kernel_buffer = nil;
    id<MTLComputeCommandEncoder> kernel_compute = nil;
    u64 segment_serial{1};
    /// Encoders the segment's command buffer got so far.
    u32 segment_encoders{};
    bool ahead_blocked{};
    /// Buffers the segment's copies read or write in order.
    std::vector<const void*> in_order_buffers;
    /// Segments the tick being recorded committed already.
    u32 tick_splits{};
    /// Render passes that are to begin a segment (WantSplitBefore).
    std::array<RenderTargetSet, 8> wanted{};
    size_t wanted_count{};
    u64 splits{};
    u64 ahead_copies{};
    u64 ahead_kernels{};
    u64 idle_fences{};
    id<MTLBuffer> visibility = nil;
    u64 render_pass_serial{};
    /// The open render encoder's counting slot (-1: not counting).
    s64 visibility_slot{-1};
    id<MTLLibrary> library = nil;
    id<MTLSamplerState> nearest = nil;
    id<MTLSamplerState> linear = nil;
    /// By format << 8 | BlitBlend.
    std::unordered_map<u64, id<MTLRenderPipelineState>> blit_pipelines;

    u64 current_tick{1};
    std::atomic<u64> submitted_tick{0};
    std::atomic<u64> completed_tick{0};
    /// Wakes WaitSubmitted when completed_tick moves.
    std::mutex completed_mutex;
    std::condition_variable completed_cv;
    /// GPU time of the command buffers completed since TakeGpuNanoseconds.
    std::atomic<u64> gpu_ns{0};
    std::deque<std::pair<u64, id<MTLCommandBuffer>>> in_flight;

    u64 encoders{};
    u64 command_buffers{};
    u64 blit_commands{};
    u64 render_passes{};
    /// Since boot (the progress trace).
    u64 draws{};
    u64 dispatches{};
    u64 committed{};
    /// Presents showing a guest layer, and those that had none to show.
    u64 presents{};
    u64 blank_presents{};
    u32 refused_passes{};
    std::chrono::steady_clock::time_point heartbeat = std::chrono::steady_clock::now();
    std::atomic<bool> completed_logged{false};
    std::atomic<u32> errors_logged{0};
};

Scheduler::Scheduler(const Device& device) : impl{std::make_unique<Impl>(device)} {}

Scheduler::~Scheduler() {
    Finish();
}

u64 Scheduler::CurrentTick() const noexcept {
    return impl->current_tick;
}

bool Scheduler::IsFree(u64 tick) const noexcept {
    return impl->completed_tick.load(std::memory_order_acquire) >= tick;
}

void Scheduler::Wait(u64 tick) {
    if (tick >= impl->current_tick) {
        Flush();
    }
    while (!impl->in_flight.empty() && impl->in_flight.front().first <= tick) {
        [impl->in_flight.front().second waitUntilCompleted];
        impl->in_flight.pop_front();
    }
}

bool Scheduler::IsSubmitted(u64 tick) const noexcept {
    return impl->submitted_tick.load(std::memory_order_acquire) >= tick;
}

void Scheduler::WaitSubmitted(u64 tick) {
    if (IsFree(tick)) {
        return;
    }
    std::unique_lock lock{impl->completed_mutex};
    impl->completed_cv.wait(lock, [&] { return IsFree(tick); });
}

bool Scheduler::HasPendingWork() const noexcept {
    return impl->command_buffer != nil || impl->pending_clear.has_value() ||
           impl->upload_buffer != nil || impl->kernel_buffer != nil || impl->tick_splits != 0;
}

u64 Scheduler::Flush() {
    impl->FlushClear();
    impl->EndEncoder();
    const id<MTLCommandBuffer> command_buffer = impl->Commands();
    const u64 tick = impl->current_tick++;
    std::atomic<u64>* const completed = &impl->completed_tick;
    std::atomic<u64>* const gpu_ns = &impl->gpu_ns;
    std::atomic<bool>* const completed_logged = &impl->completed_logged;
    std::atomic<u32>* const errors_logged = &impl->errors_logged;
    std::mutex* const completed_mutex = &impl->completed_mutex;
    std::condition_variable* const completed_cv = &impl->completed_cv;
    [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> finished) {
        const CFTimeInterval busy = finished.GPUEndTime - finished.GPUStartTime;
        if (busy > 0.0) {
            gpu_ns->fetch_add(static_cast<u64>(busy * 1e9), std::memory_order_relaxed);
        }
        if (!completed_logged->exchange(true)) {
            LOG_INFO(Render_Metal, "Metal: first command buffer completed (tick {}, status {}, "
                                   "GPU {:.2f} ms)",
                     tick, static_cast<int>(finished.status), busy * 1e3);
        }
        ReportFailure(finished, tick, errors_logged);
        u64 previous = completed->load(std::memory_order_relaxed);
        while (previous < tick &&
               !completed->compare_exchange_weak(previous, tick, std::memory_order_release)) {
        }
        {
            std::scoped_lock lock{*completed_mutex};
        }
        completed_cv->notify_all();
    }];
    impl->CommitAhead(tick);
    [command_buffer commit];
    impl->submitted_tick.store(tick, std::memory_order_release);
    ++impl->command_buffers;
    if (impl->committed++ == 0) {
        LOG_INFO(Render_Metal, "Metal: first command buffer committed (tick {})", tick);
    }
    // Until the first guest frame, a line every 5 s: where a boot that never shows one stands.
    if (impl->presents == 0) {
        const auto now = std::chrono::steady_clock::now();
        if (now - impl->heartbeat >= std::chrono::seconds{5}) {
            impl->heartbeat = now;
            LOG_INFO(Render_Metal,
                     "Metal: no guest frame presented yet; {} command buffers committed, {} "
                     "completed, {} draws, {} dispatches, {} presents without a guest picture",
                     impl->committed, impl->completed_tick.load(std::memory_order_relaxed),
                     impl->draws, impl->dispatches, impl->blank_presents);
        }
    }
    impl->command_buffer = nil;
    impl->NextSegment();
    impl->tick_splits = 0;
    impl->in_flight.emplace_back(tick, command_buffer);
    while (!impl->in_flight.empty() && IsFree(impl->in_flight.front().first)) {
        impl->in_flight.pop_front();
    }
    return tick;
}

void Scheduler::Finish() {
    Wait(Flush());
}

u64 Scheduler::FenceTick() {
    if (FenceIdleEnabled() && !HasPendingWork()) {
        // Everything before the fence is in the ticks committed so far (tick 0 is always free).
        ++impl->idle_fences;
        return impl->current_tick - 1;
    }
    return Flush();
}

void Scheduler::CopyBuffer(BufferHandle src, u64 src_offset, BufferHandle dst, u64 dst_offset,
                           u64 size, bool ahead) {
    if (size == 0 || src == nullptr || dst == nullptr) {
        return;
    }
    const bool goes_ahead = ahead && CanUploadAhead();
    if (!goes_ahead) {
        impl->NoteInOrder(src);
        impl->NoteInOrder(dst);
    }
    const id<MTLBlitCommandEncoder> encoder =
        goes_ahead ? impl->UploadEncoder() : impl->BlitEncoder();
    [encoder copyFromBuffer:AsBuffer(src)
               sourceOffset:src_offset
                   toBuffer:AsBuffer(dst)
          destinationOffset:dst_offset
                       size:size];
}

u64 Scheduler::SegmentSerial() const noexcept {
    return impl->segment_serial;
}

bool Scheduler::CanUploadAhead() const noexcept {
    return AheadEnabled() && !impl->ahead_blocked;
}

bool Scheduler::UsedInOrder(BufferHandle buffer) const noexcept {
    return std::find(impl->in_order_buffers.begin(), impl->in_order_buffers.end(), buffer) !=
           impl->in_order_buffers.end();
}

bool Scheduler::CanRecordAhead(const RenderTargetSet& targets) const noexcept {
    if (!AheadEnabled() || impl->ahead_blocked) {
        return false;
    }
    if (impl->pending_clear && !ClearFitsTargets(*impl->pending_clear, targets)) {
        return false;
    }
    return impl->segment_encoders == 0 || (impl->segment_encoders == 1 && IsRenderingTo(targets));
}

void Scheduler::BlockAhead() noexcept {
    impl->ahead_blocked = true;
}

bool Scheduler::IsRenderingTo(const RenderTargetSet& targets) const noexcept {
    return impl->render != nil && impl->render_targets == targets;
}

void Scheduler::WantSplitBefore(const RenderTargetSet& targets) {
    if (!AheadEnabled() || !AheadSplitEnabled() || impl->Wanted(targets)) {
        return;
    }
    impl->wanted[impl->wanted_count++ % impl->wanted.size()] = targets;
}

void Scheduler::SplitBefore(const RenderTargetSet& targets) {
    // A pass that other work keeps ending is not given a command buffer each time.
    constexpr u32 MaxTickSplits = 8;
    if (!AheadEnabled() || !AheadSplitEnabled() || IsRenderingTo(targets) ||
        impl->tick_splits >= MaxTickSplits || !impl->Wanted(targets)) {
        return;
    }
    // A clear waiting for this pass stays its load actions, in the new segment.
    if (impl->pending_clear && !ClearFitsTargets(*impl->pending_clear, targets)) {
        impl->FlushClear();
    }
    if (impl->segment_encoders != 0) {
        impl->Split();
    }
}

void Scheduler::FillBuffer(BufferHandle dst, u64 offset, u64 size, u8 value) {
    if (size == 0 || dst == nullptr) {
        return;
    }
    impl->NoteInOrder(dst);
    [impl->BlitEncoder() fillBuffer:AsBuffer(dst) range:NSMakeRange(offset, size) value:value];
}

void Scheduler::CopyBufferToTexture(BufferHandle src, u64 offset, u32 bytes_per_row,
                                    u32 bytes_per_image, TextureHandle dst, u32 level, u32 slice,
                                    Offset3D origin, Extent3D extent) {
    if (src == nullptr || dst == nullptr) {
        return;
    }
    impl->Touch(dst);
    impl->NoteInOrder(src);
    [impl->BlitEncoder() copyFromBuffer:AsBuffer(src)
                           sourceOffset:offset
                      sourceBytesPerRow:bytes_per_row
                    sourceBytesPerImage:bytes_per_image
                             sourceSize:MTLSizeMake(extent.width, extent.height, extent.depth)
                              toTexture:AsTexture(dst)
                       destinationSlice:slice
                       destinationLevel:level
                      destinationOrigin:MTLOriginMake(origin.x, origin.y, origin.z)];
}

void Scheduler::CopyTextureToBuffer(TextureHandle src, u32 level, u32 slice, Offset3D origin,
                                    Extent3D extent, BufferHandle dst, u64 offset,
                                    u32 bytes_per_row, u32 bytes_per_image) {
    if (src == nullptr || dst == nullptr) {
        return;
    }
    impl->Touch(src);
    impl->NoteInOrder(dst);
    [impl->BlitEncoder() copyFromTexture:AsTexture(src)
                             sourceSlice:slice
                             sourceLevel:level
                            sourceOrigin:MTLOriginMake(origin.x, origin.y, origin.z)
                              sourceSize:MTLSizeMake(extent.width, extent.height, extent.depth)
                                toBuffer:AsBuffer(dst)
                       destinationOffset:offset
                  destinationBytesPerRow:bytes_per_row
                destinationBytesPerImage:bytes_per_image];
}

void Scheduler::CopyTexture(TextureHandle src, u32 src_level, u32 src_slice, Offset3D src_origin,
                            Extent3D extent, TextureHandle dst, u32 dst_level, u32 dst_slice,
                            Offset3D dst_origin) {
    if (src == nullptr || dst == nullptr) {
        return;
    }
    impl->Touch(src);
    impl->Touch(dst);
    [impl->BlitEncoder() copyFromTexture:AsTexture(src)
                             sourceSlice:src_slice
                             sourceLevel:src_level
                            sourceOrigin:MTLOriginMake(src_origin.x, src_origin.y, src_origin.z)
                              sourceSize:MTLSizeMake(extent.width, extent.height, extent.depth)
                               toTexture:AsTexture(dst)
                        destinationSlice:dst_slice
                        destinationLevel:dst_level
                       destinationOrigin:MTLOriginMake(dst_origin.x, dst_origin.y, dst_origin.z)];
}

void Scheduler::BreakBlits() {
    impl->EndEncoder();
}

void Scheduler::Clear(const ClearPass& clear) {
    if (!ClearFoldEnabled()) {
        impl->RecordClear(clear);
        return;
    }
    // The clear waits for the next render pass: its load actions when the pass is on the same
    // attachments, saving the clear a pass of its own that stores what the next one loads.
    ClearPass wanted = clear;
    bool any = false;
    for (size_t index = 0; index < wanted.color.size(); ++index) {
        if (wanted.color[index].texture == nullptr || !wanted.clear_color[index]) {
            wanted.color[index] = {};
            wanted.clear_color[index] = false;
        } else {
            any = true;
        }
    }
    if (wanted.depth.texture == nullptr || !wanted.clear_depth) {
        wanted.depth = {};
        wanted.clear_depth = false;
    } else {
        any = true;
    }
    if (wanted.stencil.texture == nullptr || !wanted.clear_stencil) {
        wanted.stencil = {};
        wanted.clear_stencil = false;
    } else {
        any = true;
    }
    if (!any) {
        return;
    }
    // The draws and dispatches recorded so far come before it.
    if (impl->render != nil || impl->compute != nil) {
        impl->EndEncoder();
    }
    if (impl->pending_clear) {
        if (MergeClear(*impl->pending_clear, wanted)) {
            impl->HoldClear(*impl->pending_clear);
            return;
        }
        impl->FlushClear();
    }
    impl->pending_clear = wanted;
    impl->HoldClear(wanted);
}

bool Scheduler::ClearRectangle(const ClearRect& clear) {
    impl->FlushClear();
    const id<MTLTexture> color = clear.color_mask != 0 ? AsTexture(clear.color.texture) : nil;
    const id<MTLTexture> depth =
        clear.clear_depth || clear.clear_stencil ? AsTexture(clear.depth.texture) : nil;
    if (color == nil && depth == nil) {
        return false;
    }
    const MTLPixelFormat color_format = color != nil ? color.pixelFormat : MTLPixelFormatInvalid;
    const MTLPixelFormat depth_format = depth != nil ? depth.pixelFormat : MTLPixelFormatInvalid;
    const bool stencil = depth != nil && (depth_format == MTLPixelFormatStencil8 ||
                                          StencilViewFormat(depth_format) != MTLPixelFormatInvalid);
    const NSUInteger samples = color != nil ? color.sampleCount : depth.sampleCount;
    const id<MTLRenderPipelineState> pipeline =
        impl->ClearPipeline(color_format, clear.color_mask, depth_format, stencil, samples);
    if (pipeline == nil) {
        return false;
    }
    const u32 limit_w = std::clamp(clear.width, 1U, MaxRenderTargetSize);
    const u32 limit_h = std::clamp(clear.height, 1U, MaxRenderTargetSize);
    const u32 x = (std::min)(clear.rect[0], limit_w);
    const u32 y = (std::min)(clear.rect[1], limit_h);
    const u32 w = (std::min)(clear.rect[2], limit_w - x);
    const u32 h = (std::min)(clear.rect[3], limit_h - y);
    if (w == 0 || h == 0) {
        return true;
    }

    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    if (color != nil) {
        ApplyAttachment(pass.colorAttachments[0], clear.color);
        pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    }
    if (depth != nil && depth_format != MTLPixelFormatStencil8) {
        ApplyAttachment(pass.depthAttachment, clear.depth);
        pass.depthAttachment.loadAction = MTLLoadActionLoad;
        pass.depthAttachment.storeAction = MTLStoreActionStore;
    }
    if (stencil) {
        ApplyAttachment(pass.stencilAttachment, clear.depth);
        pass.stencilAttachment.loadAction = MTLLoadActionLoad;
        pass.stencilAttachment.storeAction = MTLStoreActionStore;
    }
    pass.renderTargetWidth = limit_w;
    pass.renderTargetHeight = limit_h;
    const id<MTLRenderCommandEncoder> encoder = impl->RenderEncoder(pass);
    if (encoder == nil) {
        return false;
    }
    [encoder setRenderPipelineState:pipeline];
    if (depth != nil) {
        [encoder setDepthStencilState:impl->ClearDepthState(clear.clear_depth,
                                                            stencil && clear.clear_stencil,
                                                            clear.stencil_mask)];
        [encoder setStencilReferenceValue:clear.stencil_value];
    }
    [encoder setViewport:MTLViewport{static_cast<f64>(x), static_cast<f64>(y), static_cast<f64>(w),
                                     static_cast<f64>(h), 0.0, 1.0}];
    [encoder setScissorRect:MTLScissorRect{x, y, w, h}];
    const f32 depth_value = std::clamp(clear.depth_value, 0.0f, 1.0f);
    [encoder setVertexBytes:&depth_value length:sizeof(depth_value) atIndex:0];
    if (color != nil) {
        // 16 bytes read as float4, uint4 or int4 by the format's class.
        std::array<u32, 4> bits{};
        const FormatClass format_class = ClassOf(static_cast<u32>(color_format));
        for (size_t component = 0; component < bits.size(); ++component) {
            const f64 value = clear.color_value[component];
            if (format_class == FormatClass::Uint) {
                bits[component] = static_cast<u32>(value);
            } else if (format_class == FormatClass::Sint) {
                bits[component] = static_cast<u32>(static_cast<s32>(value));
            } else {
                bits[component] = std::bit_cast<u32>(static_cast<f32>(value));
            }
        }
        [encoder setFragmentBytes:bits.data() length:sizeof(bits) atIndex:0];
    }
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
    return true;
}

bool Scheduler::Blit(const BlitPass& blit) {
    impl->FlushClear();
    const id<MTLTexture> src = AsTexture(blit.src);
    if (src == nil || blit.dst.texture == nullptr) {
        return false;
    }
    const MTLPixelFormat format = static_cast<MTLPixelFormat>(blit.dst_format);
    const id<MTLRenderPipelineState> pipeline = impl->BlitPipeline(format, blit.blend);
    if (pipeline == nil) {
        return false;
    }
    const auto [dx0, dy0, dx1, dy1] = blit.dst_rect;
    const auto [sx0, sy0, sx1, sy1] = blit.src_rect;
    const f32 width = static_cast<f32>(src.width);
    const f32 height = static_cast<f32>(src.height);
    // A mirrored destination becomes a mirrored source rectangle over a forward viewport.
    std::array<f32, 4> uv{sx0 / width, sy0 / height, sx1 / width, sy1 / height};
    if (dx1 < dx0) {
        std::swap(uv[0], uv[2]);
    }
    if (dy1 < dy0) {
        std::swap(uv[1], uv[3]);
    }
    const s32 x = (std::min)(dx0, dx1);
    const s32 y = (std::min)(dy0, dy1);
    const s32 w = std::abs(dx1 - dx0);
    const s32 h = std::abs(dy1 - dy0);
    if (w == 0 || h == 0) {
        return true;
    }

    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    const FormatClass format_class = ClassOf(blit.dst_format);
    const bool depth = format_class == FormatClass::Depth || format_class == FormatClass::DepthStencil;
    if (depth) {
        // Depth only: the fragment writes it; stencil keeps what it held or is cleared to 0.
        ApplyAttachment(pass.depthAttachment, blit.dst);
        pass.depthAttachment.loadAction = blit.keep_dst ? MTLLoadActionLoad : MTLLoadActionClear;
        pass.depthAttachment.storeAction = MTLStoreActionStore;
        if (format_class == FormatClass::DepthStencil) {
            ApplyAttachment(pass.stencilAttachment, blit.dst);
            pass.stencilAttachment.loadAction =
                blit.keep_dst ? MTLLoadActionLoad : MTLLoadActionClear;
            pass.stencilAttachment.storeAction = MTLStoreActionStore;
        }
    } else {
        ApplyAttachment(pass.colorAttachments[0], blit.dst);
        pass.colorAttachments[0].loadAction =
            blit.keep_dst ? MTLLoadActionLoad : MTLLoadActionClear;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    }
    // A depth-stencil source's stencil, read through a stencil view.
    id<MTLTexture> src_stencil = nil;
    if (format_class == FormatClass::DepthStencil &&
        StencilViewFormat(format) != MTLPixelFormatInvalid) {
        src_stencil = [src newTextureViewWithPixelFormat:StencilViewFormat(src.pixelFormat)];
        if (src_stencil == nil) {
            return false;
        }
    }
    const id<MTLRenderCommandEncoder> encoder = impl->RenderEncoder(pass);
    [encoder setRenderPipelineState:pipeline];
    if (depth) {
        [encoder setDepthStencilState:impl->DepthWriteAlways()];
        [encoder setFragmentTexture:src_stencil atIndex:1];
    }
    [encoder setViewport:MTLViewport{static_cast<f64>(x), static_cast<f64>(y),
                                     static_cast<f64>(w), static_cast<f64>(h), 0.0, 1.0}];
    const s32 limit_w = static_cast<s32>(blit.dst_width);
    const s32 limit_h = static_cast<s32>(blit.dst_height);
    const s32 sx = std::clamp(x, 0, limit_w);
    const s32 sy = std::clamp(y, 0, limit_h);
    const s32 sw = std::clamp(x + w, 0, limit_w) - sx;
    const s32 sh = std::clamp(y + h, 0, limit_h) - sy;
    if (sw <= 0 || sh <= 0) {
        [encoder endEncoding];
        return true;
    }
    [encoder setScissorRect:MTLScissorRect{static_cast<NSUInteger>(sx),
                                           static_cast<NSUInteger>(sy),
                                           static_cast<NSUInteger>(sw),
                                           static_cast<NSUInteger>(sh)}];
    [encoder setVertexBytes:uv.data() length:sizeof(uv) atIndex:0];
    [encoder setFragmentTexture:src atIndex:0];
    [encoder setFragmentSamplerState:blit.linear ? impl->linear : impl->nearest atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
    return true;
}

void Scheduler::BeginRendering(const RenderTargetSet& targets) {
    if (impl->render != nil && impl->render_targets == targets) {
        return;
    }
    // The waiting clear: this pass's load actions, or a pass of its own first.
    std::optional<ClearPass> clear;
    if (impl->pending_clear) {
        if (ClearFitsTargets(*impl->pending_clear, targets)) {
            clear = std::exchange(impl->pending_clear, std::nullopt);
            impl->pending_clear_textures.clear();
            ++impl->clears_folded;
        } else {
            impl->FlushClear();
        }
    }
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    for (size_t index = 0; index < targets.color.size(); ++index) {
        if (targets.color[index].texture == nullptr) {
            continue;
        }
        MTLRenderPassColorAttachmentDescriptor* const color = pass.colorAttachments[index];
        ApplyAttachment(color, targets.color[index]);
        color.loadAction = MTLLoadActionLoad;
        color.storeAction = MTLStoreActionStore;
        if (clear && clear->clear_color[index]) {
            const auto& value = clear->color_value[index];
            color.loadAction = MTLLoadActionClear;
            color.clearColor = MTLClearColorMake(value[0], value[1], value[2], value[3]);
        }
    }
    if (targets.has_depth) {
        ApplyAttachment(pass.depthAttachment, targets.depth);
        pass.depthAttachment.loadAction = MTLLoadActionLoad;
        pass.depthAttachment.storeAction = MTLStoreActionStore;
        if (clear && clear->clear_depth) {
            pass.depthAttachment.loadAction = MTLLoadActionClear;
            pass.depthAttachment.clearDepth = clear->depth_value;
        }
    }
    if (targets.has_stencil) {
        ApplyAttachment(pass.stencilAttachment, targets.depth);
        pass.stencilAttachment.loadAction = MTLLoadActionLoad;
        pass.stencilAttachment.storeAction = MTLStoreActionStore;
        if (clear && clear->clear_stencil) {
            pass.stencilAttachment.loadAction = MTLLoadActionClear;
            pass.stencilAttachment.clearStencil = clear->stencil_value;
        }
    }
    // Draws without attachments (occlusion queries) still need a size, within Metal's limit.
    pass.renderTargetWidth = std::clamp(targets.width, 1U, MaxRenderTargetSize);
    pass.renderTargetHeight = std::clamp(targets.height, 1U, MaxRenderTargetSize);
    const bool targetless = !targets.has_depth && !targets.has_stencil &&
                            std::ranges::all_of(targets.color, [](const Attachment& color) {
                                return color.texture == nullptr;
                            });
    if (targetless && PipelineCheckEnabled()) {
        // Without attachments nothing else tells Metal the pass has one sample per pixel, as
        // the pipelines drawn into it have (MoltenVK sets it for such a pass too).
        pass.defaultRasterSampleCount = 1;
    }
    pass.visibilityResultBuffer = impl->visibility;
    impl->render = impl->RenderEncoder(pass);
    impl->render_targets = targets;
    impl->visibility_slot = -1;
    impl->vertex_bound = {};
    impl->fragment_bound = {};
    impl->render_state = {};
    impl->render_state_valid = false;
    ++impl->render_pass_serial;
}

bool Scheduler::IsRendering() const noexcept {
    return impl->render != nil;
}

void Scheduler::EndRendering() {
    if (impl->render != nil) {
        impl->EndEncoder();
    }
}

void Scheduler::SetVisibility(std::optional<u32> slot) {
    const s64 wanted = slot ? static_cast<s64>(*slot) : -1;
    if (impl->render == nil || wanted == impl->visibility_slot) {
        return;
    }
    impl->visibility_slot = wanted;
    if (slot) {
        [impl->render setVisibilityResultMode:MTLVisibilityResultModeCounting
                                       offset:*slot * sizeof(u64)];
    } else {
        [impl->render setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
    }
}

BufferHandle Scheduler::VisibilityBuffer() const noexcept {
    return (__bridge void*)impl->visibility;
}

u64* Scheduler::VisibilityResults() noexcept {
    return static_cast<u64*>(impl->visibility.contents);
}

u64 Scheduler::RenderPassSerial() const noexcept {
    return impl->render_pass_serial;
}

void Scheduler::SetDrawState(const DrawState& state) {
    const id<MTLRenderCommandEncoder> encoder = impl->render;
    if (encoder == nil) {
        return;
    }
    // What the encoder has from the draw before stays as it is (EDEN_NX2_METAL_STATE_CACHE).
    const bool cached = StateCacheEnabled() && impl->render_state_valid;
    const DrawState& last = impl->render_state;
    const auto changed = [&](bool differs) {
        if (cached && !differs) {
            ++impl->state_skips;
            return false;
        }
        ++impl->state_sets;
        return true;
    };
    if (changed(last.pipeline != state.pipeline)) {
        [encoder setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)state.pipeline];
    }
    if (state.depth_stencil != nullptr && changed(last.depth_stencil != state.depth_stencil)) {
        [encoder setDepthStencilState:(__bridge id<MTLDepthStencilState>)state.depth_stencil];
    }
    const auto to_viewport = [](const Viewport& viewport) {
        return MTLViewport{viewport.x,      viewport.y,     viewport.width,
                           viewport.height, viewport.znear, viewport.zfar};
    };
    // Metal rejects scissors past the attachments.
    const u32 target_width = std::clamp(impl->render_targets.width, 1U, MaxRenderTargetSize);
    const u32 target_height = std::clamp(impl->render_targets.height, 1U, MaxRenderTargetSize);
    const auto to_scissor = [&](const Scissor& scissor) {
        const u32 x = (std::min)(scissor.x, target_width - 1);
        const u32 y = (std::min)(scissor.y, target_height - 1);
        const u32 width = (std::min)(scissor.width, target_width - x);
        const u32 height = (std::min)(scissor.height, target_height - y);
        return MTLScissorRect{x, y, (std::max)(width, 1U), (std::max)(height, 1U)};
    };
    if (state.viewport_count > 1) {
        // A program writes the viewport index: all of them.
        const size_t count = (std::min)(size_t{state.viewport_count}, size_t{16});
        if (changed(last.viewport_count != state.viewport_count ||
                    last.viewport != state.viewport ||
                    last.more_viewports != state.more_viewports)) {
            std::array<MTLViewport, 16> viewports{};
            viewports[0] = to_viewport(state.viewport);
            for (size_t index = 1; index < count; ++index) {
                viewports[index] = to_viewport(state.more_viewports[index - 1]);
            }
            [encoder setViewports:viewports.data() count:count];
        }
        if (changed(last.viewport_count != state.viewport_count ||
                    last.scissor != state.scissor || last.more_scissors != state.more_scissors)) {
            std::array<MTLScissorRect, 16> scissors{};
            scissors[0] = to_scissor(state.scissor);
            for (size_t index = 1; index < count; ++index) {
                scissors[index] = to_scissor(state.more_scissors[index - 1]);
            }
            [encoder setScissorRects:scissors.data() count:count];
        }
    } else {
        if (changed(last.viewport_count != 1 || last.viewport != state.viewport)) {
            [encoder setViewport:to_viewport(state.viewport)];
        }
        if (changed(last.viewport_count != 1 || last.scissor != state.scissor)) {
            [encoder setScissorRect:to_scissor(state.scissor)];
        }
    }
    if (changed(last.fill_lines != state.fill_lines)) {
        [encoder setTriangleFillMode:state.fill_lines ? MTLTriangleFillModeLines
                                                      : MTLTriangleFillModeFill];
    }
    if (changed(last.cull_mode != state.cull_mode)) {
        [encoder setCullMode:static_cast<MTLCullMode>(state.cull_mode)];
    }
    if (changed(last.front_ccw != state.front_ccw)) {
        [encoder setFrontFacingWinding:state.front_ccw ? MTLWindingCounterClockwise
                                                       : MTLWindingClockwise];
    }
    if (changed(last.depth_clamp != state.depth_clamp)) {
        [encoder setDepthClipMode:state.depth_clamp ? MTLDepthClipModeClamp
                                                    : MTLDepthClipModeClip];
    }
    if (changed(last.depth_bias != state.depth_bias || last.slope_scale != state.slope_scale ||
                last.bias_clamp != state.bias_clamp)) {
        [encoder setDepthBias:state.depth_bias slopeScale:state.slope_scale clamp:state.bias_clamp];
    }
    if (changed(last.stencil_front_ref != state.stencil_front_ref ||
                last.stencil_back_ref != state.stencil_back_ref)) {
        [encoder setStencilFrontReferenceValue:state.stencil_front_ref
                            backReferenceValue:state.stencil_back_ref];
    }
    if (changed(last.blend_color != state.blend_color)) {
        [encoder setBlendColorRed:state.blend_color[0]
                            green:state.blend_color[1]
                             blue:state.blend_color[2]
                            alpha:state.blend_color[3]];
    }
    // A draw without a depth-stencil state leaves the encoder's.
    void* const depth_stencil =
        state.depth_stencil != nullptr ? state.depth_stencil : last.depth_stencil;
    impl->render_state = state;
    impl->render_state.depth_stencil = depth_stencil;
    impl->render_state_valid = true;
}

void Scheduler::BindVertexBuffer(u32 slot, BufferHandle buffer, u32 offset) {
    if (SlotFits(slot, BufferSlots, "vertex buffer")) {
        impl->SetBuffer(ShaderStage::Vertex, slot, buffer, offset);
    }
}

void Scheduler::BindBuffer(ShaderStage stage, u32 slot, BufferHandle buffer, u32 offset) {
    if (SlotFits(slot, BufferSlots, "buffer")) {
        impl->SetBuffer(stage, slot, buffer, offset);
    }
}

void Scheduler::BindBytes(ShaderStage stage, u32 slot, const void* data, u32 size) {
    if (!SlotFits(slot, BufferSlots, "buffer")) {
        return;
    }
    ++impl->state_sets;
    switch (stage) {
    case ShaderStage::Vertex:
        if (slot < BufferSlots) {
            impl->vertex_bound.buffers[slot] = &InlineBytes;
        }
        [impl->render setVertexBytes:data length:size atIndex:slot];
        break;
    case ShaderStage::Fragment:
        if (slot < BufferSlots) {
            impl->fragment_bound.buffers[slot] = &InlineBytes;
        }
        [impl->render setFragmentBytes:data length:size atIndex:slot];
        break;
    case ShaderStage::Compute:
        if (impl->active_compute == nil) {
            return;
        }
        if (slot < BufferSlots) {
            impl->active_bound->buffers[slot] = &InlineBytes;
        }
        [impl->active_compute setBytes:data length:size atIndex:slot];
        break;
    }
}

void Scheduler::BindTexture(ShaderStage stage, u32 slot, TextureHandle texture) {
    if (!SlotFits(slot, TextureSlots, "texture")) {
        return;
    }
    switch (stage) {
    case ShaderStage::Vertex:
        if (impl->render != nil && impl->NoteObject(impl->vertex_bound.textures, slot, texture)) {
            [impl->render setVertexTexture:AsTexture(texture) atIndex:slot];
        }
        break;
    case ShaderStage::Fragment:
        if (impl->render != nil &&
            impl->NoteObject(impl->fragment_bound.textures, slot, texture)) {
            [impl->render setFragmentTexture:AsTexture(texture) atIndex:slot];
        }
        break;
    case ShaderStage::Compute:
        if (impl->active_compute != nil &&
            impl->NoteObject(impl->active_bound->textures, slot, texture)) {
            [impl->active_compute setTexture:AsTexture(texture) atIndex:slot];
        }
        break;
    }
}

void Scheduler::BindSampler(ShaderStage stage, u32 slot, void* sampler) {
    if (!SlotFits(slot, SamplerSlots, "sampler")) {
        return;
    }
    switch (stage) {
    case ShaderStage::Vertex:
        if (impl->render != nil && impl->NoteObject(impl->vertex_bound.samplers, slot, sampler)) {
            [impl->render setVertexSamplerState:AsSampler(sampler) atIndex:slot];
        }
        break;
    case ShaderStage::Fragment:
        if (impl->render != nil &&
            impl->NoteObject(impl->fragment_bound.samplers, slot, sampler)) {
            [impl->render setFragmentSamplerState:AsSampler(sampler) atIndex:slot];
        }
        break;
    case ShaderStage::Compute:
        if (impl->active_compute != nil &&
            impl->NoteObject(impl->active_bound->samplers, slot, sampler)) {
            [impl->active_compute setSamplerState:AsSampler(sampler) atIndex:slot];
        }
        break;
    }
}

void Scheduler::DrawIndirect(u32 primitive, BufferHandle buffer, u64 offset) {
    if (impl->render == nil || buffer == nullptr) {
        return;
    }
    [impl->render drawPrimitives:static_cast<MTLPrimitiveType>(primitive)
                  indirectBuffer:AsBuffer(buffer)
            indirectBufferOffset:offset];
    impl->CountDraw();
}

void Scheduler::DrawIndexedIndirect(u32 primitive, u32 index_size, BufferHandle index_buffer,
                                    u64 index_offset, BufferHandle buffer, u64 offset) {
    if (impl->render == nil || buffer == nullptr || index_buffer == nullptr) {
        return;
    }
    [impl->render drawIndexedPrimitives:static_cast<MTLPrimitiveType>(primitive)
                              indexType:index_size == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
                            indexBuffer:AsBuffer(index_buffer)
                      indexBufferOffset:index_offset
                         indirectBuffer:AsBuffer(buffer)
                   indirectBufferOffset:offset];
    impl->CountDraw();
}

void Scheduler::BeginCompute(void* pipeline, bool ahead) {
    const void** bound_pipeline = &impl->compute_pipeline;
    if (ahead && AheadEnabled()) {
        impl->active_compute = impl->KernelEncoder();
        impl->active_bound = &impl->kernel_bound;
        bound_pipeline = &impl->kernel_pipeline;
        ++impl->ahead_kernels;
    } else {
        impl->FlushClear();
        if (impl->compute == nil) {
            impl->EndEncoder();
            impl->compute = [impl->Commands() computeCommandEncoder];
            impl->compute.label = @"Compute";
            impl->compute_bound = {};
            impl->compute_pipeline = nullptr;
            ++impl->encoders;
            ++impl->segment_encoders;
        }
        impl->active_compute = impl->compute;
        impl->active_bound = &impl->compute_bound;
    }
    if (StateCacheEnabled() && *bound_pipeline == pipeline) {
        ++impl->state_skips;
        return;
    }
    *bound_pipeline = pipeline;
    ++impl->state_sets;
    [impl->active_compute setComputePipelineState:(__bridge id<MTLComputePipelineState>)pipeline];
}

void Scheduler::Dispatch(std::array<u32, 3> groups, std::array<u32, 3> group_size) {
    const id<MTLComputeCommandEncoder> encoder = impl->active_compute;
    if (encoder == nil || groups[0] == 0 || groups[1] == 0 || groups[2] == 0) {
        return;
    }
    [encoder dispatchThreadgroups:MTLSizeMake(groups[0], groups[1], groups[2])
            threadsPerThreadgroup:MTLSizeMake(group_size[0], group_size[1], group_size[2])];
    impl->CountDispatch();
}

void Scheduler::DispatchIndirect(BufferHandle buffer, u64 offset, std::array<u32, 3> group_size) {
    const id<MTLComputeCommandEncoder> encoder = impl->active_compute;
    if (encoder == nil || buffer == nullptr) {
        return;
    }
    [encoder dispatchThreadgroupsWithIndirectBuffer:AsBuffer(buffer)
                               indirectBufferOffset:offset
                              threadsPerThreadgroup:MTLSizeMake(group_size[0], group_size[1],
                                                                group_size[2])];
    impl->CountDispatch();
}

void Scheduler::DispatchStageIn(u32 first, u32 count, u32 first_instance, u32 instances) {
    const id<MTLComputeCommandEncoder> encoder = impl->active_compute;
    if (encoder == nil || count == 0 || instances == 0) {
        return;
    }
    [encoder setStageInRegion:MTLRegionMake2D(first, first_instance, count, instances)];
    [encoder dispatchThreads:MTLSizeMake(count, instances, 1)
       threadsPerThreadgroup:MTLSizeMake((std::min)(count, 64U), 1, 1)];
    impl->CountDispatch();
}

void Scheduler::SetTessellationFactorBuffer(BufferHandle buffer, u64 offset) {
    if (impl->render == nil || buffer == nullptr) {
        return;
    }
    [impl->render setTessellationFactorBuffer:AsBuffer(buffer) offset:offset instanceStride:0];
}

void Scheduler::DrawPatches(u32 control_points, u32 patch_count) {
    if (impl->render == nil || patch_count == 0) {
        return;
    }
    [impl->render drawPatches:control_points
                   patchStart:0
                   patchCount:patch_count
             patchIndexBuffer:nil
       patchIndexBufferOffset:0
                instanceCount:1
                 baseInstance:0];
    impl->CountDraw();
}

void Scheduler::Draw(u32 primitive, u32 first_vertex, u32 vertex_count, u32 instance_count,
                     u32 base_instance) {
    if (impl->render == nil || vertex_count == 0 || instance_count == 0) {
        return;
    }
    [impl->render drawPrimitives:static_cast<MTLPrimitiveType>(primitive)
                     vertexStart:first_vertex
                     vertexCount:vertex_count
                   instanceCount:instance_count
                    baseInstance:base_instance];
    impl->CountDraw();
}

void Scheduler::DrawIndexed(u32 primitive, u32 index_count, u32 index_size,
                            BufferHandle index_buffer, u64 index_offset, u32 instance_count,
                            s32 base_vertex, u32 base_instance) {
    if (impl->render == nil || index_count == 0 || instance_count == 0 || index_buffer == nullptr) {
        return;
    }
    [impl->render drawIndexedPrimitives:static_cast<MTLPrimitiveType>(primitive)
                             indexCount:index_count
                              indexType:index_size == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
                            indexBuffer:AsBuffer(index_buffer)
                      indexBufferOffset:index_offset
                          instanceCount:instance_count
                             baseVertex:base_vertex
                           baseInstance:base_instance];
    impl->CountDraw();
}

void* Scheduler::CommandBuffer() {
    impl->FlushClear();
    impl->EndEncoder();
    // What the caller encodes is not known here: nothing goes ahead of it by its targets.
    ++impl->segment_encoders;
    return (__bridge void*)impl->Commands();
}

u64 Scheduler::TakeGpuNanoseconds() noexcept {
    return impl->gpu_ns.exchange(0, std::memory_order_relaxed);
}

void Scheduler::NotePresent(bool guest_picture) {
    if (!guest_picture) {
        if (impl->blank_presents++ == 0) {
            LOG_INFO(Render_Metal, "Metal: first drawable presented, without a guest picture");
        }
        return;
    }
    if (impl->presents++ == 0) {
        LOG_INFO(Render_Metal, "Metal: first guest frame presented ({} command buffers, {} draws, "
                               "{} dispatches before it)",
                 impl->committed, impl->draws, impl->dispatches);
    }
}

std::array<u64, 11> Scheduler::TakeCounts() noexcept {
    return {std::exchange(impl->encoders, 0),      std::exchange(impl->command_buffers, 0),
            std::exchange(impl->blit_commands, 0), std::exchange(impl->render_passes, 0),
            std::exchange(impl->clears_folded, 0), std::exchange(impl->state_sets, 0),
            std::exchange(impl->state_skips, 0),   std::exchange(impl->splits, 0),
            std::exchange(impl->ahead_copies, 0),  std::exchange(impl->ahead_kernels, 0),
            std::exchange(impl->idle_fences, 0)};
}

} // namespace Metal
