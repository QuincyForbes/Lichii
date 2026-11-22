// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// EDEN_NX2_METAL_RECORDER: the scheduler's Metal encoding moved off the recording thread.
// Command buffers and encoders are proxies: their methods append compact commands (POD
// arguments, retained objects, copied bytes) to a stream that an encoder thread replays into the
// real MTLCommandBuffer and encoders, committing them in recording order. A method a proxy does
// not record is a sync point: the stream is drained (the encoder thread catches up) and the
// message goes to the real object on the calling thread (forwardingTargetForSelector:).
//
// Objective-C++ with ARC; included by metal_scheduler.mm only.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <objc/runtime.h>
#include <pthread.h>
#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"

namespace Metal::Rec {

enum class Op : u16 {
    // Command buffers (target: the command buffer proxy unless noted)
    CbNew,       ///< target +1 (released by CbCommit); obj: descriptor or null
    CbRender,    ///< target: new encoder +1 (released by EndEncoding); obj: cb; obj2: descriptor
    CbCompute,   ///< target: new encoder +1; obj: cb
    CbBlit,      ///< target: new encoder +1; obj: cb
    CbCompleted, ///< obj: the handler block
    CbSignalEvent,
    CbPresent,
    CbPresentAfter,
    CbCommit,
    Call,        ///< a function on the encoder thread, in stream order
    // Any proxy
    SetLabel,
    EndEncoding,
    // Render encoders
    RPipeline,
    RDepthStencil,
    RViewport,
    RViewports,
    RScissor,
    RScissors,
    RCull,
    RWinding,
    RDepthClip,
    RFill,
    RDepthBias,
    RStencilRef,
    RStencilRef2,
    RBlendColor,
    RDepthBounds,
    RVisibility,
    RColorStore,
    RDepthStore,
    RStencilStore,
    RBuffer,       ///< stage in Cmd::stage: 0 vertex, 1 fragment, 2 object, 3 mesh
    RBufferOffset,
    RBytes,
    RTexture,
    RSampler,
    RUseResource,
    RTessFactors,
    RDraw,
    RDrawInstanced,
    RDrawIndirect,
    RDrawIndexed,
    RDrawIndexedIndirect,
    RDrawPatches,
    RDrawMesh,
    RDrawMeshIndirect,
    // Compute encoders
    CPipeline,
    CBuffer,
    CBufferOffset,
    CBytes,
    CTexture,
    CSampler,
    CUseResource,
    CMemoryBarrier,
    CStageIn,
    CDispatch,
    CDispatchIndirect,
    CDispatchThreads,
    // Blit encoders
    BCopyBufferToBuffer,
    BCopyBufferToTexture,
    BCopyTextureToBuffer,
    BCopyTextureToTexture,
    BFill,
};

/// One command. obj and obj2 are held once per unique object in Chunk::kept until the chunk finishes.
struct Cmd {
    Op op;
    u8 stage;
    u8 pad;
    u32 payload; ///< Offset of copied bytes in the chunk's byte arena
    void* target;
    void* obj;
    void* obj2;
    alignas(8) u8 inl[48];
};
static_assert(sizeof(Cmd) == 80);

struct Chunk {
    std::vector<Cmd> cmds;
    std::vector<u8> bytes;
    std::array<void*, 128> keep_set{};
    std::vector<void*> kept;
};

template <typename T>
inline void Put(Cmd& cmd, const T& value) {
    static_assert(sizeof(T) <= sizeof(Cmd::inl) && std::is_trivially_copyable_v<T>);
    std::memcpy(cmd.inl, &value, sizeof(T));
}

template <typename T>
inline T Get(const Cmd& cmd) {
    T value;
    std::memcpy(&value, cmd.inl, sizeof(T));
    return value;
}

struct BufferArgs {
    u64 offset;
    u32 index;
};
struct BytesArgs {
    u32 length;
    u32 index;
};
struct IndexArgs {
    u32 index;
};
struct U32Args {
    u32 a;
    u32 b;
};
struct F32Args {
    f32 v[4];
};
struct VisibilityArgs {
    u64 offset;
    u32 mode;
};
struct UseArgs {
    u32 usage;
    u32 stages;
};
struct TessArgs {
    u64 offset;
    u64 stride;
};
struct DrawArgs {
    u32 type;
    u32 start;
    u32 count;
    u32 instances;
    u32 base_instance;
};
struct DrawIndirectArgs {
    u64 offset;
    u32 type;
};
struct DrawIndexedArgs {
    u64 index_offset;
    u32 type;
    u32 count;
    u32 index_type;
    u32 instances;
    s64 base_vertex;
    u32 base_instance;
};
struct DrawIndexedIndirectArgs {
    u64 index_offset;
    u64 indirect_offset;
    u32 type;
    u32 index_type;
};
struct PatchArgs {
    u64 index_offset;
    u32 control_points;
    u32 start;
    u32 count;
    u32 instances;
    u32 base_instance;
};
struct Size3 {
    u32 x;
    u32 y;
    u32 z;
};
inline Size3 ToSize3(MTLSize size) {
    return {static_cast<u32>(size.width), static_cast<u32>(size.height),
            static_cast<u32>(size.depth)};
}
inline Size3 ToSize3(MTLOrigin origin) {
    return {static_cast<u32>(origin.x), static_cast<u32>(origin.y), static_cast<u32>(origin.z)};
}
inline MTLSize ToSize(Size3 v) {
    return MTLSizeMake(v.x, v.y, v.z);
}
inline MTLOrigin ToOrigin(Size3 v) {
    return MTLOriginMake(v.x, v.y, v.z);
}
struct MeshArgs {
    Size3 groups;
    Size3 object_threads;
    Size3 mesh_threads;
};
struct MeshIndirectArgs {
    u64 offset;
    Size3 object_threads;
    Size3 mesh_threads;
};
struct DispatchArgs {
    Size3 groups;
    Size3 threads;
};
struct DispatchIndirectArgs {
    u64 offset;
    Size3 threads;
};
struct CopyBBArgs {
    u64 src_offset;
    u64 dst_offset;
    u64 size;
};
struct CopyBTArgs {
    u64 src_offset;
    u32 bytes_per_row;
    u32 bytes_per_image;
    Size3 size;
    u32 slice;
    u32 level;
    Size3 origin;
};
struct CopyTBArgs {
    u64 dst_offset;
    u32 bytes_per_row;
    u32 bytes_per_image;
    u32 slice;
    u32 level;
    Size3 origin;
    Size3 size;
};
struct CopyTTArgs { ///< In the byte arena (52 bytes)
    u32 src_slice;
    u32 src_level;
    Size3 src_origin;
    Size3 size;
    u32 dst_slice;
    u32 dst_level;
    Size3 dst_origin;
};
struct FillArgs {
    u64 location;
    u64 length;
    u32 value;
};
using CallFn = void (*)(void* context, u64 value);
struct CallArgs {
    CallFn fn;
    void* context;
    u64 value;
};

class Recorder;

} // namespace Metal::Rec

/// Base of the proxies: the recorder, and the real object once the encoder thread made it.
@interface LichiiRecProxy : NSObject {
@public
    Metal::Rec::Recorder* rec;
    id real;
}
- (void)setLabel:(NSString*)label;
- (void)endEncoding;
@end

@interface LichiiRecCommandBuffer : LichiiRecProxy
@end
@interface LichiiRecRenderEncoder : LichiiRecProxy
@end
@interface LichiiRecComputeEncoder : LichiiRecProxy
@end
@interface LichiiRecBlitEncoder : LichiiRecProxy
@end

namespace Metal::Rec {

class Recorder {
public:
    static constexpr size_t ChunkCommands = 256;
    /// Published chunks the encoder thread may lag behind before recording waits.
    static constexpr u64 MaxQueuedChunks = 256;

    explicit Recorder(id<MTLCommandQueue> queue_) : queue{queue_} {
        current = NewChunk();
        last_take = std::chrono::steady_clock::now();
        thread = std::thread([this] { Run(); });
    }

    ~Recorder() {
        Publish();
        {
            std::scoped_lock lock{mutex};
            stopping = true;
        }
        work_cv.notify_all();
        thread.join();
        ReleaseKept(*current);
        delete current;
        for (Chunk* chunk : free_chunks) {
            ReleaseKept(*chunk);
            delete chunk;
        }
    }

    Recorder(const Recorder&) = delete;
    Recorder& operator=(const Recorder&) = delete;

    /// A new command; publishes the chunk first when it is full (never between a command and
    /// its bytes).
    Cmd& Push(Op op, void* target) {
        if (current->cmds.size() >= ChunkCommands) {
            Publish();
        }
        Cmd& cmd = current->cmds.emplace_back();
        cmd.op = op;
        cmd.target = target;
        return cmd;
    }

    /// Copies bytes into the chunk of the command just pushed; returns their offset.
    u32 Bytes(const void* data, size_t size) {
        std::vector<u8>& bytes = current->bytes;
        const size_t at = (bytes.size() + 15) & ~size_t{15};
        bytes.resize(at + size);
        if (size != 0) {
            std::memcpy(bytes.data() + at, data, size);
        }
        return static_cast<u32>(at);
    }

    /// Retains `object` at most once per chunk, avoiding per-bind atomic ARC retain/release bounces.
    void* Keep(__unsafe_unretained id object) {
        if (object == nil) {
            return nullptr;
        }
        void* const raw = (__bridge void*)object;
        const size_t slot =
            ((reinterpret_cast<uintptr_t>(raw) >> 4) ^ (reinterpret_cast<uintptr_t>(raw) >> 12)) &
            (current->keep_set.size() - 1);
        if (current->keep_set[slot] == raw) {
            return raw;
        }
        current->keep_set[slot] = raw;
        current->kept.push_back((__bridge_retained void*)object);
        return raw;
    }

    /// A command buffer proxy; the real one is made by the encoder thread.
    id<MTLCommandBuffer> NewCommandBuffer(MTLCommandBufferDescriptor* descriptor) {
        LichiiRecCommandBuffer* const proxy = [[LichiiRecCommandBuffer alloc] init];
        proxy->rec = this;
        Cmd& cmd = Push(Op::CbNew, (__bridge_retained void*)proxy);
        cmd.obj = Keep(descriptor);
        return (id<MTLCommandBuffer>)proxy;
    }

    /// `fn(context, value)` on the encoder thread after the commands recorded so far.
    void Call(CallFn fn, void* context, u64 value) {
        Put(Push(Op::Call, nullptr), CallArgs{fn, context, value});
    }

    /// Hands the commands recorded so far to the encoder thread.
    void Publish() {
        if (current->cmds.empty()) {
            return;
        }
        recorded.fetch_add(current->cmds.size(), std::memory_order_relaxed);
        Chunk* next = nullptr;
        bool wake = false;
        bool backlog = false;
        {
            std::scoped_lock lock{mutex};
            queue_chunks.push_back(current);
            ++published;
            wake = sleeping;
            backlog = published - consumed > MaxQueuedChunks;
            if (!free_chunks.empty()) {
                next = free_chunks.back();
                free_chunks.pop_back();
            }
        }
        if (wake) {
            work_cv.notify_one();
        }
        current = next != nullptr ? next : NewChunk();
        if (backlog) {
            // The encoder thread is far behind (blocked on the queue's command buffer limit, or
            // slower than recording): wait for half the backlog, as encoding inline would.
            NoteDrain("backlog");
            std::unique_lock lock{mutex};
            const u64 target = published - MaxQueuedChunks / 2;
            ++drain_waiting;
            done_cv.wait(lock, [&] { return consumed >= target; });
            --drain_waiting;
        }
    }

    /// Sync point: returns once the encoder thread replayed everything recorded so far.
    void Drain(const char* reason) {
        NoteDrain(reason);
        Publish();
        std::unique_lock lock{mutex};
        const u64 target = published;
        if (consumed >= target) {
            return;
        }
        ++drain_waiting;
        done_cv.wait(lock, [&] { return consumed >= target; });
        --drain_waiting;
    }

    /// The real object behind a proxy, after a Drain (or the object itself when not a proxy).
    static id RealOf(id object) {
        if ([object isKindOfClass:[LichiiRecProxy class]]) {
            return ((LichiiRecProxy*)object)->real;
        }
        return object;
    }

    /// Since the last call, per frame: commands recorded and replayed, the encoder thread's busy
    /// share of the wall time, and drains by reason.
    std::string TakeLine(double frames) {
        const double per = frames > 0.0 ? frames : 1.0;
        const auto now = std::chrono::steady_clock::now();
        const double wall_ns = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - last_take).count());
        last_take = now;
        const double busy = static_cast<double>(busy_ns.exchange(0, std::memory_order_relaxed));
        std::string line = fmt::format(
            " (recorder: {:.0f} commands recorded/frame, {:.0f} replayed/frame, encoder thread "
            "{:.1f}% busy, {:.1f} chunks/frame, drains {:.2f}/frame",
            static_cast<double>(recorded.exchange(0, std::memory_order_relaxed)) / per,
            static_cast<double>(replayed.exchange(0, std::memory_order_relaxed)) / per,
            wall_ns > 0.0 ? busy * 100.0 / wall_ns : 0.0,
            static_cast<double>(chunks_replayed.exchange(0, std::memory_order_relaxed)) / per,
            static_cast<double>(drains.exchange(0, std::memory_order_relaxed)) / per);
        std::scoped_lock lock{reason_mutex};
        bool first = true;
        for (auto& [reason, count] : drain_reasons) {
            if (count == 0) {
                continue;
            }
            line += fmt::format("{}{}={:.2f}", first ? ": " : ", ", reason,
                                static_cast<double>(count) / per);
            first = false;
            count = 0;
        }
        line += ")";
        return line;
    }

    void NoteRefusedPass() {
        if (refused_passes++ < 4) {
            LOG_ERROR(Render_Metal, "Metal recorder: render pass refused on replay");
        }
    }

private:
    static Chunk* NewChunk() {
        Chunk* const chunk = new Chunk;
        chunk->cmds.reserve(ChunkCommands + 1);
        chunk->bytes.reserve(4096);
        chunk->kept.reserve(64);
        return chunk;
    }

    static void ReleaseKept(Chunk& chunk) {
        for (void* kept_obj : chunk.kept) {
            CFRelease(kept_obj);
        }
        chunk.kept.clear();
        chunk.keep_set.fill(nullptr);
    }

    void NoteDrain(const char* reason) {
        drains.fetch_add(1, std::memory_order_relaxed);
        std::scoped_lock lock{reason_mutex};
        for (auto& [known, count] : drain_reasons) {
            if (known == reason || std::strcmp(known, reason) == 0) {
                ++count;
                return;
            }
        }
        if (drain_reasons.size() < 32) {
            LOG_INFO(Render_Metal, "Metal recorder: first sync point for {}", reason);
            drain_reasons.emplace_back(reason, 1);
        }
    }

    void Run() {
        pthread_setname_np("MetalEncoder");
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        for (;;) {
            Chunk* chunk = nullptr;
            {
                std::unique_lock lock{mutex};
                while (queue_chunks.empty() && !stopping) {
                    sleeping = true;
                    work_cv.wait(lock);
                    sleeping = false;
                }
                if (queue_chunks.empty()) {
                    return;
                }
                chunk = queue_chunks.front();
                queue_chunks.pop_front();
            }
            const auto start = std::chrono::steady_clock::now();
            @autoreleasepool {
                for (const Cmd& cmd : chunk->cmds) {
                    Execute(cmd, *chunk);
                }
                ReleaseKept(*chunk);
            }
            busy_ns.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                   std::chrono::steady_clock::now() - start)
                                                   .count()),
                              std::memory_order_relaxed);
            replayed.fetch_add(chunk->cmds.size(), std::memory_order_relaxed);
            chunks_replayed.fetch_add(1, std::memory_order_relaxed);
            chunk->cmds.clear();
            chunk->bytes.clear();
            bool notify = false;
            {
                std::scoped_lock lock{mutex};
                ++consumed;
                notify = drain_waiting != 0;
                if (free_chunks.size() < 64) {
                    free_chunks.push_back(chunk);
                    chunk = nullptr;
                }
            }
            delete chunk;
            if (notify) {
                done_cv.notify_all();
            }
        }
    }

    void Execute(const Cmd& c, const Chunk& chunk);

    id<MTLCommandQueue> queue;
    Chunk* current{};
    std::thread thread;
    std::mutex mutex;
    std::condition_variable work_cv;
    std::condition_variable done_cv;
    std::deque<Chunk*> queue_chunks;
    std::vector<Chunk*> free_chunks;
    u64 published{};
    u64 consumed{};
    u32 drain_waiting{};
    bool sleeping{};
    bool stopping{};
    std::atomic<u64> recorded{};
    std::atomic<u64> replayed{};
    std::atomic<u64> chunks_replayed{};
    std::atomic<u64> busy_ns{};
    std::atomic<u64> drains{};
    std::mutex reason_mutex;
    std::vector<std::pair<const char*, u64>> drain_reasons;
    std::chrono::steady_clock::time_point last_take;
    u32 refused_passes{};
};

} // namespace Metal::Rec

// Recording helpers of the proxies' methods.
#define LICHII_REC(op_name)                                                                        \
    Metal::Rec::Cmd& cmd = rec->Push(Metal::Rec::Op::op_name, (__bridge void*)self)
#define LICHII_KEEP(field, object) cmd.field = rec->Keep(object)

@implementation LichiiRecProxy

- (void)setLabel:(NSString*)label {
    LICHII_REC(SetLabel);
    LICHII_KEEP(obj, label);
}

- (void)endEncoding {
    (void)rec->Push(Metal::Rec::Op::EndEncoding, (__bridge void*)self);
}

// Anything not recorded: drain, then the real object takes the message on this thread.
- (id)forwardingTargetForSelector:(SEL)selector {
    rec->Drain(sel_getName(selector));
    return real;
}

// Only reached when the real object is nil (Metal refused it): the message does nothing.
- (NSMethodSignature*)methodSignatureForSelector:(SEL)selector {
    NSMethodSignature* const signature = [super methodSignatureForSelector:selector];
    if (signature != nil) {
        return signature;
    }
    Protocol* const __unsafe_unretained protocols[] = {@protocol(MTLRenderCommandEncoder),
                                   @protocol(MTLComputeCommandEncoder),
                                   @protocol(MTLBlitCommandEncoder), @protocol(MTLCommandBuffer)};
    for (Protocol* const __unsafe_unretained protocol : protocols) {
        for (const BOOL required : {YES, NO}) {
            const objc_method_description description =
                protocol_getMethodDescription(protocol, selector, required, YES);
            if (description.types != nullptr) {
                return [NSMethodSignature signatureWithObjCTypes:description.types];
            }
        }
    }
    return nil;
}

- (void)forwardInvocation:(NSInvocation*)invocation {
    if (real != nil) {
        [invocation invokeWithTarget:real];
        return;
    }
    const NSUInteger length = invocation.methodSignature.methodReturnLength;
    if (length != 0) {
        std::vector<u8> zero(length);
        [invocation setReturnValue:zero.data()];
    }
}

@end

@implementation LichiiRecCommandBuffer

- (id<MTLRenderCommandEncoder>)renderCommandEncoderWithDescriptor:
    (MTLRenderPassDescriptor*)descriptor {
    LichiiRecRenderEncoder* const encoder = [[LichiiRecRenderEncoder alloc] init];
    encoder->rec = rec;
    // A copy: the caller may change the descriptor after this (pass timing does).
    MTLRenderPassDescriptor* const copy = [descriptor copy];
    Metal::Rec::Cmd& cmd = rec->Push(Metal::Rec::Op::CbRender, (__bridge_retained void*)encoder);
    LICHII_KEEP(obj, self);
    LICHII_KEEP(obj2, copy);
    return (id<MTLRenderCommandEncoder>)encoder;
}

- (id<MTLComputeCommandEncoder>)computeCommandEncoder {
    return [self computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
}

- (id<MTLComputeCommandEncoder>)computeCommandEncoderWithDispatchType:(MTLDispatchType)dispatchType {
    LichiiRecComputeEncoder* const encoder = [[LichiiRecComputeEncoder alloc] init];
    encoder->rec = rec;
    Metal::Rec::Cmd& cmd = rec->Push(Metal::Rec::Op::CbCompute, (__bridge_retained void*)encoder);
    Metal::Rec::Put(cmd, static_cast<uint32_t>(dispatchType));
    LICHII_KEEP(obj, self);
    return (id<MTLComputeCommandEncoder>)encoder;
}

- (id<MTLBlitCommandEncoder>)blitCommandEncoder {
    LichiiRecBlitEncoder* const encoder = [[LichiiRecBlitEncoder alloc] init];
    encoder->rec = rec;
    Metal::Rec::Cmd& cmd = rec->Push(Metal::Rec::Op::CbBlit, (__bridge_retained void*)encoder);
    LICHII_KEEP(obj, self);
    return (id<MTLBlitCommandEncoder>)encoder;
}

- (void)addCompletedHandler:(MTLCommandBufferHandler)block {
    LICHII_REC(CbCompleted);
    LICHII_KEEP(obj, [block copy]);
}

- (void)encodeSignalEvent:(id<MTLEvent>)event value:(uint64_t)value {
    LICHII_REC(CbSignalEvent);
    LICHII_KEEP(obj, event);
    Metal::Rec::Put(cmd, value);
}

- (void)presentDrawable:(id<MTLDrawable>)drawable {
    LICHII_REC(CbPresent);
    LICHII_KEEP(obj, drawable);
}

- (void)presentDrawable:(id<MTLDrawable>)drawable afterMinimumDuration:(CFTimeInterval)duration {
    LICHII_REC(CbPresentAfter);
    LICHII_KEEP(obj, drawable);
    Metal::Rec::Put(cmd, duration);
}

- (void)commit {
    (void)rec->Push(Metal::Rec::Op::CbCommit, (__bridge void*)self);
}

@end

#define LICHII_STAGE_BINDS(Stage, stage_index)                                                     \
    -(void)set##Stage##Buffer : (id<MTLBuffer>)buffer offset : (NSUInteger)offset atIndex          \
        : (NSUInteger)index {                                                                      \
        LICHII_REC(RBuffer);                                                                       \
        cmd.stage = stage_index;                                                                   \
        LICHII_KEEP(obj, buffer);                                                                  \
        Metal::Rec::Put(cmd, Metal::Rec::BufferArgs{offset, static_cast<u32>(index)});             \
    }                                                                                              \
    -(void)set##Stage##BufferOffset : (NSUInteger)offset atIndex : (NSUInteger)index {             \
        LICHII_REC(RBufferOffset);                                                                 \
        cmd.stage = stage_index;                                                                   \
        Metal::Rec::Put(cmd, Metal::Rec::BufferArgs{offset, static_cast<u32>(index)});             \
    }                                                                                              \
    -(void)set##Stage##Bytes : (const void*)bytes length : (NSUInteger)length atIndex              \
        : (NSUInteger)index {                                                                      \
        LICHII_REC(RBytes);                                                                        \
        cmd.stage = stage_index;                                                                   \
        cmd.payload = rec->Bytes(bytes, length);                                                   \
        Metal::Rec::Put(cmd,                                                                       \
                        Metal::Rec::BytesArgs{static_cast<u32>(length), static_cast<u32>(index)}); \
    }                                                                                              \
    -(void)set##Stage##Texture : (id<MTLTexture>)texture atIndex : (NSUInteger)index {             \
        LICHII_REC(RTexture);                                                                      \
        cmd.stage = stage_index;                                                                   \
        LICHII_KEEP(obj, texture);                                                                 \
        Metal::Rec::Put(cmd, Metal::Rec::IndexArgs{static_cast<u32>(index)});                      \
    }                                                                                              \
    -(void)set##Stage##SamplerState : (id<MTLSamplerState>)sampler atIndex : (NSUInteger)index {   \
        LICHII_REC(RSampler);                                                                      \
        cmd.stage = stage_index;                                                                   \
        LICHII_KEEP(obj, sampler);                                                                 \
        Metal::Rec::Put(cmd, Metal::Rec::IndexArgs{static_cast<u32>(index)});                      \
    }

@implementation LichiiRecRenderEncoder

LICHII_STAGE_BINDS(Vertex, 0)
LICHII_STAGE_BINDS(Fragment, 1)
LICHII_STAGE_BINDS(Object, 2)
LICHII_STAGE_BINDS(Mesh, 3)

- (void)setRenderPipelineState:(id<MTLRenderPipelineState>)state {
    LICHII_REC(RPipeline);
    LICHII_KEEP(obj, state);
}

- (void)setDepthStencilState:(id<MTLDepthStencilState>)state {
    LICHII_REC(RDepthStencil);
    LICHII_KEEP(obj, state);
}

- (void)setViewport:(MTLViewport)viewport {
    LICHII_REC(RViewport);
    Metal::Rec::Put(cmd, viewport);
}

- (void)setViewports:(const MTLViewport[])viewports count:(NSUInteger)count {
    LICHII_REC(RViewports);
    cmd.payload = rec->Bytes(viewports, count * sizeof(MTLViewport));
    Metal::Rec::Put(cmd, Metal::Rec::U32Args{static_cast<u32>(count), 0});
}

- (void)setScissorRect:(MTLScissorRect)rect {
    LICHII_REC(RScissor);
    Metal::Rec::Put(cmd, rect);
}

- (void)setScissorRects:(const MTLScissorRect[])rects count:(NSUInteger)count {
    LICHII_REC(RScissors);
    cmd.payload = rec->Bytes(rects, count * sizeof(MTLScissorRect));
    Metal::Rec::Put(cmd, Metal::Rec::U32Args{static_cast<u32>(count), 0});
}

- (void)setCullMode:(MTLCullMode)mode {
    LICHII_REC(RCull);
    Metal::Rec::Put(cmd, Metal::Rec::U32Args{static_cast<u32>(mode), 0});
}

- (void)setFrontFacingWinding:(MTLWinding)winding {
    LICHII_REC(RWinding);
    Metal::Rec::Put(cmd, Metal::Rec::U32Args{static_cast<u32>(winding), 0});
}

- (void)setDepthClipMode:(MTLDepthClipMode)mode {
    LICHII_REC(RDepthClip);
    Metal::Rec::Put(cmd, Metal::Rec::U32Args{static_cast<u32>(mode), 0});
}

- (void)setTriangleFillMode:(MTLTriangleFillMode)mode {
    LICHII_REC(RFill);
    Metal::Rec::Put(cmd, Metal::Rec::U32Args{static_cast<u32>(mode), 0});
}

- (void)setDepthBias:(float)bias slopeScale:(float)slope clamp:(float)clamp {
    LICHII_REC(RDepthBias);
    Metal::Rec::Put(cmd, Metal::Rec::F32Args{{bias, slope, clamp, 0.0f}});
}

- (void)setStencilReferenceValue:(uint32_t)value {
    LICHII_REC(RStencilRef);
    Metal::Rec::Put(cmd, Metal::Rec::U32Args{value, 0});
}

- (void)setStencilFrontReferenceValue:(uint32_t)front backReferenceValue:(uint32_t)back {
    LICHII_REC(RStencilRef2);
    Metal::Rec::Put(cmd, Metal::Rec::U32Args{front, back});
}

- (void)setBlendColorRed:(float)red green:(float)green blue:(float)blue alpha:(float)alpha {
    LICHII_REC(RBlendColor);
    Metal::Rec::Put(cmd, Metal::Rec::F32Args{{red, green, blue, alpha}});
}

- (void)setDepthTestMinBound:(float)min_bound maxBound:(float)max_bound {
    LICHII_REC(RDepthBounds);
    Metal::Rec::Put(cmd, Metal::Rec::F32Args{{min_bound, max_bound, 0.0f, 0.0f}});
}

- (void)setVisibilityResultMode:(MTLVisibilityResultMode)mode offset:(NSUInteger)offset {
    LICHII_REC(RVisibility);
    Metal::Rec::Put(cmd, Metal::Rec::VisibilityArgs{offset, static_cast<u32>(mode)});
}

- (void)setColorStoreAction:(MTLStoreAction)action atIndex:(NSUInteger)index {
    LICHII_REC(RColorStore);
    Metal::Rec::Put(cmd, Metal::Rec::U32Args{static_cast<u32>(action), static_cast<u32>(index)});
}

- (void)setDepthStoreAction:(MTLStoreAction)action {
    LICHII_REC(RDepthStore);
    Metal::Rec::Put(cmd, Metal::Rec::U32Args{static_cast<u32>(action), 0});
}

- (void)setStencilStoreAction:(MTLStoreAction)action {
    LICHII_REC(RStencilStore);
    Metal::Rec::Put(cmd, Metal::Rec::U32Args{static_cast<u32>(action), 0});
}

- (void)useResource:(id<MTLResource>)resource
              usage:(MTLResourceUsage)usage
             stages:(MTLRenderStages)stages {
    LICHII_REC(RUseResource);
    LICHII_KEEP(obj, resource);
    Metal::Rec::Put(cmd, Metal::Rec::UseArgs{static_cast<u32>(usage), static_cast<u32>(stages)});
}

- (void)setTessellationFactorBuffer:(id<MTLBuffer>)buffer
                             offset:(NSUInteger)offset
                     instanceStride:(NSUInteger)stride {
    LICHII_REC(RTessFactors);
    LICHII_KEEP(obj, buffer);
    Metal::Rec::Put(cmd, Metal::Rec::TessArgs{offset, stride});
}

- (void)drawPrimitives:(MTLPrimitiveType)type
           vertexStart:(NSUInteger)start
           vertexCount:(NSUInteger)count {
    LICHII_REC(RDraw);
    Metal::Rec::Put(cmd, Metal::Rec::DrawArgs{static_cast<u32>(type), static_cast<u32>(start),
                                              static_cast<u32>(count), 1, 0});
}

- (void)drawPrimitives:(MTLPrimitiveType)type
           vertexStart:(NSUInteger)start
           vertexCount:(NSUInteger)count
         instanceCount:(NSUInteger)instances
          baseInstance:(NSUInteger)base_instance {
    LICHII_REC(RDrawInstanced);
    Metal::Rec::Put(cmd, Metal::Rec::DrawArgs{static_cast<u32>(type), static_cast<u32>(start),
                                              static_cast<u32>(count), static_cast<u32>(instances),
                                              static_cast<u32>(base_instance)});
}

- (void)drawPrimitives:(MTLPrimitiveType)type
          indirectBuffer:(id<MTLBuffer>)buffer
    indirectBufferOffset:(NSUInteger)offset {
    LICHII_REC(RDrawIndirect);
    LICHII_KEEP(obj, buffer);
    Metal::Rec::Put(cmd, Metal::Rec::DrawIndirectArgs{offset, static_cast<u32>(type)});
}

- (void)drawIndexedPrimitives:(MTLPrimitiveType)type
                   indexCount:(NSUInteger)count
                    indexType:(MTLIndexType)index_type
                  indexBuffer:(id<MTLBuffer>)index_buffer
            indexBufferOffset:(NSUInteger)index_offset
                instanceCount:(NSUInteger)instances
                   baseVertex:(NSInteger)base_vertex
                 baseInstance:(NSUInteger)base_instance {
    LICHII_REC(RDrawIndexed);
    LICHII_KEEP(obj, index_buffer);
    Metal::Rec::Put(cmd, Metal::Rec::DrawIndexedArgs{
                             index_offset, static_cast<u32>(type), static_cast<u32>(count),
                             static_cast<u32>(index_type), static_cast<u32>(instances),
                             static_cast<s64>(base_vertex), static_cast<u32>(base_instance)});
}

- (void)drawIndexedPrimitives:(MTLPrimitiveType)type
                    indexType:(MTLIndexType)index_type
                  indexBuffer:(id<MTLBuffer>)index_buffer
            indexBufferOffset:(NSUInteger)index_offset
               indirectBuffer:(id<MTLBuffer>)buffer
         indirectBufferOffset:(NSUInteger)offset {
    LICHII_REC(RDrawIndexedIndirect);
    LICHII_KEEP(obj, index_buffer);
    LICHII_KEEP(obj2, buffer);
    Metal::Rec::Put(cmd, Metal::Rec::DrawIndexedIndirectArgs{index_offset, offset,
                                                             static_cast<u32>(type),
                                                             static_cast<u32>(index_type)});
}

- (void)drawPatches:(NSUInteger)control_points
                patchStart:(NSUInteger)start
                patchCount:(NSUInteger)count
          patchIndexBuffer:(id<MTLBuffer>)index_buffer
    patchIndexBufferOffset:(NSUInteger)index_offset
             instanceCount:(NSUInteger)instances
              baseInstance:(NSUInteger)base_instance {
    LICHII_REC(RDrawPatches);
    LICHII_KEEP(obj, index_buffer);
    Metal::Rec::Put(cmd, Metal::Rec::PatchArgs{index_offset, static_cast<u32>(control_points),
                                               static_cast<u32>(start), static_cast<u32>(count),
                                               static_cast<u32>(instances),
                                               static_cast<u32>(base_instance)});
}

- (void)drawMeshThreadgroups:(MTLSize)groups
    threadsPerObjectThreadgroup:(MTLSize)object_threads
      threadsPerMeshThreadgroup:(MTLSize)mesh_threads {
    LICHII_REC(RDrawMesh);
    Metal::Rec::Put(cmd, Metal::Rec::MeshArgs{Metal::Rec::ToSize3(groups),
                                              Metal::Rec::ToSize3(object_threads),
                                              Metal::Rec::ToSize3(mesh_threads)});
}

- (void)drawMeshThreadgroupsWithIndirectBuffer:(id<MTLBuffer>)buffer
                          indirectBufferOffset:(NSUInteger)offset
                   threadsPerObjectThreadgroup:(MTLSize)object_threads
                     threadsPerMeshThreadgroup:(MTLSize)mesh_threads {
    LICHII_REC(RDrawMeshIndirect);
    LICHII_KEEP(obj, buffer);
    Metal::Rec::Put(cmd, Metal::Rec::MeshIndirectArgs{offset, Metal::Rec::ToSize3(object_threads),
                                                      Metal::Rec::ToSize3(mesh_threads)});
}

@end

@implementation LichiiRecComputeEncoder

- (void)setComputePipelineState:(id<MTLComputePipelineState>)state {
    LICHII_REC(CPipeline);
    LICHII_KEEP(obj, state);
}

- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
    LICHII_REC(CBuffer);
    LICHII_KEEP(obj, buffer);
    Metal::Rec::Put(cmd, Metal::Rec::BufferArgs{offset, static_cast<u32>(index)});
}

- (void)setBufferOffset:(NSUInteger)offset atIndex:(NSUInteger)index {
    LICHII_REC(CBufferOffset);
    Metal::Rec::Put(cmd, Metal::Rec::BufferArgs{offset, static_cast<u32>(index)});
}

- (void)setBytes:(const void*)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    LICHII_REC(CBytes);
    cmd.payload = rec->Bytes(bytes, length);
    Metal::Rec::Put(cmd, Metal::Rec::BytesArgs{static_cast<u32>(length), static_cast<u32>(index)});
}

- (void)setTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index {
    LICHII_REC(CTexture);
    LICHII_KEEP(obj, texture);
    Metal::Rec::Put(cmd, Metal::Rec::IndexArgs{static_cast<u32>(index)});
}

- (void)setSamplerState:(id<MTLSamplerState>)sampler atIndex:(NSUInteger)index {
    LICHII_REC(CSampler);
    LICHII_KEEP(obj, sampler);
    Metal::Rec::Put(cmd, Metal::Rec::IndexArgs{static_cast<u32>(index)});
}

- (void)useResource:(id<MTLResource>)resource usage:(MTLResourceUsage)usage {
    LICHII_REC(CUseResource);
    LICHII_KEEP(obj, resource);
    Metal::Rec::Put(cmd, Metal::Rec::UseArgs{static_cast<u32>(usage), 0});
}

- (void)memoryBarrierWithScope:(MTLBarrierScope)scope {
    LICHII_REC(CMemoryBarrier);
    Metal::Rec::Put(cmd, Metal::Rec::U32Args{static_cast<u32>(scope), 0});
}

- (void)setStageInRegion:(MTLRegion)region {
    LICHII_REC(CStageIn);
    Metal::Rec::Put(cmd, Metal::Rec::DispatchArgs{Metal::Rec::ToSize3(region.origin),
                                                  Metal::Rec::ToSize3(region.size)});
}

- (void)dispatchThreadgroups:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads {
    LICHII_REC(CDispatch);
    Metal::Rec::Put(cmd, Metal::Rec::DispatchArgs{Metal::Rec::ToSize3(groups),
                                                  Metal::Rec::ToSize3(threads)});
}

- (void)dispatchThreadgroupsWithIndirectBuffer:(id<MTLBuffer>)buffer
                          indirectBufferOffset:(NSUInteger)offset
                         threadsPerThreadgroup:(MTLSize)threads {
    LICHII_REC(CDispatchIndirect);
    LICHII_KEEP(obj, buffer);
    Metal::Rec::Put(cmd, Metal::Rec::DispatchIndirectArgs{offset, Metal::Rec::ToSize3(threads)});
}

- (void)dispatchThreads:(MTLSize)grid threadsPerThreadgroup:(MTLSize)threads {
    LICHII_REC(CDispatchThreads);
    Metal::Rec::Put(cmd, Metal::Rec::DispatchArgs{Metal::Rec::ToSize3(grid),
                                                  Metal::Rec::ToSize3(threads)});
}

@end

@implementation LichiiRecBlitEncoder

- (void)copyFromBuffer:(id<MTLBuffer>)src
           sourceOffset:(NSUInteger)src_offset
               toBuffer:(id<MTLBuffer>)dst
      destinationOffset:(NSUInteger)dst_offset
                   size:(NSUInteger)size {
    LICHII_REC(BCopyBufferToBuffer);
    LICHII_KEEP(obj, src);
    LICHII_KEEP(obj2, dst);
    Metal::Rec::Put(cmd, Metal::Rec::CopyBBArgs{src_offset, dst_offset, size});
}

- (void)copyFromBuffer:(id<MTLBuffer>)src
           sourceOffset:(NSUInteger)offset
      sourceBytesPerRow:(NSUInteger)bytes_per_row
    sourceBytesPerImage:(NSUInteger)bytes_per_image
             sourceSize:(MTLSize)size
              toTexture:(id<MTLTexture>)dst
       destinationSlice:(NSUInteger)slice
       destinationLevel:(NSUInteger)level
      destinationOrigin:(MTLOrigin)origin {
    LICHII_REC(BCopyBufferToTexture);
    LICHII_KEEP(obj, src);
    LICHII_KEEP(obj2, dst);
    Metal::Rec::Put(cmd, Metal::Rec::CopyBTArgs{offset, static_cast<u32>(bytes_per_row),
                                                static_cast<u32>(bytes_per_image),
                                                Metal::Rec::ToSize3(size), static_cast<u32>(slice),
                                                static_cast<u32>(level),
                                                Metal::Rec::ToSize3(origin)});
}

- (void)copyFromTexture:(id<MTLTexture>)src
                 sourceSlice:(NSUInteger)slice
                 sourceLevel:(NSUInteger)level
                sourceOrigin:(MTLOrigin)origin
                  sourceSize:(MTLSize)size
                    toBuffer:(id<MTLBuffer>)dst
           destinationOffset:(NSUInteger)offset
      destinationBytesPerRow:(NSUInteger)bytes_per_row
    destinationBytesPerImage:(NSUInteger)bytes_per_image {
    LICHII_REC(BCopyTextureToBuffer);
    LICHII_KEEP(obj, src);
    LICHII_KEEP(obj2, dst);
    Metal::Rec::Put(cmd, Metal::Rec::CopyTBArgs{offset, static_cast<u32>(bytes_per_row),
                                                static_cast<u32>(bytes_per_image),
                                                static_cast<u32>(slice), static_cast<u32>(level),
                                                Metal::Rec::ToSize3(origin),
                                                Metal::Rec::ToSize3(size)});
}

- (void)copyFromTexture:(id<MTLTexture>)src
             sourceSlice:(NSUInteger)src_slice
             sourceLevel:(NSUInteger)src_level
            sourceOrigin:(MTLOrigin)src_origin
              sourceSize:(MTLSize)size
               toTexture:(id<MTLTexture>)dst
        destinationSlice:(NSUInteger)dst_slice
        destinationLevel:(NSUInteger)dst_level
       destinationOrigin:(MTLOrigin)dst_origin {
    LICHII_REC(BCopyTextureToTexture);
    LICHII_KEEP(obj, src);
    LICHII_KEEP(obj2, dst);
    const Metal::Rec::CopyTTArgs args{static_cast<u32>(src_slice), static_cast<u32>(src_level),
                                      Metal::Rec::ToSize3(src_origin), Metal::Rec::ToSize3(size),
                                      static_cast<u32>(dst_slice), static_cast<u32>(dst_level),
                                      Metal::Rec::ToSize3(dst_origin)};
    cmd.payload = rec->Bytes(&args, sizeof(args));
}

- (void)fillBuffer:(id<MTLBuffer>)buffer range:(NSRange)range value:(uint8_t)value {
    LICHII_REC(BFill);
    LICHII_KEEP(obj, buffer);
    Metal::Rec::Put(cmd, Metal::Rec::FillArgs{range.location, range.length, value});
}

@end

#undef LICHII_STAGE_BINDS
#undef LICHII_KEEP
#undef LICHII_REC

namespace Metal::Rec {

inline void Recorder::Execute(const Cmd& c, const Chunk& chunk) {
    // Chunk::kept owns obj and obj2 across the whole chunk, so no per-command ARC retain/release.
    const __unsafe_unretained id obj = (__bridge id)c.obj;
    const __unsafe_unretained id obj2 = (__bridge id)c.obj2;
    LichiiRecProxy* const __unsafe_unretained proxy = (__bridge LichiiRecProxy*)c.target;
    const u8* const bytes = chunk.bytes.data() + c.payload;
    switch (c.op) {
    case Op::CbNew: {
        MTLCommandBufferDescriptor* const descriptor = obj;
        proxy->real = descriptor != nil ? [queue commandBufferWithDescriptor:descriptor]
                                        : [queue commandBuffer];
        return;
    }
    case Op::CbRender: {
        const id<MTLCommandBuffer> buffer = ((LichiiRecProxy*)obj)->real;
        proxy->real = [buffer renderCommandEncoderWithDescriptor:(MTLRenderPassDescriptor*)obj2];
        if (proxy->real == nil) {
            NoteRefusedPass();
        }
        return;
    }
    case Op::CbCompute: {
        const id<MTLCommandBuffer> buffer = ((LichiiRecProxy*)obj)->real;
        proxy->real = [buffer
            computeCommandEncoderWithDispatchType:static_cast<MTLDispatchType>(Get<uint32_t>(c))];
        return;
    }
    case Op::CbBlit: {
        const id<MTLCommandBuffer> buffer = ((LichiiRecProxy*)obj)->real;
        proxy->real = [buffer blitCommandEncoder];
        return;
    }
    case Op::CbCompleted:
        [(id<MTLCommandBuffer>)proxy->real addCompletedHandler:(MTLCommandBufferHandler)obj];
        return;
    case Op::CbSignalEvent:
        [(id<MTLCommandBuffer>)proxy->real encodeSignalEvent:(id<MTLEvent>)obj
                                                       value:Get<uint64_t>(c)];
        return;
    case Op::CbPresent:
        [(id<MTLCommandBuffer>)proxy->real presentDrawable:(id<MTLDrawable>)obj];
        return;
    case Op::CbPresentAfter:
        [(id<MTLCommandBuffer>)proxy->real presentDrawable:(id<MTLDrawable>)obj
                                      afterMinimumDuration:Get<CFTimeInterval>(c)];
        return;
    case Op::CbCommit:
        [(id<MTLCommandBuffer>)proxy->real commit];
        // The creation's reference (CbNew); whoever holds the proxy keeps it (and the real one).
        (void)CFBridgingRelease(c.target);
        return;
    case Op::Call: {
        const CallArgs args = Get<CallArgs>(c);
        args.fn(args.context, args.value);
        return;
    }
    case Op::SetLabel:
        [proxy->real setLabel:(NSString*)obj];
        return;
    case Op::EndEncoding:
        [(id<MTLCommandEncoder>)proxy->real endEncoding];
        (void)CFBridgingRelease(c.target);
        return;
    default:
        break;
    }

    if (c.op >= Op::RPipeline && c.op <= Op::RDrawMeshIndirect) {
        const id<MTLRenderCommandEncoder> __unsafe_unretained e = proxy->real;
        switch (c.op) {
        case Op::RPipeline:
            [e setRenderPipelineState:obj];
            break;
        case Op::RDepthStencil:
            [e setDepthStencilState:obj];
            break;
        case Op::RViewport:
            [e setViewport:Get<MTLViewport>(c)];
            break;
        case Op::RViewports:
            [e setViewports:reinterpret_cast<const MTLViewport*>(bytes)
                      count:Get<U32Args>(c).a];
            break;
        case Op::RScissor:
            [e setScissorRect:Get<MTLScissorRect>(c)];
            break;
        case Op::RScissors:
            [e setScissorRects:reinterpret_cast<const MTLScissorRect*>(bytes)
                         count:Get<U32Args>(c).a];
            break;
        case Op::RCull:
            [e setCullMode:static_cast<MTLCullMode>(Get<U32Args>(c).a)];
            break;
        case Op::RWinding:
            [e setFrontFacingWinding:static_cast<MTLWinding>(Get<U32Args>(c).a)];
            break;
        case Op::RDepthClip:
            [e setDepthClipMode:static_cast<MTLDepthClipMode>(Get<U32Args>(c).a)];
            break;
        case Op::RFill:
            [e setTriangleFillMode:static_cast<MTLTriangleFillMode>(Get<U32Args>(c).a)];
            break;
        case Op::RDepthBias: {
            const F32Args args = Get<F32Args>(c);
            [e setDepthBias:args.v[0] slopeScale:args.v[1] clamp:args.v[2]];
            break;
        }
        case Op::RStencilRef:
            [e setStencilReferenceValue:Get<U32Args>(c).a];
            break;
        case Op::RStencilRef2: {
            const U32Args args = Get<U32Args>(c);
            [e setStencilFrontReferenceValue:args.a backReferenceValue:args.b];
            break;
        }
        case Op::RBlendColor: {
            const F32Args args = Get<F32Args>(c);
            [e setBlendColorRed:args.v[0] green:args.v[1] blue:args.v[2] alpha:args.v[3]];
            break;
        }
        case Op::RDepthBounds: {
#if defined(__MAC_26_0) || defined(__IPHONE_26_0)
            if (@available(macOS 26.0, iOS 26.0, *)) {
                const F32Args args = Get<F32Args>(c);
                [e setDepthTestMinBound:args.v[0] maxBound:args.v[1]];
            }
#endif
            break;
        }
        case Op::RVisibility: {
            const VisibilityArgs args = Get<VisibilityArgs>(c);
            [e setVisibilityResultMode:static_cast<MTLVisibilityResultMode>(args.mode)
                                offset:args.offset];
            break;
        }
        case Op::RColorStore: {
            const U32Args args = Get<U32Args>(c);
            [e setColorStoreAction:static_cast<MTLStoreAction>(args.a) atIndex:args.b];
            break;
        }
        case Op::RDepthStore:
            [e setDepthStoreAction:static_cast<MTLStoreAction>(Get<U32Args>(c).a)];
            break;
        case Op::RStencilStore:
            [e setStencilStoreAction:static_cast<MTLStoreAction>(Get<U32Args>(c).a)];
            break;
        case Op::RBuffer: {
            const BufferArgs args = Get<BufferArgs>(c);
            switch (c.stage) {
            case 0:
                [e setVertexBuffer:obj offset:args.offset atIndex:args.index];
                break;
            case 1:
                [e setFragmentBuffer:obj offset:args.offset atIndex:args.index];
                break;
            case 2:
                [e setObjectBuffer:obj offset:args.offset atIndex:args.index];
                break;
            default:
                [e setMeshBuffer:obj offset:args.offset atIndex:args.index];
                break;
            }
            break;
        }
        case Op::RBufferOffset: {
            const BufferArgs args = Get<BufferArgs>(c);
            switch (c.stage) {
            case 0:
                [e setVertexBufferOffset:args.offset atIndex:args.index];
                break;
            case 1:
                [e setFragmentBufferOffset:args.offset atIndex:args.index];
                break;
            case 2:
                [e setObjectBufferOffset:args.offset atIndex:args.index];
                break;
            default:
                [e setMeshBufferOffset:args.offset atIndex:args.index];
                break;
            }
            break;
        }
        case Op::RBytes: {
            const BytesArgs args = Get<BytesArgs>(c);
            switch (c.stage) {
            case 0:
                [e setVertexBytes:bytes length:args.length atIndex:args.index];
                break;
            case 1:
                [e setFragmentBytes:bytes length:args.length atIndex:args.index];
                break;
            case 2:
                [e setObjectBytes:bytes length:args.length atIndex:args.index];
                break;
            default:
                [e setMeshBytes:bytes length:args.length atIndex:args.index];
                break;
            }
            break;
        }
        case Op::RTexture: {
            const u32 index = Get<IndexArgs>(c).index;
            switch (c.stage) {
            case 0:
                [e setVertexTexture:obj atIndex:index];
                break;
            case 1:
                [e setFragmentTexture:obj atIndex:index];
                break;
            case 2:
                [e setObjectTexture:obj atIndex:index];
                break;
            default:
                [e setMeshTexture:obj atIndex:index];
                break;
            }
            break;
        }
        case Op::RSampler: {
            const u32 index = Get<IndexArgs>(c).index;
            switch (c.stage) {
            case 0:
                [e setVertexSamplerState:obj atIndex:index];
                break;
            case 1:
                [e setFragmentSamplerState:obj atIndex:index];
                break;
            case 2:
                [e setObjectSamplerState:obj atIndex:index];
                break;
            default:
                [e setMeshSamplerState:obj atIndex:index];
                break;
            }
            break;
        }
        case Op::RUseResource: {
            const UseArgs args = Get<UseArgs>(c);
            [e useResource:obj
                     usage:static_cast<MTLResourceUsage>(args.usage)
                    stages:static_cast<MTLRenderStages>(args.stages)];
            break;
        }
        case Op::RTessFactors: {
            const TessArgs args = Get<TessArgs>(c);
            [e setTessellationFactorBuffer:obj offset:args.offset instanceStride:args.stride];
            break;
        }
        case Op::RDraw: {
            const DrawArgs args = Get<DrawArgs>(c);
            [e drawPrimitives:static_cast<MTLPrimitiveType>(args.type)
                  vertexStart:args.start
                  vertexCount:args.count];
            break;
        }
        case Op::RDrawInstanced: {
            const DrawArgs args = Get<DrawArgs>(c);
            [e drawPrimitives:static_cast<MTLPrimitiveType>(args.type)
                  vertexStart:args.start
                  vertexCount:args.count
                instanceCount:args.instances
                 baseInstance:args.base_instance];
            break;
        }
        case Op::RDrawIndirect: {
            const DrawIndirectArgs args = Get<DrawIndirectArgs>(c);
            [e drawPrimitives:static_cast<MTLPrimitiveType>(args.type)
                      indirectBuffer:obj
                indirectBufferOffset:args.offset];
            break;
        }
        case Op::RDrawIndexed: {
            const DrawIndexedArgs args = Get<DrawIndexedArgs>(c);
            [e drawIndexedPrimitives:static_cast<MTLPrimitiveType>(args.type)
                          indexCount:args.count
                           indexType:static_cast<MTLIndexType>(args.index_type)
                         indexBuffer:obj
                   indexBufferOffset:args.index_offset
                       instanceCount:args.instances
                          baseVertex:static_cast<NSInteger>(args.base_vertex)
                        baseInstance:args.base_instance];
            break;
        }
        case Op::RDrawIndexedIndirect: {
            const DrawIndexedIndirectArgs args = Get<DrawIndexedIndirectArgs>(c);
            [e drawIndexedPrimitives:static_cast<MTLPrimitiveType>(args.type)
                           indexType:static_cast<MTLIndexType>(args.index_type)
                         indexBuffer:obj
                   indexBufferOffset:args.index_offset
                      indirectBuffer:obj2
                indirectBufferOffset:args.indirect_offset];
            break;
        }
        case Op::RDrawPatches: {
            const PatchArgs args = Get<PatchArgs>(c);
            [e drawPatches:args.control_points
                            patchStart:args.start
                            patchCount:args.count
                      patchIndexBuffer:obj
                patchIndexBufferOffset:args.index_offset
                         instanceCount:args.instances
                          baseInstance:args.base_instance];
            break;
        }
        case Op::RDrawMesh: {
            const MeshArgs args = Get<MeshArgs>(c);
            [e drawMeshThreadgroups:ToSize(args.groups)
                threadsPerObjectThreadgroup:ToSize(args.object_threads)
                  threadsPerMeshThreadgroup:ToSize(args.mesh_threads)];
            break;
        }
        case Op::RDrawMeshIndirect: {
            const MeshIndirectArgs args = Get<MeshIndirectArgs>(c);
            [e drawMeshThreadgroupsWithIndirectBuffer:obj
                                 indirectBufferOffset:args.offset
                          threadsPerObjectThreadgroup:ToSize(args.object_threads)
                            threadsPerMeshThreadgroup:ToSize(args.mesh_threads)];
            break;
        }
        default:
            break;
        }
        return;
    }

    if (c.op >= Op::CPipeline && c.op <= Op::CDispatchThreads) {
        const id<MTLComputeCommandEncoder> __unsafe_unretained e = proxy->real;
        switch (c.op) {
        case Op::CPipeline:
            [e setComputePipelineState:obj];
            break;
        case Op::CBuffer: {
            const BufferArgs args = Get<BufferArgs>(c);
            [e setBuffer:obj offset:args.offset atIndex:args.index];
            break;
        }
        case Op::CBufferOffset: {
            const BufferArgs args = Get<BufferArgs>(c);
            [e setBufferOffset:args.offset atIndex:args.index];
            break;
        }
        case Op::CBytes: {
            const BytesArgs args = Get<BytesArgs>(c);
            [e setBytes:bytes length:args.length atIndex:args.index];
            break;
        }
        case Op::CTexture:
            [e setTexture:obj atIndex:Get<IndexArgs>(c).index];
            break;
        case Op::CSampler:
            [e setSamplerState:obj atIndex:Get<IndexArgs>(c).index];
            break;
        case Op::CUseResource:
            [e useResource:obj usage:static_cast<MTLResourceUsage>(Get<UseArgs>(c).usage)];
            break;
        case Op::CMemoryBarrier:
            [e memoryBarrierWithScope:static_cast<MTLBarrierScope>(Get<U32Args>(c).a)];
            break;
        case Op::CStageIn: {
            const DispatchArgs args = Get<DispatchArgs>(c);
            [e setStageInRegion:MTLRegionMake3D(args.groups.x, args.groups.y, args.groups.z,
                                                args.threads.x, args.threads.y, args.threads.z)];
            break;
        }
        case Op::CDispatch: {
            const DispatchArgs args = Get<DispatchArgs>(c);
            [e dispatchThreadgroups:ToSize(args.groups) threadsPerThreadgroup:ToSize(args.threads)];
            break;
        }
        case Op::CDispatchIndirect: {
            const DispatchIndirectArgs args = Get<DispatchIndirectArgs>(c);
            [e dispatchThreadgroupsWithIndirectBuffer:obj
                                 indirectBufferOffset:args.offset
                                threadsPerThreadgroup:ToSize(args.threads)];
            break;
        }
        case Op::CDispatchThreads: {
            const DispatchArgs args = Get<DispatchArgs>(c);
            [e dispatchThreads:ToSize(args.groups) threadsPerThreadgroup:ToSize(args.threads)];
            break;
        }
        default:
            break;
        }
        return;
    }

    const id<MTLBlitCommandEncoder> __unsafe_unretained e = proxy->real;
    switch (c.op) {
    case Op::BCopyBufferToBuffer: {
        const CopyBBArgs args = Get<CopyBBArgs>(c);
        [e copyFromBuffer:obj
                 sourceOffset:args.src_offset
                     toBuffer:obj2
            destinationOffset:args.dst_offset
                         size:args.size];
        break;
    }
    case Op::BCopyBufferToTexture: {
        const CopyBTArgs args = Get<CopyBTArgs>(c);
        [e copyFromBuffer:obj
                   sourceOffset:args.src_offset
              sourceBytesPerRow:args.bytes_per_row
            sourceBytesPerImage:args.bytes_per_image
                     sourceSize:ToSize(args.size)
                      toTexture:obj2
               destinationSlice:args.slice
               destinationLevel:args.level
              destinationOrigin:ToOrigin(args.origin)];
        break;
    }
    case Op::BCopyTextureToBuffer: {
        const CopyTBArgs args = Get<CopyTBArgs>(c);
        [e copyFromTexture:obj
                         sourceSlice:args.slice
                         sourceLevel:args.level
                        sourceOrigin:ToOrigin(args.origin)
                          sourceSize:ToSize(args.size)
                            toBuffer:obj2
                   destinationOffset:args.dst_offset
              destinationBytesPerRow:args.bytes_per_row
            destinationBytesPerImage:args.bytes_per_image];
        break;
    }
    case Op::BCopyTextureToTexture: {
        CopyTTArgs args;
        std::memcpy(&args, bytes, sizeof(args));
        [e copyFromTexture:obj
                 sourceSlice:args.src_slice
                 sourceLevel:args.src_level
                sourceOrigin:ToOrigin(args.src_origin)
                  sourceSize:ToSize(args.size)
                   toTexture:obj2
            destinationSlice:args.dst_slice
            destinationLevel:args.dst_level
           destinationOrigin:ToOrigin(args.dst_origin)];
        break;
    }
    case Op::BFill: {
        const FillArgs args = Get<FillArgs>(c);
        [e fillBuffer:obj
                range:NSMakeRange(args.location, args.length)
                value:static_cast<uint8_t>(args.value)];
        break;
    }
    default:
        break;
    }
}

} // namespace Metal::Rec
