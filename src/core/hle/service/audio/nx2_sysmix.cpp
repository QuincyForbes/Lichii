// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "audio_core/sink/silent_input_stream.h"
#include "audio_core/sink/sink.h"
#include "audio_core/sink/sink_stream.h"
#include "common/logging.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/ipc.h"
#include "core/hle/kernel/k_event.h"
#include "core/hle/kernel/k_light_lock.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/service/audio/errors.h"
#include "core/hle/service/audio/nx2_sysmix.h"
#include "core/hle/service/kernel_helpers.h"
#include "core/memory.h"

namespace Service::Audio {
namespace {
constexpr u64 ReadOffset = 0x44;
constexpr u64 WriteOffset = 0x1004;
constexpr u64 PayloadOffset = 0x1040;
constexpr u32 BlockFrames = 240;

u64 AlignUp(u64 value, u64 alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

bool IsVirtualOutputLoopbackName(std::string_view stream_name) {
    constexpr std::string_view VirtualOutput = "AudioVirtualOutput0";
    constexpr std::string_view Suffix = "<=AudioVirtualOutput0";
    return stream_name == VirtualOutput || stream_name.ends_with(Suffix);
}

bool IsVirtualOutputLoopbackIpc(Kernel::KernelCore& kernel, Kernel::KProcess& process,
                                u32 match_cmd_id, u64 stream_buffer, u64 stream_size,
                                u32 stream_flags, u32 stream_id) {
    if (Kernel::GetCurrentThreadPointer(kernel) == nullptr) {
        return false;
    }
    auto& memory = process.GetMemory();
    Kernel::KScopedLightLock lk{process.GetListLock()};
    for (const auto& thread : process.GetThreadList()) {
        const u64 tls = GetInteger(thread.GetTlsAddress());
        if (tls == 0) {
            continue;
        }
        std::array<u32, IPC::COMMAND_BUFFER_LENGTH> cmd_buf{};
        if (!memory.ReadBlock(tls, cmd_buf.data(), sizeof(cmd_buf))) {
            continue;
        }
        IPC::CommandHeader header{};
        std::memcpy(&header, cmd_buf.data(), sizeof(header));
        if (header.IsCloseCommand() || header.IsTipc() ||
            (header.type != IPC::CommandType::Request &&
             header.type != IPC::CommandType::RequestWithContext)) {
            continue;
        }
        std::size_t idx = 2;
        if (header.enable_handle_descriptor) {
            if (idx >= cmd_buf.size()) {
                continue;
            }
            IPC::HandleDescriptorHeader hd{};
            std::memcpy(&hd, &cmd_buf[idx++], sizeof(hd));
            idx +=
                (hd.send_current_pid ? 2u : 0u) + hd.num_handles_to_copy + hd.num_handles_to_move;
        }
        const u32 num_x = header.num_buf_x_descriptors;
        const u32 num_a = header.num_buf_a_descriptors;
        const u32 num_b = header.num_buf_b_descriptors;
        const u32 num_w = header.num_buf_w_descriptors;
        if (idx + 2u * num_x + 3u * (num_a + num_b + num_w) > cmd_buf.size()) {
            continue;
        }
        VAddr name_addr = 0;
        u64 name_size = 0;
        if (num_x > 0) {
            IPC::BufferDescriptorX desc_x{};
            std::memcpy(&desc_x, &cmd_buf[idx], sizeof(desc_x));
            name_addr = desc_x.Address();
            name_size = desc_x.Size();
        }
        idx += 2u * num_x;
        if (num_a > 0) {
            IPC::BufferDescriptorABW desc_a{};
            std::memcpy(&desc_a, &cmd_buf[idx], sizeof(desc_a));
            if (desc_a.Size() != 0) {
                name_addr = desc_a.Address();
                name_size = desc_a.Size();
            }
        }
        idx += 3u * (num_a + num_b + num_w);
        idx = (idx + 3u) & ~std::size_t{3};
        constexpr u32 SfciMagic = Common::MakeMagic('S', 'F', 'C', 'I');
        if (idx < cmd_buf.size() && cmd_buf[idx] != SfciMagic && idx + 4u < cmd_buf.size() &&
            cmd_buf[idx + 4u] == SfciMagic) {
            idx += 4u;
        }
        if (idx + 6u > cmd_buf.size() || cmd_buf[idx] != SfciMagic ||
            cmd_buf[idx + 2u] != match_cmd_id) {
            continue;
        }
        if (match_cmd_id == 1u) {
            if (idx + 12u > cmd_buf.size()) {
                continue;
            }
            const u32 req_device = cmd_buf[idx + 4u];
            const u32 req_flags = cmd_buf[idx + 5u];
            const u64 req_buffer = u64{cmd_buf[idx + 8u]} | (u64{cmd_buf[idx + 9u]} << 32);
            const u64 req_size = u64{cmd_buf[idx + 10u]} | (u64{cmd_buf[idx + 11u]} << 32);
            if (req_device != 2u || req_flags != stream_flags || req_buffer != stream_buffer ||
                req_size != stream_size) {
                continue;
            }
        } else if (match_cmd_id == 700u) {
            if (cmd_buf[idx + 4u] != stream_id || cmd_buf[idx + 5u] != 2u) {
                continue;
            }
        }
        if (name_addr == 0 || name_size == 0) {
            return false;
        }
        std::array<char, 0x100> raw_name{};
        const std::size_t to_read =
            static_cast<std::size_t>(std::min<u64>(name_size, raw_name.size() - 1));
        if (!memory.ReadBlock(name_addr, raw_name.data(), to_read)) {
            return false;
        }
        const char* const nul = std::char_traits<char>::find(raw_name.data(), to_read, '\0');
        const std::string_view stream_name{
            raw_name.data(), nul ? static_cast<std::size_t>(nul - raw_name.data()) : to_read};
        return IsVirtualOutputLoopbackName(stream_name);
    }
    return false;
}
} // namespace

struct SysmixOutput::Impl {
    Impl(Core::System& system_, Kernel::KProcess& process_, AudioCore::Sink::Sink& sink_, u32 id_,
         u64 buffer_, u64 size_, u32 creation_flags_, bool input_)
        : system{system_}, process{system.Kernel(), &process_}, sink{sink_},
          context{system, "sysmix"}, id{id_}, buffer{buffer_}, size{size_},
          creation_flags{creation_flags_}, input_stream{input_},
          loopback_input{input_ && IsVirtualOutputLoopbackIpc(system_.Kernel(), process_, 1u,
                                                              buffer_, size_, creation_flags_,
                                                              id_)} {
        state_event = context.CreateEvent("sysmix state");
        data_event = context.CreateEvent("sysmix data");
    }

    ~Impl() {
        Close();
        context.CloseEvent(data_event);
        context.CloseEvent(state_event);
    }

    void Close() {
        std::scoped_lock lock{mutex};
        open = false;
        engaged = false;
        data_mode = 0;
        ReleaseHost();
        if (state_event) {
            state_event->Signal(system.Kernel());
        }
    }

    void ReleaseHost() {
        if (host == silent_input.get()) {
            silent_input.reset();
        } else {
            sink.CloseStream(host);
        }
        host = nullptr;
    }

    /// Stand in with synthesized 48 kHz silence for virtual-output loopback or missing microphones.
    void UseSilentInput(u32 guest_channels, bool warn = true) {
        if (warn) {
            LOG_WARNING(Service_Audio,
                        "sysmix stream {}: host microphone unavailable, capturing silence", id);
        } else {
            LOG_INFO(Service_Audio,
                     "sysmix stream {}: virtual output loopback, capturing silence", id);
        }
        ReleaseHost();
        silent_input = std::make_unique<AudioCore::Sink::SilentInputStream>(system, guest_channels);
        host = silent_input.get();
    }

    /// Stop admission and notify the guest after invalid memory or a fatal host-stream failure.
    Result Fault(Result result = ResultInvalidAddressInfo) {
        faulted = true;
        engaged = false;
        ++counters.faults;
        if (host) {
            host->MarkFaulted();
        }
        state_event->Signal(system.Kernel());
        LOG_ERROR(Service_Audio, "sysmix stream {} fault {:#x}: copied={} accepted={} blocks={}",
                  id, result.raw, counters.copied_frames, counters.accepted_frames,
                  counters.admitted_blocks);
        return result;
    }

    /// Read valid ring cursors and verify that the guest has not changed our owned cursor.
    bool ReadOccupancy(u32& read, u32& write, u32& filled) {
        auto& memory = process->GetMemory();
        if (!memory.Read32Acquire(buffer + ReadOffset, read) ||
            !memory.Read32Acquire(buffer + WriteOffset, write) ||
            (input_stream ? write : read) != consumer || read >= slots || write >= slots) {
            return false;
        }
        filled = static_cast<u32>((u64{write} + slots - read) % slots);
        return true;
    }

    /// Signal when captured input or free output space reaches the notification threshold.
    void Notify(u32 filled) {
        const auto available = input_stream ? filled : slots - 1 - filled;
        if (data_mode == 2 && available >= static_cast<u32>(threshold)) {
            data_event->Signal(system.Kernel());
        }
    }

    // emu_us is the guest clock; the wall clock is the timestamp every log line carries.
    // Comparing their deltas over consecutive windows shows whether emulated time runs at the
    // host's rate (48000 frames of output per second of wall time).
    void LogStatus(u32 filled) {
        const u64 emu_us = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                system.CoreTiming().GetGlobalTimeNs())
                .count());
        LOG_INFO(Service_Audio,
                 "sysmix stream {} output: copied={} accepted={} nonzero={} blocks={} full={} "
                 "queued={} retried={} padded={} deferred={} filled={} emu_us={}",
                 id, counters.copied_frames, counters.accepted_frames, counters.nonzero_frames,
                 counters.admitted_blocks, counters.backpressure,
                 host ? host->GetQueueSize() : 0, counters.retried_frames, counters.padded_frames,
                 counters.deferred, filled, emu_us);
    }

    Core::System& system;
    Kernel::KScopedAutoObject<Kernel::KProcess> process;
    AudioCore::Sink::Sink& sink;
    KernelHelpers::ServiceContext context;
    Kernel::KEvent* state_event{};
    Kernel::KEvent* data_event{};
    AudioCore::Sink::SinkStream* host{};
    std::unique_ptr<AudioCore::Sink::SinkStream> silent_input;
    mutable std::mutex mutex;
    u32 id;
    u64 buffer;
    u64 size;
    u32 creation_flags;
    bool input_stream;
    bool loopback_input;
    u32 channels{2};
    AudioCore::SampleFormat format{};
    u32 frame_bytes{};
    u32 slots{};
    u32 consumer{};
    u32 data_mode{};
    s32 threshold{};
    f32 gain{1.0f};
    bool open{};
    bool engaged{};
    bool faulted{};
    std::array<u8, BlockFrames * 2 * sizeof(f32)> input{};
    std::array<s16, BlockFrames * 2> output{};
    SysmixOutputCounters counters{};
};

SysmixOutput::SysmixOutput(Core::System& system, Kernel::KProcess& process,
                           AudioCore::Sink::Sink& sink, u32 id, u64 buffer, u64 size,
                           u32 creation_flags, bool input)
    : impl{std::make_unique<Impl>(system, process, sink, id, buffer, size, creation_flags, input)} {
}

SysmixOutput::~SysmixOutput() = default;

/// Validate direction-specific creation flags and an aligned, mapped guest ring allocation.
Result SysmixOutput::ValidateRegion(Kernel::KProcess& process, u64 buffer, u64 size,
                                    u32 creation_flags, bool input) {
    R_UNLESS(creation_flags == (input ? 2U : 0x24U), ResultNotSupported);
    R_UNLESS(buffer != 0 && (buffer & 0xfff) == 0 && (size & 0xfff) == 0 && size >= 0x2000 &&
                 size <= std::numeric_limits<u64>::max() - buffer &&
                 process.GetMemory().IsValidVirtualAddressRange(buffer, size),
             ResultInvalidAddressInfo);
    R_SUCCEED();
}

bool SysmixOutput::IsFormatSupported(u32 sample_type, u32 channels, u32 rate) {
    return (sample_type == static_cast<u32>(AudioCore::SampleFormat::PcmInt16) ||
            sample_type == static_cast<u32>(AudioCore::SampleFormat::PcmFloat)) &&
           channels == 2 && rate == 48000;
}

/// Accept 48 kHz S16 or F32 microphone capture with one or two guest channels.
bool SysmixOutput::IsInputFormatSupported(u32 sample_type, u32 channels, u32 rate) {
    return (channels == 1 || channels == 2) && IsFormatSupported(sample_type, 2, rate);
}

bool SysmixOutput::IsInput() const {
    return impl->input_stream;
}

/// Validate the ring layout and open the host stream, leaving it disengaged until requested.
Result SysmixOutput::Open(u32 sample_type, u32 channels, u32 rate, u32 frames, u32 prefill,
                          u32 flags) {
    auto& p = *impl;
    std::scoped_lock lock{p.mutex};
    R_UNLESS(!p.open, ResultOperationFailed);
    R_UNLESS((p.input_stream ? IsInputFormatSupported(sample_type, channels, rate)
                             : IsFormatSupported(sample_type, channels, rate)) &&
                 prefill == 0 && flags == 0,
             ResultNotSupported);
    R_UNLESS(frames > 0 && frames < static_cast<u32>(std::numeric_limits<s32>::max()),
             ResultInvalidArgument);
    R_TRY(ValidateRegion(*p.process, p.buffer, p.size, p.creation_flags, p.input_stream));
    const u64 slots = u64{frames} + 1;
    const auto format = static_cast<AudioCore::SampleFormat>(sample_type);
    const u32 frame_bytes =
        channels * (format == AudioCore::SampleFormat::PcmInt16 ? sizeof(s16) : sizeof(f32));
    const u64 timestamp_bytes = p.input_stream ? 0 : std::min<u64>(16, slots) * 8;
    const u64 minimum = AlignUp(0x1000 + AlignUp(0x40 + timestamp_bytes, 64) +
                                    AlignUp(slots * frame_bytes, 64),
                                0x1000);
    R_UNLESS(p.size >= minimum && PayloadOffset + slots * frame_bytes <= p.size,
             ResultInsufficientBuffer);
    R_UNLESS(p.state_event && p.data_event, ResultOutOfSessions);
    u32 read{}, write{};
    auto& memory = p.process->GetMemory();
    R_UNLESS(memory.Read32Acquire(p.buffer + ReadOffset, read) &&
                 memory.Read32Acquire(p.buffer + WriteOffset, write) && read < slots &&
                 write < slots,
             ResultInvalidAddressInfo);
    if (p.input_stream && p.loopback_input) {
        p.UseSilentInput(channels, false);
    } else {
#if defined(__APPLE__) && TARGET_OS_IPHONE
        if (p.input_stream) {
            p.UseSilentInput(channels);
        } else {
            p.host = p.sink.AcquireSinkStream(p.system, channels, "sysmix-" + std::to_string(p.id),
                                              AudioCore::Sink::StreamType::Sysmix);
        }
#else
        p.host = p.sink.AcquireSinkStream(p.system, channels, "sysmix-" + std::to_string(p.id),
                                          p.input_stream ? AudioCore::Sink::StreamType::In
                                                         : AudioCore::Sink::StreamType::Sysmix);
        if (p.input_stream && (!p.host || !p.host->IsReady())) {
            p.UseSilentInput(channels);
        }
#endif
    }
    if (!p.host || !p.host->IsReady()) {
        p.ReleaseHost();
        R_RETURN(ResultOperationFailed);
    }
    p.slots = static_cast<u32>(slots);
    p.format = format;
    p.frame_bytes = frame_bytes;
    p.consumer = p.input_stream ? write : read;
    p.channels = channels;
    p.faulted = false;
    p.open = true;
    p.engaged = false;
    p.gain = 1.0f;
    p.counters = {};
    p.state_event->Signal(p.system.Kernel());
    p.Notify(static_cast<u32>((write + slots - read) % slots));
    LOG_INFO(Service_Audio,
             "sysmix stream {} host ready: input={} sample type {} channels={} 48000 slots={} "
             "payload={:#x}",
             p.id, p.input_stream, sample_type, channels, p.slots, PayloadOffset);
    R_SUCCEED();
}

void SysmixOutput::Close() {
    impl->Close();
}

Result SysmixOutput::SetEngaged(bool engaged) {
    auto& p = *impl;
    std::scoped_lock lock{p.mutex};
    R_UNLESS(p.open && !p.faulted && p.host && p.host->IsReady(), ResultOperationFailed);
    if (p.engaged == engaged) {
        R_SUCCEED();
    }
    if (engaged) {
        if (p.input_stream && !p.loopback_input &&
            IsVirtualOutputLoopbackIpc(p.system.Kernel(), *p.process, 700u, p.buffer, p.size,
                                       p.creation_flags, p.id)) {
            p.loopback_input = true;
            p.UseSilentInput(p.channels, false);
        }
        p.host->Start();
        if (p.input_stream && (!p.host->IsReady() || p.host->IsPaused())) {
            p.UseSilentInput(p.channels);
            p.host->Start();
        }
        R_UNLESS(p.host->IsReady() && !p.host->IsPaused(), ResultOperationFailed);
    } else {
        p.host->Stop();
        R_UNLESS(p.host->IsReady(), p.Fault(ResultOperationFailed));
    }
    p.engaged = engaged;
    p.state_event->Signal(p.system.Kernel());
    R_SUCCEED();
}

Result SysmixOutput::SetDataNotification(u32 mode, s32 threshold) {
    auto& p = *impl;
    std::scoped_lock lock{p.mutex};
    R_UNLESS(mode == 0 || mode == 2, ResultNotSupported);
    R_UNLESS(p.open && !p.faulted && threshold >= 0 && static_cast<u32>(threshold) < p.slots,
             ResultInvalidArgument);
    u32 read{}, write{}, filled{};
    R_UNLESS(p.ReadOccupancy(read, write, filled), p.Fault());
    p.data_mode = mode;
    p.threshold = threshold;
    p.Notify(filled);
    R_SUCCEED();
}

/**
 * Discard up to frames of buffered input; zero also drains pending host capture.
 * Cursor contention is retried, then returns OperationFailed with purged zero while capture stays
 * engaged. Invalid memory or cursor ownership still faults the stream.
 */
Result SysmixOutput::Purge(u32 frames, u32& purged) {
    auto& p = *impl;
    std::scoped_lock lock{p.mutex};
    purged = 0;
    R_UNLESS(p.input_stream && p.open && !p.faulted, ResultOperationFailed);
    auto& memory = p.process->GetMemory();
    u32 read{}, write{}, filled{};
    // The guest owns the read cursor and may advance it while this call is in flight.
    for (u32 attempt = 0; attempt < 8; ++attempt) {
        R_UNLESS(p.ReadOccupancy(read, write, filled), p.Fault());
        const auto count = frames == 0 ? filled : std::min(frames, filled);
        if (!memory.WriteExclusive32(p.buffer + ReadOffset, (read + count) % p.slots, read)) {
            continue;
        }
        R_UNLESS(p.ReadOccupancy(read, write, filled), p.Fault());
        purged = count;
        if (frames == 0) {
            p.host->DiscardInputSamples();
        }
        p.Notify(filled);
        R_SUCCEED();
    }
    R_UNLESS(p.ReadOccupancy(read, write, filled), p.Fault());
    R_RETURN(ResultOperationFailed);
}

Result SysmixOutput::SetGain(f32 gain, s64 ramp_ns) {
    R_UNLESS(ramp_ns == 0, ResultNotSupported);
    R_UNLESS(std::isfinite(gain) && gain >= 0.0f && gain <= 128.0f, ResultInvalidArgument);
    std::scoped_lock lock{impl->mutex};
    impl->gain = gain;
    R_SUCCEED();
}

f32 SysmixOutput::GetGain() const {
    std::scoped_lock lock{impl->mutex};
    return impl->gain;
}

u32 SysmixOutput::GetState() const {
    std::scoped_lock lock{impl->mutex};
    return impl->open && !impl->faulted ? (impl->engaged ? 2 : 1) : 0;
}

/// Transfer bounded PCM between host audio and the guest ring, preserving cursor ownership.
void SysmixOutput::Pump() {
    auto& p = *impl;
    std::unique_lock lock{p.mutex, std::try_to_lock};
    if (!lock || !p.open || p.faulted || p.system.IsShuttingDown()) {
        return;
    }
    ++p.counters.ticks;
    u32 read{}, write{}, filled{};
    if (!p.ReadOccupancy(read, write, filled)) {
        p.Fault();
        return;
    }
    if (!p.engaged) {
        p.Notify(filled);
        return;
    }
    if (!p.host->IsReady()) {
        p.Fault(ResultOperationFailed);
        return;
    }
    if (p.input_stream) {
        const auto available = std::min(p.slots - 1 - filled, BlockFrames);
        if (!available) {
            ++p.counters.backpressure;
            p.Notify(filled);
            return;
        }
        const auto taken = static_cast<u32>(p.host->DrainInputSamples(p.output, available));
        if (!taken) {
            p.Notify(filled);
            return;
        }
        u32 nonzero{};
        for (u32 frame = 0; frame < taken; ++frame) {
            bool audible{};
            for (u32 channel = 0; channel < p.channels; ++channel) {
                const auto index = frame * p.channels + channel;
                const auto value =
                    std::clamp(static_cast<f32>(p.output[index]) * p.gain, -32768.0f, 32767.0f);
                audible |= value != 0;
                if (p.format == AudioCore::SampleFormat::PcmInt16) {
                    const auto sample = static_cast<s16>(value);
                    std::memcpy(p.input.data() + index * sizeof(sample), &sample, sizeof(sample));
                } else {
                    const auto sample = value / 32768.0f;
                    std::memcpy(p.input.data() + index * sizeof(sample), &sample, sizeof(sample));
                }
            }
            nonzero += audible;
        }
        const auto first = std::min(taken, p.slots - write);
        const auto second = taken - first;
        auto& memory = p.process->GetMemory();
        if (!memory.WriteBlock(p.buffer + PayloadOffset + u64{write} * p.frame_bytes,
                               p.input.data(), first * p.frame_bytes) ||
            (second &&
             !memory.WriteBlock(p.buffer + PayloadOffset, p.input.data() + first * p.frame_bytes,
                                second * p.frame_bytes)) ||
            !memory.WriteExclusive32(p.buffer + WriteOffset, (write + taken) % p.slots, write)) {
            p.Fault();
            return;
        }
        p.consumer = (write + taken) % p.slots;
        p.counters.copied_frames += taken;
        p.counters.accepted_frames += taken;
        p.counters.nonzero_frames += nonzero;
        ++p.counters.admitted_blocks;
        if (!p.ReadOccupancy(read, write, filled)) {
            p.Fault();
            return;
        }
        p.Notify(filled);
        if (p.counters.admitted_blocks <= 3 || p.counters.ticks % 400 == 0) {
            LOG_INFO(Service_Audio, "sysmix input {}: captured={} nonzero={} filled={}", p.id,
                     p.counters.accepted_frames, p.counters.nonzero_frames, filled);
        }
        return;
    }
    // Host callbacks need several guest blocks at once (for example 1024 versus 240 frames).
    // Once the host has requested frames, refill the bounded host queue from available whole
    // blocks so submillisecond timer jitter cannot insert silence while PCM is waiting in the
    // guest ring. Before a host callback runs (including discard sinks with no callback), admit
    // one block per tick to preserve guest PCM pacing.
    const bool refill_headroom = p.host->HasHostCallback();
    const u32 queue_limit = refill_headroom ? p.host->SysmixQueueLimit() : 1U;
    const u32 queued = refill_headroom ? p.host->GetQueueSize() : 0U;
    if (refill_headroom && queued >= queue_limit) {
        ++p.counters.backpressure;
        if (p.counters.ticks % 400 == 0) {
            p.LogStatus(filled);
        }
        return;
    }
    bool log_status = p.counters.admitted_blocks < 3 || p.counters.ticks % 400 == 0;
    for (u32 block_index = queued; block_index < queue_limit; ++block_index) {
        const u32 taken = std::min(filled, BlockFrames);
        // A short block is padded with silence. That is right for a real underrun, but while the
        // host still has audio to play the guest is only a little behind this tick, and padding
        // would cut a gap into continuous audio. Wait for a whole block until the host drains, so a
        // guest that waits on fewer than BlockFrames frames (a high notification threshold) still
        // makes progress. The queue size alone is not enough: it is zero while the last block is
        // still playing.
        if (taken < BlockFrames && p.host->HasPendingAudio()) {
            ++p.counters.deferred;
            p.Notify(filled);
            return;
        }
        const u32 first = std::min(taken, p.slots - read);
        const u32 second = taken - first;
        auto& memory = p.process->GetMemory();
        if ((first && !memory.ReadBlock(p.buffer + PayloadOffset + u64{read} * p.frame_bytes,
                                        p.input.data(), first * p.frame_bytes)) ||
            (second &&
             !memory.ReadBlock(p.buffer + PayloadOffset, p.input.data() + first * p.frame_bytes,
                               second * p.frame_bytes))) {
            p.Fault();
            return;
        }
        p.counters.copied_frames += taken;
        p.output.fill(0);
        u32 nonzero{};
        for (u32 frame = 0; frame < taken; ++frame) {
            bool audible{};
            for (u32 channel = 0; channel < 2; ++channel) {
                const u32 index = frame * 2 + channel;
                f32 value;
                if (p.format == AudioCore::SampleFormat::PcmInt16) {
                    s16 sample;
                    std::memcpy(&sample, p.input.data() + index * sizeof(sample), sizeof(sample));
                    value = static_cast<f32>(sample) / 32768.0f;
                } else {
                    std::memcpy(&value, p.input.data() + index * sizeof(value), sizeof(value));
                }
                if (std::isfinite(value)) {
                    const auto sample =
                        static_cast<s32>(std::clamp(value * p.gain, -1.0f, 1.0f) * 32768.0f);
                    p.output[index] = static_cast<s16>(std::clamp(sample, -32768, 32767));
                    audible |= p.output[index] != 0;
                }
            }
            nonzero += audible;
        }
        AudioCore::Sink::SinkBuffer block{BlockFrames, 0, p.counters.admitted_blocks, false};
        if (!p.host->TryAppendOutputBlock(block, p.output)) {
            if (!p.host->IsReady()) {
                p.Fault(ResultOperationFailed);
                return;
            }
            ++p.counters.backpressure;
            p.counters.retried_frames += taken;
            if (p.counters.backpressure <= 3) {
                LOG_INFO(Service_Audio, "sysmix stream {} host full: accepted={} blocks={}", p.id,
                         p.counters.accepted_frames, p.counters.admitted_blocks);
            }
            if (p.counters.ticks % 400 == 0) {
                p.LogStatus(filled);
            }
            return;
        }
        ++p.counters.admitted_blocks;
        if (taken < BlockFrames) {
            p.counters.padded_frames += BlockFrames - taken;
        }
        const u32 next = (read + taken) % p.slots;
        u32 checked{};
        if (!memory.Read32Acquire(p.buffer + ReadOffset, checked) || checked != read ||
            !memory.WriteExclusive32(p.buffer + ReadOffset, next, read) ||
            !memory.Read32Acquire(p.buffer + ReadOffset, checked) || checked != next) {
            p.Fault();
            return;
        }
        p.consumer = next;
        p.counters.accepted_frames += taken;
        p.counters.nonzero_frames += nonzero;
        if (!p.ReadOccupancy(read, write, filled)) {
            p.Fault();
            return;
        }
        p.Notify(filled);
        log_status |= nonzero && p.counters.nonzero_frames == nonzero;
        // Never pad another block merely to fill host headroom. Leave a short guest tail for the
        // next tick; the existing empty-host path above can still pad a genuine underrun.
        if (filled < BlockFrames) {
            break;
        }
    }
    if (log_status) {
        p.LogStatus(filled);
    }
}

Kernel::KReadableEvent* SysmixOutput::GetStateEvent() {
    return impl->state_event ? &impl->state_event->GetReadableEvent() : nullptr;
}

Kernel::KReadableEvent* SysmixOutput::GetDataEvent() {
    return impl->data_event ? &impl->data_event->GetReadableEvent() : nullptr;
}

SysmixOutputCounters SysmixOutput::GetCounters() const {
    std::scoped_lock lock{impl->mutex};
    return impl->counters;
}

bool SysmixOutput::IsOwnedBy(const Kernel::KProcess* process) const {
    return impl->process.GetPointerUnsafe() == process;
}

} // namespace Service::Audio
