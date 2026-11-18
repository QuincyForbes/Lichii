// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>

#include <fmt/format.h>

#include "common/cityhash.h"
#include "common/logging.h"
#include "common/stb.h"
#include "common/zstd_compression.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/renderer_metal/metal_formats.h"
#include "video_core/renderer_metal/metal_frame_dump.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/texture_cache/formatter.h"

namespace Metal {

namespace {

using VideoCore::Surface::PixelFormat;

constexpr size_t SIZE_CAP = size_t{2} << 30;
constexpr size_t STAGING_BUDGET = size_t{256} << 20;

template <typename T>
T Load(const u8* pixel, int index) {
    T value;
    std::memcpy(&value, pixel + index * sizeof(T), sizeof(T));
    return value;
}

template <typename T, typename Convert>
int Channels(const u8* pixel, int count, std::array<float, 4>& out, Convert&& convert) {
    for (int channel = 0; channel < count; ++channel) {
        out[channel] = convert(Load<T>(pixel, channel));
    }
    return count;
}

/// Unsigned float with a 5-bit exponent, as in half, B10G11R11 and friends.
float PackedFloat(u32 bits, int mantissa_bits) {
    const u32 exponent = bits >> mantissa_bits;
    const u32 mantissa = bits & ((1U << mantissa_bits) - 1);
    if (exponent == 0) {
        return std::ldexp(static_cast<float>(mantissa), -14 - mantissa_bits);
    }
    if (exponent == 31) {
        return mantissa != 0 ? std::numeric_limits<float>::quiet_NaN()
                             : std::numeric_limits<float>::infinity();
    }
    return std::ldexp(static_cast<float>(mantissa | (1U << mantissa_bits)),
                      static_cast<int>(exponent) - 15 - mantissa_bits);
}

float Half(u16 bits) {
    const float magnitude = PackedFloat(bits & 0x7fff, 10);
    return (bits & 0x8000) != 0 ? -magnitude : magnitude;
}

float Unorm8(u8 value) {
    return value / 255.0f;
}

float Unorm16(u16 value) {
    return value / 65535.0f;
}

float Identity(float value) {
    return value;
}

/// The same decode as Vulkan's frame dump: the number of channels, or 0 without a preview.
int DecodePixel(PixelFormat format, const u8* pixel, std::array<float, 4>& out) {
    out = {0.0f, 0.0f, 0.0f, 1.0f};
    switch (format) {
    case PixelFormat::A8B8G8R8_UNORM:
    case PixelFormat::A8B8G8R8_SRGB:
        return Channels<u8>(pixel, 4, out, Unorm8);
    case PixelFormat::B8G8R8A8_UNORM:
    case PixelFormat::B8G8R8A8_SRGB:
        out = {Unorm8(pixel[2]), Unorm8(pixel[1]), Unorm8(pixel[0]), Unorm8(pixel[3])};
        return 4;
    case PixelFormat::R8_UNORM:
        return Channels<u8>(pixel, 1, out, Unorm8);
    case PixelFormat::R8G8_UNORM:
        return Channels<u8>(pixel, 2, out, Unorm8);
    case PixelFormat::R16_UNORM:
        return Channels<u16>(pixel, 1, out, Unorm16);
    case PixelFormat::R16G16_UNORM:
        return Channels<u16>(pixel, 2, out, Unorm16);
    case PixelFormat::R16G16B16A16_UNORM:
        return Channels<u16>(pixel, 4, out, Unorm16);
    case PixelFormat::R16_FLOAT:
        return Channels<u16>(pixel, 1, out, Half);
    case PixelFormat::R16G16_FLOAT:
        return Channels<u16>(pixel, 2, out, Half);
    case PixelFormat::R16G16B16A16_FLOAT:
        return Channels<u16>(pixel, 4, out, Half);
    case PixelFormat::R32_FLOAT:
        return Channels<float>(pixel, 1, out, Identity);
    case PixelFormat::R32G32_FLOAT:
        return Channels<float>(pixel, 2, out, Identity);
    case PixelFormat::R32G32B32A32_FLOAT:
        return Channels<float>(pixel, 4, out, Identity);
    case PixelFormat::B10G11R11_FLOAT: {
        const u32 value = Load<u32>(pixel, 0);
        out = {PackedFloat(value & 0x7ff, 6), PackedFloat((value >> 11) & 0x7ff, 6),
               PackedFloat(value >> 22, 5), 1.0f};
        return 3;
    }
    case PixelFormat::A2B10G10R10_UNORM: {
        const u32 value = Load<u32>(pixel, 0);
        out = {(value & 0x3ff) / 1023.0f, ((value >> 10) & 0x3ff) / 1023.0f,
               ((value >> 20) & 0x3ff) / 1023.0f, (value >> 30) / 3.0f};
        return 4;
    }
    default:
        return 0;
    }
}

bool IsFloat(PixelFormat format) {
    switch (format) {
    case PixelFormat::R16_FLOAT:
    case PixelFormat::R16G16_FLOAT:
    case PixelFormat::R16G16B16A16_FLOAT:
    case PixelFormat::R32_FLOAT:
    case PixelFormat::R32G32_FLOAT:
    case PixelFormat::R32G32B32A32_FLOAT:
    case PixelFormat::B10G11R11_FLOAT:
        return true;
    default:
        return false;
    }
}

/// Float targets hold linear values, so the preview encodes them to sRGB; UNORM bytes are kept.
u8 PreviewByte(float value, bool encode) {
    if (!(value > 0.0f)) {
        return 0;
    }
    if (value >= 1.0f) {
        return 255;
    }
    if (encode) {
        value = value <= 0.0031308f ? value * 12.92f
                                    : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
    }
    return static_cast<u8>(value * 255.0f + 0.5f);
}

std::string JsonNumber(double value) {
    return std::isfinite(value) ? fmt::format("{:.6g}", value) : "null";
}

template <typename T>
std::string JsonArray(std::span<T> values) {
    std::string out{"["};
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out += ',';
        }
        if constexpr (std::is_floating_point_v<T>) {
            out += JsonNumber(values[i]);
        } else {
            out += fmt::format("{}", values[i]);
        }
    }
    return out + "]";
}

} // Anonymous namespace

u32 FrameDump::HostTexelBytes(u32 format) noexcept {
    if (format >= 10 && format <= 14) {
        return 1;
    }
    if ((format >= 20 && format <= 25) || (format >= 30 && format <= 34) ||
        (format >= 40 && format <= 43)) {
        return 2;
    }
    if ((format >= 53 && format <= 55) || (format >= 60 && format <= 65) ||
        (format >= 70 && format <= 74) || (format >= 80 && format <= 81) ||
        (format >= 90 && format <= 94)) {
        return 4;
    }
    if ((format >= 103 && format <= 105) || (format >= 110 && format <= 115)) {
        return 8;
    }
    if (format >= 123 && format <= 125) {
        return 16;
    }
    return 0;
}

FrameDump::FrameDump(const Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_} {
    const char* const dir = std::getenv("EDEN_NX2_FRAME_DUMP");
    size_cap = SIZE_CAP;
    if (const char* const cap = std::getenv("EDEN_NX2_FRAME_DUMP_CAP_MIB")) {
        if (const u64 mib = std::strtoull(cap, nullptr, 10); mib != 0) {
            size_cap = static_cast<size_t>(mib) << 20;
        }
    }
    start = std::chrono::steady_clock::now();
    if (!dir || *dir == '\0') {
        return;
    }
    enabled = true;
    if (const char* const value = std::getenv("EDEN_NX2_FRAME_DUMP_SAMPLED");
        value != nullptr && *value != '\0' && *value != '0') {
        env_sampled = sampled = buffers = true;
    }
    if (const char* const from = std::getenv("EDEN_NX2_FRAME_DUMP_SAMPLED_FROM")) {
        sampled_from_pass = static_cast<u32>(std::strtoul(from, nullptr, 10));
    }
    directory = dir;
    base_directory = directory;
    env_directory = directory;
    const char* const at = std::getenv("EDEN_NX2_FRAME_DUMP_AT");
    const std::string_view value = at ? at : "60";
    if (const char* const pair_value = std::getenv("EDEN_NX2_FRAME_DUMP_PAIR")) {
        pair = *pair_value != '\0' && *pair_value != '0';
    }
    if (value == "file") {
        // Captures once <dir>/capture.now exists, so a frame can be picked while playing.
        trigger_file = directory / "capture.now";
        trigger_seconds = std::numeric_limits<double>::infinity();
    } else if (value.ends_with('f')) {
        trigger_frame = std::strtoull(value.data(), nullptr, 10);
        trigger_seconds = std::numeric_limits<double>::infinity();
    } else {
        trigger_seconds = std::strtod(value.data(), nullptr);
    }
    LOG_INFO(Render_Metal, "Frame dump armed: dir={} at={}", directory.string(), value);
}

FrameDump::~FrameDump() {
    StopWriter();
}

void FrameDump::StopWriter() {
    if (!writer.joinable()) {
        return;
    }
    {
        std::scoped_lock lock{queue_mutex};
        stop_writer = true;
    }
    queue_ready.notify_one();
    writer.join();
    stop_writer = false;
}

void FrameDump::Arm(u64 id, std::filesystem::path capture_directory, bool with_sampled,
                    bool with_buffers) {
    std::scoped_lock lock{request_mutex};
    request_id = id;
    request_sampled = with_sampled;
    request_buffers = with_buffers;
    request_directory = std::move(capture_directory);
    requested = true;
}

bool FrameDump::BeginPass(const void* key, std::span<const Source> colors) {
    if (key == pass_key) {
        return false;
    }
    const bool recorded = EndPass();
    pass_key = key;
    pass_sources.assign(colors.begin(), colors.end());
    pass_hashes.clear();
    pass_sampled.clear();
    pass_null_sampled.clear();
    pass_first_draw = draws;
    return recorded;
}

void FrameDump::NoteDraw(std::span<const u64> shader_hashes) {
    ++draws;
    for (const u64 hash : shader_hashes) {
        if (hash != 0 && pass_hashes.size() < 32 &&
            std::ranges::find(pass_hashes, hash) == pass_hashes.end()) {
            pass_hashes.push_back(hash);
        }
    }
}

void FrameDump::NoteSkipped(bool compute, u64 program, GPUVAddr rt0, std::string_view reason) {
    std::string text{reason};
    std::ranges::replace(text, '"', '\'');
    std::ranges::replace(text, '\\', '/');
    Enqueue({.tags = fmt::format(
                 R"({{"kind":"skipped","compute":{},"after_draws":{},"after_dispatches":{},"program":"{:#x}","rt0":"{:#x}","reason":"{}"}})",
                 compute, draws, dispatches, program, rt0, text)});
}

bool FrameDump::EndPass() {
    if (!pass_key) {
        return false;
    }
    pass_key = nullptr;
    std::string shaders;
    for (const u64 hash : pass_hashes) {
        shaders += fmt::format("{}\"{:016x}\"", shaders.empty() ? "" : ",", hash);
    }
    bool recorded = false;
    for (const Source& source : pass_sources) {
        recorded |= Record(
            source,
            fmt::format("{:04}_pass{:03}_d{}-{}_rt{}", sequence++, pass_index, pass_first_draw,
                        draws, source.slot),
            fmt::format(R"("kind":"pass","pass":{},"first_draw":{},"draw_count":{},"slot":{},"shaders":[{}])",
                        pass_index, pass_first_draw, draws - pass_first_draw, source.slot,
                        shaders));
    }
    ++pass_index;
    return recorded;
}

void FrameDump::CaptureDispatch(u64 shader_hash, std::span<const Source> written) {
    const u32 dispatch = dispatches++;
    for (const Source& source : written) {
        Record(source,
               fmt::format("{:04}_dispatch{:04}_{:016x}_img{}", sequence++, dispatch, shader_hash,
                           source.slot),
               fmt::format(R"("kind":"dispatch","dispatch":{},"after_draws":{},"slot":{},"shaders":["{:016x}"])",
                           dispatch, draws, source.slot, shader_hash));
    }
}

void FrameDump::CapturePresent(const Source& source) {
    Record(source, fmt::format("{:04}_present", sequence++),
           fmt::format(R"("kind":"present","after_draws":{})", draws));
}

bool FrameDump::Record(const Source& source, std::string name, std::string tags) {
    name += fmt::format("_{}_{}x{}_{:x}", source.format, source.width, source.height,
                        source.gpu_addr);
    const u32 texel = HostTexelBytes(source.host_format);
    if (source.msaa || texel == 0 || source.texture == nullptr || source.width == 0 ||
        source.height == 0) {
        Enqueue({.tags = fmt::format(R"({{"name":"{}",{},"skipped":"{}"}})", name, tags,
                                     source.msaa ? "msaa" : "format")});
        return false;
    }
    const size_t size = size_t{source.width} * source.height * texel;
    if (recorded_bytes + size > size_cap) {
        truncated = true;
        return false;
    }
    u8* contents = nullptr;
    Object buffer = CreateBuffer(device, size, true, &contents);
    if (!buffer || contents == nullptr) {
        return false;
    }
    recorded_bytes += size;
    scheduler.CopyTextureToBuffer(source.texture, source.level, source.layer, {}, {source.width, source.height, 1},
                                  buffer.Get(), 0, source.width * texel,
                                  static_cast<u32>(size));
    pending_bytes += size;
    readbacks.push_back({.raw = false,
                         .buffer = std::move(buffer),
                         .contents = contents,
                         .source = source,
                         .name = std::move(name),
                         .tags = std::move(tags),
                         .bytes = size});
    if (pending_bytes >= STAGING_BUDGET) {
        Drain();
    }
    return true;
}

bool FrameDump::RecordSampled(const Sampled& texture, u32 index, std::string name,
                              std::string tags) {
    tags += texture.sampler;
    if (texture.source && texture.source->host_format != 0 && !texture.block_compressed) {
        Source source = *texture.source;
        source.slot = index;
        return Record(source, std::move(name), std::move(tags));
    }
    Enqueue({.tags = fmt::format(
                 R"({{"name":"{}",{},"format":"{}","width":{},"height":{},"gpu_addr":"{:#x}","skipped":"format"}})",
                 name, tags, texture.source ? texture.source->format : VideoCore::Surface::PixelFormat{},
                 texture.width, texture.height, texture.key_addr)});
    return false;
}

bool FrameDump::NoteSampled(std::span<const Sampled> textures) {
    bool recorded = false;
    if (!sampled || pass_index < sampled_from_pass) {
        return false;
    }
    for (const Sampled& texture : textures) {
        if (texture.null) {
            const std::pair stage_tic{texture.stage, texture.tic};
            if (std::ranges::find(pass_null_sampled, stage_tic) == pass_null_sampled.end()) {
                pass_null_sampled.push_back(stage_tic);
                Enqueue({.tags = fmt::format(
                             R"({{"kind":"sampled","pass":{},"draw":{},"stage":{},"tic":{},"null":true}})",
                             pass_index, draws - 1, texture.stage, texture.tic)});
            }
            continue;
        }
        const std::pair key{texture.key_addr, texture.key_extra};
        if (std::ranges::find(pass_sampled, key) != pass_sampled.end()) {
            continue;
        }
        const u32 index = static_cast<u32>(pass_sampled.size());
        pass_sampled.push_back(key);
        recorded |= RecordSampled(
            texture, index,
            fmt::format("{:04}_pass{:03}_tex{:02}_s{}_tic{}", sequence++, pass_index, index,
                        texture.stage, texture.tic),
            fmt::format(R"("kind":"sampled","pass":{},"draw":{},"stage":{},"tic":{},"index":{})",
                        pass_index, draws - 1, texture.stage, texture.tic, index));
    }
    return recorded;
}

void FrameDump::NoteDispatchSampled(u64 shader_hash, std::span<const Sampled> textures) {
    if (!sampled || pass_index < sampled_from_pass) {
        return;
    }
    for (u32 index = 0; index < textures.size(); ++index) {
        const Sampled& texture = textures[index];
        if (texture.null) {
            continue;
        }
        const std::tuple<u64, GPUVAddr, u32> seen{shader_hash, texture.key_addr, texture.key_extra};
        if (std::ranges::find(dispatch_sampled, seen) != dispatch_sampled.end()) {
            continue;
        }
        dispatch_sampled.push_back(seen);
        RecordSampled(
            texture, index,
            fmt::format("{:04}_dispatch{:04}_{:016x}_tex{:02}_tic{}", sequence++, dispatches,
                        shader_hash, index, texture.tic),
            fmt::format(
                R"("kind":"sampled","dispatch":{},"after_draws":{},"stage":{},"tic":{},"index":{},"shaders":["{:016x}"])",
                dispatches, draws, texture.stage, texture.tic, index, shader_hash));
    }
}

bool FrameDump::RecordBuffer(BufferHandle source, u32 offset, u32 size, std::string tags) {
    if (recorded_bytes + size > size_cap) {
        truncated = true;
        return false;
    }
    u8* contents = nullptr;
    Object buffer = CreateBuffer(device, size, true, &contents);
    if (!buffer || contents == nullptr) {
        return false;
    }
    recorded_bytes += size;
    scheduler.CopyBuffer(source, offset, buffer.Get(), 0, size);
    pending_bytes += size;
    readbacks.push_back({.raw = true,
                         .buffer = std::move(buffer),
                         .contents = contents,
                         .source = {},
                         .name = {},
                         .tags = std::move(tags),
                         .bytes = size});
    return true;
}

bool FrameDump::NoteBuffers(bool compute, std::span<const u64> shader_hashes,
                            std::span<BoundBuffer> bound) {
    if (!buffers) {
        return false;
    }
    bool recorded = false;
    const auto owner = [&] {
        return compute ? fmt::format(R"("dispatch":{},"after_draws":{})", dispatches, draws)
                       : fmt::format(R"("pass":{},"draw":{})", pass_index, draws - 1);
    };
    for (BoundBuffer& buffer : bound) {
        const size_t program = compute ? 0 : buffer.stage + 1;
        const u64 shader = program < shader_hashes.size() ? shader_hashes[program] : 0;
        const std::tuple<u64, bool, u32, u64, u32> key{shader, buffer.storage, buffer.index,
                                                       buffer.device_addr, buffer.size};
        if (std::ranges::find(seen_buffers, key) != seen_buffers.end()) {
            continue;
        }
        seen_buffers.push_back(key);
        const bool from_host = buffer.guest.empty();
        std::string tags = fmt::format(
            R"("kind":"buffer",{},"stage":{},"shader":"{:016x}","type":"{}","index":{},"written":{},"gpu_addr":"{:#x}","device_addr":"{:#x}","bound_size":{},"size":{},"gpu_modified":{},"source":"{}")",
            owner(), buffer.stage, shader, buffer.storage ? "storage" : "uniform", buffer.index,
            buffer.written, buffer.gpu_addr, buffer.device_addr, buffer.bound_size, buffer.size,
            buffer.gpu_modified, from_host ? "host" : "guest");
        if (!from_host) {
            Enqueue({.tags = std::move(tags), .data = std::move(buffer.guest), .raw = true});
            continue;
        }
        recorded |= RecordBuffer(buffer.host, buffer.host_offset, buffer.size, std::move(tags));
    }
    for (const BoundBuffer& buffer : bound) {
        if (!buffer.also_host || buffer.host == nullptr) {
            continue;
        }
        const size_t program = compute ? 0 : buffer.stage + 1;
        const u64 shader = program < shader_hashes.size() ? shader_hashes[program] : 0;
        recorded |= RecordBuffer(
            buffer.host, buffer.host_offset, buffer.size,
            fmt::format(
                R"("kind":"buffer",{},"stage":{},"shader":"{:016x}","type":"{}","index":{},"written":{},"gpu_addr":"{:#x}","device_addr":"{:#x}","bound_size":{},"size":{},"gpu_modified":{},"source":"host_copy")",
                owner(), buffer.stage, shader, buffer.storage ? "storage" : "uniform",
                buffer.index, buffer.written, buffer.gpu_addr, buffer.device_addr,
                buffer.bound_size, buffer.size, buffer.gpu_modified));
    }
    return recorded;
}

void FrameDump::NoteLaunch(u64 shader_hash, const std::array<u32, 3>& grid,
                           const std::array<u32, 3>& block, std::optional<GPUVAddr> indirect,
                           std::optional<std::array<u32, 3>> indirect_grid) {
    std::string tags = fmt::format(
        R"({{"kind":"launch","dispatch":{},"after_draws":{},"shaders":["{:016x}"],"grid":[{},{},{}],"block":[{},{},{}])",
        dispatches, draws, shader_hash, grid[0], grid[1], grid[2], block[0], block[1], block[2]);
    if (indirect) {
        tags += fmt::format(R"(,"indirect":"{:#x}")", *indirect);
    }
    if (indirect_grid) {
        tags += fmt::format(R"(,"indirect_grid_guest":[{},{},{}])", (*indirect_grid)[0],
                            (*indirect_grid)[1], (*indirect_grid)[2]);
    }
    Enqueue({.tags = tags + "}"});
}

void FrameDump::NoteDrawRecord(const DrawRecord& draw) {
    // Metal has no pipeline statistics: the primitive and fragment counts are null.
    Enqueue({.tags = fmt::format(
                 R"({{"kind":"draw","draw":{},"primitives_in":null,"primitives":null,"fragments":null,"mesh":{},"indirect":{},"first":{},"elements":{},"instances":{},"alpha_test":{},"alpha_ref":{},"alpha_to_coverage":{}}})",
                 draws - 1, draw.mesh, draw.indirect, draw.first, draw.elements, draw.instances,
                 draw.alpha_test, JsonNumber(draw.alpha_ref), draw.alpha_to_coverage)});
}

void FrameDump::WriteBuffer(const Job& job) {
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(job.data.data()),
                                        job.data.size());
    const std::string name = fmt::format("buffer_{:016x}.bin.zst", hash);
    if (std::ranges::find(written_buffers, hash) == written_buffers.end()) {
        written_buffers.push_back(hash);
        WriteFile(name, Common::Compression::CompressDataZSTDDefault(job.data.data(),
                                                                     job.data.size()));
    }
    manifest << fmt::format(R"({{{},"hash":"{:016x}","raw":"{}"}})", job.tags, hash, name) << '\n';
}

void FrameDump::Drain() {
    if (readbacks.empty()) {
        return;
    }
    scheduler.Finish();
    for (Readback& readback : readbacks) {
        Enqueue({.source = readback.source,
                 .name = std::move(readback.name),
                 .tags = std::move(readback.tags),
                 .data = {readback.contents, readback.contents + readback.bytes},
                 .raw = readback.raw});
    }
    readbacks.clear();
    pending_bytes = 0;
}

void FrameDump::Enqueue(Job job) {
    {
        std::scoped_lock lock{queue_mutex};
        queue.push_back(std::move(job));
    }
    queue_ready.notify_one();
}

void FrameDump::WriterLoop() {
    for (;;) {
        Job job;
        {
            std::unique_lock lock{queue_mutex};
            queue_ready.wait(lock, [this] { return !queue.empty() || stop_writer; });
            if (queue.empty()) {
                return;
            }
            job = std::move(queue.front());
            queue.pop_front();
        }
        if (job.summary) {
            manifest << fmt::format(
                            R"({{"kind":"summary","images":{},{},"bytes":{},"truncated":{}}})",
                            images, job.tags, written_bytes, job.truncated)
                     << '\n';
            manifest.close();
            LOG_INFO(Render_Metal, "Frame dump done: {} images, {}, {} MiB in {}{}", images,
                     job.tags, written_bytes >> 20, directory.string(),
                     job.truncated ? fmt::format(" (stopped at the {} MiB cap)", size_cap >> 20)
                                   : "");
            if (job.finish_id != 0) {
                VideoCore::Nx2Debug::FinishCapture(job.finish_id, {});
            }
            continue;
        }
        if (job.raw) {
            WriteBuffer(job);
        } else if (job.name.empty()) {
            manifest << job.tags << '\n';
        } else {
            Write(job);
        }
        manifest.flush();
    }
}

void FrameDump::Write(const Job& job) {
    const Source& source = job.source;
    const u32 bytes_per_pixel = HostTexelBytes(source.host_format);
    const size_t pixels = size_t{source.width} * source.height;
    const std::span<const u8> data = job.data;
    WriteFile(job.name + ".raw.zst",
              Common::Compression::CompressDataZSTDDefault(data.data(), data.size()));
    ++images;

    std::array<float, 4> pixel;
    const int channels = DecodePixel(source.format, data.data(), pixel);
    std::string stats;
    std::string png{"null"};
    if (channels != 0) {
        constexpr double inf = std::numeric_limits<double>::infinity();
        std::array<double, 4> lo{inf, inf, inf, inf};
        std::array<double, 4> hi{-inf, -inf, -inf, -inf};
        std::array<double, 4> sum{};
        std::array<u64, 4> nans{};
        std::array<u64, 4> infs{};
        const bool encode = IsFloat(source.format);
        std::vector<u8> rgb(pixels * 3);
        for (size_t index = 0; index < pixels; ++index) {
            DecodePixel(source.format, data.data() + index * bytes_per_pixel, pixel);
            bool finite = true;
            for (int c = 0; c < channels; ++c) {
                const double value = pixel[c];
                if (std::isnan(value)) {
                    ++nans[c];
                    finite = false;
                } else if (std::isinf(value)) {
                    ++infs[c];
                    finite = false;
                } else {
                    lo[c] = std::min(lo[c], value);
                    hi[c] = std::max(hi[c], value);
                    sum[c] += value;
                }
            }
            u8* const out = &rgb[index * 3];
            if (!finite) {
                out[0] = 255;
                out[1] = 0;
                out[2] = 255;
                continue;
            }
            for (int c = 0; c < 3; ++c) {
                out[c] = PreviewByte(pixel[channels == 1 ? 0 : c], encode);
            }
        }
        std::array<double, 4> mean{};
        for (int c = 0; c < channels; ++c) {
            const u64 finite_count = pixels - nans[c] - infs[c];
            mean[c] = finite_count != 0 ? sum[c] / static_cast<double>(finite_count) : inf;
        }
        const auto first = [channels](auto& values) {
            return std::span(values.data(), static_cast<size_t>(channels));
        };
        stats = fmt::format(R"(,"min":{},"max":{},"mean":{},"nan":{},"inf":{})",
                            JsonArray(first(lo)), JsonArray(first(hi)), JsonArray(first(mean)),
                            JsonArray(first(nans)), JsonArray(first(infs)));
        std::vector<u8> encoded;
        const int ok = stbi_write_png_to_func(
            [](void* context, void* bytes, int size) {
                auto* const out = static_cast<std::vector<u8>*>(context);
                const auto* const begin = static_cast<const u8*>(bytes);
                out->insert(out->end(), begin, begin + size);
            },
            &encoded, static_cast<int>(source.width), static_cast<int>(source.height), 3,
            rgb.data(), static_cast<int>(source.width * 3));
        if (ok != 0) {
            WriteFile(job.name + ".png", encoded);
            png = fmt::format("\"{}.png\"", job.name);
        }
    }
    manifest << fmt::format(
                    R"({{"name":"{}",{},"format":"{}","width":{},"height":{},"depth":1,"level":{},"layer":{},"gpu_addr":"{:#x}","rescaled":{},"host_format":{},"raw":"{}.raw.zst","png":{}{}}})",
                    job.name, job.tags, source.format, source.width, source.height,
                    source.level, source.layer, source.gpu_addr, source.rescaled, source.host_format, job.name, png, stats)
             << '\n';
}

void FrameDump::WriteFile(const std::string& name, std::span<const u8> data) {
    std::ofstream file{directory / name, std::ios::binary};
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size()));
    written_bytes += data.size();
}

void FrameDump::ResetCapture() {
    pending_bytes = recorded_bytes = written_bytes = 0;
    sequence = images = 0;
    truncated = false;
    pass_key = nullptr;
    pass_sources.clear();
    pass_hashes.clear();
    pass_index = pass_first_draw = draws = dispatches = 0;
    pass_sampled.clear();
    pass_null_sampled.clear();
    dispatch_sampled.clear();
    recorded_blocks.clear();
    seen_buffers.clear();
    written_buffers.clear();
    readbacks.clear();
}

void FrameDump::StartCapture(double elapsed) {
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    manifest.open(directory / "manifest.jsonl", std::ios::trunc);
    if (!manifest) {
        LOG_ERROR(Render_Metal, "Frame dump: cannot write {}", directory.string());
        if (active_id != 0) {
            VideoCore::Nx2Debug::FinishCapture(
                active_id, {.error = "cannot write " + directory.string()});
            active_id = 0;
        } else {
            done = true;
        }
        return;
    }
    manifest << fmt::format(R"({{"kind":"frame","frame":{},"seconds":{:.3f}}})", frame, elapsed)
             << '\n';
    writer = std::thread{[this] { WriterLoop(); }};
    capturing = true;
    LOG_INFO(Render_Metal, "Frame dump: capturing frame {} ({:.1f} s after boot) into {}", frame,
             elapsed, directory.string());
}

void FrameDump::TickFrame() {
    if (!enabled && !requested && !capturing) {
        return;
    }
    ++frame;
    if (capturing) {
        EndPass();
        Drain();
        Enqueue({.tags = fmt::format(R"("passes":{},"draws":{},"dispatches":{})", pass_index,
                                     draws, dispatches),
                 .summary = true,
                 .truncated = truncated,
                 .finish_id = active_id});
        capturing = false;
        const bool was_request = active_id != 0;
        active_id = 0;
        // The summary is written (and a request answered) before the writer is stopped.
        StopWriter();
        ResetCapture();
        if (was_request) {
            sampled = buffers = env_sampled;
            directory = env_directory;
        } else if (trigger_file.empty()) {
            done = true;
        } else {
            // File trigger: each new capture.now takes another capture, into capture-2, ...
            std::error_code error;
            std::filesystem::remove(trigger_file, error);
            directory = base_directory / fmt::format("capture-{}", ++captures + 1);
            LOG_INFO(Render_Metal, "Frame dump re-armed: the next capture.now captures into {}",
                     directory.string());
            pair_next = pair && !std::exchange(pair_next, false);
        }
        if (!pair_next && !requested) {
            return;
        }
    }
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (requested) {
        std::scoped_lock lock{request_mutex};
        active_id = request_id;
        sampled = request_sampled || env_sampled;
        buffers = request_buffers || env_sampled;
        directory = request_directory;
        requested = false;
        StartCapture(elapsed);
        return;
    }
    if (!enabled || done) {
        return;
    }
    std::error_code error;
    if (frame < trigger_frame && elapsed < trigger_seconds && !pair_next &&
        (trigger_file.empty() || frame % 30 != 0 ||
         !std::filesystem::exists(trigger_file, error))) {
        return;
    }
    StartCapture(elapsed);
}

} // namespace Metal
