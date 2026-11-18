// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/settings.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/post_processing/fx_chain.h"
#include "video_core/post_processing/fx_effect.h"
#include "video_core/renderer_metal/metal_device.h"
#include "video_core/renderer_metal/metal_formats.h"
#include "video_core/renderer_metal/metal_flash_watch.h"
#include "video_core/renderer_metal/metal_frame_dump.h"
#include "video_core/renderer_metal/metal_frame_gen.h"
#include "video_core/renderer_metal/metal_fsr3_upscaler.h"
#include "video_core/renderer_metal/metal_post_process.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_smoke_present.h"

namespace SmokePresent {

namespace {

using Metal::Object;

/// A BGRA8 or RGBA8 texture filled by `pixel(x, y)` (four bytes in memory order).
template <typename Pixel>
Object Picture(Metal::Device& device, Metal::Scheduler& scheduler, uint32_t format, uint32_t width,
               uint32_t height, Pixel pixel, bool render_target = true) {
    Object texture = Metal::CreateTexture(device, {.kind = Metal::TextureKind::e2D,
                                                   .format = format,
                                                   .width = width,
                                                   .height = height,
                                                   .render_target = render_target});
    uint8_t* bytes = nullptr;
    const Object upload = Metal::CreateBuffer(device, uint64_t{width} * height * 4, true, &bytes);
    if (!texture || bytes == nullptr) {
        return {};
    }
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const std::array<uint8_t, 4> value = pixel(x, y);
            std::memcpy(bytes + (uint64_t{y} * width + x) * 4, value.data(), 4);
        }
    }
    scheduler.CopyBufferToTexture(upload.Get(), 0, width * 4, 0, texture.Get(), 0, 0, {},
                                  {width, height, 1});
    scheduler.Finish();
    return texture;
}

/// `texture` (`texel` bytes per pixel) read back.
std::vector<uint8_t> ReadBack(Metal::Device& device, Metal::Scheduler& scheduler,
                              Metal::TextureHandle texture, uint32_t width, uint32_t height,
                              uint32_t texel) {
    uint8_t* bytes = nullptr;
    const Object buffer =
        Metal::CreateBuffer(device, uint64_t{width} * height * texel, true, &bytes);
    if (bytes == nullptr) {
        return {};
    }
    scheduler.CopyTextureToBuffer(texture, 0, 0, {}, {width, height, 1}, buffer.Get(), 0,
                                  width * texel, 0);
    scheduler.Finish();
    return {bytes, bytes + uint64_t{width} * height * texel};
}

float Half(uint16_t bits) {
    const uint32_t exponent = (bits >> 10) & 31;
    const uint32_t mantissa = bits & 1023;
    float value;
    if (exponent == 0) {
        value = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exponent == 31) {
        value = mantissa != 0 ? NAN : INFINITY;
    } else {
        value = std::ldexp(static_cast<float>(mantissa | 1024), static_cast<int>(exponent) - 25);
    }
    return (bits & 0x8000) != 0 ? -value : value;
}

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream file{path, std::ios::binary};
    std::stringstream out;
    out << file.rdbuf();
    return out.str();
}

size_t Count(const std::string& text, const std::string& needle) {
    size_t count = 0;
    for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + 1)) {
        ++count;
    }
    return count;
}

} // Anonymous namespace

/// N08: every bundled effect of dist/post_shaders through the chain (compiled by the shared
/// ReShade front end, translated to MSL, drawn as render passes), then two of them chained.
int RunPostProcess() {
#ifdef HAS_RESHADE
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "lichii-smoke-post-shaders";
    std::filesystem::remove_all(root);
    Common::FS::SetEdenPath(Common::FS::EdenPath::PostShaderDir, root);
    const auto& catalog = VideoCore::GetFxCatalog();
    Settings::values.post_shader_enabled.SetValue(true);

    constexpr uint32_t Size = 64;
    const Object source = Picture(device, scheduler, Metal::Fmt::RGBA8Unorm, Size, Size,
                                  [](uint32_t x, uint32_t y) {
                                      const bool left = x < Size / 2;
                                      const bool top = y < Size / 2;
                                      return std::array<uint8_t, 4>{
                                          static_cast<uint8_t>(left ? 200 : 60),
                                          static_cast<uint8_t>(top ? 160 : 40),
                                          static_cast<uint8_t>((x + y) * 2), 255};
                                  });
    if (!source || catalog.empty()) {
        std::printf("metal-smoke: FAIL post-processing: no source texture or no effects\n");
        return 1;
    }
    const auto run = [&](const char* label) {
        Metal::PostProcess post(device, scheduler);
        const Metal::TextureHandle out = post.Apply(source.Get(), {Size, Size});
        const size_t built = post.BuiltEffects();
        const uint64_t passes = post.TakePasses();
        struct Result {
            size_t built;
            uint64_t passes;
            bool changed;
            bool finite;
            bool drawn;
            double mean;
        } result{built, passes, out != source.Get(), true, false, 0.0};
        if (out != nullptr && out != source.Get()) {
            const auto bytes = ReadBack(device, scheduler, out, Size, Size, 8);
            double sum = 0.0;
            for (size_t i = 0; i + 1 < bytes.size(); i += 2) {
                uint16_t bits;
                std::memcpy(&bits, bytes.data() + i, 2);
                const float value = Half(bits);
                result.finite = result.finite && std::isfinite(value);
                sum += value;
            }
            result.mean = sum / (bytes.size() / 2.0);
            result.drawn = !bytes.empty() && result.mean > 0.0;
        }
        std::printf("metal-smoke: post-processing %s: %zu built, %llu passes, mean %.3f: %s\n",
                    label, result.built, static_cast<unsigned long long>(result.passes),
                    result.mean,
                    result.changed && result.finite && result.drawn ? "ok" : "no output");
        return result.changed && result.finite && result.drawn && result.built != 0;
    };
    size_t usable = 0;
    size_t good = 0;
    std::string bad;
    for (const VideoCore::FxEffectDesc& effect : catalog) {
        if (!effect.Valid()) {
            continue;
        }
        ++usable;
        VideoCore::FxChain::Instance().Clear();
        VideoCore::FxChain::Instance().Append(effect.file, effect.techniques.front());
        if (run(effect.file.c_str())) {
            ++good;
        } else {
            bad += " " + effect.file;
        }
    }
    bool passed = usable != 0 && good == usable;
    // Two effects in one chain: the first one's output is the second's input.
    VideoCore::FxChain::Instance().Clear();
    size_t chained = 0;
    for (const VideoCore::FxEffectDesc& effect : catalog) {
        if (effect.Valid() && chained < 2) {
            VideoCore::FxChain::Instance().Append(effect.file, effect.techniques.front());
            ++chained;
        }
    }
    const bool chain_ok = chained == 2 && run("chain of two");
    passed = passed && chain_ok;
    std::printf("metal-smoke: post-processing: %zu of %zu usable effects drew%s%s, chain of two %s: "
                "%s\n",
                good, usable, bad.empty() ? "" : "; not:", bad.c_str(), chain_ok ? "ok" : "FAIL",
                passed ? "PASS" : "FAIL");
    VideoCore::FxChain::Instance().Clear();
    Settings::values.post_shader_enabled.SetValue(false);
    return passed ? 0 : 1;
#else
    std::printf("metal-smoke: post-processing: built without ENABLE_RESHADE, skipped\n");
    return 0;
#endif
}

/// N17 and N18: the frame dump's passes, dispatches and presented image through the writer, and
/// the debug server's capture request (RequestCapture, TakeCapture, Arm, WaitCapture).
int RunFrameDump() {
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "lichii-smoke-frame-dump";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    setenv("EDEN_NX2_FRAME_DUMP", (root / "env").c_str(), 1);
    setenv("EDEN_NX2_FRAME_DUMP_AT", "2f", 1);
    constexpr uint32_t Size = 8;
    const Object picture = Picture(device, scheduler, Metal::Fmt::RGBA8Unorm, Size, Size,
                                   [](uint32_t, uint32_t) {
                                       return std::array<uint8_t, 4>{255, 0, 0, 255};
                                   });
    if (!picture) {
        std::printf("metal-smoke: FAIL frame dump texture\n");
        return 1;
    }
    Metal::FrameDump::Source source{
        .texture = picture.Get(),
        .format = VideoCore::Surface::PixelFormat::A8B8G8R8_UNORM,
        .width = Size,
        .height = Size,
        .gpu_addr = 0x1000,
        .slot = 0,
        .host_format = Metal::Fmt::RGBA8Unorm,
    };
    bool passed = true;
    {
        Metal::FrameDump dump(device, scheduler);
        dump.TickFrame();
        passed = passed && !dump.Capturing();
        dump.TickFrame(); // frame 2: capturing
        const bool capturing = dump.Capturing();
        const std::array<Metal::FrameDump::Source, 1> colors{source};
        dump.BeginPass(&dump, colors);
        const std::array<uint64_t, 2> hashes{0x1111222233334444ULL, 0};
        dump.NoteDraw(hashes);
        source.slot = 0;
        dump.CaptureDispatch(0xaaaabbbbccccddddULL, colors);
        dump.CapturePresent(source);
        dump.TickFrame();
        const std::string manifest = ReadFile(root / "env" / "manifest.jsonl");
        const bool pass_line = Count(manifest, "\"kind\":\"pass\"") == 1 &&
                               manifest.find("1111222233334444") != std::string::npos;
        const bool dispatch_line = Count(manifest, "\"kind\":\"dispatch\"") == 1;
        const bool present_line = Count(manifest, "\"kind\":\"present\"") == 1;
        const bool stats = manifest.find("\"max\":[1,0,0,1]") != std::string::npos &&
                           manifest.find("\"min\":[1,0,0,1]") != std::string::npos;
        const bool summary = manifest.find("\"passes\":1,\"draws\":1,\"dispatches\":1") !=
                             std::string::npos;
        size_t pngs = 0;
        size_t raws = 0;
        for (const auto& entry : std::filesystem::directory_iterator(root / "env")) {
            const std::string extension = entry.path().extension().string();
            const std::string name = entry.path().filename().string();
            pngs += extension == ".png" && ReadFile(entry.path()).starts_with("\x89PNG");
            raws += name.ends_with(".raw.zst") && std::filesystem::file_size(entry.path()) > 0;
        }
        const bool ok = capturing && pass_line && dispatch_line && present_line && stats &&
                        summary && pngs == 3 && raws == 3 && !dump.Capturing();
        std::printf("metal-smoke: frame dump (EDEN_NX2_FRAME_DUMP): pass %d, dispatch %d, present %d, "
                    "stats %d, summary %d, %zu PNGs, %zu raw: %s\n",
                    pass_line, dispatch_line, present_line, stats, summary, pngs, raws,
                    ok ? "PASS" : "FAIL");
        passed = passed && ok;

        // The debug server's capture_frame: the request, the rasterizer taking it at a frame
        // end, the capture of the next frame and the answer.
        VideoCore::Nx2Debug::BeginSession();
        const std::filesystem::path requested = root / "requested";
        std::filesystem::create_directories(requested);
        VideoCore::Nx2Debug::RequestCapture({.id = 7, .directory = requested});
        const bool has = VideoCore::Nx2Debug::HasRequests();
        const auto taken = VideoCore::Nx2Debug::TakeCapture();
        if (taken) {
            dump.Arm(taken->id, taken->directory);
        }
        dump.TickFrame(); // starts
        const bool started = dump.Capturing();
        dump.CapturePresent(source);
        dump.TickFrame(); // ends
        const auto answer = VideoCore::Nx2Debug::WaitCapture(7, std::chrono::seconds{10});
        const std::string requested_manifest = ReadFile(requested / "manifest.jsonl");
        const bool server = has && taken && started && answer && answer->error.empty() &&
                            Count(requested_manifest, "\"kind\":\"present\"") == 1 &&
                            requested_manifest.find("\"kind\":\"summary\"") != std::string::npos &&
                            Count(requested_manifest, "\"kind\":\"sampled\"") == 0;
        std::printf("metal-smoke: frame dump (debug server capture_frame): request %d, taken %d, "
                    "answered %d, manifest with the present: %s\n",
                    has, taken.has_value(), answer.has_value(), server ? "PASS" : "FAIL");
        passed = passed && server;

        // The sampled capture (capture_frame with sampled and buffers): a texture a draw samples
        // (and a null one), a uniform buffer from guest bytes, a storage buffer the GPU wrote
        // (read from the host buffer), a dispatch's texture and group counts, a draw record and
        // a skipped draw.
        const std::filesystem::path sampled_dir = root / "sampled";
        std::filesystem::create_directories(sampled_dir);
        uint8_t* host_bytes = nullptr;
        const Object host_buffer = Metal::CreateBuffer(device, 64, true, &host_bytes);
        for (uint32_t i = 0; i < 64; ++i) {
            host_bytes[i] = static_cast<uint8_t>(i * 3);
        }
        VideoCore::Nx2Debug::RequestCapture({.id = 8, .directory = sampled_dir, .sampled = true,
                                             .buffers = true});
        const auto sampled_request = VideoCore::Nx2Debug::TakeCapture();
        if (sampled_request) {
            dump.Arm(sampled_request->id, sampled_request->directory, sampled_request->sampled,
                     sampled_request->buffers);
        }
        dump.TickFrame();
        const std::array<Metal::FrameDump::Source, 1> colors2{source};
        dump.BeginPass(&source, colors2);
        const std::array<uint64_t, 6> hashes2{0, 0x1111, 0, 0, 0, 0x2222};
        dump.NoteDraw(hashes2);
        Metal::FrameDump::Sampled texture{.stage = 4, .tic = 9, .source = source,
                                          .sampler = ",\"sampler\":{\"tsc\":[\"00000000\"]}",
                                          .key_addr = 0x1000, .key_extra = 1};
        Metal::FrameDump::Sampled missing{.stage = 4, .tic = 10, .null = true};
        const std::array<Metal::FrameDump::Sampled, 3> textures{texture, missing, texture};
        dump.NoteSampled(textures);
        std::vector<Metal::FrameDump::BoundBuffer> bound(2);
        bound[0] = {.stage = 4, .storage = false, .index = 2, .gpu_addr = 0x2000,
                    .device_addr = 0x3000, .bound_size = 16, .size = 16};
        bound[0].guest.assign(16, 0x5a);
        bound[1] = {.stage = 4, .storage = true, .index = 0, .written = true, .gpu_addr = 0x4000,
                    .device_addr = 0x5000, .bound_size = 64, .size = 64, .gpu_modified = true,
                    .host = host_buffer.Get(), .host_offset = 0};
        dump.NoteBuffers(false, hashes2, bound);
        dump.NoteDrawRecord({.first = 0, .elements = 6, .instances = 1});
        dump.NoteSkipped(false, 0x3333, 0x1000, "program not built");
        dump.NoteDispatchSampled(0x4444, std::array{texture});
        dump.NoteLaunch(0x4444, {8, 4, 1}, {16, 16, 1}, std::nullopt, std::nullopt);
        dump.TickFrame();
        const auto sampled_answer = VideoCore::Nx2Debug::WaitCapture(8, std::chrono::seconds{10});
        const std::string sampled_manifest = ReadFile(sampled_dir / "manifest.jsonl");
        const bool sampled_ok =
            sampled_request && sampled_answer && sampled_answer->error.empty() &&
            Count(sampled_manifest, "\"kind\":\"sampled\"") == 3 && // texture, null, dispatch
            sampled_manifest.find("\"null\":true") != std::string::npos &&
            Count(sampled_manifest, "\"kind\":\"buffer\"") == 2 &&
            sampled_manifest.find("\"source\":\"host\"") != std::string::npos &&
            sampled_manifest.find("\"source\":\"guest\"") != std::string::npos &&
            Count(sampled_manifest, "\"kind\":\"launch\"") == 1 &&
            sampled_manifest.find("\"grid\":[8,4,1]") != std::string::npos &&
            Count(sampled_manifest, "\"kind\":\"draw\"") == 1 &&
            Count(sampled_manifest, "\"kind\":\"skipped\"") == 1;
        size_t buffer_files = 0;
        for (const auto& entry : std::filesystem::directory_iterator(sampled_dir)) {
            buffer_files += entry.path().filename().string().starts_with("buffer_");
        }
        std::printf("metal-smoke: frame dump (sampled): %zu sampled, %zu buffers (%zu files), %zu "
                    "launch, %zu draw, %zu skipped: %s\n",
                    Count(sampled_manifest, "\"kind\":\"sampled\""),
                    Count(sampled_manifest, "\"kind\":\"buffer\""), buffer_files,
                    Count(sampled_manifest, "\"kind\":\"launch\""),
                    Count(sampled_manifest, "\"kind\":\"draw\""),
                    Count(sampled_manifest, "\"kind\":\"skipped\""),
                    sampled_ok && buffer_files == 2 ? "PASS" : "FAIL");
        passed = passed && sampled_ok && buffer_files == 2;
        VideoCore::Nx2Debug::EndSession();
    }
    unsetenv("EDEN_NX2_FRAME_DUMP");
    unsetenv("EDEN_NX2_FRAME_DUMP_AT");
    return passed ? 0 : 1;
}

/// N09 and N10: a square that moves four pixels between two frames; the frame made halfway
/// between has it two pixels along, by FSR 3's flow and interpolation shaders, and by MetalFX
/// from the same flow.
int RunFrameGen() {
#ifdef HAS_LSFG
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    constexpr uint32_t Size = 128;
    constexpr uint32_t Format = Metal::Fmt::BGRA8Unorm;
    // A white 12 x 12 square on black with its left edge at `left`; B G R A in memory.
    const auto square = [&](uint32_t left) {
        return Picture(device, scheduler, Format, Size, Size, [left](uint32_t x, uint32_t y) {
            const bool inside = x >= left && x < left + 12 && y >= 58 && y < 70;
            const uint8_t value = inside ? 255 : 0;
            return std::array<uint8_t, 4>{value, value, value, 255};
        });
    };
    // The square moves four pixels a frame: 40, 40, 44, 48 ... 64. The last pair is 60 and 64.
    std::vector<Object> pictures;
    for (const uint32_t left : {40U, 40U, 44U, 48U, 52U, 56U, 60U, 64U}) {
        pictures.push_back(square(left));
    }
    if (std::ranges::any_of(pictures, [](const Object& picture) { return !picture; })) {
        std::printf("metal-smoke: FAIL frame generation textures\n");
        return 1;
    }
    Settings::values.frame_gen.SetValue(true);
    Settings::values.frame_gen_multiplier.SetValue(2);
    int result = 0;
    for (const auto type : {Settings::FrameGenType::Fsr3, Settings::FrameGenType::MetalFx}) {
        Settings::values.frame_gen_type.SetValue(type);
        const bool metalfx = type == Settings::FrameGenType::MetalFx;
        Metal::FrameGen frame_gen(device, scheduler);
        size_t generated_count = 0;
        // Two frames to start, the pacer's first plan, then a pair at a time (MetalFX starts its
        // history over at the first one it makes and needs a few to settle).
        Metal::TextureHandle generated_texture = nullptr;
        for (const Object& frame : pictures) {
            std::this_thread::sleep_for(std::chrono::milliseconds{16});
            generated_count = frame_gen.Process(frame.Get(), {Size, Size}, Format, {Size, Size});
            if (generated_count == 1) {
                generated_texture = frame_gen.Generate(0);
            }
        }
        const Metal::TextureHandle generated = generated_count == 1 ? generated_texture : nullptr;
        if (generated == nullptr) {
            std::printf("metal-smoke: frame generation (%s): %zu frames planned, none made: FAIL\n",
                        metalfx ? "MetalFX" : "FSR 3", generated_count);
            result = 1;
            continue;
        }
        const auto bytes = ReadBack(device, scheduler, generated, Size, Size, 4);
        const auto at = [&](uint32_t x) {
            return bytes.empty() ? 0 : static_cast<int>(bytes[(uint64_t{64} * Size + x) * 4 + 1]);
        };
        // The square was at 60..71, then 64..75: the middle one is 62..73. 60 and 61 are in the
        // first only and 72 to 75 in the last: a cross-fade would show about half there.
        const int before = at(56);
        const int mid_left = at(62);
        const int mid_right = at(71);
        const int a_only = at(60);
        std::string profile;
        for (uint32_t x = 54; x <= 76; x += 2) {
            profile += " " + std::to_string(at(x));
        }
        std::printf("metal-smoke: frame generation row 64, columns 54 to 76 step 2:%s\n",
                    profile.c_str());
        const char* const maker = frame_gen.LastMaker();
        const bool moved = mid_left > 190 && mid_right > 190 && a_only < 100 && before < 60;
        // MetalFX reconstructs from the same flow with its own filter: held to the middle being
        // lit and the far side dark, not to the shader's exact edges.
        const bool loose = mid_left > 128 && mid_right > 128 && before < 60;
        const bool expected_maker = metalfx ? std::strcmp(maker, "MetalFX") == 0 ||
                                                  std::strcmp(maker, "FSR 3") == 0
                                            : std::strcmp(maker, "FSR 3") == 0;
        const bool ok = expected_maker && (metalfx ? loose : moved);
        std::printf("metal-smoke: frame generation (%s, made by %s): columns 56 %d, 60 %d, 62 %d, "
                    "71 %d (a cross-fade gives about 128 at 60 and 62): %s\n",
                    metalfx ? "MetalFX" : "FSR 3", maker, before, a_only, mid_left, mid_right,
                    ok ? "PASS" : "FAIL");
        result = result != 0 ? result : (ok ? 0 : 1);
    }
    Settings::values.frame_gen.SetValue(false);
    return result;
#else
    std::printf("metal-smoke: frame generation: built without ENABLE_LSFG, skipped\n");
    return 0;
#endif
}

/// N21: a presented picture that is dark except for one bright frame, and a pass target and a
/// dispatch image that flash with it, through the tile reduction and the analysis of the frames
/// around it; a flash_<frame>.jsonl and .tiles come out and name the presented flash.
int RunFlashWatch() {
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "lichii-smoke-flash-watch";
    std::filesystem::remove_all(root);
    setenv("EDEN_NX2_FLASH_WATCH", root.c_str(), 1);
    constexpr uint32_t Size = 64;
    const auto flat = [&](uint8_t value) {
        return Picture(device, scheduler, Metal::Fmt::RGBA8Unorm, Size, Size,
                       [value](uint32_t, uint32_t) { return std::array<uint8_t, 4>{value, value, value, 255}; });
    };
    const Object dark = flat(40);
    const Object bright = flat(240);
    if (!dark || !bright) {
        std::printf("metal-smoke: FAIL flash watch textures\n");
        return 1;
    }
    constexpr size_t FlashFrame = 3;
    bool armed = false;
    {
        Metal::FlashWatch watch(device, scheduler);
        armed = watch.Enabled() && watch.WatchesPasses();
        for (size_t frame = 0; frame < 9; ++frame) {
            const Object& picture = frame == FlashFrame ? bright : dark;
            Metal::FlashWatch::Source source{
                .texture = picture.Get(),
                .format = VideoCore::Surface::PixelFormat::A8B8G8R8_UNORM,
                .width = Size,
                .height = Size,
                .gpu_addr = 0x1000,
                .slot = 0,
            };
            const std::array<Metal::FlashWatch::Source, 1> colors{source};
            watch.BeginPass(&watch, colors);
            const std::array<uint64_t, 1> hashes{0xabcdULL};
            watch.NoteDraw(hashes);
            watch.NoteDispatch(0x7777, colors);
            watch.NotePresent(source);
            watch.TickFrame();
            scheduler.Flush();
        }
        scheduler.Finish();
        for (int i = 0; i < 4; ++i) {
            watch.TickFrame();
        }
    }
    unsetenv("EDEN_NX2_FLASH_WATCH");
    std::string report;
    size_t tile_bytes = 0;
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        const std::string name = entry.path().filename().string();
        if (name.ends_with(".jsonl")) {
            report = ReadFile(entry.path());
        } else if (name.ends_with(".tiles")) {
            tile_bytes = std::filesystem::file_size(entry.path());
        }
    }
    const bool flashed = report.find("\"kind\":\"flash\"") != std::string::npos &&
                         report.find("\"frame\":3,") != std::string::npos &&
                         report.find("\"trigger\":\"present:") != std::string::npos;
    const bool outputs = Count(report, "\"kind\":\"output\"") == 3 &&
                         report.find("\"type\":\"pass\"") != std::string::npos &&
                         report.find("\"type\":\"dispatch\"") != std::string::npos;
    // 3 outputs of 32 x 18 tiles, 16 bytes each, in 5 frames.
    const bool tiles = tile_bytes == size_t{3} * 32 * 18 * 16 * 5;
    std::printf("metal-smoke: flash watch: armed %d, report for frame 3 naming the present %d, %zu "
                "outputs, tiles file %zu bytes: %s\n",
                armed, flashed, Count(report, "\"kind\":\"output\""), tile_bytes,
                armed && flashed && outputs && tiles ? "PASS" : "FAIL");
    return armed && flashed && outputs && tiles ? 0 : 1;
}

/// O04: FSR 3.1 (the SDK's nine passes through the translator) upscales a still horizontal
/// gradient 64x64 -> 128x128 over several frames; the output follows the gradient.
int RunFsr3Upscale() {
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    constexpr uint32_t Render = 64;
    constexpr uint32_t Display = 128;
    const Object color = Picture(device, scheduler, Metal::Fmt::RGBA8Unorm, Render, Render,
                                 [](uint32_t x, uint32_t) {
                                     const uint8_t value = static_cast<uint8_t>(x * 4);
                                     return std::array<uint8_t, 4>{value, value, value, 255};
                                 });
    const Object motion = Picture(device, scheduler, Metal::Fmt::RG16Float, Render, Render,
                                  [](uint32_t, uint32_t) { return std::array<uint8_t, 4>{0, 0, 0, 0}; });
    const Object depth = Metal::CreateTexture(device, {.kind = Metal::TextureKind::e2D,
                                                       .format = Metal::Fmt::Depth32Float,
                                                       .width = Render,
                                                       .height = Render,
                                                       .render_target = true});
    if (!color || !motion || !depth) {
        std::printf("metal-smoke: FAIL FSR 3.1 textures\n");
        return 1;
    }
    Metal::ClearPass clear{};
    clear.depth = {.texture = depth.Get()};
    clear.clear_depth = true;
    clear.depth_value = 0.5;
    clear.width = Render;
    clear.height = Render;
    scheduler.Clear(clear);
    scheduler.Finish();
    Metal::Fsr3Upscaler upscaler(device, scheduler);
    if (!upscaler.Prepare()) {
        std::printf("metal-smoke: FSR 3.1 passes not built: FAIL\n");
        return 1;
    }
    Metal::TextureHandle out = nullptr;
    for (int frame = 0; frame < 6; ++frame) {
        out = upscaler.Upscale({.color = color.Get(),
                                .depth = depth.Get(),
                                .motion = motion.Get(),
                                .render_size = {Render, Render},
                                .upscale_size = {Display, Display},
                                .jitter = {0.0f, 0.0f},
                                .motion_vector_scale = {1.0f, 1.0f},
                                .frame_time_ms = 16.0f,
                                .reset = frame == 0,
                                .sharpen = false,
                                .sharpness = 1.0f});
        scheduler.Flush();
    }
    if (out == nullptr) {
        std::printf("metal-smoke: FSR 3.1: no output: FAIL\n");
        return 1;
    }
    const auto bytes = ReadBack(device, scheduler, out, Display, Display, 8);
    double worst = 0.0;
    bool finite = !bytes.empty();
    for (uint32_t x = 8; x < Display - 8; x += 8) {
        uint16_t bits;
        std::memcpy(&bits, bytes.data() + (uint64_t{64} * Display + x) * 8, 2);
        const float value = Half(bits);
        finite = finite && std::isfinite(value);
        const double expected = (x / 2.0 * 4.0) / 255.0;
        worst = (std::max)(worst, std::abs(value - expected));
    }
    std::printf("metal-smoke: FSR 3.1 upscale 64x64 -> 128x128, a gradient over 6 frames: worst "
                "difference %.3f from the gradient: %s\n",
                worst, finite && worst < 0.08 ? "PASS" : "FAIL");
    return finite && worst < 0.08 ? 0 : 1;
}

int RunAll() {
    if (std::getenv("LICHII_SMOKE_LOG") != nullptr) {
        Common::Log::Initialize();
        Common::Log::SetColorConsoleBackendEnabled(true);
        Common::Log::Start();
    }
    int result = 0;
    for (const auto test : {RunPostProcess, RunFrameDump, RunFlashWatch, RunFrameGen, RunFsr3Upscale}) {
        const int outcome = test();
        result = result != 0 ? result : outcome;
    }
    return result;
}

} // namespace SmokePresent
