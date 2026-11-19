// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-replay: runs a game's stored pipelines (its Vulkan pipeline cache files) and the
// draws, clears and dispatches of a feature census (EDEN_NX2_FEATURE_CENSUS) through the Metal
// renderer's translation and descriptor checks, without a Metal device, and reports what the
// renderer would refuse. tools/nx2/metal-replay.md.

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/cityhash.h"
#include "common/common_types.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/stage.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/feature_census.h"
#include "video_core/gpu.h"
#include "video_core/pipeline_cache_files.h"
#include "video_core/renderer_metal/metal_draw_desc.h"
#include "video_core/renderer_metal/metal_formats.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_index_convert.h"
#include "video_core/renderer_metal/metal_pipeline_translate.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/shader_environment.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/samples_helper.h"

namespace Metal {
namespace {

namespace Census = VideoCommon::FeatureCensus;
using VideoCommon::FileEnvironment;
using VideoCore::Surface::PixelFormat;
using Hashes = std::array<u64, 6>;
using Topology = Maxwell::PrimitiveTopology;

constexpr std::array<const char*, 6> SLOT_NAMES{
    "vertex_a", "vertex_b", "tess_control", "tess_eval", "geometry", "fragment",
};
constexpr std::array<const char*, 11> MISC_NAMES{
    "DrawTexture",      "InlineToMemory",  "ConditionalRendering", "TransformFeedbackBinding",
    "SurfaceCopy",      "TemporalUpscale", "DmaCopy",              "DownloadMemory",
    "FragmentBarrier",  "TiledCacheBarrier", "CounterReset",
};
constexpr std::array<char, 8> CACHE_MAGIC{'y', 'u', 'z', 'u', 'c', 'a', 'c', 'h'};
constexpr std::array<u32, 1> CACHE_VERSIONS{VideoCommon::VulkanPipelineCacheVersion};
/// The sample counts Apple's GPUs take; the renderer asks the device (DevicePipelineLimits).
constexpr PipelineLimits LIMITS{.sample_counts = (1U << 1) | (1U << 2) | (1U << 4)};
const VideoCommon::GuessMissing GUESS{};

// Counts by name, printed largest first.

struct Tally {
    u64 rows{};
    u64 weight{};
    std::vector<std::string> examples;
};
using Tallies = std::map<std::string, Tally>;

void Count(Tallies& tallies, std::string name, u64 weight, std::string_view example = {}) {
    Tally& tally = tallies[std::move(name)];
    ++tally.rows;
    tally.weight += weight;
    if (!example.empty() && tally.examples.size() < 3) {
        tally.examples.emplace_back(example);
    }
}

template <typename... Args>
void Line(std::string& out, fmt::format_string<Args...> format, Args&&... args) {
    fmt::format_to(std::back_inserter(out), format, std::forward<Args>(args)...);
    out.push_back('\n');
}

void Print(std::string& out, std::string_view title, const Tallies& tallies, std::string_view unit,
           std::string_view weight_unit = {}) {
    Line(out, "");
    Line(out, "{}", title);
    if (tallies.empty()) {
        Line(out, "  (none)");
        return;
    }
    std::vector<const Tallies::value_type*> order;
    for (const auto& entry : tallies) {
        order.push_back(&entry);
    }
    std::ranges::sort(order, [](const auto* lhs, const auto* rhs) {
        return lhs->second.weight != rhs->second.weight ? lhs->second.weight > rhs->second.weight
                                                        : lhs->second.rows > rhs->second.rows;
    });
    size_t printed = 0;
    for (const auto* entry : order) {
        if (printed++ == 48) {
            Line(out, "  ... and {} more", order.size() - 48);
            break;
        }
        if (weight_unit.empty()) {
            Line(out, "  {:>9} {}: {}", entry->second.rows, unit, entry->first);
        } else {
            Line(out, "  {:>9} {}, {:>11} {}: {}", entry->second.rows, unit, entry->second.weight,
                 weight_unit, entry->first);
        }
        for (const std::string& example : entry->second.examples) {
            Line(out, "              e.g. {}", example);
        }
    }
}

std::string OneLine(std::string_view text, size_t limit) {
    std::string line;
    for (const char c : text.substr(0, limit)) {
        line.push_back(c == '\n' || c == '\r' ? ' ' : c);
    }
    if (text.size() > limit) {
        line += "...";
    }
    return line;
}

std::string HashText(const Hashes& hashes) {
    std::string text;
    for (size_t slot = 0; slot < hashes.size(); ++slot) {
        if (hashes[slot] != 0) {
            text += fmt::format("{}{}={:016x}", text.empty() ? "" : " ", SLOT_NAMES[slot],
                                hashes[slot]);
        }
    }
    return text;
}

// The pipeline cache files, read as VideoCommon::LoadPipelines reads them. That function deletes
// a file it can't read, so the replay has its own reader.

struct GraphicsRecordKey {
    Hashes unique_hashes;
    Vulkan::FixedPipelineState state;
};

struct ComputeRecordKey {
    u64 unique_hash;
    u32 shared_memory_size;
    std::array<u32, 3> workgroup_size;
};

struct Record {
    bool is_compute{};
    /// Graphics: one environment per program of the key, in slot order.
    bool consistent{};
    bool has_mesh{};
    std::vector<FileEnvironment> envs;
    GraphicsRecordKey graphics{Zeroed<GraphicsRecordKey>()};
    ComputeRecordKey compute{};
    /// Environments note what was read from them, and take the guess for a translation.
    std::mutex mutex;
};

struct EnvRef {
    Record* record{};
    u32 env{};
};
using Candidate = std::vector<EnvRef>;

struct HashesHash {
    size_t operator()(const Hashes& hashes) const noexcept {
        return static_cast<size_t>(Common::CityHash64(
            reinterpret_cast<const char*>(hashes.data()), sizeof(hashes)));
    }
};

struct Caches {
    std::vector<std::unique_ptr<Record>> records;
    std::unordered_map<Hashes, std::vector<Record*>, HashesHash> by_programs;
    /// A program's newest environment by slot, for keys no record has as a whole.
    std::array<std::unordered_map<u64, EnvRef>, 6> by_slot;
    std::unordered_map<u64, std::vector<Record*>> by_compute;
    std::vector<std::string> notes;
    u64 graphics{};
    u64 compute{};
    u64 inconsistent{};
};

std::unique_ptr<Record> ReadRecord(std::ifstream& file, std::streamoff end) {
    u32 num_envs{};
    file.read(reinterpret_cast<char*>(&num_envs), sizeof(num_envs));
    if (!file || num_envs == 0 || num_envs > Maxwell::MaxShaderProgram) {
        return nullptr;
    }
    std::vector<FileEnvironment> envs(num_envs);
    for (FileEnvironment& env : envs) {
        // Code size, then the entry count of each table: none can be larger than the file.
        const std::streamoff begin = static_cast<std::streamoff>(file.tellg());
        std::array<u64, 6> counts{};
        file.read(reinterpret_cast<char*>(counts.data()), sizeof(counts));
        if (!file) {
            return nullptr;
        }
        file.seekg(begin);
        const u64 left = static_cast<u64>(end - begin);
        if (std::ranges::any_of(counts, [left](u64 count) { return count > left; })) {
            return nullptr;
        }
        env.Deserialize(file);
        if (!file ||
            static_cast<u32>(env.ShaderStage()) > static_cast<u32>(Shader::Stage::Mesh)) {
            return nullptr;
        }
    }
    auto record = std::make_unique<Record>();
    record->is_compute = envs.front().ShaderStage() == Shader::Stage::Compute;
    const auto compute_envs = std::ranges::count_if(envs, [](const FileEnvironment& env) {
        return env.ShaderStage() == Shader::Stage::Compute;
    });
    if (record->is_compute ? num_envs != 1 : compute_envs != 0) {
        return nullptr;
    }
    if (record->is_compute) {
        file.read(reinterpret_cast<char*>(&record->compute), sizeof(record->compute));
    } else {
        file.read(reinterpret_cast<char*>(&record->graphics), sizeof(record->graphics));
    }
    if (!file) {
        return nullptr;
    }
    record->has_mesh = std::ranges::any_of(envs, [](const FileEnvironment& env) {
        return env.ShaderStage() == Shader::Stage::Task || env.ShaderStage() == Shader::Stage::Mesh;
    });
    record->envs = std::move(envs);
    return record;
}

void Index(Caches& caches, Record& record) {
    if (record.is_compute) {
        ++caches.compute;
        caches.by_compute[record.compute.unique_hash].push_back(&record);
        return;
    }
    ++caches.graphics;
    const Hashes& hashes = record.graphics.unique_hashes;
    const auto programs = std::ranges::count_if(hashes, [](u64 hash) { return hash != 0; });
    record.consistent = static_cast<size_t>(programs) == record.envs.size();
    if (!record.consistent) {
        ++caches.inconsistent;
        return;
    }
    caches.by_programs[hashes].push_back(&record);
    u32 env = 0;
    for (size_t slot = 0; slot < hashes.size(); ++slot) {
        if (hashes[slot] != 0) {
            caches.by_slot[slot][hashes[slot]] = EnvRef{&record, env++};
        }
    }
}

void ReadCache(Caches& caches, const std::filesystem::path& path) {
    const std::string name = path.filename().string();
    std::ifstream file{path, std::ios::binary | std::ios::ate};
    if (!file.is_open()) {
        caches.notes.push_back(fmt::format("{}: can't be opened", name));
        return;
    }
    const std::streamoff end = static_cast<std::streamoff>(file.tellg());
    file.seekg(0);
    std::array<char, 8> magic{};
    u32 version{};
    file.read(magic.data(), magic.size()).read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!file || magic != CACHE_MAGIC) {
        caches.notes.push_back(fmt::format("{}: not a pipeline cache file", name));
        return;
    }
    if (std::ranges::find(CACHE_VERSIONS, version) == CACHE_VERSIONS.end()) {
        caches.notes.push_back(
            fmt::format("{}: version {} is not one the replay reads", name, version));
        return;
    }
    u64 records = 0;
    std::streamoff at = static_cast<std::streamoff>(file.tellg());
    while (at < end) {
        std::unique_ptr<Record> record = ReadRecord(file, end);
        if (!record) {
            caches.notes.push_back(fmt::format(
                "{}: record {} at byte {} is damaged, {} bytes after it not read", name, records,
                at, end - at));
            break;
        }
        Index(caches, *record);
        caches.records.push_back(std::move(record));
        ++records;
        at = static_cast<std::streamoff>(file.tellg());
    }
    caches.notes.push_back(
        fmt::format("{}: version {}, {} records, {} bytes", name, version, records, end));
}

// The census file: {"version":1, "<category>":{"total":n,"dropped":n,"columns":[...],
// "rows":[[count,word,...],...]}, ...}, as feature_census.cpp writes it.

struct CensusRow {
    u64 count{};
    std::vector<u32> words;
};

struct CensusCategory {
    u64 total{};
    u64 dropped{};
    std::vector<std::string> columns; ///< Without the leading count
    std::vector<CensusRow> rows;

    [[nodiscard]] size_t Column(std::string_view name) const {
        const auto it = std::ranges::find(columns, name);
        if (it == columns.end()) {
            throw std::runtime_error{fmt::format("the census has no column {}", name)};
        }
        return static_cast<size_t>(it - columns.begin());
    }
};
using CensusData = std::map<std::string, CensusCategory, std::less<>>;

class JsonReader {
public:
    explicit JsonReader(std::string text_) : text{std::move(text_)} {}

    [[nodiscard]] bool Take(char c) {
        Skip();
        if (pos < text.size() && text[pos] == c) {
            ++pos;
            return true;
        }
        return false;
    }

    void Expect(char c) {
        if (!Take(c)) {
            throw std::runtime_error{fmt::format("census: '{}' expected at byte {}", c, pos)};
        }
    }

    [[nodiscard]] std::string String() {
        Expect('"');
        const size_t begin = pos;
        while (pos < text.size() && text[pos] != '"') {
            ++pos;
        }
        std::string value = text.substr(begin, pos - begin);
        Expect('"');
        return value;
    }

    [[nodiscard]] u64 Number() {
        Skip();
        const size_t begin = pos;
        u64 value = 0;
        while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
            value = value * 10 + static_cast<u64>(text[pos] - '0');
            ++pos;
        }
        if (pos == begin) {
            throw std::runtime_error{fmt::format("census: number expected at byte {}", pos)};
        }
        return value;
    }

private:
    void Skip() {
        while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\n' || text[pos] == '\r' ||
                                     text[pos] == '\t')) {
            ++pos;
        }
    }

    std::string text;
    size_t pos{};
};

CensusCategory ReadCategory(JsonReader& json) {
    CensusCategory category;
    json.Expect('{');
    do {
        const std::string key = json.String();
        json.Expect(':');
        if (key == "total") {
            category.total = json.Number();
        } else if (key == "dropped") {
            category.dropped = json.Number();
        } else if (key == "columns") {
            json.Expect('[');
            do {
                category.columns.push_back(json.String());
            } while (json.Take(','));
            json.Expect(']');
            if (category.columns.empty() || category.columns.front() != "count") {
                throw std::runtime_error{"census: a category's first column is not the count"};
            }
            category.columns.erase(category.columns.begin());
        } else if (key == "rows") {
            json.Expect('[');
            while (json.Take('[')) {
                CensusRow row;
                row.count = json.Number();
                while (json.Take(',')) {
                    row.words.push_back(static_cast<u32>(json.Number()));
                }
                json.Expect(']');
                if (row.words.size() != category.columns.size()) {
                    throw std::runtime_error{"census: a row is not as wide as its columns"};
                }
                category.rows.push_back(std::move(row));
                if (!json.Take(',')) {
                    break;
                }
            }
            json.Expect(']');
        } else {
            throw std::runtime_error{fmt::format("census: unknown key {}", key)};
        }
    } while (json.Take(','));
    json.Expect('}');
    return category;
}

CensusData ReadCensus(const std::filesystem::path& path) {
    std::ifstream file{path, std::ios::binary};
    if (!file.is_open()) {
        throw std::runtime_error{fmt::format("{} can't be opened", path.string())};
    }
    std::ostringstream text;
    text << file.rdbuf();
    JsonReader json{text.str()};
    CensusData census;
    json.Expect('{');
    do {
        const std::string key = json.String();
        json.Expect(':');
        if (key == "version") {
            if (json.Number() != 1) {
                throw std::runtime_error{"census: not version 1"};
            }
        } else {
            census.emplace(key, ReadCategory(json));
        }
    } while (json.Take(','));
    json.Expect('}');
    return census;
}

// What a key's programs came to.

struct Outcome {
    bool ok{};
    /// No stored environment had every value this translation read; the missing ones were
    /// guessed (VideoCommon::GuessMissing).
    bool guessed{};
    bool missing_env{};
    std::string cause;
    std::string detail;
    bool mesh{};
    bool has_task{};
    u32 mesh_topology{};
    /// Drawn with Metal mesh functions (NativeMesh); else why not.
    bool native_mesh{};
    std::string native_cause;
    /// The mesh pipeline's emulation, if it has one (its fallback when native).
    bool mesh_emulated{};
    u64 mesh_hash{};
    u64 task_hash{};
    NativeMesh native_shape{};
    bool tessellated{};
    ShaderTranslate::TessellationModes modes{};
    /// By Tessellation::IndexSizes variant for a tessellation pipeline, else [0].
    std::array<VertexProgramUse, 3> vertex_use{};
    /// ProgramFeature bits of the programs (texture rows of metal-coverage.md).
    u32 features{};
};

struct GraphicsJob {
    GraphicsPipelineKey key;
    std::vector<Candidate> candidates;
    Outcome outcome;
    u64 records{};
    u64 census_rows{};
};

struct ComputeJob {
    u64 hash{};
    std::vector<Record*> candidates;
    Outcome outcome;
    u64 records{};
    u64 census_rows{};
};

struct Jobs {
    std::vector<GraphicsJob> graphics;
    std::unordered_map<GraphicsPipelineKey, size_t> graphics_index;
    std::vector<ComputeJob> compute;
    std::unordered_map<u64, size_t> compute_index;
};

/// The translated functions, counted by their text and, with a directory, written there.
class MslSink {
public:
    explicit MslSink(std::filesystem::path directory_) : directory{std::move(directory_)} {
        if (!directory.empty()) {
            std::filesystem::create_directories(directory);
            index.open(directory / "index.txt", std::ios::binary | std::ios::trunc);
        }
    }

    void Add(std::string_view role, const StageProgram& program) {
        if (program.msl.empty()) {
            return;
        }
        const u64 hash = Common::CityHash64(program.msl.data(), program.msl.size());
        std::scoped_lock lock{mutex};
        if (!seen.insert(hash).second) {
            return;
        }
        ++functions;
        bytes += program.msl.size();
        if (directory.empty()) {
            return;
        }
        const std::string name = fmt::format("{:016x}_{}.metal", hash, role);
        std::ofstream file{directory / name, std::ios::binary | std::ios::trunc};
        file << program.msl;
        index << name << ' ' << program.entry_point << '\n';
    }

    u64 functions{};
    u64 bytes{};

private:
    std::filesystem::path directory;
    std::ofstream index;
    std::mutex mutex;
    std::unordered_set<u64> seen;
};

/// What the texture rows of metal-coverage.md ask of a game's programs and the census does not
/// count (F09, F11, G19, R14).
enum ProgramFeature : u32 {
    WritesImage = 1U << 0,
    WritesTypelessImage = 1U << 1,
    ByteAddressedImage = 1U << 2,
    ReadsMultisample = 1U << 3,
    StoresLayer = 1U << 4,
    StoresViewportIndex = 1U << 5,
    ReadsDepthTexture = 1U << 6,
    ReadsTexelBuffer = 1U << 7,
    UsesImageBuffer = 1U << 8,
    ProgramFeatureCount = 9,
};

constexpr std::array<std::string_view, ProgramFeatureCount> ProgramFeatureNames{
    "writes a storage image",       "writes a storage image without a format",
    "byte-addressed storage image", "reads a multisampled texture",
    "stores the layer",             "stores the viewport index",
    "reads a depth texture",       "reads a texel buffer",
    "uses an image buffer",
};

u32 ProgramFeatures(const Shader::Info& info) {
    u32 features = 0;
    for (const auto& desc : info.image_descriptors) {
        if (desc.is_written) {
            features |= WritesImage;
            if (desc.format == Shader::ImageFormat::Typeless) {
                features |= WritesTypelessImage;
            }
        }
        if (desc.is_byte_addressed) {
            features |= ByteAddressedImage;
        }
    }
    for (const auto& desc : info.texture_descriptors) {
        if (desc.is_multisample) {
            features |= ReadsMultisample;
        }
        if (desc.is_depth) {
            features |= ReadsDepthTexture;
        }
    }
    if (!info.texture_buffer_descriptors.empty()) {
        features |= ReadsTexelBuffer;
    }
    if (!info.image_buffer_descriptors.empty()) {
        features |= UsesImageBuffer;
    }
    if (info.stores[Shader::IR::Attribute::Layer]) {
        features |= StoresLayer;
    }
    if (info.stores[Shader::IR::Attribute::ViewportIndex]) {
        features |= StoresViewportIndex;
    }
    return features;
}

void Summarise(const TranslatedPipeline& pipeline, Outcome& outcome, MslSink& sink) {
    outcome.ok = true;
    outcome.cause.clear();
    outcome.detail.clear();
    if (pipeline.native_mesh) {
        const NativeMesh& native = *pipeline.native_mesh;
        outcome.mesh = true;
        outcome.native_mesh = true;
        outcome.has_task = !native.object.entry_point.empty();
        outcome.mesh_topology = native.topology;
        outcome.native_shape.payload_bytes = native.payload_bytes;
        outcome.native_shape.object_threads = native.object_threads;
        outcome.native_shape.mesh_threads = native.mesh_threads;
        outcome.native_shape.threadgroup_bytes = native.threadgroup_bytes;
        sink.Add("object", native.object);
        sink.Add("mesh", native.mesh);
        sink.Add("mesh_fragment", native.fragment);
    }
    if (pipeline.mesh) {
        const MeshEmulation& mesh = *pipeline.mesh;
        outcome.mesh = true;
        outcome.mesh_emulated = true;
        outcome.has_task = mesh.has_task;
        outcome.mesh_topology = mesh.topology;
        sink.Add(mesh.has_task ? "task_kernel" : "mesh_kernel", mesh.first.program);
        outcome.features |= ProgramFeatures(mesh.first.program.info) |
                            ProgramFeatures(mesh.mesh.program.info);
        if (mesh.has_task) {
            sink.Add("mesh_kernel", mesh.mesh.program);
            sink.Add("mesh_distribution", mesh.distribution.program);
        }
        sink.Add("mesh_vertex", pipeline.vertex);
    } else if (pipeline.native_mesh) {
        return;
    } else if (pipeline.tessellation) {
        const Tessellation& tessellation = *pipeline.tessellation;
        outcome.tessellated = true;
        outcome.modes = tessellation.modes;
        for (size_t variant = 0; variant < tessellation.vertex.size(); ++variant) {
            outcome.vertex_use[variant] = VertexUse(tessellation.vertex[variant]);
            sink.Add("tess_vertex_kernel", tessellation.vertex[variant]);
        }
        sink.Add("tess_control_kernel", tessellation.control.program);
        outcome.features |= ProgramFeatures(tessellation.control.program.info);
        sink.Add("tess_evaluation", pipeline.vertex);
    } else {
        outcome.vertex_use[0] = VertexUse(pipeline.vertex);
        sink.Add("vertex", pipeline.vertex);
    }
    sink.Add("fragment", pipeline.fragment);
    outcome.features |=
        ProgramFeatures(pipeline.vertex.info) | ProgramFeatures(pipeline.fragment.info);
}

/// The environment lacks a value: another record's may have it.
bool EnvIncomplete(std::string_view detail) {
    return detail.find("Uncached") != std::string_view::npos ||
           detail.find("Out of bounds address") != std::string_view::npos;
}

std::vector<std::unique_lock<std::mutex>> LockRecords(std::vector<Record*> records) {
    std::ranges::sort(records);
    records.erase(std::unique(records.begin(), records.end()), records.end());
    std::vector<std::unique_lock<std::mutex>> locks;
    for (Record* const record : records) {
        locks.emplace_back(record->mutex);
    }
    return locks;
}

std::optional<TranslatedPipeline> TryGraphics(PipelineTranslator& translator,
                                              const GraphicsPipelineKey& key,
                                              const Candidate& candidate, bool guess,
                                              Outcome& outcome) {
    std::vector<Record*> records;
    for (const EnvRef& ref : candidate) {
        records.push_back(ref.record);
    }
    const auto locks = LockRecords(std::move(records));
    std::vector<Shader::Environment*> envs;
    for (const EnvRef& ref : candidate) {
        FileEnvironment& env = ref.record->envs[ref.env];
        env.SetGuessMissing(guess ? &GUESS : nullptr);
        envs.push_back(&env);
    }
    std::optional<TranslatedPipeline> pipeline;
    try {
        pipeline = translator.Graphics(key, envs);
        outcome.native_cause = translator.NativeMeshCause();
        if (!pipeline) {
            outcome.cause = translator.Cause();
            outcome.detail = translator.Detail();
        }
    } catch (const std::exception& exception) {
        outcome.cause = "exception";
        outcome.detail = exception.what();
    }
    for (const EnvRef& ref : candidate) {
        ref.record->envs[ref.env].SetGuessMissing(nullptr);
    }
    return pipeline;
}

void RunGraphics(GraphicsJob& job, PipelineTranslator& translator, MslSink& sink) {
    Outcome& outcome = job.outcome;
    SCOPE_EXIT {
        if (!outcome.mesh) {
            return;
        }
        // The mesh program follows the task program (slot 1) or is in slot 1 itself.
        const auto& hashes = job.key.unique_hashes;
        if (outcome.has_task) {
            outcome.task_hash = hashes[1];
            for (size_t index = 2; index < 5; ++index) {
                if (hashes[index] != 0) {
                    outcome.mesh_hash = hashes[index];
                    break;
                }
            }
        } else {
            outcome.mesh_hash = hashes[1];
        }
    };
    if (job.candidates.empty()) {
        outcome.missing_env = true;
        outcome.cause = "no environment in the caches";
        return;
    }
    for (const Candidate& candidate : job.candidates) {
        if (const auto pipeline = TryGraphics(translator, job.key, candidate, false, outcome)) {
            Summarise(*pipeline, outcome, sink);
            return;
        }
        if (!EnvIncomplete(outcome.detail)) {
            return;
        }
    }
    outcome.guessed = true;
    if (const auto pipeline =
            TryGraphics(translator, job.key, job.candidates.front(), true, outcome)) {
        Summarise(*pipeline, outcome, sink);
    }
}

std::optional<StageProgram> TryCompute(PipelineTranslator& translator, u64 hash, Record& record,
                                       bool guess, Outcome& outcome) {
    std::scoped_lock lock{record.mutex};
    FileEnvironment& env = record.envs.front();
    env.SetGuessMissing(guess ? &GUESS : nullptr);
    std::optional<StageProgram> program;
    try {
        program = translator.Compute(hash, env);
        if (!program) {
            outcome.cause = translator.Cause();
            outcome.detail = translator.Detail();
        }
    } catch (const std::exception& exception) {
        outcome.cause = "exception";
        outcome.detail = exception.what();
    }
    env.SetGuessMissing(nullptr);
    return program;
}

void RunCompute(ComputeJob& job, PipelineTranslator& translator, MslSink& sink) {
    Outcome& outcome = job.outcome;
    if (job.candidates.empty()) {
        outcome.missing_env = true;
        outcome.cause = "no environment in the caches";
        return;
    }
    const auto done = [&](const StageProgram& program) {
        outcome.ok = true;
        outcome.cause.clear();
        outcome.detail.clear();
        sink.Add("compute", program);
        outcome.features |= ProgramFeatures(program.info);
    };
    for (Record* const record : job.candidates) {
        if (const auto program = TryCompute(translator, job.hash, *record, false, outcome)) {
            done(*program);
            return;
        }
        if (!EnvIncomplete(outcome.detail)) {
            return;
        }
    }
    outcome.guessed = true;
    if (const auto program =
            TryCompute(translator, job.hash, *job.candidates.front(), true, outcome)) {
        done(*program);
    }
}

void RunJobs(Jobs& jobs, bool ios, u32 apple_family, bool native_mesh, u32 threads,
             MslSink& sink) {
    const size_t total = jobs.graphics.size() + jobs.compute.size();
    std::atomic<size_t> next{0};
    const auto work = [&] {
        PipelineTranslator translator{ios, apple_family, native_mesh};
        for (;;) {
            const size_t index = next.fetch_add(1);
            if (index >= total) {
                return;
            }
            if (index < jobs.graphics.size()) {
                RunGraphics(jobs.graphics[index], translator, sink);
            } else {
                RunCompute(jobs.compute[index - jobs.graphics.size()], translator, sink);
            }
            if ((index + 1) % 500 == 0) {
                fmt::print(stderr, "translated {} of {}\n", index + 1, total);
            }
        }
    };
    std::vector<std::thread> workers;
    for (u32 thread = 0; thread < threads; ++thread) {
        workers.emplace_back(work);
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
}

// Keys.

std::vector<Candidate> GraphicsCandidates(const Caches& caches, const Hashes& hashes,
                                          Record* own) {
    std::vector<Candidate> candidates;
    const auto whole = [&](Record* record) {
        Candidate candidate;
        for (u32 env = 0; env < record->envs.size(); ++env) {
            candidate.push_back(EnvRef{record, env});
        }
        candidates.push_back(std::move(candidate));
    };
    if (own != nullptr) {
        whole(own);
    }
    if (const auto it = caches.by_programs.find(hashes); it != caches.by_programs.end()) {
        for (auto record = it->second.rbegin();
             record != it->second.rend() && candidates.size() < 4; ++record) {
            if (*record != own) {
                whole(*record);
            }
        }
    }
    Candidate mixed;
    for (size_t slot = 0; slot < hashes.size(); ++slot) {
        if (hashes[slot] == 0) {
            continue;
        }
        const auto it = caches.by_slot[slot].find(hashes[slot]);
        if (it == caches.by_slot[slot].end()) {
            return candidates;
        }
        mixed.push_back(it->second);
    }
    if (!mixed.empty()) {
        candidates.push_back(std::move(mixed));
    }
    return candidates;
}

size_t GraphicsJobOf(Jobs& jobs, const Caches& caches, const GraphicsPipelineKey& key,
                     Record* own) {
    const auto [it, is_new] = jobs.graphics_index.try_emplace(key, jobs.graphics.size());
    if (is_new) {
        jobs.graphics.push_back(GraphicsJob{
            .key = key,
            .candidates = GraphicsCandidates(caches, key.unique_hashes, own),
            .outcome = {},
        });
    }
    return it->second;
}

size_t ComputeJobOf(Jobs& jobs, const Caches& caches, u64 hash) {
    const auto [it, is_new] = jobs.compute_index.try_emplace(hash, jobs.compute.size());
    if (is_new) {
        ComputeJob job{.hash = hash, .candidates = {}, .outcome = {}};
        if (const auto records = caches.by_compute.find(hash);
            records != caches.by_compute.end()) {
            for (auto record = records->second.rbegin();
                 record != records->second.rend() && job.candidates.size() < 4; ++record) {
                job.candidates.push_back(*record);
            }
        }
        jobs.compute.push_back(std::move(job));
    }
    return it->second;
}

/// The registers a stored Vulkan key says the draw had, as far as the Metal key reads them.
void RegsOfRecord(Maxwell& regs, const Record& record) {
    using Tess = Maxwell::Tessellation;
    const Vulkan::FixedPipelineState& state = record.graphics.state;
    regs.reg_array.fill(0);
    regs.mandated_early_z = state.early_z.Value();
    regs.depth_mode = state.ndc_minus_one_to_one != 0 ? Maxwell::DepthMode::MinusOneToOne
                                                      : Maxwell::DepthMode::ZeroToOne;
    regs.window_origin.mode.Assign(state.y_negate != 0 ? Maxwell::WindowOrigin::Mode::LowerLeft
                                                       : Maxwell::WindowOrigin::Mode::UpperLeft);
    if (state.attachment0_dual_source_blend != 0) {
        regs.blend.enable[0] = 1;
        regs.blend.color_source = Maxwell::Blend::Factor::Source1Color_GL;
    }
    regs.alpha_test_enabled = 1;
    regs.alpha_test_func =
        Vulkan::FixedPipelineState::UnpackComparisonOp(state.alpha_test_func.Value());
    regs.alpha_test_ref = std::bit_cast<f32>(state.alpha_test_ref);
    for (size_t index = 0; index < state.color_formats.size(); ++index) {
        regs.rt[index].format = static_cast<Tegra::RenderTargetFormat>(state.color_formats[index]);
    }
    regs.zeta.format = static_cast<Tegra::DepthFormat>(state.depth_format.Value());
    regs.zeta_enable = state.depth_enabled.Value();
    regs.anti_alias_samples_mode = state.msaa_mode.Value();
    // The key keeps the winding of triangles only; point and line output is not in it.
    regs.tessellation.params.domain_type.Assign(
        static_cast<Tess::DomainType>(state.tessellation_primitive.Value()));
    regs.tessellation.params.spacing.Assign(
        static_cast<Tess::Spacing>(state.tessellation_spacing.Value()));
    regs.tessellation.params.output_primitives.Assign(state.tessellation_clockwise != 0
                                                          ? Tess::OutputPrimitives::Triangles_CW
                                                          : Tess::OutputPrimitives::Triangles_CCW);
    regs.patch_vertices = state.patch_control_points_minus_one.Value() + 1;
    regs.point_size = std::bit_cast<f32>(state.point_size);
    if (state.dynamic_state.logic_op_enable != 0) {
        regs.logic_op.enable = 1;
        regs.logic_op.op = Vulkan::FixedPipelineState::UnpackLogicOp(
            state.dynamic_state.logic_op.Value());
    }
    if (state.viewport_swizzles[0] != 0) {
        // A stored swizzle of viewport 0 (zero: the record has none).
        regs.viewport_scale_offset_enabled = 1;
        regs.viewport_transform[0].swizzle.raw = state.viewport_swizzles[0];
    }
    if (record.has_mesh) {
        regs.reg_array[Maxwell::MeshControl] = 1;
    }
}

/// GraphicsPipelineKey::attribute_types of a stored key (vk_pipeline_cache's MakeRuntimeInfo).
u64 AttributeTypes(const Vulkan::FixedPipelineState& state) {
    if (state.dynamic_vertex_input != 0) {
        return state.attribute_types;
    }
    u64 types = 0;
    for (size_t index = 0; index < state.attributes.size(); ++index) {
        const auto& attribute = state.attributes[index];
        u64 type = 0;
        if (attribute.enabled != 0) {
            switch (attribute.Type()) {
            case Maxwell::VertexAttribute::Type::UnusedEnumDoNotUseBecauseItWillGoAway:
                break;
            case Maxwell::VertexAttribute::Type::SInt:
                type = 2;
                break;
            case Maxwell::VertexAttribute::Type::UInt:
                type = 3;
                break;
            default:
                type = 1;
                break;
            }
        }
        types |= type << (index * 2);
    }
    return types;
}

GraphicsPipelineKey KeyOfRecord(Maxwell& regs, const Record& record) {
    RegsOfRecord(regs, record);
    GraphicsPipelineKey key;
    key.unique_hashes = record.graphics.unique_hashes;
    PipelineTranslator::RefreshKey(key, regs, record.graphics.state.topology.Value());
    key.attribute_types = AttributeTypes(record.graphics.state);
    return key;
}

// What the rasterizer does with a draw, from the registers alone.

/// PrimitiveType (metal_graphics.mm) has a value.
bool IsNative(Topology topology) {
    switch (topology) {
    case Topology::Points:
    case Topology::Lines:
    case Topology::LineStrip:
    case Topology::Triangles:
    case Topology::TriangleStrip:
        return true;
    default:
        return false;
    }
}

std::optional<PixelFormat> DepthPixelFormat(Tegra::DepthFormat format) {
    switch (format) {
    case Tegra::DepthFormat::Z24_UNORM_S8_UINT:
    case Tegra::DepthFormat::S8Z24_UNORM:
    case Tegra::DepthFormat::Z32_FLOAT:
    case Tegra::DepthFormat::Z16_UNORM:
    case Tegra::DepthFormat::S8_UINT:
    case Tegra::DepthFormat::Z32_FLOAT_X24S8_UINT:
    case Tegra::DepthFormat::X8Z24_UNORM:
        return VideoCore::Surface::PixelFormatFromDepthFormat(format);
    default:
        return std::nullopt;
    }
}

u32 SampleCount(Tegra::Texture::MsaaMode mode) {
    using Tegra::Texture::MsaaMode;
    const u32 raw = static_cast<u32>(mode);
    if (raw > static_cast<u32>(MsaaMode::Msaa4x2_VC24) ||
        (raw > static_cast<u32>(MsaaMode::Msaa4x4) &&
         raw < static_cast<u32>(MsaaMode::Msaa2x2_VC4))) {
        return 1;
    }
    return static_cast<u32>(VideoCommon::NumSamples(mode));
}

/// The Metal format of a colour target, 0 with `problem` set when the target would be dropped.
u32 ColorFormat(Tegra::RenderTargetFormat format, std::string& problem) {
    const auto pixel_format = VideoCore::Surface::TryPixelFormatFromRenderTargetFormat(format);
    if (!pixel_format) {
        problem = fmt::format("render target format {:#x} is unknown", static_cast<u32>(format));
        return 0;
    }
    const FormatInfo info = SurfaceFormat(*pixel_format, false);
    if (info.format == 0 || !info.attachable) {
        problem = fmt::format(
            "render target format {:#x} (pixel format {}) has no Metal target format, target "
            "dropped",
            static_cast<u32>(format), static_cast<u32>(*pixel_format));
        return 0;
    }
    return info.format;
}

struct DepthTarget {
    u32 format{};
    bool has_depth{};
    bool has_stencil{};
};

DepthTarget DepthFormatOf(Tegra::DepthFormat format, std::string& problem) {
    auto pixel_format = DepthPixelFormat(format);
    if (!pixel_format) {
        // PixelFormatFromDepthFormat's answer, which both backends get from the texture cache.
        problem = fmt::format("depth format {:#x} is unknown: taken as S8_UINT_D24_UNORM, as "
                              "Vulkan takes it",
                              static_cast<u32>(format));
        pixel_format = PixelFormat::S8_UINT_D24_UNORM;
    }
    const FormatInfo info = SurfaceFormat(*pixel_format, false);
    if (info.format == 0) {
        problem = fmt::format("depth format {:#x} has no Metal format, target dropped",
                              static_cast<u32>(format));
        return {};
    }
    const FormatClass format_class = ClassOf(info.format);
    return DepthTarget{
        .format = info.format,
        .has_depth =
            format_class == FormatClass::Depth || format_class == FormatClass::DepthStencil,
        .has_stencil =
            format_class == FormatClass::Stencil || format_class == FormatClass::DepthStencil,
    };
}

struct DrawLayout {
    size_t flags{};
    size_t topology{};
    size_t index_format{};
    size_t color_targets{};
    size_t zeta_target{};
    std::array<size_t, 6> hi{};
    std::array<size_t, 6> lo{};
    size_t ranges{};
};

DrawLayout LayoutOf(const CensusCategory& draws) {
    DrawLayout layout{
        .flags = draws.Column("flags"),
        .topology = draws.Column("topology"),
        .index_format = draws.Column("index_format"),
        .color_targets = draws.Column("color_targets"),
        .zeta_target = draws.Column("zeta_target"),
    };
    for (size_t slot = 0; slot < SLOT_NAMES.size(); ++slot) {
        layout.hi[slot] = draws.Column(fmt::format("{}.hi", SLOT_NAMES[slot]));
        layout.lo[slot] = draws.Column(fmt::format("{}.lo", SLOT_NAMES[slot]));
    }
    // The registers follow in DrawRanges' order; a census of another build has other columns.
    size_t column = layout.zeta_target + 1;
    layout.ranges = column;
    for (const Census::DrawRange& range : Census::DrawRanges()) {
        const std::string name =
            range.count > 1 ? fmt::format("{}[0]", range.name) : std::string{range.name};
        if (column >= draws.columns.size() || draws.columns[column] != name) {
            throw std::runtime_error{fmt::format(
                "the census' draw column {} is not {}: written by another build", column, name)};
        }
        column += range.count;
    }
    if (column != draws.columns.size()) {
        throw std::runtime_error{"the census' draws have columns this build doesn't know"};
    }
    return layout;
}

struct DrawRow {
    u64 count{};
    u32 flags{};
    Topology topology{};
    u32 index_format{};
    u32 color_targets{};
    u32 zeta_target{};
    Hashes hashes{};
    size_t job{};
};

DrawRow RowOf(const DrawLayout& layout, const CensusRow& row) {
    DrawRow draw{
        .count = row.count,
        .flags = row.words[layout.flags],
        .topology = static_cast<Topology>(row.words[layout.topology]),
        .index_format = row.words[layout.index_format],
        .color_targets = row.words[layout.color_targets],
        .zeta_target = row.words[layout.zeta_target],
    };
    for (size_t slot = 0; slot < draw.hashes.size(); ++slot) {
        draw.hashes[slot] = u64{row.words[layout.hi[slot]]} << 32 | row.words[layout.lo[slot]];
    }
    return draw;
}

void FillRegs(Maxwell& regs, const DrawLayout& layout, std::span<const u32> words) {
    regs.reg_array.fill(0);
    size_t column = layout.ranges;
    for (const Census::DrawRange& range : Census::DrawRanges()) {
        for (u32 index = 0; index < range.count; ++index) {
            regs.reg_array[range.first + index * range.step] = words[column++];
        }
    }
}

struct DrawResults {
    Tallies paths;
    Tallies refusals;
    Tallies findings;
    std::unordered_set<u64> pipeline_states;
    std::unordered_set<u64> depth_states;
    Tessellation tessellation; ///< ApplyTessellation's argument
    u64 rows{};
    u64 draws{};
};

/// RasterizerMetal::Draw, DrawIndirect and PrepareDraw up to the pipeline state, with the
/// census' approximations: a target is bound when its format is set, a stream when it is enabled.
void ReplayDraw(DrawResults& results, const Maxwell& regs, const DrawRow& row,
                const Outcome& outcome) {
    const std::string example = HashText(row.hashes);
    ++results.rows;
    results.draws += row.count;
    const auto refuse = [&](std::string cause) {
        Count(results.refusals, std::move(cause), row.count, example);
    };
    if (!outcome.ok) {
        refuse(outcome.missing_env
                   ? "not replayed: a program of the draw has no environment in the caches"
                   : fmt::format("pipeline: {}", outcome.cause));
        return;
    }
    const bool indexed = row.index_format != ~0U;
    const bool u8_indices =
        indexed && row.index_format == static_cast<u32>(Maxwell::IndexFormat::UnsignedByte);
    std::string path;
    if ((row.flags & Census::DrawByteCount) != 0) {
        refuse("transform feedback draw");
        return;
    }
    if (outcome.mesh && (row.flags & Census::DrawIndirect) != 0) {
        path = outcome.native_mesh
                   ? "native: indirect mesh draw (records rewritten into Metal's by a kernel)"
                   : "emulated: indirect mesh draw (records read on the CPU), kernels (compute)";
    } else if (outcome.mesh && (row.flags & Census::DrawFromGpuArguments) != 0) {
        path = outcome.native_mesh
                   ? "native: mesh functions, task count the GPU wrote (indirect)"
                   : "emulated: task count read on the CPU, kernels (compute)";
    } else if ((row.flags & Census::DrawIndirect) != 0) {
        if (outcome.tessellated) {
            refuse("indirect tessellated draw");
            return;
        }
        if (u8_indices) {
            refuse("indirect draw with 8-bit indices");
            return;
        }
        if (!IsNative(row.topology)) {
            refuse("indirect draw of a topology Metal can't draw");
            return;
        }
        path = (row.flags & Census::DrawIndirectCount) != 0
                   ? "emulated: indirect loop (draw count from a buffer)"
                   : "native: indirect (a loop when the draw count is above one)";
    } else if (outcome.mesh) {
        path = outcome.native_mesh ? outcome.has_task ? "native: object and mesh functions"
                                                      : "native: mesh function"
               : outcome.has_task  ? "emulated: task and mesh kernels (compute)"
                                   : "emulated: mesh kernel (compute)";
    } else if (outcome.tessellated) {
        path = "emulated: vertex and control kernels (compute), Metal tessellation";
    } else {
        const bool convert = IndexConvert::NeedsConversion(row.topology) || u8_indices;
        if (!IsNative(IndexConvert::ConvertedTopology(row.topology))) {
            refuse("topology Metal can't draw");
            return;
        }
        path = convert ? "emulated: rewritten indices" : "native";
    }

    static const int function{};
    auto desc = Zeroed<RenderPipelineDesc>();
    desc.vertex_function = &function;
    desc.fragment_function = &function;
    std::string problem;
    if (row.color_targets != 0) {
        const u32 count = (std::min)(regs.rt_control.count.Value(), u32{Maxwell::NumRenderTargets});
        for (u32 index = 0; index < count; ++index) {
            const Tegra::RenderTargetFormat format = regs.rt[regs.rt_control.Map(index)].format;
            if (format == Tegra::RenderTargetFormat::NONE) {
                continue;
            }
            desc.color_formats[index] = ColorFormat(format, problem);
            if (desc.color_formats[index] == 0) {
                Count(results.findings, problem, row.count, example);
            }
        }
    }
    DepthTarget depth;
    if (row.zeta_target != 0) {
        problem.clear();
        depth = DepthFormatOf(regs.zeta.format, problem);
        if (!problem.empty()) {
            Count(results.findings, problem, row.count, example);
        }
    }
    desc.depth_format = depth.has_depth ? depth.format : 0U;
    desc.stencil_format = depth.has_stencil ? depth.format : 0U;
    desc.samples = SampleCount(regs.anti_alias_samples_mode);
    FillRegisterState(desc, regs,
                      outcome.mesh ? static_cast<Topology>(outcome.mesh_topology) : row.topology,
                      outcome.mesh);
    for (size_t index = 0; index < Maxwell::NumVertexArrays; ++index) {
        const auto& stream = regs.vertex_streams[index];
        FillStream(desc, regs, index, stream.enable != 0 && !outcome.mesh, stream.stride.Value());
    }
    const size_t variant =
        !outcome.tessellated || !indexed ? 0
        : row.index_format == static_cast<u32>(Maxwell::IndexFormat::UnsignedShort) ? 1
                                                                                    : 2;
    const VertexProgramUse use = outcome.mesh ? VertexProgramUse{} : outcome.vertex_use[variant];
    VertexSources sources;
    std::string reason;
    if (!SanitizeVertexLayout(desc, use.loaded, use.resource_slots, sources, reason)) {
        refuse(fmt::format("vertex layout: {}", reason));
        return;
    }
    if (outcome.tessellated) {
        reason = CheckVertexLayout(desc);
        if (!reason.empty()) {
            refuse(fmt::format("vertex kernel's stage-in layout: {}", reason));
            return;
        }
        results.tessellation.modes = outcome.modes;
        ApplyTessellation(desc, results.tessellation);
        if (outcome.modes.point_mode) {
            Count(results.findings, "tessellation with point output, drawn as triangles", row.count,
                  example);
        }
    }
    reason = CheckPipeline(desc, LIMITS);
    if (!reason.empty()) {
        refuse(fmt::format("render pipeline state: {}", reason));
        return;
    }
    Count(results.paths, std::move(path), row.count);
    desc.vertex_function = nullptr;
    desc.fragment_function = nullptr;
    results.pipeline_states.insert(
        Common::CityHash64(reinterpret_cast<const char*>(&desc), sizeof(desc)) ^
        (u64{row.job} + 1) * 0x9E3779B97F4A7C15ULL);
    const DepthStencilDesc depth_desc = DepthStencilState(regs, depth.has_depth, depth.has_stencil);
    results.depth_states.insert(
        Common::CityHash64(reinterpret_cast<const char*>(&depth_desc), sizeof(depth_desc)));
}

/// The state checks a stored key allows: targets, sample count and the stored topology.
void CheckRecord(Tallies& findings, Tallies& paths, const Record& record,
                 const Outcome& outcome) {
    const Vulkan::FixedPipelineState& state = record.graphics.state;
    const std::string example = HashText(record.graphics.unique_hashes);
    std::string problem;
    for (const u8 format : state.color_formats) {
        if (format != 0 &&
            ColorFormat(static_cast<Tegra::RenderTargetFormat>(format), problem) == 0) {
            Count(findings, problem, 1, example);
        }
    }
    if (state.depth_enabled != 0) {
        problem.clear();
        DepthFormatOf(static_cast<Tegra::DepthFormat>(state.depth_format.Value()), problem);
        if (!problem.empty()) {
            Count(findings, problem, 1, example);
        }
    }
    const u32 samples = SampleCount(state.msaa_mode.Value());
    if ((LIMITS.sample_counts >> samples & 1) == 0) {
        Count(findings, fmt::format("{} samples per pixel", samples), 1, example);
    }
    if (!outcome.ok) {
        return;
    }
    const Topology topology = state.topology.Value();
    if (outcome.mesh) {
        Count(paths,
              outcome.native_mesh ? outcome.has_task ? "native: object and mesh functions"
                                                     : "native: mesh function"
              : outcome.has_task  ? "emulated: task and mesh kernels (compute)"
                                  : "emulated: mesh kernel (compute)",
              1);
    } else if (outcome.tessellated) {
        Count(paths, "emulated: vertex and control kernels (compute), Metal tessellation", 1);
    } else if (!IsNative(IndexConvert::ConvertedTopology(topology))) {
        Count(findings, "topology Metal can't draw", 1, example);
    } else {
        Count(paths, IndexConvert::NeedsConversion(topology) ? "emulated: rewritten indices"
                                                             : "native topology",
              1);
    }
}

void TallyOutcome(Tallies& results, Tallies& details, const Outcome& outcome, u64 weight,
                  std::string_view example, std::string_view kind) {
    if (outcome.ok) {
        Count(results,
              fmt::format("translated{}: {}", outcome.guessed ? " with guessed reads" : "", kind),
              weight);
        return;
    }
    const std::string cause = fmt::format(
        "refused: {}{}", outcome.cause, outcome.guessed ? " (with guessed reads)" : "");
    Count(results, cause, weight, example);
    if (!outcome.detail.empty()) {
        Count(details, fmt::format("{}: {}", outcome.cause, OneLine(outcome.detail, 200)), weight,
              example);
    }
}

std::string_view KindOf(const Outcome& outcome) {
    if (outcome.mesh) {
        if (outcome.native_mesh) {
            if (!outcome.mesh_emulated) {
                return outcome.has_task ? "task and mesh, Metal mesh functions only (no emulation)"
                                        : "mesh, Metal mesh function only (no emulation)";
            }
            return outcome.has_task ? "task and mesh, Metal mesh functions (emulation as fallback)"
                                    : "mesh, Metal mesh function (emulation as fallback)";
        }
        return outcome.has_task ? "task and mesh, emulated" : "mesh, emulated";
    }
    return outcome.tessellated ? "tessellation" : "vertex and fragment";
}

/// The task and mesh programs of the translated keys: which are drawn with Metal mesh functions
/// in every pipeline they are in, and why the others are not; the native ones' shapes.
void ReportMeshPrograms(std::string& out, const Jobs& jobs) {
    struct Program {
        bool task{};
        u64 native{};
        u64 emulated{};
        std::string cause;
    };
    std::map<u64, Program> programs;
    NativeMesh largest{};
    Tallies causes;
    for (const GraphicsJob& job : jobs.graphics) {
        const Outcome& outcome = job.outcome;
        if (!outcome.ok || !outcome.mesh) {
            continue;
        }
        for (const u64 hash : {outcome.mesh_hash, outcome.task_hash}) {
            if (hash == 0) {
                continue;
            }
            Program& program = programs[hash];
            program.task = hash == outcome.task_hash;
            (outcome.native_mesh ? program.native : program.emulated) += 1;
            if (!outcome.native_mesh && program.cause.empty()) {
                program.cause = outcome.native_cause;
            }
        }
        if (outcome.native_mesh) {
            const NativeMesh& shape = outcome.native_shape;
            largest.payload_bytes = (std::max)(largest.payload_bytes, shape.payload_bytes);
            largest.object_threads = (std::max)(largest.object_threads, shape.object_threads);
            largest.mesh_threads = (std::max)(largest.mesh_threads, shape.mesh_threads);
            largest.threadgroup_bytes =
                (std::max)(largest.threadgroup_bytes, shape.threadgroup_bytes);
        } else {
            Count(causes, outcome.native_cause.empty() ? "(no cause)" : outcome.native_cause, 1,
                  HashText(job.key.unique_hashes));
        }
    }
    std::array<std::array<u64, 3>, 2> counts{};
    for (const auto& [hash, program] : programs) {
        const size_t column = program.emulated == 0 ? 0 : program.native == 0 ? 2 : 1;
        ++counts[program.task ? 1 : 0][column];
    }
    Line(out, "");
    Line(out, "Task and mesh programs as Metal object and mesh functions (Metal: {} vertices, {} "
              "primitives, {} B payload, {} threads, {} B threadgroup memory)",
         NativeMeshLimits::MaxVertices, NativeMeshLimits::MaxPrimitives,
         NativeMeshLimits::MaxPayloadBytes, NativeMeshLimits::MaxThreads,
         NativeMeshLimits::MaxThreadgroupBytes);
    for (const bool task : {false, true}) {
        const auto& row = counts[task ? 1 : 0];
        Line(out, "  {} programs: {} native in every pipeline, {} in some, {} in none",
             task ? "task" : "mesh", row[0], row[1], row[2]);
    }
    Line(out, "  largest native: payload {} B, object {} threads, mesh {} threads, ~{} B mesh "
              "threadgroup memory",
         largest.payload_bytes, largest.object_threads, largest.mesh_threads,
         largest.threadgroup_bytes);
    Print(out, "Mesh pipelines not native, by cause", causes, "keys");
}

void ReplayClears(std::string& out, const CensusCategory& clears) {
    const size_t surface = clears.Column("clear_surface");
    const size_t control = clears.Column("clear_control");
    const size_t scissor = clears.Column("scissor_test.enable");
    const size_t stencil_mask = clears.Column("stencil_front_mask");
    // As the rasterizer's EDEN_NX2_METAL_CLEAR_DRAWS.
    const char* const value = std::getenv("EDEN_NX2_METAL_CLEAR_DRAWS");
    const bool draws = value == nullptr || value[0] != '0';
    Tallies tallies;
    for (const CensusRow& row : clears.rows) {
        const u32 raw = row.words[surface];
        const u32 channels = raw >> 2 & 0xF;
        const bool stencil = (raw & 2) != 0;
        const u32 mask = row.words[stencil_mask];
        if (channels == 0 && (raw & 3) == 0) {
            Count(tallies, "nothing to clear", row.count);
        } else if (channels != 0 && channels != 0xF) {
            Count(tallies,
                  draws ? "drawn as a triangle: clear of some colour channels"
                        : "refused: clear of some colour channels",
                  row.count);
        } else if (stencil && mask != 0xFF && mask != 0) {
            Count(tallies,
                  draws ? "drawn as a triangle: clear of some stencil bits"
                        : "refused: clear of some stencil bits",
                  row.count);
        } else if ((row.words[control] >> 8 & 1) != 0 && row.words[scissor] != 0) {
            Count(tallies,
                  draws ? "native when the scissor covers the target, else drawn as a triangle "
                          "(the rectangle is not in the census)"
                        : "native when the scissor covers the target, else refused: clear of "
                          "part of the target (the rectangle is not in the census)",
                  row.count);
        } else {
            Count(tallies, "native", row.count);
        }
    }
    Print(out, "Clears", tallies, "states", "clears");
}

void ReplayImages(std::string& out, const CensusCategory& images) {
    const size_t format = images.Column("format");
    Tallies tallies;
    for (const CensusRow& row : images.rows) {
        const auto pixel_format = static_cast<PixelFormat>(row.words[format]);
        if (row.words[format] >= static_cast<u32>(PixelFormat::Max)) {
            Count(tallies, fmt::format("pixel format {} is unknown", row.words[format]),
                  row.count);
        } else if (SurfaceFormat(pixel_format, false).format == 0) {
            Count(tallies,
                  fmt::format("pixel format {} has no Metal format, image not made",
                              row.words[format]),
                  row.count);
        } else {
            Count(tallies, "has a Metal format", row.count);
        }
    }
    Print(out, "Images by format", tallies, "kinds", "images");
}

struct Arguments {
    std::filesystem::path census;
    std::vector<std::filesystem::path> caches;
    std::vector<std::filesystem::path> spirv;
    std::filesystem::path msl_out;
    std::filesystem::path report;
    bool ios{true};
    /// Task/mesh pipelines as Metal mesh functions, as on a GPU with Metal 3 mesh pipelines.
    bool native_mesh{true};
    u32 apple_family{10};
    /// --title <hex>: the program id translation sees (title-specific edits, O07).
    u64 title{};
    u32 jobs{(std::max)(std::thread::hardware_concurrency() / 2, 1U)};
};

std::optional<Arguments> ParseArguments(int argc, char** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; ++index) {
        const std::string_view name = argv[index];
        if (name == "--macos") {
            arguments.ios = false;
            continue;
        }
        if (name == "--no-native-mesh") {
            arguments.native_mesh = false;
            continue;
        }
        if (index + 1 >= argc) {
            return std::nullopt;
        }
        const char* const value = argv[++index];
        if (name == "--census") {
            arguments.census = value;
        } else if (name == "--cache") {
            arguments.caches.emplace_back(value);
        } else if (name == "--spirv") {
            arguments.spirv.emplace_back(value);
        } else if (name == "--msl-out") {
            arguments.msl_out = value;
        } else if (name == "--report") {
            arguments.report = value;
        } else if (name == "--apple-family") {
            arguments.apple_family = static_cast<u32>((std::max)(std::atoi(value), 0));
        } else if (name == "--title") {
            arguments.title = std::strtoull(value, nullptr, 16);
        } else if (name == "--jobs") {
            arguments.jobs = static_cast<u32>((std::max)(std::atoi(value), 1));
        } else {
            return std::nullopt;
        }
    }
    if (arguments.caches.empty() && arguments.spirv.empty()) {
        return std::nullopt;
    }
    return arguments;
}

/// --spirv: SPIR-V files of any origin (<name>.vert.spv, .frag.spv or .comp.spv) through the
/// renderer's translation, the MSL or the refusal printed. 1 when one was refused.
int TranslateFiles(const Arguments& arguments) {
    int result = 0;
    for (const std::filesystem::path& path : arguments.spirv) {
        const std::string stage_name = path.stem().extension().string();
        const ShaderTranslate::Stage stage =
            stage_name == ".vert"   ? ShaderTranslate::Stage::Vertex
            : stage_name == ".frag" ? ShaderTranslate::Stage::Fragment
            : stage_name == ".task" ? ShaderTranslate::Stage::Object
            : stage_name == ".mesh" ? ShaderTranslate::Stage::Mesh
                                    : ShaderTranslate::Stage::Compute;
        std::ifstream file{path, std::ios::binary};
        const std::string bytes{std::istreambuf_iterator<char>{file},
                                std::istreambuf_iterator<char>{}};
        std::vector<u32> words(bytes.size() / sizeof(u32));
        std::memcpy(words.data(), bytes.data(), words.size() * sizeof(u32));
        ShaderTranslate::Options options;
        options.ios = arguments.ios;
        options.apple_family = arguments.apple_family;
        const ShaderTranslate::Result msl = ShaderTranslate::SpirvToMsl(words, stage, options);
        if (msl.source.empty()) {
            fmt::print("// {}: refused: {}\n", path.filename().string(), msl.error);
            result = 1;
        } else {
            fmt::print("// {}: entry point {}\n{}\n", path.filename().string(), msl.entry_point,
                       msl.source);
        }
    }
    return result;
}

int Run(const Arguments& arguments) {
    std::string out;
    Line(out,
         "lichii-metal-replay: MSL for {}, GPU family Apple{}{}, default settings (no resolution "
         "scaling)",
         arguments.ios ? "iOS" : "macOS", arguments.apple_family,
         ShaderTranslate::DeviceMslEnabled() ? "" : " (EDEN_NX2_METAL_MSL_DEVICE=0: not used)");

    Caches caches;
    for (const std::filesystem::path& path : arguments.caches) {
        fmt::print(stderr, "reading {}\n", path.string());
        ReadCache(caches, path);
    }
    Line(out, "");
    Line(out, "Pipeline caches (later ones are taken as newer)");
    for (const std::string& note : caches.notes) {
        Line(out, "  {}", note);
    }
    Line(out, "  {} graphics records, {} compute records", caches.graphics, caches.compute);
    if (caches.inconsistent != 0) {
        Line(out, "  {} graphics records whose programs and environments don't match, left out",
             caches.inconsistent);
    }

    std::optional<CensusData> census;
    if (!arguments.census.empty()) {
        census = ReadCensus(arguments.census);
    }

    // Stored keys first, so a key the census shares takes its record's environments.
    Jobs jobs;
    const auto regs = std::make_unique<Maxwell>();
    std::vector<std::pair<const Record*, size_t>> record_jobs;
    for (const auto& record : caches.records) {
        if (record->is_compute) {
            ++jobs.compute[ComputeJobOf(jobs, caches, record->compute.unique_hash)].records;
        } else if (record->consistent) {
            const size_t job =
                GraphicsJobOf(jobs, caches, KeyOfRecord(*regs, *record), record.get());
            ++jobs.graphics[job].records;
            record_jobs.emplace_back(record.get(), job);
        }
    }
    const size_t stored_graphics = jobs.graphics.size();
    const size_t stored_compute = jobs.compute.size();

    DrawLayout layout;
    std::vector<DrawRow> draws;
    std::vector<std::pair<size_t, const CensusRow*>> dispatches;
    if (census) {
        if (const auto it = census->find("draws"); it != census->end() && !it->second.rows.empty()) {
            layout = LayoutOf(it->second);
            for (const CensusRow& row : it->second.rows) {
                DrawRow draw = RowOf(layout, row);
                FillRegs(*regs, layout, row.words);
                GraphicsPipelineKey key;
                key.unique_hashes = draw.hashes;
                PipelineTranslator::RefreshKey(key, *regs, draw.topology);
                draw.job = GraphicsJobOf(jobs, caches, key, nullptr);
                ++jobs.graphics[draw.job].census_rows;
                draws.push_back(draw);
            }
        }
        if (const auto it = census->find("dispatches");
            it != census->end() && !it->second.rows.empty()) {
            const size_t hi = it->second.Column("compute.hi");
            const size_t lo = it->second.Column("compute.lo");
            for (const CensusRow& row : it->second.rows) {
                const size_t job =
                    ComputeJobOf(jobs, caches, u64{row.words[hi]} << 32 | row.words[lo]);
                ++jobs.compute[job].census_rows;
                dispatches.emplace_back(job, &row);
            }
        }
    }

    fmt::print(stderr, "{} graphics keys, {} compute programs, {} threads\n", jobs.graphics.size(),
               jobs.compute.size(), arguments.jobs);
    MslSink sink{arguments.msl_out};
    RunJobs(jobs, arguments.ios, arguments.apple_family, arguments.native_mesh, arguments.jobs,
            sink);

    // The stored pipelines.
    {
        Tallies results;
        Tallies details;
        for (size_t index = 0; index < stored_graphics; ++index) {
            const GraphicsJob& job = jobs.graphics[index];
            TallyOutcome(results, details, job.outcome, job.records,
                         HashText(job.key.unique_hashes), KindOf(job.outcome));
        }
        Line(out, "");
        Line(out, "Stored graphics pipelines: {} records, {} distinct Metal keys",
             record_jobs.size(), stored_graphics);
        Print(out, "Translation of the stored graphics keys", results, "keys", "records");
        Print(out, "What the failing step said", details, "keys", "records");

        Tallies findings;
        Tallies paths;
        for (const auto& [record, job] : record_jobs) {
            CheckRecord(findings, paths, *record, jobs.graphics[job].outcome);
        }
        Print(out,
              "Stored graphics records by how they would be drawn (the stored topology stands "
              "for its class under dynamic state)",
              paths, "records");
        Print(out, "Stored graphics records: target, sample count and topology problems", findings,
              "records");

        Tallies compute_results;
        Tallies compute_details;
        for (size_t index = 0; index < stored_compute; ++index) {
            const ComputeJob& job = jobs.compute[index];
            TallyOutcome(compute_results, compute_details, job.outcome, job.records,
                         fmt::format("compute={:016x}", job.hash), "compute");
        }
        Print(out, "Translation of the stored compute programs", compute_results, "programs",
              "records");
        Print(out, "What the failing step said", compute_details, "programs", "records");

        Tallies features;
        for (size_t index = 0; index < stored_graphics; ++index) {
            const GraphicsJob& job = jobs.graphics[index];
            for (u32 bit = 0; bit < ProgramFeatureCount; ++bit) {
                if ((job.outcome.features >> bit) & 1) {
                    Count(features, fmt::format("graphics: {}", ProgramFeatureNames[bit]),
                          job.records, HashText(job.key.unique_hashes));
                }
            }
        }
        for (size_t index = 0; index < stored_compute; ++index) {
            const ComputeJob& job = jobs.compute[index];
            for (u32 bit = 0; bit < ProgramFeatureCount; ++bit) {
                if ((job.outcome.features >> bit) & 1) {
                    Count(features, fmt::format("compute: {}", ProgramFeatureNames[bit]),
                          job.records, fmt::format("compute={:016x}", job.hash));
                }
            }
        }
        Print(out, "Program features of the translated pipelines (texture rows)", features,
              "pipelines", "records");
    }
    ReportMeshPrograms(out, jobs);

    if (census) {
        Line(out, "");
        Line(out, "Census {}", arguments.census.filename().string());
        for (const auto& [name, category] : *census) {
            Line(out, "  {:<12} {:>7} rows, {:>11} noted, {} dropped", name, category.rows.size(),
                 category.total, category.dropped);
        }

        Tallies results;
        Tallies details;
        for (size_t index = 0; index < jobs.graphics.size(); ++index) {
            const GraphicsJob& job = jobs.graphics[index];
            if (job.census_rows != 0) {
                TallyOutcome(results, details, job.outcome, job.census_rows,
                             HashText(job.key.unique_hashes), KindOf(job.outcome));
            }
        }
        Print(out, "Translation of the census' graphics keys", results, "keys", "draw states");
        Print(out, "What the failing step said", details, "keys", "draw states");

        DrawResults draw_results;
        if (const auto it = census->find("draws"); it != census->end()) {
            for (size_t index = 0; index < draws.size(); ++index) {
                FillRegs(*regs, layout, it->second.rows[index].words);
                ReplayDraw(draw_results, *regs, draws[index],
                           jobs.graphics[draws[index].job].outcome);
            }
        }
        Line(out, "");
        Line(out,
             "Census draws: {} states, {} draws; {} distinct render pipeline states, {} distinct "
             "depth-stencil states",
             draw_results.rows, draw_results.draws, draw_results.pipeline_states.size(),
             draw_results.depth_states.size());
        Print(out, "Draws by path", draw_results.paths, "states", "draws");
        Print(out, "Draws refused", draw_results.refusals, "states", "draws");
        Print(out, "Draws drawn with something missing", draw_results.findings, "states", "draws");

        if (const auto it = census->find("clears"); it != census->end() && !it->second.rows.empty()) {
            ReplayClears(out, it->second);
        }

        Tallies dispatch_tallies;
        Tallies dispatch_details;
        if (!dispatches.empty()) {
            const size_t indirect = census->find("dispatches")->second.Column("indirect");
            for (const auto& [job, row] : dispatches) {
                const Outcome& outcome = jobs.compute[job].outcome;
                const std::string example = fmt::format("compute={:016x}", jobs.compute[job].hash);
                if (outcome.ok) {
                    Count(dispatch_tallies,
                          fmt::format("native{}{}", row->words[indirect] != 0 ? ", indirect" : "",
                                      outcome.guessed ? " (translated with guessed reads)" : ""),
                          row->count);
                } else if (outcome.missing_env) {
                    Count(dispatch_tallies,
                          "not replayed: the program has no environment in the caches", row->count,
                          example);
                } else {
                    Count(dispatch_tallies, fmt::format("refused: {}", outcome.cause), row->count,
                          example);
                    Count(dispatch_details,
                          fmt::format("{}: {}", outcome.cause, OneLine(outcome.detail, 200)),
                          row->count, example);
                }
            }
        }
        Print(out, "Dispatches", dispatch_tallies, "programs", "dispatches");
        Print(out, "What the failing step said", dispatch_details, "programs", "dispatches");

        if (const auto it = census->find("misc"); it != census->end() && !it->second.rows.empty()) {
            const size_t kind = it->second.Column("kind");
            Tallies tallies;
            for (const CensusRow& row : it->second.rows) {
                const u32 value = row.words[kind];
                const std::string name = value < MISC_NAMES.size() ? MISC_NAMES[value]
                                                                   : fmt::format("kind {}", value);
                Count(tallies, value == 0 ? fmt::format("{}: refused", name) : name, row.count);
            }
            Print(out, "Other rasterizer calls (only DrawTexture is judged here)", tallies, "rows",
                  "calls");
        }
        if (const auto it = census->find("images"); it != census->end() && !it->second.rows.empty()) {
            ReplayImages(out, it->second);
        }
    }

    Line(out, "");
    Line(out, "MSL: {} distinct functions, {} bytes{}", sink.functions, sink.bytes,
         arguments.msl_out.empty() ? "" : ", written to the --msl-out directory");
    Line(out, "  Not compiled: this tool has no Metal compiler. Compile the --msl-out files on a "
              "Mac (tools/nx2/metal-replay.md).");
    Line(out, "");
    Line(out, "What the replay takes for granted");
    Line(out, "  - Translation with default settings; the programs' reads come from the stored");
    Line(out, "    environments, and a value none of them has is guessed where noted.");
    Line(out, "  - Sample counts 1, 2 and 4 are taken, as Apple's GPUs do.");
    Line(out, "  - A stored key has no blending, vertex layout or exact topology under dynamic");
    Line(out, "    state: those are checked for census draws only.");
    Line(out, "  - Census draws: a colour target is bound when its format is set, a vertex stream");
    Line(out, "    when it is enabled; sizes, scissor rectangles and draw counts are not known.");
    Line(out, "  - Dispatches the renderer replaces (the Ocarina upscale) are counted as native.");
    Line(out, "  - Queries, copies, image views and samplers are not replayed.");

    std::fputs(out.c_str(), stdout);
    if (!arguments.report.empty()) {
        std::ofstream file{arguments.report, std::ios::binary | std::ios::trunc};
        file << out;
    }
    return 0;
}

} // Anonymous namespace
} // namespace Metal

int main(int argc, char** argv) {
    const auto arguments = Metal::ParseArguments(argc, argv);
    if (!arguments) {
        std::fputs("usage: lichii-metal-replay --cache <pipeline cache file> [--cache <newer "
                   "file> ...] [--census <nx2_feature_census.json>] [--msl-out <directory>] "
                   "[--report <file>] [--macos] [--no-native-mesh] [--apple-family <n>] "
                   "[--title <hex>] [--jobs <n>]\n"
                   "       lichii-metal-replay --spirv <file.vert.spv, .frag.spv, .comp.spv, "
                   ".task.spv or .mesh.spv> [--spirv ...] [--macos] [--apple-family <n>]\n",
                   stderr);
        return 2;
    }
    try {
        if (!arguments->spirv.empty()) {
            return Metal::TranslateFiles(*arguments);
        }
        Settings::SetCurrentProgramID(arguments->title);
        const int result = Metal::Run(*arguments);
        if (arguments->title != 0) {
            fmt::print("programs edited by the Ocarina visual workaround "
                       "(EDEN_NX2_OOT_VISUAL_WORKAROUND=1): {}\n",
                       Metal::PipelineTranslator::VisualWorkaroundEdits());
        }
        return result;
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "lichii-metal-replay: %s\n", exception.what());
        return 1;
    }
}
