// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "common/alignment.h"
#include "common/cityhash.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/exception.h"
#include "video_core/control/channel_state.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/framebuffer_config.h"
#include "video_core/gpu.h"
#include "video_core/host1x/host1x.h"
#include "video_core/memory_manager.h"
#include "video_core/oot_network.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_index_convert.h"
#include "video_core/renderer_metal/metal_ocarina.h"
#include "video_core/renderer_metal/metal_rasterizer.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/shader_handle_program.h"
#include "video_core/textures/texture.h"

namespace Metal {

namespace {
/// Logs a message the first time lag is seen unset.
#define LOG_ONCE(flag, ...)                                                                        \
    do {                                                                                           \
        static std::atomic_bool flag{false};                                                       \
        if (!flag.exchange(true)) {                                                                \
            LOG_WARNING(Render_Metal, __VA_ARGS__);                                                \
        }                                                                                          \
    } while (false)
} // Anonymous namespace

AccelerateDMA::AccelerateDMA(BufferCache& buffer_cache_, TextureCache& texture_cache_)
    : buffer_cache{buffer_cache_}, texture_cache{texture_cache_} {}

bool AccelerateDMA::BufferCopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount) {
    std::scoped_lock lock{buffer_cache.mutex};
    return buffer_cache.DMACopy(src_address, dest_address, amount);
}

bool AccelerateDMA::BufferClear(GPUVAddr src_address, u64 amount, u32 value) {
    std::scoped_lock lock{buffer_cache.mutex};
    return buffer_cache.DMAClear(src_address, amount, value);
}

template <bool IS_IMAGE_UPLOAD>
bool AccelerateDMA::DmaBufferImageCopy(const Tegra::DMA::ImageCopy& copy_info,
                                       const Tegra::DMA::BufferOperand& buffer_operand,
                                       const Tegra::DMA::ImageOperand& image_operand) {
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    const auto image_id = texture_cache.DmaImageId(image_operand, IS_IMAGE_UPLOAD);
    if (image_id == VideoCommon::NULL_IMAGE_ID) {
        return false;
    }
    const u32 buffer_size = static_cast<u32>(buffer_operand.pitch * buffer_operand.height);
    static constexpr auto sync_info = VideoCommon::ObtainBufferSynchronize::FullSynchronize;
    const auto post_op = IS_IMAGE_UPLOAD ? VideoCommon::ObtainBufferOperation::DoNothing
                                         : VideoCommon::ObtainBufferOperation::MarkAsWritten;
    const auto [buffer, offset] =
        buffer_cache.ObtainBuffer(buffer_operand.address, buffer_size, sync_info, post_op);

    const auto [image, copy] = texture_cache.DmaBufferImageCopy(
        copy_info, buffer_operand, image_operand, image_id, IS_IMAGE_UPLOAD);
    const std::span copy_span{&copy, 1};

    if constexpr (IS_IMAGE_UPLOAD) {
        texture_cache.PrepareImage(image_id, true, false);
        image->UploadMemory(buffer->Handle(), offset, copy_span);
    } else {
        if (offset % VideoCore::Surface::BytesPerBlock(image->info.format)) {
            return false;
        }
        texture_cache.DownloadImageIntoBuffer(image, buffer->Handle(), offset, copy_span,
                                              buffer_operand.address, buffer_size);
    }
    return true;
}

bool AccelerateDMA::ImageToBuffer(const Tegra::DMA::ImageCopy& copy_info,
                                  const Tegra::DMA::ImageOperand& image_operand,
                                  const Tegra::DMA::BufferOperand& buffer_operand) {
    return DmaBufferImageCopy<false>(copy_info, buffer_operand, image_operand);
}

bool AccelerateDMA::BufferToImage(const Tegra::DMA::ImageCopy& copy_info,
                                  const Tegra::DMA::BufferOperand& buffer_operand,
                                  const Tegra::DMA::ImageOperand& image_operand) {
    return DmaBufferImageCopy<true>(copy_info, buffer_operand, image_operand);
}

RasterizerMetal::RasterizerMetal(Tegra::GPU& gpu_, Tegra::MaxwellDeviceMemoryManager& device_memory,
                                 const Device& device_, Scheduler& scheduler_)
    : gpu{gpu_}, device{device_}, scheduler{scheduler_}, staging_pool{device_, scheduler_},
      texture_cache_runtime{device_, scheduler_, staging_pool},
      texture_cache{texture_cache_runtime, device_memory},
      buffer_cache_runtime{device_, scheduler_, staging_pool},
      buffer_cache{device_memory, buffer_cache_runtime}, pipeline_cache{device_memory, device_},
      ocarina{device_, scheduler_},
      accelerate_dma{buffer_cache, texture_cache},
      fence_manager{*this, gpu_, texture_cache, buffer_cache, query_cache, scheduler_} {}

RasterizerMetal::~RasterizerMetal() = default;

void RasterizerMetal::Draw(bool is_indexed, u32 instance_count) {
    ++draws;
    GraphicsPipeline* const pipeline = pipeline_cache.CurrentGraphicsPipeline();
    if (pipeline == nullptr || !pipeline->BuildFunctions(device)) {
        return;
    }
    ++draws_with_pipeline;
    // EDEN_NX2_METAL_DRAW=0: build pipelines but encode no draw (milestone 2 behaviour).
    static const bool encode = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_DRAW");
        return value == nullptr || value[0] != '0';
    }();
    if (!encode) {
        return;
    }
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    index_rewrite = {};
    if (const MeshEmulation* const mesh = pipeline->Mesh()) {
        // The draw's count is its mesh workgroups (the index count for indexed draws).
        const u32 groups = is_indexed ? draw_state.index_buffer.count
                                      : draw_state.vertex_buffer.count;
        if (groups == 0) {
            return;
        }
        gpu_memory->FlushCaching();
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        // Written texel buffers read through aligned copies go back once the work is encoded.
        SCOPE_EXIT {
            FinishTexelWritebacks();
        };
        if (!PrepareDraw(*pipeline, is_indexed, groups)) {
            return;
        }
        CountOcclusion();
        const auto topology = static_cast<Maxwell::PrimitiveTopology>(mesh->topology);
        if (mesh->has_task) {
            // The distribution pass wrote the vertex count.
            scheduler.DrawIndirect(*PrimitiveType(topology), mesh_buffers.args.buffer,
                                   mesh_buffers.args.offset +
                                       Shader::Backend::SPIRV::MESH_EMULATION_DRAW_ARGS_OFFSET);
        } else {
            scheduler.Draw(*PrimitiveType(topology), 0, groups * mesh->vertices_per_group, 1, 0);
        }
        ++draws_encoded;
        ++mesh_draws;
        mesh_groups += groups;
        return;
    }
    const bool u8_indices =
        is_indexed && draw_state.index_buffer.format == Maxwell::IndexFormat::UnsignedByte;
    const bool convert = IndexConvert::NeedsConversion(draw_state.topology) || u8_indices;
    const std::optional<u32> primitive =
        PrimitiveType(IndexConvert::ConvertedTopology(draw_state.topology));
    if (!primitive) {
        LOG_ONCE(warned_topology, "Metal: topology {} is not drawn yet",
                 static_cast<u32>(draw_state.topology));
        return;
    }
    const u32 count = is_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count;
    const u32 index_size =
        is_indexed ? static_cast<u32>(draw_state.index_buffer.FormatSizeInBytes()) : 0U;
    const u32 converted = convert ? IndexConvert::ConvertedCount(draw_state.topology, count) : 0U;
    const bool restart = maxwell3d->regs.primitive_restart.enabled != 0;
    if (convert && converted == 0) {
        return;
    }
    gpu_memory->FlushCaching();
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    // Written texel buffers read through aligned copies go back once the work is encoded.
    SCOPE_EXIT {
        FinishTexelWritebacks();
    };
    // Indices the GPU wrote are not in guest memory yet: a kernel rewrites them from the bound
    // index buffer before the render pass (EDEN_NX2_METAL_GPU_INDICES=1 for every indexed
    // rewrite, =0 never).
    static const char gpu_indices = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_GPU_INDICES");
        return value != nullptr ? value[0] : 'a';
    }();
    if (convert && is_indexed && draw_state.inline_index_draw_indexes.empty() &&
        gpu_indices != '0') {
        const GPUVAddr start = draw_state.index_buffer.StartAddress() +
                               GPUVAddr{draw_state.index_buffer.first} * index_size;
        const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(start);
        index_rewrite.wanted =
            gpu_indices == '1' ||
            (device_addr && buffer_cache.IsRegionGpuModified(*device_addr,
                                                             size_t{count} * index_size));
        index_rewrite.params = {
            .mode = IndexConvert::RewriteMode(draw_state.topology),
            .index_size = index_size,
            .count = count,
            .restart = restart ? 1U : 0U,
            .converted = converted,
            .source_offset = draw_state.index_buffer.first * index_size,
        };
    }
    if (!PrepareDraw(*pipeline, is_indexed)) {
        return;
    }
    CountOcclusion();
    if (convert) {
        // 32-bit indices in the stream ring: from the GPU rewrite, or rewritten on the CPU from
        // the guest's indices (or the vertex range).
        StagingBufferRef indices = index_rewrite.output;
        if (indices.buffer != nullptr) {
            ++indices_rewritten_on_gpu;
        } else {
            std::span<const u8> source;
            if (is_indexed) {
                if (!draw_state.inline_index_draw_indexes.empty()) {
                    source = draw_state.inline_index_draw_indexes;
                } else {
                    converted_source.resize(size_t{count} * index_size);
                    gpu_memory->ReadBlockUnsafe(
                        draw_state.index_buffer.StartAddress() +
                            GPUVAddr{draw_state.index_buffer.first} * index_size,
                        converted_source.data(), converted_source.size());
                    source = converted_source;
                }
            }
            indices = staging_pool.Stream(size_t{converted} * sizeof(u32), 4);
            const std::span<u32> out(reinterpret_cast<u32*>(indices.mapped_span.data()),
                                     converted);
            IndexConvert::Convert(draw_state.topology, source, index_size,
                                  is_indexed ? 0U : draw_state.vertex_buffer.first, count, restart,
                                  out);
        }
        scheduler.DrawIndexed(*primitive, converted, 4, indices.buffer, indices.offset,
                              instance_count,
                              is_indexed ? static_cast<s32>(draw_state.base_index) : 0,
                              draw_state.base_instance);
        ++draws_converted;
    } else if (is_indexed) {
        const BufferCacheRuntime::IndexState& index = buffer_cache_runtime.Index();
        scheduler.DrawIndexed(*primitive, draw_state.index_buffer.count, index_size, index.buffer,
                              u64{index.offset} + u64{draw_state.index_buffer.first} * index_size,
                              instance_count, static_cast<s32>(draw_state.base_index),
                              draw_state.base_instance);
    } else {
        scheduler.Draw(*primitive, draw_state.vertex_buffer.first, draw_state.vertex_buffer.count,
                       instance_count, draw_state.base_instance);
    }
    ++draws_encoded;
}

std::vector<TextureHandle> RasterizerMetal::PrepareTexelBuffers(std::span<const Binding> bindings,
                                                                const Shader::Info& info) {
    // The buffer cache appends a stage's texture buffers, then its image buffers.
    const size_t read_only = Shader::NumDescriptors(info.texture_buffer_descriptors);
    std::vector<TextureHandle> result;
    for (const Binding& binding : bindings) {
        if (binding.kind != Binding::Kind::TexelBuffer) {
            continue;
        }
        const bool writable = result.size() >= read_only;
        const VideoCore::Surface::PixelFormat guest_format =
            binding.format == VideoCore::Surface::PixelFormat::Invalid
                ? VideoCore::Surface::PixelFormat::R8_UNORM
                : binding.format;
        const FormatInfo format = SurfaceFormat(device, guest_format, false);
        const u32 bytes_per_texel = VideoCore::Surface::BytesPerBlock(guest_format);
        const u32 elements = bytes_per_texel != 0 ? binding.size / bytes_per_texel : 0;
        if (format.format == 0 || elements == 0 || binding.buffer == nullptr) {
            result.push_back(nullptr);
            continue;
        }
        const u32 alignment = (std::max)(TextureBufferAlignment(device, format.format), 1U);
        BufferHandle buffer = binding.buffer;
        u64 offset = binding.offset;
        if (offset % alignment != 0) {
            // Through an aligned copy (ends an open render pass: counted as a texel copy). A
            // written one is copied back after the draw or dispatch (FinishTexelWritebacks).
            const StagingBufferRef copy = staging_pool.Stream(binding.size, alignment);
            scheduler.CopyBuffer(binding.buffer, binding.offset, copy.buffer, copy.offset,
                                 binding.size);
            if (writable) {
                texel_writebacks.push_back({copy.buffer, copy.offset, binding.buffer,
                                            binding.offset, binding.size});
            }
            buffer = copy.buffer;
            offset = copy.offset;
            ++texel_copies;
        }
        const TexelViewKey key{buffer, offset, elements, format.format, writable};
        Object& view = texel_view_cache[key];
        if (!view) {
            view = CreateTextureBufferView(buffer, offset, elements, bytes_per_texel, format.format,
                                           writable);
        }
        result.push_back(view.Get());
    }
    return result;
}

void RasterizerMetal::RewriteIndices() {
    const BufferCacheRuntime::IndexState& index = buffer_cache_runtime.Index();
    if (!index_kernel && !index_kernel_failed) {
        std::string error;
        index_kernel_library =
            CreateLibrary(device, IndexConvert::RewriteKernelSource(), error);
        const Object function = index_kernel_library
                                    ? CreateFunction(index_kernel_library.Get(), "rewrite_indices")
                                    : Object{};
        index_kernel =
            function ? CreateComputePipelineState(device, function.Get(), error) : Object{};
        if (!index_kernel) {
            index_kernel_failed = true;
            LOG_ERROR(Render_Metal, "Metal: index rewrite kernel failed to build: {}", error);
        }
    }
    if (!index_kernel || index.buffer == nullptr) {
        // The CPU rewrites them from guest memory instead.
        return;
    }
    IndexConvert::RewriteParams params = index_rewrite.params;
    params.source_offset += index.offset;
    index_rewrite.output = staging_pool.Stream(size_t{params.converted} * sizeof(u32), 4);
    scheduler.BeginCompute(index_kernel.Get());
    scheduler.BindBuffer(ShaderStage::Compute, 0, index.buffer, 0);
    scheduler.BindBuffer(ShaderStage::Compute, 1, index_rewrite.output.buffer,
                         static_cast<u32>(index_rewrite.output.offset));
    scheduler.BindBytes(ShaderStage::Compute, 2, &params, static_cast<u32>(sizeof(params)));
    constexpr u32 GroupSize = 64;
    scheduler.Dispatch({(params.converted + GroupSize - 1) / GroupSize, 1, 1}, {GroupSize, 1, 1});
}

void RasterizerMetal::BindMeshBuffers(const StageProgram& program, ShaderStage metal_stage,
                                      const MeshEmulation& mesh) {
    // The output buffer's binding, then (with a task program) payload, offsets and arguments.
    const std::array<const StagingBufferRef*, 4> buffers{
        &mesh_buffers.output, &mesh_buffers.payload, &mesh_buffers.offsets, &mesh_buffers.args};
    const size_t count = mesh.has_task ? buffers.size() : 1;
    for (const ShaderTranslate::ResourceSlot& slot : program.slots) {
        if (slot.kind != ShaderTranslate::ResourceKind::Buffer ||
            slot.set == ShaderTranslate::PushConstantSet || slot.binding < mesh.output_binding ||
            slot.binding - mesh.output_binding >= count) {
            continue;
        }
        const StagingBufferRef& buffer = *buffers[slot.binding - mesh.output_binding];
        scheduler.BindBuffer(metal_stage, slot.index, buffer.buffer,
                             static_cast<u32>(buffer.offset));
    }
}

RasterizerMetal::MeshArena& RasterizerMetal::AcquireMeshArena() {
    const u64 tick = scheduler.CurrentTick();
    if (mesh_arena_current < mesh_arenas.size() && mesh_arenas[mesh_arena_current].tick == tick) {
        return mesh_arenas[mesh_arena_current];
    }
    // At most four command buffers in flight hold one: past that, wait for the oldest.
    constexpr size_t MaxMeshArenas = 4;
    const auto reusable = [this](const MeshArena& arena) {
        return arena.counted || scheduler.IsFree(arena.tick);
    };
    if (mesh_arenas.size() >= MaxMeshArenas && std::ranges::none_of(mesh_arenas, reusable)) {
        scheduler.Wait(std::ranges::min_element(mesh_arenas, {}, &MeshArena::tick)->tick);
    }
    // A finished command buffer's counter holds the words its draws asked for, including those
    // that did not fit: later arenas grow to hold them.
    constexpr u64 MaxArenaSize = u64{1} << 30;
    for (MeshArena& arena : mesh_arenas) {
        if (arena.counted || !scheduler.IsFree(arena.tick)) {
            continue;
        }
        arena.counted = true;
        u32 words;
        std::memcpy(&words, arena.counter_contents, sizeof(words));
        const u64 needed = u64{words} * sizeof(u32);
        if (needed > arena.size && mesh_arena_size < MaxArenaSize) {
            mesh_arena_size = (std::min)(std::bit_ceil(needed + needed / 4), MaxArenaSize);
            LOG_INFO(Render_Metal,
                     "Metal: mesh emulation output of a command buffer needed {} MiB, arenas grow "
                     "to {} MiB",
                     needed >> 20, mesh_arena_size >> 20);
        }
    }
    std::erase_if(mesh_arenas, [this](const MeshArena& arena) {
        return arena.counted && arena.size != mesh_arena_size;
    });
    auto it = std::ranges::find_if(mesh_arenas, &MeshArena::counted);
    if (it == mesh_arenas.end()) {
        MeshArena arena{.size = mesh_arena_size};
        arena.buffer = CreateBuffer(device, mesh_arena_size, false, nullptr);
        arena.counter = CreateBuffer(device, sizeof(u32), true, &arena.counter_contents);
        it = mesh_arenas.insert(mesh_arenas.end(), std::move(arena));
    }
    // The GPU is done with it, and this command buffer has not been committed.
    std::memset(it->counter_contents, 0, sizeof(u32));
    it->tick = tick;
    it->counted = false;
    mesh_arena_current = static_cast<size_t>(std::distance(mesh_arenas.begin(), it));
    return *it;
}

void RasterizerMetal::FinishTexelWritebacks() {
    for (const TexelWriteback& writeback : texel_writebacks) {
        scheduler.CopyBuffer(writeback.copy, writeback.copy_offset, writeback.buffer,
                             writeback.offset, writeback.size);
    }
    texel_writebacks.clear();
}

void RasterizerMetal::BindStageResources(StageProgram& program, ShaderStage metal_stage,
                                         std::span<const Binding> pending,
                                         std::span<const TextureHandle> texel_buffer_views,
                                         std::span<const VideoCommon::ImageViewInOut> views,
                                         std::span<const VideoCommon::SamplerId> samplers,
                                         u32 render_width, u32 render_height) {
    const Shader::Info& info = program.info;
    std::vector<u32> texture_slots;
    std::vector<u32> sampler_slots;
    size_t next_buffer = 0;
    for (const ShaderTranslate::ResourceSlot& slot : program.slots) {
        for (u32 element = 0; element < slot.count; ++element) {
            const u32 slot_index = slot.index + element;
            switch (slot.kind) {
            case ShaderTranslate::ResourceKind::Buffer:
                if (slot.set == ShaderTranslate::PushConstantSet) {
                    // Shader::Backend::SPIRV::RescalingLayout (no rescaling, down factor 1)
                    // or RenderAreaLayout.
                    std::array<u32, 8> words{};
                    if (info.uses_render_area && !info.uses_rescaling_uniform) {
                        words[0] = std::bit_cast<u32>(static_cast<f32>(render_width));
                        words[1] = std::bit_cast<u32>(static_cast<f32>(render_height));
                    } else {
                        words[6] = std::bit_cast<u32>(1.0f);
                    }
                    scheduler.BindBytes(metal_stage, slot_index, words.data(),
                                        static_cast<u32>(sizeof(words)));
                    break;
                }
                while (next_buffer < pending.size() &&
                       pending[next_buffer].kind != Binding::Kind::Buffer) {
                    ++next_buffer;
                }
                if (next_buffer < pending.size()) {
                    scheduler.BindBuffer(metal_stage, slot_index, pending[next_buffer].buffer,
                                         pending[next_buffer].offset);
                    ++next_buffer;
                }
                break;
            case ShaderTranslate::ResourceKind::Texture:
                texture_slots.push_back(slot_index);
                break;
            case ShaderTranslate::ResourceKind::Sampler:
                sampler_slots.push_back(slot_index);
                break;
            }
        }
    }

    // Texture slots in binding order: texel buffers, image buffers, textures, images.
    const size_t texel_buffers = Shader::NumDescriptors(info.texture_buffer_descriptors) +
                                 Shader::NumDescriptors(info.image_buffer_descriptors);
    for (size_t element = 0; element < texel_buffers && element < texture_slots.size(); ++element) {
        scheduler.BindTexture(metal_stage, texture_slots[element],
                              element < texel_buffer_views.size() ? texel_buffer_views[element]
                                                                  : nullptr);
    }
    size_t texture_slot = texel_buffers;
    size_t view = texel_buffers;
    size_t sampler = 0;
    for (const auto& desc : info.texture_descriptors) {
        for (u32 element = 0; element < desc.count; ++element) {
            ImageView& image_view = texture_cache.GetImageView(views[view++].id);
            if (texture_slot < texture_slots.size()) {
                scheduler.BindTexture(metal_stage, texture_slots[texture_slot],
                                      image_view.Handle(desc.type));
            }
            ++texture_slot;
            if (sampler < sampler_slots.size()) {
                scheduler.BindSampler(
                    metal_stage, sampler_slots[sampler],
                    texture_cache.GetSampler(samplers[sampler])
                        .Handle());
            }
            ++sampler;
        }
    }
    for (const auto& desc : info.image_descriptors) {
        for (u32 element = 0; element < desc.count; ++element) {
            ImageView& image_view = texture_cache.GetImageView(views[view++].id);
            if (desc.is_written) {
                texture_cache.MarkModification(image_view.image_id);
            }
            if (texture_slot < texture_slots.size()) {
                scheduler.BindTexture(metal_stage, texture_slots[texture_slot],
                                      image_view.Handle(desc.type));
            }
            ++texture_slot;
        }
    }
}

bool RasterizerMetal::PrepareDraw(GraphicsPipeline& pipeline, bool is_indexed, u32 mesh_workgroups) {
    // Buffer cache stage indices (vertex B, the mesh program after a task program, pixel) and
    // the translated programs. An emulated mesh pipeline's stage 0 is its task or mesh kernel,
    // dispatched before the render pass.
    MeshEmulation* const mesh = pipeline.Mesh();
    const bool has_task = mesh != nullptr && mesh->has_task;
    const std::array<size_t, 3> stages{0, has_task ? size_t{mesh->mesh_stage} : 0, 4};
    const std::array<StageProgram*, 3> programs{
        mesh != nullptr ? &mesh->first.program : &pipeline.Vertex(),
        has_task ? &mesh->mesh.program : nullptr, &pipeline.Fragment()};
    const auto absent = [&](size_t index) {
        return programs[index] == nullptr || programs[index]->msl.empty();
    };
    const auto& regs = maxwell3d->regs;
    const Maxwell::PrimitiveTopology topology =
        mesh != nullptr ? static_cast<Maxwell::PrimitiveTopology>(mesh->topology)
                        : maxwell3d->draw_manager.draw_state.topology;

    std::array<u32, VideoCommon::NUM_STAGES> uniform_masks{};
    for (size_t index = 0; index < stages.size(); ++index) {
        if (absent(index)) {
            continue;
        }
        const Shader::Info& info = programs[index]->info;
        uniform_masks[stages[index]] = info.constant_buffer_mask;
        std::ranges::copy_n(info.constant_buffer_used_sizes.begin(),
                            uniform_buffer_sizes[stages[index]].size(),
                            uniform_buffer_sizes[stages[index]].begin());
    }
    texture_cache.SynchronizeDescriptors(false);
    buffer_cache.SetUniformBuffersState(uniform_masks, &uniform_buffer_sizes);

    // Storage buffers, and every texture and image handle, as OpenGL's ConfigureImpl reads them.
    std::vector<VideoCommon::ImageViewInOut> views;
    std::vector<VideoCommon::SamplerId> samplers;
    std::array<size_t, 3> first_view{};
    std::array<size_t, 3> first_sampler{};
    const bool via_header_index = regs.sampler_binding == Maxwell::SamplerBinding::ViaHeaderBinding;
    for (size_t index = 0; index < stages.size(); ++index) {
        first_view[index] = views.size();
        first_sampler[index] = samplers.size();
        if (absent(index)) {
            continue;
        }
        const size_t stage = stages[index];
        const Shader::Info& info = programs[index]->info;
        buffer_cache.UnbindGraphicsStorageBuffers(stage);
        size_t ssbo_index = 0;
        for (const auto& desc : info.storage_buffers_descriptors) {
            buffer_cache.BindGraphicsStorageBuffer(stage, ssbo_index++, desc.cbuf_index,
                                                   desc.cbuf_offset, desc.is_written);
        }
        const auto& cbufs = maxwell3d->state.shader_stages[stage].const_buffers;
        const auto read_handle = [&](const auto& desc, u32 element) {
            if (desc.handle_program != ~0U) {
                return Tegra::Texture::TexturePair(
                    VideoCommon::EvaluateTextureHandle(
                        info, desc.handle_program, *gpu_memory,
                        [&](u32 cbuf) -> GPUVAddr {
                            return cbuf < cbufs.size() && cbufs[cbuf].enabled
                                       ? cbufs[cbuf].address
                                       : GPUVAddr{};
                        },
                        buffer_cache_runtime.GetStorageBufferAlignment()),
                    via_header_index);
            }
            const u32 offset = desc.cbuf_offset + (element << desc.size_shift);
            const GPUVAddr addr = cbufs[desc.cbuf_index].address + offset;
            if constexpr (requires { desc.has_secondary; }) {
                if (desc.has_secondary) {
                    const GPUVAddr second = cbufs[desc.secondary_cbuf_index].address +
                                            desc.secondary_cbuf_offset;
                    const u32 raw = (gpu_memory->Read<u32>(addr) << desc.shift_left) |
                                    (gpu_memory->Read<u32>(second) << desc.secondary_shift_left);
                    return Tegra::Texture::TexturePair(raw, via_header_index);
                }
            }
            return Tegra::Texture::TexturePair(gpu_memory->Read<u32>(addr), via_header_index);
        };
        for (const auto& desc : info.texture_buffer_descriptors) {
            for (u32 element = 0; element < desc.count; ++element) {
                views.push_back({.index = read_handle(desc, element).first});
            }
        }
        for (const auto& desc : info.image_buffer_descriptors) {
            for (u32 element = 0; element < desc.count; ++element) {
                views.push_back({.index = read_handle(desc, element).first});
            }
        }
        for (const auto& desc : info.texture_descriptors) {
            for (u32 element = 0; element < desc.count; ++element) {
                const auto handle = read_handle(desc, element);
                views.push_back({.index = handle.first});
                samplers.push_back(texture_cache.GetSamplerId(handle.second, false));
            }
        }
        for (const auto& desc : info.image_descriptors) {
            for (u32 element = 0; element < desc.count; ++element) {
                views.push_back(
                    {.index = read_handle(desc, element).first, .blacklist = desc.is_written});
            }
        }
    }
    texture_cache.FillImageViews(views, false, true);
    texture_cache.UpdateRenderTargets(false);
    const Framebuffer* const framebuffer = texture_cache.GetFramebuffer();
    if (framebuffer == nullptr) {
        return false;
    }

    // Texel buffers come from the buffer cache.
    for (size_t index = 0; index < stages.size(); ++index) {
        if (absent(index)) {
            continue;
        }
        const size_t stage = stages[index];
        const Shader::Info& info = programs[index]->info;
        buffer_cache.UnbindGraphicsTextureBuffers(stage);
        size_t view = first_view[index];
        size_t tbo_index = 0;
        const auto add = [&](const auto& desc, bool is_image) {
            for (u32 element = 0; element < desc.count; ++element) {
                ImageView& image_view = texture_cache.GetImageView(views[view++].id);
                bool is_written = false;
                if constexpr (requires { desc.is_written; }) {
                    is_written = desc.is_written;
                }
                buffer_cache.BindGraphicsTextureBuffer(stage, tbo_index++, image_view.GpuAddr(),
                                                       image_view.BufferSize(), image_view.format,
                                                       is_written, is_image);
            }
        };
        for (const auto& desc : info.texture_buffer_descriptors) {
            add(desc, false);
        }
        for (const auto& desc : info.image_buffer_descriptors) {
            add(desc, true);
        }
    }
    buffer_cache.UpdateGraphicsBuffers(is_indexed);
    buffer_cache.BindHostGeometryBuffers(is_indexed);

    // Each stage's buffer bindings, and texel buffer views (copies of misaligned ranges go
    // before the render pass).
    std::array<std::vector<Binding>, 3> stage_bindings;
    std::array<std::vector<TextureHandle>, 3> texel_views;
    for (size_t index = 0; index < stages.size(); ++index) {
        if (absent(index)) {
            continue;
        }
        buffer_cache_runtime.PendingBindings().clear();
        buffer_cache.BindHostStageBuffers(stages[index]);
        stage_bindings[index] = buffer_cache_runtime.PendingBindings();
        texel_views[index] = PrepareTexelBuffers(stage_bindings[index], programs[index]->info);
    }

    // The render pipeline state: targets, blending, vertex layout.
    RenderTargetSet targets{};
    auto state_desc = Zeroed<RenderPipelineDesc>();
    for (size_t rt = 0; rt < Maxwell::NumRenderTargets; ++rt) {
        targets.color[rt] = framebuffer->Colors()[rt];
        state_desc.color_formats[rt] = framebuffer->Colors()[rt].texture != nullptr
                                     ? framebuffer->ColorFormats()[rt]
                                     : 0U;
        const auto& mask = regs.color_mask[regs.color_mask_common ? 0 : rt];
        BlendTargetState& blend = state_desc.blend[rt];
        blend.write_mask = (mask.R != 0 ? 1U : 0U) | (mask.G != 0 ? 2U : 0U) |
                           (mask.B != 0 ? 4U : 0U) | (mask.A != 0 ? 8U : 0U);
        const bool per_target = regs.blend_per_target_enabled != 0;
        blend.enable = regs.blend.enable[per_target ? rt : 0] != 0 ? 1U : 0U;
        if (per_target) {
            const auto& src = regs.blend_per_target[rt];
            blend.color_op = src.color_op;
            blend.alpha_op = src.alpha_op;
            blend.color_source = src.color_source;
            blend.color_dest = src.color_dest;
            blend.alpha_source = src.alpha_source;
            blend.alpha_dest = src.alpha_dest;
        } else {
            blend.color_op = regs.blend.color_op;
            blend.alpha_op = regs.blend.alpha_op;
            blend.color_source = regs.blend.color_source;
            blend.color_dest = regs.blend.color_dest;
            blend.alpha_source = regs.blend.alpha_source;
            blend.alpha_dest = regs.blend.alpha_dest;
        }
    }
    targets.depth = framebuffer->Depth();
    targets.has_depth = framebuffer->HasDepth();
    targets.has_stencil = framebuffer->HasStencil();
    targets.width = framebuffer->Width();
    targets.height = framebuffer->Height();
    state_desc.depth_format = targets.has_depth ? framebuffer->DepthFormat() : 0U;
    state_desc.stencil_format = targets.has_stencil ? framebuffer->DepthFormat() : 0U;
    state_desc.samples = framebuffer->Samples();
    state_desc.topology_class = TopologyClass(topology);
    state_desc.rasterize = regs.rasterize_enable != 0 ? 1U : 0U;
    state_desc.alpha_to_coverage = regs.anti_alias_alpha_control.alpha_to_coverage != 0 ? 1U : 0U;
    for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
        const auto& input = regs.vertex_attrib_format[index];
        VertexAttributeState& attribute = state_desc.attributes[index];
        // The mesh emulation's vertex shader reads its output buffer, no vertex streams.
        attribute.enabled = input.constant != 0 || mesh != nullptr ? 0U : 1U;
        attribute.stream = input.buffer;
        attribute.offset = input.offset;
        attribute.type = input.type;
        attribute.size = input.size;
    }
    const auto& vertex_states = buffer_cache_runtime.Vertices();
    for (size_t index = 0; index < Maxwell::NumVertexArrays; ++index) {
        VertexStreamState& stream = state_desc.streams[index];
        stream.enabled = vertex_states[index].buffer != nullptr && mesh == nullptr ? 1U : 0U;
        stream.stride = vertex_states[index].stride;
        stream.divisor = regs.vertex_stream_instances.IsInstancingEnabled(index)
                             ? regs.vertex_streams[index].frequency
                             : 0U;
    }
    void* const state = pipeline.RenderState(device, state_desc);
    if (state == nullptr) {
        return false;
    }

    auto depth_stencil = Zeroed<DepthStencilDesc>();
    depth_stencil.depth_test = regs.depth_test_enable != 0 && targets.has_depth ? 1U : 0U;
    depth_stencil.depth_write = regs.depth_write_enabled != 0 ? 1U : 0U;
    depth_stencil.depth_func = regs.depth_test_func;
    depth_stencil.stencil_test = regs.stencil_enable != 0 && targets.has_stencil ? 1U : 0U;
    depth_stencil.front = {regs.stencil_front_op.fail, regs.stencil_front_op.zfail,
                           regs.stencil_front_op.zpass, regs.stencil_front_op.func,
                           regs.stencil_front_func_mask, regs.stencil_front_mask};
    depth_stencil.back = regs.stencil_two_side_enable != 0
                             ? StencilFaceState{regs.stencil_back_op.fail,
                                                regs.stencil_back_op.zfail,
                                                regs.stencil_back_op.zpass,
                                                regs.stencil_back_op.func,
                                                regs.stencil_back_func_mask,
                                                regs.stencil_back_mask}
                             : depth_stencil.front;
    const u64 depth_stencil_hash = Common::CityHash64(
        reinterpret_cast<const char*>(&depth_stencil), sizeof(depth_stencil));
    Object& depth_stencil_state = depth_stencil_states[depth_stencil_hash];
    if (!depth_stencil_state) {
        depth_stencil_state = CreateDepthStencilState(device, depth_stencil);
    }

    if (index_rewrite.wanted) {
        RewriteIndices();
    }
    if (mesh != nullptr) {
        void* const first = pipeline.KernelState(device, mesh->first);
        void* const mesh_kernel = has_task ? pipeline.KernelState(device, mesh->mesh) : nullptr;
        void* const distribution =
            has_task ? pipeline.KernelState(device, mesh->distribution) : nullptr;
        if (first == nullptr || mesh_workgroups == 0 ||
            (has_task && (mesh_kernel == nullptr || distribution == nullptr))) {
            return false;
        }
        namespace SPIRV = Shader::Backend::SPIRV;
        mesh_buffers = {};
        u32 capacity_words = 0;
        if (has_task) {
            // The mesh workgroups the tasks ask for are only known on the GPU: their output goes
            // to the command buffer's arena.
            MeshArena& arena = AcquireMeshArena();
            mesh_buffers.output = {.buffer = arena.buffer.Get()};
            mesh_buffers.counter = {.buffer = arena.counter.Get()};
            capacity_words = static_cast<u32>(arena.size / sizeof(u32));
            const auto stream = [&](u64 size) {
                return staging_pool.Stream((std::max)(size, u64{16}), 16);
            };
            mesh_buffers.payload = stream(u64{mesh_workgroups} * mesh->payload_stride * 4);
            mesh_buffers.offsets = stream(u64{mesh_workgroups} * 4);
            mesh_buffers.args = stream(SPIRV::MESH_EMULATION_ARGS_WORDS * 4);
        } else {
            mesh_buffers.output = staging_pool.Stream(
                (std::max)(u64{mesh_workgroups} * mesh->block_words * sizeof(u32), u64{16}), 16);
        }
        const auto begin_kernel = [&](size_t index, MeshKernel& kernel, void* kernel_state) {
            scheduler.BeginCompute(kernel_state);
            BindStageResources(kernel.program, ShaderStage::Compute, stage_bindings[index],
                               texel_views[index], std::span(views).subspan(first_view[index]),
                               std::span(samplers).subspan(first_sampler[index]), targets.width,
                               targets.height);
            BindMeshBuffers(kernel.program, ShaderStage::Compute, *mesh);
        };
        begin_kernel(0, mesh->first, first);
        scheduler.Dispatch({mesh_workgroups, 1, 1}, mesh->first.workgroup_size);
        if (has_task) {
            // The distribution pass: payload, offsets, arguments and the arena counter at
            // bindings 0-3, and its constants.
            const SPIRV::MeshEmulationDistributionConstants constants{
                .tasks = mesh_workgroups,
                .payload_words = mesh->payload_stride,
                .capacity_words = capacity_words,
                .block_words = mesh->block_words,
                .vertices_per_group = mesh->vertices_per_group,
                // Vulkan's guaranteed maxComputeWorkGroupCount[0].
                .max_groups = 65535,
            };
            const std::array<const StagingBufferRef*, 4> distribution_buffers{
                &mesh_buffers.payload, &mesh_buffers.offsets, &mesh_buffers.args,
                &mesh_buffers.counter};
            scheduler.BeginCompute(distribution);
            for (const ShaderTranslate::ResourceSlot& slot : mesh->distribution.program.slots) {
                if (slot.kind != ShaderTranslate::ResourceKind::Buffer) {
                    continue;
                }
                if (slot.set == ShaderTranslate::PushConstantSet) {
                    scheduler.BindBytes(ShaderStage::Compute, slot.index, &constants,
                                        static_cast<u32>(sizeof(constants)));
                } else if (slot.binding < distribution_buffers.size()) {
                    const StagingBufferRef& buffer = *distribution_buffers[slot.binding];
                    scheduler.BindBuffer(ShaderStage::Compute, slot.index, buffer.buffer,
                                         static_cast<u32>(buffer.offset));
                }
            }
            scheduler.Dispatch({1, 1, 1}, {1, 1, 1});
            begin_kernel(1, mesh->mesh, mesh_kernel);
            scheduler.DispatchIndirect(mesh_buffers.args.buffer, mesh_buffers.args.offset,
                                       mesh->mesh.workgroup_size);
        }
    }

    scheduler.BeginRendering(targets);
    DrawState draw{};
    draw.pipeline = state;
    draw.depth_stencil = depth_stencil_state.Get();
    // Viewport 0, as Vulkan's GetViewportState makes it (the shader flips y for Metal).
    const auto& transform = regs.viewport_transform[0];
    if (regs.viewport_scale_offset_enabled) {
        draw.viewport.x = transform.translate_x - transform.scale_x;
        draw.viewport.width = transform.scale_x * 2.0f;
        draw.viewport.y = transform.translate_y - transform.scale_y;
        draw.viewport.height = transform.scale_y * 2.0f;
        if (regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft) {
            draw.viewport.y += static_cast<f32>(regs.surface_clip.height);
            draw.viewport.height = -draw.viewport.height;
        }
        if (transform.swizzle.y == Maxwell::ViewportSwizzle::NegativeY) {
            draw.viewport.y += draw.viewport.height;
            draw.viewport.height = -draw.viewport.height;
        }
        const f32 reduce_z = regs.depth_mode == Maxwell::DepthMode::MinusOneToOne ? 1.0f : 0.0f;
        draw.viewport.znear = transform.translate_z - transform.scale_z * reduce_z;
        draw.viewport.zfar = transform.translate_z + transform.scale_z;
    } else {
        draw.viewport.x = static_cast<f32>(regs.surface_clip.x);
        draw.viewport.y = static_cast<f32>(regs.surface_clip.y);
        draw.viewport.width = static_cast<f32>(regs.surface_clip.width);
        draw.viewport.height = static_cast<f32>(regs.surface_clip.height);
    }
    if (draw.viewport.width == 0.0f) {
        draw.viewport.width = 1.0f;
    }
    if (draw.viewport.height == 0.0f) {
        draw.viewport.height = 1.0f;
    }
    const auto& scissor = regs.scissor_test[0];
    if (scissor.enable) {
        const bool lower_left = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft;
        const s32 clip_height = static_cast<s32>(regs.surface_clip.height);
        const s32 min_y = (std::max)(lower_left ? clip_height - static_cast<s32>(scissor.max_y)
                                                : static_cast<s32>(scissor.min_y.Value()),
                                     0);
        const s32 max_y = (std::max)(lower_left ? clip_height - static_cast<s32>(scissor.min_y)
                                                : static_cast<s32>(scissor.max_y.Value()),
                                     0);
        draw.scissor = {static_cast<u32>(scissor.min_x.Value()), static_cast<u32>(min_y),
                        static_cast<u32>(scissor.max_x.Value() - scissor.min_x.Value()),
                        static_cast<u32>((std::max)(max_y - min_y, 0))};
    } else {
        draw.scissor = {0, 0, targets.width, targets.height};
    }
    if (regs.gl_cull_test_enabled != 0) {
        switch (regs.gl_cull_face) {
        case Maxwell::CullFace::Front:
            draw.cull_mode = 1;
            break;
        case Maxwell::CullFace::Back:
            draw.cull_mode = 2;
            break;
        case Maxwell::CullFace::FrontAndBack:
            // Nothing is drawn but points and lines.
            if (TopologyClass(topology) == 2) {
                return false;
            }
            break;
        }
    }
    bool front_ccw = regs.gl_front_face == Maxwell::FrontFace::CounterClockWise;
    if (regs.window_origin.flip_y != 0) {
        front_ccw = !front_ccw;
    }
    draw.front_ccw = front_ccw;
    draw.depth_clamp =
        regs.viewport_clip_control.geometry_clip !=
            Maxwell::ViewportClipControl::GeometryClip::Passthrough &&
        regs.viewport_clip_control.geometry_clip !=
            Maxwell::ViewportClipControl::GeometryClip::FrustumXYZ &&
        regs.viewport_clip_control.geometry_clip !=
            Maxwell::ViewportClipControl::GeometryClip::FrustumZ;
    if (regs.polygon_offset_fill_enable != 0 || regs.polygon_offset_line_enable != 0 ||
        regs.polygon_offset_point_enable != 0) {
        // The hardware halves the depth bias units (as OpenGL and Vulkan apply it).
        draw.depth_bias = regs.depth_bias / 2.0f;
        draw.slope_scale = regs.slope_scale_depth_bias;
        draw.bias_clamp = regs.depth_bias_clamp;
    }
    draw.stencil_front_ref = regs.stencil_front_ref;
    draw.stencil_back_ref =
        regs.stencil_two_side_enable != 0 ? regs.stencil_back_ref : regs.stencil_front_ref;
    draw.blend_color = {regs.blend_color.r, regs.blend_color.g, regs.blend_color.b,
                        regs.blend_color.a};
    scheduler.SetDrawState(draw);

    for (size_t index = 0; index < vertex_states.size() && mesh == nullptr; ++index) {
        if (vertex_states[index].buffer != nullptr) {
            scheduler.BindVertexBuffer(VertexBufferSlot(static_cast<u32>(index)),
                                       vertex_states[index].buffer, vertex_states[index].offset);
        }
    }

    // Each stage's buffers (appended by the buffer cache in binding order), textures, samplers
    // and images, into the slots the translator gave them.
    for (size_t index = 0; index < stages.size(); ++index) {
        if (absent(index) || index == 1) {
            continue;
        }
        const ShaderStage metal_stage = index == 0 ? ShaderStage::Vertex : ShaderStage::Fragment;
        if (index == 0 && mesh != nullptr) {
            BindMeshBuffers(pipeline.Vertex(), metal_stage, *mesh);
            continue;
        }
        BindStageResources(*programs[index], metal_stage, stage_bindings[index], texel_views[index],
                           std::span(views).subspan(first_view[index]),
                           std::span(samplers).subspan(first_sampler[index]), targets.width,
                           targets.height);
    }
    return true;
}

void RasterizerMetal::DrawIndirect() {
    ++draws;
    GraphicsPipeline* const pipeline = pipeline_cache.CurrentGraphicsPipeline();
    if (pipeline == nullptr || !pipeline->BuildFunctions(device)) {
        return;
    }
    ++draws_with_pipeline;
    const auto& params = maxwell3d->draw_manager.indirect_state;
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    if (params.is_byte_count) {
        LOG_ONCE(warned_byte_count, "Metal: transform feedback draws are not drawn yet");
        return;
    }
    if (pipeline->Mesh() != nullptr) {
        // As on Vulkan: the emulation needs the workgroup count on the CPU (mesh draws whose
        // count the GPU wrote come through Draw, DrawMeshTasksFromGpu being false).
        LOG_ONCE(warned_mesh_indirect, "Metal: indirect emulated mesh draws are skipped");
        return;
    }
    const bool u8_indices = params.is_indexed && maxwell3d->regs.index_buffer.format ==
                                                     Maxwell::IndexFormat::UnsignedByte;
    const std::optional<u32> primitive = PrimitiveType(draw_state.topology);
    if (!primitive || u8_indices) {
        // The arguments are on the GPU: rewriting their indices needs a compute pass.
        LOG_ONCE(warned_indirect, "Metal: indirect draws of topology {} or 8-bit indices are not "
                                  "drawn yet",
                 static_cast<u32>(draw_state.topology));
        return;
    }
    gpu_memory->FlushCaching();
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    // Written texel buffers read through aligned copies go back once the work is encoded.
    SCOPE_EXIT {
        FinishTexelWritebacks();
    };
    index_rewrite = {};
    buffer_cache.SetDrawIndirect(&params);
    SCOPE_EXIT {
        buffer_cache.SetDrawIndirect(nullptr);
    };
    if (!PrepareDraw(*pipeline, params.is_indexed)) {
        return;
    }
    CountOcclusion();
    const auto [indirect, indirect_offset] = buffer_cache.GetDrawIndirectBuffer();
    size_t count = params.max_draw_counts;
    if (params.include_count) {
        // Metal has no draw count from a buffer: the guest's count as it is in memory.
        count = (std::min)(count, size_t{gpu_memory->Read<u32>(params.count_start_address)});
        ++indirect_counts_read;
    }
    const size_t stride = params.stride != 0
                              ? params.stride
                              : (params.is_indexed ? 5 : 4) * sizeof(u32);
    const BufferCacheRuntime::IndexState& index = buffer_cache_runtime.Index();
    const u32 index_size = index.format == Maxwell::IndexFormat::UnsignedShort ? 2U : 4U;
    for (size_t draw = 0; draw < count; ++draw) {
        const u64 offset = u64{indirect_offset} + draw * stride;
        if (params.is_indexed) {
            scheduler.DrawIndexedIndirect(*primitive, index_size, index.buffer, index.offset,
                                          indirect->Handle(), offset);
        } else {
            scheduler.DrawIndirect(*primitive, indirect->Handle(), offset);
        }
    }
    ++draws_encoded;
}

void RasterizerMetal::DrawTexture() {
    ++draws;
}

void RasterizerMetal::Clear(u32 layer_count) {
    if (!maxwell3d) {
        return;
    }
    gpu_memory->FlushCaching();
    const auto& regs = maxwell3d->regs;
    const bool any_color = regs.clear_surface.R || regs.clear_surface.G ||
                           regs.clear_surface.B || regs.clear_surface.A;
    const bool all_color = regs.clear_surface.R && regs.clear_surface.G &&
                           regs.clear_surface.B && regs.clear_surface.A;
    if (!any_color && !regs.clear_surface.Z && !regs.clear_surface.S) {
        return;
    }
    if (any_color && !all_color) {
        // A masked clear is a draw on Metal (milestone 3).
        static std::atomic_bool warned{false};
        if (!warned.exchange(true)) {
            LOG_WARNING(Render_Metal, "Metal: clears of some colour channels are not done yet");
        }
    }
    std::scoped_lock lock{texture_cache.mutex};
    texture_cache.UpdateRenderTargets(true);
    const Framebuffer* const framebuffer = texture_cache.GetFramebuffer();
    if (framebuffer == nullptr) {
        return;
    }
    ClearPass pass{};
    pass.width = framebuffer->Width();
    pass.height = framebuffer->Height();
    const u32 rt = regs.clear_surface.RT;
    if (all_color && rt < NUM_RT && framebuffer->Colors()[rt].texture != nullptr) {
        pass.color[rt] = framebuffer->Colors()[rt];
        pass.clear_color[rt] = true;
        for (size_t component = 0; component < 4; ++component) {
            pass.color_value[rt][component] = regs.clear_color[component];
        }
    }
    if (regs.clear_surface.Z && framebuffer->HasDepth()) {
        pass.depth = framebuffer->Depth();
        pass.clear_depth = true;
        pass.depth_value = regs.clear_depth;
    }
    if (regs.clear_surface.S && framebuffer->HasStencil()) {
        pass.stencil = framebuffer->Depth();
        pass.clear_stencil = true;
        pass.stencil_value = static_cast<u32>(regs.clear_stencil);
    }
    scheduler.Clear(pass);
    ++clears;
}

void RasterizerMetal::DispatchCompute() {
    ++dispatches;
    gpu_memory->FlushCaching();
    ComputePipeline* const pipeline = pipeline_cache.CurrentComputePipeline();
    void* const state = pipeline != nullptr ? pipeline->State(device) : nullptr;
    if (state == nullptr) {
        // As OpenGL and Vulkan report it: KeplerCompute::DispatchNativeCompute takes a silent
        // return for a successful dispatch.
        if (kepler_compute->IsNativeMode()) {
            throw Shader::RuntimeError("Native compute pipeline unavailable");
        }
        return;
    }
    ++dispatches_with_pipeline;
    static const bool encode = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_DISPATCH");
        return value == nullptr || value[0] != '0';
    }();
    if (!encode) {
        return;
    }
    StageProgram& program = pipeline->Program();
    const Shader::Info& info = program.info;
    const auto& qmd = kepler_compute->launch_description;
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    // Written texel buffers read through aligned copies go back once the work is encoded.
    SCOPE_EXIT {
        FinishTexelWritebacks();
    };

    VideoCommon::ComputeUniformBufferSizes uniform_sizes{};
    std::ranges::copy_n(info.constant_buffer_used_sizes.begin(), uniform_sizes.size(),
                        uniform_sizes.begin());
    buffer_cache.SetComputeUniformBufferState(info.constant_buffer_mask, &uniform_sizes);
    buffer_cache.UnbindComputeStorageBuffers();
    size_t ssbo_index = 0;
    for (const auto& desc : info.storage_buffers_descriptors) {
        buffer_cache.BindComputeStorageBuffer(ssbo_index++, desc.cbuf_index, desc.cbuf_offset,
                                              desc.is_written, desc.indirect_offset,
                                              desc.write_extent);
    }
    texture_cache.SynchronizeDescriptors(true);

    const auto& cbufs = qmd.const_buffer_config;
    const bool via_header_index = qmd.linked_tsc != 0;
    const auto read_handle = [&](const auto& desc, u32 element) {
        if (desc.handle_program != ~0U) {
            return Tegra::Texture::TexturePair(
                VideoCommon::EvaluateTextureHandle(
                    info, desc.handle_program, *gpu_memory,
                    [&](u32 cbuf) -> GPUVAddr {
                        return cbuf < cbufs.size() && ((qmd.const_buffer_enable_mask >> cbuf) & 1)
                                   ? cbufs[cbuf].Address()
                                   : GPUVAddr{};
                    },
                    buffer_cache_runtime.GetStorageBufferAlignment()),
                via_header_index);
        }
        const GPUVAddr addr =
            cbufs[desc.cbuf_index].Address() + desc.cbuf_offset + (element << desc.size_shift);
        if constexpr (requires { desc.has_secondary; }) {
            if (desc.has_secondary) {
                const GPUVAddr second =
                    cbufs[desc.secondary_cbuf_index].Address() + desc.secondary_cbuf_offset;
                const u32 raw = (gpu_memory->Read<u32>(addr) << desc.shift_left) |
                                (gpu_memory->Read<u32>(second) << desc.secondary_shift_left);
                return Tegra::Texture::TexturePair(raw, via_header_index);
            }
        }
        return Tegra::Texture::TexturePair(gpu_memory->Read<u32>(addr), via_header_index);
    };
    std::vector<VideoCommon::ImageViewInOut> views;
    std::vector<VideoCommon::SamplerId> samplers;
    for (const auto& desc : info.texture_buffer_descriptors) {
        for (u32 element = 0; element < desc.count; ++element) {
            views.push_back({.index = read_handle(desc, element).first});
        }
    }
    for (const auto& desc : info.image_buffer_descriptors) {
        for (u32 element = 0; element < desc.count; ++element) {
            views.push_back({.index = read_handle(desc, element).first});
        }
    }
    for (const auto& desc : info.texture_descriptors) {
        for (u32 element = 0; element < desc.count; ++element) {
            const auto handle = read_handle(desc, element);
            views.push_back({.index = handle.first});
            samplers.push_back(texture_cache.GetSamplerId(handle.second, true));
        }
    }
    for (const auto& desc : info.image_descriptors) {
        for (u32 element = 0; element < desc.count; ++element) {
            views.push_back(
                {.index = read_handle(desc, element).first, .blacklist = desc.is_written});
        }
    }
    texture_cache.FillImageViews(views, true);
    // EDEN_NX2_OOT_BILINEAR_UPSCALE: Ocarina's upscale dispatch as one bilinear tap, as Vulkan's
    // TryOcarinaBilinear takes it (views: the sampled scene, then the written target).
    if (OcarinaUpscale::BilinearEnabled() &&
        Settings::GetCurrentProgramID() == VideoCommon::OotNetwork::ProgramId &&
        !kepler_compute->GetIndirectComputeAddress() &&
        OcarinaUpscale::IsBilinearPass(pipeline->Key().unique_hash, info)) {
        using VideoCore::Surface::PixelFormat;
        ImageView& color = texture_cache.GetImageView(views[0].id);
        ImageView& target = texture_cache.GetImageView(views[1].id);
        if (color.format == PixelFormat::E5B9G9R9_FLOAT &&
            target.format == PixelFormat::B10G11R11_FLOAT && color.Samples() <= 1 &&
            target.Samples() <= 1 && color.range == VideoCommon::SubresourceRange{} &&
            target.range == VideoCommon::SubresourceRange{} &&
            target.size.width > color.size.width && target.size.height > color.size.height &&
            color.ImageHandle() != target.ImageHandle() &&
            ocarina.Bilinear(color.Handle(Shader::TextureType::Color2D),
                             {color.size.width, color.size.height},
                             {color.size.width, color.size.height},
                             target.Handle(Shader::TextureType::Color2D),
                             {target.size.width, target.size.height})) {
            texture_cache.MarkModification(target.image_id);
            return;
        }
    }

    buffer_cache.UnbindComputeTextureBuffers();
    size_t view = 0;
    size_t tbo_index = 0;
    const auto add_buffer = [&](const auto& desc, bool is_image) {
        for (u32 element = 0; element < desc.count; ++element) {
            ImageView& image_view = texture_cache.GetImageView(views[view++].id);
            bool is_written = false;
            if constexpr (requires { desc.is_written; }) {
                is_written = desc.is_written;
            }
            buffer_cache.BindComputeTextureBuffer(tbo_index++, image_view.GpuAddr(),
                                                  image_view.BufferSize(), image_view.format,
                                                  is_written, is_image);
        }
    };
    for (const auto& desc : info.texture_buffer_descriptors) {
        add_buffer(desc, false);
    }
    for (const auto& desc : info.image_buffer_descriptors) {
        add_buffer(desc, true);
    }
    buffer_cache.UpdateComputeBuffers();
    buffer_cache_runtime.PendingBindings().clear();
    buffer_cache.BindHostComputeBuffers();
    const std::vector<Binding> bindings = buffer_cache_runtime.PendingBindings();
    const std::vector<TextureHandle> texel_views = PrepareTexelBuffers(bindings, info);

    scheduler.BeginCompute(state);
    BindStageResources(program, ShaderStage::Compute, bindings, texel_views, views, samplers, 0,
                       0);
    const std::array<u32, 3> group_size{qmd.block_dim_x, qmd.block_dim_y, qmd.block_dim_z};
    if (const auto indirect = kepler_compute->GetIndirectComputeAddress()) {
        const auto [buffer, offset] = buffer_cache.ObtainBuffer(
            *indirect, 12, VideoCommon::ObtainBufferSynchronize::FullSynchronize,
            VideoCommon::ObtainBufferOperation::DiscardWrite);
        scheduler.DispatchIndirect(buffer->Handle(), offset, group_size);
    } else {
        scheduler.Dispatch({qmd.grid_dim_x, qmd.grid_dim_y, qmd.grid_dim_z}, group_size);
    }
}

void RasterizerMetal::ResetCounter(VideoCommon::QueryType type) {
    if (type == VideoCommon::QueryType::ZPassPixelCount64 ||
        type == VideoCommon::QueryType::ZPassPixelCount) {
        occlusion_slots.clear();
        occlusion_pass = ~u64{0};
    }
}

u32 RasterizerMetal::AllocateVisibilitySlot() {
    const u32 slot = next_visibility_slot;
    next_visibility_slot = (next_visibility_slot + 1) % Scheduler::VisibilitySlots;
    // A lap later: the GPU wrote the slot long ago, but make sure.
    if (!scheduler.IsFree(visibility_ticks[slot])) {
        scheduler.Wait(visibility_ticks[slot]);
    }
    visibility_ticks[slot] = scheduler.CurrentTick();
    scheduler.VisibilityResults()[slot] = 0;
    return slot;
}

void RasterizerMetal::CountOcclusion() {
    if (maxwell3d->regs.zpass_pixel_count_enable == 0) {
        scheduler.SetVisibility(std::nullopt);
        return;
    }
    // A slot per render pass of the counting segment: results are summed at the query.
    if (occlusion_pass != scheduler.RenderPassSerial()) {
        occlusion_slots.push_back(AllocateVisibilitySlot());
        occlusion_pass = scheduler.RenderPassSerial();
    }
    scheduler.SetVisibility(occlusion_slots.back());
}

void RasterizerMetal::Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
                            VideoCommon::QueryPropertiesFlags flags, u32 payload, u32) {
    if (type != VideoCommon::QueryType::ZPassPixelCount64 &&
        type != VideoCommon::QueryType::ZPassPixelCount) {
        QueryFallback(gpu_addr, type, flags, payload);
        return;
    }
    if (!gpu_memory) {
        return;
    }
    ++occlusion_queries;
    // Written once the GPU has finished the draws counted so far (the fence's command buffer
    // ends their render pass, so later draws count into new slots).
    SignalFence([this, gpu_addr, flags, memory_manager = gpu_memory,
                 slots = occlusion_slots] {
        const u64* const results = scheduler.VisibilityResults();
        u64 samples = 0;
        for (const u32 slot : slots) {
            samples += results[slot];
        }
        if (True(flags & VideoCommon::QueryPropertiesFlags::HasTimeout)) {
            memory_manager->Write<u64>(gpu_addr + 8, gpu.GetTicks());
            memory_manager->Write<u64>(gpu_addr, samples);
        } else {
            memory_manager->Write<u32>(gpu_addr, static_cast<u32>(samples));
        }
    });
}

void RasterizerMetal::QueryFallback(GPUVAddr gpu_addr, VideoCommon::QueryType type,
                                    VideoCommon::QueryPropertiesFlags flags, u32 payload) {
    if (!gpu_memory) {
        return;
    }
    // Host queries come with milestone 4: counters read as 1 (something passed), as OpenGL's
    // fallback does.
    if (type != VideoCommon::QueryType::Payload) {
        payload = 1U;
    }
    std::function<void()> func([this, gpu_addr, flags, memory_manager = gpu_memory, payload] {
        if (True(flags & VideoCommon::QueryPropertiesFlags::HasTimeout)) {
            memory_manager->Write<u64>(gpu_addr + 8, gpu.GetTicks());
            memory_manager->Write<u64>(gpu_addr, static_cast<u64>(payload));
        } else {
            memory_manager->Write<u32>(gpu_addr, payload);
        }
    });
    if (True(flags & VideoCommon::QueryPropertiesFlags::IsAFence)) {
        SignalFence(std::move(func));
        return;
    }
    func();
}

void RasterizerMetal::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                                u32 size) {
    std::scoped_lock lock{buffer_cache.mutex};
    buffer_cache.BindGraphicsUniformBuffer(stage, index, gpu_addr, size);
}

void RasterizerMetal::DisableGraphicsUniformBuffer(size_t stage, u32 index) {
    buffer_cache.DisableGraphicsUniformBuffer(stage, index);
}

void RasterizerMetal::FlushAll() {}

void RasterizerMetal::FlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0) {
        return;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.DownloadMemory(addr, size);
    }
    if (True(which & VideoCommon::CacheType::BufferCache)) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.DownloadMemory(addr, size);
    }
}

bool RasterizerMetal::MustFlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (True(which & VideoCommon::CacheType::BufferCache)) {
        std::scoped_lock lock{buffer_cache.mutex};
        if (buffer_cache.IsRegionGpuModified(addr, size)) {
            return true;
        }
    }
    if (!Settings::IsGPULevelHigh()) {
        return false;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        return texture_cache.IsRegionGpuModified(addr, size);
    }
    return false;
}

VideoCore::RasterizerDownloadArea RasterizerMetal::GetFlushArea(DAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        if (const auto area = texture_cache.GetFlushArea(addr, size)) {
            return *area;
        }
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        if (const auto area = buffer_cache.GetFlushArea(addr, size)) {
            return *area;
        }
    }
    return VideoCore::RasterizerDownloadArea{
        .start_address = Common::AlignDown(addr, Core::DEVICE_PAGESIZE),
        .end_address = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE),
        .preemtive = true,
    };
}

void RasterizerMetal::InvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0) {
        return;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    if (True(which & VideoCommon::CacheType::BufferCache)) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    if (True(which & VideoCommon::CacheType::ShaderCache)) {
        pipeline_cache.InvalidateRegion(addr, size);
    }
}

bool RasterizerMetal::OnCPUWrite(DAddr addr, u64 size) {
    {
        std::scoped_lock lock{buffer_cache.mutex};
        if (buffer_cache.OnCPUWrite(addr, size)) {
            return true;
        }
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
    return false;
}

void RasterizerMetal::OnCacheInvalidation(DAddr addr, u64 size) {
    if (addr == 0 || size == 0) {
        return;
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
}

void RasterizerMetal::InvalidateGPUCache() {
    gpu.InvalidateGPUCache();
}

void RasterizerMetal::UnmapMemory(DAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size, false);
    }
    pipeline_cache.OnCacheInvalidation(addr, size);
}

void RasterizerMetal::ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) {
    std::scoped_lock lock{texture_cache.mutex};
    texture_cache.UnmapGPUMemory(as_id, addr, size);
}

void RasterizerMetal::SignalFence(std::function<void()>&& func) {
    fence_manager.SignalFence(std::move(func));
}

void RasterizerMetal::SyncOperation(std::function<void()>&& func) {
    fence_manager.SyncOperation(std::move(func));
}

void RasterizerMetal::SignalSyncPoint(u32 value) {
    fence_manager.SignalSyncPoint(value);
}

void RasterizerMetal::SignalReference() {
    fence_manager.SignalOrdering();
}

void RasterizerMetal::ReleaseFences(bool force) {
    fence_manager.WaitPendingFences(force);
}

void RasterizerMetal::FlushAndInvalidateRegion(DAddr addr, u64 size,
                                               VideoCommon::CacheType which) {
    if (Settings::IsGPULevelHigh()) {
        FlushRegion(addr, size, which);
    }
    InvalidateRegion(addr, size, which);
}

void RasterizerMetal::WaitForIdle() {
    SignalReference();
}

void RasterizerMetal::FragmentBarrier() {}

void RasterizerMetal::TiledCacheBarrier() {}

void RasterizerMetal::FlushCommands() {
    if (scheduler.HasPendingWork()) {
        scheduler.Flush();
    }
}

void RasterizerMetal::TickFrame() {
    texel_view_cache.clear();
    fence_manager.TickFrame();
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.TickFrame();
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.TickFrame();
    }
    staging_pool.TickFrame();
}

bool RasterizerMetal::AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                                            const Tegra::Engines::Fermi2D::Surface& dst,
                                            const Tegra::Engines::Fermi2D::Config& copy_config) {
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.BlitImage(dst, src, copy_config);
}

bool RasterizerMetal::AccelerateTemporalUpscale(
    const Tegra::Engines::Fermi2D::Surface& src, const Tegra::Engines::Fermi2D::Surface& dst,
    const Tegra::Engines::Fermi2D::Surface& motion_vectors, const Tegra::Texture::TICEntry* depth,
    std::array<f32, 2> jitter, std::array<f32, 2> motion_scale, bool use_history) {
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.TemporalUpscale(dst, src, motion_vectors, depth, jitter, motion_scale,
                                          use_history);
}

Tegra::Engines::AccelerateDMAInterface& RasterizerMetal::AccessAccelerateDMA() {
    return accelerate_dma;
}

void RasterizerMetal::AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                               std::span<const u8> memory) {
    const auto cpu_addr = gpu_memory->GpuToCpuAddress(address);
    if (!cpu_addr) {
        gpu_memory->WriteBlock(address, memory.data(), copy_size);
        return;
    }
    gpu_memory->WriteBlockUnsafe(address, memory.data(), copy_size);
    {
        std::unique_lock lock{buffer_cache.mutex};
        if (!buffer_cache.InlineMemory(*cpu_addr, copy_size, memory)) {
            buffer_cache.WriteMemory(*cpu_addr, copy_size);
        }
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(*cpu_addr, copy_size);
    }
    pipeline_cache.InvalidateRegion(*cpu_addr, copy_size);
}

void RasterizerMetal::LoadDiskResources(u64, std::stop_token,
                                        const VideoCore::DiskResourceLoadCallback&) {}

void RasterizerMetal::InitializeChannel(Tegra::Control::ChannelState& channel) {
    CreateChannel(channel);
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    texture_cache.CreateChannel(channel);
    buffer_cache.CreateChannel(channel);
    pipeline_cache.CreateChannel(channel);
}

void RasterizerMetal::BindChannel(Tegra::Control::ChannelState& channel) {
    const s32 channel_id = channel.bind_id;
    BindToChannel(channel_id);
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    texture_cache.BindToChannel(channel_id);
    buffer_cache.BindToChannel(channel_id);
    pipeline_cache.BindToChannel(channel_id);
}

void RasterizerMetal::ReleaseChannel(s32 channel_id) {
    EraseChannel(channel_id);
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    texture_cache.EraseChannel(channel_id);
    buffer_cache.EraseChannel(channel_id);
    pipeline_cache.EraseChannel(channel_id);
}

std::optional<DisplayTexture> RasterizerMetal::AccelerateDisplay(
    const Tegra::FramebufferConfig& config, DAddr framebuffer_addr) {
    if (framebuffer_addr == 0) {
        return std::nullopt;
    }
    std::scoped_lock lock{texture_cache.mutex};
    const auto [image_view, scaled] =
        texture_cache.TryFindFramebufferImageView(config, framebuffer_addr);
    if (image_view == nullptr) {
        return std::nullopt;
    }
    const auto& resolution = Settings::values.resolution_info;
    DisplayTexture texture{
        .texture = image_view->Handle(Shader::TextureType::Color2D),
        .width = image_view->size.width,
        .height = image_view->size.height,
    };
    texture.scaled_width = scaled ? resolution.ScaleUp(texture.width) : texture.width;
    texture.scaled_height = scaled ? resolution.ScaleUp(texture.height) : texture.height;
    if (texture.texture == nullptr) {
        return std::nullopt;
    }
    return texture;
}

} // namespace Metal
