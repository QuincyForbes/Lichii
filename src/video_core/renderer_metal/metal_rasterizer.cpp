// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <mutex>

#include "common/alignment.h"
#include "common/cityhash.h"
#include "common/logging.h"
#include "common/settings.h"
#include "shader_recompiler/exception.h"
#include "video_core/control/channel_state.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/framebuffer_config.h"
#include "video_core/gpu.h"
#include "video_core/host1x/host1x.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_metal/metal_graphics.h"
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
    const std::optional<u32> primitive = PrimitiveType(draw_state.topology);
    if (!primitive) {
        LOG_ONCE(warned_topology, "Metal: topology {} is not drawn yet",
                 static_cast<u32>(draw_state.topology));
        return;
    }
    gpu_memory->FlushCaching();
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    if (!PrepareDraw(*pipeline, is_indexed)) {
        return;
    }
    if (is_indexed) {
        const BufferCacheRuntime::IndexState& index = buffer_cache_runtime.Index();
        if (index.format == Maxwell::IndexFormat::UnsignedByte) {
            // Metal has no 8-bit indices: they need widening on the GPU first.
            LOG_ONCE(warned_u8, "Metal: 8-bit index buffers are not drawn yet");
            return;
        }
        const u32 index_size = index.format == Maxwell::IndexFormat::UnsignedShort ? 2U : 4U;
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

void RasterizerMetal::BindStageResources(StageProgram& program, ShaderStage metal_stage,
                                         std::span<const VideoCommon::ImageViewInOut> views,
                                         std::span<const VideoCommon::SamplerId> samplers,
                                         u32 render_width, u32 render_height) {
    const Shader::Info& info = program.info;
    std::vector<Binding>& pending = buffer_cache_runtime.PendingBindings();
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
    if (texel_buffers != 0) {
        LOG_ONCE(warned_texel, "Metal: texel buffers are not bound yet");
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

bool RasterizerMetal::PrepareDraw(GraphicsPipeline& pipeline, bool is_indexed) {
    // Buffer cache stage indices (vertex B, pixel) and the translated programs.
    constexpr std::array<size_t, 2> stages{0, 4};
    const std::array<StageProgram*, 2> programs{&pipeline.Vertex(), &pipeline.Fragment()};
    const auto& regs = maxwell3d->regs;

    std::array<u32, VideoCommon::NUM_STAGES> uniform_masks{};
    for (size_t index = 0; index < stages.size(); ++index) {
        if (programs[index]->msl.empty()) {
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
    std::array<size_t, 2> first_view{};
    std::array<size_t, 2> first_sampler{};
    const bool via_header_index = regs.sampler_binding == Maxwell::SamplerBinding::ViaHeaderBinding;
    for (size_t index = 0; index < stages.size(); ++index) {
        first_view[index] = views.size();
        first_sampler[index] = samplers.size();
        if (programs[index]->msl.empty()) {
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
        if (programs[index]->msl.empty()) {
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
    state_desc.topology_class = TopologyClass(maxwell3d->draw_manager.draw_state.topology);
    state_desc.rasterize = regs.rasterize_enable != 0 ? 1U : 0U;
    state_desc.alpha_to_coverage = regs.anti_alias_alpha_control.alpha_to_coverage != 0 ? 1U : 0U;
    for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
        const auto& input = regs.vertex_attrib_format[index];
        VertexAttributeState& attribute = state_desc.attributes[index];
        attribute.enabled = input.constant != 0 ? 0U : 1U;
        attribute.stream = input.buffer;
        attribute.offset = input.offset;
        attribute.type = input.type;
        attribute.size = input.size;
    }
    const auto& vertex_states = buffer_cache_runtime.Vertices();
    for (size_t index = 0; index < Maxwell::NumVertexArrays; ++index) {
        VertexStreamState& stream = state_desc.streams[index];
        stream.enabled = vertex_states[index].buffer != nullptr ? 1U : 0U;
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
            if (TopologyClass(maxwell3d->draw_manager.draw_state.topology) == 2) {
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

    for (size_t index = 0; index < vertex_states.size(); ++index) {
        if (vertex_states[index].buffer != nullptr) {
            scheduler.BindVertexBuffer(VertexBufferSlot(static_cast<u32>(index)),
                                       vertex_states[index].buffer, vertex_states[index].offset);
        }
    }

    // Each stage's buffers (appended by the buffer cache in binding order), textures, samplers
    // and images, into the slots the translator gave them.
    for (size_t index = 0; index < stages.size(); ++index) {
        StageProgram& program = *programs[index];
        if (program.msl.empty()) {
            continue;
        }
        const size_t stage = stages[index];
        const ShaderStage metal_stage = index == 0 ? ShaderStage::Vertex : ShaderStage::Fragment;
        buffer_cache_runtime.PendingBindings().clear();
        buffer_cache.BindHostStageBuffers(stage);

        BindStageResources(program, metal_stage, std::span(views).subspan(first_view[index]),
                           std::span(samplers).subspan(first_sampler[index]), targets.width,
                           targets.height);
    }
    return true;
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

    scheduler.BeginCompute(state);
    BindStageResources(program, ShaderStage::Compute, views, samplers, 0, 0);
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

void RasterizerMetal::ResetCounter(VideoCommon::QueryType) {}

void RasterizerMetal::Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
                            VideoCommon::QueryPropertiesFlags flags, u32 payload, u32) {
    QueryFallback(gpu_addr, type, flags, payload);
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
