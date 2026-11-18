// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Metal resources and formats for plain C++ (metal_resources.mm). Formats are MTLPixelFormat
// values, swizzles MTLTextureSwizzle values.

#pragma once

#include <array>
#include <functional>
#include <string>
#include <string_view>

#include "common/common_types.h"
#include "video_core/renderer_metal/metal_object.h"
#include "video_core/surface.h"
#include "video_core/textures/texture.h"

namespace Metal {

class Device;
class PipelineArchive;

enum class TextureKind : u8 {
    e1D,
    e1DArray,
    e2D,
    e2DArray,
    e2DMultisample,
    e2DMultisampleArray,
    e3D,
    Cube,
    CubeArray,
};

struct TextureDesc {
    TextureKind kind{TextureKind::e2D};
    u32 format{};
    u32 width{1};
    u32 height{1};
    u32 depth{1};
    u32 levels{1};
    u32 layers{1};
    u32 samples{1};
    bool render_target{};
    bool shader_write{};
};

/// MTLTextureSwizzleRed..Alpha: the identity.
inline constexpr std::array<u8, 4> IdentitySwizzle{2, 3, 4, 5};

/// How a texture's texels are read and written by shaders.
enum class FormatClass : u8 {
    Float,
    Uint,
    Sint,
    Depth,
    Stencil,
    DepthStencil,
};

struct FormatInfo {
    u32 format{};      ///< MTLPixelFormatInvalid (0) when Metal has no counterpart
    bool attachable{}; ///< Can be a render target
    bool storage{};    ///< Can be written by shaders
    FormatClass format_class{FormatClass::Float};
    std::array<u8, 4> swizzle{IdentitySwizzle}; ///< Applied before the guest's own swizzle
};

/// Device::MemoryBudget for plain C++.
[[nodiscard]] u64 DeviceMemoryBudget(const Device& device);

/// Device::AppleFamily for plain C++.
[[nodiscard]] u32 DeviceAppleFamily(const Device& device);

/// Device::SupportsMeshShaders for plain C++.
[[nodiscard]] bool DeviceSupportsMeshShaders(const Device& device);

/// Runs work inside an autorelease pool: threads of plain C++ that make Metal objects have none.
void InAutoreleasePool(const std::function<void()>& work);

/// The host format of a guest format. `converted` images hold the CPU-decoded data instead
/// (RGBA8 / R8 / RG8 / RGBA16F, VideoCommon::ConvertImage).
[[nodiscard]] FormatInfo SurfaceFormat(const Device& device, VideoCore::Surface::PixelFormat format,
                                       bool converted);

/// Whether the guest format is decoded on the CPU before upload (no native ASTC / BC).
[[nodiscard]] bool IsConverted(const Device& device, VideoCore::Surface::PixelFormat format);

[[nodiscard]] FormatClass ClassOf(u32 mtl_format);

/// The format CAMetalLayer presents (BGRA8Unorm).
[[nodiscard]] u32 DrawableFormat();

[[nodiscard]] Object CreateTexture(const Device& device, const TextureDesc& desc);

/// A view of `texture` in another format, type, range or swizzle; nil on failure.
[[nodiscard]] Object CreateTextureView(TextureHandle texture, u32 format, TextureKind kind,
                                       u32 base_level, u32 levels, u32 base_layer, u32 layers,
                                       std::array<u8, 4> swizzle);

/// Bytes a texture buffer view's offset must be a multiple of, for the format.
[[nodiscard]] u32 TextureBufferAlignment(const Device& device, u32 format);

/// A texture_buffer view of `elements` texels of `buffer` from `offset` (TextureBufferAlignment
/// aligned); empty on failure.
[[nodiscard]] Object CreateTextureBufferView(BufferHandle buffer, u64 offset, u32 elements,
                                             u32 bytes_per_texel, u32 format, bool writable);

/// A buffer in shared (CPU-visible, `contents` set) or private storage.
[[nodiscard]] Object CreateBuffer(const Device& device, u64 size, bool shared, u8** contents);

/// What a sampler state takes besides the guest's descriptor: Vulkan's sampler variants
/// (vk_texture_cache.cpp, Sampler::MakeKey) for the view it samples.
struct SamplerOptions {
    /// EDEN_NX2_METAL_SAMPLERS=0: the guest's descriptor as this backend first read it (no LOD
    /// bias, reduction or variants; Clamp and the mirror-once modes clamp without mirroring).
    bool plain{};
    f32 added_lod_bias{};      ///< Added to the guest's bias (the Ocarina upscalers')
    bool reduce_anisotropy{};  ///< Only the guest's own anisotropy (the view takes no more)
    bool force_nearest{};      ///< No filtering (integer views)
    bool drop_compare{};       ///< No comparison (a plain sample of a compare sampler)
    bool srgb_border{};        ///< The sRGB border colour (sRGB views)
};

/// A sampler for a guest sampler descriptor (TSC entry).
[[nodiscard]] Object CreateSampler(const Device& device, const Tegra::Texture::TSCEntry& tsc,
                                   const SamplerOptions& options = {});

/// Whether sampler states take a LOD bias and a min / max reduction: Metal 4's
/// MTLSamplerDescriptor lodBias and reductionMode (macOS / iOS 26 SDK and OS) on a Metal 4 GPU
/// (Apple7 and later: A14, M1). Metal applies the reduction only with linear min, mag and mip
/// filters.
[[nodiscard]] bool SamplerBiasSupported(const Device& device);

/// What a guest sampler adds to the shader's mask (ShaderTranslate::Options::sampler_reduction),
/// before it is shifted to its slot: bit 0 min, bit 1 max (only with linear min and mag filters),
/// bit 2 a linear mip filter, bit 3 a nearest one. Zero for the weighted average.
[[nodiscard]] u64 SamplerReductionBits(const Tegra::Texture::TSCEntry& tsc);

/// A bilinear sampler clamping to the edge, for the renderer's own passes.
[[nodiscard]] Object CreateLinearClampSampler(const Device& device);

/// An MTLLibrary built from MSL source (Metal's compiler, at run time); empty with `error` set
/// on failure.
[[nodiscard]] Object CreateLibrary(const Device& device, std::string_view source,
                                   std::string& error);

/// An MTLFunction of a library; empty when it has none of that name.
[[nodiscard]] Object CreateFunction(const void* library, std::string_view name);

/// An MTLComputePipelineState; empty with `error` set on failure. With `archive`, its GPU code
/// is looked up there first and added on a miss.
[[nodiscard]] Object CreateComputePipelineState(const Device& device, const void* function,
                                                std::string& error,
                                                PipelineArchive* archive = nullptr);

/// What a texture is, for checks before Metal is given it.
struct TextureFacts {
    u32 format{};  ///< MTLPixelFormat, 0 for a null texture
    u32 samples{1};
    u32 levels{1};
    u32 slices{1}; ///< Array slices (cube faces count), 1 for 3D
    u32 depth{1};  ///< Of level 0 (3D)
};

[[nodiscard]] TextureFacts DescribeTexture(TextureHandle texture);

/// Bytes the device has allocated for this process (Device::AllocatedSize), for plain C++.
[[nodiscard]] u64 DeviceAllocatedSize(const Device& device);

/// Bytes the resource (texture or buffer) holds on the GPU.
[[nodiscard]] u64 AllocatedSize(const void* resource);

void SetLabel(const void* resource, std::string_view label);

} // namespace Metal
