// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Metal resources and formats for plain C++ (metal_resources.mm). Formats are MTLPixelFormat
// values, swizzles MTLTextureSwizzle values.

#pragma once

#include <array>
#include <string_view>

#include "common/common_types.h"
#include "video_core/renderer_metal/metal_object.h"
#include "video_core/surface.h"
#include "video_core/textures/texture.h"

namespace Metal {

class Device;

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

/// A buffer in shared (CPU-visible, `contents` set) or private storage.
[[nodiscard]] Object CreateBuffer(const Device& device, u64 size, bool shared, u8** contents);

/// A sampler for a guest sampler descriptor (TSC entry).
[[nodiscard]] Object CreateSampler(const Device& device, const Tegra::Texture::TSCEntry& tsc);

/// Bytes the resource (texture or buffer) holds on the GPU.
[[nodiscard]] u64 AllocatedSize(const void* resource);

void SetLabel(const void* resource, std::string_view label);

} // namespace Metal
