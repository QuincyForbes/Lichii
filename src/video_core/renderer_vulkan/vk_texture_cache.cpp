// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <limits>
#include <array>
#include <atomic>
#include <cstdlib>
#include <optional>
#include <set>
#include <span>
#include <tuple>
#include <memory>
#include <utility>
#include <vector>
#include <boost/container/small_vector.hpp>
#include <bit>
#include <numeric>
#include "common/alignment.h"
#include "common/bit_util.h"
#include "common/div_ceil.h"
#include "common/literals.h"
#include "common/platform_events.h"
#include "common/settings.h"
#include "shader_recompiler/byte_addressed_image.h"
#include "shader_recompiler/exception.h"

#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/nx2.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/render_trace.h"
#include "video_core/gpu_logging/gpu_logging.h"

#include "video_core/engines/fermi_2d.h"
#include "video_core/renderer_vulkan/blit_image.h"
#include "video_core/renderer_vulkan/framebuffer_limits.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/vk_compute_pass.h"
#include "video_core/renderer_vulkan/vk_ocarina_camera_motion.h"
#include "video_core/renderer_vulkan/vk_render_pass_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/formatter.h"
#include "video_core/texture_cache/samples_helper.h"
#include "video_core/texture_cache/util.h"
#include "video_core/dlss_contract.h"
#include "video_core/vulkan_common/vulkan_device.h"
#ifdef EGLANTINE_STREAMLINE
#include "video_core/vulkan_common/streamline.h"
#endif
#include "video_core/vulkan_common/unbacked_fallback.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/textures/decoders.h"

namespace Vulkan {

using Tegra::Engines::Fermi2D;
using Tegra::Texture::SwizzleSource;
using Tegra::Texture::TextureMipmapFilter;
using VideoCommon::BufferImageCopy;
using VideoCommon::ImageFlagBits;
using VideoCommon::ImageInfo;
using VideoCommon::ImageType;
using VideoCommon::SubresourceRange;
using VideoCore::Surface::BytesPerBlock;
using VideoCore::Surface::HasAlpha;
using VideoCore::Surface::IsPixelFormatASTC;
using VideoCore::Surface::IsPixelFormatInteger;
using VideoCore::Surface::IsPixelFormatSRGB;
using VideoCore::Surface::SurfaceType;

namespace {
constexpr bool ENABLE_MSAA_TILER_RESOLVE = true;
constexpr bool ENABLE_MSAA_RESOLVE_CONSUME = true;
// EDEN_NX2_MSAA_DISCARD=0: MSAA render passes keep their samples (STORE_OP_STORE) even when they
// begin with a clear. A discarding pass that something ends early (a copy needs the render pass
// closed) loses the samples drawn so far, and its resumed half loads garbage.
bool MsaaDiscardEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_MSAA_DISCARD");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

[[nodiscard]] constexpr bool NeedsExplicitBorderColorFormat(VkFormat format) {
    switch (format) {
    case VK_FORMAT_B4G4R4A4_UNORM_PACK16:
    case VK_FORMAT_B5G6R5_UNORM_PACK16:
    case VK_FORMAT_B5G5R5A1_UNORM_PACK16:
        return true;
    default:
        return false;
    }
}

constexpr VkBorderColor ConvertBorderColor(const std::array<float, 4>& color) {
    if (color == std::array<float, 4>{0, 0, 0, 0}) {
        return VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    } else if (color == std::array<float, 4>{0, 0, 0, 1}) {
        return VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    } else if (color == std::array<float, 4>{1, 1, 1, 1}) {
        return VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    }
    if (color[0] + color[1] + color[2] > 1.35f) {
        // If color elements are brighter than roughly 0.5 average, use white border
        return VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    } else if (color[3] > 0.5f) {
        return VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    } else {
        return VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    }
}

[[nodiscard]] VkImageType ConvertImageType(const ImageType type) {
    switch (type) {
    case ImageType::e1D:
        return VK_IMAGE_TYPE_1D;
    case ImageType::e2D:
    case ImageType::Linear:
        return VK_IMAGE_TYPE_2D;
    case ImageType::e3D:
        return VK_IMAGE_TYPE_3D;
    case ImageType::Buffer:
        break;
    }
    ASSERT_MSG(false, "Invalid image type={}", type);
    return {};
}

[[nodiscard]] VkSampleCountFlagBits ConvertSampleCount(u32 num_samples) {
    switch (num_samples) {
    case 1:
        return VK_SAMPLE_COUNT_1_BIT;
    case 2:
        return VK_SAMPLE_COUNT_2_BIT;
    case 4:
        return VK_SAMPLE_COUNT_4_BIT;
    case 8:
        return VK_SAMPLE_COUNT_8_BIT;
    case 16:
        return VK_SAMPLE_COUNT_16_BIT;
    default:
        ASSERT_MSG(false, "Invalid number of samples={}", num_samples);
        return VK_SAMPLE_COUNT_1_BIT;
    }
}

[[nodiscard]] VkImageUsageFlags ImageUsageFlags(const MaxwellToVK::FormatInfo& info,
                                                PixelFormat format, bool allow_storage = true) {
    VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                              VK_IMAGE_USAGE_SAMPLED_BIT;
    if (info.attachable) {
        switch (VideoCore::Surface::GetFormatType(format)) {
        case VideoCore::Surface::SurfaceType::ColorTexture:
            usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            break;
        case VideoCore::Surface::SurfaceType::Depth:
        case VideoCore::Surface::SurfaceType::Stencil:
        case VideoCore::Surface::SurfaceType::DepthStencil:
            usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            break;
        default:
            ASSERT_MSG(false, "Invalid surface type");
            break;
        }
    }
    if (info.storage && allow_storage) {
        usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
    return usage;
}

// False when the driver could not build the GPU ASTC decoder (Apple's shader compiler crashes on
// it for AMD GPUs under MoltenVK). Every ASTC image then decodes on the CPU.
std::atomic_bool gpu_astc_decoder_available{true};

// An ASTC 2D array whose RGBA8 decode exceeds 1 GiB (title 1's 8192x8192x68 terrain textures).
// These stay on the GPU decoder as RGBA8 with trimmed levels (TrimmedHostInfo) even when other
// ASTC images are re-compressed to BC1/BC3 on the CPU: re-compressing them would not fit either.
[[nodiscard]] bool IsHugeAstcArray(const Device& device, const ImageInfo& info) {
    if (!IsPixelFormatASTC(info.format) || device.IsOptimalAstcSupported() ||
        !gpu_astc_decoder_available || info.type != ImageType::e2D || info.size.depth != 1) {
        return false;
    }
    u64 bytes = 0;
    for (s32 level = 0; level < info.resources.levels; ++level) {
        bytes += u64{(std::max)(info.size.width >> level, 1U)} *
                 (std::max)(info.size.height >> level, 1U) * 4;
    }
    return bytes * info.resources.layers > (u64{1} << 30);
}

// Whether a block-compressed copy of `info` has a mip level whose size in 4x4 blocks is not its
// level 0 block count shifted down (341 texels: 86 blocks at level 0, 85 texels = 22 blocks at
// level 2, not 86 >> 2 = 21). On AMD (RX 6600 XT, Windows driver) the BC1/BC3 copies of such
// images show dark lines along the seams of distant terrain in title 1; RGBA8 copies don't.
[[nodiscard]] bool HasUnevenBlockMips(const ImageInfo& info) {
    const u32 blocks_x = Common::DivCeil(info.size.width, 4U);
    const u32 blocks_y = Common::DivCeil(info.size.height, 4U);
    for (s32 level = 1; level < info.resources.levels; ++level) {
        const u32 level_x = Common::DivCeil((std::max)(info.size.width >> level, 1U), 4U);
        const u32 level_y = Common::DivCeil((std::max)(info.size.height >> level, 1U), 4U);
        if (level_x != (std::max)(blocks_x >> level, 1U) ||
            level_y != (std::max)(blocks_y >> level, 1U)) {
            return true;
        }
    }
    return false;
}

// ASTC that would be re-compressed to BC1/BC3 but has uneven block mips (HasUnevenBlockMips) on
// an AMD driver, which samples those wrong: kept as the GPU decoder's RGBA8 instead, like huge
// arrays.
[[nodiscard]] bool IsUnevenMipRgba8Astc(const Device& device, const ImageInfo& info) {
    if (!IsPixelFormatASTC(info.format) || device.IsOptimalAstcSupported() ||
        info.type != ImageType::e2D || info.size.depth != 1 || info.resources.levels <= 1 ||
        Settings::values.accelerate_astc.GetValue() != Settings::AstcDecodeMode::Gpu ||
        Settings::GetAstcRecompression() == Settings::AstcRecompression::Uncompressed) {
        return false;
    }
    const VkDriverIdKHR id = device.GetDriverID();
    if (id != VK_DRIVER_ID_AMD_PROPRIETARY && id != VK_DRIVER_ID_AMD_OPEN_SOURCE &&
        id != VK_DRIVER_ID_MESA_RADV) {
        return false;
    }
    return HasUnevenBlockMips(info) && !IsHugeAstcArray(device, info);
}

// A large image over Switch 2 sparse blocks nobody has written (title 1's terrain layer pools, 2.9
// GB of R8/RG16/R16 that stay empty in the attract demo) is all zeros: it gets no device memory
// until a write makes the texture cache replace it.
[[nodiscard]] bool IsUnwrittenSparse(const Device& device, const ImageInfo& info, GPUVAddr gpu_addr) {
    if (!device.CanUseUnbackedSparseImages()) {
        return false;
    }
    constexpr u32 MinSize = 32U << 20;
    const u64 size = VideoCommon::CalculateGuestSizeInBytes(info);
    return size >= MinSize && info.type != ImageType::e1D && info.type != ImageType::Buffer &&
           info.type != ImageType::Linear && Tegra::Nx2SparseUnwritten(gpu_addr, size);
}

// A large color array in a Switch 2 sparse region (title 1's terrain arrays: 13-26% of their
// blocks written) gets memory by layer, as its layers are written: unbacked at first, then
// Image::BindLayers. Only formats uploaded layer by layer: huge ASTC arrays (GPU decode) and
// formats the host samples as they are.
[[nodiscard]] bool UsesSparseLayers(const Device& device, const ImageInfo& info, GPUVAddr gpu_addr) {
    if (!device.CanUseUnbackedSparseImages() || !device.IsSparseResidencyImage2DSupported() ||
        info.type != ImageType::e2D ||
        info.resources.layers <= 1 || info.num_samples != 1 ||
        VideoCore::Surface::GetFormatType(info.format) != VideoCore::Surface::SurfaceType::ColorTexture) {
        return false;
    }
    const bool layer_uploads =
        IsHugeAstcArray(device, info) ||
        (!IsPixelFormatASTC(info.format) &&
         (!IsPixelFormatBCn(info.format) || device.IsOptimalBcnSupported()));
    const u64 size = VideoCommon::CalculateGuestSizeInBytes(info);
    return layer_uploads && size >= (32U << 20) && Tegra::Nx2InSparseRegion(gpu_addr, size);
}

// How SparseTiles images leave the blocks without data: unbound (reads give zero), or all bound to
// one zeroed block where the driver can't sample unbound memory (AMD) but lets blocks share
// memory. EDEN_NX2_SPARSE_ALIASED=1 takes the shared block wherever it is supported, =0 never.
enum class SparseTilesMode {
    None,
    Unbound,
    Aliased,
};

[[nodiscard]] SparseTilesMode SparseTilesModeOf(const Device& device) {
    static const char force_aliased = [] {
        const char* value = std::getenv("EDEN_NX2_SPARSE_ALIASED");
        return value == nullptr ? '\0' : value[0];
    }();
    if (!device.IsSparseResidencyImage2DSupported()) {
        return SparseTilesMode::None;
    }
    const bool aliased = device.IsSparseResidencyAliasedSupported() && force_aliased != '0';
    if (aliased && force_aliased == '1') {
        return SparseTilesMode::Aliased;
    }
    if (device.CanUseUnbackedSparseImages()) {
        return SparseTilesMode::Unbound;
    }
    return aliased ? SparseTilesMode::Aliased : SparseTilesMode::None;
}

// Large NX2 masks and BC6H lightmap arrays get memory by sparse block where guest memory
// is mapped. A whole BC6H layer can cost 85 MiB even if only a few guest tiles are present.
[[nodiscard]] bool UsesSparseTiles(const Device& device, const ImageInfo& info, GPUVAddr gpu_addr) {
    const auto mode = SparseTilesModeOf(device);
    if (mode == SparseTilesMode::None || info.type != ImageType::e2D || info.num_samples != 1 ||
        info.size.depth != 1 || info.block.width != 0 ||
        VideoCore::Surface::GetFormatType(info.format) !=
            VideoCore::Surface::SurfaceType::ColorTexture) {
        return false;
    }
    const bool plain_image = info.resources.layers == 1 &&
                             VideoCore::Surface::DefaultBlockWidth(info.format) == 1 &&
                             VideoCore::Surface::DefaultBlockHeight(info.format) == 1;
    const bool hdr_array = info.native_tiling && info.resources.layers > 1 &&
                           (info.format == PixelFormat::BC6H_UFLOAT ||
                            info.format == PixelFormat::BC6H_SFLOAT) &&
                           mode == SparseTilesMode::Unbound && device.IsOptimalBcnSupported();
    const u64 size = VideoCommon::CalculateGuestSizeInBytes(info);
    return (plain_image || hdr_array) && size >= (32U << 20) &&
           Tegra::Nx2InSparseRegion(gpu_addr, size);
}

// Vulkan's standard sparse block shape for single-sampled 2D images (64 KiB blocks), in texel
// blocks, for a texel block of `block_bytes`. Only valid when the device reports
// residencyStandard2DBlockShape.
[[nodiscard]] bool StandardSparseBlockShape(const Device& device, u64 block_bytes,
                                            VkExtent3D& shape) {
    if (!device.GetPhysical().GetProperties().sparseProperties.residencyStandard2DBlockShape) {
        return false;
    }
    switch (block_bytes) {
    case 1:
        shape = {256, 256, 1};
        return true;
    case 2:
        shape = {256, 128, 1};
        return true;
    case 4:
        shape = {128, 128, 1};
        return true;
    case 8:
        shape = {128, 64, 1};
        return true;
    case 16:
        shape = {64, 64, 1};
        return true;
    default:
        return false;
    }
}

// The most device memory one SparseLayers image binds: 768 MiB, 384 MiB on GPUs with less than
// 7 GiB (title 1's BC6H arrays are 85 MB a layer).
[[nodiscard]] VkDeviceSize SparseLayersCap(const Device& device) {
    return device.GetDeviceLocalMemory() < (u64{7} << 30) ? (VkDeviceSize{384} << 20)
                                                           : (VkDeviceSize{768} << 20);
}

// The host format of a huge ASTC array or of ASTC with uneven block mips: the GPU decoder's RGBA8
// (sRGB kept for sampling).
[[nodiscard]] std::optional<VkFormat> HugeAstcHostFormat(const Device& device, const ImageInfo& info) {
    if (!IsHugeAstcArray(device, info) && !IsUnevenMipRgba8Astc(device, info)) {
        return std::nullopt;
    }
    return VideoCore::Surface::IsPixelFormatSRGB(info.format) ? VK_FORMAT_A8B8G8R8_SRGB_PACK32
                                                              : VK_FORMAT_A8B8G8R8_UNORM_PACK32;
}

constexpr std::array<VkFormat, 2> HugeAstcViewFormats{VK_FORMAT_A8B8G8R8_UNORM_PACK32,
                                                       VK_FORMAT_A8B8G8R8_SRGB_PACK32};

// Opaque ASTC (ImageInfo::astc_opaque) re-compressed to BC1 instead of BC3.
[[nodiscard]] bool IsOpaqueBc1Astc(const Device& device, const ImageInfo& info) {
    return info.astc_opaque && IsPixelFormatASTC(info.format) && !IsHugeAstcArray(device, info) &&
           !IsUnevenMipRgba8Astc(device, info);
}

constexpr std::array<VkFormat, 2> OpaqueAstcViewFormats{VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
                                                         VK_FORMAT_BC1_RGBA_SRGB_BLOCK};

[[nodiscard]] std::optional<VkFormat> AstcHostFormat(const Device& device, const ImageInfo& info) {
    if (IsOpaqueBc1Astc(device, info)) {
        return VideoCore::Surface::IsPixelFormatSRGB(info.format) ? VK_FORMAT_BC1_RGBA_SRGB_BLOCK
                                                                  : VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    }
    return HugeAstcHostFormat(device, info);
}

// ASTC that the GPU decodes and re-compresses to BC1/BC3 itself (TranscodeAstcToBcn): no CPU
// decode, so streamed textures are ready the frame they are first used instead of seconds later.
[[nodiscard]] bool WillUseGpuBcnTranscode(const Device& device, const ImageInfo& info) {
    if (!IsPixelFormatASTC(info.format) || device.IsOptimalAstcSupported() ||
        !gpu_astc_decoder_available || IsHugeAstcArray(device, info) ||
        IsUnevenMipRgba8Astc(device, info) || info.type != ImageType::e2D || info.size.depth != 1) {
        return false;
    }
    return Settings::values.accelerate_astc.GetValue() == Settings::AstcDecodeMode::Gpu &&
           Settings::GetAstcRecompression() != Settings::AstcRecompression::Uncompressed;
}

[[nodiscard]] bool WillUseAcceleratedAstcDecode(const Device& device, const ImageInfo& info) {
    if (!IsPixelFormatASTC(info.format) || device.IsOptimalAstcSupported() ||
        !gpu_astc_decoder_available) {
        return false;
    }
    if (IsHugeAstcArray(device, info) || IsUnevenMipRgba8Astc(device, info)) {
        return true;
    }
    if (Settings::values.accelerate_astc.GetValue() != Settings::AstcDecodeMode::Gpu) {
        return false;
    }
    return Settings::GetAstcRecompression() == Settings::AstcRecompression::Uncompressed &&
           info.size.depth == 1;
}

// The most a trimmed ASTC array keeps on the host: 512 MiB, 256 MiB on GPUs with less than 7 GiB
// (title 1's two terrain arrays took 724 MiB of a 6 GB card next to ~1.2 GB of other sparse terrain
// textures). EDEN_NX2_ASTC_ARRAY_MIB=<MiB> sets it.
[[nodiscard]] u64 TrimmedAstcArrayLimit(const Device& device) {
    static const u64 forced = [] {
        const char* value = std::getenv("EDEN_NX2_ASTC_ARRAY_MIB");
        return value != nullptr ? std::strtoull(value, nullptr, 10) << 20 : 0ULL;
    }();
    if (forced != 0) {
        return forced;
    }
    return device.GetDeviceLocalMemory() < (u64{7} << 30) ? (u64{256} << 20) : (u64{512} << 20);
}

// The host shape of an image whose decoded copy would not fit in VRAM: a GPU-decoded ASTC array
// over 1 GiB as RGBA8 (title 1's 8192x8192x68 ASTC 8x8 virtual textures are 24 GB each) keeps only
// its mip levels from `level_offset` on, so host mip 0 is guest mip `level_offset`. Normalized
// coordinates and derivative LOD still address it correctly; explicit LODs sample one level lower.
[[nodiscard]] ImageInfo TrimmedHostInfo(const Device& device, const ImageInfo& info, u32& level_offset) {
    level_offset = 0;
    if (!WillUseAcceleratedAstcDecode(device, info) || info.type != ImageType::e2D) {
        return info;
    }
    const auto decoded_bytes = [&](u32 skip) {
        u64 bytes = 0;
        for (s32 level = static_cast<s32>(skip); level < info.resources.levels; ++level) {
            bytes += u64{(std::max)(info.size.width >> level, 1U)} *
                     (std::max)(info.size.height >> level, 1U) * 4;
        }
        return bytes * info.resources.layers;
    };
    if (decoded_bytes(0) <= (u64{1} << 30)) {
        return info;
    }
    const u64 limit = TrimmedAstcArrayLimit(device);
    while (level_offset + 1 < static_cast<u32>(info.resources.levels) &&
           decoded_bytes(level_offset) > limit) {
        ++level_offset;
    }
    ImageInfo host = info;
    host.size.width = (std::max)(info.size.width >> level_offset, 1U);
    host.size.height = (std::max)(info.size.height >> level_offset, 1U);
    host.resources.levels -= static_cast<s32>(level_offset);
    LOG_WARNING(Render_Vulkan, "{}x{}x{} ASTC image would be {} MiB decoded; keeping mip levels {}+ ({}x{})",
                info.size.width, info.size.height, info.resources.layers, decoded_bytes(0) >> 20,
                level_offset, host.size.width, host.size.height);
    return host;
}

[[nodiscard]] bool IsCubeCompatible(const Device& device, const ImageInfo& info) {
    return info.type == ImageType::e2D && info.resources.layers >= 6 &&
           info.size.width == info.size.height && !device.HasBrokenCubeImageCompatibility();
}

[[nodiscard]] VkImageCreateInfo MakeImageCreateInfo(const Device& device, const ImageInfo& info,
                                                    std::optional<VkFormat> format_override = {}) {
    auto format_info =
        MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, false, info.format);
    if (format_override) {
        format_info.format = *format_override;
        format_info.attachable = false;
        format_info.storage = true;
    }
    VkImageCreateFlags flags{};
    if (IsCubeCompatible(device, info)) {
        flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    }
    if (info.type == ImageType::e3D) {
        flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    }
    const auto [samples_x, samples_y] = VideoCommon::SamplesLog2(info.num_samples);
    const bool allow_storage =
        info.num_samples == 1 || device.IsStorageImageMultisampleSupported();
    return VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = flags,
        .imageType = ConvertImageType(info.type),
        .format = format_info.format,
        .extent{
            .width = info.size.width >> samples_x,
            .height = info.size.height >> samples_y,
            .depth = info.size.depth,
        },
        .mipLevels = static_cast<u32>(info.resources.levels),
        .arrayLayers = static_cast<u32>(info.resources.layers),
        .samples = ConvertSampleCount(info.num_samples),
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = ImageUsageFlags(format_info, info.format, allow_storage),
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
}

[[nodiscard]] VkImageCreateInfo MakeMsaaScratchImageCreateInfo(const Device& device,
                                                               const ImageInfo& info,
                                                               VkImageUsageFlags usage) {
    ImageInfo temp_info = info;
    temp_info.num_samples = 1;
    VkImageCreateInfo image_ci = MakeImageCreateInfo(device, temp_info);
    image_ci.format =
        MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, true, info.format).format;
    image_ci.usage = usage;
    return image_ci;
}

[[nodiscard]] vk::Image MakeImage(const Device& device, const MemoryAllocator& allocator,
                                  const ImageInfo& info, std::span<const VkFormat> view_formats,
                                  std::optional<VkFormat> format_override = {},
                                  bool unbacked = false, bool aliased = false,
                                  bool may_fail = false, TextureCacheRuntime* pool = nullptr,
                                  ImagePoolKey* pool_key = nullptr, bool* pooled = nullptr) {
    if (info.type == ImageType::Buffer) {
        return vk::Image{};
    }
    VkImageCreateInfo image_ci = MakeImageCreateInfo(device, info, format_override);
    const VkImageFormatListCreateInfo image_format_list = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
        .pNext = nullptr,
        .viewFormatCount = static_cast<u32>(view_formats.size()),
        .pViewFormats = view_formats.data(),
    };
    if (view_formats.size() > 1) {
        image_ci.flags |=
            VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;

        const bool has_storage_compatible_view =
            std::any_of(view_formats.begin(), view_formats.end(), [&device](VkFormat view_format) {
                return device.IsFormatSupported(view_format, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT,
                                                FormatType::Optimal);
            });
        if (has_storage_compatible_view) {
            image_ci.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
        }

        if (device.IsKhrImageFormatListSupported()) {
            image_ci.pNext = &image_format_list;
        }
    }
    // EXP42: single-level, single-layer render targets can come from (and go back to) the pool.
    constexpr VkImageUsageFlags attachment_usage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (pool_key != nullptr && !unbacked && image_ci.imageType == VK_IMAGE_TYPE_2D &&
        image_ci.mipLevels == 1 && image_ci.arrayLayers == 1 &&
        image_ci.samples == VK_SAMPLE_COUNT_1_BIT && (image_ci.usage & attachment_usage) != 0) {
        *pool_key = ImagePoolKey{
            .valid = true,
            .flags = image_ci.flags,
            .type = image_ci.imageType,
            .format = image_ci.format,
            .width = image_ci.extent.width,
            .height = image_ci.extent.height,
            .depth = image_ci.extent.depth,
            .levels = image_ci.mipLevels,
            .layers = image_ci.arrayLayers,
            .samples = image_ci.samples,
            .tiling = image_ci.tiling,
            .usage = image_ci.usage,
            .view_formats = view_formats.data(),
            .view_format_count = view_formats.size(),
        };
        if (pool != nullptr) {
            if (vk::Image image = pool->TakePooledImage(*pool_key)) {
                if (pooled != nullptr) {
                    *pooled = true;
                }
                return image;
            }
        }
    }
    if (unbacked) {
        if (vk::Image image = allocator.CreateUnbackedImage(image_ci, aliased)) {
            return image;
        }
    }
    const MemoryTagScope memory_tag{MemoryTags::CurrentOr(MemoryTag::GuestImage)};
    return may_fail ? allocator.TryCreateImage(image_ci) : allocator.CreateImage(image_ci);
}

[[nodiscard]] vk::ImageView MakeStorageView(const vk::Device& device, u32 level, VkImage image,
                                            VkFormat format, u32 base_layer = 0,
                                            u32 layer_count = VK_REMAINING_ARRAY_LAYERS) {
    static constexpr VkImageViewUsageCreateInfo storage_image_view_usage_create_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
        .pNext = nullptr,
        .usage = VK_IMAGE_USAGE_STORAGE_BIT,
    };
    return device.CreateImageView(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = &storage_image_view_usage_create_info,
        .flags = 0,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
        .format = format,
        .components{
            .r = VK_COMPONENT_SWIZZLE_IDENTITY,
            .g = VK_COMPONENT_SWIZZLE_IDENTITY,
            .b = VK_COMPONENT_SWIZZLE_IDENTITY,
            .a = VK_COMPONENT_SWIZZLE_IDENTITY,
        },
        .subresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = level,
            .levelCount = 1,
            .baseArrayLayer = base_layer,
            .layerCount = layer_count,
        },
    });
}

[[nodiscard]] VkImageAspectFlags ImageAspectMask(PixelFormat format) {
    switch (VideoCore::Surface::GetFormatType(format)) {
    case VideoCore::Surface::SurfaceType::ColorTexture:
        return VK_IMAGE_ASPECT_COLOR_BIT;
    case VideoCore::Surface::SurfaceType::Depth:
        return VK_IMAGE_ASPECT_DEPTH_BIT;
    case VideoCore::Surface::SurfaceType::Stencil:
        return VK_IMAGE_ASPECT_STENCIL_BIT;
    case VideoCore::Surface::SurfaceType::DepthStencil:
        return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    default:
        ASSERT_MSG(false, "Invalid surface type");
        return VkImageAspectFlags{};
    }
}

[[nodiscard]] bool IsLdrAstcFormat(VkFormat format) {
    return format >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && format <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK;
}

[[nodiscard]] VkImageAspectFlags ImageViewAspectMask(const VideoCommon::ImageViewInfo& info) {
    if (info.IsRenderTarget()) {
        return ImageAspectMask(info.format);
    }
    bool any_r =
        std::ranges::any_of(info.Swizzle(), [](SwizzleSource s) { return s == SwizzleSource::R; });
    switch (info.format) {
    case PixelFormat::D24_UNORM_S8_UINT:
    case PixelFormat::D32_FLOAT_S8_UINT:
        // R = depth, G = stencil
        return any_r ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_STENCIL_BIT;
    case PixelFormat::S8_UINT_D24_UNORM:
        // R = stencil, G = depth
        return any_r ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
    case PixelFormat::D16_UNORM:
    case PixelFormat::D32_FLOAT:
    case PixelFormat::X8_D24_UNORM:
        return VK_IMAGE_ASPECT_DEPTH_BIT;
    case PixelFormat::S8_UINT:
        return VK_IMAGE_ASPECT_STENCIL_BIT;
    default:
        return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

[[nodiscard]] VkComponentSwizzle ComponentSwizzle(SwizzleSource swizzle) {
    switch (swizzle) {
    case SwizzleSource::Zero:
        return VK_COMPONENT_SWIZZLE_ZERO;
    case SwizzleSource::R:
        return VK_COMPONENT_SWIZZLE_R;
    case SwizzleSource::G:
        return VK_COMPONENT_SWIZZLE_G;
    case SwizzleSource::B:
        return VK_COMPONENT_SWIZZLE_B;
    case SwizzleSource::A:
        return VK_COMPONENT_SWIZZLE_A;
    case SwizzleSource::OneFloat:
    case SwizzleSource::OneInt:
        return VK_COMPONENT_SWIZZLE_ONE;
    }
    ASSERT_MSG(false, "Invalid swizzle={}", swizzle);
    return VK_COMPONENT_SWIZZLE_ZERO;
}

void SanitizeDepthStencilSwizzle(std::array<SwizzleSource, 4>& swizzle,
                                 bool supports_depth_stencil_swizzle_one) {
    if (supports_depth_stencil_swizzle_one) {
        return;
    }
    std::replace_if(swizzle.begin(), swizzle.end(),
                    [](SwizzleSource value) {
                        return value == SwizzleSource::OneFloat ||
                               value == SwizzleSource::OneInt;
                    },
                    SwizzleSource::Zero);
}

[[nodiscard]] VkImageViewType ImageViewType(Shader::TextureType type) {
    switch (type) {
    case Shader::TextureType::Color1D:
        return VK_IMAGE_VIEW_TYPE_1D;
    case Shader::TextureType::Color2D:
    case Shader::TextureType::Color2DRect:
        return VK_IMAGE_VIEW_TYPE_2D;
    case Shader::TextureType::ColorCube:
        return VK_IMAGE_VIEW_TYPE_CUBE;
    case Shader::TextureType::Color3D:
        return VK_IMAGE_VIEW_TYPE_3D;
    case Shader::TextureType::ColorArray1D:
        return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
    case Shader::TextureType::ColorArray2D:
        return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    case Shader::TextureType::ColorArrayCube:
        return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    case Shader::TextureType::Buffer:
        ASSERT_MSG(false, "Texture buffers can't be image views");
        return VK_IMAGE_VIEW_TYPE_1D;
    }
    ASSERT_MSG(false, "Invalid image view type={}", type);
    return VK_IMAGE_VIEW_TYPE_2D;
}

[[nodiscard]] VkImageViewType ImageViewType(VideoCommon::ImageViewType type) {
    switch (type) {
    case VideoCommon::ImageViewType::e1D:
        return VK_IMAGE_VIEW_TYPE_1D;
    case VideoCommon::ImageViewType::e2D:
    case VideoCommon::ImageViewType::Rect:
        return VK_IMAGE_VIEW_TYPE_2D;
    case VideoCommon::ImageViewType::Cube:
        return VK_IMAGE_VIEW_TYPE_CUBE;
    case VideoCommon::ImageViewType::e3D:
        return VK_IMAGE_VIEW_TYPE_3D;
    case VideoCommon::ImageViewType::e1DArray:
        return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
    case VideoCommon::ImageViewType::e2DArray:
        return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    case VideoCommon::ImageViewType::CubeArray:
        return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    case VideoCommon::ImageViewType::Buffer:
        ASSERT_MSG(false, "Texture buffers can't be image views");
        return VK_IMAGE_VIEW_TYPE_1D;
    }
    ASSERT_MSG(false, "Invalid image view type={}", type);
    return VK_IMAGE_VIEW_TYPE_2D;
}

[[nodiscard]] VkImageSubresourceLayers MakeImageSubresourceLayers(
    VideoCommon::SubresourceLayers subresource, VkImageAspectFlags aspect_mask) {
    return VkImageSubresourceLayers{
        .aspectMask = aspect_mask,
        .mipLevel = static_cast<u32>(subresource.base_level),
        .baseArrayLayer = static_cast<u32>(subresource.base_layer),
        .layerCount = static_cast<u32>(subresource.num_layers),
    };
}

[[nodiscard]] VkOffset3D MakeOffset3D(VideoCommon::Offset3D offset3d) {
    return VkOffset3D{
        .x = offset3d.x,
        .y = offset3d.y,
        .z = offset3d.z,
    };
}

[[nodiscard]] VkExtent3D MakeExtent3D(VideoCommon::Extent3D extent3d) {
    return VkExtent3D{
        .width = static_cast<u32>(extent3d.width),
        .height = static_cast<u32>(extent3d.height),
        .depth = static_cast<u32>(extent3d.depth),
    };
}

[[nodiscard]] VkImageCopy MakeImageCopy(const VideoCommon::ImageCopy& copy,
                                        VkImageAspectFlags aspect_mask) noexcept {
    return VkImageCopy{
        .srcSubresource = MakeImageSubresourceLayers(copy.src_subresource, aspect_mask),
        .srcOffset = MakeOffset3D(copy.src_offset),
        .dstSubresource = MakeImageSubresourceLayers(copy.dst_subresource, aspect_mask),
        .dstOffset = MakeOffset3D(copy.dst_offset),
        .extent = MakeExtent3D(copy.extent),
    };
}

[[nodiscard]] VkBufferImageCopy MakeBufferImageCopy(const VideoCommon::ImageCopy& copy, bool is_src,
                                                    VkImageAspectFlags aspect_mask) noexcept {
    return VkBufferImageCopy{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = MakeImageSubresourceLayers(
            is_src ? copy.src_subresource : copy.dst_subresource, aspect_mask),
        .imageOffset = MakeOffset3D(is_src ? copy.src_offset : copy.dst_offset),
        .imageExtent = MakeExtent3D(copy.extent),
    };
}

[[maybe_unused]] [[nodiscard]] boost::container::small_vector<VkBufferCopy, 16>
TransformBufferCopies(std::span<const VideoCommon::BufferCopy> copies, size_t buffer_offset) {
    boost::container::small_vector<VkBufferCopy, 16> result(copies.size());
    std::ranges::transform(
        copies, result.begin(), [buffer_offset](const VideoCommon::BufferCopy& copy) {
            return VkBufferCopy{
                .srcOffset = static_cast<VkDeviceSize>(copy.src_offset + buffer_offset),
                .dstOffset = static_cast<VkDeviceSize>(copy.dst_offset),
                .size = static_cast<VkDeviceSize>(copy.size),
            };
        });
    return result;
}

[[nodiscard]] boost::container::small_vector<VkBufferImageCopy, 16> TransformBufferImageCopies(
    std::span<const BufferImageCopy> copies, size_t buffer_offset, VkImageAspectFlags aspect_mask) {
    struct Maker {
        VkBufferImageCopy operator()(const BufferImageCopy& copy) const {
            return VkBufferImageCopy{
                .bufferOffset = copy.buffer_offset + buffer_offset,
                .bufferRowLength = copy.buffer_row_length,
                .bufferImageHeight = copy.buffer_image_height,
                .imageSubresource =
                    {
                        .aspectMask = aspect_mask,
                        .mipLevel = static_cast<u32>(copy.image_subresource.base_level),
                        .baseArrayLayer = static_cast<u32>(copy.image_subresource.base_layer),
                        .layerCount = static_cast<u32>(copy.image_subresource.num_layers),
                    },
                .imageOffset =
                    {
                        .x = copy.image_offset.x,
                        .y = copy.image_offset.y,
                        .z = copy.image_offset.z,
                    },
                .imageExtent =
                    {
                        .width = copy.image_extent.width,
                        .height = copy.image_extent.height,
                        .depth = copy.image_extent.depth,
                    },
            };
        }
        size_t buffer_offset;
        VkImageAspectFlags aspect_mask;
    };
    if (aspect_mask == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
        boost::container::small_vector<VkBufferImageCopy, 16> result(copies.size() * 2);
        std::ranges::transform(copies, result.begin(),
                               Maker{buffer_offset, VK_IMAGE_ASPECT_DEPTH_BIT});
        std::ranges::transform(copies, result.begin() + copies.size(),
                               Maker{buffer_offset, VK_IMAGE_ASPECT_STENCIL_BIT});
        return result;
    } else {
        boost::container::small_vector<VkBufferImageCopy, 16> result(copies.size());
        std::ranges::transform(copies, result.begin(), Maker{buffer_offset, aspect_mask});
        return result;
    }
}

[[nodiscard]] VkImageSubresourceRange MakeSubresourceRange(VkImageAspectFlags aspect_mask,
                                                           const SubresourceRange& range) {
    return VkImageSubresourceRange{
        .aspectMask = aspect_mask,
        .baseMipLevel = static_cast<u32>(range.base.level),
        .levelCount = static_cast<u32>(range.extent.levels),
        .baseArrayLayer = static_cast<u32>(range.base.layer),
        .layerCount = static_cast<u32>(range.extent.layers),
    };
}

// A guest view's levels on a host image that keeps only levels >= host_level_offset.
[[nodiscard]] SubresourceRange HostRange(const Image& image, SubresourceRange range) {
    const s32 skip = static_cast<s32>(image.host_level_offset);
    if (skip == 0) {
        return range;
    }
    const s32 host_levels = image.info.resources.levels - skip;
    const s32 base = std::clamp(range.base.level - skip, 0, host_levels - 1);
    const s32 end = std::clamp(range.base.level + range.extent.levels - skip, base + 1, host_levels);
    range.base.level = base;
    range.extent.levels = end - base;
    return range;
}

[[nodiscard]] VkImageSubresourceRange MakeSubresourceRange(const ImageView* image_view) {
    SubresourceRange range = image_view->range;
    if (True(image_view->flags & VideoCommon::ImageViewFlagBits::Slice)) {
        // Slice image views always affect a single layer, but their subresource range corresponds
        // to the slice. Override the value to affect a single layer.
        range.base.layer = 0;
        range.extent.layers = 1;
    }
    return MakeSubresourceRange(ImageAspectMask(image_view->format), range);
}

[[nodiscard]] VkImageSubresourceLayers MakeSubresourceLayers(const ImageView* image_view) {
    return VkImageSubresourceLayers{
        .aspectMask = ImageAspectMask(image_view->format),
        .mipLevel = static_cast<u32>(image_view->range.base.level),
        .baseArrayLayer = static_cast<u32>(image_view->range.base.layer),
        .layerCount = static_cast<u32>(image_view->range.extent.layers),
    };
}

[[nodiscard]] SwizzleSource ConvertGreenRed(SwizzleSource value) {
    switch (value) {
    case SwizzleSource::G:
        return SwizzleSource::R;
    default:
        return value;
    }
}

[[nodiscard]] SwizzleSource SwapBlueRed(SwizzleSource value) {
    switch (value) {
    case SwizzleSource::R:
        return SwizzleSource::B;
    case SwizzleSource::B:
        return SwizzleSource::R;
    default:
        return value;
    }
}

[[nodiscard]] SwizzleSource SwapGreenRed(SwizzleSource value) {
    switch (value) {
    case SwizzleSource::R:
        return SwizzleSource::G;
    case SwizzleSource::G:
        return SwizzleSource::R;
    default:
        return value;
    }
}

[[nodiscard]] SwizzleSource SwapSpecial(SwizzleSource value) {
    switch (value) {
    case SwizzleSource::A:
        return SwizzleSource::R;
    case SwizzleSource::R:
        return SwizzleSource::A;
    case SwizzleSource::G:
        return SwizzleSource::B;
    case SwizzleSource::B:
        return SwizzleSource::G;
    default:
        return value;
    }
}
struct RangedBarrierRange {
    u32 min_mip = (std::numeric_limits<u32>::max)();
    u32 max_mip = (std::numeric_limits<u32>::min)();
    u32 min_layer = (std::numeric_limits<u32>::max)();
    u32 max_layer = (std::numeric_limits<u32>::min)();

    void AddLayers(const VkImageSubresourceLayers& layers) {
        min_mip = (std::min)(min_mip, layers.mipLevel);
        max_mip = (std::max)(max_mip, layers.mipLevel + 1);
        min_layer = (std::min)(min_layer, layers.baseArrayLayer);
        max_layer = (std::max)(max_layer, layers.baseArrayLayer + layers.layerCount);
    }

    VkImageSubresourceRange SubresourceRange(VkImageAspectFlags aspect_mask) const noexcept {
        return VkImageSubresourceRange{
            .aspectMask = aspect_mask,
            .baseMipLevel = min_mip,
            .levelCount = max_mip - min_mip,
            .baseArrayLayer = min_layer,
            .layerCount = max_layer - min_layer,
        };
    }
};
void CopyBufferToImage(vk::CommandBuffer cmdbuf, VkBuffer src_buffer, VkImage image,
                       VkImageAspectFlags aspect_mask, bool is_initialized,
                       std::span<const VkBufferImageCopy> copies) {
    static constexpr VkAccessFlags WRITE_ACCESS_FLAGS =
                                           VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                           VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    static constexpr VkAccessFlags READ_ACCESS_FLAGS = VK_ACCESS_SHADER_READ_BIT |
                                                       VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;

    //  Compute exact mip/layer range being written to
    RangedBarrierRange range;
    for (const auto& region : copies) {
        range.AddLayers(region.imageSubresource);
    }
    const VkImageSubresourceRange subresource_range = range.SubresourceRange(aspect_mask);

    const VkImageMemoryBarrier read_barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = static_cast<VkAccessFlags>(is_initialized ? WRITE_ACCESS_FLAGS
                                                                       : VK_ACCESS_NONE),
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = is_initialized ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = subresource_range,
    };

    const VkImageMemoryBarrier write_barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = WRITE_ACCESS_FLAGS | READ_ACCESS_FLAGS,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = subresource_range,
    };

    cmdbuf.PipelineBarrier(is_initialized
                               ? vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER
                               : VkPipelineStageFlags{VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT},
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, read_barrier);
    cmdbuf.CopyBufferToImage(src_buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, copies);
    // TODO: Move this to another API
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                           vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER, 0, nullptr, nullptr,
                           write_barrier);
}

[[nodiscard]] VkImageBlit MakeImageBlit(const Region2D& dst_region, const Region2D& src_region,
                                        const VkImageSubresourceLayers& dst_layers,
                                        const VkImageSubresourceLayers& src_layers) {
    return VkImageBlit{
        .srcSubresource = src_layers,
        .srcOffsets =
            {
                {
                    .x = src_region.start.x,
                    .y = src_region.start.y,
                    .z = 0,
                },
                {
                    .x = src_region.end.x,
                    .y = src_region.end.y,
                    .z = 1,
                },
            },
        .dstSubresource = dst_layers,
        .dstOffsets =
            {
                {
                    .x = dst_region.start.x,
                    .y = dst_region.start.y,
                    .z = 0,
                },
                {
                    .x = dst_region.end.x,
                    .y = dst_region.end.y,
                    .z = 1,
                },
            },
    };
}

[[nodiscard]] bool HaveSameExtent(const Region2D& dst_region, const Region2D& src_region) {
    return dst_region.end.x - dst_region.start.x == src_region.end.x - src_region.start.x &&
           dst_region.end.y - dst_region.start.y == src_region.end.y - src_region.start.y;
}

[[nodiscard]] VkImageResolve MakeImageResolve(const Region2D& dst_region,
                                              const Region2D& src_region,
                                              const VkImageSubresourceLayers& dst_layers,
                                              const VkImageSubresourceLayers& src_layers) {
    return VkImageResolve{
        .srcSubresource = src_layers,
        .srcOffset =
            {
                .x = src_region.start.x,
                .y = src_region.start.y,
                .z = 0,
            },
        .dstSubresource = dst_layers,
        .dstOffset =
            {
                .x = dst_region.start.x,
                .y = dst_region.start.y,
                .z = 0,
            },
        .extent =
            {
                .width = static_cast<u32>(dst_region.end.x - dst_region.start.x),
                .height = static_cast<u32>(dst_region.end.y - dst_region.start.y),
                .depth = 1,
            },
    };
}

void TryTransformSwizzleIfNeeded(PixelFormat format, std::array<SwizzleSource, 4>& swizzle,
                                 bool emulate_bgr565, bool emulate_a4b4g4r4) {
    switch (format) {
    case PixelFormat::A1B5G5R5_UNORM:
        std::ranges::transform(swizzle, swizzle.begin(), SwapBlueRed);
        break;
    case PixelFormat::B5G6R5_UNORM:
        if (emulate_bgr565) {
            std::ranges::transform(swizzle, swizzle.begin(), SwapBlueRed);
        }
        break;
    case PixelFormat::A5B5G5R1_UNORM:
        std::ranges::transform(swizzle, swizzle.begin(), SwapSpecial);
        break;
    case PixelFormat::G4R4_UNORM:
        std::ranges::transform(swizzle, swizzle.begin(), SwapGreenRed);
        break;
    case PixelFormat::A4B4G4R4_UNORM:
        if (emulate_a4b4g4r4) {
            std::ranges::reverse(swizzle);
        }
        break;
    default:
        break;
    }
}



[[nodiscard]] VkFormat Format(Shader::ImageFormat format) {
    switch (format) {
    case Shader::ImageFormat::Typeless:
        break;
    case Shader::ImageFormat::R8_SINT:
        return VK_FORMAT_R8_SINT;
    case Shader::ImageFormat::R8_UINT:
        return VK_FORMAT_R8_UINT;
    case Shader::ImageFormat::R16_UINT:
        return VK_FORMAT_R16_UINT;
    case Shader::ImageFormat::R16_SINT:
        return VK_FORMAT_R16_SINT;
    case Shader::ImageFormat::R32_UINT:
        return VK_FORMAT_R32_UINT;
    case Shader::ImageFormat::R32G32_UINT:
        return VK_FORMAT_R32G32_UINT;
    case Shader::ImageFormat::R32G32B32A32_UINT:
        return VK_FORMAT_R32G32B32A32_UINT;
    }
    ASSERT_MSG(false, "Invalid image format={}", format);
    return VK_FORMAT_R32_UINT;
}

void BlitScale(Scheduler& scheduler, VkImage src_image, VkImage dst_image, const ImageInfo& info,
               VkImageAspectFlags aspect_mask, const Settings::ResolutionScalingInfo& resolution,
               bool up_scaling = true) {
    const bool is_2d = info.type == ImageType::e2D;
    const auto resources = info.resources;
    const VkExtent2D extent{
        .width = info.size.width,
        .height = info.size.height,
    };
    // Depth and integer formats must use NEAREST filter for blits.
    const bool is_color{aspect_mask == VK_IMAGE_ASPECT_COLOR_BIT};
    const bool is_bilinear{is_color && !IsPixelFormatInteger(info.format)};
    const VkFilter vk_filter = is_bilinear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([dst_image, src_image, extent, resources, aspect_mask, resolution, is_2d,
                      vk_filter, up_scaling](vk::CommandBuffer cmdbuf) {
        const u32 scaled_width = resolution.ScaleUp(extent.width);
        const u32 scaled_height = is_2d ? resolution.ScaleUp(extent.height) : extent.height;
        const VkOffset2D src_size{
            .x = static_cast<s32>(up_scaling ? extent.width : scaled_width),
            .y = static_cast<s32>(up_scaling ? extent.height : scaled_height),
        };
        const VkOffset2D dst_size{
            .x = static_cast<s32>(up_scaling ? scaled_width : extent.width),
            .y = static_cast<s32>(up_scaling ? scaled_height : extent.height),
        };
        boost::container::small_vector<VkImageBlit, 4> regions;
        regions.reserve(resources.levels);
        for (s32 level = 0; level < resources.levels; level++) {
            regions.push_back({
                .srcSubresource{
                    .aspectMask = aspect_mask,
                    .mipLevel = static_cast<u32>(level),
                    .baseArrayLayer = 0,
                    .layerCount = static_cast<u32>(resources.layers),
                },
                .srcOffsets{
                    {
                        .x = 0,
                        .y = 0,
                        .z = 0,
                    },
                    {
                        .x = (std::max)(1, src_size.x >> level),
                        .y = (std::max)(1, src_size.y >> level),
                        .z = 1,
                    },
                },
                .dstSubresource{
                    .aspectMask = aspect_mask,
                    .mipLevel = static_cast<u32>(level),
                    .baseArrayLayer = 0,
                    .layerCount = static_cast<u32>(resources.layers),
                },
                .dstOffsets{
                    {
                        .x = 0,
                        .y = 0,
                        .z = 0,
                    },
                    {
                        .x = (std::max)(1, dst_size.x >> level),
                        .y = (std::max)(1, dst_size.y >> level),
                        .z = 1,
                    },
                },
            });
        }
        const VkImageSubresourceRange subresource_range{
            .aspectMask = aspect_mask,
            .baseMipLevel = 0,
            .levelCount = VK_REMAINING_MIP_LEVELS,
            .baseArrayLayer = 0,
            .layerCount = VK_REMAINING_ARRAY_LAYERS,
        };
        const std::array read_barriers{
            VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = src_image,
                .subresourceRange = subresource_range,
            },
            VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT |
                                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, // Discard contents
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = dst_image,
                .subresourceRange = subresource_range,
            },
        };
        const std::array write_barriers{
            VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = 0,
                .dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = src_image,
                .subresourceRange = subresource_range,
            },
            VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = dst_image,
                .subresourceRange = subresource_range,
            },
        };
        cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       0, nullptr, nullptr, read_barriers);
        cmdbuf.BlitImage(src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst_image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, regions, vk_filter);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, vk::PIPELINE_STAGE_GRAPHICS_COMPUTE,
                       0, nullptr, nullptr, write_barriers);
    });
}

// Uncompressed ASTC needs more video memory than a Switch 2 title leaves under a 12 GiB budget
// (Mario Kart World filled a 16 GB card's 13.5 GB budget; a 12 GB card has ~11.2 GB): there the
// textures went unbacked (garbage) and the game crashed. Use BC3 for this session.
void UseBc3ForSwitch2OnSmallCards(const Device& device) {
    using namespace Common::Literals;
    if (!Settings::CurrentProgramIsSwitch2() || device.IsOptimalAstcSupported() ||
        Settings::GetAstcRecompression() != Settings::AstcRecompression::Uncompressed) {
        return;
    }
    const u64 budget = device.GetDeviceMemoryBudget();
    const u64 memory = budget != 0 ? budget : device.GetDeviceLocalMemory();
    const char* keep = std::getenv("EDEN_NX2_KEEP_ASTC");
    if (memory >= 20_GiB || (keep != nullptr && keep[0] == '1')) {
        return;
    }
    LOG_WARNING(Render_Vulkan,
                "Switch 2 program with a {} MiB video memory budget: ASTC textures are "
                "re-compressed to BC3 for this session (uncompressed ASTC needs a 20 GiB budget). "
                "Set EDEN_NX2_KEEP_ASTC=1 to keep them uncompressed",
                memory >> 20);
    Settings::SetAstcRecompressionOverride(Settings::AstcRecompression::Bc3);
}
} // Anonymous namespace

TextureCacheRuntime::TextureCacheRuntime(const Device& device_, Scheduler& scheduler_,
                                         MemoryAllocator& memory_allocator_,
                                         StagingBufferPool& staging_buffer_pool_,
                                         BlitImageHelper& blit_image_helper_,
                                         RenderPassCache& render_pass_cache_,
                                         DescriptorPool& descriptor_pool_,
                                         ComputePassDescriptorQueue& compute_pass_descriptor_queue_)
    : device{device_}, scheduler{scheduler_}, memory_allocator{memory_allocator_},
      staging_buffer_pool{staging_buffer_pool_}, blit_image_helper{blit_image_helper_},
      render_pass_cache{render_pass_cache_}, descriptor_pool{descriptor_pool_},
      compute_pass_descriptor_queue{compute_pass_descriptor_queue_},
      resolution{Settings::values.resolution_info} {
    // Before anything below reads astc_recompression (passes, view formats).
    UseBc3ForSwitch2OnSmallCards(device);
    // Huge ASTC arrays always decode on the GPU (IsHugeAstcArray), whatever the decode setting.
    gpu_astc_decoder_available = true;
    if (Settings::values.accelerate_astc.GetValue() == Settings::AstcDecodeMode::Gpu ||
        !device.IsOptimalAstcSupported()) {
        try {
            astc_decoder_pass.emplace(device, scheduler, descriptor_pool, staging_buffer_pool,
                                      compute_pass_descriptor_queue, memory_allocator);
            if (Settings::GetAstcRecompression() != Settings::AstcRecompression::Uncompressed) {
                bc_encoder_pass.emplace(device, scheduler, descriptor_pool,
                                        compute_pass_descriptor_queue);
            }
        } catch (const vk::Exception& exception) {
            LOG_WARNING(Render_Vulkan,
                        "The driver could not build the GPU ASTC decoder ({}): ASTC textures "
                        "decode on the CPU",
                        exception.what());
            astc_decoder_pass.reset();
            bc_encoder_pass.reset();
            gpu_astc_decoder_available = false;
        }
    }
    if (!device.IsKhrImageFormatListSupported()) {
        return;
    }
    for (size_t index_a = 0; index_a < VideoCore::Surface::MaxPixelFormat; index_a++) {
        const auto image_format = static_cast<PixelFormat>(index_a);
        if (IsPixelFormatASTC(image_format) && !device.IsOptimalAstcSupported() &&
            Settings::GetAstcRecompression() == Settings::AstcRecompression::Uncompressed) {
            view_formats[index_a].push_back(VK_FORMAT_A8B8G8R8_UNORM_PACK32);
        }
        for (size_t index_b = 0; index_b < VideoCore::Surface::MaxPixelFormat; index_b++) {
            const auto view_format = static_cast<PixelFormat>(index_b);
            if (VideoCore::Surface::IsViewCompatible(image_format, view_format, false, true)) {
                const auto view_info =
                    MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, true, view_format);
                view_formats[index_a].push_back(view_info.format);
            }
        }
    }

    if (Settings::values.gpu_unswizzle_enabled.GetValue()) {
        bl3d_unswizzle_pass.emplace(device, scheduler, descriptor_pool,
                                   staging_buffer_pool, compute_pass_descriptor_queue);
    }
}

void TextureCacheRuntime::Finish() {
    scheduler.Finish();
}

StagingBufferRef TextureCacheRuntime::UploadStagingBuffer(size_t size, bool deferred) {
    return staging_buffer_pool.Request(size, MemoryUsage::Upload, deferred);
}

StagingBufferRef TextureCacheRuntime::DownloadStagingBuffer(size_t size, bool deferred) {
    return staging_buffer_pool.Request(size, MemoryUsage::Download, deferred);
}

void TextureCacheRuntime::FreeDeferredStagingBuffer(StagingBufferRef& ref) {
    staging_buffer_pool.FreeDeferred(ref);
}

bool TextureCacheRuntime::ShouldReinterpret(Image& dst, Image& src) {
    if (VideoCore::Surface::GetFormatType(dst.info.format) ==
            VideoCore::Surface::SurfaceType::DepthStencil &&
        !device.IsExtShaderStencilExportSupported()) {
        return true;
    }

    if (VideoCore::Surface::GetFormatType(src.info.format) ==
            VideoCore::Surface::SurfaceType::DepthStencil &&
        !device.IsExtShaderStencilExportSupported()) {
        return true;
    }

    if (dst.info.format == PixelFormat::D32_FLOAT_S8_UINT ||
        src.info.format == PixelFormat::D32_FLOAT_S8_UINT) {
        return true;
    }
    return false;
}

VkBuffer TextureCacheRuntime::GetTemporaryBuffer(size_t needed_size) {
    const auto level = (8 * sizeof(size_t)) - std::countl_zero(needed_size - 1ULL);
    if (buffers[level]) {
        return *buffers[level];
    }
    const auto new_size = Common::NextPow2(needed_size);
    static constexpr VkBufferUsageFlags flags =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const VkBufferCreateInfo temp_ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = new_size,
        .usage = flags,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    const MemoryTagScope memory_tag{MemoryTag::TextureHelper};
    buffers[level] = memory_allocator.CreateBuffer(temp_ci, MemoryUsage::DeviceLocal);
    return *buffers[level];
}

VkImageView TextureCacheRuntime::GetOrCreateResolveShadow(VkImage msaa_image, VkFormat format,
                                                          VkExtent2D extent, u32 layers,
                                                          VkImageAspectFlags aspect_mask) {
    ResolveShadow& shadow = resolve_shadows[msaa_image];
    if (shadow.image && shadow.format == format && shadow.extent.width == extent.width &&
        shadow.extent.height == extent.height && shadow.layers == layers &&
        shadow.aspect_mask == aspect_mask) {
        return *shadow.view;
    }
    VkImageUsageFlags shadow_usage =
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if ((aspect_mask & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) != 0) {
        shadow_usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    } else {
        shadow_usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    }
    if (shadow.image) {
        pending_resolve_shadows.emplace_back(scheduler.CurrentTick(), std::move(shadow));
    }
    const MemoryTagScope memory_tag{MemoryTag::TextureHelper};
    shadow.image = memory_allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = {extent.width, extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = layers,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = shadow_usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    shadow.view = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = *shadow.image,
        .viewType = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .components{},
        .subresourceRange{
            .aspectMask = aspect_mask,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = layers,
        },
    });
    shadow.format = format;
    shadow.extent = extent;
    shadow.layers = layers;
    shadow.aspect_mask = aspect_mask;
    shadow.up_to_date = false;
    return *shadow.view;
}

const TextureCacheRuntime::ResolveShadow* TextureCacheRuntime::GetValidResolveShadow(
    VkImage msaa_image) const {
    const auto it = resolve_shadows.find(msaa_image);
    if (it == resolve_shadows.end() || !it->second.up_to_date || !it->second.image) {
        return nullptr;
    }
    return &it->second;
}

void TextureCacheRuntime::MarkResolveShadowUpToDate(VkImage msaa_image) {
    const auto it = resolve_shadows.find(msaa_image);
    if (it != resolve_shadows.end()) {
        it->second.up_to_date = true;
    }
}

void TextureCacheRuntime::InvalidateResolveShadow(VkImage msaa_image) {
    const auto it = resolve_shadows.find(msaa_image);
    if (it != resolve_shadows.end()) {
        it->second.up_to_date = false;
    }
}

void TextureCacheRuntime::EraseResolveShadow(VkImage msaa_image) {
    const auto it = resolve_shadows.find(msaa_image);
    if (it == resolve_shadows.end()) {
        return;
    }
    if (it->second.image) {
        pending_resolve_shadows.emplace_back(scheduler.CurrentTick(), std::move(it->second));
    }
    resolve_shadows.erase(it);
}

void TextureCacheRuntime::BarrierFeedbackLoop() {
    // The draw samples the bound depth/stencil image: ending the pass makes what earlier draws
    // of the pass wrote to it visible. If none of them wrote it, there is nothing to make
    // visible (Eglantine bdaf482d3f). Off unless EDEN_NX2_DEPTH_READ_KEEP_PASS=1: depth clears
    // recorded inside an open pass (RasterizerVulkan::Clear) do not note their write yet.
    static const bool keep_pass = [] {
        const char* const value = std::getenv("EDEN_NX2_DEPTH_READ_KEEP_PASS");
        return value != nullptr && value[0] == '1';
    }();
    if (keep_pass && scheduler.IsRenderPassActive() && !scheduler.RenderPassWroteDepthStencil()) {
        return;
    }
    scheduler.RequestOutsideRenderPassOperationContext();
}

ByteAddressedShadow TextureCacheRuntime::CreateByteAddressedShadow(const ImageView& view) {
    const auto* image{view.BackingImage()};
    const auto extent{PackedByteAddressedExtent(view.size.width, view.size.height)};
    if (!image || !extent || view.format != PixelFormat::R16_FLOAT ||
        (view.type != VideoCommon::ImageViewType::e2D &&
         view.type != VideoCommon::ImageViewType::Rect) ||
        view.Samples() != VK_SAMPLE_COUNT_1_BIT || view.range.base.level != 0 ||
        view.range.base.layer != 0 || view.range.extent.layers != 1 ||
        view.range.extent.levels != 1 || view.size.depth != 1 || view.IsRescaled() ||
        image->info.is_sparse || image->info.type != ImageType::e2D ||
        True(image->flags & (ImageFlagBits::SparseLayers | ImageFlagBits::SparseTiles)) ||
        image->info.resources.levels != 1 ||
        image->info.resources.layers != 1 || image->info.size != view.size ||
        image->info.format != PixelFormat::R16_FLOAT ||
        view.ImageHandle() != image->Handle() ||
        (image->UsageFlags() & (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)) !=
            (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT) ||
        !device.IsFormatSupported(VK_FORMAT_R16_SFLOAT,
                                  VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT,
                                  FormatType::Optimal) ||
        !device.IsFormatSupported(VK_FORMAT_R32_UINT,
                                  VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                      VK_FORMAT_FEATURE_TRANSFER_DST_BIT,
                                  FormatType::Optimal)) {
        throw Shader::RuntimeError("Unsupported packed R16 byte-addressed image binding");
    }
    const MemoryTagScope memory_tag{MemoryTag::TextureHelper};
    PendingByteAddressedShadow pending{
        .tick = (std::numeric_limits<u64>::max)(),
        .image = memory_allocator.CreateImage(VkImageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_R32_UINT,
            .extent = {extent->width, extent->height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        }),
        .view = {},
        .bridge = memory_allocator.CreateBuffer(VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = u64{view.size.width} * view.size.height * 2,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        }, MemoryUsage::DeviceLocal),
    };
    pending.view = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = *pending.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = VK_FORMAT_R32_UINT,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    });
    const ByteAddressedShadow result{
        .original = image->Handle(),
        .packed = *pending.image,
        .view = *pending.view,
        .bridge = *pending.bridge,
        .extent = *extent,
    };
    pending_byte_addressed_shadows.push_back(std::move(pending));
    return result;
}

void TextureCacheRuntime::RetireByteAddressedShadow(const ByteAddressedShadow& shadow) {
    for (auto& pending : pending_byte_addressed_shadows) {
        if (*pending.image == shadow.packed) {
            pending.tick = scheduler.CurrentTick();
            return;
        }
    }
}

void TextureCacheRuntime::CopyByteAddressedShadow(vk::CommandBuffer cmdbuf,
                                                 const ByteAddressedShadow& shadow, bool unpack) {
    const VkImage src{unpack ? shadow.packed : shadow.original};
    const VkImage dst{unpack ? shadow.original : shadow.packed};
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    const std::array pre{
        VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = src,
            .subresourceRange = range,
        },
        VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = unpack ? VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT : 0U,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = unpack ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = dst,
            .subresourceRange = range,
        },
    };
    const VkBufferMemoryBarrier bridge_reuse{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = unpack ? VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT : 0U,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = shadow.bridge,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           0, {}, bridge_reuse, pre);
    const VkBufferImageCopy src_copy{
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageExtent = {shadow.extent.width * (unpack ? 1U : 2U), shadow.extent.height, 1},
    };
    const VkBufferImageCopy dst_copy{
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageExtent = {shadow.extent.width * (unpack ? 2U : 1U), shadow.extent.height, 1},
    };
    cmdbuf.CopyImageToBuffer(src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, shadow.bridge, src_copy);
    const VkMemoryBarrier bridge_barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           0, bridge_barrier);
    cmdbuf.CopyBufferToImage(shadow.bridge, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, dst_copy);
    const std::array post{
        VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = src,
            .subresourceRange = range,
        },
        VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = dst,
            .subresourceRange = range,
        },
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           0, {}, {}, post);
}

void TextureCacheRuntime::ReinterpretImage(Image& dst, Image& src,
                                           std::span<const VideoCommon::ImageCopy> copies) {
    if (ENABLE_MSAA_RESOLVE_CONSUME) {
        InvalidateResolveShadow(dst.Handle());
    }
    boost::container::small_vector<VkBufferImageCopy, 16> vk_in_copies(copies.size());
    boost::container::small_vector<VkBufferImageCopy, 16> vk_out_copies(copies.size());
    const VkImageAspectFlags src_aspect_mask = src.AspectMask();
    const VkImageAspectFlags dst_aspect_mask = dst.AspectMask();
    // A VkBufferImageCopy region may name only one aspect (VUID-VkBufferImageCopy-aspectMask-09103).
    // Combined depth-stencil images were copied with DEPTH|STENCIL in a single region, which is
    // undefined behaviour and loses the device on NVIDIA (no VK_EXT_shader_stencil_export, so every
    // colour <-> depth-stencil alias lands here). Reinterpret through the depth aspect only.
    const auto copy_aspect = [](VkImageAspectFlags mask) {
        return mask == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)
                   ? VkImageAspectFlags{VK_IMAGE_ASPECT_DEPTH_BIT}
                   : mask;
    };
    const VkImageAspectFlags src_copy_aspect = copy_aspect(src_aspect_mask);
    const VkImageAspectFlags dst_copy_aspect = copy_aspect(dst_aspect_mask);

    const auto bpp_in = BytesPerBlock(src.info.format) / DefaultBlockWidth(src.info.format);
    const auto bpp_out = BytesPerBlock(dst.info.format) / DefaultBlockWidth(dst.info.format);
    std::ranges::transform(copies, vk_in_copies.begin(),
                           [src_copy_aspect, bpp_in, bpp_out](const auto& copy) {
                               auto copy2 = copy;
                               copy2.src_offset.x = (bpp_out * copy.src_offset.x) / bpp_in;
                               copy2.extent.width = (bpp_out * copy.extent.width) / bpp_in;
                               return MakeBufferImageCopy(copy2, true, src_copy_aspect);
                           });
    std::ranges::transform(copies, vk_out_copies.begin(), [dst_copy_aspect](const auto& copy) {
        return MakeBufferImageCopy(copy, false, dst_copy_aspect);
    });
    const u32 img_bpp = BytesPerBlock(dst.info.format);
    size_t total_size = 0;
    for (const auto& copy : copies) {
        total_size += copy.extent.width * copy.extent.height * copy.extent.depth * img_bpp;
    }
    const VkBuffer copy_buffer = GetTemporaryBuffer(total_size);
    const VkImage dst_image = dst.Handle();
    const VkImage src_image = src.Handle();
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([dst_image, src_image, copy_buffer, src_aspect_mask, dst_aspect_mask,
                      vk_in_copies, vk_out_copies](vk::CommandBuffer cmdbuf) {
        RangedBarrierRange dst_range;
        RangedBarrierRange src_range;
        for (const VkBufferImageCopy& copy : vk_in_copies) {
            src_range.AddLayers(copy.imageSubresource);
        }
        for (const VkBufferImageCopy& copy : vk_out_copies) {
            dst_range.AddLayers(copy.imageSubresource);
        }
        static constexpr VkMemoryBarrier READ_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        };
        static constexpr VkMemoryBarrier WRITE_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        const std::array pre_barriers{
            VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = src_image,
                .subresourceRange = src_range.SubresourceRange(src_aspect_mask),
            },
        };
        const std::array middle_in_barrier{
            VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = 0,
                .dstAccessMask = 0,
                .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = src_image,
                .subresourceRange = src_range.SubresourceRange(src_aspect_mask),
            },
        };
        const std::array middle_out_barrier{
            VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = dst_image,
                .subresourceRange = dst_range.SubresourceRange(dst_aspect_mask),
            },
        };
        const std::array post_barriers{
            VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                 VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = dst_image,
                .subresourceRange = dst_range.SubresourceRange(dst_aspect_mask),
            },
        };
        cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       0, {}, {}, pre_barriers);

        cmdbuf.CopyImageToBuffer(src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, copy_buffer,
                                 vk_in_copies);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, vk::PIPELINE_STAGE_GRAPHICS_COMPUTE,
                       0, WRITE_BARRIER, nullptr, middle_in_barrier);

        cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       0, READ_BARRIER, {}, middle_out_barrier);
        // middle_out_barrier moved dst to TRANSFER_DST_OPTIMAL; naming GENERAL here mismatched the
        // actual layout (VUID-vkCmdCopyBufferToImage-dstImageLayout-00180).
        cmdbuf.CopyBufferToImage(copy_buffer, dst_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 vk_out_copies);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, vk::PIPELINE_STAGE_GRAPHICS_COMPUTE,
                       0, {}, {}, post_barriers);
    });
}

bool TextureCacheRuntime::WantsTemporalUpscaleDepth() const {
    // FSR 3.1 replaces DLSS; the temporal stand-in only runs when FSR cannot (no depth, a device
    // without the formats, a failed start). EDEN_NX2_DLSS_REPLACEMENT=standin forces
    // the stand-in for A/B runs.
    static const bool force_standin = [] {
        const char* const value = std::getenv("EDEN_NX2_DLSS_REPLACEMENT");
        return value && std::string_view{value} == "standin";
    }();
    if (force_standin) {
        return false;
    }
    if (UsesNvidiaDlss()) {
        return true;
    }
#ifdef LICHII_METALFX
    if (Settings::MetalFxTemporalChosen() && !metalfx_unsupported) {
        return true;
    }
#endif
    if (Settings::AmdFsr4Chosen() && !fsr4_unsupported) {
        return true;
    }
    return !fsr3_unsupported;
}

bool TextureCacheRuntime::UsesNvidiaDlss() const {
#ifdef EGLANTINE_STREAMLINE
    static const std::string_view replacement = [] {
        const char* const value = std::getenv("EDEN_NX2_DLSS_REPLACEMENT");
        return value ? std::string_view{value} : std::string_view{};
    }();
    if (replacement == "fsr" || replacement == "standin" || replacement == "native") {
        return false;
    }
    const bool wanted = Settings::NvidiaDlssChosen() || replacement == "dlss";
    return wanted && !dlss_unsupported && Streamline::IsDlssAvailable();
#else
    return false;
#endif
}

std::optional<VideoCore::GameDlssProbe> TextureCacheRuntime::ProbeGameDlssOutput(
    ImageView& input, ImageView& output, std::optional<std::array<u32, 2>> input_region) {
    if (!dlss_network_probe) {
        dlss_network_probe = std::make_unique<DlssNetworkProbe>(
            device, scheduler, descriptor_pool, compute_pass_descriptor_queue, memory_allocator);
    }
    return dlss_network_probe->Probe(input.Handle(Shader::TextureType::Color2D),
                                     output.Handle(Shader::TextureType::Color2D),
                                     input_region.value_or(std::array<u32, 2>{}));
}

bool TextureCacheRuntime::UpscaleWithFsr3(Framebuffer* dst_framebuffer, ImageView& input_color,
                                          ImageView& motion_vectors, ImageView& depth,
                                          std::array<u32, 2> render_size,
                                          std::array<f32, 2> jitter,
                                          std::array<f32, 2> motion_scale) {
    if (!WantsTemporalUpscaleDepth()) {
        return false;
    }
    const VkExtent2D extent = dst_framebuffer->RenderArea();
    // A gap in the evaluations (a menu, a movie, a load) starts the history again.
    const auto now = std::chrono::steady_clock::now();
    const f32 frame_time_ms =
        fsr3_last_evaluation
            ? std::chrono::duration<f32, std::milli>(now - *fsr3_last_evaluation).count()
            : 1000.0f / 30.0f;
    const bool reset = !fsr3_last_evaluation || frame_time_ms > 250.0f;
    fsr3_last_evaluation = now;
#ifdef EGLANTINE_STREAMLINE
    if (UpscaleWithDlss(dst_framebuffer, input_color, motion_vectors, depth, render_size, jitter,
                        motion_scale, reset)) {
        fsr4_evaluated_last = false;
        return true;
    }
#endif
#ifdef LICHII_METALFX
    if (UpscaleWithMetalFx(dst_framebuffer, input_color, motion_vectors, depth, render_size,
                           jitter, motion_scale, reset)) {
        fsr4_evaluated_last = false;
        return true;
    }
#endif
    // FSR's own RCAS, set by the sharpness slider: FSR takes 0-1 for 2 to 0 stops, and the slider
    // stores stops x 100, so this is the percentage the slider shows; 0% (stored 200) is off.
    // EDEN_NX2_TEMPORAL_SHARPEN=<stops>|off overrides the slider.
    static const char* const sharpen_override = std::getenv("EDEN_NX2_TEMPORAL_SHARPEN");
    const int slider = Settings::values.fsr_sharpening_slider.GetValue();
    bool sharpen = slider < 200;
    f32 stops = static_cast<f32>(slider) / 100.0f;
    if (sharpen_override) {
        sharpen = std::string_view{sharpen_override} != "off";
        stops = static_cast<f32>(std::strtod(sharpen_override, nullptr));
    }
    const f32 sharpness = std::clamp(1.0f - stops / 2.0f, 0.0f, 1.0f);
    if (UpscaleWithFsr4(dst_framebuffer, input_color, motion_vectors, depth, render_size, jitter,
                        motion_scale, frame_time_ms, reset, sharpen, sharpness)) {
        fsr4_evaluated_last = true;
        return true;
    }
    const bool fsr3_reset = reset || std::exchange(fsr4_evaluated_last, false);
    if (fsr3_unsupported) {
        return false;
    }
    if (!fsr3_upscaler) {
        if (!Fsr3Upscaler::IsSupported(device)) {
            LOG_WARNING(Render_Vulkan, "FSR 3.1 needs formats or features this device lacks; "
                                       "using the temporal stand-in");
            fsr3_unsupported = true;
            return false;
        }
        try {
            fsr3_upscaler = std::make_unique<Fsr3Upscaler>(device, scheduler, memory_allocator,
                                                           staging_buffer_pool, descriptor_pool,
                                                           compute_pass_descriptor_queue);
        } catch (const vk::Exception& exception) {
            LOG_ERROR(Render_Vulkan, "FSR 3.1 failed to start ({}); using the temporal stand-in",
                      exception.what());
            fsr3_unsupported = true;
            return false;
        }
    }
    try {
        fsr3_upscaler->Upscale({
            .color = input_color.Handle(Shader::TextureType::Color2D),
            .depth = depth.DepthView(),
            .motion_vectors = motion_vectors.Handle(Shader::TextureType::Color2D),
            .render_size = {render_size[0], render_size[1]},
            .upscale_size = extent,
            .jitter = jitter,
            .motion_vector_scale = {motion_scale[0] * static_cast<f32>(render_size[0]),
                                    motion_scale[1] * static_cast<f32>(render_size[1])},
            .frame_time_ms = frame_time_ms,
            .reset = fsr3_reset,
            .sharpen = sharpen,
            .sharpness = sharpness,
        });
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan, "FSR 3.1 evaluation unavailable ({}); using the temporal stand-in",
                  exception.what());
        fsr3_unsupported = true;
        return false;
    }
    DrawUpscaledOutput(dst_framebuffer, fsr3_upscaler->OutputView(), fsr3_upscaler->OutputImage(),
                       fsr3_upscaler->PointSampler());
    static bool logged{};
    if (!logged) {
        logged = true;
        LOG_INFO(Render_Vulkan, "DLSS replaced by FSR 3.1: {}x{} -> {}x{}", render_size[0],
                 render_size[1], extent.width, extent.height);
    }
    return true;
}

bool TextureCacheRuntime::UpscaleWithFsr4(Framebuffer* dst_framebuffer, ImageView& input_color,
                                          ImageView& motion_vectors, ImageView& depth,
                                          std::array<u32, 2> render_size,
                                          std::array<f32, 2> jitter,
                                          std::array<f32, 2> motion_scale, f32 frame_time_ms,
                                          bool reset, bool sharpen, f32 sharpness) {
    if (!Settings::AmdFsr4Chosen() || fsr4_unsupported) {
        return false;
    }
    if (!fsr4_upscaler) {
        if (!Fsr4Upscaler::IsWanted(device)) {
            fsr4_unsupported = true;
            return false;
        }
        try {
            fsr4_upscaler = std::make_unique<Fsr4Upscaler>(
                device, scheduler, memory_allocator, Fsr4Upscaler::Profile::HdrReversedDepth);
        } catch (const vk::Exception& exception) {
            LOG_WARNING(Render_Vulkan, "AMD FSR 4 unavailable ({}); using FSR 3.1",
                        exception.what());
            fsr4_unsupported = true;
            return false;
        }
    }
    if (fsr4_upscaler->Failed()) {
        fsr4_unsupported = true;
        return false;
    }
    const auto view_host_extent = [&](const ImageView& view) -> VkExtent2D {
        const Image* const backing = view.BackingImage();
        if (backing && backing->IsRescaled()) {
            return {resolution.ScaleUp(view.size.width), resolution.ScaleUp(view.size.height)};
        }
        return {view.size.width, view.size.height};
    };
    const auto image = [&](ImageView& view, VkImageAspectFlags aspect) {
        return Fsr4Upscaler::Image{
            .image = view.ImageHandle(),
            .format = MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, false, view.format)
                          .format,
            .range{
                .aspectMask = aspect,
                .baseMipLevel = static_cast<u32>(view.range.base.level),
                .levelCount = 1,
                .baseArrayLayer = static_cast<u32>(view.range.base.layer),
                .layerCount = 1,
            },
            .barrier_aspect = ImageAspectMask(view.format),
            .extent = view_host_extent(view),
        };
    };
    const VkExtent2D extent = dst_framebuffer->RenderArea();
    const VkExtent2D color_extent = view_host_extent(input_color);
    const bool dynamic_region =
        color_extent.width > render_size[0] || color_extent.height > render_size[1];
    const std::optional<VkExtent2D> max_render_size =
        dynamic_region
            ? std::optional{VkExtent2D{std::clamp(color_extent.width, render_size[0], extent.width),
                                       std::clamp(color_extent.height, render_size[1],
                                                  extent.height)}}
            : std::nullopt;
    bool upscaled = false;
    try {
        upscaled = fsr4_upscaler->Upscale({
            .color = image(input_color, VK_IMAGE_ASPECT_COLOR_BIT),
            .depth = image(depth, VK_IMAGE_ASPECT_DEPTH_BIT),
            .motion_vectors = image(motion_vectors, VK_IMAGE_ASPECT_COLOR_BIT),
            .render_size = {render_size[0], render_size[1]},
            .upscale_size = extent,
            .max_render_size = max_render_size,
            .frame_time_ms = frame_time_ms,
            .reset = reset,
            .sharpen = sharpen,
            .sharpness = sharpness,
            .jitter = jitter,
            .motion_vector_scale = {motion_scale[0] * static_cast<f32>(render_size[0]),
                                    motion_scale[1] * static_cast<f32>(render_size[1])},
        });
    } catch (const vk::Exception& exception) {
        LOG_WARNING(Render_Vulkan, "AMD FSR 4 evaluation unavailable ({}); using FSR 3.1",
                    exception.what());
    }
    if (!upscaled || fsr4_upscaler->Failed()) {
        fsr4_unsupported = true;
        return false;
    }
    DrawUpscaledOutput(dst_framebuffer, fsr4_upscaler->OutputView(), fsr4_upscaler->OutputImage(),
                       fsr4_upscaler->PointSampler());
    static bool logged{};
    if (!logged) {
        logged = true;
        LOG_INFO(Render_Vulkan, "DLSS replaced by AMD FSR 4: {}x{} -> {}x{}", render_size[0],
                 render_size[1], extent.width, extent.height);
    }
    return true;
}

void TextureCacheRuntime::DrawUpscaledOutput(Framebuffer* dst_framebuffer, VkImageView view,
                                             VkImage output, VkSampler sampler) {
    // Draw the result into the DLSS output, which keeps its own format; the blit leaves the
    // upscaler's output image in SHADER_READ_ONLY_OPTIMAL, so return it to GENERAL for the next
    // frame.
    const VkExtent2D extent = dst_framebuffer->RenderArea();
    const Region2D region{
        .start = {0, 0},
        .end = {static_cast<s32>(extent.width), static_cast<s32>(extent.height)},
    };
    blit_image_helper.BlitColor(dst_framebuffer, view, output, sampler, region, region,
                                {extent.width, extent.height, 1});
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([output](vk::CommandBuffer cmdbuf) {
        const VkImageMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = output,
            .subresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, barrier);
    });
}

#ifdef EGLANTINE_STREAMLINE
StreamlineDlssUpscaler* TextureCacheRuntime::DlssUpscaler() {
    if (!UsesNvidiaDlss()) {
        return nullptr;
    }
    if (!dlss_upscaler) {
        // Off in the settings, or Streamline didn't start: it has said why.
        if (!Streamline::IsDlssAvailable()) {
            dlss_unsupported = true;
            return nullptr;
        }
        try {
            dlss_upscaler =
                std::make_unique<StreamlineDlssUpscaler>(device, scheduler, memory_allocator);
        } catch (const vk::Exception& exception) {
            LOG_WARNING(Render_Vulkan, "NVIDIA DLSS unavailable ({})", exception.what());
            dlss_unsupported = true;
            return nullptr;
        }
    }
    return dlss_upscaler.get();
}

namespace {
StreamlineDlssUpscaler::Image DlssImage(const Device& device,
                                        const Settings::ResolutionScalingInfo& resolution,
                                        ImageView& view, VkImageView handle,
                                        VkImageAspectFlags aspect) {
    const Image* const backing = view.BackingImage();
    const VkExtent2D extent =
        backing && backing->IsRescaled()
            ? VkExtent2D{resolution.ScaleUp(view.size.width), resolution.ScaleUp(view.size.height)}
            : VkExtent2D{view.size.width, view.size.height};
    return StreamlineDlssUpscaler::Image{
        .view = handle,
        .image = view.ImageHandle(),
        .format =
            MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, false, view.format).format,
        .range{
            .aspectMask = aspect,
            .baseMipLevel = static_cast<u32>(view.range.base.level),
            .levelCount = 1,
            .baseArrayLayer = static_cast<u32>(view.range.base.layer),
            .layerCount = 1,
        },
        .barrier_aspect = ImageAspectMask(view.format),
        .extent = extent,
    };
}
} // Anonymous namespace

bool TextureCacheRuntime::UpscaleWithDlss(Framebuffer* dst_framebuffer, ImageView& input_color,
                                          ImageView& motion_vectors, ImageView& depth,
                                          std::array<u32, 2> render_size,
                                          std::array<f32, 2> jitter,
                                          std::array<f32, 2> motion_scale, bool reset) {
    if (!DlssUpscaler()) {
        return false;
    }
    const auto image = [&](ImageView& view, VkImageView handle, VkImageAspectFlags aspect) {
        return DlssImage(device, resolution, view, handle, aspect);
    };
    const bool native_resolution =
        Settings::values.nvidia_dlss_quality.GetValue() == Settings::DlssQuality::Dlaa;
    const VkExtent2D extent = native_resolution ? VkExtent2D{render_size[0], render_size[1]}
                                                : dst_framebuffer->RenderArea();
    const auto color_img =
        image(input_color, input_color.Handle(Shader::TextureType::Color2D), VK_IMAGE_ASPECT_COLOR_BIT);
    const bool dynamic_region =
        color_img.extent.width > render_size[0] || color_img.extent.height > render_size[1];
    const std::optional<VkExtent2D> max_render_size =
        dynamic_region ? std::optional{VkExtent2D{
                             std::clamp(color_img.extent.width, render_size[0], extent.width),
                             std::clamp(color_img.extent.height, render_size[1], extent.height)}}
                       : std::nullopt;
    bool upscaled{};
    try {
        upscaled = dlss_upscaler->Upscale({
            .color = color_img,
            .depth = image(depth, depth.DepthView(), VK_IMAGE_ASPECT_DEPTH_BIT),
            .motion_vectors = image(motion_vectors,
                                    motion_vectors.Handle(Shader::TextureType::Color2D),
                                    VK_IMAGE_ASPECT_COLOR_BIT),
            .render_size = {render_size[0], render_size[1]},
            .upscale_size = extent,
            .max_render_size = max_render_size,
            .jitter = jitter,
            .motion_vector_scale = {motion_scale[0] * static_cast<f32>(render_size[0]),
                                    motion_scale[1] * static_cast<f32>(render_size[1])},
            .reset = reset,
        });
    } catch (const vk::Exception& exception) {
        LOG_WARNING(Render_Vulkan, "NVIDIA DLSS evaluation unavailable ({}); using FSR 3.1",
                    exception.what());
    }
    if (!upscaled) {
        dlss_unsupported = true;
        return false;
    }
    DrawUpscaledOutput(dst_framebuffer, dlss_upscaler->OutputView(), dlss_upscaler->OutputImage(),
                       native_resolution ? dlss_upscaler->LinearSampler()
                                         : dlss_upscaler->PointSampler());
    static bool logged{};
    if (!logged) {
        logged = true;
        LOG_INFO(Render_Vulkan, "DLSS replaced by NVIDIA DLSS: {}x{} -> {}x{}", render_size[0],
                 render_size[1], extent.width, extent.height);
    }
    return true;
}
#endif

#ifdef LICHII_METALFX
bool TextureCacheRuntime::UpscaleWithMetalFx(Framebuffer* dst_framebuffer, ImageView& input_color,
                                             ImageView& motion_vectors, ImageView& depth,
                                             std::array<u32, 2> render_size,
                                             std::array<f32, 2> jitter,
                                             std::array<f32, 2> motion_scale, bool reset) {
    if (!Settings::MetalFxTemporalChosen() || metalfx_unsupported) {
        return false;
    }
    if (!metalfx_upscaler) {
        try {
            metalfx_upscaler =
                std::make_unique<MetalFxTemporalUpscaler>(device, scheduler, memory_allocator);
        } catch (const vk::Exception& exception) {
            LOG_WARNING(Render_Vulkan, "MetalFX temporal unavailable ({}); using FSR 3.1",
                        exception.what());
            metalfx_unsupported = true;
            return false;
        }
    }
    const auto image = [&](ImageView& view, VkImageView handle, VkImageAspectFlags aspect) {
        return MetalFxTemporalUpscaler::Image{
            .view = handle,
            .image = view.ImageHandle(),
            .format = MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, false, view.format)
                          .format,
            .range{
                .aspectMask = aspect,
                .baseMipLevel = static_cast<u32>(view.range.base.level),
                .levelCount = 1,
                .baseArrayLayer = static_cast<u32>(view.range.base.layer),
                .layerCount = 1,
            },
            .barrier_aspect = ImageAspectMask(view.format),
        };
    };
    const VkExtent2D extent = dst_framebuffer->RenderArea();
    const bool upscaled = metalfx_upscaler->Upscale({
        .color = image(input_color, input_color.Handle(Shader::TextureType::Color2D),
                       VK_IMAGE_ASPECT_COLOR_BIT),
        .depth = image(depth, depth.DepthView(), VK_IMAGE_ASPECT_DEPTH_BIT),
        .motion_vectors = image(motion_vectors,
                                motion_vectors.Handle(Shader::TextureType::Color2D),
                                VK_IMAGE_ASPECT_COLOR_BIT),
        .render_size = {render_size[0], render_size[1]},
        .upscale_size = extent,
        .jitter = jitter,
        .motion_vector_scale = {motion_scale[0] * static_cast<f32>(render_size[0]),
                                motion_scale[1] * static_cast<f32>(render_size[1])},
        .reset = reset,
    });
    if (!upscaled) {
        LOG_WARNING(Render_Vulkan, "MetalFX temporal declined the evaluation; using FSR 3.1");
        metalfx_unsupported = true;
        return false;
    }
    DrawUpscaledOutput(dst_framebuffer, metalfx_upscaler->OutputView(),
                       metalfx_upscaler->OutputImage(), metalfx_upscaler->PointSampler());
    static bool logged{};
    if (!logged) {
        logged = true;
        LOG_INFO(Render_Vulkan, "DLSS replaced by MetalFX temporal: {}x{} -> {}x{}",
                 render_size[0], render_size[1], extent.width, extent.height);
    }
    return true;
}
#endif
bool TextureCacheRuntime::UpscaleWithCameraMotion(Framebuffer* dst_framebuffer,
                                                  ImageView& input_color, ImageView& depth,
                                                  std::array<u32, 2> render_size,
                                                  const std::array<f32, 16>& reprojection,
                                                  std::array<f32, 2> jitter, bool reset) {
#ifdef EGLANTINE_STREAMLINE
    const VkExtent2D render_extent{render_size[0], render_size[1]};
    const VkExtent2D upscale_extent = dst_framebuffer->RenderArea();
    if (upscale_extent.width < render_extent.width ||
        upscale_extent.height < render_extent.height ||
        input_color.Samples() != VK_SAMPLE_COUNT_1_BIT ||
        depth.Samples() != VK_SAMPLE_COUNT_1_BIT) {
        return false;
    }
    if (!DlssUpscaler()) {
        return false;
    }
    if (!camera_motion_pass) {
        camera_motion_pass = std::make_unique<CameraMotionPass>(
            device, scheduler, descriptor_pool, compute_pass_descriptor_queue, memory_allocator);
    }
    camera_motion_pass->Record(depth.DepthView(), render_extent, reprojection);
    const StreamlineDlssUpscaler::Image motion_vectors{
        .view = camera_motion_pass->View(),
        .image = camera_motion_pass->Image(),
        .format = CameraMotionPass::FORMAT,
        .range{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .barrier_aspect = VK_IMAGE_ASPECT_COLOR_BIT,
    };
    bool upscaled{};
    try {
        upscaled = dlss_upscaler->Upscale({
            .color = DlssImage(device, resolution, input_color,
                               input_color.Handle(Shader::TextureType::Color2D),
                               VK_IMAGE_ASPECT_COLOR_BIT),
            .depth = DlssImage(device, resolution, depth, depth.DepthView(),
                               VK_IMAGE_ASPECT_DEPTH_BIT),
            .motion_vectors = motion_vectors,
            .render_size = render_extent,
            .upscale_size = upscale_extent,
            .jitter = jitter,
            // The motion vectors are in render pixels.
            .motion_vector_scale = {1.0f, 1.0f},
            .reset = reset,
            .depth_inverted = false,
            .hdr = false,
        });
    } catch (const vk::Exception& exception) {
        LOG_WARNING(Render_Vulkan, "NVIDIA DLSS evaluation unavailable ({})", exception.what());
    }
    if (!upscaled) {
        dlss_unsupported = true;
        return false;
    }
    DrawUpscaledOutput(dst_framebuffer, dlss_upscaler->OutputView(), dlss_upscaler->OutputImage(),
                       dlss_upscaler->PointSampler());
    static bool logged{};
    if (!logged) {
        logged = true;
        LOG_INFO(Render_Vulkan, "Anti-aliasing replaced by NVIDIA DLSS: {}x{} -> {}x{}",
                 render_extent.width, render_extent.height, upscale_extent.width,
                 upscale_extent.height);
    }
    return true;
#else
    return false;
#endif
}

bool TextureCacheRuntime::TemporalUpscale(Framebuffer* dst_framebuffer, ImageView& dst,
                                          ImageView& input_color, ImageView& motion_vectors,
                                          ImageView* depth, std::array<u32, 2> render_size,
                                          std::array<f32, 2> jitter,
                                          std::array<f32, 2> motion_scale, bool use_history) {
    if (dst.Samples() != VK_SAMPLE_COUNT_1_BIT || ImageAspectMask(dst.format) != VK_IMAGE_ASPECT_COLOR_BIT) {
        return false;
    }
    const auto plain = [](const ImageView& view) {
        return view.Samples() == VK_SAMPLE_COUNT_1_BIT && view.range.base.level == 0 &&
               view.range.base.layer == 0;
    };
    // A view's size in host pixels, after resolution scaling.
    const auto host_extent = [&](const ImageView& view) -> std::optional<std::array<u32, 2>> {
        const Image* image = view.BackingImage();
        if (!image) {
            return std::nullopt;
        }
        return image->IsRescaled()
            ? std::array{resolution.ScaleUp(view.size.width), resolution.ScaleUp(view.size.height)}
            : std::array{view.size.width, view.size.height};
    };
    const auto fits = [&](const ImageView& view, std::array<u32, 2> requested) {
        const auto view_extent = host_extent(view);
        return view_extent && VideoCommon::DlssContract::FitsExtent(requested, *view_extent);
    };
    const VkExtent2D output_extent = dst_framebuffer->RenderArea();
    if (!plain(dst) || !plain(input_color) || !fits(input_color, render_size) ||
        !fits(dst, {output_extent.width, output_extent.height}) ||
        !VideoCommon::DlssContract::ValidColorFormats(input_color.format, dst.format, use_history) ||
        input_color.ImageHandle() == dst.ImageHandle() ||
        !VideoCommon::DlssContract::ValidGeometry(render_size,
                                                 {output_extent.width, output_extent.height},
                                                 jitter, motion_scale)) {
        return false;
    }
    // When the game renders only a region of the input (render_size smaller than the input), its
    // motion and depth targets may be of another size: they only have to hold that region.
    const bool whole_input = host_extent(input_color) == render_size;
    const auto same_size = [&](const ImageView& view) {
        return !whole_input ||
               (input_color.size.width == view.size.width && input_color.size.height == view.size.height);
    };
    if (use_history && (!plain(motion_vectors) || !fits(motion_vectors, render_size) ||
                        motion_vectors.format != PixelFormat::R16G16_FLOAT ||
                        !same_size(motion_vectors))) {
        return false;
    }
    if (use_history && depth && (!plain(*depth) || !fits(*depth, render_size) ||
                                depth->format != PixelFormat::D32_FLOAT || !same_size(*depth))) {
        return false;
    }
    if (use_history && depth && UpscaleWithFsr3(dst_framebuffer, input_color, motion_vectors, *depth, render_size,
                                 jitter, motion_scale)) {
        return true;
    }
    const VkFormat format =
        MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, false, dst.format).format;
    // The framebuffer's area is the output's size on the host (scaled with resolution scaling).
    const VkExtent2D extent = dst_framebuffer->RenderArea();
    // The history is kept at half-float precision: in the output's format (B10G11R11 here, 5-6
    // mantissa bits) a frame's small share of the blend rounded away, so the history never
    // converged on the detail the jittered frames add.
    constexpr VkFormat history_format = VK_FORMAT_R16G16B16A16_SFLOAT;
    bool reset = false;
    if (!temporal_history.images[0] || temporal_history.format != format ||
        temporal_history.extent.width != extent.width ||
        temporal_history.extent.height != extent.height) {
        if (temporal_history.images[0]) {
            scheduler.Finish(); // the old history may still be in use
        }
        const MemoryTagScope memory_tag{MemoryTag::Upscaler};
        for (size_t i = 0; i < temporal_history.images.size(); ++i) {
            temporal_history.images[i] = memory_allocator.CreateImage(VkImageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .imageType = VK_IMAGE_TYPE_2D,
                .format = history_format,
                .extent = {extent.width, extent.height, 1},
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            });
            temporal_history.views[i] = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .image = *temporal_history.images[i],
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = history_format,
                .components{},
                .subresourceRange{
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            });
        }
        temporal_history.current = 0;
        temporal_history.format = format;
        temporal_history.extent = extent;
        reset = true;
        LOG_INFO(Render_Vulkan, "Temporal upscale into {}x{} (format {})", extent.width, extent.height,
                 static_cast<u32>(format));
    }

    // The shader reads the previous result from one history image and stores this one to the
    // other: the game may draw into its DLSS output afterwards (post-processing), which must not
    // feed back. Both stay in the GENERAL layout, which the blit helper's descriptors use.
    const size_t read_index = temporal_history.current;
    const size_t write_index = read_index ^ 1;
    const VkImage history = *temporal_history.images[read_index];
    const VkImage next_history = *temporal_history.images[write_index];
    const auto barrier = [](VkImage image, VkAccessFlags src_access, VkAccessFlags dst_access,
                            VkImageLayout old_layout) {
        return VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = src_access,
            .dstAccessMask = dst_access,
            .oldLayout = old_layout,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
    };
    if (reset) {
        // New history images have no contents yet; the first frame does not use what it reads.
        scheduler.RequestOutsideRenderPassOperationContext();
        scheduler.Record([history, next_history, barrier](vk::CommandBuffer cmdbuf) {
            constexpr VkAccessFlags access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            const std::array barriers{
                barrier(history, 0, access, VK_IMAGE_LAYOUT_UNDEFINED),
                barrier(next_history, 0, access, VK_IMAGE_LAYOUT_UNDEFINED),
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, {}, {}, barriers);
        });
    }
    const Region2D dst_region{
        .start = {0, 0},
        .end = {static_cast<s32>(extent.width), static_cast<s32>(extent.height)},
    };
    // Each frame weighs at most 0.1 against the history (scaled by how close its samples land to
    // the pixel; the neighbourhood clip limits what older samples contribute). Still pixels use a
    // sample kernel a third as wide as for moving ones (the jittered frames fill in the detail) and
    // a clip of 1.5 deviations instead of 1.
    // EDEN_NX2_TEMPORAL_WEIGHT / _KERNEL / _CLIP / _CLIP_MOTION / _KERNEL_MOTION
    // override them (all 1 = the previous stand-in's filter).
    const auto tuned = [](const char* name, f32 fallback) {
        const char* value = std::getenv(name);
        return value ? static_cast<f32>(std::strtod(value, nullptr)) : fallback;
    };
    static const f32 current_weight = tuned("EDEN_NX2_TEMPORAL_WEIGHT", 0.1f);
    static const f32 kernel_scale = tuned("EDEN_NX2_TEMPORAL_KERNEL", 3.0f);
    static const f32 clip_width = tuned("EDEN_NX2_TEMPORAL_CLIP", 1.5f);
    static const f32 clip_motion = tuned("EDEN_NX2_TEMPORAL_CLIP_MOTION", 1.0f);
    static const f32 kernel_motion = tuned("EDEN_NX2_TEMPORAL_KERNEL_MOTION", 1.0f);
    blit_image_helper.TemporalUpscale(dst_framebuffer, input_color, motion_vectors,
                                      *temporal_history.views[read_index],
                                      *temporal_history.views[write_index], dst_region, jitter,
                                      motion_scale, !use_history ? -1.0f : reset ? 1.0f : current_weight, kernel_scale,
                                      clip_width, clip_motion, kernel_motion, render_size);
    // RCAS on the output, set by the FSR sharpness slider (stops as FSR takes them, 0% = stored 200
    // turns it off). It reads the unsharpened result just stored, which stays the history.
    // EDEN_NX2_TEMPORAL_SHARPEN=<stops>|off overrides the slider.
    static const char* const sharpen_override = std::getenv("EDEN_NX2_TEMPORAL_SHARPEN");
    const int slider = Settings::values.fsr_sharpening_slider.GetValue();
    bool sharpen = slider < 200;
    f32 sharpness = static_cast<f32>(slider) / 100.0f;
    if (sharpen_override) {
        sharpen = std::string_view{sharpen_override} != "off";
        sharpness = static_cast<f32>(std::strtod(sharpen_override, nullptr));
    }
    const VkImage output = dst.ImageHandle();
    const u32 output_level = static_cast<u32>(dst.range.base.level);
    const u32 output_layer = static_cast<u32>(dst.range.base.layer);
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([history, next_history, barrier, output, output_level, output_layer,
                      sharpen](vk::CommandBuffer cmdbuf) {
        // The next frame reads what was stored now and stores over what was read now; the
        // sharpening pass reads what was stored now and draws over the output again.
        const std::array barriers{
            barrier(next_history, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_GENERAL),
            barrier(history, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_GENERAL),
            VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = output,
                .subresourceRange{
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = output_level,
                    .levelCount = 1,
                    .baseArrayLayer = output_layer,
                    .layerCount = 1,
                },
            },
        };
        constexpr VkPipelineStageFlags stages =
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        cmdbuf.PipelineBarrier(stages, stages, 0, {}, {},
                               vk::Span<VkImageMemoryBarrier>(barriers.data(),
                                                              sharpen ? barriers.size() : 2));
    });
    if (sharpen) {
        blit_image_helper.TemporalSharpen(dst_framebuffer, *temporal_history.views[write_index],
                                          dst_region, sharpness);
    }
    temporal_history.current = write_index;
    return true;
}

void TextureCacheRuntime::BlitImage(Framebuffer* dst_framebuffer, ImageView& dst, ImageView& src,
                                    const Region2D& dst_region, const Region2D& src_region,
                                    Tegra::Engines::Fermi2D::Filter filter,
                                    Tegra::Engines::Fermi2D::Operation operation) {
    const VkImageAspectFlags aspect_mask = ImageAspectMask(src.format);
    const bool is_dst_msaa = dst.Samples() != VK_SAMPLE_COUNT_1_BIT;
    const bool is_src_msaa = src.Samples() != VK_SAMPLE_COUNT_1_BIT;
    if (aspect_mask != ImageAspectMask(dst.format)) {
        UNIMPLEMENTED_MSG("Incompatible blit from format {} to {}", src.format, dst.format);
        return;
    }
    if (aspect_mask == VK_IMAGE_ASPECT_COLOR_BIT && !is_src_msaa && !is_dst_msaa) {
        blit_image_helper.BlitColor(dst_framebuffer, src, dst_region, src_region, filter,
                                    operation);
        return;
    }
    ASSERT(src.format == dst.format);
    if (is_src_msaa && !is_dst_msaa &&
        (aspect_mask & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) != 0) {
        if ((aspect_mask & VK_IMAGE_ASPECT_DEPTH_BIT) == 0) {
            UNIMPLEMENTED_MSG("Stencil-only MSAA resolve is not supported");
            return;
        }
        blit_image_helper.ResolveDepthStencil(dst_framebuffer, src, dst_region, src_region);
        return;
    }
    if (aspect_mask == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
        const auto format = src.format;
        const auto can_blit_depth_stencil = [this, format] {
            switch (format) {
            case VideoCore::Surface::PixelFormat::D24_UNORM_S8_UINT:
            case VideoCore::Surface::PixelFormat::S8_UINT_D24_UNORM:
                return device.IsBlitDepth24Stencil8Supported();
            case VideoCore::Surface::PixelFormat::D32_FLOAT_S8_UINT:
                return device.IsBlitDepth32Stencil8Supported();
            default:
                UNREACHABLE();
            }
        }();
        // Use shader-based depth/stencil blits if hardware doesn't support the format
        // Note: MSAA resolves (MSAA->single) use vkCmdResolveImage which works fine
        if (!can_blit_depth_stencil) {
            UNIMPLEMENTED_IF(is_src_msaa || is_dst_msaa);
            blit_image_helper.BlitDepthStencil(dst_framebuffer, src, dst_region, src_region,
                                               filter, operation);
            return;
        }
    }
    ASSERT(!(is_dst_msaa && !is_src_msaa));
    ASSERT(operation == Fermi2D::Operation::SrcCopy);

    const bool is_msaa_to_msaa = is_src_msaa && is_dst_msaa;
    if (is_msaa_to_msaa && aspect_mask == VK_IMAGE_ASPECT_COLOR_BIT) {
        blit_image_helper.BlitColorMSAA(dst_framebuffer, src, dst_region, src_region);
        return;
    }
    if (is_msaa_to_msaa) {
        blit_image_helper.BlitDepthStencilMSAA(dst_framebuffer, src, dst_region, src_region);
        return;
    }

    const bool is_resolve = is_src_msaa && !is_dst_msaa;
    if (is_resolve && !HaveSameExtent(dst_region, src_region)) {
        blit_image_helper.BlitColorMSAA(dst_framebuffer, src, dst_region, src_region);
        return;
    }

    const VkImage dst_image = dst.ImageHandle();
    const VkImage src_image = src.ImageHandle();
    const VkImageSubresourceLayers dst_layers = MakeSubresourceLayers(&dst);
    const VkImageSubresourceLayers src_layers = MakeSubresourceLayers(&src);
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([filter, dst_region, src_region, dst_image, src_image, dst_layers, src_layers,
                      aspect_mask, is_resolve](vk::CommandBuffer cmdbuf) {
        const std::array read_barriers{
            VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT |
                                 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = src_image,
                .subresourceRange{
                    .aspectMask = aspect_mask,
                    .baseMipLevel = 0,
                    .levelCount = VK_REMAINING_MIP_LEVELS,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            },
            VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT |
                                 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = dst_image,
                .subresourceRange{
                    .aspectMask = aspect_mask,
                    .baseMipLevel = 0,
                    .levelCount = VK_REMAINING_MIP_LEVELS,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            },
        };
        VkImageMemoryBarrier write_barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = dst_image,
            .subresourceRange{
                .aspectMask = aspect_mask,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        };
        cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       0, nullptr, nullptr, read_barriers);
        if (is_resolve) {
            cmdbuf.ResolveImage(src_image, VK_IMAGE_LAYOUT_GENERAL, dst_image,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                MakeImageResolve(dst_region, src_region, dst_layers, src_layers));
        } else {
            const bool is_linear = filter == Fermi2D::Filter::Bilinear;
            const VkFilter vk_filter = is_linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
            cmdbuf.BlitImage(
                src_image, VK_IMAGE_LAYOUT_GENERAL, dst_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                MakeImageBlit(dst_region, src_region, dst_layers, src_layers), vk_filter);
        }
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                               vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER, 0, write_barrier);
    });
}

void TextureCacheRuntime::ConvertImage(Framebuffer* dst, ImageView& dst_view, ImageView& src_view) {
    if (!dst->RenderPass()) {
        return;
    }

    switch (dst_view.format) {
    case PixelFormat::R16_UNORM:
        if (src_view.format == PixelFormat::D16_UNORM) {
            return blit_image_helper.ConvertD16ToR16(dst, src_view);
        }
        break;
    case PixelFormat::A8B8G8R8_SRGB:
    case PixelFormat::B8G8R8A8_SRGB:
    case PixelFormat::B8G8R8A8_UNORM:
        if (src_view.format == PixelFormat::D32_FLOAT) {
            return blit_image_helper.ConvertD32FToABGR8(dst, src_view);
        }
        break;
    case PixelFormat::A8B8G8R8_UNORM:
        if (src_view.format == PixelFormat::X8_D24_UNORM) {
            return blit_image_helper.ConvertX8D24ToABGR8(dst, src_view);
        }
        if (src_view.format == PixelFormat::S8_UINT_D24_UNORM) {
            return blit_image_helper.ConvertD24S8ToABGR8(dst, src_view);
        }
        if (src_view.format == PixelFormat::D24_UNORM_S8_UINT) {
            return blit_image_helper.ConvertS8D24ToABGR8(dst, src_view);
        }
        if (src_view.format == PixelFormat::D32_FLOAT) {
            return blit_image_helper.ConvertD32FToABGR8(dst, src_view);
        }
        break;
    case PixelFormat::R32_FLOAT:
        if (src_view.format == PixelFormat::D32_FLOAT) {
            return blit_image_helper.ConvertD32ToR32(dst, src_view);
        }
        break;
    case PixelFormat::D16_UNORM:
        if (src_view.format == PixelFormat::R16_UNORM) {
            return blit_image_helper.ConvertR16ToD16(dst, src_view);
        }
        break;
    case PixelFormat::X8_D24_UNORM:
        if (src_view.format == PixelFormat::A8B8G8R8_UNORM) {
            return blit_image_helper.ConvertABGR8ToX8D24(dst, src_view);
        }
        break;
    case PixelFormat::S8_UINT_D24_UNORM:
        if (src_view.format == PixelFormat::A8B8G8R8_UNORM ||
            src_view.format == PixelFormat::B8G8R8A8_UNORM) {
            return blit_image_helper.ConvertABGR8ToD24S8(dst, src_view);
        }
        break;
    case PixelFormat::D32_FLOAT:
        if (src_view.format == PixelFormat::A8B8G8R8_UNORM ||
            src_view.format == PixelFormat::B8G8R8A8_UNORM ||
            src_view.format == PixelFormat::A8B8G8R8_SRGB ||
            src_view.format == PixelFormat::B8G8R8A8_SRGB) {
            return blit_image_helper.ConvertABGR8ToD32F(dst, src_view);
        }
        if (src_view.format == PixelFormat::R32_FLOAT) {
            return blit_image_helper.ConvertR32ToD32(dst, src_view);
        }
        break;
    case PixelFormat::D24_UNORM_S8_UINT:
        if (src_view.format == PixelFormat::A8B8G8R8_UNORM ||
            src_view.format == PixelFormat::B8G8R8A8_UNORM ||
            src_view.format == PixelFormat::A8B8G8R8_SRGB ||
            src_view.format == PixelFormat::B8G8R8A8_SRGB) {
            return blit_image_helper.ConvertABGR8ToD24S8(dst, src_view);
        }
        break;
    default:
        break;
    }

    if (src_view.format == PixelFormat::D32_FLOAT &&
        VideoCore::Surface::GetFormatType(dst_view.format) == SurfaceType::ColorTexture &&
        (dst_view.format == PixelFormat::B5G6R5_UNORM ||
         Settings::values.fix_bloom_effects.GetValue())) {
        const Region2D region{
            .start = {0, 0},
            .end = {static_cast<s32>(dst->RenderArea().width),
                    static_cast<s32>(dst->RenderArea().height)},
        };
        return blit_image_helper.BlitColor(dst, src_view, region, region,
                                        Tegra::Engines::Fermi2D::Filter::Point,
                                        Tegra::Engines::Fermi2D::Operation::SrcCopy);
    }

    LOG_DEBUG(Render_Vulkan, "Unimplemented texture conversion from {} to {} format type", src_view.format, dst_view.format);
}

VkFormat TextureCacheRuntime::GetSupportedFormat(VkFormat requested_format,
                                                VkFormatFeatureFlags required_features) const {
    if (requested_format == VK_FORMAT_A8B8G8R8_SRGB_PACK32 &&
        (required_features & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
        // Force valid depth format when sRGB requested in depth context
        return VK_FORMAT_D24_UNORM_S8_UINT;
    }
    return requested_format;
}

// Helper functions for format compatibility checks
bool TextureCacheRuntime::IsFormatDitherable(PixelFormat format) {
    switch (format) {
    case PixelFormat::B8G8R8A8_UNORM:
    case PixelFormat::A8B8G8R8_UNORM:
    case PixelFormat::B8G8R8A8_SRGB:
    case PixelFormat::A8B8G8R8_SRGB:
        return true;
    default:
        return false;
    }
}

bool TextureCacheRuntime::IsFormatScalable(PixelFormat format) {
    switch (format) {
    case PixelFormat::B8G8R8A8_UNORM:
    case PixelFormat::A8B8G8R8_UNORM:
    case PixelFormat::R16G16B16A16_FLOAT:
    case PixelFormat::R32G32B32A32_FLOAT:
        return true;
    default:
        return false;
    }
}

void TextureCacheRuntime::CopyImage(Image& dst, Image& src,
                                    std::span<const VideoCommon::ImageCopy> copies) {
    if (ENABLE_MSAA_RESOLVE_CONSUME) {
        InvalidateResolveShadow(dst.Handle());
    }
    // As per the size-compatible formats section of vulkan, copy manually via ReinterpretImage
    // these images that aren't size-compatible
    if (BytesPerBlock(src.info.format) != BytesPerBlock(dst.info.format)) {
#ifdef _WIN32
        // On Windows, linear images cause device loss when used in image copies.
        // Tested with TitleID: 0x010067300059A00 (Mario + Rabbids Kingdom Battle)
        if (src.info.type == ImageType::Linear || dst.info.type == ImageType::Linear) {
            return;
        }
#endif
        auto oneCopy = VideoCommon::ImageCopy{
            .src_offset = VideoCommon::Offset3D(0, 0, 0),
            .dst_offset = VideoCommon::Offset3D(0, 0, 0),
            .extent = dst.info.size
        };
        return ReinterpretImage(dst, src, std::span{&oneCopy, 1});
    }
    boost::container::small_vector<VkImageCopy, 16> vk_copies(copies.size());
    const VkImageAspectFlags aspect_mask = dst.AspectMask();
    ASSERT(aspect_mask == src.AspectMask());

    std::ranges::transform(copies, vk_copies.begin(), [aspect_mask](const auto& copy) {
        return MakeImageCopy(copy, aspect_mask);
    });
    const VkImage dst_image = dst.Handle();
    const VkImage src_image = src.Handle();
    scheduler.RequestOutsideRenderPassKeepClear(std::array{dst_image, src_image});
    scheduler.Record([dst_image, src_image, aspect_mask, vk_copies](vk::CommandBuffer cmdbuf) {
        RangedBarrierRange dst_range;
        RangedBarrierRange src_range;
        for (const VkImageCopy& copy : vk_copies) {
            dst_range.AddLayers(copy.dstSubresource);
            src_range.AddLayers(copy.srcSubresource);
        }
        const std::array pre_barriers{
            VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = src_image,
                .subresourceRange = src_range.SubresourceRange(aspect_mask),
            },
            VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = dst_image,
                .subresourceRange = dst_range.SubresourceRange(aspect_mask),
            },
        };
        const std::array post_barriers{
            VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = 0,
                .dstAccessMask = 0,
                .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = src_image,
                .subresourceRange = src_range.SubresourceRange(aspect_mask),
            },
            VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                 VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = dst_image,
                .subresourceRange = dst_range.SubresourceRange(aspect_mask),
            },
        };
        cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, nullptr, nullptr, pre_barriers);
        cmdbuf.CopyImage(src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst_image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, vk_copies);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                               vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER, 0, nullptr, nullptr,
                               post_barriers);
    });
}

void TextureCacheRuntime::CopyImageMSAA(Image& dst, Image& src,
                                        std::span<const VideoCommon::ImageCopy> copies) {
    const bool msaa_to_non_msaa = src.info.num_samples > 1 && dst.info.num_samples == 1;
    const u32 num_samples = msaa_to_non_msaa ? src.info.num_samples : dst.info.num_samples;
    if (ENABLE_MSAA_RESOLVE_CONSUME && msaa_to_non_msaa && copies.size() == 1 &&
        src.info.format == dst.info.format) {
        const VideoCommon::ImageCopy& copy = copies.front();
        const ResolveShadow* const shadow = GetValidResolveShadow(src.Handle());
        if (shadow != nullptr && shadow->aspect_mask == dst.AspectMask() &&
            copy.src_offset.x == 0 && copy.src_offset.y == 0 &&
            copy.src_subresource.base_level == 0 &&
            static_cast<u32>(copy.extent.width) <= shadow->extent.width &&
            static_cast<u32>(copy.extent.height) <= shadow->extent.height &&
            static_cast<u32>(copy.src_subresource.base_layer + copy.src_subresource.num_layers) <=
                shadow->layers) {
            const VkImageAspectFlags aspect_mask = shadow->aspect_mask;
            VkPipelineStageFlags attachment_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            VkAccessFlags attachment_write = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            VkAccessFlags attachment_read_write =
                VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            if ((aspect_mask & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) != 0) {
                attachment_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
                attachment_write = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                attachment_read_write = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            }
            const VkImage shadow_image = *shadow->image;
            const VkImage dst_image = dst.Handle();
            const VkImageCopy region{
                .srcSubresource{
                    .aspectMask = aspect_mask,
                    .mipLevel = 0,
                    .baseArrayLayer = static_cast<u32>(copy.src_subresource.base_layer),
                    .layerCount = static_cast<u32>(copy.src_subresource.num_layers),
                },
                .srcOffset = {0, 0, 0},
                .dstSubresource{
                    .aspectMask = aspect_mask,
                    .mipLevel = static_cast<u32>(copy.dst_subresource.base_level),
                    .baseArrayLayer = static_cast<u32>(copy.dst_subresource.base_layer),
                    .layerCount = static_cast<u32>(copy.dst_subresource.num_layers),
                },
                .dstOffset = {copy.dst_offset.x, copy.dst_offset.y, copy.dst_offset.z},
                .extent = {copy.extent.width, copy.extent.height, 1},
            };
            scheduler.RequestOutsideRenderPassOperationContext();
            scheduler.Record([shadow_image, dst_image, region, aspect_mask, attachment_stage,
                              attachment_write,
                              attachment_read_write](vk::CommandBuffer cmdbuf) {
                const std::array pre_barriers{
                    VkImageMemoryBarrier{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = attachment_write,
                        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = shadow_image,
                        .subresourceRange{aspect_mask, 0, VK_REMAINING_MIP_LEVELS, 0,
                                          VK_REMAINING_ARRAY_LAYERS},
                    },
                    VkImageMemoryBarrier{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | attachment_write |
                                         VK_ACCESS_TRANSFER_WRITE_BIT,
                        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = dst_image,
                        .subresourceRange{aspect_mask, 0, VK_REMAINING_MIP_LEVELS, 0,
                                          VK_REMAINING_ARRAY_LAYERS},
                    },
                };
                const std::array post_barriers{
                    VkImageMemoryBarrier{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = 0,
                        .dstAccessMask = 0,
                        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = shadow_image,
                        .subresourceRange{aspect_mask, 0, VK_REMAINING_MIP_LEVELS, 0,
                                          VK_REMAINING_ARRAY_LAYERS},
                    },
                    VkImageMemoryBarrier{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | attachment_read_write |
                                         VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = dst_image,
                        .subresourceRange{aspect_mask, 0, VK_REMAINING_MIP_LEVELS, 0,
                                          VK_REMAINING_ARRAY_LAYERS},
                    },
                };
                cmdbuf.PipelineBarrier(attachment_stage | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, nullptr, nullptr,
                                       pre_barriers);
                cmdbuf.CopyImage(shadow_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst_image,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region);
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, nullptr, nullptr,
                                       post_barriers);
            });
            return;
        }
    }
    const VkImageAspectFlags dst_aspect_mask = dst.AspectMask();
    if ((dst_aspect_mask & VK_IMAGE_ASPECT_DEPTH_BIT) != 0) {
        const bool copies_stencil = (dst_aspect_mask & VK_IMAGE_ASPECT_STENCIL_BIT) != 0 &&
                                    device.IsExtShaderStencilExportSupported();
        blit_image_helper.CopyMSAADepth(render_pass_cache, dst.Handle(), dst.info.format,
                                        src.Handle(), src.info.format, num_samples, copies,
                                        copies_stencil, msaa_to_non_msaa);
        return;
    }
    if ((dst_aspect_mask & VK_IMAGE_ASPECT_COLOR_BIT) == 0) {
        UNIMPLEMENTED_MSG("Copying images with different samples is not supported.");
        return;
    }
    blit_image_helper.CopyMSAA(render_pass_cache, dst.Handle(), dst.info.format, src.Handle(),
                               src.info.format, num_samples, copies, msaa_to_non_msaa);
}

u64 TextureCacheRuntime::GetDeviceLocalMemory() const {
    return device.GetDeviceLocalMemory();
}

u64 TextureCacheRuntime::GetDeviceMemoryUsage() const {
    return device.GetDeviceMemoryUsage();
}

u64 TextureCacheRuntime::CurrentTick() const noexcept {
    return scheduler.CurrentTick();
}

bool TextureCacheRuntime::IsFree(u64 tick) const noexcept {
    return scheduler.IsFree(tick);
}

u64 TextureCacheRuntime::GetDeviceMemoryBudget() const {
    return device.GetDeviceMemoryBudget();
}

u64 TextureCacheRuntime::GetDeviceMemorySlack() const {
    return device.GetDeviceMemorySlack();
}

u32 TextureCacheRuntime::GetMaxScaledExtent() const noexcept {
    return (std::min)({device.GetMaxImageDimension2D(), device.GetMaxFramebufferWidth(),
                       device.GetMaxFramebufferHeight()});
}

bool TextureCacheRuntime::CanDownloadMsaa(const VideoCommon::ImageInfo& info) const {
    const VkImageAspectFlags aspect_mask = ImageAspectMask(info.format);
    if ((aspect_mask & VK_IMAGE_ASPECT_COLOR_BIT) != 0) {
        return true;
    }
    if ((aspect_mask & VK_IMAGE_ASPECT_DEPTH_BIT) == 0) {
        return false;
    }
    if ((aspect_mask & VK_IMAGE_ASPECT_STENCIL_BIT) != 0) {
        return device.IsExtShaderStencilExportSupported();
    }
    return true;
}

bool TextureCacheRuntime::CanReportMemoryUsage() const {
    return device.CanReportMemoryUsage();
}

void TextureCacheRuntime::LogMemoryStatistics() const {
    Vulkan::LogMemoryStatistics(device);
}

bool TextureCacheRuntime::UseOpaqueAstcFormat() const {
    // EDEN_NX2_ASTC_BC1=0 keeps every re-compressed ASTC image in BC3.
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_ASTC_BC1");
        return value == nullptr || value[0] != '0';
    }();
    return enabled && !device.IsOptimalAstcSupported() &&
           Settings::GetAstcRecompression() == Settings::AstcRecompression::Bc3;
}

void TextureCacheRuntime::FlushDeferredClear() {
    scheduler.FlushDeferredClear();
}

bool TextureCacheRuntime::KeepsDeferredClearAcrossCopies() const {
    return Scheduler::KeepsDeferredClear();
}

VkImage TextureCacheRuntime::AcquireMsaaScratchImage(const VkImageCreateInfo& image_ci) {
    const MsaaScratchKey key{
        .format = image_ci.format,
        .type = image_ci.imageType,
        .width = image_ci.extent.width,
        .height = image_ci.extent.height,
        .depth = image_ci.extent.depth,
        .levels = image_ci.mipLevels,
        .layers = image_ci.arrayLayers,
        .usage = image_ci.usage,
        .flags = image_ci.flags,
    };
    for (MsaaScratchImage& scratch : msaa_scratch_images) {
        if (scratch.key != key || !scheduler.IsFree(scratch.tick)) {
            continue;
        }
        scratch.tick = (std::numeric_limits<u64>::max)();
        scratch.unused_frames = 0;
        return *scratch.image;
    }
    const MemoryTagScope memory_tag{MemoryTag::TextureHelper};
    MsaaScratchImage scratch{
        .key = key,
        .image = memory_allocator.CreateImage(image_ci),
        .tick = (std::numeric_limits<u64>::max)(),
        .unused_frames = 0,
    };
    const VkImage handle = *scratch.image;
    msaa_scratch_images.push_back(std::move(scratch));
    return handle;
}

void TextureCacheRuntime::ReleaseMsaaScratchImage(VkImage image) {
    for (MsaaScratchImage& scratch : msaa_scratch_images) {
        if (*scratch.image == image) {
            scratch.tick = scheduler.CurrentTick();
            return;
        }
    }
}

namespace {
bool PlainGpuUnswizzleEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_GPU_UNSWIZZLE_2D");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}
bool PlainGpuUnswizzleVerify() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_GPU_UNSWIZZLE_VERIFY");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}
bool ImageAllocationReuseEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_IMAGE_ALLOCATION_REUSE");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}
}

bool TextureCacheRuntime::CanGpuUnswizzle(const Image& image) const {
    constexpr auto unsupported = ImageFlagBits::Converted | ImageFlagBits::AcceleratedUpload |
                                 ImageFlagBits::AsynchronousDecode | ImageFlagBits::IsDecoding;
    if (!PlainGpuUnswizzleEnabled() || plain_unswizzle_failed || !image.info.native_tiling ||
        resolution.up_scale != 1 || resolution.down_shift != 0 || image.HasScaled() ||
        !VideoCommon::IsAttachmentBackingUsable(image) || True(image.flags & unsupported) ||
        image.AspectMask() == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
        return false;
    }
    const auto params = VideoCommon::MakePlainUnswizzleParameters(image.info);
    return params && params->word_count >= 65536 &&
           u64{params->source_words} * 4 <= device.GetMaxStorageBufferRange() &&
           u64{params->word_count} * 4 <= device.GetMaxStorageBufferRange() &&
           Common::DivCeil(params->word_count, 256U) <= device.GetMaxComputeWorkGroupCount()[0];
}

VkBuffer TextureCacheRuntime::DispatchPlainUnswizzle(
    const VideoCommon::PlainUnswizzleParameters& params, const StagingBufferRef& source, bool guarded) {
    const u64 output_bytes = u64{params.word_count} * 4;
    const VkBuffer output = GetTemporaryBuffer(output_bytes + (guarded ? 64 : 0));
    if (guarded) {
        scheduler.RequestOutsideRenderPassOperationContext();
        scheduler.Record([output, output_bytes](vk::CommandBuffer cmdbuf) {
            const VkMemoryBarrier before{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT, 0, before);
            cmdbuf.FillBuffer(output, output_bytes, 64, 0xcdcdcdcd);
        });
    }
    plain_unswizzle_pass->Dispatch(source.buffer, source.offset, output, params, guarded ? 64 : 0);
    return output;
}

StagingBufferRef TextureCacheRuntime::QueuePlainUnswizzleReadback(
    const VideoCommon::PlainUnswizzleParameters& params, VkBuffer output, bool deferred) {
    const size_t bytes = size_t{params.word_count} * 4;
    const auto download = DownloadStagingBuffer(bytes + 64, deferred);
    scheduler.Record([output, download, bytes](vk::CommandBuffer cmdbuf) {
        const VkMemoryBarrier before{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0, before);
        cmdbuf.CopyBuffer(output, download.buffer, VkBufferCopy{0, download.offset, bytes + 64});
        const VkBufferMemoryBarrier after{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = download.buffer,
            .offset = download.offset,
            .size = bytes + 64,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                              0, nullptr, after, nullptr);
    });
    return download;
}

bool TextureCacheRuntime::CheckPlainUnswizzle(const ImageInfo& info,
                                             const VideoCommon::PlainUnswizzleParameters& params,
                                             const StagingBufferRef& source, VkBuffer output) {
    const size_t bytes = size_t{params.word_count} * 4;
    std::vector<u8> expected(bytes);
    const auto swizzle = VideoCommon::FullUploadSwizzles(info).front();
    Tegra::Texture::UnswizzleTexture(expected, source.mapped_span, BytesPerBlock(info.format),
                                     info.size.width, info.size.height, 1, swizzle.block.height,
                                     swizzle.block.depth, VideoCommon::CalculateLevelStrideAlignment(info, 0),
                                     info.native_tiling);
    const auto download = QueuePlainUnswizzleReadback(params, output, false);
    Finish();
    const bool equal = std::equal(expected.begin(), expected.end(), download.mapped_span.begin());
    const auto guard = download.mapped_span.subspan(bytes, 64);
    const bool intact = std::all_of(guard.begin(), guard.end(), [](u8 value) { return value == 0xcd; });
    if (!equal || !intact) {
        LOG_ERROR(Render_Vulkan, "Plain GPU unswizzle verification failed: {}x{} format {} equal={} guard={}",
                  info.size.width, info.size.height, static_cast<u32>(info.format), equal, intact);
    }
    return equal && intact;
}

bool TextureCacheRuntime::TestPlainUnswizzle() {
    u32 passed = 0;
    constexpr std::array widths{4U, 17U, 65U, 81U, 97U, 128U, 257U, 960U};
    constexpr std::array heights{1U, 7U, 8U, 9U, 33U, 65U, 97U, 540U};
    for (const auto format : {PixelFormat::A8B8G8R8_UNORM, PixelFormat::R16G16B16A16_FLOAT,
                              PixelFormat::R8_UNORM, PixelFormat::D32_FLOAT}) {
        for (u32 index = 0; index < widths.size(); ++index) {
            ImageInfo info;
            info.type = ImageType::e2D;
            info.format = format;
            info.size = {widths[index], heights[index], 1};
            info.resources = {.levels = 1, .layers = 1};
            info.block = {0, index % 6, 0};
            info.tile_width_spacing = index % 3;
            info.native_tiling = (index % 2) != 0;
            const auto params = VideoCommon::MakePlainUnswizzleParameters(info);
            if (!params) {
                continue;
            }
            auto input = UploadStagingBuffer(size_t{params->source_words} * 4);
            for (size_t i = 0; i < size_t{params->source_words} * 4; ++i) {
                input.mapped_span[i] = static_cast<u8>((i * 71) ^ (i >> 7) ^ index);
            }
            const VkBuffer output = DispatchPlainUnswizzle(*params, input, true);
            if (!CheckPlainUnswizzle(info, *params, input, output)) {
                return false;
            }
            ++passed;
        }
    }
    struct PendingCheck {
        StagingBufferRef download;
        std::vector<u8> expected;
    };
    std::vector<PendingCheck> pending;
    for (u32 index = 0; index < 4; ++index) {
        ImageInfo info;
        info.type = ImageType::e2D;
        info.format = PixelFormat::A8B8G8R8_UNORM;
        info.size = {97, 65, 1};
        info.resources = {.levels = 1, .layers = 1};
        info.block = {0, index, 0};
        info.native_tiling = (index % 2) != 0;
        const auto params = VideoCommon::MakePlainUnswizzleParameters(info).value();
        auto input = UploadStagingBuffer(size_t{params.source_words} * 4);
        for (size_t i = 0; i < size_t{params.source_words} * 4; ++i) {
            input.mapped_span[i] = static_cast<u8>((i * 83) ^ (i >> 9) ^ (index * 37));
        }
        std::vector<u8> expected(size_t{params.word_count} * 4 + 64, 0xcd);
        const auto swizzle = VideoCommon::FullUploadSwizzles(info).front();
        Tegra::Texture::UnswizzleTexture(expected, input.mapped_span, 4, 97, 65, 1,
                                         swizzle.block.height, 0,
                                         VideoCommon::CalculateLevelStrideAlignment(info, 0),
                                         info.native_tiling);
        const VkBuffer output = DispatchPlainUnswizzle(params, input, true);
        pending.push_back({QueuePlainUnswizzleReadback(params, output, true), std::move(expected)});
    }
    Finish();
    bool batch_valid = true;
    for (auto& check : pending) {
        batch_valid &= std::equal(check.expected.begin(), check.expected.end(),
                                  check.download.mapped_span.begin());
        FreeDeferredStagingBuffer(check.download);
    }
    if (!batch_valid) {
        LOG_ERROR(Render_Vulkan, "Plain GPU unswizzle queued scratch-reuse verification failed");
        return false;
    }
    passed += 4;
    LOG_INFO(Render_Vulkan, "Plain GPU unswizzle self-test: {} byte-exact guarded cases passed (4 queued)", passed);
    return passed != 0;
}

bool TextureCacheRuntime::UploadGpuUnswizzle(Image& image, std::span<const u8> source) {
    const auto params = VideoCommon::MakePlainUnswizzleParameters(image.info);
    if (!params || source.size() < u64{params->source_words} * 4) {
        return false;
    }
    if (!plain_unswizzle_pass) {
        plain_unswizzle_pass.emplace(device, scheduler, descriptor_pool, compute_pass_descriptor_queue);
    }
    if (PlainGpuUnswizzleVerify() && !plain_unswizzle_tested) {
        plain_unswizzle_tested = true;
        if (!TestPlainUnswizzle()) {
            plain_unswizzle_failed = true;
            return false;
        }
    }
    auto input = UploadStagingBuffer(size_t{params->source_words} * 4);
    if (input.offset % device.GetStorageBufferAlignment() != 0) {
        return false;
    }
    std::memcpy(input.mapped_span.data(), source.data(), size_t{params->source_words} * 4);
    const bool verify_early = PlainGpuUnswizzleVerify() && plain_verified_early < 64;
    const bool verify_late = PlainGpuUnswizzleVerify() && plain_unswizzle_frames >= 2500 &&
                             plain_verified_early >= 64 && plain_verified_late < 64;
    const VkBuffer output = DispatchPlainUnswizzle(*params, input, verify_early || verify_late);
    if (verify_early || verify_late) {
        if (!CheckPlainUnswizzle(image.info, *params, input, output)) {
            plain_unswizzle_failed = true;
            return false;
        }
        ++(verify_early ? plain_verified_early : plain_verified_late);
        if ((plain_verified_early + plain_verified_late) % 16 == 0) {
            LOG_INFO(Render_Vulkan, "Plain GPU unswizzle verification: {} early / {} late byte-exact uploads",
                     plain_verified_early, plain_verified_late);
        }
    }
    const std::array copies{VideoCommon::BufferImageCopy{
        .buffer_offset = 0,
        .buffer_size = size_t{params->word_count} * 4,
        .buffer_row_length = image.info.size.width,
        .buffer_image_height = image.info.size.height,
        .image_subresource = {0, 0, 1},
        .image_offset = {0, 0, 0},
        .image_extent = image.info.size,
    }};
    image.UploadMemory(output, 0, copies);
    ++plain_unswizzle_uploads;
    plain_unswizzle_bytes += u64{params->word_count} * 4;
    return true;
}

vk::Image TextureCacheRuntime::AcquireImageAllocation(const ImageInfo& info) {
    if (!ImageAllocationReuseEnabled() || resolution.up_scale != 1 || resolution.down_shift != 0) {
        return {};
    }
    if (const auto key = VideoCommon::ReusableImageAllocationKey(info)) {
        return image_allocation_pool.Acquire(*key);
    }
    return {};
}

void TextureCacheRuntime::RecycleImage(Image& image, u64 completed_tick) {
    constexpr auto unsupported = VideoCommon::ImageFlagBits::Converted |
                                 VideoCommon::ImageFlagBits::IsDecoding;
    if (!ImageAllocationReuseEnabled() || resolution.up_scale != 1 || resolution.down_shift != 0 ||
        !image.original_image || image.original_image.IsUnbacked() || image.scaled_image ||
        image.HasScaled() || !VideoCommon::IsAttachmentBackingUsable(image) ||
        True(image.flags & unsupported)) {
        return;
    }
    const auto key = VideoCommon::ReusableImageAllocationKey(image.info);
    if (!key) {
        return;
    }
    const VkImage handle = *image.original_image;
    const auto memory = image.original_image.GetMemoryBlock();
    if (image_allocation_pool.Recycle(*key, image.original_image,
                                      memory.size, completed_tick,
                                      [this](u64 tick) { return scheduler.IsFree(tick); },
                                      memory.block, memory.block_size)) {
        EraseResolveShadow(handle);
    }
}

void TextureCacheRuntime::RecycleFinishedTranscodes() {
    if (pending_transcodes.empty() || !scheduler.IsFree(pending_transcodes.front().tick)) {
        return;
    }
    static constexpr u64 MaxFreeTranscodeBytes = u64{256} << 20;
    static const bool evict_lru = [] {
        const char* value = std::getenv("EDEN_NX2_ASTC_POOL_V2");
        return value == nullptr || value[0] != '0';
    }();
    const auto decode_bytes = [](const PendingTranscode& t) {
        u64 bytes{};
        for (u32 level = 0; level < t.levels; ++level) {
            bytes += u64{(std::max)(t.width >> level, 1U)} * (std::max)(t.height >> level, 1U) * 4;
        }
        return bytes * t.layers;
    };
    u64 free_bytes{};
    for (const PendingTranscode& t : free_transcodes) {
        free_bytes += decode_bytes(t);
    }
    while (!pending_transcodes.empty() && scheduler.IsFree(pending_transcodes.front().tick)) {
        PendingTranscode& done = pending_transcodes.front();
        const u64 bytes = decode_bytes(done);
        while (evict_lru && free_bytes + bytes > MaxFreeTranscodeBytes && !free_transcodes.empty()) {
            const auto oldest = std::ranges::min_element(
                free_transcodes, {}, [](const PendingTranscode& t) { return t.last_use; });
            free_bytes -= decode_bytes(*oldest);
            free_transcodes.erase(oldest);
        }
        if (free_bytes + bytes <= MaxFreeTranscodeBytes) {
            done.unused_frames = 0;
            free_bytes += bytes;
            free_transcodes.push_back(std::move(done));
        }
        pending_transcodes.pop_front();
    }
}

vk::Image TextureCacheRuntime::TakePooledImage(const ImagePoolKey& key) {
    if (!key.valid || VideoCore::Nx2Debug::Experiment(42) == 0) {
        return vk::Image{};
    }
    // Newest first: its memory is the most likely to be resident still.
    for (auto it = image_pool.rbegin(); it != image_pool.rend(); ++it) {
        if (it->key != key) {
            continue;
        }
        vk::Image image = std::move(it->image);
        image_pool_bytes -= it->bytes;
        image_pool.erase(std::next(it).base());
        ++image_pool_taken;
        return image;
    }
    ++image_pool_missed;
    return vk::Image{};
}

void TextureCacheRuntime::PoolImage(const ImagePoolKey& key, vk::Image image, u64 bytes) {
    if (!key.valid || !image || bytes > ImagePoolMaxBytes) {
        return;
    }
    image_pool.push_back(PooledImage{
        .key = key,
        .image = std::move(image),
        .bytes = bytes,
        .frame = image_pool_frame,
    });
    image_pool_bytes += bytes;
    ++image_pool_given;
    TrimImagePool(ImagePoolCap());
}

u64 TextureCacheRuntime::ImagePoolCap() {
    return VideoCore::Nx2Debug::Experiment(42) >= 2 ? ImagePoolMaxBytesDedicated : ImagePoolMaxBytes;
}

void TextureCacheRuntime::TrimImagePool(u64 max_bytes) {
    while (!image_pool.empty() && image_pool_bytes > max_bytes) {
        image_pool_bytes -= image_pool.front().bytes;
        image_pool.pop_front();
        ++image_pool_evicted;
    }
}

void TextureCacheRuntime::TickFrame() {
    if (PlainGpuUnswizzleEnabled()) {
        static const bool log_plain = std::getenv("EDEN_NX2_TC_LOG") != nullptr;
        if (++plain_unswizzle_frames % 600 == 0 && log_plain) {
            LOG_INFO(Render_Vulkan, "Plain GPU unswizzle totals: {} uploads, {} MiB, verification failed={}",
                     plain_unswizzle_uploads, plain_unswizzle_bytes >> 20, plain_unswizzle_failed);
        }
    }
    if (ImageAllocationReuseEnabled()) {
        const u32 pressure_events = MemoryPressureEvents();
        if (pressure_events != allocation_pool_pressure_events) {
            allocation_pool_pressure_frames = 60;
            allocation_pool_pressure_events = pressure_events;
        }
        if (++allocation_pool_frames % 30 == 0) {
            const u64 budget = GetDeviceMemoryBudget();
            const u64 used = GetDeviceMemoryUsage();
            if (budget != 0 && (used >= budget || budget - used < (u64{512} << 20))) {
                allocation_pool_pressure_frames = 60;
            }
        }
        image_allocation_pool.Tick(allocation_pool_pressure_frames != 0);
        if (allocation_pool_pressure_frames != 0) {
            --allocation_pool_pressure_frames;
        }
        static const bool log_pool = std::getenv("EDEN_NX2_TC_LOG") != nullptr;
        if (log_pool && allocation_pool_frames % 600 == 0) {
            LOG_INFO(Render_Vulkan,
                     "Image allocation reuse totals: {} hits, {} misses, {} recycled, {} evicted; "
                     "{} images {} MiB allocations / {} MiB blocks held",
                     image_allocation_pool.Hits(), image_allocation_pool.Misses(),
                     image_allocation_pool.Recycled(), image_allocation_pool.Evicted(), image_allocation_pool.Size(),
                     image_allocation_pool.AllocationBytes() >> 20,
                     image_allocation_pool.Bytes() >> 20);
        }
    }
    ++image_pool_frame;
    {
        // EXP42's pool: images older than ImagePoolMaxFrames go, and all of them when the
        // switch is off or device memory ran short (an image did not fit or could not be made).
        const u32 pressure = memory_allocator.MemoryPressureEvents();
        const u32 failed = memory_allocator.FailedImages();
        // Host memory pressure is only signalled on iOS (shared memory: the OS low-memory warning
        // or the process nearing its limit).
        const u32 host_pressure = Common::PlatformEvents::MemoryPressureGeneration();
        const bool short_of_memory = pressure != image_pool_seen_pressure ||
                                     failed != image_pool_seen_failed ||
                                     host_pressure != image_pool_seen_host_pressure;
        image_pool_seen_pressure = pressure;
        image_pool_seen_failed = failed;
        image_pool_seen_host_pressure = host_pressure;
        const int pool_switch = VideoCore::Nx2Debug::Experiment(42);
        if (pool_switch == 0 || short_of_memory) {
            TrimImagePool(0);
        }
        while (!image_pool.empty() &&
               image_pool_frame - image_pool.front().frame > ImagePoolMaxFrames) {
            image_pool_bytes -= image_pool.front().bytes;
            image_pool.pop_front();
            ++image_pool_evicted;
        }
        if (image_pool_frame % 600 == 0 &&
            (pool_switch != 0 || image_pool_taken + image_pool_given != 0)) {
            LOG_INFO(Render_Vulkan,
                     "Image pool (EXP42) last 600 frames: {} taken, {} missed, {} pooled, {} "
                     "evicted; holds {} images, {} MiB",
                     image_pool_taken, image_pool_missed, image_pool_given, image_pool_evicted,
                     image_pool.size(), image_pool_bytes >> 20);
            image_pool_taken = image_pool_missed = image_pool_given = image_pool_evicted = 0;
        }
    }
    while (!pending_byte_addressed_shadows.empty() &&
           scheduler.IsFree(pending_byte_addressed_shadows.front().tick)) {
        pending_byte_addressed_shadows.pop_front();
    }
    static constexpr u32 MAX_UNUSED_SCRATCH_FRAMES = 60;
    RecycleFinishedTranscodes();
    std::erase_if(free_transcodes, [](PendingTranscode& t) {
        return ++t.unused_frames > MAX_UNUSED_SCRATCH_FRAMES;
    });
    std::erase_if(msaa_scratch_images, [this](MsaaScratchImage& scratch) {
        if (!scheduler.IsFree(scratch.tick)) {
            scratch.unused_frames = 0;
            return false;
        }
        return ++scratch.unused_frames > MAX_UNUSED_SCRATCH_FRAMES;
    });
    std::erase_if(pending_resolve_shadows, [this](const auto& pending) {
        return scheduler.IsFree(pending.first);
    });
}

Image::Image(TextureCacheRuntime& runtime_, const ImageInfo& info_, GPUVAddr gpu_addr_,
             VAddr cpu_addr_)
    : VideoCommon::ImageBase(info_, gpu_addr_, cpu_addr_), scheduler{&runtime_.scheduler},
      runtime{&runtime_}, aspect_mask(ImageAspectMask(info.format)) {
    const auto host_info = TrimmedHostInfo(runtime_.device, info, host_level_offset);
    const bool unbacked = UsesSparseLayers(runtime_.device, info, gpu_addr_) ||
                          UsesSparseTiles(runtime_.device, info, gpu_addr_) ||
                          IsUnwrittenSparse(runtime_.device, info, gpu_addr_);
    if (!unbacked) {
        original_image = runtime_.AcquireImageAllocation(host_info);
    }
    if (!original_image) {
        original_image = MakeImage(
            runtime_.device, runtime_.memory_allocator, host_info,
            IsHugeAstcArray(runtime_.device, info) || IsUnevenMipRgba8Astc(runtime_.device, info)
                ? std::span<const VkFormat>{HugeAstcViewFormats}
            : IsOpaqueBc1Astc(runtime_.device, info)
                ? std::span<const VkFormat>{OpaqueAstcViewFormats}
                : runtime->ViewFormats(info.format),
            AstcHostFormat(runtime_.device, info), unbacked,
            UsesSparseTiles(runtime_.device, info, gpu_addr_) &&
                SparseTilesModeOf(runtime_.device) == SparseTilesMode::Aliased,
            true, &runtime_, &original_pool_key, &from_pool);
    }
    if (OutOfMemory()) {
        return;
    }
    if (original_image.IsUnbacked()) {
        flags |= VideoCommon::ImageFlagBits::Unbacked;
        const u64 texels = u64{host_info.size.width} * host_info.size.height *
                           host_info.size.depth * static_cast<u64>(host_info.resources.layers);
        if (UsesSparseTiles(runtime_.device, info, gpu_addr_) && InitSparseTiles()) {
            flags |= VideoCommon::ImageFlagBits::SparseTiles;
        } else if (UsesSparseLayers(runtime_.device, info, gpu_addr_) && InitSparseLayers()) {
            flags |= VideoCommon::ImageFlagBits::SparseLayers;
        } else if (texels <= MaxBackedImageTexels) {
            // Not too big to back (see MemoryAllocator::CreateImage): unbacked because nothing
            // was written there yet, so the first write gets it replaced.
            flags |= VideoCommon::ImageFlagBits::UnwrittenSparse;
        }
    }
    if (host_level_offset != 0) {
        flags |= VideoCommon::ImageFlagBits::TrimmedLevels;
    }
    if (IsHugeAstcArray(runtime->device, info)) {
        flags |= VideoCommon::ImageFlagBits::AcceleratedUpload;
        flags |= VideoCommon::ImageFlagBits::Converted;
        flags |= VideoCommon::ImageFlagBits::CostlyLoad;
    } else if (IsPixelFormatASTC(info.format) && !runtime->device.IsOptimalAstcSupported()) {
        switch (Settings::values.accelerate_astc.GetValue()) {
        case Settings::AstcDecodeMode::Gpu:
            if (WillUseAcceleratedAstcDecode(runtime->device, info) ||
                WillUseGpuBcnTranscode(runtime->device, info)) {
                flags |= VideoCommon::ImageFlagBits::AcceleratedUpload;
            } else if (!gpu_astc_decoder_available) {
                // Without the GPU decoder, decode off the GPU thread as CpuAsynchronous does.
                flags |= VideoCommon::ImageFlagBits::AsynchronousDecode;
            }
            if (IsUnevenMipRgba8Astc(runtime->device, info)) {
                static std::atomic<u32> uneven_images{0};
                const u32 count = ++uneven_images;
                if (count <= 4 || (count & (count - 1)) == 0) {
                    LOG_INFO(Render_Vulkan,
                             "ASTC {}x{}x{} with {} levels kept as RGBA8 (uneven BC block mips), "
                             "{} images so far",
                             info.size.width, info.size.height, info.resources.layers,
                             info.resources.levels, count);
                }
            }
            break;
        case Settings::AstcDecodeMode::CpuAsynchronous:
            flags |= VideoCommon::ImageFlagBits::AsynchronousDecode;
            break;
        default:
            break;
        }
        flags |= VideoCommon::ImageFlagBits::Converted;
        flags |= VideoCommon::ImageFlagBits::CostlyLoad;
    }
    if (IsPixelFormatBCn(info.format) && !runtime->device.IsOptimalBcnSupported()) {
        flags |= VideoCommon::ImageFlagBits::Converted;
        flags |= VideoCommon::ImageFlagBits::CostlyLoad;
    }
    if (runtime->device.HasDebuggingToolAttached()) {
        original_image.SetObjectNameEXT(VideoCommon::Name(*this).c_str());
    }
    current_image = &Image::original_image;
    const s32 host_levels = info.resources.levels - static_cast<s32>(host_level_offset);
    storage_image_views.resize(host_levels);
    if (WillUseAcceleratedAstcDecode(runtime->device, info)) {
        const auto& device = runtime->device.GetLogical();
        const VkFormat storage_format = VK_FORMAT_A8B8G8R8_UNORM_PACK32;
        for (s32 level = 0; level < host_levels; ++level) {
            storage_image_views[level] =
                MakeStorageView(device, level, *original_image, storage_format);
        }
    }
}

Image::Image(const VideoCommon::NullImageParams& params) : VideoCommon::ImageBase{params} {}

void Image::ReleaseToPool() {
    using VideoCommon::ImageFlagBits;
    if (runtime == nullptr || VideoCore::Nx2Debug::Experiment(42) == 0) {
        return;
    }
    constexpr ImageFlagBits unusual = ImageFlagBits::Sparse | ImageFlagBits::SparseLayers |
                                      ImageFlagBits::SparseTiles | ImageFlagBits::UnwrittenSparse |
                                      ImageFlagBits::Unbacked;
    if (True(flags & unusual)) {
        return;
    }
    constexpr u64 MinPooledBytes = u64{256} << 10;
    const auto give = [this](vk::Image& image, const ImagePoolKey& key) {
        if (!key.valid || !image || image.IsUnbacked()) {
            return;
        }
        // Memory placed in host memory over budget is freed as before. Dedicated allocations
        // (block 0) too with EXP42=1; EXP42=2 pools them as well (2026-11-10: in the Stalchild
        // fight 2,295 images a 600 frames asked the pool and only 309 were ever given back, the
        // big render targets such as a 1280x720 one remade up to 282 times being dedicated).
        // The image keeps its own allocation either way.
        const vk::Image::MemoryBlock block = image.GetMemoryBlock();
        const bool dedicated = block.block == 0;
        if ((dedicated && VideoCore::Nx2Debug::Experiment(42) < 2) || block.size < MinPooledBytes ||
            !image.IsDeviceLocal()) {
            return;
        }
        if (ENABLE_MSAA_RESOLVE_CONSUME) {
            runtime->EraseResolveShadow(*image);
        }
        runtime->PoolImage(key, std::move(image), block.size);
    };
    give(original_image, original_pool_key);
    give(scaled_image, scaled_pool_key);
}

Image::~Image() {
    if (ENABLE_MSAA_RESOLVE_CONSUME && runtime != nullptr) {
        if (original_image) {
            runtime->EraseResolveShadow(*original_image);
        }
        if (scaled_image) {
            runtime->EraseResolveShadow(*scaled_image);
        }
    }
}

void Image::AllocateComputeUnswizzleBuffer(u32 max_slices) {
    using VideoCore::Surface::BytesPerBlock;

    const u32 block_bytes  = BytesPerBlock(info.format); // 8 for BC1, 16 for BC6H
    const u32 block_width  = 4;
    const u32 block_height = 4;

    // BCn is 4x4x1 blocks
    const u32 blocks_x = (info.size.width  + block_width  - 1) / block_width;
    const u32 blocks_y = (info.size.height + block_height - 1) / block_height;
    const u32 blocks_z = (std::min)(max_slices, info.size.depth);

    const u64 block_count =
        static_cast<u64>(blocks_x) *
        static_cast<u64>(blocks_y) *
        static_cast<u64>(blocks_z);

    const VkDeviceSize required_size = block_count * block_bytes;
    if (has_compute_unswizzle_buffer && required_size <= compute_unswizzle_buffer_size) {
        return;
    }

    compute_unswizzle_buffer_size = required_size;

    VkBufferCreateInfo ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = compute_unswizzle_buffer_size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };

    const MemoryTagScope memory_tag{MemoryTag::TextureHelper};
    compute_unswizzle_buffer =
        runtime->memory_allocator.CreateBuffer(ci, MemoryUsage::DeviceLocal);

    has_compute_unswizzle_buffer = true;
}

void Image::UploadMemory(VkBuffer buffer, VkDeviceSize offset,
                         std::span<const VideoCommon::BufferImageCopy> copies) {
    // TODO: Move this to another API
    if (ENABLE_MSAA_RESOLVE_CONSUME && runtime != nullptr) {
        runtime->InvalidateResolveShadow(Handle());
    }
    const bool is_rescaled = True(flags & ImageFlagBits::Rescaled);
    if (is_rescaled) {
        ScaleDown(true);
    }

    const bool msaa_upload_is_depth = (aspect_mask & VK_IMAGE_ASPECT_DEPTH_BIT) != 0;
    const bool wants_msaa_upload = info.num_samples > 1
        && ((aspect_mask & VK_IMAGE_ASPECT_COLOR_BIT) != 0 || msaa_upload_is_depth);

    if (wants_msaa_upload) {
        const bool msaa_upload_copies_stencil =
            msaa_upload_is_depth && (aspect_mask & VK_IMAGE_ASPECT_STENCIL_BIT) != 0 &&
            runtime->device.IsExtShaderStencilExportSupported();
        VkImageAspectFlags upload_aspect_mask = aspect_mask;
        if (msaa_upload_is_depth) {
            upload_aspect_mask = VK_IMAGE_ASPECT_DEPTH_BIT;
            if (msaa_upload_copies_stencil) {
                upload_aspect_mask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            }
        }
        const VkImageCreateInfo image_ci = MakeMsaaScratchImageCreateInfo(
            runtime->device, info, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        const VkImage temp_vk_image = runtime->AcquireMsaaScratchImage(image_ci);

        scheduler->RequestOutsideRenderPassOperationContext();
        auto vk_copies = TransformBufferImageCopies(copies, offset, upload_aspect_mask);
        const VkBuffer src_buffer = buffer;
        const VkImageAspectFlags vk_aspect_mask = upload_aspect_mask;

        scheduler->Record([src_buffer, temp_vk_image, vk_aspect_mask,
                           vk_copies](vk::CommandBuffer cmdbuf) {
            CopyBufferToImage(cmdbuf, src_buffer, temp_vk_image, vk_aspect_mask, false, vk_copies);
        });

        boost::container::small_vector<VideoCommon::ImageCopy, 16> image_copies;
        image_copies.reserve(copies.size());
        for (const auto& copy : copies) {
            image_copies.push_back(VideoCommon::MsaaUploadCopy(copy, info.num_samples));
        }

        if (msaa_upload_is_depth) {
            runtime->blit_image_helper.CopyMSAADepth(runtime->render_pass_cache, Handle(),
                                                     info.format, temp_vk_image, info.format,
                                                     info.num_samples,
                                                     {image_copies.data(), image_copies.size()},
                                                     msaa_upload_copies_stencil, false);
        } else {
            runtime->blit_image_helper.CopyMSAA(runtime->render_pass_cache, Handle(), info.format,
                                                temp_vk_image, info.format, info.num_samples,
                                                {image_copies.data(), image_copies.size()}, false);
        }
        initialized = true;
        runtime->ReleaseMsaaScratchImage(temp_vk_image);

        if (is_rescaled) {
            ScaleUp();
        }
        return;
    }

    if (info.num_samples > 1) {
        if (is_rescaled) {
            ScaleUp();
        }
        return;
    }

    scheduler->RequestOutsideRenderPassOperationContext();
    auto vk_copies = TransformBufferImageCopies(copies, offset, aspect_mask);
    const VkBuffer src_buffer = buffer;
    const VkImage vk_image = *original_image;
    const VkImageAspectFlags vk_aspect_mask = aspect_mask;
    const bool was_initialized = std::exchange(initialized, true);

    scheduler->Record([src_buffer, vk_image, vk_aspect_mask, was_initialized,
                       vk_copies](vk::CommandBuffer cmdbuf) {
        CopyBufferToImage(cmdbuf, src_buffer, vk_image, vk_aspect_mask, was_initialized, vk_copies);
    });

    if (is_rescaled) {
        ScaleUp();
    }
}

void Image::UploadMemory(const StagingBufferRef& map, std::span<const BufferImageCopy> copies) {
    UploadMemory(map.buffer, map.offset, copies);
}

void Image::DownloadMemory(VkBuffer buffer, size_t offset,
                           std::span<const VideoCommon::BufferImageCopy> copies) {
    std::array buffer_handles{
        buffer,
    };
    std::array buffer_offsets{
        offset,
    };
    DownloadMemory(buffer_handles, buffer_offsets, copies);
}

void Image::DownloadMemory(std::span<VkBuffer> buffers_span, std::span<size_t> offsets_span,
                            std::span<const VideoCommon::BufferImageCopy> copies) {
    if (!original_image) {
        LOG_CRITICAL(Render_Vulkan, "Missing image at download: gpu={:#x}, cpu={:#x}, type={}, format={}, flags={:#x}, size={}, dimensions={}x{}",
                     gpu_addr, cpu_addr, static_cast<u32>(info.type), static_cast<u32>(info.format),
                     static_cast<u32>(flags), unswizzled_size_bytes, info.size.width, info.size.height);
    }
    const bool is_rescaled = True(flags & ImageFlagBits::Rescaled);
    if (is_rescaled) {
        ScaleDown();
    }

    if (info.num_samples > 1) {
        if (runtime->CanDownloadMsaa(info)) {
            const bool msaa_download_is_depth =
                (aspect_mask & VK_IMAGE_ASPECT_DEPTH_BIT) != 0;
            const bool msaa_download_copies_stencil =
                msaa_download_is_depth && (aspect_mask & VK_IMAGE_ASPECT_STENCIL_BIT) != 0 &&
                runtime->device.IsExtShaderStencilExportSupported();
            VkImageUsageFlags scratch_usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            if (msaa_download_is_depth) {
                scratch_usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            } else {
                scratch_usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            }
            const VkImageCreateInfo image_ci =
                MakeMsaaScratchImageCreateInfo(runtime->device, info, scratch_usage);
            const VkImage temp_vk_image = runtime->AcquireMsaaScratchImage(image_ci);

            const VkImageAspectFlags temp_aspect_mask = aspect_mask;
            const VkAccessFlags attachment_access =
                msaa_download_is_depth ? (VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
                                       : (VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                          VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
            const VkPipelineStageFlags attachment_stage =
                msaa_download_is_depth ? (VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                          VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT)
                                       : VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

            scheduler->RequestOutsideRenderPassOperationContext();
            scheduler->Record([temp_vk_image, temp_aspect_mask, attachment_access,
                               attachment_stage](vk::CommandBuffer cmdbuf) {
                const VkImageMemoryBarrier init_barrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = 0,
                    .dstAccessMask = attachment_access,
                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = temp_vk_image,
                    .subresourceRange{
                        .aspectMask = temp_aspect_mask,
                        .baseMipLevel = 0,
                        .levelCount = VK_REMAINING_MIP_LEVELS,
                        .baseArrayLayer = 0,
                        .layerCount = VK_REMAINING_ARRAY_LAYERS,
                    },
                };
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, attachment_stage, 0,
                                       init_barrier);
            });

            boost::container::small_vector<VideoCommon::ImageCopy, 16> image_copies;
            image_copies.reserve(copies.size());
            for (const auto& copy : copies) {
                VideoCommon::ImageCopy image_copy;
                image_copy.src_offset = copy.image_offset;
                image_copy.dst_offset = copy.image_offset;
                image_copy.src_subresource = copy.image_subresource;
                image_copy.dst_subresource = copy.image_subresource;
                image_copy.extent = copy.image_extent;
                image_copies.push_back(image_copy);
            }

            if (msaa_download_is_depth) {
                runtime->blit_image_helper.CopyMSAADepth(
                    runtime->render_pass_cache, temp_vk_image, info.format, Handle(), info.format,
                    info.num_samples, {image_copies.data(), image_copies.size()},
                    msaa_download_copies_stencil, true);
            } else {
                runtime->blit_image_helper.CopyMSAA(runtime->render_pass_cache, temp_vk_image,
                                                    info.format, Handle(), info.format,
                                                    info.num_samples,
                                                    {image_copies.data(), image_copies.size()},
                                                    true);
            }

            boost::container::small_vector<VkBuffer, 8> buffers_vector{};
            boost::container::small_vector<boost::container::small_vector<VkBufferImageCopy, 16>, 8>
                vk_copies;
            for (size_t index = 0; index < buffers_span.size(); index++) {
                buffers_vector.emplace_back(buffers_span[index]);
                vk_copies.emplace_back(
                    TransformBufferImageCopies(copies, offsets_span[index], aspect_mask));
            }

            scheduler->RequestOutsideRenderPassOperationContext();
            scheduler->Record([buffers = std::move(buffers_vector), image = temp_vk_image,
                               aspect_mask_ = aspect_mask, vk_copies](vk::CommandBuffer cmdbuf) {
                const VkImageMemoryBarrier read_barrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = image,
                    .subresourceRange{
                        .aspectMask = aspect_mask_,
                        .baseMipLevel = 0,
                        .levelCount = VK_REMAINING_MIP_LEVELS,
                        .baseArrayLayer = 0,
                        .layerCount = VK_REMAINING_ARRAY_LAYERS,
                    },
                };
                cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                       0, read_barrier);

                for (size_t index = 0; index < buffers.size(); index++) {
                    cmdbuf.CopyImageToBuffer(image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffers[index],
                                             vk_copies[index]);
                }

                const VkMemoryBarrier memory_write_barrier{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                };
                const VkImageMemoryBarrier image_write_barrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = 0,
                    .dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = image,
                    .subresourceRange{
                        .aspectMask = aspect_mask_,
                        .baseMipLevel = 0,
                        .levelCount = VK_REMAINING_MIP_LEVELS,
                        .baseArrayLayer = 0,
                        .layerCount = VK_REMAINING_ARRAY_LAYERS,
                    },
                };
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, vk::PIPELINE_STAGE_GRAPHICS_COMPUTE,
                                       0, memory_write_barrier, nullptr, image_write_barrier);
            });
            runtime->ReleaseMsaaScratchImage(temp_vk_image);
        }
        if (is_rescaled) {
            ScaleUp(true);
        }
        return;
    } else {
        boost::container::small_vector<VkBuffer, 8> buffers_vector{};
        boost::container::small_vector<boost::container::small_vector<VkBufferImageCopy, 16>, 8>
            vk_copies;
        for (size_t index = 0; index < buffers_span.size(); index++) {
            buffers_vector.emplace_back(buffers_span[index]);
            vk_copies.emplace_back(
                TransformBufferImageCopies(copies, offsets_span[index], aspect_mask));
        }
        scheduler->RequestOutsideRenderPassOperationContext();
        scheduler->Record([buffers = std::move(buffers_vector), image = *original_image,
                           aspect_mask_ = aspect_mask, vk_copies](vk::CommandBuffer cmdbuf) {
            const VkImageMemoryBarrier read_barrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = image,
                .subresourceRange{
                    .aspectMask = aspect_mask_,
                    .baseMipLevel = 0,
                    .levelCount = VK_REMAINING_MIP_LEVELS,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            };
            cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   0, read_barrier);

            for (size_t index = 0; index < buffers.size(); index++) {
                cmdbuf.CopyImageToBuffer(image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffers[index],
                                         vk_copies[index]);
            }

            const VkMemoryBarrier memory_write_barrier{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            };
            const VkImageMemoryBarrier image_write_barrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = 0,
                .dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = image,
                .subresourceRange{
                    .aspectMask = aspect_mask_,
                    .baseMipLevel = 0,
                    .levelCount = VK_REMAINING_MIP_LEVELS,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, vk::PIPELINE_STAGE_GRAPHICS_COMPUTE,
                                   0, memory_write_barrier, nullptr, image_write_barrier);
        });
    }

    if (is_rescaled) {
        ScaleUp(true);
    }
}

void Image::DownloadMemory(const StagingBufferRef& map, std::span<const BufferImageCopy> copies) {
    std::array buffers{
        map.buffer,
    };
    std::array offsets{
        static_cast<size_t>(map.offset),
    };
    DownloadMemory(buffers, offsets, copies);
}

VkImageView Image::StorageImageView(s32 level) noexcept {
    auto& view = storage_image_views[level];
    if (!view) {
        auto format_info =
            MaxwellToVK::SurfaceFormat(runtime->device, FormatType::Optimal, true, info.format);
        if (WillUseAcceleratedAstcDecode(runtime->device, info)) {
            format_info.format = VK_FORMAT_A8B8G8R8_UNORM_PACK32;
        }
        view = MakeStorageView(runtime->device.GetLogical(), level, *(this->*current_image),
                               format_info.format);
    }
    return *view;
}

bool Image::InitSparseLayers() {
    const Device& device = runtime->device;
    const VkImage handle = *original_image;
    const auto requirements = device.GetLogical().GetImageSparseMemoryRequirements(handle);
    const auto it = std::ranges::find_if(requirements, [](const auto& r) {
        return (r.formatProperties.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) != 0;
    });
    if (it == requirements.end()) {
        return false;
    }
    SparseLayerMemory& s = sparse_layers;
    s.allocator = &runtime->memory_allocator;
    s.requirements = device.GetLogical().GetImageMemoryRequirements(handle);
    s.sparse = *it;
    u32 offset = host_level_offset;
    const ImageInfo host = TrimmedHostInfo(device, info, offset);
    s.width = host.size.width;
    s.height = host.size.height;
    s.levels = static_cast<u32>(host.resources.levels);
    // Sparse block size in texels (drivers give it in texels or in compressed blocks).
    const bool huge_astc = IsHugeAstcArray(device, info);
    const u32 block_width = huge_astc ? 1 : VideoCore::Surface::DefaultBlockWidth(info.format);
    const u32 block_height = huge_astc ? 1 : VideoCore::Surface::DefaultBlockHeight(info.format);
    const u64 block_bytes = huge_astc ? 4 : VideoCore::Surface::BytesPerBlock(info.format);
    VkExtent3D granularity = it->formatProperties.imageGranularity;
    if (granularity.width == 0 || granularity.height == 0) {
        // Seen with AMD's driver: the sparse block size comes back as 0, and the size computation
        // below divided by it. Use the standard 2D block shape when the device promises it.
        if (!StandardSparseBlockShape(device, block_bytes, granularity)) {
            LOG_WARNING(Render_Vulkan,
                        "Sparse layered image {:#x}: the driver reports no sparse block size and "
                        "no standard block shape; not using sparse layers",
                        gpu_addr);
            return false;
        }
        granularity.width *= block_width;
        granularity.height *= block_height;
        LOG_INFO(Render_Vulkan,
                 "Sparse layered image {:#x}: no block size from the driver, using the standard "
                 "{}x{} texel block",
                 gpu_addr, granularity.width, granularity.height);
    }
    s.tile_width = granularity.width;
    s.tile_height = granularity.height;
    const u64 tile = s.requirements.alignment;
    if (u64{granularity.width / block_width} * (granularity.height / block_height) * block_bytes !=
            tile &&
        u64{granularity.width} * granularity.height * block_bytes == tile) {
        s.tile_width *= block_width;
        s.tile_height *= block_height;
    }
    const u32 tail_lod = (std::min)(it->imageMipTailFirstLod, s.levels);
    VkDeviceSize bytes = 0;
    for (u32 level = 0; level < tail_lod; ++level) {
        const u32 width = (std::max)(s.width >> level, 1U);
        const u32 height = (std::max)(s.height >> level, 1U);
        bytes += u64{Common::DivCeil(width, s.tile_width)} * Common::DivCeil(height, s.tile_height) *
                 tile;
    }
    if (tail_lod < s.levels &&
        (it->formatProperties.flags & VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT) == 0) {
        bytes += it->imageMipTailSize;
    }
    s.layer_bytes = Common::AlignUp(bytes, tile);
    s.layer_memory.assign(static_cast<size_t>(info.resources.layers), VK_NULL_HANDLE);
    LOG_INFO(Render_Vulkan,
             "Sparse layered image {:#x}: {}x{} {} levels {} layers, {} KiB a layer, block {}x{}",
             gpu_addr, s.width, s.height, s.levels, info.resources.layers, s.layer_bytes >> 10,
             s.tile_width, s.tile_height);
    return true;
}

bool Image::InitSparseTiles() {
    const Device& device = runtime->device;
    const VkImage handle = *original_image;
    const auto requirements = device.GetLogical().GetImageSparseMemoryRequirements(handle);
    const auto it = std::ranges::find_if(requirements, [](const auto& r) {
        return (r.formatProperties.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) != 0;
    });
    if (it == requirements.end()) {
        return false;
    }
    SparseLayerMemory& s = sparse_layers;
    s.requirements = device.GetLogical().GetImageMemoryRequirements(handle);
    VkExtent3D granularity = it->formatProperties.imageGranularity;
    const u32 bytes = VideoCore::Surface::BytesPerBlock(info.format);
    const u32 block_width = VideoCore::Surface::DefaultBlockWidth(info.format);
    const u32 block_height = VideoCore::Surface::DefaultBlockHeight(info.format);
    if (u64{granularity.width} * granularity.height * bytes == s.requirements.alignment) {
        granularity.width *= block_width;
        granularity.height *= block_height;
    }
    // Rectangles are uploaded by whole GOB columns (64 bytes across).
    const u32 row_bytes = granularity.width / block_width * bytes;
    if (granularity.width == 0 || granularity.height == 0 || row_bytes % 64 != 0 ||
        granularity.width % block_width != 0 || granularity.height % block_height != 0 ||
        u64{row_bytes} * (granularity.height / block_height) != s.requirements.alignment) {
        return false;
    }
    s.allocator = &runtime->memory_allocator;
    s.sparse = *it;
    s.width = info.size.width;
    s.height = info.size.height;
    s.levels = static_cast<u32>(info.resources.levels);
    s.tile_width = granularity.width;
    s.tile_height = granularity.height;
    s.tail_level = (std::min)(it->imageMipTailFirstLod, s.levels);
    const size_t layers = static_cast<size_t>(info.resources.layers);
    if (s.tail_level < s.levels && layers > 1 &&
        !(s.sparse.formatProperties.flags & VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT)) {
        s.layer_tails.resize(layers);
    }
    s.level_tiles.resize(s.tail_level);
    for (u32 level = 0; level < s.tail_level; ++level) {
        const u32 width = (std::max)(s.width >> level, 1U);
        const u32 height = (std::max)(s.height >> level, 1U);
        s.level_tiles[level].assign(
            size_t{Common::DivCeil(width, s.tile_width)} * Common::DivCeil(height, s.tile_height) *
                layers,
            VK_NULL_HANDLE);
    }
    if (SparseTilesModeOf(device) == SparseTilesMode::Aliased && !BindZeroBlock()) {
        LOG_WARNING(Render_Vulkan, "Sparse tiled image {:#x}: no zeroed block to share", gpu_addr);
        s.Release();
        s.allocator = nullptr;
        s.bound_bytes = 0;
        return false;
    }
    LOG_INFO(Render_Vulkan,
             "Sparse tiled image {:#x}: {}x{} {} levels {} layers, format {}, block {}x{}, "
             "mip tail from {}{}",
             gpu_addr, s.width, s.height, s.levels, layers, static_cast<u32>(info.format),
             s.tile_width, s.tile_height, s.tail_level,
             s.zero_block != VK_NULL_HANDLE ? ", empty blocks share a zeroed one" : "");
    return true;
}

bool Image::BindZeroBlock() {
    SparseLayerMemory& s = sparse_layers;
    const Device& device = runtime->device;
    const u64 tile = s.requirements.alignment;
    VkDeviceMemory memory{};
    VkDeviceSize offset{};
    s.zero_block = s.allocator->AllocateSparseMemory(
        VkMemoryRequirements{
            .size = tile,
            .alignment = tile,
            .memoryTypeBits = s.requirements.memoryTypeBits,
        },
        memory, offset);
    if (s.zero_block == VK_NULL_HANDLE) {
        return false;
    }
    s.bound_bytes += tile;
    std::vector<VkSparseImageMemoryBind> image_binds;
    for (u32 level = 0; level < s.tail_level; ++level) {
        const u32 width = (std::max)(s.width >> level, 1U);
        const u32 height = (std::max)(s.height >> level, 1U);
        for (u32 y = 0; y < height; y += s.tile_height) {
            for (u32 x = 0; x < width; x += s.tile_width) {
                image_binds.push_back(VkSparseImageMemoryBind{
                    .subresource{
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .mipLevel = level,
                        .arrayLayer = 0,
                    },
                    .offset{static_cast<s32>(x), static_cast<s32>(y), 0},
                    .extent{(std::min)(s.tile_width, width - x), (std::min)(s.tile_height, height - y),
                            1},
                    .memory = memory,
                    .memoryOffset = offset,
                    .flags = 0,
                });
            }
        }
    }
    // The mip tail gets memory of its own: it is small, and the clear below zeroes it too.
    std::vector<VkSparseMemoryBind> opaque_binds;
    if (s.tail_level < s.levels) {
        const VkMemoryRequirements requirements{
            .size = Common::AlignUp(s.sparse.imageMipTailSize, tile),
            .alignment = tile,
            .memoryTypeBits = s.requirements.memoryTypeBits,
        };
        VkDeviceMemory tail_memory{};
        VkDeviceSize tail_offset{};
        s.tail_memory = s.allocator->AllocateSparseMemory(requirements, tail_memory, tail_offset);
        if (s.tail_memory == VK_NULL_HANDLE) {
            return false;
        }
        s.bound_bytes += requirements.size;
        opaque_binds.push_back(VkSparseMemoryBind{
            .resourceOffset = s.sparse.imageMipTailOffset,
            .size = s.sparse.imageMipTailSize,
            .memory = tail_memory,
            .memoryOffset = tail_offset,
            .flags = 0,
        });
    }
    const VkSparseImageMemoryBindInfo image_bind_info{
        .image = *original_image,
        .bindCount = static_cast<u32>(image_binds.size()),
        .pBinds = image_binds.data(),
    };
    const VkSparseImageOpaqueMemoryBindInfo opaque_bind_info{
        .image = *original_image,
        .bindCount = static_cast<u32>(opaque_binds.size()),
        .pBinds = opaque_binds.data(),
    };
    const VkBindSparseInfo bind_info{
        .sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = nullptr,
        .bufferBindCount = 0,
        .pBufferBinds = nullptr,
        .imageOpaqueBindCount = opaque_binds.empty() ? 0U : 1U,
        .pImageOpaqueBinds = opaque_binds.empty() ? nullptr : &opaque_bind_info,
        .imageBindCount = image_binds.empty() ? 0U : 1U,
        .pImageBinds = image_binds.empty() ? nullptr : &image_bind_info,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = nullptr,
    };
    vk::Fence fence = device.GetLogical().CreateFence(VkFenceCreateInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    });
    VkResult result;
    {
        std::scoped_lock lock{scheduler->submit_mutex};
        result = device.GetGraphicsQueue().BindSparse(bind_info, *fence);
    }
    if (result != VK_SUCCESS) {
        LOG_ERROR(Render_Vulkan, "Sparse tiled image {:#x}: binding the zeroed block failed ({})",
                  gpu_addr, static_cast<int>(result));
        return false;
    }
    fence.Wait();
    // Every block now reads the shared block (and the tail its own memory): zero them all.
    ClearTiles();
    return true;
}

void Image::ClearTiles() {
    scheduler->RequestOutsideRenderPassOperationContext();
    const VkImage vk_image = *original_image;
    const bool was_initialized = std::exchange(initialized, true);
    scheduler->Record([vk_image, was_initialized](vk::CommandBuffer cmdbuf) {
        static constexpr VkImageSubresourceRange range{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = VK_REMAINING_MIP_LEVELS,
            .baseArrayLayer = 0,
            .layerCount = VK_REMAINING_ARRAY_LAYERS,
        };
        const VkImageMemoryBarrier to_transfer{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = was_initialized ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = vk_image,
            .subresourceRange = range,
        };
        const VkImageMemoryBarrier to_general{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                             VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = vk_image,
            .subresourceRange = range,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                               to_transfer);
        cmdbuf.ClearColorImage(vk_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VkClearColorValue{},
                               range);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                               to_general);
    });
}

bool Image::BackTilesWhole() {
    SparseLayerMemory& s = sparse_layers;
    if (s.zero_block == VK_NULL_HANDLE) {
        return true;
    }
    std::vector<VideoCommon::LevelRect> rects;
    for (u32 level = 0; level < s.levels; ++level) {
        rects.push_back({static_cast<s32>(level), 0, 0, (std::max)(s.width >> level, 1U),
                         (std::max)(s.height >> level, 1U)});
    }
    if (!BindTiles(rects)) {
        return false;
    }
    // No block reads the shared one any more.
    s.allocator->FreeSparseMemory(s.zero_block);
    s.zero_block = VK_NULL_HANDLE;
    s.bound_bytes -= s.requirements.alignment;
    ClearTiles();
    return true;
}

bool Image::BindTiles(std::span<const VideoCommon::LevelRect> rects) {
    SparseLayerMemory& s = sparse_layers;
    if (s.allocator == nullptr) {
        return false;
    }
    const Device& device = runtime->device;
    const u64 tile = s.requirements.alignment;
    const VkMemoryRequirements tile_requirements{
        .size = tile,
        .alignment = tile,
        .memoryTypeBits = s.requirements.memoryTypeBits,
    };
    std::vector<VkSparseImageMemoryBind> image_binds;
    std::vector<VkSparseMemoryBind> opaque_binds;
    std::vector<VmaAllocation*> fresh; // Blocks given memory by this call
    std::vector<VmaAllocation*> fresh_tails;
    const VkDeviceSize tail_bytes = Common::AlignUp(s.sparse.imageMipTailSize, tile);
    // A failed bind leaves the blocks as they were, so the next call tries them again.
    const auto roll_back = [&] {
        for (VmaAllocation* const allocation : fresh) {
            s.allocator->FreeSparseMemory(*allocation);
            *allocation = VK_NULL_HANDLE;
            s.bound_bytes -= tile;
        }
        for (VmaAllocation* const allocation : fresh_tails) {
            s.allocator->FreeSparseMemory(*allocation);
            *allocation = VK_NULL_HANDLE;
            s.bound_bytes -= tail_bytes;
        }
    };
    const auto bind_tail = [&](u32 layer) {
        VmaAllocation& allocation = s.layer_tails.empty() ? s.tail_memory : s.layer_tails[layer];
        if (allocation != VK_NULL_HANDLE) {
            return true;
        }
        const VkMemoryRequirements requirements{
            .size = tail_bytes,
            .alignment = tile,
            .memoryTypeBits = s.requirements.memoryTypeBits,
        };
        VkDeviceMemory memory{};
        VkDeviceSize offset{};
        allocation = s.allocator->AllocateSparseMemory(requirements, memory, offset);
        if (allocation == VK_NULL_HANDLE) {
            LOG_WARNING(Render_Vulkan, "Sparse tiled image {:#x}: out of device memory for the mip "
                                       "tail",
                        gpu_addr);
            return false;
        }
        fresh_tails.push_back(&allocation);
        s.bound_bytes += tail_bytes;
        opaque_binds.push_back(VkSparseMemoryBind{
            .resourceOffset =
                s.sparse.imageMipTailOffset +
                (s.layer_tails.empty() ? 0 : layer * s.sparse.imageMipTailStride),
            .size = s.sparse.imageMipTailSize,
            .memory = memory,
            .memoryOffset = offset,
            .flags = 0,
        });
        return true;
    };
    for (const VideoCommon::LevelRect& rect : rects) {
        const u32 level = static_cast<u32>(rect.level);
        const u32 layer = static_cast<u32>(rect.layer);
        if (level >= s.tail_level) {
            if (!bind_tail(layer)) {
                roll_back();
                return false;
            }
            continue;
        }
        const u32 width = (std::max)(s.width >> level, 1U);
        const u32 height = (std::max)(s.height >> level, 1U);
        const u32 tiles_x = Common::DivCeil(width, s.tile_width);
        const size_t layer_offset = size_t{layer} * Common::DivCeil(height, s.tile_height) * tiles_x;
        for (u32 ty = rect.y / s.tile_height; ty <= (rect.y + rect.height - 1) / s.tile_height;
             ++ty) {
            for (u32 tx = rect.x / s.tile_width; tx <= (rect.x + rect.width - 1) / s.tile_width;
                 ++tx) {
                VmaAllocation& allocation = s.level_tiles[level][layer_offset + size_t{ty} * tiles_x + tx];
                if (allocation != VK_NULL_HANDLE) {
                    continue;
                }
                VkDeviceMemory memory{};
                VkDeviceSize offset{};
                allocation = s.allocator->AllocateSparseMemory(tile_requirements, memory, offset);
                if (allocation == VK_NULL_HANDLE) {
                    LOG_WARNING(Render_Vulkan, "Sparse tiled image {:#x}: out of device memory",
                                gpu_addr);
                    roll_back();
                    return false;
                }
                fresh.push_back(&allocation);
                s.bound_bytes += tile;
                const u32 x = tx * s.tile_width;
                const u32 y = ty * s.tile_height;
                image_binds.push_back(VkSparseImageMemoryBind{
                    .subresource{
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .mipLevel = level,
                        .arrayLayer = layer,
                    },
                    .offset{static_cast<s32>(x), static_cast<s32>(y), 0},
                    .extent{(std::min)(s.tile_width, width - x), (std::min)(s.tile_height, height - y),
                            1},
                    .memory = memory,
                    .memoryOffset = offset,
                    .flags = 0,
                });
            }
        }
    }
    if (image_binds.empty() && opaque_binds.empty()) {
        return true;
    }
    const VkSparseImageMemoryBindInfo image_bind_info{
        .image = *original_image,
        .bindCount = static_cast<u32>(image_binds.size()),
        .pBinds = image_binds.data(),
    };
    const VkSparseImageOpaqueMemoryBindInfo opaque_bind_info{
        .image = *original_image,
        .bindCount = static_cast<u32>(opaque_binds.size()),
        .pBinds = opaque_binds.data(),
    };
    const VkBindSparseInfo bind_info{
        .sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = nullptr,
        .bufferBindCount = 0,
        .pBufferBinds = nullptr,
        .imageOpaqueBindCount = opaque_binds.empty() ? 0U : 1U,
        .pImageOpaqueBinds = opaque_binds.empty() ? nullptr : &opaque_bind_info,
        .imageBindCount = image_binds.empty() ? 0U : 1U,
        .pImageBinds = image_binds.empty() ? nullptr : &image_bind_info,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = nullptr,
    };
    // Bound before the command buffer that uploads the blocks is submitted (as BindLayers).
    vk::Fence fence = device.GetLogical().CreateFence(VkFenceCreateInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    });
    VkResult result;
    {
        std::scoped_lock lock{scheduler->submit_mutex};
        result = device.GetGraphicsQueue().BindSparse(bind_info, *fence);
    }
    if (result != VK_SUCCESS) {
        LOG_ERROR(Render_Vulkan, "Sparse tiled image {:#x}: binding failed ({})", gpu_addr,
                  static_cast<int>(result));
        roll_back();
        return false;
    }
    fence.Wait();
    return true;
}

std::vector<u32> Image::BindLayers(std::span<const u32> layers) {
    SparseLayerMemory& s = sparse_layers;
    std::vector<u32> bound;
    if (s.layer_memory.empty()) {
        return bound;
    }
    const Device& device = runtime->device;
    const u64 tile = s.requirements.alignment;
    const u32 tail_lod = (std::min)(s.sparse.imageMipTailFirstLod, s.levels);
    const bool has_tail = tail_lod < s.levels;
    const bool single_tail =
        (s.sparse.formatProperties.flags & VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT) != 0;
    const VkDeviceSize cap = SparseLayersCap(device);
    std::vector<VkSparseImageMemoryBind> image_binds;
    std::vector<VkSparseMemoryBind> opaque_binds;
    const bool tail_was_bound = s.tail_memory != VK_NULL_HANDLE;
    std::vector<u32> fresh; // Layers given memory by this call
    // A failed bind leaves the layers as they were, so the next call tries them again.
    const auto roll_back = [&] {
        for (const u32 layer : fresh) {
            s.allocator->FreeSparseMemory(s.layer_memory[layer]);
            s.layer_memory[layer] = VK_NULL_HANDLE;
            s.bound_bytes -= s.layer_bytes;
        }
        if (s.tail_memory != VK_NULL_HANDLE && !tail_was_bound) {
            s.allocator->FreeSparseMemory(s.tail_memory);
            s.tail_memory = VK_NULL_HANDLE;
        }
    };
    for (const u32 layer : layers) {
        if (s.layer_memory[layer] != VK_NULL_HANDLE) {
            bound.push_back(layer);
            continue;
        }
        if (s.bound_bytes + s.layer_bytes > cap) {
            if (!std::exchange(s.warned, true)) {
                LOG_WARNING(Render_Vulkan, "Sparse layered image {:#x}: {} MiB bound, more layers "
                                           "stay without memory",
                            gpu_addr, s.bound_bytes >> 20);
            }
            continue;
        }
        const VkMemoryRequirements requirements{
            .size = s.layer_bytes,
            .alignment = tile,
            .memoryTypeBits = s.requirements.memoryTypeBits,
        };
        VkDeviceMemory memory{};
        VkDeviceSize offset{};
        const VmaAllocation allocation = s.allocator->AllocateSparseMemory(requirements, memory, offset);
        if (allocation == VK_NULL_HANDLE) {
            LOG_WARNING(Render_Vulkan, "Sparse layered image {:#x}: out of device memory", gpu_addr);
            break;
        }
        s.layer_memory[layer] = allocation;
        s.bound_bytes += s.layer_bytes;
        fresh.push_back(layer);
        for (u32 level = 0; level < tail_lod; ++level) {
            const u32 width = (std::max)(s.width >> level, 1U);
            const u32 height = (std::max)(s.height >> level, 1U);
            image_binds.push_back(VkSparseImageMemoryBind{
                .subresource{
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .mipLevel = level,
                    .arrayLayer = layer,
                },
                .offset{0, 0, 0},
                .extent{width, height, 1},
                .memory = memory,
                .memoryOffset = offset,
                .flags = 0,
            });
            offset += u64{Common::DivCeil(width, s.tile_width)} *
                      Common::DivCeil(height, s.tile_height) * tile;
        }
        if (has_tail && !single_tail) {
            opaque_binds.push_back(VkSparseMemoryBind{
                .resourceOffset = s.sparse.imageMipTailOffset + layer * s.sparse.imageMipTailStride,
                .size = s.sparse.imageMipTailSize,
                .memory = memory,
                .memoryOffset = offset,
                .flags = 0,
            });
        }
        bound.push_back(layer);
    }
    if (has_tail && single_tail && s.tail_memory == VK_NULL_HANDLE && !bound.empty()) {
        const VkMemoryRequirements requirements{
            .size = Common::AlignUp(s.sparse.imageMipTailSize, tile),
            .alignment = tile,
            .memoryTypeBits = s.requirements.memoryTypeBits,
        };
        VkDeviceMemory memory{};
        VkDeviceSize offset{};
        s.tail_memory = s.allocator->AllocateSparseMemory(requirements, memory, offset);
        if (s.tail_memory != VK_NULL_HANDLE) {
            opaque_binds.push_back(VkSparseMemoryBind{
                .resourceOffset = s.sparse.imageMipTailOffset,
                .size = s.sparse.imageMipTailSize,
                .memory = memory,
                .memoryOffset = offset,
                .flags = 0,
            });
        } else {
            LOG_WARNING(Render_Vulkan, "Sparse layered image {:#x}: out of device memory for the "
                                       "mip tail",
                        gpu_addr);
            roll_back();
            return {};
        }
    }
    if (image_binds.empty() && opaque_binds.empty()) {
        return bound;
    }
    const VkSparseImageMemoryBindInfo image_bind_info{
        .image = *original_image,
        .bindCount = static_cast<u32>(image_binds.size()),
        .pBinds = image_binds.data(),
    };
    const VkSparseImageOpaqueMemoryBindInfo opaque_bind_info{
        .image = *original_image,
        .bindCount = static_cast<u32>(opaque_binds.size()),
        .pBinds = opaque_binds.data(),
    };
    const VkBindSparseInfo bind_info{
        .sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = nullptr,
        .bufferBindCount = 0,
        .pBufferBinds = nullptr,
        .imageOpaqueBindCount = opaque_binds.empty() ? 0U : 1U,
        .pImageOpaqueBinds = opaque_binds.empty() ? nullptr : &opaque_bind_info,
        .imageBindCount = image_binds.empty() ? 0U : 1U,
        .pImageBinds = image_binds.empty() ? nullptr : &image_bind_info,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = nullptr,
    };
    // Bound before the command buffer that uploads the layers is submitted (as the multi-range
    // buffer cache does).
    vk::Fence fence = device.GetLogical().CreateFence(VkFenceCreateInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    });
    VkResult result;
    {
        std::scoped_lock lock{scheduler->submit_mutex};
        result = device.GetGraphicsQueue().BindSparse(bind_info, *fence);
    }
    if (result != VK_SUCCESS) {
        LOG_ERROR(Render_Vulkan, "Sparse layered image {:#x}: binding failed ({})", gpu_addr,
                  static_cast<int>(result));
        roll_back();
        return {};
    }
    fence.Wait();
    return bound;
}

VkImageView Image::StorageImageLayerView(s32 level, u32 layer) {
    const u32 layers = static_cast<u32>(info.resources.layers);
    if (storage_layer_views.empty()) {
        storage_layer_views.resize(storage_image_views.size() * layers);
    }
    auto& view = storage_layer_views[static_cast<size_t>(level) * layers + layer];
    if (!view) {
        view = MakeStorageView(runtime->device.GetLogical(), level, *(this->*current_image),
                               VK_FORMAT_A8B8G8R8_UNORM_PACK32, layer, 1);
    }
    return *view;
}

bool Image::IsRescaled() const noexcept {
    return True(flags & ImageFlagBits::Rescaled);
}

bool Image::ScaleUp(bool ignore) {
    const auto& resolution = runtime->resolution;
    if (!resolution.active) {
        return false;
    }
    if (True(flags & ImageFlagBits::Rescaled)) {
        return false;
    }
    ASSERT(info.type != ImageType::Linear);
    flags |= ImageFlagBits::Rescaled;
    has_scaled = true;
    if (!scaled_image) {
        const bool is_2d = info.type == ImageType::e2D;
        const u32 scaled_width = resolution.ScaleUp(info.size.width);
        const u32 scaled_height = is_2d ? resolution.ScaleUp(info.size.height) : info.size.height;
        auto scaled_info = info;
        scaled_info.size.width = scaled_width;
        scaled_info.size.height = scaled_height;
        const MemoryTagScope memory_tag{MemoryTag::RescaledImage};
        scaled_image = MakeImage(runtime->device, runtime->memory_allocator, scaled_info,
                                 runtime->ViewFormats(info.format), {}, false, false, false,
                                 runtime, &scaled_pool_key);
        ignore = false;
    }
    current_image = &Image::scaled_image;
    if (ignore) {
        return true;
    }
    if (aspect_mask == 0) {
        aspect_mask = ImageAspectMask(info.format);
    }
    if (NeedsScaleHelper()) {
        if (!BlitScaleHelper(true)) {
            current_image = &Image::original_image;
            return false;
        }
    } else {
        BlitScale(*scheduler, *original_image, *scaled_image, info, aspect_mask, resolution);
    }
    return true;
}

bool Image::ScaleDown(bool ignore) {
    const auto& resolution = runtime->resolution;
    if (!resolution.active) {
        return false;
    }
    if (False(flags & ImageFlagBits::Rescaled)) {
        return false;
    }
    ASSERT(info.type != ImageType::Linear);
    flags &= ~ImageFlagBits::Rescaled;
    current_image = &Image::original_image;
    if (ignore) {
        return true;
    }
    if (aspect_mask == 0) {
        aspect_mask = ImageAspectMask(info.format);
    }
    if (NeedsScaleHelper()) {
        if (!BlitScaleHelper(false)) {
            current_image = &Image::scaled_image;
            flags |= ImageFlagBits::Rescaled;
            return false;
        }
    } else {
        BlitScale(*scheduler, *scaled_image, *original_image, info, aspect_mask, resolution, false);
    }
    return true;
}

bool Image::BlitScaleHelper(bool scale_up) {
    using namespace VideoCommon;
    static constexpr auto BLIT_OPERATION = Tegra::Engines::Fermi2D::Operation::SrcCopy;
    const bool is_color{aspect_mask == VK_IMAGE_ASPECT_COLOR_BIT};
    const bool is_bilinear{is_color && !IsPixelFormatInteger(info.format)};
    const auto operation = is_bilinear ? Tegra::Engines::Fermi2D::Filter::Bilinear
                                       : Tegra::Engines::Fermi2D::Filter::Point;

    const bool is_2d = info.type == ImageType::e2D;
    const auto& resolution = runtime->resolution;
    const u32 scaled_width = resolution.ScaleUp(info.size.width);
    const u32 scaled_height = is_2d ? resolution.ScaleUp(info.size.height) : info.size.height;
    std::unique_ptr<ImageView>& blit_view = scale_up ? scale_view : normal_view;
    std::optional<Framebuffer>& blit_framebuffer = scale_up ? scale_framebuffer : normal_framebuffer;
    if (!blit_view) {
        const auto view_info = ImageViewInfo(ImageViewType::e2D, info.format);
        blit_view = std::make_unique<ImageView>(*runtime, view_info, NULL_IMAGE_ID, *this);
    }

    const auto [samples_x, samples_y] = VideoCommon::SamplesLog2(info.num_samples);
    const u32 src_width = (scale_up ? info.size.width : scaled_width) >> samples_x;
    const u32 src_height = (scale_up ? info.size.height : scaled_height) >> samples_y;
    const u32 dst_width = (scale_up ? scaled_width : info.size.width) >> samples_x;
    const u32 dst_height = (scale_up ? scaled_height : info.size.height) >> samples_y;
    const Region2D src_region{
        .start = {0, 0},
        .end = {s32(src_width), s32(src_height)},
    };
    const Region2D dst_region{
        .start = {0, 0},
        .end = {s32(dst_width), s32(dst_height)},
    };
    const VkExtent2D extent{
        .width = (std::max)(scaled_width, info.size.width) >> samples_x,
        .height = (std::max)(scaled_height, info.size.height) >> samples_y,
    };

    auto* view_ptr = blit_view.get();
    if (aspect_mask == VK_IMAGE_ASPECT_COLOR_BIT) {
        if (!blit_framebuffer)
            blit_framebuffer.emplace(*runtime, view_ptr, nullptr, extent, scale_up);
        if (info.num_samples > 1) {
            runtime->blit_image_helper.BlitColorMSAA(&*blit_framebuffer, *blit_view,
                dst_region, src_region);
        } else {
            runtime->blit_image_helper.BlitColor(&*blit_framebuffer, *blit_view,
                dst_region, src_region, operation, BLIT_OPERATION);
        }
    } else if ((aspect_mask & VK_IMAGE_ASPECT_DEPTH_BIT) != 0) {
        if (!blit_framebuffer)
            blit_framebuffer.emplace(*runtime, nullptr, view_ptr, extent, scale_up);
        const bool has_stencil = (aspect_mask & VK_IMAGE_ASPECT_STENCIL_BIT) != 0;
        const bool can_blit_stencil =
            has_stencil && runtime->device.IsExtShaderStencilExportSupported();
        if (info.num_samples > 1) {
            runtime->blit_image_helper.BlitDepthStencilMSAA(&*blit_framebuffer, *blit_view,
                dst_region, src_region);
        } else if (can_blit_stencil) {
            runtime->blit_image_helper.BlitDepthStencil(&*blit_framebuffer, *blit_view,
                dst_region, src_region, operation, BLIT_OPERATION);
        } else {
            runtime->blit_image_helper.BlitDepth(&*blit_framebuffer, *blit_view, dst_region,
                                                 src_region);
        }
    } else {
        // TODO: Use helper blits where applicable
        flags &= ~ImageFlagBits::Rescaled;
        LOG_ERROR(Render_Vulkan, "Device does not support scaling format {}", info.format);
        return false;
    }
    return true;
}

bool Image::NeedsScaleHelper() const {
    const auto& device = runtime->device;
    if (info.num_samples > 1) {
        return true;
    }
    static constexpr auto OPTIMAL_FORMAT = FormatType::Optimal;
    const auto vk_format =
        MaxwellToVK::SurfaceFormat(device, OPTIMAL_FORMAT, false, info.format).format;
    const auto blit_usage = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
    const bool needs_blit_helper = !device.IsFormatSupported(vk_format, blit_usage, OPTIMAL_FORMAT);
    return needs_blit_helper;
}

ImageView::ImageView(TextureCacheRuntime& runtime, const VideoCommon::ImageViewInfo& info,
                     ImageId image_id_, Image& image)
    : VideoCommon::ImageViewBase{info, image.info, image_id_, image.gpu_addr},
      device{&runtime.device}, image_handle{image.Handle()},
      samples(ConvertSampleCount(image.info.num_samples)) {
    using Shader::TextureType;

    const VkImageAspectFlags aspect_mask = ImageViewAspectMask(info);
    std::array<SwizzleSource, 4> swizzle{
        SwizzleSource::R,
        SwizzleSource::G,
        SwizzleSource::B,
        SwizzleSource::A,
    };
    if (!info.IsRenderTarget()) {
        swizzle = info.Swizzle();
        TryTransformSwizzleIfNeeded(format, swizzle,
                        device->MustEmulateBGR565(),
                        !device->IsExt4444FormatsSupported());
        if ((aspect_mask & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) != 0) {
            std::ranges::transform(swizzle, swizzle.begin(), ConvertGreenRed);
            SanitizeDepthStencilSwizzle(swizzle, device->SupportsDepthStencilSwizzleOne());
        }
    }
    auto format_info = MaxwellToVK::SurfaceFormat(*device, FormatType::Optimal, true, format);
    if (IsHugeAstcArray(*device, image.info) || IsUnevenMipRgba8Astc(*device, image.info)) {
        format_info.format = VideoCore::Surface::IsPixelFormatSRGB(format)
                                 ? VK_FORMAT_A8B8G8R8_SRGB_PACK32
                                 : VK_FORMAT_A8B8G8R8_UNORM_PACK32;
    } else if (IsOpaqueBc1Astc(*device, image.info) && IsPixelFormatASTC(format)) {
        format_info.format = VideoCore::Surface::IsPixelFormatSRGB(format)
                                 ? VK_FORMAT_BC1_RGBA_SRGB_BLOCK
                                 : VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    }
    if (device->ApiVersion() >= VK_API_VERSION_1_3) {
        const VkFormatProperties3 properties3 =
            device->GetPhysical().GetFormatProperties3(format_info.format);
        supports_depth_comparison =
            (properties3.optimalTilingFeatures &
             VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT) != 0;
        supports_minmax_filter = (properties3.optimalTilingFeatures &
                                  VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_MINMAX_BIT) != 0;
    } else {
        const VkFormatFeatureFlags format_features =
            device->GetPhysical().GetFormatProperties(format_info.format).optimalTilingFeatures;
        supports_depth_comparison =
            (aspect_mask & VK_IMAGE_ASPECT_DEPTH_BIT) != 0 &&
            (format_features & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0;
        supports_minmax_filter =
            (format_features & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_MINMAX_BIT) != 0;
    }
    if (VideoCommon::RenderTrace::Enabled() &&
        (aspect_mask & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) != 0) {
        LOG_DEBUG(Render_Vulkan,
                  "Render trace depth view: docked={} address={:#x} guest_format={} vk_format={} "
                  "image={}x{} layers={} view={}x{} base_level={} base_layer={} layers={} "
                  "aspects={:#x} compare_supported={} render_target={}",
                  Settings::IsDockedMode(), image.gpu_addr, info.format,
                  static_cast<u32>(format_info.format), image.info.size.width, image.info.size.height,
                  image.info.resources.layers, size.width, size.height, range.base.level,
                  range.base.layer, range.extent.layers, aspect_mask, supports_depth_comparison,
                  info.IsRenderTarget());
    }
    requires_border_color_format = NeedsExplicitBorderColorFormat(format_info.format);
    swizzle_mapping = VkComponentMapping{
        .r = ComponentSwizzle(swizzle[0]),
        .g = ComponentSwizzle(swizzle[1]),
        .b = ComponentSwizzle(swizzle[2]),
        .a = ComponentSwizzle(swizzle[3]),
    };
    has_identity_swizzle = swizzle[0] == SwizzleSource::R && swizzle[1] == SwizzleSource::G &&
                           swizzle[2] == SwizzleSource::B && swizzle[3] == SwizzleSource::A;
    const VkImageUsageFlags requested_view_usage = ImageUsageFlags(format_info, format);
    const VkImageUsageFlags image_usage = image.UsageFlags();
    const VkImageUsageFlags clamped_view_usage = requested_view_usage & image_usage;
    const VkImageViewUsageCreateInfo image_view_usage{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
        .pNext = nullptr,
        .usage = clamped_view_usage,
    };
    const VkImageViewASTCDecodeModeEXT astc_decode_mode{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_ASTC_DECODE_MODE_EXT,
        .pNext = &image_view_usage,
        .decodeMode = VK_FORMAT_R8G8B8A8_UNORM,
    };
    const void* view_next = &image_view_usage;
    if (device->IsExtAstcDecodeModeSupported() && IsLdrAstcFormat(format_info.format)) {
        view_next = &astc_decode_mode;
    }
    const VkImageViewCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = view_next,
        .flags = 0,
        .image = image.Handle(),
        .viewType = VkImageViewType{},
        .format = format_info.format,
        .components = swizzle_mapping,
        .subresourceRange = MakeSubresourceRange(aspect_mask, HostRange(image, info.range)),
    };
    const auto create = [&](TextureType tex_type, std::optional<u32> num_layers) {
        VkImageViewCreateInfo ci{create_info};
        ci.viewType = ImageViewType(tex_type);
        if (num_layers) {
            ci.subresourceRange.layerCount = *num_layers;
        }
        vk::ImageView handle = device->GetLogical().CreateImageView(ci);
        if (device->HasDebuggingToolAttached()) {
            handle.SetObjectNameEXT(VideoCommon::Name(*this, gpu_addr).c_str());
        }
        image_views[static_cast<size_t>(tex_type)] = std::move(handle);
    };
    switch (info.type) {
    case VideoCommon::ImageViewType::e1D:
    case VideoCommon::ImageViewType::e1DArray:
        create(TextureType::Color1D, 1);
        create(TextureType::ColorArray1D, std::nullopt);
        render_target = Handle(TextureType::ColorArray1D);
        break;
    case VideoCommon::ImageViewType::e2D:
    case VideoCommon::ImageViewType::e2DArray:
    case VideoCommon::ImageViewType::Rect:
        create(TextureType::Color2D, 1);
        create(TextureType::ColorArray2D, std::nullopt);
        render_target = Handle(Shader::TextureType::ColorArray2D);
        // Native shaders can sample six layers of a 2D array as a cube (point-light shadows).
        // Adapted from Duck912/eden-fefw a1b23ac3.
        if (info.type == VideoCommon::ImageViewType::e2DArray && info.range.extent.layers >= 6 &&
            image.info.num_samples == 1 && IsCubeCompatible(*device, image.info)) {
            create(TextureType::ColorCube, 6);
            if (info.range.extent.layers % 6 == 0) {
                create(TextureType::ColorArrayCube, std::nullopt);
            }
        }
        break;
    case VideoCommon::ImageViewType::e3D:
        create(TextureType::Color3D, std::nullopt);
        render_target = Handle(Shader::TextureType::Color3D);
        break;
    case VideoCommon::ImageViewType::Cube:
    case VideoCommon::ImageViewType::CubeArray:
        create(TextureType::ColorCube, 6);
        create(TextureType::ColorArrayCube, std::nullopt);
        break;
    case VideoCommon::ImageViewType::Buffer:
        ASSERT(false);
        break;
    }
}

ImageView::ImageView(TextureCacheRuntime& runtime, const VideoCommon::ImageViewInfo& info, ImageId image_id_, Image& image, const SlotVector<Image>& slot_imgs)
    : ImageView{runtime, info, image_id_, image}
{
    slot_images = &slot_imgs;
}

ImageView::ImageView(TextureCacheRuntime& runtime, const VideoCommon::ImageInfo& info,
                     const VideoCommon::ImageViewInfo& view_info, GPUVAddr gpu_addr_)
    : VideoCommon::ImageViewBase{info, view_info, gpu_addr_},
      buffer_size{static_cast<u32>(VideoCommon::CalculateGuestSizeInBytes(info))} {}

ImageView::ImageView(TextureCacheRuntime& runtime, const VideoCommon::NullImageViewParams& params)
    : VideoCommon::ImageViewBase{params}, device{&runtime.device} {
    if (device->HasNullDescriptor()) {
        return;
    }

    // Handle fallback for devices without nullDescriptor
    ImageInfo info{};
    info.format = PixelFormat::A8B8G8R8_UNORM;

    const MemoryTagScope memory_tag{MemoryTag::NullImage};
    null_image = MakeImage(*device, runtime.memory_allocator, info, {});
    image_handle = *null_image;
    for (u32 i = 0; i < Shader::NUM_TEXTURE_TYPES; i++) {
        image_views[i] = MakeView(VK_FORMAT_A8B8G8R8_UNORM_PACK32, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    // MakeView builds a 2D view when the backing image is not 3D or 6+ layers; Metal rejects a 2D
    // texture bound to a 3D or cube sampler, so give those slots a backing image of their own type.
    ImageInfo info_3d = info;
    info_3d.type = ImageType::e3D;
    null_image_3d = MakeImage(*device, runtime.memory_allocator, info_3d, {});

    ImageInfo info_cube = info;
    info_cube.type = ImageType::e2D;
    info_cube.resources.layers = 6;
    null_image_cube = MakeImage(*device, runtime.memory_allocator, info_cube, {});

    const auto make_typed_null_view = [&](VkImage img, VkImageViewType view_type, u32 layers) {
        return device->GetLogical().CreateImageView({
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = img,
            .viewType = view_type,
            .format = VK_FORMAT_A8B8G8R8_UNORM_PACK32,
            .components = {},
            .subresourceRange =
                {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = layers,
                },
        });
    };
    image_views[static_cast<size_t>(Shader::TextureType::Color3D)] =
        make_typed_null_view(*null_image_3d, VK_IMAGE_VIEW_TYPE_3D, 1);
    image_views[static_cast<size_t>(Shader::TextureType::ColorCube)] =
        make_typed_null_view(*null_image_cube, VK_IMAGE_VIEW_TYPE_CUBE, 6);
    image_views[static_cast<size_t>(Shader::TextureType::ColorArrayCube)] =
        make_typed_null_view(*null_image_cube, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY, 6);
}

ImageView::~ImageView() = default;

VkImageView ImageView::DepthView() {
    if (!image_handle) {
        return VK_NULL_HANDLE;
    }
    if (depth_view) {
        return *depth_view;
    }
    const auto& info = MaxwellToVK::SurfaceFormat(*device, FormatType::Optimal, true, format);
    depth_view = MakeView(info.format, VK_IMAGE_ASPECT_DEPTH_BIT);
    return *depth_view;
}

VkImageView ImageView::StencilView() {
    if (!image_handle) {
        return VK_NULL_HANDLE;
    }
    if (stencil_view) {
        return *stencil_view;
    }
    const auto& info = MaxwellToVK::SurfaceFormat(*device, FormatType::Optimal, true, format);
    stencil_view = MakeView(info.format, VK_IMAGE_ASPECT_STENCIL_BIT);
    return *stencil_view;
}

VkImageView ImageView::ColorView() {
    if (!image_handle) {
        return VK_NULL_HANDLE;
    }
    if (color_view) {
        return *color_view;
    }
    color_view = MakeView(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
    return *color_view;
}

VkImageView ImageView::StorageView(Shader::TextureType texture_type,
                                   Shader::ImageFormat image_format, bool is_byte_addressed) {
    if (is_byte_addressed &&
        (texture_type != Shader::TextureType::Color2D ||
         image_format != Shader::ImageFormat::R32_UINT ||
         !Shader::IsByteAddressed32BitColorFormat(static_cast<Shader::TexturePixelFormat>(format)) ||
         (type != VideoCommon::ImageViewType::e2D && type != VideoCommon::ImageViewType::Rect) ||
         samples != VK_SAMPLE_COUNT_1_BIT || range.extent.layers != 1 || range.extent.levels != 1 ||
         size.depth != 1 || size.width == 0 || size.height == 0 || !image_handle)) {
        throw Shader::RuntimeError("Unsupported byte-addressed B32 image binding: format {} type {} "
                                   "samples {} layers {} levels {} size {}x{}x{}",
                                   format, type, samples, range.extent.layers,
                                   range.extent.levels, size.width, size.height, size.depth);
    }
    if (image_handle) {
        if (image_format == Shader::ImageFormat::Typeless) {
            if (!typeless_storage_view) {
                auto info = MaxwellToVK::SurfaceFormat(*device, FormatType::Optimal, true, format);
                typeless_storage_view = MakeView(info.format, VK_IMAGE_ASPECT_COLOR_BIT, texture_type);
            }
            return *typeless_storage_view;
        }
        if (!storage_views)
            storage_views.emplace();
        auto& views{storage_views->formats[static_cast<size_t>(image_format)]};
        auto& view{views[size_t(texture_type)]};
        if (!view)
            view = MakeView(Format(image_format), VK_IMAGE_ASPECT_COLOR_BIT, texture_type);
        return *view;
    }
    return VK_NULL_HANDLE;
}

bool ImageView::IsRescaled() const noexcept {
    // The null view (a garbled or unbound texture id) has no image.
    if (slot_images == nullptr || !image_id) {
        return false;
    }
    return (*slot_images)[image_id].IsRescaled();
}

vk::ImageView ImageView::MakeView(VkFormat vk_format, VkImageAspectFlags aspect_mask,
                                  std::optional<Shader::TextureType> texture_type) {
    VkImageViewType view_type = ImageViewType(type);
    VkImageSubresourceRange subresource_range = MakeSubresourceRange(aspect_mask, range);
    const auto is_3d = [](VkImageViewType t) { return t == VK_IMAGE_VIEW_TYPE_3D; };
    if (texture_type && is_3d(ImageViewType(*texture_type)) != is_3d(view_type)) {
        // A 3D view of a 2D image (or the reverse) is invalid in Vulkan and aborts on Metal; keep
        // the image's own type when the shader's declared dimension disagrees with it.
        LOG_WARNING(Render_Vulkan,
                    "Shader image type {} does not match view type {} at 0x{:x} (format={} "
                    "size={}x{}x{}), using the view type",
                    *texture_type, type, gpu_addr, format, size.width, size.height, size.depth);
        texture_type.reset();
    }
    if (texture_type) {
        view_type = ImageViewType(*texture_type);
        switch (view_type) {
        case VK_IMAGE_VIEW_TYPE_1D_ARRAY:
        case VK_IMAGE_VIEW_TYPE_2D_ARRAY:
        case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
            break;
        default:
            subresource_range.layerCount = 1;
            break;
        }
    }
    return device->GetLogical().CreateImageView({
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = image_handle,
        .viewType = view_type,
        .format = vk_format,
        .components{
            .r = VK_COMPONENT_SWIZZLE_IDENTITY,
            .g = VK_COMPONENT_SWIZZLE_IDENTITY,
            .b = VK_COMPONENT_SWIZZLE_IDENTITY,
            .a = VK_COMPONENT_SWIZZLE_IDENTITY,
        },
        .subresourceRange = subresource_range,
    });
}

CustomBorderColorBudget::~CustomBorderColorBudget() {
    Release();
}

CustomBorderColorBudget::CustomBorderColorBudget(CustomBorderColorBudget&& rhs) noexcept
    : device_ptr{std::exchange(rhs.device_ptr, nullptr)}, held{std::exchange(rhs.held, 0)} {}

CustomBorderColorBudget& CustomBorderColorBudget::operator=(
    CustomBorderColorBudget&& rhs) noexcept {
    if (this != &rhs) {
        Release();
        device_ptr = std::exchange(rhs.device_ptr, nullptr);
        held = std::exchange(rhs.held, 0);
    }
    return *this;
}

bool CustomBorderColorBudget::TryAcquire(const Device& device, size_t count) {
    if (!device.TryReserveCustomBorderColorSamplers(count)) {
        return false;
    }
    device_ptr = &device;
    held += count;
    return true;
}

void CustomBorderColorBudget::Release() noexcept {
    if (device_ptr != nullptr) {
        device_ptr->ReleaseCustomBorderColorSamplers(held);
    }
    device_ptr = nullptr;
    held = 0;
}

f32 TextureCacheRuntime::OcarinaFsrLodBias() {
    if (!ocarina_fsr_lod_bias) {
        const bool ocarina = Settings::GetCurrentProgramID() == 0x0400486024DEC000ULL;
        f32 bias = 0.0f;
        if (ocarina) {
            const auto backend = ResolveOcarinaUpscaler(device);
            const auto effective_res = Settings::OcarinaUpscaledResolution(
                Settings::values.oot_resolution.GetValue(), Settings::values.upscaler.GetValue(),
                Settings::IsDockedMode());
            const auto mode = Settings::OcarinaDlssModeFor(effective_res);
            bias = VideoCommon::OcarinaFsrLodBias(backend, mode);
            if (bias == 0.0f && backend == VideoCommon::OcarinaUpscaleBackend::Dlss) {
                bias = VideoCommon::OcarinaDlssLodBias(mode);
            }
        }
        ocarina_fsr_lod_bias = bias;
    }
    return *ocarina_fsr_lod_bias;
}

bool TextureCacheRuntime::CanRescaleImage(const VideoCommon::ImageInfo& info) const noexcept {
    if (!resolution.active) {
        return true;
    }
    const bool is_2d = info.type == ImageType::e2D;
    const u64 scaled_width = resolution.ScaleUp(info.size.width);
    const u64 scaled_height = is_2d ? resolution.ScaleUp(info.size.height) : info.size.height;
    const auto& limits = device.GetPhysical().GetProperties().limits;
    const u64 max_dim = std::min<u64>({limits.maxImageDimension2D, limits.maxFramebufferWidth,
                                       limits.maxFramebufferHeight});
    const u64 depth = (std::max)(info.size.depth, 1U);
    const u64 layers = static_cast<u64>((std::max)(info.resources.layers, 1));
    return CanRescaleImageExtent(scaled_width, scaled_height, depth, layers, max_dim);
}

Sampler::Sampler(TextureCacheRuntime& runtime, const Tegra::Texture::TSCEntry& tsc) {
    const auto& device = runtime.device;
    device_ptr = &device;
    descriptor = tsc;
    border_color = tsc.BorderColor();
    srgb_border_color = tsc.SrgbBorderColor();

    const f32 max_anisotropy = std::clamp(tsc.MaxAnisotropy(), 1.0f, 16.0f);
    default_anisotropy = static_cast<f32>(1U << tsc.max_anisotropy);

    const VkFilter mag_filter{MaxwellToVK::Sampler::Filter(tsc.mag_filter)};
    const VkFilter min_filter{MaxwellToVK::Sampler::Filter(tsc.min_filter)};
    const VkSamplerMipmapMode mipmap_mode{MaxwellToVK::Sampler::MipmapMode(tsc.mipmap_filter)};

    const VkSamplerAddressMode wrap_u{
        MaxwellToVK::Sampler::WrapMode(device, tsc.wrap_u, tsc.mag_filter)};
    const VkSamplerAddressMode wrap_v{
        MaxwellToVK::Sampler::WrapMode(device, tsc.wrap_v, tsc.mag_filter)};
    const VkSamplerAddressMode wrap_p{
        MaxwellToVK::Sampler::WrapMode(device, tsc.wrap_p, tsc.mag_filter)};
    const bool samples_border = wrap_u == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ||
                                wrap_v == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ||
                                wrap_p == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;

    reduction_mode = MaxwellToVK::SamplerReduction(tsc.reduction_filter);

    has_added_anisotropy = max_anisotropy > default_anisotropy;
    has_linear_filtering = mag_filter == VK_FILTER_LINEAR || min_filter == VK_FILTER_LINEAR ||
                           mipmap_mode == VK_SAMPLER_MIPMAP_MODE_LINEAR;
    has_depth_comparison = tsc.depth_compare_enabled != 0;
    has_minmax_reduction = reduction_mode != VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE_EXT;
    has_srgb_border_color = tsc.srgb_conversion != 0 && srgb_border_color != border_color;
    if (has_minmax_reduction && !device.IsExtSamplerFilterMinmaxSupported()) {
        LOG_WARNING(Render_Vulkan, "VK_EXT_sampler_filter_minmax is required");
        has_minmax_reduction = false;
    }
    has_custom_border_colors = samples_border && device.IsCustomBorderColorUsable();
    needs_swizzle_mapping = has_custom_border_colors && device.NeedsBorderColorSwizzleMapping();

    if (has_custom_border_colors && GPU::Logging::IsActive()) {
        GPU::Logging::GPULogger::GetInstance().LogExtensionUsage(
            "VK_EXT_custom_border_color", "Sampler::Sampler");
    }
    if (device.IsExtBorderColorSwizzleSupported() && GPU::Logging::IsActive()) {
        GPU::Logging::GPULogger::GetInstance().LogExtensionUsage(
            "VK_EXT_border_color_swizzle", "Sampler::Sampler");
    }

    f32 min_lod = 0.0f;
    f32 max_lod = 0.25f;
    if (tsc.mipmap_filter != TextureMipmapFilter::None) {
        min_lod = tsc.MinLod();
        max_lod = tsc.MaxLod();
        // EDEN_NX2_FORCE_LOD0: sample only the top mip level (debug aid for mip layout problems).
        if (static const bool force_lod0 = std::getenv("EDEN_NX2_FORCE_LOD0") != nullptr; force_lod0) {
            min_lod = 0.0f;
            max_lod = 0.25f;
        }
    }
    // Ocarina with a jittered temporal upscaler: the scene renders below the output size, so
    // textures would pick the render size's mips; the bias keeps the output's detail
    // (VideoCommon::OcarinaFsrLodBias).
    f32 lod_bias = tsc.LodBias();
    ocarina_fsr_failed = &runtime.ocarina_fsr_failed;
    if (const f32 fsr_lod_bias = runtime.OcarinaFsrLodBias();
        fsr_lod_bias != 0.0f && tsc.mipmap_filter != TextureMipmapFilter::None) {
        const f32 biased = std::clamp(lod_bias + fsr_lod_bias, -15.0f, 15.0f);
        added_lod_bias = biased - lod_bias;
        lod_bias = biased;
    }
    base_ci = VkSamplerCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = mag_filter,
        .minFilter = min_filter,
        .mipmapMode = mipmap_mode,
        .addressModeU = wrap_u,
        .addressModeV = wrap_v,
        .addressModeW = wrap_p,
        .mipLodBias = lod_bias,
        .anisotropyEnable = static_cast<VkBool32>(max_anisotropy > 1.0f),
        .maxAnisotropy = max_anisotropy,
        .compareEnable = static_cast<VkBool32>(tsc.depth_compare_enabled),
        .compareOp = MaxwellToVK::Sampler::DepthCompareFunction(tsc.depth_compare_func),
        .minLod = min_lod,
        .maxLod = max_lod,
        .borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT,
        .unnormalizedCoordinates = VK_FALSE,
    };
    variants.reserve(1);
    Emplace(VariantKey{});
}

Sampler::VariantKey Sampler::MakeKey(const ImageView& image_view, bool is_depth) const noexcept {
    VariantKey key{};
    key.reduce_anisotropy = has_added_anisotropy && !image_view.SupportsAnisotropy();
    key.force_nearest = has_linear_filtering && IsPixelFormatInteger(image_view.format);
    key.drop_reduction = has_minmax_reduction && !image_view.SupportsMinmaxFilter();
    key.drop_custom_border = has_custom_border_colors && image_view.RequiresBorderColorFormat();
    key.srgb_border = has_srgb_border_color && IsPixelFormatSRGB(image_view.format);
    // A compare-enabled TSC read by a plain (non-Dref) sample: Maxwell ignores the comparison,
    // Vulkan leaves it undefined and AMD returns the 0/1 compare result (Ocarina forest fog
    // composite read alpha 0 -> black screen). Use a variant without comparison.
    key.drop_compare = has_depth_comparison && !is_depth;
    // The negative LOD bias is for the game's textures in both docked and handheld modes while a
    // temporal upscaler is active. Images the GPU wrote (render targets: bloom and blur chains,
    // reflections, fog) keep the game's own bias, and so does every image after the upscaler
    // failed this session (the game's pass runs again).
    if (added_lod_bias != 0.0f) {
        const Image* const image = image_view.BackingImage();
        key.drop_lod_bias = !VideoCommon::KeepOcarinaFsrLodBias(
            true, *ocarina_fsr_failed,
            image != nullptr && True(image->flags & ImageFlagBits::GpuModified));
    }
    if (needs_swizzle_mapping && !key.drop_custom_border && !image_view.HasIdentitySwizzle()) {
        const VkComponentMapping& mapping = image_view.Swizzle();
        key.swizzle = {mapping.r, mapping.g, mapping.b, mapping.a};
    }
    return key;
}

VkSampler Sampler::Find(const VariantKey& key) const noexcept {
    const auto it = std::ranges::find(variants, key, &Variant::key);
    if (it == variants.end()) {
        return VK_NULL_HANDLE;
    }
    return *it->sampler;
}

VkSampler Sampler::Emplace(VariantKey key) {
    bool custom_border = has_custom_border_colors && !key.drop_custom_border;
    if (custom_border && !custom_border_color_budget.TryAcquire(*device_ptr, 1)) {
        static bool warned_budget = false;
        if (!warned_budget) {
            warned_budget = true;
        }
        custom_border = false;
        key.drop_custom_border = true;
        key.swizzle = {};
        if (const VkSampler existing = Find(key); existing != VK_NULL_HANDLE) {
            return existing;
        }
    }

    std::array<float, 4> color = border_color;
    if (key.srgb_border) {
        color = srgb_border_color;
    }
    const VkSamplerCustomBorderColorCreateInfoEXT border_ci{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT,
        .pNext = nullptr,
        .customBorderColor = std::bit_cast<VkClearColorValue>(color),
        .format = VK_FORMAT_UNDEFINED,
    };
    const VkSamplerBorderColorComponentMappingCreateInfoEXT mapping_ci{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_BORDER_COLOR_COMPONENT_MAPPING_CREATE_INFO_EXT,
        .pNext = &border_ci,
        .components{key.swizzle[0], key.swizzle[1], key.swizzle[2], key.swizzle[3]},
        .srgb = VK_FALSE,
    };
    const void* chain = nullptr;
    if (custom_border) {
        chain = &border_ci;
        if (key.HasSwizzle()) {
            chain = &mapping_ci;
        }
    }
    const VkSamplerReductionModeCreateInfoEXT reduction_ci{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO_EXT,
        .pNext = chain,
        .reductionMode = reduction_mode,
    };
    if (has_minmax_reduction && !key.drop_reduction) {
        chain = &reduction_ci;
    }

    VkSamplerCreateInfo create_info = base_ci;
    if (key.drop_lod_bias) {
        create_info.mipLodBias -= added_lod_bias;
    }
    if (key.drop_compare) {
        create_info.compareEnable = VK_FALSE;
    }
    create_info.pNext = chain;
    if (key.force_nearest) {
        create_info.magFilter = VK_FILTER_NEAREST;
        create_info.minFilter = VK_FILTER_NEAREST;
        create_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        create_info.anisotropyEnable = VK_FALSE;
        create_info.maxAnisotropy = 1.0f;
    } else if (key.reduce_anisotropy) {
        create_info.anisotropyEnable = static_cast<VkBool32>(default_anisotropy > 1.0f);
        create_info.maxAnisotropy = default_anisotropy;
    }
    if (!custom_border) {
        create_info.borderColor = ConvertBorderColor(color);
    }
    variants.push_back(Variant{
        .key = key,
        .sampler = device_ptr->GetLogical().CreateSampler(create_info),
    });
    return *variants.back().sampler;
}

VkSampler Sampler::HandleFor(const ImageView& image_view, bool is_depth) {
    VariantKey key = MakeKey(image_view, is_depth);
    if (variants.size() >= MAX_VARIANTS) {
        key.srgb_border = false;
        key.swizzle = {};
    }
    VkSampler handle = Find(key);
    if (handle == VK_NULL_HANDLE) {
        handle = Emplace(key);
    }
    if (VideoCommon::RenderTrace::Enabled() && (is_depth || has_depth_comparison)) {
        static thread_local std::set<std::tuple<VkSampler, VkImage, s32, s32, s32, bool, bool>> seen;
        if (seen.size() < 4096 &&
            seen.emplace(handle, image_view.ImageHandle(), image_view.range.base.level,
                         image_view.range.base.layer, image_view.range.extent.layers,
                         is_depth, Settings::IsDockedMode()).second) {
            LOG_DEBUG(Render_Vulkan,
                      "Render trace depth sampler: docked={} address={:#x} format={} {}x{} "
                      "base_level={} base_layer={} layers={} shader_depth={} compare_enable={} compare_op={} "
                      "compare_supported={} min_filter={} mag_filter={} border={} address_modes={}/{}/{}",
                      Settings::IsDockedMode(), image_view.GpuAddr(), image_view.format,
                      image_view.size.width, image_view.size.height, image_view.range.base.level,
                      image_view.range.base.layer,
                      image_view.range.extent.layers, is_depth, base_ci.compareEnable,
                      static_cast<u32>(base_ci.compareOp), image_view.SupportsDepthComparison(),
                      static_cast<u32>(key.force_nearest ? VK_FILTER_NEAREST : base_ci.minFilter),
                      static_cast<u32>(key.force_nearest ? VK_FILTER_NEAREST : base_ci.magFilter),
                      static_cast<u32>(base_ci.borderColor), static_cast<u32>(base_ci.addressModeU),
                      static_cast<u32>(base_ci.addressModeV), static_cast<u32>(base_ci.addressModeW));
        }
    }
    return handle;
}

Framebuffer::Framebuffer(TextureCacheRuntime& runtime, std::span<ImageView*, NUM_RT> color_buffers,
                         ImageView* depth_buffer, const VideoCommon::RenderTargets& key)
    : render_area{VkExtent2D{
          .width = key.size.width,
          .height = key.size.height,
      }} {
    CreateFramebuffer(runtime, color_buffers, depth_buffer, key.is_rescaled);
    if (runtime.device.HasDebuggingToolAttached()) {
        framebuffer.SetObjectNameEXT(VideoCommon::Name(key).c_str());
    }
}

Framebuffer::Framebuffer(TextureCacheRuntime& runtime, ImageView* color_buffer,
                         ImageView* depth_buffer, VkExtent2D extent, bool is_rescaled_)
    : render_area{extent} {
    std::array<ImageView*, NUM_RT> color_buffers{color_buffer};
    CreateFramebuffer(runtime, color_buffers, depth_buffer, is_rescaled_);
}

Framebuffer::~Framebuffer() = default;

void Framebuffer::CreateFramebuffer(TextureCacheRuntime& runtime,
                                    std::span<ImageView*, NUM_RT> color_buffers,
                                    ImageView* depth_buffer, bool is_rescaled_) {
    boost::container::small_vector<VkImageView, NUM_RT * 2 + 2> attachments;
    RenderPassKey renderpass_key{};
    // A framebuffer can't have more layers than any of its attachments: one with more layers
    // than a color target makes the driver write the extra layers past that image's memory.
    s32 num_layers = (std::numeric_limits<s32>::max)();

    is_rescaled = is_rescaled_;
    const auto& resolution = runtime.resolution;

    const auto max_extent = runtime.device.GetMaxFramebufferExtent();
    u32 width = max_extent.width;
    u32 height = max_extent.height;
    for (size_t index = 0; index < NUM_RT; ++index) {
        const ImageView* const color_buffer = color_buffers[index];
        if (!color_buffer) {
            renderpass_key.color_formats[index] = PixelFormat::Invalid;
            continue;
        }
        width = (std::min)(width, is_rescaled ? resolution.ScaleUp(color_buffer->size.width)
                                            : color_buffer->size.width);
        height = (std::min)(height, is_rescaled ? resolution.ScaleUp(color_buffer->size.height)
                                              : color_buffer->size.height);
        attachments.push_back(color_buffer->RenderTarget());
        renderpass_key.color_formats[index] = color_buffer->format;
        num_layers = (std::min)(num_layers, color_buffer->range.extent.layers);
        images[num_images] = color_buffer->ImageHandle();
        image_ranges[num_images] = MakeSubresourceRange(color_buffer);
        rt_map[index] = num_images;
        samples = color_buffer->Samples();
        ++num_images;
    }
    const size_t num_colors = attachments.size();
    VkImage depth_image = VK_NULL_HANDLE;
    VkImageAspectFlags depth_aspect_mask = 0;
    if (depth_buffer) {
        width = (std::min)(width, is_rescaled ? resolution.ScaleUp(depth_buffer->size.width)
                                            : depth_buffer->size.width);
        height = (std::min)(height, is_rescaled ? resolution.ScaleUp(depth_buffer->size.height)
                                              : depth_buffer->size.height);
        attachments.push_back(depth_buffer->RenderTarget());
        renderpass_key.depth_format = depth_buffer->format;
        num_layers = (std::min)(num_layers, depth_buffer->range.extent.layers);
        images[num_images] = depth_buffer->ImageHandle();
        const VkImageSubresourceRange subresource_range = MakeSubresourceRange(depth_buffer);
        image_ranges[num_images] = subresource_range;
        samples = depth_buffer->Samples();
        ++num_images;
        has_depth = (subresource_range.aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) != 0;
        has_stencil = (subresource_range.aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT) != 0;
        depth_image = depth_buffer->ImageHandle();
        depth_aspect_mask = subresource_range.aspectMask;
    } else {
        renderpass_key.depth_format = PixelFormat::Invalid;
    }
    if (num_layers == (std::numeric_limits<s32>::max)()) {
        num_layers = 1;
    }
    layers = static_cast<u32>((std::max)(num_layers, 1));
    LOG_DEBUG(Render_Vulkan,
              "Framebuffer layers: layers={}, color0_layers={}, depth_layers={}, attachments={}",
              layers, color_buffers[0] ? color_buffers[0]->range.extent.layers : 0,
              depth_buffer ? depth_buffer->range.extent.layers : 0, attachments.size());
    renderpass_key.samples = samples;
    const bool do_resolve_color = ENABLE_MSAA_TILER_RESOLVE &&
        samples != VK_SAMPLE_COUNT_1_BIT && num_colors > 0 && runtime.device.IsTiler();
    renderpass_key.resolve_color = do_resolve_color;

    const bool do_resolve_depth_stencil = ENABLE_MSAA_TILER_RESOLVE &&
        samples != VK_SAMPLE_COUNT_1_BIT && depth_image != VK_NULL_HANDLE &&
        runtime.device.IsTiler() &&
        SupportsDepthStencilResolve(runtime.device, renderpass_key.depth_format);
    renderpass_key.resolve_depth_stencil = do_resolve_depth_stencil;

    discard_msaa_color = ENABLE_MSAA_RESOLVE_CONSUME && MsaaDiscardEnabled() && do_resolve_color;
    discard_msaa_depth_stencil =
        ENABLE_MSAA_RESOLVE_CONSUME && MsaaDiscardEnabled() && do_resolve_depth_stencil;

    renderpass = runtime.render_pass_cache.Get(renderpass_key);
    render_pass_key = renderpass_key;
    render_pass_cache = &runtime.render_pass_cache;
    runtime_ptr = &runtime;
    render_area.width = (std::min)(render_area.width, width);
    render_area.height = (std::min)(render_area.height, height);
    const FramebufferSize size = ClampFramebufferSize(
        {render_area.width, render_area.height, layers},
        {runtime.device.GetMaxFramebufferWidth(), runtime.device.GetMaxFramebufferHeight(),
         runtime.device.GetMaxFramebufferLayers()});
    render_area = VkExtent2D{.width = size.width, .height = size.height};
    layers = size.layers;

    if (do_resolve_color) {
        for (size_t index = 0; index < NUM_RT; ++index) {
            const PixelFormat format = renderpass_key.color_formats[index];
            if (format == PixelFormat::Invalid) {
                continue;
            }
            const VkFormat vk_format =
                MaxwellToVK::SurfaceFormat(runtime.device, FormatType::Optimal, true, format).format;
            const VkImage msaa_image = images[rt_map[index]];
            attachments.push_back(runtime.GetOrCreateResolveShadow(
                msaa_image, vk_format, render_area, layers, VK_IMAGE_ASPECT_COLOR_BIT));
            resolve_shadow_images[num_resolve_shadows++] = msaa_image;
        }
    }

    if (do_resolve_depth_stencil) {
        const VkFormat vk_format =
            MaxwellToVK::SurfaceFormat(runtime.device, FormatType::Optimal, true,
                                       renderpass_key.depth_format)
                .format;
        attachments.push_back(runtime.GetOrCreateResolveShadow(depth_image, vk_format, render_area,
                                                               layers, depth_aspect_mask));
        resolve_shadow_images[num_resolve_shadows++] = depth_image;
    }

    num_color_buffers = static_cast<u32>(num_colors);
    num_attachment_views = static_cast<u32>(attachments.size());
    std::ranges::copy(attachments, attachment_views.begin());
    framebuffer = runtime.device.GetLogical().CreateFramebuffer({
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .renderPass = renderpass,
        .attachmentCount = static_cast<u32>(attachments.size()),
        .pAttachments = attachments.data(),
        .width = render_area.width,
        .height = render_area.height,
        .layers = layers,
    });
}

void Framebuffer::MarkResolveShadowsUpToDate() const {
    if (runtime_ptr == nullptr) {
        return;
    }
    for (u32 index = 0; index < num_resolve_shadows; ++index) {
        runtime_ptr->MarkResolveShadowUpToDate(resolve_shadow_images[index]);
    }
}

VkRenderPass Framebuffer::RenderPassVariant(u32 color_clear_mask, bool depth_stencil_clear,
                                            u32 color_discard_mask,
                                            bool depth_stencil_discard) const {
    if (color_clear_mask == 0 && !depth_stencil_clear && color_discard_mask == 0 &&
        !depth_stencil_discard) {
        return renderpass;
    }
    static_assert(NUM_RT <= 8);
    const u32 variant_key = color_clear_mask | (color_discard_mask << 8) |
                            (static_cast<u32>(depth_stencil_clear) << 16) |
                            (static_cast<u32>(depth_stencil_discard) << 17);
    for (u32 index = 0; index < num_memoized_variants; ++index) {
        if (variant_keys[index] == variant_key) {
            return variant_render_passes[index];
        }
    }
    RenderPassKey key = render_pass_key;
    key.color_clear_mask = color_clear_mask;
    key.depth_stencil_clear = depth_stencil_clear;
    key.color_discard_mask = color_discard_mask;
    key.depth_stencil_discard = depth_stencil_discard;
    const VkRenderPass variant = render_pass_cache->Get(key);
    if (num_memoized_variants < variant_keys.size()) {
        variant_keys[num_memoized_variants] = variant_key;
        variant_render_passes[num_memoized_variants] = variant;
        ++num_memoized_variants;
    }
    return variant;
}

void TextureCacheRuntime::AccelerateImageUpload(
    Image& image, const StagingBufferRef& map,
    std::span<const VideoCommon::SwizzleParameters> swizzles,
    u32 z_start, u32 z_count) {

    if (IsPixelFormatASTC(image.info.format)) {
        if (bc_encoder_pass && WillUseGpuBcnTranscode(device, image.info)) {
            return TranscodeAstcToBcn(image, map, swizzles);
        }
        return astc_decoder_pass->Assemble(image, map, swizzles);
    }

    if (!Settings::values.gpu_unswizzle_enabled.GetValue() || !bl3d_unswizzle_pass) {
        if (IsPixelFormatBCn(image.info.format) && image.info.type == ImageType::e3D) {
            ASSERT(false && "GPU unswizzle is disabled for BCn 3D texture");
        }
        ASSERT(false);
        return;
    }

    if (bl3d_unswizzle_pass && IsPixelFormatBCn(image.info.format) && image.info.type == ImageType::e3D && image.info.resources.levels == 1 && image.info.resources.layers == 1) {
        return bl3d_unswizzle_pass->Unswizzle(image, map, swizzles, z_start, z_count);
    }

    ASSERT(false);
}

void TextureCacheRuntime::AccelerateImageUploadLayers(
    Image& image, const StagingBufferRef& map,
    std::span<const VideoCommon::SwizzleParameters> swizzles, std::span<const u32> layers) {
    astc_decoder_pass->AssembleLayers(image, map, swizzles, layers);
}

void TextureCacheRuntime::TranscodeAstcToBcn(
    Image& image, const StagingBufferRef& map,
    std::span<const VideoCommon::SwizzleParameters> swizzles) {
    const ImageInfo& info = image.info;
    const u32 width = info.size.width;
    const u32 height = info.size.height;
    const u32 levels = static_cast<u32>(info.resources.levels);
    const u32 layers = static_cast<u32>(info.resources.layers);
    // The BC encoder declares layout(rgba8). Identical byte layout is not enough:
    // its storage image view must have the matching R8G8B8A8_UNORM format.
    constexpr VkFormat DecodeFormat = VK_FORMAT_R8G8B8A8_UNORM;
    static const bool bucket_scratch = [] {
        const char* value = std::getenv("EDEN_NX2_ASTC_SCRATCH_BUCKET");
        return value == nullptr || value[0] != '0';
    }();
    static const bool recycle_on_lookup = [] {
        const char* value = std::getenv("EDEN_NX2_ASTC_POOL_V2");
        return value == nullptr || value[0] != '0';
    }();
    static const bool allow_larger = [] {
        const char* value = std::getenv("EDEN_NX2_ASTC_POOL_FIT");
        return value == nullptr || value[0] != '0';
    }();
    constexpr u32 ScratchAlignment = 64;
    const u32 scratch_width = bucket_scratch ? Common::AlignUp(width, ScratchAlignment) : width;
    const u32 scratch_height = bucket_scratch ? Common::AlignUp(height, ScratchAlignment) : height;
    if (recycle_on_lookup) {
        RecycleFinishedTranscodes();
    }
    const auto fits = [&](const PendingTranscode& t) {
        if (t.levels != levels || t.layers != layers) {
            return false;
        }
        if (t.width == scratch_width && t.height == scratch_height) {
            return true;
        }
        return allow_larger && t.width >= scratch_width && t.height >= scratch_height &&
               t.width <= scratch_width * 2 && t.height <= scratch_height * 2;
    };
    const auto reusable = std::ranges::min_element(
        free_transcodes, [&](const PendingTranscode& a, const PendingTranscode& b) {
            const bool fits_a = fits(a), fits_b = fits(b);
            if (fits_a != fits_b) {
                return fits_a;
            }
            return u64{a.width} * a.height < u64{b.width} * b.height;
        });
    PendingTranscode pending;
    if (reusable != free_transcodes.end() && fits(*reusable)) {
        pending = std::move(*reusable);
        free_transcodes.erase(reusable);
    } else {
        const VkImageCreateInfo decode_ci{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = DecodeFormat,
            .extent = {scratch_width, scratch_height, 1},
            .mipLevels = levels,
            .arrayLayers = layers,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_STORAGE_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        const MemoryTagScope memory_tag{MemoryTag::TextureHelper};
        vk::Image decode = memory_allocator.TryCreateImage(decode_ci, true);
        if (!decode && !free_transcodes.empty()) {
            // The GPU is done with these: their memory goes back to the driver now.
            free_transcodes.clear();
            decode = memory_allocator.TryCreateImage(decode_ci, true);
        }
        if (!decode) {
            static u32 logged{};
            if (logged++ < 32) {
                LOG_ERROR(Render_Vulkan,
                          "No memory to decode ASTC image {:#x} ({}x{}, {} levels); uploading it "
                          "again when it is next used",
                          image.gpu_addr, width, height, levels);
            }
            image.flags |= VideoCommon::ImageFlagBits::CpuModified;
            TransitionImageLayout(image);
            return;
        }
        pending = PendingTranscode{
            .tick = 0,
            .image = std::move(decode),
            .views = {},
            .width = scratch_width,
            .height = scratch_height,
            .levels = levels,
            .layers = layers,
            .unused_frames = 0,
            .last_use = 0,
        };
        for (u32 level = 0; level < levels; ++level) {
            pending.views.push_back(
                MakeStorageView(device.GetLogical(), level, *pending.image, DecodeFormat));
        }
    }
    static const bool one_submit = [] {
        const char* value = std::getenv("EDEN_NX2_ASTC_ONE_SUBMIT");
        return value == nullptr || value[0] != '0';
    }();
    astc_decoder_pass->AssembleInto(
        info, 0, image.guest_size_bytes,
        ASTCDecoderPass::Target{
            .image = *pending.image,
            .aspect_mask = VK_IMAGE_ASPECT_COLOR_BIT,
            .initialized = false,
            .level_view = [&pending](s32 level) { return *pending.views[level]; },
            .wait = !one_submit,
            .compute_only = true,
        },
        map, swizzles);

    // Encode every level into one buffer, then copy it into the image.
    const bool bc3 = !IsOpaqueBc1Astc(device, info) &&
                     Settings::GetAstcRecompression() != Settings::AstcRecompression::Bc1;
    const u64 block_bytes = bc3 ? 16 : 8;
    boost::container::small_vector<VideoCommon::BufferImageCopy, 16> copies;
    boost::container::small_vector<BcEncoderPass::LevelEncode, 16> encodes;
    copies.reserve(levels);
    encodes.reserve(levels);
    u64 total = 0;
    for (u32 level = 0; level < levels; ++level) {
        const u32 level_width = (std::max)(width >> level, 1U);
        const u32 level_height = (std::max)(height >> level, 1U);
        const u32 blocks_x = Common::DivCeil(level_width, 4U);
        const u32 blocks_y = Common::DivCeil(level_height, 4U);
        const u64 size = u64{blocks_x} * blocks_y * layers * block_bytes;
        copies.push_back({
            .buffer_offset = total,
            .buffer_size = size,
            .buffer_row_length = blocks_x * 4,
            .buffer_image_height = blocks_y * 4,
            .image_subresource{
                .base_level = static_cast<s32>(level),
                .base_layer = 0,
                .num_layers = static_cast<s32>(layers),
            },
            .image_offset{0, 0, 0},
            .image_extent{level_width, level_height, 1},
        });
        encodes.push_back({
            .source = *pending.views[level],
            .width = level_width,
            .height = level_height,
            .out_offset = total,
        });
        total += size;
    }
    const auto output = staging_buffer_pool.Request(total, MemoryUsage::DeviceLocal);
    bc_encoder_pass->EncodeLevels(std::span{encodes.data(), encodes.size()}, layers, bc3,
                                  output.buffer, output.offset, total, true);
    image.UploadMemory(output.buffer, output.offset, copies);
    pending.tick = scheduler.CurrentTick();
    pending.last_use = ++transcode_use_counter;
    pending_transcodes.push_back(std::move(pending));
}

void TextureCacheRuntime::TransitionImageLayout(Image& image) {
    if (!image.ExchangeInitialization()) {
        VkImageMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_NONE,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image.Handle(),
            .subresourceRange{
                .aspectMask = image.AspectMask(),
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        };
        scheduler.RequestOutsideRenderPassOperationContext();
        scheduler.Record([barrier](vk::CommandBuffer cmdbuf) {
            cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE,
                                   vk::PIPELINE_STAGE_GRAPHICS_COMPUTE, 0, barrier);
        });
    }
}

} // namespace Vulkan
