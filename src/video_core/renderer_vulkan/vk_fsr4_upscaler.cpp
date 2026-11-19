// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// AMD FSR 4 through AMD's FidelityFX runtime (Windows) or the fsr4-vk community Vulkan port
// (Linux). The runtime takes plain Vulkan images: the inputs in SHADER_READ_ONLY_OPTIMAL
// (FFX "compute read"), the output in GENERAL ("unordered access"), and records into our command
// buffer. Picking the FSR 4 provider by version follows CheezerWheezer's FSR 4 integration.

#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <boost/container/static_vector.hpp>
#include <fmt/format.h>

#include "common/logging.h"
#include "common/settings.h"
#include "video_core/renderer_vulkan/vk_fsr4_upscaler.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_upscaler_common.h"
#include "video_core/vulkan_common/ffx_runtime.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

// After vulkan.h, which sets the platform macros the wrapper's Win32 entry points need.
#ifndef _WIN32
#define __declspec(x)
#endif
#include "api/include/vk/ffx_api_vk.h"
#include "upscalers/include/ffx_upscale.h"
#ifndef _WIN32
#undef __declspec
#endif

namespace Vulkan {
namespace {

FfxApiResource Resource(VkImage image, VkFormat format, VkExtent2D size, u32 usage, u32 state) {
    FfxApiResource resource{};
    resource.resource = reinterpret_cast<void*>(image);
    resource.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    resource.description.format = ffxApiGetSurfaceFormatVK(format);
    resource.description.width = size.width;
    resource.description.height = size.height;
    resource.description.depth = 1;
    resource.description.mipCount = 1;
    resource.description.flags = 0;
    resource.description.usage = usage;
    resource.state = state;
    return resource;
}

void Message(uint32_t type, const wchar_t* message) {
    std::string text;
    for (const wchar_t* c = message; c && *c; ++c) {
        text.push_back(*c > 0 && *c < 128 ? static_cast<char>(*c) : '?');
    }
    if (type == FFX_API_MESSAGE_TYPE_ERROR) {
        LOG_ERROR(Render_Vulkan, "FSR 4: {}", text);
    } else {
        LOG_INFO(Render_Vulkan, "FSR 4: {}", text);
    }
}

VkDeviceContext DeviceContext(const Device& device) {
    return VkDeviceContext{
        .vkDevice = *device.GetLogical(),
        .vkPhysicalDevice = static_cast<VkPhysicalDevice>(device.GetPhysical()),
        .vkDeviceProcAddr = device.GetDispatchLoader().vkGetDeviceProcAddr,
    };
}

struct Provider {
    u64 id{};
    std::string name;
};

#if defined(__linux__) && !defined(__ANDROID__)
/// The fsr4-vk model CheckFsr4Port picked for the device's enabled features. fsr4-vk would pick
/// by what the GPU supports, which can include features this device doesn't enable.
std::optional<Provider> FindFsr4(const FfxRuntime::Api& api, const Device& device) {
    const VideoCommon::Fsr4PortSupport support =
        VideoCommon::CheckFsr4Port(device.Fsr4PortDevice());
    if (!support.model) {
        FfxRuntime::SetProblem(fmt::format("Unsupported on {}: {}; using FSR 3.1.",
                                           device.GetModelName(), support.reason));
        return std::nullopt;
    }
    const std::string_view suffix =
        *support.model == VideoCommon::Fsr4PortModel::Fp8 ? " FP8" : " INT8";
    VkDeviceContext vk_context = DeviceContext(device);
    u64 count = 0;
    ffxQueryDescGetVersions versions{};
    versions.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    versions.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    versions.device = &vk_context;
    versions.outputCount = &count;
    if (api.query(nullptr, &versions.header) != FFX_API_RETURN_OK || count == 0) {
        LOG_WARNING(Render_Vulkan, "FSR 4: fsr4-vk offers no upscaler; using FSR 3.1");
        FfxRuntime::SetProblem("fsr4-vk offers no upscaler for this GPU; using FSR 3.1.");
        return std::nullopt;
    }
    std::vector<u64> ids(count);
    std::vector<const char*> names(count);
    versions.versionIds = ids.data();
    versions.versionNames = names.data();
    if (api.query(nullptr, &versions.header) != FFX_API_RETURN_OK) {
        FfxRuntime::SetProblem("Failed to query fsr4-vk versions; using FSR 3.1.");
        return std::nullopt;
    }
    for (u64 i = 0; i < count; ++i) {
        const std::string_view name = names[i] ? names[i] : "";
        if (name.starts_with('4') && name.ends_with(suffix)) {
            LOG_INFO(Render_Vulkan,
                     "FSR 4 (fsr4-vk community Vulkan port): using its {} model on {}: {}", name,
                     device.GetModelName(), support.reason);
            return Provider{ids[i], std::string{name}};
        }
    }
    LOG_WARNING(Render_Vulkan, "FSR 4: fsr4-vk lacks its{} model; using FSR 3.1", suffix);
    FfxRuntime::SetProblem(fmt::format("fsr4-vk lacks its{} model; using FSR 3.1.", suffix));
    return std::nullopt;
}
#else
/// The newest FSR 4 (or later) upscaler AMD's runtime offers for this device. The runtime also
/// offers FSR 3.1 on GPUs without FSR 4; that one is left to the built-in FSR 3.1.
std::optional<Provider> FindFsr4(const FfxRuntime::Api& api, const Device& device) {
    VkDeviceContext vk_context = DeviceContext(device);
    u64 count = 0;
    ffxQueryDescGetVersions versions{};
    versions.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    versions.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    versions.device = &vk_context;
    versions.outputCount = &count;
    if (api.query(nullptr, &versions.header) != FFX_API_RETURN_OK || count == 0) {
        LOG_INFO(Render_Vulkan, "AMD FSR 4: the FidelityFX runtime offers no upscaler for {}",
                 device.GetModelName());
        FfxRuntime::SetProblem(fmt::format(
            "AMD's FidelityFX runtime offers no upscaler for {}; using FSR 3.1.",
            device.GetModelName()));
        return std::nullopt;
    }
    std::vector<u64> ids(count);
    std::vector<const char*> names(count);
    versions.versionIds = ids.data();
    versions.versionNames = names.data();
    if (api.query(nullptr, &versions.header) != FFX_API_RETURN_OK) {
        FfxRuntime::SetProblem("Failed to query AMD's FidelityFX runtime; using FSR 3.1.");
        return std::nullopt;
    }
    std::string offered;
    std::optional<Provider> fsr4;
    for (u64 i = 0; i < count; ++i) {
        const std::string name = names[i] ? names[i] : "";
        offered += (offered.empty() ? "" : ", ") + name;
        // Newest first.
        if (!fsr4 && !name.empty() && name[0] >= '4' && name[0] <= '9') {
            fsr4 = Provider{ids[i], name};
        }
    }
    LOG_INFO(Render_Vulkan, "AMD FSR 4: upscalers offered for {}: {}{}", device.GetModelName(),
             offered, fsr4 ? "" : " (no FSR 4; using FSR 3.1)");
    if (!fsr4) {
        FfxRuntime::SetProblem(fmt::format(
            "AMD's driver offers no FSR 4 upscaler for {}; using FSR 3.1.",
            device.GetModelName()));
    }
    return fsr4;
}
#endif
} // Anonymous namespace

bool Fsr4Upscaler::IsWanted(const Device& device) {
    if (!Settings::AmdFsr4Chosen()) {
        return false;
    }
#if defined(__linux__) && !defined(__ANDROID__)
    const VideoCommon::Fsr4PortSupport support =
        VideoCommon::CheckFsr4Port(device.Fsr4PortDevice());
    if (!support.model) {
        LOG_INFO(Render_Vulkan,
                 "FSR 4 (fsr4-vk community Vulkan port) can't run on {} ({}): {}; using FSR 3.1",
                 device.GetModelName(), device.GetDriverName(), support.reason);
        FfxRuntime::SetProblem(fmt::format("Unsupported on {} ({}): {}; using FSR 3.1.",
                                           device.GetModelName(), device.GetDriverName(),
                                           support.reason));
    }
    return support.model.has_value();
#elif defined(_WIN32)
    if (device.GetDriverID() != VK_DRIVER_ID_AMD_PROPRIETARY) {
        FfxRuntime::SetProblem(
            fmt::format("AMD FSR 4 needs an AMD Radeon GPU on AMD's Windows driver (found {}); "
                        "using FSR 3.1.",
                        device.GetModelName()));
        return false;
    }
    return true;
#else
    FfxRuntime::SetProblem("AMD FSR 4 is supported on Windows and Linux only; using FSR 3.1.");
    return false;
#endif
}

Fsr4Upscaler::Fsr4Upscaler(const Device& device_, Scheduler& scheduler_,
                           MemoryAllocator& memory_allocator_, Profile profile_)
    : device{device_}, scheduler{scheduler_}, memory_allocator{memory_allocator_},
      profile{profile_}, api{FfxRuntime::Acquire()} {
    const auto provider = api ? FindFsr4(*api, device) : std::nullopt;
    if (!provider) {
        throw vk::Exception(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    provider_id = provider->id;
    provider_name = provider->name;
    FfxRuntime::SetAvailable(true, provider_name);
    point_sampler = CreateUpscalerSampler(device, VK_FILTER_NEAREST);
    if (profile == Profile::OcarinaLdr) {
        CreateExposure();
    }
}

Fsr4Upscaler::~Fsr4Upscaler() {
    DestroyContext();
}

void Fsr4Upscaler::DestroyContext() {
    if (context == nullptr) {
        return;
    }
    scheduler.Finish(); // recorded dispatches may still use it
    ffxContext ctx = context;
    api->destroy_context(&ctx, nullptr);
    context = nullptr;
}

bool Fsr4Upscaler::CreateContext(VkExtent2D max_render_size, VkExtent2D upscale_size,
                                 bool dynamic_resolution) {
    DestroyContext();
    ffxCreateBackendVKDesc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    backend.header.pNext = nullptr;
    backend.deviceContext = DeviceContext(device);
    ffxOverrideVersion version_override{};
    version_override.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
    version_override.header.pNext = &backend.header;
    version_override.versionId = provider_id;
    ffxCreateContextDescUpscaleVersion api_version{};
    api_version.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;
    api_version.header.pNext = &version_override.header;
    api_version.version = FFX_UPSCALER_VERSION;

    ffxCreateContextDescUpscale create{};
    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    create.header.pNext = &api_version.header;
    u32 flags = dynamic_resolution ? u32{FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION} : 0u;
    if (profile == Profile::HdrReversedDepth) {
        flags |= FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_DEPTH_INVERTED |
                 FFX_UPSCALE_ENABLE_DEPTH_INFINITE | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
    }
    create.flags = flags;
    create.maxRenderSize = {max_render_size.width, max_render_size.height};
    create.maxUpscaleSize = {upscale_size.width, upscale_size.height};
    create.fpMessage = Message;
    ffxContext ctx{};
    const ffxReturnCode_t result = api->create_context(&ctx, &create.header, nullptr);
    if (result != FFX_API_RETURN_OK) {
        LOG_ERROR(Render_Vulkan, "FSR 4: ffxCreateContext {}x{} -> {}x{} failed ({}); using FSR 3.1",
                  max_render_size.width, max_render_size.height, upscale_size.width,
                  upscale_size.height, result);
        FfxRuntime::SetProblem(
            fmt::format("FSR 4 context creation failed (code {}); FSR 3.1 took over.", result));
        return false;
    }
    context = ctx;
    context_render_capacity = max_render_size;
    upscale_extent = upscale_size;
    context_dynamic_resolution = dynamic_resolution;
    FfxRuntime::SetAvailable(true, provider_name);
#if defined(__linux__) && !defined(__ANDROID__)
    LOG_INFO(Render_Vulkan, "FSR 4: fsr4-vk {} context {}x{} -> {}x{}{}{}", provider_name,
             max_render_size.width, max_render_size.height, upscale_size.width, upscale_size.height,
             profile == Profile::HdrReversedDepth ? " (HDR)" : "",
             dynamic_resolution ? " (dynamic render size)" : "");
#else
    LOG_INFO(Render_Vulkan, "FSR 4: AMD FSR {} context {}x{} -> {}x{}{}{}", provider_name,
             max_render_size.width, max_render_size.height, upscale_size.width, upscale_size.height,
             profile == Profile::HdrReversedDepth ? " (HDR)" : "",
             dynamic_resolution ? " (dynamic render size)" : "");
#endif
    return true;
}

void Fsr4Upscaler::CreateExposure() {
    const MemoryTagScope memory_tag{MemoryTag::Upscaler};
    exposure_image = memory_allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R32_SFLOAT,
        .extent = {1, 1, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([image = *exposure_image](vk::CommandBuffer cmdbuf) {
        cmdbuf.PipelineBarrier(
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
            MakeUpscalerImageBarrier(image, UPSCALER_COLOR_RANGE, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                                     VK_IMAGE_LAYOUT_UNDEFINED,
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL));
        VkClearColorValue one{};
        one.float32[0] = 1.0f;
        cmdbuf.ClearColorImage(image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, one,
                               UPSCALER_COLOR_RANGE);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                               MakeUpscalerImageBarrier(image, UPSCALER_COLOR_RANGE,
                                                        VK_ACCESS_TRANSFER_WRITE_BIT,
                                                        VK_ACCESS_SHADER_READ_BIT,
                                                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
    });
}

void Fsr4Upscaler::CreateOutput(VkExtent2D upscale_size) {
    CreateUpscalerOutput(device, memory_allocator, scheduler, upscale_size, output_image,
                         output_view);
}

bool Fsr4Upscaler::Upscale(const Inputs& in) {
    if (failed.load()) {
        return false;
    }
    const auto render_capacity = in.max_render_size.value_or(in.render_size);
    const bool dynamic_resolution = in.max_render_size.has_value();
    const bool new_context = context == nullptr ||
                             render_capacity.width != context_render_capacity.width ||
                             render_capacity.height != context_render_capacity.height ||
                             in.upscale_size.width != upscale_extent.width ||
                             in.upscale_size.height != upscale_extent.height ||
                             dynamic_resolution != context_dynamic_resolution;
    if (new_context) {
        if (!CreateContext(render_capacity, in.upscale_size, dynamic_resolution)) {
            failed = true;
            return false;
        }
        CreateOutput(in.upscale_size);
    }

    const auto image_extent = [&](const Image& img) {
        return img.extent.width != 0 && img.extent.height != 0 ? img.extent : in.render_size;
    };
    ffxDispatchDescUpscale dispatch{};
    dispatch.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    dispatch.header.pNext = nullptr;
    dispatch.color = Resource(in.color.image, in.color.format, image_extent(in.color), 0,
                              FFX_API_RESOURCE_STATE_COMPUTE_READ);
    dispatch.depth = Resource(in.depth.image, in.depth.format, image_extent(in.depth),
                              FFX_API_RESOURCE_USAGE_DEPTHTARGET,
                              FFX_API_RESOURCE_STATE_COMPUTE_READ);
    dispatch.motionVectors = Resource(in.motion_vectors.image, in.motion_vectors.format,
                                      image_extent(in.motion_vectors), 0,
                                      FFX_API_RESOURCE_STATE_COMPUTE_READ);
    if (profile == Profile::OcarinaLdr && exposure_image) {
        dispatch.exposure = Resource(*exposure_image, VK_FORMAT_R32_SFLOAT, {1, 1}, 0,
                                     FFX_API_RESOURCE_STATE_COMPUTE_READ);
    }
    if (in.reactive) {
        dispatch.reactive = Resource(in.reactive->image, in.reactive->format,
                                     image_extent(*in.reactive), 0,
                                     FFX_API_RESOURCE_STATE_COMPUTE_READ);
    }
    dispatch.output = Resource(*output_image, UPSCALER_OUTPUT_FORMAT, in.upscale_size,
                               FFX_API_RESOURCE_USAGE_UAV, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
    dispatch.jitterOffset = {in.jitter[0], in.jitter[1]};
    dispatch.motionVectorScale = {in.motion_vector_scale[0], in.motion_vector_scale[1]};
    dispatch.renderSize = {in.render_size.width, in.render_size.height};
    dispatch.upscaleSize = {in.upscale_size.width, in.upscale_size.height};
    dispatch.enableSharpening = in.sharpen;
    dispatch.sharpness = in.sharpness;
    dispatch.frameTimeDelta = in.frame_time_ms;
    dispatch.preExposure = 1.0f;
    dispatch.reset = in.reset || new_context;
    if (profile == Profile::HdrReversedDepth) {
        dispatch.cameraNear = (std::numeric_limits<f32>::max)();
        dispatch.cameraFar = in.camera_near;
    } else {
        dispatch.cameraNear = in.camera_near;
        dispatch.cameraFar = in.camera_far;
    }
    dispatch.cameraFovAngleVertical = in.vertical_fov;
    dispatch.viewSpaceToMetersFactor = 1.0f;
    dispatch.flags = 0;

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([this, in, dispatch,
                      output = *output_image](vk::CommandBuffer cmdbuf) {
        // The inputs stay in GENERAL otherwise; FSR 4 reads them as sampled images.
        boost::container::static_vector<VkImageMemoryBarrier, 5> to_read{
            MakeUpscalerInputToReadBarrier(in.color),
            MakeUpscalerInputToReadBarrier(in.depth),
            MakeUpscalerInputToReadBarrier(in.motion_vectors),
            MakeUpscalerImageBarrier(output, UPSCALER_COLOR_RANGE, VK_ACCESS_SHADER_READ_BIT,
                                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                     VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL),
        };
        if (in.reactive) {
            to_read.push_back(MakeUpscalerInputToReadBarrier(*in.reactive));
        }
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, {}, {}, to_read);

        ffxDispatchDescUpscale recorded = dispatch;
        recorded.commandList = static_cast<void*>(*cmdbuf);
        ffxContext ctx = context;
        const ffxReturnCode_t result = api->dispatch(&ctx, &recorded.header);
        if (result != FFX_API_RETURN_OK) {
            if (!failed.exchange(true)) {
                LOG_ERROR(Render_Vulkan, "FSR 4: ffxDispatch failed ({}); using FSR 3.1", result);
                FfxRuntime::SetProblem(
                    fmt::format("FSR 4 dispatch failed (code {}); FSR 3.1 took over.", result));
            }
        } else {
            FfxRuntime::NoteEvaluation();
        }

        boost::container::static_vector<VkImageMemoryBarrier, 5> back{
            MakeUpscalerInputToGeneralBarrier(in.color),
            MakeUpscalerInputToGeneralBarrier(in.depth),
            MakeUpscalerInputToGeneralBarrier(in.motion_vectors),
            MakeUpscalerImageBarrier(output, UPSCALER_COLOR_RANGE, VK_ACCESS_SHADER_WRITE_BIT,
                                     VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                                     VK_IMAGE_LAYOUT_GENERAL),
        };
        if (in.reactive) {
            back.push_back(MakeUpscalerInputToGeneralBarrier(*in.reactive));
        }
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, {}, {}, back);
    });
    scheduler.InvalidateState();
    return true;
}

} // namespace Vulkan
