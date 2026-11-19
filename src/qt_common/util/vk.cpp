// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <utility>
#include <vector>

#include "qt_common/qt_common.h"

#include "common/dynamic_library.h"
#include "common/logging.h"
#include "qt_common/util/vk.h"
#include "video_core/fsr4_support.h"
#include "video_core/vulkan_common/ffx_files.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_instance.h"
#include "video_core/vulkan_common/vulkan_library.h"
#include "video_core/vulkan_common/vulkan_surface.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "vulkan/vulkan_core.h"

class QWindow;

namespace VkDeviceInfo {
static_assert(Settings::DeviceProfile::VkDriverAmdProprietary == VK_DRIVER_ID_AMD_PROPRIETARY);
static_assert(Settings::DeviceProfile::VkDriverAmdOpenSource == VK_DRIVER_ID_AMD_OPEN_SOURCE);
static_assert(Settings::DeviceProfile::VkDriverMesaRadv == VK_DRIVER_ID_MESA_RADV);
static_assert(Settings::DeviceProfile::VkDriverNvidiaProprietary ==
              VK_DRIVER_ID_NVIDIA_PROPRIETARY);
static_assert(Settings::DeviceProfile::VkDriverIntelProprietaryWindows ==
              VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS);
static_assert(Settings::DeviceProfile::VkDriverIntelOpenSourceMesa ==
              VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA);
static_assert(Settings::DeviceProfile::VkDriverQualcommProprietary ==
              VK_DRIVER_ID_QUALCOMM_PROPRIETARY);
static_assert(Settings::DeviceProfile::VkDriverMoltenVk == VK_DRIVER_ID_MOLTENVK);
static_assert(Settings::DeviceProfile::VkDriverMesaTurnip == VK_DRIVER_ID_MESA_TURNIP);

Record::Record(std::string_view name_, const std::vector<VkPresentModeKHR>& vsync_modes_,
               bool has_broken_compute_, VkDriverId driver_id_, bool dlss_capable_,
               std::string_view driver_version_, bool integrated_, u64 vram_mib_,
               Capabilities capabilities_, bool fsr4_capable_)
    : name{name_}, vsync_support{vsync_modes_}, has_broken_compute{has_broken_compute_},
      driver_id{driver_id_}, dlss_capable{dlss_capable_}, fsr4_capable{fsr4_capable_},
      driver_version{driver_version_}, integrated{integrated_}, vram_mib{vram_mib_},
      vendor_id{capabilities_.vendor_id}, capabilities{std::move(capabilities_)} {}

Record::~Record() = default;

void PopulateRecords(std::vector<Record>& records, QWindow* window) try {
    using namespace Vulkan;

    // Create a test window with a Vulkan surface type for checking present modes.
    QWindow test_window(window);
    test_window.setSurfaceType(QWindow::VulkanSurface);
    test_window.create();
    auto wsi = QtCommon::GetWindowSystemInfo(&test_window);

    vk::InstanceDispatch dld;
    const auto library = OpenLibrary();
    const vk::Instance instance = CreateInstance(*library, dld, VK_API_VERSION_1_1, wsi.type);
    const std::vector<VkPhysicalDevice> physical_devices = instance.EnumeratePhysicalDevices();
    vk::SurfaceKHR surface;
    try {
        surface = CreateSurface(instance, wsi);
    } catch (const Vulkan::vk::Exception& exception) {
        LOG_WARNING(Frontend, "Could not create test Vulkan surface: {}", exception.what());
    }

    records.clear();
    records.reserve(physical_devices.size());
    for (const VkPhysicalDevice device : physical_devices) {
        try {
            const auto physical_device = vk::PhysicalDevice(device, dld);
            const VkPhysicalDeviceProperties base_props = physical_device.GetProperties();
            const std::string raw_name = base_props.deviceName;

            std::vector<VkPresentModeKHR> present_modes = default_present_modes;
            if (*surface) {
                try {
                    const auto modes = physical_device.GetSurfacePresentModesKHR(*surface);
                    if (!modes.empty()) {
                        present_modes = modes;
                    }
                } catch (const Vulkan::vk::Exception& exception) {
                    LOG_WARNING(Frontend, "Could not query present modes of {}: {}", raw_name,
                                exception.what());
                }
            }

            VkPhysicalDeviceDriverProperties driver_properties{};
            driver_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
            driver_properties.pNext = nullptr;
            VkPhysicalDeviceProperties2 properties{};
            properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
            properties.pNext = &driver_properties;
            properties.properties = base_props;
            if (base_props.apiVersion >= VK_API_VERSION_1_1 &&
                dld.vkGetPhysicalDeviceProperties2 != nullptr) {
                dld.vkGetPhysicalDeviceProperties2(physical_device, &properties);
            }

            const auto driverID = driver_properties.driverID;
            const bool has_broken_compute{
                Vulkan::Device::CheckBrokenCompute(driverID, properties.properties.driverVersion)};

            Capabilities capabilities;
            capabilities.vendor_id = properties.properties.vendorID;
            capabilities.driver_id = static_cast<u32>(driverID);
            try {
                for (const VkExtensionProperties& extension :
                     physical_device.EnumerateDeviceExtensionProperties()) {
                    capabilities.extensions.emplace_back(extension.extensionName);
                }
            } catch (const Vulkan::vk::Exception& exception) {
                LOG_WARNING(Frontend, "Could not list the extensions of {}: {}", raw_name,
                            exception.what());
            }

            const bool dlss_capable = Settings::DeviceProfile::IsNvidiaDlssCapable(
                static_cast<u32>(driverID), raw_name, capabilities.extensions);

#if defined(_WIN32)
            const bool fsr4_capable = VideoCommon::IsFsr4CapableGpu(
                properties.properties.vendorID, driverID == VK_DRIVER_ID_AMD_PROPRIETARY, raw_name);
#elif defined(__linux__) && !defined(__ANDROID__)
            VkPhysicalDeviceShaderIntegerDotProductProperties dot_properties{};
            dot_properties.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES;
            if (properties.properties.apiVersion >= VK_API_VERSION_1_3) {
                VkPhysicalDeviceProperties2 dot_query{};
                dot_query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
                dot_query.pNext = &dot_properties;
                dld.vkGetPhysicalDeviceProperties2(physical_device, &dot_query);
            }
            const bool fsr4_capable =
                !Vulkan::FfxRuntime::PortLibraryPath().empty() &&
                VideoCommon::IsFsr4PortGpu(
                    properties.properties.vendorID, properties.properties.deviceID,
                    driverID == VK_DRIVER_ID_MESA_RADV,
                    dot_properties.integerDotProduct4x8BitPackedMixedSignednessAccelerated !=
                        VK_FALSE);
#else
            constexpr bool fsr4_capable = false;
#endif

            std::string driver_string = Vulkan::vk::GetDriverName(driver_properties);
            if (driver_string.empty()) {
                driver_string = "Unknown";
            }
            const std::string name = fmt::format("{} ({})", raw_name, driver_string);

            const auto memory = physical_device.GetMemoryProperties().memoryProperties;
            u64 vram = 0;
            for (u32 i = 0; i < memory.memoryHeapCount; ++i) {
                if ((memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                    vram = (std::max)(vram, u64{memory.memoryHeaps[i].size});
                }
            }
            const bool integrated =
                properties.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;

            records.push_back(VkDeviceInfo::Record(
                name, present_modes, has_broken_compute, driverID, dlss_capable,
                static_cast<const char*>(driver_properties.driverInfo), integrated, vram >> 20,
                std::move(capabilities), fsr4_capable));
        } catch (const Vulkan::vk::Exception& exception) {
            LOG_WARNING(Frontend, "Could not query Vulkan physical device {}: {}", records.size(),
                        exception.what());
            records.push_back(VkDeviceInfo::Record("Unknown Vulkan Device", default_present_modes,
                                                   false, VkDriverId{}, false));
        }
    }
} catch (const Vulkan::vk::Exception& exception) {
    LOG_ERROR(Frontend, "Failed to enumerate devices with error: {}", exception.what());
}

} // namespace VkDeviceInfo
