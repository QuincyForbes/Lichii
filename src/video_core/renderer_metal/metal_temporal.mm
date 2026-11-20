// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>

#include <algorithm>

#include "common/logging.h"
#include "video_core/renderer_metal/metal_bridge.h"
#include "video_core/renderer_metal/metal_device.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_temporal.h"

namespace Metal {

namespace {

constexpr MTLPixelFormat OutputFormat = MTLPixelFormatRGBA16Float;

struct ScalerKey {
    MTLPixelFormat color{};
    MTLPixelFormat depth{};
    MTLPixelFormat motion{};
    MTLPixelFormat reactive{};
    NSUInteger input_width{};
    NSUInteger input_height{};
    std::array<u32, 2> output{};
    bool auto_exposure{};

    bool operator==(const ScalerKey&) const noexcept = default;
};

bool HasUsage(id<MTLTexture> texture, MTLTextureUsage usage) {
    return (texture.usage & usage) == usage;
}

} // Anonymous namespace

struct TemporalUpscaler::Impl {
    id scaler = nil;
    id<MTLTexture> output = nil;
    ScalerKey key{};
    ScalerKey failed_key{};
    bool has_failed_key{};
    bool first_evaluation{};
    bool failed{};
};

TemporalUpscaler::TemporalUpscaler() : impl{std::make_unique<Impl>()} {}

TemporalUpscaler::~TemporalUpscaler() = default;

bool TemporalUpscaler::IsSupported(const Device& device) {
    if (@available(iOS 16.0, macOS 13.0, *)) {
        return [MTLFXTemporalScalerDescriptor supportsDevice:device.GetDevice()];
    }
    return false;
}

bool TemporalUpscaler::ReactiveMaskSupported() {
    if (@available(iOS 17.4, macOS 14.4, *)) {
        return true;
    }
    return false;
}

bool TemporalUpscaler::ReactiveMaskSupported(const Device& device) {
    return IsSupported(device) && ReactiveMaskSupported();
}

bool TemporalUpscaler::Failed() const noexcept {
    return impl->failed;
}

TextureHandle TemporalUpscaler::Upscale(const Device& device, Scheduler& scheduler,
                                        const Inputs& inputs) {
    Impl& state = *impl;
    if (state.failed) {
        return nullptr;
    }
    // Permanent GPU/OS absence disables the scaler; per-frame validation declines that frame only.
    const auto decline_permanent = [&state](const char* reason) -> TextureHandle {
        LOG_WARNING(Render_Metal, "MetalFX temporal: declined, {}", reason);
        state.failed = true;
        return nullptr;
    };
    if (@available(iOS 16.0, macOS 13.0, *)) {
        const id<MTLTexture> color = AsTexture(inputs.color);
        const id<MTLTexture> depth = AsTexture(inputs.depth);
        const id<MTLTexture> motion = AsTexture(inputs.motion);
        const id<MTLTexture> reactive =
            ReactiveMaskSupported() ? AsTexture(inputs.reactive) : nil;
        if (color == nil || depth == nil || motion == nil || inputs.render_size[0] == 0 ||
            inputs.render_size[1] == 0 || inputs.output_size[0] == 0 ||
            inputs.output_size[1] == 0) {
            return nullptr;
        }
        if (depth.width != color.width || depth.height != color.height ||
            motion.width != color.width || motion.height != color.height ||
            (reactive != nil &&
             (reactive.width != color.width || reactive.height != color.height))) {
            return nullptr;
        }
        if (color.width < inputs.render_size[0] || color.height < inputs.render_size[1]) {
            return nullptr;
        }
        f32 min_scale = 1.0f;
        f32 max_scale = 3.0f;
        if (@available(iOS 17.0, macOS 14.0, *)) {
            min_scale = [MTLFXTemporalScalerDescriptor
                supportedInputContentMinScaleForDevice:device.GetDevice()];
            max_scale = [MTLFXTemporalScalerDescriptor
                supportedInputContentMaxScaleForDevice:device.GetDevice()];
        }
        const f32 scale_x =
            static_cast<f32>(inputs.output_size[0]) / static_cast<f32>(inputs.render_size[0]);
        const f32 scale_y =
            static_cast<f32>(inputs.output_size[1]) / static_cast<f32>(inputs.render_size[1]);
        if (scale_x < min_scale || scale_x > max_scale || scale_y < min_scale ||
            scale_y > max_scale) {
            return nullptr;
        }
        const ScalerKey key{
            .color = color.pixelFormat,
            .depth = depth.pixelFormat,
            .motion = motion.pixelFormat,
            .reactive = reactive != nil ? reactive.pixelFormat : MTLPixelFormatInvalid,
            .input_width = color.width,
            .input_height = color.height,
            .output = inputs.output_size,
            .auto_exposure = inputs.auto_exposure,
        };
        if (state.has_failed_key && state.failed_key == key) {
            return nullptr;
        }
        if (state.scaler == nil || !(state.key == key)) {
            if (![MTLFXTemporalScalerDescriptor supportsDevice:device.GetDevice()]) {
                return decline_permanent("not supported on this GPU");
            }
            if (state.output != nil) {
                scheduler.Finish();
            }
            MTLFXTemporalScalerDescriptor* descriptor =
                [[MTLFXTemporalScalerDescriptor alloc] init];
            descriptor.colorTextureFormat = key.color;
            descriptor.depthTextureFormat = key.depth;
            descriptor.motionTextureFormat = key.motion;
            descriptor.outputTextureFormat = OutputFormat;
            descriptor.inputWidth = key.input_width;
            descriptor.inputHeight = key.input_height;
            descriptor.outputWidth = key.output[0];
            descriptor.outputHeight = key.output[1];
            descriptor.autoExposureEnabled = key.auto_exposure ? YES : NO;
            // Dynamic Resolution Scaling updates inputContentWidth/Height per frame without
            // rebuilding the scaler or resetting history.
            descriptor.inputContentPropertiesEnabled = YES;
            descriptor.inputContentMinScale = min_scale;
            descriptor.inputContentMaxScale = max_scale;
            if (key.reactive != MTLPixelFormatInvalid) {
                if (@available(iOS 17.4, macOS 14.4, *)) {
                    descriptor.reactiveMaskTextureEnabled = YES;
                    descriptor.reactiveMaskTextureFormat = key.reactive;
                }
            }
            id<MTLFXTemporalScaler> scaler = nil;
            // An exception MetalFX raises for a descriptor it rejects would end the app.
            @try {
                scaler = [descriptor newTemporalScalerWithDevice:device.GetDevice()];
            } @catch (NSException* exception) {
                const char* const reason = exception.reason.UTF8String;
                LOG_ERROR(Render_Metal, "MetalFX temporal: creating the scaler raised: {}",
                          reason != nullptr ? reason : "no reason given");
                scaler = nil;
            }
            if (scaler == nil) {
                LOG_WARNING(Render_Metal, "MetalFX temporal: no scaler for {}x{} in {}x{} -> {}x{}",
                            inputs.render_size[0], inputs.render_size[1], key.input_width,
                            key.input_height, key.output[0], key.output[1]);
                state.failed_key = key;
                state.has_failed_key = true;
                return nullptr;
            }
            MTLTextureDescriptor* output = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:OutputFormat
                                             width:key.output[0]
                                            height:key.output[1]
                                         mipmapped:NO];
            // MTLFXTemporalScaler.outputTexture must be private.
            output.storageMode = MTLStorageModePrivate;
            output.usage = scaler.outputTextureUsage | MTLTextureUsageShaderRead;
            state.output = [device.GetDevice() newTextureWithDescriptor:output];
            if (state.output == nil) {
                state.failed_key = key;
                state.has_failed_key = true;
                return nullptr;
            }
            state.scaler = scaler;
            state.key = key;
            state.has_failed_key = false;
            state.first_evaluation = true;
            LOG_INFO(Render_Metal,
                     "MetalFX temporal: {}x{} in {}x{} textures -> {}x{}, auto exposure {}, "
                     "reactive mask {}",
                     inputs.render_size[0], inputs.render_size[1], key.input_width,
                     key.input_height, key.output[0], key.output[1],
                     key.auto_exposure ? "on" : "off",
                     key.reactive != MTLPixelFormatInvalid ? "on" : "off");
        }
        id<MTLFXTemporalScaler> scaler = state.scaler;
        if (!HasUsage(color, scaler.colorTextureUsage) ||
            !HasUsage(depth, scaler.depthTextureUsage) ||
            !HasUsage(motion, scaler.motionTextureUsage)) {
            return nullptr;
        }
        if (@available(iOS 17.4, macOS 14.4, *)) {
            if (reactive != nil) {
                if (!HasUsage(reactive, scaler.reactiveTextureUsage)) {
                    return nullptr;
                }
                scaler.reactiveMaskTexture = reactive;
            } else {
                scaler.reactiveMaskTexture = nil;
            }
        }
        scaler.colorTexture = color;
        scaler.depthTexture = depth;
        scaler.motionTexture = motion;
        scaler.outputTexture = state.output;
        // Every evaluation, as the steps of MTLFXTemporalScaler list them.
        scaler.inputContentWidth = inputs.render_size[0];
        scaler.inputContentHeight = inputs.render_size[1];
        scaler.jitterOffsetX = inputs.jitter[0];
        scaler.jitterOffsetY = inputs.jitter[1];
        scaler.motionVectorScaleX = inputs.motion_vector_scale[0];
        scaler.motionVectorScaleY = inputs.motion_vector_scale[1];
        scaler.depthReversed = inputs.depth_reversed ? YES : NO;
        scaler.preExposure = 1.0f;
        scaler.reset = inputs.reset || state.first_evaluation;
        state.first_evaluation = false;
        const id<MTLCommandBuffer> command_buffer =
            (__bridge id<MTLCommandBuffer>)scheduler.CommandBuffer();
        @try {
            [scaler encodeToCommandBuffer:command_buffer];
        } @catch (NSException* exception) {
            const char* const reason = exception.reason.UTF8String;
            LOG_ERROR(Render_Metal, "MetalFX temporal: encoding raised: {}",
                      reason != nullptr ? reason : "no reason given");
            return nullptr;
        }
        return (__bridge void*)state.output;
    }
    return decline_permanent("needs iOS 16 or macOS 13");
}

} // namespace Metal
