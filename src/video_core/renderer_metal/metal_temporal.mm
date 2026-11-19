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
    std::array<u32, 2> render{};
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
    if (@available(iOS 18.0, macOS 15.0, *)) {
        return true;
    }
    return false;
}

TextureHandle TemporalUpscaler::Upscale(const Device& device, Scheduler& scheduler,
                                        const Inputs& inputs) {
    Impl& state = *impl;
    if (state.failed) {
        return nullptr;
    }
    // The caller falls back to the game's own upscaler; this says why, once.
    const auto decline = [&state](const char* reason) -> TextureHandle {
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
        if (color == nil || depth == nil || motion == nil) {
            return decline("an input has no texture");
        }
        if (color.width < inputs.render_size[0] || color.height < inputs.render_size[1] ||
            depth.width < inputs.render_size[0] || depth.height < inputs.render_size[1] ||
            motion.width < inputs.render_size[0] || motion.height < inputs.render_size[1]) {
            return decline("an input is smaller than the render size");
        }
        const ScalerKey key{
            .color = color.pixelFormat,
            .depth = depth.pixelFormat,
            .motion = motion.pixelFormat,
            .reactive = reactive != nil ? reactive.pixelFormat : MTLPixelFormatInvalid,
            .input_width = color.width,
            .input_height = color.height,
            .render = inputs.render_size,
            .output = inputs.output_size,
            .auto_exposure = inputs.auto_exposure,
        };
        if (state.scaler == nil || !(state.key == key)) {
            if (state.output != nil) {
                scheduler.Finish();
            }
            if (![MTLFXTemporalScalerDescriptor supportsDevice:device.GetDevice()]) {
                return decline("not supported on this GPU");
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
            // The game renders into part of larger textures: MetalFX reads the render size.
            if (key.render[0] != key.input_width || key.render[1] != key.input_height) {
                const f32 scale_x = static_cast<f32>(key.output[0]) / key.render[0];
                const f32 scale_y = static_cast<f32>(key.output[1]) / key.render[1];
                descriptor.inputContentPropertiesEnabled = YES;
                descriptor.inputContentMinScale = (std::min)(scale_x, scale_y);
                descriptor.inputContentMaxScale = (std::max)(scale_x, scale_y);
            }
            if (key.reactive != MTLPixelFormatInvalid) {
                if (@available(iOS 18.0, macOS 15.0, *)) {
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
                            key.render[0], key.render[1], key.input_width, key.input_height,
                            key.output[0], key.output[1]);
                state.failed = true;
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
                return decline("no output texture");
            }
            state.scaler = scaler;
            state.key = key;
            state.first_evaluation = true;
            LOG_INFO(Render_Metal,
                     "MetalFX temporal: {}x{} in {}x{} textures -> {}x{}, reactive mask {}",
                     key.render[0], key.render[1], key.input_width, key.input_height,
                     key.output[0], key.output[1],
                     key.reactive != MTLPixelFormatInvalid ? "on" : "off");
        }
        id<MTLFXTemporalScaler> scaler = state.scaler;
        if (!HasUsage(color, scaler.colorTextureUsage) ||
            !HasUsage(depth, scaler.depthTextureUsage) ||
            !HasUsage(motion, scaler.motionTextureUsage)) {
            return decline("an input lacks the usage MetalFX needs");
        }
        if (reactive != nil) {
            if (@available(iOS 18.0, macOS 15.0, *)) {
                if (!HasUsage(reactive, scaler.reactiveTextureUsage)) {
                    return decline("the reactive mask lacks the usage MetalFX needs");
                }
                scaler.reactiveMaskTexture = reactive;
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
            return decline("encoding raised an exception");
        }
        return (__bridge void*)state.output;
    }
    return decline("needs iOS 16 or macOS 13");
}

} // namespace Metal
