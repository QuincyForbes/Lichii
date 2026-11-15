// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <utility>

namespace Metal {

/// Releases an object retained by Object (metal_object.mm).
void ReleaseObject(void* object) noexcept;

/// A retained Objective-C object (an MTLTexture, MTLBuffer, ...) owned by plain C++ code. C++
/// passes the pointer around; the Objective-C++ files bridge it back (metal_bridge.h). Keeping
/// Metal types out of the texture and buffer cache headers lets those compile on every platform
/// (LICHII_METAL_COMPILE_CHECK).
class Object {
public:
    Object() = default;

    /// Takes over one reference (CFBridgingRetain / __bridge_retained).
    explicit Object(void* retained) noexcept : ptr{retained} {}

    ~Object() {
        Reset();
    }

    Object(const Object&) = delete;
    Object& operator=(const Object&) = delete;

    Object(Object&& other) noexcept : ptr{std::exchange(other.ptr, nullptr)} {}

    Object& operator=(Object&& other) noexcept {
        if (this != &other) {
            Reset();
            ptr = std::exchange(other.ptr, nullptr);
        }
        return *this;
    }

    void Reset() noexcept {
        if (ptr != nullptr) {
            ReleaseObject(std::exchange(ptr, nullptr));
        }
    }

    [[nodiscard]] void* Get() const noexcept {
        return ptr;
    }

    explicit operator bool() const noexcept {
        return ptr != nullptr;
    }

private:
    void* ptr{};
};

/// An unretained id<MTLBuffer> (the buffer caches' BufferType).
using BufferHandle = void*;

/// An unretained id<MTLTexture>.
using TextureHandle = void*;

} // namespace Metal
