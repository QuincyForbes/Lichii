// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/framebuffer_limits.h"

namespace {
constexpr Vulkan::FramebufferSize APPLE_LIMITS{16384, 16384, 2048};
} // Anonymous namespace

TEST_CASE("Attachment-less framebuffers stay within the device limits",
          "[video_core][framebuffer_limits]") {
    const auto size = Vulkan::ClampFramebufferSize({32768, 32768, 1}, APPLE_LIMITS);
    REQUIRE(size.width == 16384);
    REQUIRE(size.height == 16384);
    REQUIRE(size.layers == 1);
}

TEST_CASE("Framebuffer sizes within the device limits are unchanged",
          "[video_core][framebuffer_limits]") {
    const auto size = Vulkan::ClampFramebufferSize({1920, 1080, 6}, APPLE_LIMITS);
    REQUIRE(size.width == 1920);
    REQUIRE(size.height == 1080);
    REQUIRE(size.layers == 6);
}

TEST_CASE("Framebuffer layers are clamped to the device limit",
          "[video_core][framebuffer_limits]") {
    const auto size = Vulkan::ClampFramebufferSize({64, 64, 4096}, APPLE_LIMITS);
    REQUIRE(size.layers == 2048);
}

TEST_CASE("Framebuffers keep at least one pixel and one layer",
          "[video_core][framebuffer_limits]") {
    const auto size = Vulkan::ClampFramebufferSize({0, 0, 0}, APPLE_LIMITS);
    REQUIRE(size.width == 1);
    REQUIRE(size.height == 1);
    REQUIRE(size.layers == 1);
}

TEST_CASE("A framebuffer without attachments ends where the scissor does",
          "[video_core][framebuffer_limits]") {
    REQUIRE(Vulkan::ScissorReach(0, 1, 32768) == 1);
    REQUIRE(Vulkan::ScissorReach(0, 256, 32768) == 256);
    REQUIRE(Vulkan::ScissorReach(100, 200, 32768) == 512);
    REQUIRE(Vulkan::ScissorReach(-8, 24, 32768) == 16);
}

TEST_CASE("A scissor that is off or past the surface clip leaves the framebuffer its size",
          "[video_core][framebuffer_limits]") {
    REQUIRE(Vulkan::ScissorReach(0, 0x7fffffff, 32768) == 32768);
    REQUIRE(Vulkan::ScissorReach(0x7fffffff, 0x7fffffff, 32768) == 32768);
    REQUIRE(Vulkan::ScissorReach(0, 1200, 1000) == 1000);
    REQUIRE(Vulkan::ScissorReach(0, 600, 1000) == 1000);
}

TEST_CASE("An empty scissor keeps one pixel", "[video_core][framebuffer_limits]") {
    REQUIRE(Vulkan::ScissorReach(0, 0, 32768) == 1);
    REQUIRE(Vulkan::ScissorReach(-64, 32, 32768) == 1);
    REQUIRE(Vulkan::ScissorReach(0, 16, 0) == 1);
}
