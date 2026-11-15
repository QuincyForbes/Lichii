#!/bin/bash
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# Builds the unsigned IPA on the Mac the way .github/workflows/ios.yml does (same configure
# flags), for when a CI artifact is not wanted. bench_mac.py sign then signs it for the phone.
#   bash build-ipa.sh <source checkout> [<build dir>]     -> <build dir>/Lichii.ipa
# Needs Homebrew: brew install ninja ccache nasm glslang meson pkgconf cmake
set -euo pipefail
src=$(cd "${1:?source checkout}" && pwd)
build=${2:-$src/build-ios}
for tool in cmake ninja nasm glslangValidator meson pkg-config; do
  command -v "$tool" >/dev/null || { echo "missing $tool: brew install ninja ccache nasm glslang meson pkgconf cmake" >&2; exit 1; }
done
if [ ! -d "$src/ffmpeg-ios" ]; then
  (cd "$src" && OUT="$src/ffmpeg-ios" bash .ci/ios/build-ffmpeg.sh)
fi
launcher=()
if command -v ccache >/dev/null; then
  launcher=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
fi
cmake -S "$src" -B "$build" -G Ninja \
  -DCMAKE_Swift_COMPILER_TARGET=arm64-apple-ios17.0 \
  -DLICHII_FFMPEG_DIR="$src/ffmpeg-ios" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=17.0 \
  -DCMAKE_C_FLAGS=-miphoneos-version-min=17.0 -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
  -DCMAKE_SYSTEM_PROCESSOR=arm64 -DBOOST_CONTEXT_ARCHITECTURE=arm64 -DBOOST_CONTEXT_ABI=aapcs \
  -DBOOST_CONTEXT_BINARY_FORMAT=mach-o -DBOOST_CONTEXT_ASSEMBLER=gas "${launcher[@]}" \
  -DCPMUTIL_FORCE_BUNDLED=ON -DENABLE_QT=OFF -DYUZU_CMD=ON -DYUZU_ROOM=OFF \
  -DYUZU_ROOM_STANDALONE=OFF -DENABLE_WEB_SERVICE=OFF -DENABLE_UPDATE_CHECKER=OFF \
  -DENABLE_LIBUSB=OFF -DENABLE_CUBEB=OFF -DENABLE_RESHADE=OFF -DUSE_DISCORD_PRESENCE=OFF \
  -DENABLE_WERROR=OFF -DYUZU_TESTS=OFF -DBUILD_TESTING=OFF -DDYNARMIC_TESTS=OFF \
  -DYUZU_USE_BUNDLED_FFMPEG=ON -DYUZU_USE_BUNDLED_SDL3=ON -DYUZU_USE_BUNDLED_SIRIT=ON \
  -DYUZU_USE_BUNDLED_OPENSSL=OFF -DOPENSSL_TARGET_PLATFORM=ios64-xcrun \
  -DYUZU_USE_BUNDLED_MOLTENVK=ON -DLICHII_BUILD_TAG=ios-local \
  -DLICHII_BUILD_PLATFORM=ios-arm64
cmake --build "$build" --target yuzu-cmd --parallel
app="$build/bin/Lichii.app"
test -x "$app/Lichii"
rm -rf "$build/Payload" && mkdir "$build/Payload"
cp -R "$app" "$build/Payload/"
(cd "$build" && rm -f Lichii.ipa && /usr/bin/zip -qry Lichii.ipa Payload)
echo "$build/Lichii.ipa"
