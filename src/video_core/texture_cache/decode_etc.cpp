// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>

#include "video_core/texture_cache/decode_etc.h"

namespace VideoCommon {

using VideoCore::Surface::PixelFormat;

namespace {

struct Rgba {
    s32 r, g, b, a;
};

constexpr std::array<std::array<s32, 2>, 8> Modifiers{{
    {2, 8}, {5, 17}, {9, 29}, {13, 42}, {18, 60}, {24, 80}, {33, 106}, {47, 183},
}};
constexpr std::array<s32, 8> Distances{3, 6, 11, 16, 23, 32, 41, 64};
constexpr std::array<std::array<s32, 8>, 16> EacModifiers{{
    {-3, -6, -9, -15, 2, 5, 8, 14},   {-3, -7, -10, -13, 2, 6, 9, 12},
    {-2, -5, -8, -13, 1, 4, 7, 12},   {-2, -4, -6, -13, 1, 3, 5, 12},
    {-3, -6, -8, -12, 2, 5, 7, 11},   {-3, -7, -9, -11, 2, 6, 8, 10},
    {-4, -7, -8, -11, 3, 6, 7, 10},   {-3, -5, -8, -11, 2, 4, 7, 10},
    {-2, -6, -8, -10, 1, 5, 7, 9},    {-2, -5, -8, -10, 1, 4, 7, 9},
    {-2, -4, -8, -10, 1, 3, 7, 9},    {-2, -5, -7, -10, 1, 4, 6, 9},
    {-3, -4, -7, -10, 2, 3, 6, 9},    {-1, -2, -3, -10, 0, 1, 2, 9},
    {-4, -6, -8, -9, 3, 5, 7, 8},     {-3, -5, -7, -9, 2, 4, 6, 8},
}};

s32 Clamp8(s32 value) {
    return std::clamp(value, 0, 255);
}

s32 Extend4(s32 v) {
    return v * 17;
}
s32 Extend5(s32 v) {
    return (v << 3) | (v >> 2);
}
s32 Extend6(s32 v) {
    return (v << 2) | (v >> 4);
}
s32 Extend7(s32 v) {
    return (v << 1) | (v >> 6);
}
s32 SignExtend3(s32 v) {
    return v >= 4 ? v - 8 : v;
}

/// The 16 texels of an ETC2 colour block, [x][y] by x * 4 + y, alpha 255 or 0 (punch-through).
void DecodeColorBlock(const u8* b, bool punch, std::array<Rgba, 16>& out) {
    const u32 msb_bits = (u32{b[4]} << 8) | b[5];
    const u32 lsb_bits = (u32{b[6]} << 8) | b[7];
    const auto index_of = [&](u32 k) { return ((msb_bits >> k) & 1U) << 1 | ((lsb_bits >> k) & 1U); };
    const bool diff = punch || (b[3] & 2) != 0;
    const bool flip = (b[3] & 1) != 0;
    // PTA: the diff bit is the opaque bit; the block is always differential.
    const bool opaque = punch ? (b[3] & 2) != 0 : true;
    const auto paint = [&](const std::array<Rgba, 4>& colors) {
        for (u32 k = 0; k < 16; ++k) {
            out[k] = colors[index_of(k)];
        }
    };
    {
        s32 r1, g1, b1, r2, g2, b2;
        bool overflow_r = false;
        bool overflow_g = false;
        bool overflow_b = false;
        if (diff) {
            const s32 rr = b[0] >> 3;
            const s32 gg = b[1] >> 3;
            const s32 bb = b[2] >> 3;
            const s32 r = rr + SignExtend3(b[0] & 7);
            const s32 g = gg + SignExtend3(b[1] & 7);
            const s32 bl = bb + SignExtend3(b[2] & 7);
            overflow_r = r < 0 || r > 31;
            overflow_g = !overflow_r && (g < 0 || g > 31);
            overflow_b = !overflow_r && !overflow_g && (bl < 0 || bl > 31);
            r1 = Extend5(rr);
            g1 = Extend5(gg);
            b1 = Extend5(bb);
            r2 = Extend5(r & 31);
            g2 = Extend5(g & 31);
            b2 = Extend5(bl & 31);
        } else {
            r1 = Extend4(b[0] >> 4);
            g1 = Extend4(b[1] >> 4);
            b1 = Extend4(b[2] >> 4);
            r2 = Extend4(b[0] & 15);
            g2 = Extend4(b[1] & 15);
            b2 = Extend4(b[2] & 15);
        }
        if (overflow_r) {
            // T mode.
            const s32 r0 = Extend4(((b[0] >> 3) & 3) << 2 | (b[0] & 3));
            const s32 g0 = Extend4(b[1] >> 4);
            const s32 bl0 = Extend4(b[1] & 15);
            const s32 rr1 = Extend4(b[2] >> 4);
            const s32 gg1 = Extend4(b[2] & 15);
            const s32 bb1 = Extend4(b[3] >> 4);
            const s32 d = Distances[((b[3] >> 2) & 3) << 1 | (b[3] & 1)];
            const auto add = [&](s32 v) { return Clamp8(v); };
            paint({Rgba{r0, g0, bl0, 255},
                   Rgba{add(rr1 + d), add(gg1 + d), add(bb1 + d), 255},
                   Rgba{rr1, gg1, bb1, 255},
                   Rgba{add(rr1 - d), add(gg1 - d), add(bb1 - d), 255}});
            if (!opaque) {
                for (u32 k = 0; k < 16; ++k) {
                    if (index_of(k) == 2) {
                        out[k] = {0, 0, 0, 0};
                    }
                }
            }
            return;
        }
        if (overflow_g) {
            // H mode.
            const s32 r0 = (b[0] >> 3) & 15;
            const s32 g0 = ((b[0] & 7) << 1) | ((b[1] >> 4) & 1);
            const s32 bl0 = (b[1] & 8) | ((b[1] & 3) << 1) | ((b[2] >> 7) & 1);
            const s32 rr1 = (b[2] >> 3) & 15;
            const s32 gg1 = ((b[2] & 7) << 1) | ((b[3] >> 7) & 1);
            const s32 bb1 = (b[3] >> 3) & 15;
            const s32 c0 = (r0 << 8) | (g0 << 4) | bl0;
            const s32 c1 = (rr1 << 8) | (gg1 << 4) | bb1;
            const s32 d = Distances[(b[3] & 4) | ((b[3] & 1) << 1) | (c0 >= c1 ? 1 : 0)];
            const s32 er0 = Extend4(r0), eg0 = Extend4(g0), eb0 = Extend4(bl0);
            const s32 er1 = Extend4(rr1), eg1 = Extend4(gg1), eb1 = Extend4(bb1);
            paint({Rgba{Clamp8(er0 + d), Clamp8(eg0 + d), Clamp8(eb0 + d), 255},
                   Rgba{Clamp8(er0 - d), Clamp8(eg0 - d), Clamp8(eb0 - d), 255},
                   Rgba{Clamp8(er1 + d), Clamp8(eg1 + d), Clamp8(eb1 + d), 255},
                   Rgba{Clamp8(er1 - d), Clamp8(eg1 - d), Clamp8(eb1 - d), 255}});
            if (!opaque) {
                for (u32 k = 0; k < 16; ++k) {
                    if (index_of(k) == 2) {
                        out[k] = {0, 0, 0, 0};
                    }
                }
            }
            return;
        }
        if (overflow_b) {
            // Planar mode.
            const s32 ro = Extend6((b[0] >> 1) & 0x3f);
            const s32 go = Extend7(((b[0] & 1) << 6) | ((b[1] >> 1) & 0x3f));
            const s32 bo = Extend6(((b[1] & 1) << 5) | (b[2] & 0x18) | ((b[2] & 3) << 1) |
                                   (b[3] >> 7));
            const s32 rh = Extend6(((b[3] & 0x7c) >> 1) | (b[3] & 1));
            const s32 gh = Extend7(b[4] >> 1);
            const s32 bh = Extend6(((b[4] & 1) << 5) | (b[5] >> 3));
            const s32 rv = Extend6(((b[5] & 7) << 3) | (b[6] >> 5));
            const s32 gv = Extend7(((b[6] & 0x1f) << 2) | (b[7] >> 6));
            const s32 bv = Extend6(b[7] & 0x3f);
            for (s32 x = 0; x < 4; ++x) {
                for (s32 y = 0; y < 4; ++y) {
                    out[x * 4 + y] = {Clamp8((x * (rh - ro) + y * (rv - ro) + 4 * ro + 2) >> 2),
                                      Clamp8((x * (gh - go) + y * (gv - go) + 4 * go + 2) >> 2),
                                      Clamp8((x * (bh - bo) + y * (bv - bo) + 4 * bo + 2) >> 2),
                                      255};
                }
            }
            return;
        }
        // ETC1-style individual / differential.
        const s32 table1 = (b[3] >> 5) & 7;
        const s32 table2 = (b[3] >> 2) & 7;
        for (s32 x = 0; x < 4; ++x) {
            for (s32 y = 0; y < 4; ++y) {
                const bool second = flip ? y >= 2 : x >= 2;
                const s32 table = second ? table2 : table1;
                const u32 k = static_cast<u32>(x * 4 + y);
                const u32 index = index_of(k);
                s32 modifier;
                s32 alpha = 255;
                if (!opaque) {
                    const s32 m = Modifiers[table][1];
                    modifier = index == 1 ? m : index == 3 ? -m : 0;
                    alpha = index == 2 ? 0 : 255;
                } else {
                    const s32 m = Modifiers[table][index & 1];
                    modifier = (index & 2) != 0 ? -m : m;
                }
                const s32 base_r = second ? r2 : r1;
                const s32 base_g = second ? g2 : g1;
                const s32 base_b = second ? b2 : b1;
                out[k] = alpha == 0 ? Rgba{0, 0, 0, 0}
                                    : Rgba{Clamp8(base_r + modifier), Clamp8(base_g + modifier),
                                           Clamp8(base_b + modifier), 255};
            }
        }
    }
}

/// An EAC block's 16 raw 8-bit values (the alpha of RGBA) or 11-bit ones (`wide`; signed: the
/// base is a signed byte and the result in -1023..1023).
void DecodeEacBlock(const u8* b, bool wide, bool is_signed, std::array<s32, 16>& out) {
    const s32 base = is_signed ? std::max<s32>(static_cast<s8>(b[0]), -127) : b[0];
    const s32 multiplier = b[1] >> 4;
    const auto& table = EacModifiers[b[1] & 15];
    u64 bits = 0;
    for (u32 i = 2; i < 8; ++i) {
        bits = (bits << 8) | b[i];
    }
    for (u32 k = 0; k < 16; ++k) {
        const s32 modifier = table[(bits >> (45 - 3 * k)) & 7];
        if (!wide) {
            out[k] = Clamp8(base + modifier * multiplier);
        } else if (is_signed) {
            out[k] = std::clamp(base * 8 + modifier * (multiplier == 0 ? 1 : multiplier * 8),
                                -1023, 1023);
        } else {
            out[k] = std::clamp(base * 8 + 4 + modifier * (multiplier == 0 ? 1 : multiplier * 8),
                                0, 2047);
        }
    }
}

} // Anonymous namespace

bool IsEtcFormat(PixelFormat format) {
    switch (format) {
    case PixelFormat::ETC2_RGB_UNORM:
    case PixelFormat::ETC2_RGBA_UNORM:
    case PixelFormat::ETC2_RGB_PTA_UNORM:
    case PixelFormat::ETC2_RGB_SRGB:
    case PixelFormat::ETC2_RGBA_SRGB:
    case PixelFormat::ETC2_RGB_PTA_SRGB:
    case PixelFormat::EAC_R11_UNORM:
    case PixelFormat::EAC_R11_SNORM:
    case PixelFormat::EAC_R11G11_UNORM:
    case PixelFormat::EAC_R11G11_SNORM:
        return true;
    default:
        return false;
    }
}

u32 EtcDecodedBytes(PixelFormat format) {
    return format == PixelFormat::EAC_R11_UNORM || format == PixelFormat::EAC_R11_SNORM ? 2 : 4;
}

void DecodeEtc(std::span<const u8> input, std::span<u8> output, const BufferImageCopy& copy,
               PixelFormat format) {
    const u32 width = copy.image_extent.width;
    const u32 height = copy.image_extent.height;
    const u32 planes = copy.image_subresource.num_layers * copy.image_extent.depth;
    const u32 blocks_x = (width + 3) / 4;
    const u32 blocks_y = (height + 3) / 4;
    const bool rg = format == PixelFormat::EAC_R11G11_UNORM || format == PixelFormat::EAC_R11G11_SNORM;
    const bool r11 = format == PixelFormat::EAC_R11_UNORM || format == PixelFormat::EAC_R11_SNORM;
    const bool is_signed =
        format == PixelFormat::EAC_R11_SNORM || format == PixelFormat::EAC_R11G11_SNORM;
    const bool rgba = format == PixelFormat::ETC2_RGBA_UNORM || format == PixelFormat::ETC2_RGBA_SRGB;
    const bool pta = format == PixelFormat::ETC2_RGB_PTA_UNORM || format == PixelFormat::ETC2_RGB_PTA_SRGB;
    const u32 block_bytes = rg || rgba ? 16 : 8;
    const u32 texel_bytes = EtcDecodedBytes(format);
    for (u32 plane = 0; plane < planes; ++plane) {
        for (u32 by = 0; by < blocks_y; ++by) {
            for (u32 bx = 0; bx < blocks_x; ++bx) {
                const size_t in = (size_t{plane} * blocks_x * blocks_y + size_t{by} * blocks_x + bx) *
                                  block_bytes;
                if (in + block_bytes > input.size()) {
                    return;
                }
                const u8* const block = input.data() + in;
                std::array<Rgba, 16> color{};
                std::array<s32, 16> first{};
                std::array<s32, 16> second{};
                if (r11 || rg) {
                    DecodeEacBlock(block, true, is_signed, first);
                    if (rg) {
                        DecodeEacBlock(block + 8, true, is_signed, second);
                    }
                } else if (rgba) {
                    DecodeEacBlock(block, false, false, first);
                    DecodeColorBlock(block + 8, false, color);
                } else {
                    DecodeColorBlock(block, pta, color);
                }
                for (u32 x = 0; x < 4; ++x) {
                    for (u32 y = 0; y < 4; ++y) {
                        const u32 px = bx * 4 + x;
                        const u32 py = by * 4 + y;
                        if (px >= width || py >= height) {
                            continue;
                        }
                        const size_t out = ((size_t{plane} * height + py) * width + px) * texel_bytes;
                        if (out + texel_bytes > output.size()) {
                            return;
                        }
                        const u32 k = x * 4 + y;
                        u8* const dst = output.data() + out;
                        if (r11 || rg) {
                            // Unsigned: 0..2047 to 0..65535; signed: -1023..1023 to snorm16.
                            const auto wide = [&](s32 v) -> u16 {
                                return static_cast<u16>(
                                    is_signed ? static_cast<s16>((v * 32767 + (v < 0 ? -511 : 511)) / 1023)
                                              : (v * 65535 + 1023) / 2047);
                            };
                            const u16 a = wide(first[k]);
                            std::memcpy(dst, &a, 2);
                            if (rg) {
                                const u16 g = wide(second[k]);
                                std::memcpy(dst + 2, &g, 2);
                            }
                        } else {
                            dst[0] = static_cast<u8>(color[k].r);
                            dst[1] = static_cast<u8>(color[k].g);
                            dst[2] = static_cast<u8>(color[k].b);
                            dst[3] = static_cast<u8>(rgba ? first[k] : color[k].a);
                        }
                    }
                }
            }
        }
    }
}

} // namespace VideoCommon
