// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/arm64/abi.h"

#include <bit>

#include <boost/container/static_vector.hpp>
#include "common/common_types.h"
#include <oaknut/oaknut.hpp>

namespace Dynarmic::Backend::Arm64 {

using namespace oaknut::util;

static constexpr std::size_t gpr_size = 8;
static constexpr std::size_t fpr_size = 16;

using RegIndexList = boost::container::static_vector<int, 32>;

struct FrameInfo {
    RegIndexList gprs;
    RegIndexList fprs;
    std::size_t frame_size;
    std::size_t gprs_size;
    std::size_t fprs_size;
};

static RegIndexList ListToIndexes(u32 list) {
    RegIndexList indexes;
    for (u32 mask = list; mask != 0; mask &= mask - 1) {
        indexes.push_back(std::countr_zero(mask));
    }
    return indexes;
}

static FrameInfo CalculateFrameInfo(RegisterList rl, std::size_t frame_size) {
    const auto gprs = ListToIndexes(static_cast<u32>(rl));
    const auto fprs = ListToIndexes(static_cast<u32>(rl >> 32));

    const std::size_t num_gprs = gprs.size();
    const std::size_t num_fprs = fprs.size();

    const std::size_t gprs_size = (num_gprs + 1) / 2 * 16;
    const std::size_t fprs_size = num_fprs * 16;

    return {
        gprs,
        fprs,
        frame_size,
        gprs_size,
        fprs_size,
    };
}

#define DO_IT(TYPE, REG_TYPE, PAIR_OP, SINGLE_OP, OFFSET)                                                                                       \
    if (frame_info.TYPE##s.size() > 0) {                                                                                                        \
        for (std::size_t i = 0; i < frame_info.TYPE##s.size() - 1; i += 2) {                                                                    \
            code.PAIR_OP(oaknut::REG_TYPE{frame_info.TYPE##s[i]}, oaknut::REG_TYPE{frame_info.TYPE##s[i + 1]}, SP, (OFFSET) + i * TYPE##_size); \
        }                                                                                                                                       \
        if (frame_info.TYPE##s.size() % 2 == 1) {                                                                                               \
            const std::size_t i = frame_info.TYPE##s.size() - 1;                                                                                \
            code.SINGLE_OP(oaknut::REG_TYPE{frame_info.TYPE##s[i]}, SP, (OFFSET) + i * TYPE##_size);                                            \
        }                                                                                                                                       \
    }

void ABI_PushRegisters(oaknut::CodeGenerator& code, RegisterList rl, std::size_t frame_size) {
    if (rl == 0 && frame_size == 0) {
        return;
    }

    const FrameInfo frame_info = CalculateFrameInfo(rl, frame_size);
    const std::size_t regs_size = frame_info.gprs_size + frame_info.fprs_size;

    if (frame_info.frame_size == 0) {
        if (frame_info.fprs.empty()) {
            if (frame_info.gprs.size() == 1) {
                code.STR(oaknut::XReg{frame_info.gprs[0]}, SP, PRE_INDEXED, -16);
                return;
            }
            if (frame_info.gprs.size() == 2) {
                code.STP(oaknut::XReg{frame_info.gprs[0]}, oaknut::XReg{frame_info.gprs[1]}, SP, PRE_INDEXED, -16);
                return;
            }
        } else if (frame_info.gprs.empty()) {
            if (frame_info.fprs.size() == 1) {
                code.STR(oaknut::QReg{frame_info.fprs[0]}, SP, PRE_INDEXED, -16);
                return;
            }
            if (frame_info.fprs.size() == 2) {
                code.STP(oaknut::QReg{frame_info.fprs[0]}, oaknut::QReg{frame_info.fprs[1]}, SP, PRE_INDEXED, -32);
                return;
            }
        }
    }

    if (regs_size > 0) {
        code.SUB(SP, SP, regs_size);

        DO_IT(gpr, XReg, STP, STR, 0)
        DO_IT(fpr, QReg, STP, STR, frame_info.gprs_size)
    }

    if (frame_info.frame_size > 0) {
        code.SUB(SP, SP, frame_info.frame_size);
    }
}

void ABI_PopRegisters(oaknut::CodeGenerator& code, RegisterList rl, std::size_t frame_size) {
    if (rl == 0 && frame_size == 0) {
        return;
    }

    const FrameInfo frame_info = CalculateFrameInfo(rl, frame_size);
    const std::size_t regs_size = frame_info.gprs_size + frame_info.fprs_size;

    if (frame_info.frame_size == 0) {
        if (frame_info.fprs.empty()) {
            if (frame_info.gprs.size() == 1) {
                code.LDR(oaknut::XReg{frame_info.gprs[0]}, SP, POST_INDEXED, 16);
                return;
            }
            if (frame_info.gprs.size() == 2) {
                code.LDP(oaknut::XReg{frame_info.gprs[0]}, oaknut::XReg{frame_info.gprs[1]}, SP, POST_INDEXED, 16);
                return;
            }
        } else if (frame_info.gprs.empty()) {
            if (frame_info.fprs.size() == 1) {
                code.LDR(oaknut::QReg{frame_info.fprs[0]}, SP, POST_INDEXED, 16);
                return;
            }
            if (frame_info.fprs.size() == 2) {
                code.LDP(oaknut::QReg{frame_info.fprs[0]}, oaknut::QReg{frame_info.fprs[1]}, SP, POST_INDEXED, 32);
                return;
            }
        }
    }

    if (frame_info.frame_size > 0) {
        code.ADD(SP, SP, frame_info.frame_size);
    }

    if (regs_size > 0) {
        DO_IT(gpr, XReg, LDP, LDR, 0)
        DO_IT(fpr, QReg, LDP, LDR, frame_info.gprs_size)

        code.ADD(SP, SP, regs_size);
    }
}

}  // namespace Dynarmic::Backend::Arm64
