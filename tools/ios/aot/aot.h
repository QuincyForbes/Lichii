// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct aot_state {
    uint64_t x[31];
    uint64_t pc;
    uint32_t svc;
};

enum aot_stop { AOT_BUDGET, AOT_SVC, AOT_UNKNOWN_PC };

enum aot_stop aot_run(struct aot_state *s, unsigned budget);

#ifdef __cplusplus
}
#endif
