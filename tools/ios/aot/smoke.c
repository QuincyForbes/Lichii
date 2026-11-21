// SPDX-License-Identifier: GPL-2.0-or-later
#include "aot.h"

int main(void) {
    struct aot_state s = {0};
    if (aot_run(&s, 2) != AOT_BUDGET || s.pc != 24 || s.x[30] != 8)
        return 1;
    if (aot_run(&s, 16) != AOT_SVC || s.pc != 12 || s.svc != 0x42 || s.x[0] != 40)
        return 2;
    s.x[0] = 41; // Synthetic service result; resume directly after SVC.
    if (aot_run(&s, 16) != AOT_SVC || s.pc != 20 || s.svc != 0 || s.x[0] != 42)
        return 3;
    s.x[1] = UINT64_C(0x100000000);
    if (aot_run(&s, 2) != AOT_UNKNOWN_PC || s.pc != s.x[1])
        return 4;
    s.pc = 1;
    return aot_run(&s, 1) != AOT_UNKNOWN_PC;
}
