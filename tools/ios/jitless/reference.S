// SPDX-License-Identifier: GPL-2.0-or-later
// Built only in the macOS test oracle, never linked into the runner.
.text
.p2align 2
.globl _reference
_reference:
1:  add x0, x0, #5
    sub x1, x1, #1
    cbnz x1, 1b
    str w0, [x2, #4]
    ldr w0, [x2, #4]
    ret
.globl _reference_end
_reference_end:
    nop
