"""Compile one raw, little-endian A64 text section (guest base zero) to C11."""

import argparse
from pathlib import Path
import struct


def lower(pc, word):
    rd = word & 31
    rn = (word >> 5) & 31
    advance = f"s->pc = {pc + 4};"
    if word & 0xffe0001f == 0xd4000001:  # SVC #imm16
        return f"{advance} s->svc = {(word >> 5) & 65535}; return AOT_SVC;"
    if word & 0xff800000 == 0xd2800000 and rd != 31:  # MOVZ Xd
        value = ((word >> 5) & 65535) << (16 * ((word >> 21) & 3))
        return f"s->x[{rd}] = UINT64_C({value}); {advance}"
    if word & 0xff800000 == 0x91000000 and 31 not in (rn, rd):  # ADD Xd, Xn, #imm
        imm = ((word >> 10) & 4095) << (12 * ((word >> 22) & 1))
        return f"s->x[{rd}] = s->x[{rn}] + {imm}; {advance}"
    branch = word & 0xfffffc1f
    if branch in (0xd61f0000, 0xd63f0000, 0xd65f0000) and rn != 31:  # BR, BLR, RET
        link = f" s->x[30] = {pc + 4};" if branch == 0xd63f0000 else ""
        return f"s->pc = s->x[{rn}];{link}"
    raise ValueError(f"unsupported A64 word 0x{word:08x} at guest PC 0x{pc:x}")


def translate(text):
    if not text or len(text) % 4:
        raise ValueError("expected a nonempty sequence of 4-byte instructions")
    cases = []
    for i, (word,) in enumerate(struct.iter_unpack("<I", text)):
        pc = i * 4
        body = lower(pc, word)
        cases.append(f"        case {pc}: {{ {body} break; }}")
    return '''#include "aot.h"

enum aot_stop aot_run(struct aot_state *s, unsigned budget) {
    while (budget--) {
        switch (s->pc) {
%s
        default: return AOT_UNKNOWN_PC;
        }
    }
    return AOT_BUDGET;
}
''' % "\n".join(cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        args.output.write_text(translate(args.text.read_bytes()))
    except ValueError as e:
        parser.exit(1, f"{e}\n")
