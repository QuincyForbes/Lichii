// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Instruction semantics adapted from Mesa mme_tu104_sim.c and mme_tu104.c:
// Copyright © 2022 Collabora Ltd. SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace Tegra {

// TU104/GA10x MME has two ALUs and two outputs per 96-bit instruction.
class TuringMacroInterpreter {
public:
    using Word = std::uint32_t;
    struct Result {
        bool success;
        Word pc;
        std::size_t consumed;
        std::string_view error;
    };

    // One instruction's fields, pre-decoded once per uploaded macro generation. Packed into 24
    // bytes so typical MME macros fit in 2-3 L1 cache lines on arm64/x86-64.
    struct Decoded {
        std::array<std::uint16_t, 2> imm{};
        std::array<std::uint8_t, 2> op{}, dst{}, src_a{}, src_b{}, out_method{}, out_emit{};
        std::uint8_t pred_reg{}, pred_mode{}, mask_true{15}, mask_false{15};
        bool end{};
        std::array<bool, 2> load{};
    };
    static_assert(sizeof(Decoded) <= 32);

    static Decoded Decode(const std::array<Word, 3>& bits) {
        const auto field = [&](Word low, Word count) {
            std::uint64_t value = bits[low / 32];
            if (low / 32 < 2) value |= std::uint64_t{bits[low / 32 + 1]} << 32;
            return Word((value >> (low % 32)) & ((std::uint64_t{1} << count) - 1));
        };
        Decoded d;
        d.imm = {static_cast<std::uint16_t>(field(30, 16)),
                 static_cast<std::uint16_t>(field(66, 16))};
        d.op = {static_cast<std::uint8_t>(field(10, 5)),
                static_cast<std::uint8_t>(field(46, 5))};
        d.dst = {static_cast<std::uint8_t>(field(15, 5)),
                 static_cast<std::uint8_t>(field(51, 5))};
        d.src_a = {static_cast<std::uint8_t>(field(20, 5)),
                   static_cast<std::uint8_t>(field(56, 5))};
        d.src_b = {static_cast<std::uint8_t>(field(25, 5)),
                   static_cast<std::uint8_t>(field(61, 5))};
        d.out_method = {static_cast<std::uint8_t>(field(82, 3)),
                        static_cast<std::uint8_t>(field(89, 3))};
        d.out_emit = {static_cast<std::uint8_t>(field(85, 4)),
                      static_cast<std::uint8_t>(field(92, 4))};
        d.pred_reg = static_cast<std::uint8_t>(field(5, 5));
        d.pred_mode = static_cast<std::uint8_t>(field(1, 4));
        d.end = field(0, 1) != 0;
        constexpr std::array<std::string_view, 16> predicates{
            "UUUU", "TTTT", "FFFF", "TTUU", "FFUU", "TFUU", "TUUU", "FUUU",
            "UUTT", "UUTF", "UUTU", "UUFT", "UUFF", "UUFU", "UUUT", "UUUF"};
        d.mask_true = d.mask_false = 15;
        if (d.pred_mode) {
            d.mask_true = d.mask_false = 0;
            for (Word i = 0; i < 4; ++i) {
                if (predicates[d.pred_mode][i] != 'T') d.mask_true |= static_cast<std::uint8_t>(1U << i);
                if (predicates[d.pred_mode][i] != 'F') d.mask_false |= static_cast<std::uint8_t>(1U << i);
            }
        }
        for (Word i = 0; i < 2; ++i) {
            const Word reg = 28 + i, out = 3 + i;
            d.load[i] = d.pred_reg == reg || d.src_a[0] == reg || d.src_a[1] == reg ||
                        d.src_b[0] == reg || d.src_b[1] == reg || d.out_method[0] == out ||
                        d.out_method[1] == out || d.out_emit[0] == out || d.out_emit[1] == out;
        }
        return d;
    }

    /// Pre-decodes `code` (3 words per instruction) once per `generation`.
    void Prepare(std::span<const Word> code, std::uint64_t generation) {
        const std::size_t count = code.size() / 3;
        if (has_generation && prepared_generation == generation &&
            prepared_code == code.data() && decoded.size() == count) {
            return;
        }
        DecodeAll(code);
        has_generation = true;
        prepared_generation = generation;
        prepared_code = code.data();
    }

    template <typename Read, typename Emit>
    // `parameters` is the load FIFO; `emit` may append to it (MME_DMA_READ_FIFOED).
    Result Execute(std::span<const Word> code, Word start,
                   std::vector<Word>& parameters, Read read, Emit emit,
                   bool track_data_ram = true) {
        const std::size_t code_count = code.size() / 3;
        if (!has_generation || prepared_code != code.data() || decoded.size() != code_count) {
            DecodeAll(code);
            if (has_generation) {
                prepared_code = code.data();
            }
        }
        // 0..23: MME general registers; 24: RZ (kept 0); 28..29: current instruction's FIFO loads.
        std::array<Word, 32> regs{};
        Word valid_regs = 0x3f000000U;
        Word carry{}, method{}, increment{};
        Word pc = start, loop_count{}, loop_start{}, loop_end{};
        std::size_t consumed{};
        bool end_next{}, stopped{};
        const char* error = nullptr;
        const auto fail = [&](const char* message) {
            if (error == nullptr) {
                error = message;
            }
        };
        read_registers = false;
        reread_data_ram = false;
        data_ram_reads.clear();
        data_ram_writes.clear();
        if (track_data_ram && data_ram_reads.capacity() < 16) {
            data_ram_reads.reserve(16);
            data_ram_writes.reserve(16);
        }
        const Word* param_data = parameters.data();
        std::size_t param_size = parameters.size();
        for (Word steps = 0; steps < 1000000; ++steps) {
            if (static_cast<std::size_t>(pc) >= code_count) {
                fail("instruction outside uploaded RAM");
                break;
            }
            const Decoded& d = decoded[pc];
            if (d.load[0]) {
                if (consumed >= param_size) {
                    fail("parameter FIFO underflow");
                    break;
                }
                regs[28] = param_data[consumed++];
            }
            if (d.load[1]) {
                if (consumed >= param_size) {
                    fail("parameter FIFO underflow");
                    break;
                }
                regs[29] = param_data[consumed++];
            }
            // Both ALUs read the register file before either destination is committed: their
            // operands are read up front. Carry chaining remains explicit between the lanes.
            const auto get = [&](Word reg, Word lane) -> Word {
                if (!(valid_regs & (1U << reg))) {
                    fail(reg < 24 ? "uninitialized register read" : "invalid source register");
                    return 0;
                }
                if ((reg - 25U) > 2U) {
                    return regs[reg];
                }
                if (reg == 25) {
                    return Word(std::int32_t(std::int16_t(d.imm[lane])));
                }
                if (reg == 26) {
                    return Word(std::int32_t(std::int16_t(d.imm[1 - lane])));
                }
                return (Word{d.imm[0]} << 16) | d.imm[1];
            };
            const Word mask = d.pred_mode ? (get(d.pred_reg, 0) ? d.mask_true : d.mask_false) : 15U;
            std::array<Word, 2> xs{}, ys{}, alu{};
            if (mask & 1U) {
                xs[0] = get(d.src_a[0], 0);
                ys[0] = get(d.src_b[0], 0);
            }
            if (mask & 2U) {
                xs[1] = get(d.src_a[1], 1);
                ys[1] = get(d.src_b[1], 1);
            }
            if (error != nullptr) {
                break;
            }
            Word next_pc = pc + 1;
            bool branch_cancels_end{};
            for (Word lane = 0; lane < 2 && error == nullptr; ++lane) {
                if (!(mask & (1U << lane))) continue;
                const Word x = xs[lane], y = ys[lane];
                const Word imm = d.imm[lane];
                Word value{};
                const auto condition = [&](Word comparison) {
                    switch (comparison) {
                    case 17: case 25: return std::int32_t(x) < std::int32_t(y);
                    case 18: case 26: return x < y;
                    case 19: case 27: return std::int32_t(x) <= std::int32_t(y);
                    case 20: case 28: return x <= y;
                    default: return x == y;
                    }
                };
                switch (d.op[lane]) {
                case 0: value = x + y; carry = value < x; break;
                case 1: value = x + y + carry; break;
                case 2: value = x - y; carry = value > x; break;
                case 3: value = x - y - carry; break;
                case 4: case 6: {
                    const std::uint64_t a = d.op[lane] == 4 ? std::uint64_t(std::int64_t(std::int32_t(x))) : x;
                    const std::uint64_t b = d.op[lane] == 4 ? std::uint64_t(std::int64_t(std::int32_t(y))) : y;
                    const auto product = a * b;
                    value = Word(product); carry = Word(product >> 32); break;
                }
                case 5: value = carry; break;
                case 7:
                    // EXTENDED: x = 0x1000 is the load barrier (Mesa mme_tu104_load_barrier),
                    // nothing to do in a sequential interpreter.
                    if (x != 0x1000) fail("unsupported extended operation");
                    break;
                case 8: value = Word(std::countl_zero(x)); break;
                case 9: value = x << (y & 31); break;
                case 10: value = x >> (y & 31); break;
                case 11: value = Word(std::int32_t(x) >> (y & 31)); break;
                case 12: value = x & y; break;
                case 13: value = ~(x & y); break;
                case 14: value = x | y; break;
                case 15: value = x ^ y; break;
                case 16: {
                    const Word dest = (imm >> 10) & 63, count = (imm >> 5) & 31;
                    const Word source = imm & 31;
                    if (dest >= 32) { fail("invalid merge destination"); break; }
                    const Word bits_mask = (1U << count) - 1;
                    value = (x & ~(bits_mask << dest)) | (((y >> source) & bits_mask) << dest);
                    break;
                }
                case 17: case 18: case 19: case 20: case 21:
                    value = condition(d.op[lane]) ? ~0U : 0; break;
                case 22: value = read((x + y) & 0xffff); read_registers = true; break;
                case 23:
                    if (loop_count || imm < 2) { fail("invalid loop"); break; }
                    // The native SDK emits zero-count loops around FIFO reads.
                    // Unlike Mesa's test simulator, these must skip the body.
                    if (x == 0) { next_pc = pc + imm; break; }
                    loop_count = x - 1; loop_start = pc; loop_end = pc + imm - 1;
                    break;
                case 24: next_pc = pc + (imm & 0x7fff); break;
                case 25: case 26: case 27: case 28: case 29:
                    if (condition(d.op[lane]) == bool(imm & 0x8000)) {
                        const auto offset = std::int32_t(imm << 19) >> 19;
                        if (offset == -4096) stopped = true;
                        else {
                            next_pc = Word(std::int32_t(pc) + offset);
                            // Native back edges use bit 14 to suppress the
                            // fallthrough termination when the branch is taken.
                            branch_cancels_end = (imm & 0x4000) != 0;
                        }
                    }
                    break;
                case 30:
                    if (x >= active_dram->size()) { fail("data RAM read out of bounds"); break; }
                    value = (*active_dram)[x];
                    if (track_data_ram && !reread_data_ram) {
                        if (std::ranges::find(data_ram_writes, x, &DataRamAccess::first) !=
                            data_ram_writes.end()) {
                            reread_data_ram = true;
                        } else {
                            data_ram_reads.emplace_back(x, value);
                        }
                    }
                    break;
                case 31:
                    if (x >= active_dram->size()) { fail("data RAM write out of bounds"); break; }
                    (*active_dram)[x] = y;
                    if (track_data_ram && !reread_data_ram) {
                        data_ram_writes.emplace_back(x, y);
                    }
                    break;
                }
                alu[lane] = value;
                if (d.dst[lane] < 24) {
                    regs[d.dst[lane]] = value;
                    valid_regs |= 1U << d.dst[lane];
                } else if (d.dst[lane] != 24) {
                    fail("invalid destination register");
                }
            }
            const auto output = [&](Word selector) -> Word {
                switch (selector) {
                case 1: return alu[0]; case 2: return alu[1];
                case 3: return regs[28]; case 4: return regs[29];
                case 5: return d.imm[0]; case 6: return d.imm[1];
                case 8: return Word{d.imm[0]} >> 12; case 9: return Word{d.imm[1]} >> 12;
                case 10: return (Word{d.imm[0]} << 16) | d.imm[1];
                default: fail("invalid output selector"); return 0;
                }
            };
            for (Word lane = 0; lane < 2 && error == nullptr; ++lane) {
                if (!(mask & (4U << lane))) continue;
                if (d.out_method[lane]) {
                    const Word address = output(d.out_method[lane]);
                    method = address & 0xfff; increment = (address >> 12) & 15;
                }
                if (d.out_emit[lane]) {
                    const Word value = output(d.out_emit[lane]);
                    if (error == nullptr) {
                        if (!emit(method, value)) {
                            fail("unsupported emitted method");
                        } else {
                            param_data = parameters.data();
                            param_size = parameters.size();
                        }
                    }
                    method += increment;
                }
            }
            if (error != nullptr) break;
            if (end_next || stopped) return {true, pc, consumed, {}};
            end_next = d.end && !branch_cancels_end;
            if (loop_count && pc == loop_end) { --loop_count; next_pc = loop_start + 1; }
            pc = next_pc;
        }
        if (error == nullptr) error = "instruction limit exceeded";
        return {false, pc, consumed, error};
    }

    using DataRam = std::array<Word, 0xc00>;
    DataRam dram{};
    // The data RAM DREAD/DWRITE use: a channel's own copy when the caller sets one. Macro code is
    // shared by the channels of a process, but each channel keeps its own data RAM (title 1's
    // channels store their own QMD ring base in word 0x27).
    DataRam* active_dram{&dram};
    // Decoded instructions by PC, pre-decoded once per uploaded macro generation.
    std::vector<Decoded> decoded;
    // What the last Execute read besides its parameters, for the macro memo: whether it read a
    // register (MME_READ), the data RAM words it read and wrote (address, value) in order, and
    // whether it read a word it had written itself.
    using DataRamAccess = std::pair<Word, Word>;
    bool read_registers{};
    bool reread_data_ram{};
    std::vector<DataRamAccess> data_ram_reads;
    std::vector<DataRamAccess> data_ram_writes;

private:
    void DecodeAll(std::span<const Word> code) {
        const std::size_t count = code.size() / 3;
        decoded.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            decoded[i] = Decode({code[i * 3 + 2], code[i * 3 + 1], code[i * 3]});
        }
    }

    bool has_generation{};
    std::uint64_t prepared_generation{~std::uint64_t{0}};
    const Word* prepared_code{};
};

} // namespace Tegra
