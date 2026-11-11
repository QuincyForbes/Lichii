// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <random>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/macro.h"
#include "video_core/macro_turing.h"

namespace {

using Word = Tegra::TuringMacroInterpreter::Word;

struct InsSpec {
    bool end = false;
    u32 pred_mode = 0;
    u32 pred_reg = 24;
    std::array<u32, 2> op{0, 0};
    std::array<u32, 2> dst{24, 24};
    std::array<u32, 2> src_a{24, 24};
    std::array<u32, 2> src_b{24, 24};
    std::array<u32, 2> imm{0, 0};
    std::array<u32, 2> out_method{0, 0};
    std::array<u32, 2> out_emit{0, 0};
};

std::array<Word, 3> EncodeInstruction(const InsSpec& s) {
    std::array<Word, 3> bits{};
    const auto put = [&](u32 low, u32 count, u32 val) {
        for (u32 b = 0; b < count; ++b) {
            if ((val >> b) & 1U) {
                const u32 idx = low + b;
                bits[idx / 32] |= 1U << (idx % 32);
            }
        }
    };
    put(0, 1, s.end ? 1U : 0U);
    put(1, 4, s.pred_mode);
    put(5, 5, s.pred_reg);
    put(10, 5, s.op[0]);
    put(15, 5, s.dst[0]);
    put(20, 5, s.src_a[0]);
    put(25, 5, s.src_b[0]);
    put(30, 16, s.imm[0]);
    put(46, 5, s.op[1]);
    put(51, 5, s.dst[1]);
    put(56, 5, s.src_a[1]);
    put(61, 5, s.src_b[1]);
    put(66, 16, s.imm[1]);
    put(82, 3, s.out_method[0]);
    put(85, 4, s.out_emit[0]);
    put(89, 3, s.out_method[1]);
    put(92, 4, s.out_emit[1]);
    // TuringMacroInterpreter stores each 96-bit instruction as {bits[2], bits[1], bits[0]} at pc*3.
    return {bits[2], bits[1], bits[0]};
}

void AppendInstruction(std::vector<Word>& code, const InsSpec& spec) {
    const auto words = EncodeInstruction(spec);
    code.insert(code.end(), words.begin(), words.end());
}

// Bit-exact copy of the pre-optimization TuringMacroInterpreter for differential verification.
class ReferenceTuringInterpreter {
public:
    using Result = Tegra::TuringMacroInterpreter::Result;
    using DataRam = Tegra::TuringMacroInterpreter::DataRam;
    using DataRamAccess = Tegra::TuringMacroInterpreter::DataRamAccess;

    struct OldDecoded {
        std::array<Word, 3> raw{~0U, ~0U, ~0U};
        std::array<Word, 2> imm, op, dst, src_a, src_b, out_method, out_emit;
        Word pred_reg, pred_mode;
        Word mask_true, mask_false;
        bool end;
        std::array<bool, 2> load;
    };

    static OldDecoded Decode(const std::array<Word, 3>& bits) {
        const auto field = [&](Word low, Word count) {
            std::uint64_t value = bits[low / 32];
            if (low / 32 < 2) {
                value |= std::uint64_t{bits[low / 32 + 1]} << 32;
            }
            return Word((value >> (low % 32)) & ((std::uint64_t{1} << count) - 1));
        };
        OldDecoded d;
        d.raw = bits;
        d.imm = {field(30, 16), field(66, 16)};
        d.op = {field(10, 5), field(46, 5)};
        d.dst = {field(15, 5), field(51, 5)};
        d.src_a = {field(20, 5), field(56, 5)};
        d.src_b = {field(25, 5), field(61, 5)};
        d.out_method = {field(82, 3), field(89, 3)};
        d.out_emit = {field(85, 4), field(92, 4)};
        d.pred_reg = field(5, 5);
        d.pred_mode = field(1, 4);
        d.end = field(0, 1) != 0;
        constexpr std::array<std::string_view, 16> predicates{
            "UUUU", "TTTT", "FFFF", "TTUU", "FFUU", "TFUU", "TUUU", "FUUU",
            "UUTT", "UUTF", "UUTU", "UUFT", "UUFF", "UUFU", "UUUT", "UUUF"};
        d.mask_true = d.mask_false = 15;
        if (d.pred_mode) {
            d.mask_true = d.mask_false = 0;
            for (Word i = 0; i < 4; ++i) {
                if (predicates[d.pred_mode][i] != 'T') {
                    d.mask_true |= 1U << i;
                }
                if (predicates[d.pred_mode][i] != 'F') {
                    d.mask_false |= 1U << i;
                }
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

    template <typename Read, typename Emit>
    Result Execute(std::span<const Word> code, Word start, std::vector<Word>& parameters, Read read,
                   Emit emit) {
        std::array<Word, 24> regs{};
        Word initialized{}, carry{}, method{}, increment{};
        Word pc = start, loop_count{}, loop_start{}, loop_end{};
        std::size_t consumed{};
        bool end_next{}, stopped{};
        std::string_view error;
        const auto fail = [&](std::string_view message) {
            if (error.empty()) {
                error = message;
            }
        };
        if (decoded.size() < code.size() / 3) {
            decoded.resize(code.size() / 3);
        }
        read_registers = false;
        reread_data_ram = false;
        data_ram_reads.clear();
        data_ram_writes.clear();
        for (Word steps = 0; steps < 1000000; ++steps) {
            if (static_cast<std::size_t>(pc) * 3 + 2 >= code.size()) {
                fail("instruction outside uploaded RAM");
                break;
            }
            const std::array<Word, 3> bits{code[pc * 3 + 2], code[pc * 3 + 1], code[pc * 3]};
            OldDecoded& d = decoded[pc];
            if (d.raw != bits) {
                d = Decode(bits);
            }
            std::array<Word, 2> load{}, alu{};
            for (Word i = 0; i < 2; ++i) {
                if (d.load[i]) {
                    if (consumed >= parameters.size()) {
                        fail("parameter FIFO underflow");
                        break;
                    }
                    load[i] = parameters[consumed++];
                }
            }
            const auto get = [&](Word reg, Word lane) -> Word {
                if (reg < 24) {
                    if (!(initialized & (1U << reg))) {
                        fail("uninitialized register read");
                    }
                    return regs[reg];
                }
                switch (reg) {
                case 24: return 0;
                case 25: return Word(std::int32_t(std::int16_t(d.imm[lane])));
                case 26: return Word(std::int32_t(std::int16_t(d.imm[1 - lane])));
                case 27: return (d.imm[0] << 16) | d.imm[1];
                case 28: return load[0];
                case 29: return load[1];
                default: fail("invalid source register"); return 0;
                }
            };
            const Word mask = d.pred_mode ? (get(d.pred_reg, 0) ? d.mask_true : d.mask_false) : 15;
            std::array<Word, 2> xs{}, ys{};
            for (Word lane = 0; lane < 2; ++lane) {
                if (mask & (1U << lane)) {
                    xs[lane] = get(d.src_a[lane], lane);
                    ys[lane] = get(d.src_b[lane], lane);
                }
            }
            Word next_pc = pc + 1;
            bool branch_cancels_end{};
            for (Word lane = 0; lane < 2 && error.empty(); ++lane) {
                if (!(mask & (1U << lane))) {
                    continue;
                }
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
                    const std::uint64_t a =
                        d.op[lane] == 4 ? std::uint64_t(std::int64_t(std::int32_t(x))) : x;
                    const std::uint64_t b =
                        d.op[lane] == 4 ? std::uint64_t(std::int64_t(std::int32_t(y))) : y;
                    const auto product = a * b;
                    value = Word(product);
                    carry = Word(product >> 32);
                    break;
                }
                case 5: value = carry; break;
                case 7:
                    if (x != 0x1000) {
                        fail("unsupported extended operation");
                    }
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
                    if (dest >= 32) {
                        fail("invalid merge destination");
                        break;
                    }
                    const Word bits_mask = (1U << count) - 1;
                    value = (x & ~(bits_mask << dest)) | (((y >> source) & bits_mask) << dest);
                    break;
                }
                case 17: case 18: case 19: case 20: case 21:
                    value = condition(d.op[lane]) ? ~0U : 0;
                    break;
                case 22:
                    value = read((x + y) & 0xffff);
                    read_registers = true;
                    break;
                case 23:
                    if (loop_count || imm < 2) {
                        fail("invalid loop");
                        break;
                    }
                    if (x == 0) {
                        next_pc = pc + imm;
                        break;
                    }
                    loop_count = x - 1;
                    loop_start = pc;
                    loop_end = pc + imm - 1;
                    break;
                case 24: next_pc = pc + (imm & 0x7fff); break;
                case 25: case 26: case 27: case 28: case 29:
                    if (condition(d.op[lane]) == bool(imm & 0x8000)) {
                        const auto offset = std::int32_t(imm << 19) >> 19;
                        if (offset == -4096) {
                            stopped = true;
                        } else {
                            next_pc = Word(std::int32_t(pc) + offset);
                            branch_cancels_end = (imm & 0x4000) != 0;
                        }
                    }
                    break;
                case 30:
                    if (x >= active_dram->size()) {
                        fail("data RAM read out of bounds");
                        break;
                    }
                    value = (*active_dram)[x];
                    if (std::ranges::find(data_ram_writes, x, &DataRamAccess::first) !=
                        data_ram_writes.end()) {
                        reread_data_ram = true;
                    }
                    data_ram_reads.emplace_back(x, value);
                    break;
                case 31:
                    if (x >= active_dram->size()) {
                        fail("data RAM write out of bounds");
                        break;
                    }
                    (*active_dram)[x] = y;
                    data_ram_writes.emplace_back(x, y);
                    break;
                }
                alu[lane] = value;
                if (d.dst[lane] < 24) {
                    regs[d.dst[lane]] = value;
                    initialized |= 1U << d.dst[lane];
                } else if (d.dst[lane] != 24) {
                    fail("invalid destination register");
                }
            }
            const auto output = [&](Word selector) -> Word {
                switch (selector) {
                case 1: return alu[0];
                case 2: return alu[1];
                case 3: return load[0];
                case 4: return load[1];
                case 5: return d.imm[0];
                case 6: return d.imm[1];
                case 8: return d.imm[0] >> 12;
                case 9: return d.imm[1] >> 12;
                case 10: return (d.imm[0] << 16) | d.imm[1];
                default: fail("invalid output selector"); return 0;
                }
            };
            for (Word lane = 0; lane < 2 && error.empty(); ++lane) {
                if (!(mask & (4U << lane))) {
                    continue;
                }
                if (d.out_method[lane]) {
                    const Word address = output(d.out_method[lane]);
                    method = address & 0xfff;
                    increment = (address >> 12) & 15;
                }
                if (d.out_emit[lane]) {
                    const Word value = output(d.out_emit[lane]);
                    if (error.empty() && !emit(method, value)) {
                        fail("unsupported emitted method");
                    }
                    method += increment;
                }
            }
            if (!error.empty()) {
                break;
            }
            if (end_next || stopped) {
                return {true, pc, consumed, {}};
            }
            end_next = d.end && !branch_cancels_end;
            if (loop_count && pc == loop_end) {
                --loop_count;
                next_pc = loop_start + 1;
            }
            pc = next_pc;
        }
        if (error.empty()) {
            error = "instruction limit exceeded";
        }
        return {false, pc, consumed, error};
    }

    DataRam dram{};
    DataRam* active_dram{&dram};
    std::vector<OldDecoded> decoded;
    bool read_registers{};
    bool reread_data_ram{};
    std::vector<DataRamAccess> data_ram_reads;
    std::vector<DataRamAccess> data_ram_writes;
};

void CheckEquivalent(std::span<const Word> code, Word start, const std::vector<Word>& params_in,
                     const Tegra::TuringMacroInterpreter::DataRam& initial_dram,
                     u64 generation = 1, u32 reject_method = ~0U) {
    ReferenceTuringInterpreter ref;
    Tegra::TuringMacroInterpreter opt_prepared;
    Tegra::TuringMacroInterpreter opt_direct;
    Tegra::TuringMacroInterpreter opt_untracked;

    ref.dram = initial_dram;
    opt_prepared.dram = initial_dram;
    opt_direct.dram = initial_dram;
    opt_untracked.dram = initial_dram;

    std::vector<Word> p_ref = params_in;
    std::vector<Word> p_prep = params_in;
    std::vector<Word> p_dir = params_in;
    std::vector<Word> p_untrack = params_in;

    std::vector<std::pair<Word, Word>> out_ref, out_prep, out_dir, out_untrack;
    const auto read_fn = [](Word addr) -> Word { return (addr * 0x9e3779b1U) ^ 0x5a5a5a5aU; };
    const auto make_emit = [&](std::vector<std::pair<Word, Word>>& out, std::vector<Word>& fifo) {
        return [&, reject_method](Word method, Word value) {
            if (method == reject_method) {
                return false;
            }
            out.emplace_back(method, value);
            if (method == 0x158 && value <= 16) {
                for (Word k = 0; k < value; ++k) {
                    fifo.push_back(0x1000U + k + static_cast<Word>(out.size()));
                }
            }
            return true;
        };
    };

    opt_prepared.Prepare(code, generation);
    const auto res_ref = ref.Execute(code, start, p_ref, read_fn, make_emit(out_ref, p_ref));
    const auto res_prep =
        opt_prepared.Execute(code, start, p_prep, read_fn, make_emit(out_prep, p_prep), true);
    const auto res_dir =
        opt_direct.Execute(code, start, p_dir, read_fn, make_emit(out_dir, p_dir), true);
    opt_untracked.Prepare(code, generation);
    const auto res_untrack = opt_untracked.Execute(code, start, p_untrack, read_fn,
                                                   make_emit(out_untrack, p_untrack), false);

    REQUIRE(res_prep.success == res_ref.success);
    REQUIRE(res_prep.pc == res_ref.pc);
    REQUIRE(res_prep.consumed == res_ref.consumed);
    REQUIRE(res_prep.error == res_ref.error);
    REQUIRE(out_prep == out_ref);
    REQUIRE(p_prep == p_ref);
    REQUIRE(opt_prepared.dram == ref.dram);
    REQUIRE(opt_prepared.read_registers == ref.read_registers);
    REQUIRE(opt_prepared.reread_data_ram == ref.reread_data_ram);
    if (!ref.reread_data_ram) {
        REQUIRE(opt_prepared.data_ram_reads == ref.data_ram_reads);
        REQUIRE(opt_prepared.data_ram_writes == ref.data_ram_writes);
    } else {
        REQUIRE_FALSE(opt_prepared.data_ram_writes.empty());
    }

    REQUIRE(res_dir.success == res_ref.success);
    REQUIRE(res_dir.pc == res_ref.pc);
    REQUIRE(res_dir.consumed == res_ref.consumed);
    REQUIRE(res_dir.error == res_ref.error);
    REQUIRE(out_dir == out_ref);
    REQUIRE(opt_direct.dram == ref.dram);

    REQUIRE(res_untrack.success == res_ref.success);
    REQUIRE(res_untrack.pc == res_ref.pc);
    REQUIRE(res_untrack.consumed == res_ref.consumed);
    REQUIRE(res_untrack.error == res_ref.error);
    REQUIRE(out_untrack == out_ref);
    REQUIRE(opt_untracked.dram == ref.dram);
    REQUIRE(opt_untracked.data_ram_reads.empty());
    REQUIRE(opt_untracked.data_ram_writes.empty());
}

} // namespace

TEST_CASE("Turing macro uploads reach entry points past instruction 0x1000",
          "[video_core][macro]") {
    Tegra::MacroEngine engine{false};
    engine.native_turing = true;
    // Ocarina of Time calls macros starting at instructions 0x1010 and 0x1018.
    for (const u32 start : {0x1010U, 0x1018U}) {
        CAPTURE(start);
        engine.ClearCode(start);
        for (u32 word = 0; word < 3; ++word) {
            engine.AddCode(0, 0xabc0 + word);
        }
        for (u32 word = 0; word < 3; ++word) {
            REQUIRE(engine.native_code[start * 3 + word] == 0xabc0 + word);
        }
    }
    // An instruction pointer past the RAM leaves every word alone, even when * 3 wraps a u32.
    for (const u32 start : {0x2aabU, 0x55555556U}) {
        CAPTURE(start);
        const auto before = engine.native_code;
        engine.ClearCode(start);
        engine.AddCode(0, 0xdead);
        REQUIRE(engine.native_code == before);
    }
}

TEST_CASE("TuringMacroInterpreter compacts Decoded and reuses Prepare across generations",
          "[video_core][macro]") {
    STATIC_REQUIRE(sizeof(Tegra::TuringMacroInterpreter::Decoded) <= 32);

    std::vector<Word> code;
    AppendInstruction(code, InsSpec{
                                .end = true,
                                .op = {0, 0},
                                .dst = {0, 24},
                                .src_a = {28, 24},
                                .src_b = {25, 24},
                                .imm = {10, 0x1200},
                                .out_method = {6, 0},
                                .out_emit = {1, 0},
                            });
    AppendInstruction(code, InsSpec{});

    Tegra::TuringMacroInterpreter interp;
    interp.Prepare(code, 7);
    std::vector<Word> params{32};
    std::vector<std::pair<Word, Word>> emitted;
    auto res = interp.Execute(
        code, 0, params, [](Word) { return 0U; },
        [&](Word m, Word v) {
            emitted.emplace_back(m, v);
            return true;
        });
    REQUIRE(res.success);
    REQUIRE(emitted == std::vector<std::pair<Word, Word>>{{0x200, 42}});

    // Mutate code and advance generation: Prepare re-decodes; same generation keeps cached decode.
    code.clear();
    AppendInstruction(code, InsSpec{
                                .end = true,
                                .op = {0, 0},
                                .dst = {0, 24},
                                .src_a = {28, 24},
                                .src_b = {25, 24},
                                .imm = {100, 0x1300},
                                .out_method = {6, 0},
                                .out_emit = {1, 0},
                            });
    AppendInstruction(code, InsSpec{});
    interp.Prepare(code, 8);
    emitted.clear();
    res = interp.Execute(
        code, 0, params, [](Word) { return 0U; },
        [&](Word m, Word v) {
            emitted.emplace_back(m, v);
            return true;
        });
    REQUIRE(res.success);
    REQUIRE(emitted == std::vector<std::pair<Word, Word>>{{0x300, 132}});
}

TEST_CASE("TuringMacroInterpreter matches reference across ALU, control flow, RAM, FIFO, and errors",
          "[video_core][macro]") {
    Tegra::TuringMacroInterpreter::DataRam dram{};
    for (std::size_t i = 0; i < dram.size(); ++i) {
        dram[i] = static_cast<Word>(0x10000U + i * 7U);
    }

    // 1. Carry chaining, signed/unsigned mul, merge, clz, shifts, immediate modes, dual-lane swap.
    {
        std::vector<Word> code;
        // pc 0: r0 = load0 + 0xffff (imm0 = -1 -> carry!), r1 = load1 + imm0(lane1) + carry
        AppendInstruction(code, InsSpec{
                                    .op = {0, 1},
                                    .dst = {0, 1},
                                    .src_a = {28, 29},
                                    .src_b = {25, 26},
                                    .imm = {0xffff, 0x0010},
                                    .out_method = {6, 0},
                                    .out_emit = {1, 2},
                                });
        // pc 1: swap r0 and r1 simultaneously (tests pre-commit operand reads), mul signed + carry
        AppendInstruction(code, InsSpec{
                                    .op = {0, 4},
                                    .dst = {1, 2},
                                    .src_a = {0, 0},
                                    .src_b = {24, 1},
                                    .imm = {0x1100, 0},
                                    .out_method = {5, 0},
                                    .out_emit = {1, 2},
                                });
        // pc 2: r3 = carry (op 5), r4 = MERGE(r1, r2, dest=4, count=8, src=2)
        AppendInstruction(code, InsSpec{
                                    .op = {5, 16},
                                    .dst = {3, 4},
                                    .src_a = {24, 1},
                                    .src_b = {24, 2},
                                    .imm = {0, (4U << 10) | (8U << 5) | 2U},
                                    .out_emit = {1, 2},
                                });
        // pc 3: CLZ, SAR, IMM32 (src 27) and output selector 10
        AppendInstruction(code, InsSpec{
                                    .end = true,
                                    .op = {8, 11},
                                    .dst = {5, 6},
                                    .src_a = {27, 0},
                                    .src_b = {24, 25},
                                    .imm = {0x00f0, 0x0005},
                                    .out_method = {0, 0},
                                    .out_emit = {1, 10},
                                });
        // pc 4: delay slot of end, extended load barrier (0x1000) + unsigned mul
        AppendInstruction(code, InsSpec{
                                    .op = {7, 6},
                                    .dst = {24, 7},
                                    .src_a = {25, 5},
                                    .src_b = {24, 6},
                                    .imm = {0x1000, 0},
                                    .out_emit = {0, 2},
                                });
        CheckEquivalent(code, 0, {0x80000010U, 0x12345678U}, dram);
    }

    // 2. Hardware loop (zero count & non-zero count), DWRITE + DREAD reread, MME_DMA_READ_FIFOED.
    for (const Word count : {0U, 1U, 5U}) {
        CAPTURE(count);
        std::vector<Word> code;
        // pc 0: emit 0x158 (MME_DMA_READ_FIFOED) with load0 words, set r0 = load1 (loop count)
        AppendInstruction(code, InsSpec{
                                    .op = {0, 0},
                                    .dst = {24, 0},
                                    .src_a = {24, 29},
                                    .src_b = {24, 24},
                                    .imm = {0x0158, 0},
                                    .out_method = {5, 0},
                                    .out_emit = {3, 0},
                                });
        // pc 1: LOOP r0, imm=3 (body is pc 2..3), also DWRITE dram[10] = r0
        AppendInstruction(code, InsSpec{
                                    .op = {23, 31},
                                    .dst = {24, 24},
                                    .src_a = {0, 25},
                                    .src_b = {24, 0},
                                    .imm = {3, 10},
                                });
        // pc 2: load from appended FIFO and emit to 0x400
        AppendInstruction(code, InsSpec{
                                    .op = {30, 0},
                                    .dst = {2, 3},
                                    .src_a = {25, 28},
                                    .src_b = {24, 24},
                                    .imm = {10, 0x1400},
                                    .out_method = {0, 6},
                                    .out_emit = {0, 2},
                                });
        // pc 3: loop end, emit DREAD value
        AppendInstruction(code, InsSpec{
                                    .op = {0, 0},
                                    .dst = {2, 24},
                                    .src_a = {2, 24},
                                    .src_b = {25, 24},
                                    .imm = {1, 0},
                                    .out_emit = {1, 0},
                                });
        // pc 4: end + MME_READ register (op 22)
        AppendInstruction(code, InsSpec{
                                    .end = true,
                                    .op = {22, 30},
                                    .dst = {4, 5},
                                    .src_a = {25, 25},
                                    .src_b = {24, 24},
                                    .imm = {0x20, 11},
                                    .out_emit = {1, 2},
                                });
        // pc 5: delay slot
        AppendInstruction(code, InsSpec{});
        CheckEquivalent(code, 0, {6U, count}, dram);
    }

    // 3. Conditional back-edge branch with branch_cancels_end (bit 14) and immediate stop (-4096).
    {
        std::vector<Word> code;
        // pc 0: r0 = load0
        AppendInstruction(code, InsSpec{.op = {0, 0}, .dst = {0, 24}, .src_a = {28, 24}});
        // pc 1: r0 = r0 - 1; if r0 != 0 (EQ false -> bit 15 = 0) branch to pc 1 with bit 14 + end!
        const u32 back_imm = (0U << 15) | (1U << 14); // offset = 0 -> stays at pc 1 while r0 != 0
        AppendInstruction(code, InsSpec{
                                    .end = true,
                                    .op = {2, 29},
                                    .dst = {0, 24},
                                    .src_a = {0, 0},
                                    .src_b = {25, 24},
                                    .imm = {1, back_imm},
                                    .out_method = {5, 0},
                                    .out_emit = {1, 0},
                                });
        // pc 2: delay slot once loop exits; stop (-4096 = 0x1000 in 13-bit signed)
        AppendInstruction(code, InsSpec{
                                    .op = {29, 0},
                                    .src_a = {24, 24},
                                    .src_b = {24, 24},
                                    .imm = {0x8000U | 0x1000U, 0},
                                });
        CheckEquivalent(code, 0, {4U}, dram);
    }

    // 4. All 16 predicate modes for both zero and non-zero predicate registers.
    for (u32 pred_mode = 0; pred_mode < 16; ++pred_mode) {
        for (const Word pred_val : {0U, 5U}) {
            CAPTURE(pred_mode, pred_val);
            std::vector<Word> code;
            AppendInstruction(code, InsSpec{
                                        .end = true,
                                        .op = {0, 0},
                                        .dst = {0, 1},
                                        .src_a = {28, 29},
                                    });
            AppendInstruction(code, InsSpec{
                                        .pred_mode = pred_mode,
                                        .pred_reg = 0,
                                        .op = {0, 2},
                                        .dst = {2, 3},
                                        .src_a = {1, 1},
                                        .src_b = {25, 25},
                                        .imm = {0x220, 0x330},
                                        .out_method = {5, 6},
                                        .out_emit = {1, 2},
                                    });
            CheckEquivalent(code, 0, {pred_val, 0x1000U}, dram);
        }
    }

    // 5. All error paths match reference exactly.
    {
        // Outside uploaded RAM
        std::vector<Word> empty_code;
        CheckEquivalent(empty_code, 0, {}, dram);
        // FIFO underflow on load0 and load1
        std::vector<Word> c;
        AppendInstruction(c, InsSpec{.src_a = {28, 29}});
        CheckEquivalent(c, 0, {}, dram);
        CheckEquivalent(c, 0, {1U}, dram);
        // Uninitialized register read
        c.clear();
        AppendInstruction(c, InsSpec{.src_a = {7, 24}});
        CheckEquivalent(c, 0, {}, dram);
        // Invalid source register (30, 31)
        c.clear();
        AppendInstruction(c, InsSpec{.src_a = {30, 24}});
        CheckEquivalent(c, 0, {}, dram);
        c.clear();
        AppendInstruction(c, InsSpec{.src_b = {24, 31}});
        CheckEquivalent(c, 0, {}, dram);
        // Invalid destination register (25..31)
        c.clear();
        AppendInstruction(c, InsSpec{.dst = {25, 24}});
        CheckEquivalent(c, 0, {}, dram);
        // Unsupported extended op
        c.clear();
        AppendInstruction(c, InsSpec{.op = {7, 0}, .src_a = {25, 24}, .imm = {0x999, 0}});
        CheckEquivalent(c, 0, {}, dram);
        // Invalid merge destination (>= 32)
        c.clear();
        AppendInstruction(c, InsSpec{.op = {16, 0}, .imm = {32U << 10, 0}});
        CheckEquivalent(c, 0, {}, dram);
        // Invalid loop (imm < 2)
        c.clear();
        AppendInstruction(c, InsSpec{.op = {23, 0}, .imm = {1, 0}});
        CheckEquivalent(c, 0, {}, dram);
        // Data RAM read / write out of bounds
        c.clear();
        AppendInstruction(c, InsSpec{.op = {30, 0}, .src_a = {25, 24}, .imm = {0xc00, 0}});
        CheckEquivalent(c, 0, {}, dram);
        c.clear();
        AppendInstruction(c, InsSpec{.op = {31, 0}, .src_a = {25, 24}, .imm = {0xc00, 0}});
        CheckEquivalent(c, 0, {}, dram);
        // Invalid output selector (7)
        c.clear();
        AppendInstruction(c, InsSpec{.out_method = {7, 0}});
        CheckEquivalent(c, 0, {}, dram);
        // Unsupported emitted method
        c.clear();
        AppendInstruction(c, InsSpec{.imm = {0x250, 1}, .out_method = {5, 0}, .out_emit = {6, 0}});
        CheckEquivalent(c, 0, {}, dram, 1, 0x250);
    }

    // 6. Randomized straight-line and predicated MME programs.
    std::mt19937 rng{0xC0FFEE42U};
    constexpr std::array<u32, 22> safe_ops{
        0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 30};
    constexpr std::array<u32, 9> valid_out{0, 1, 2, 3, 4, 5, 6, 8, 9};
    for (u32 trial = 0; trial < 300; ++trial) {
        CAPTURE(trial);
        std::vector<Word> code;
        // Initialize r0..r3 from FIFO and immediates
        AppendInstruction(code, InsSpec{
                                    .op = {0, 0},
                                    .dst = {0, 1},
                                    .src_a = {28, 29},
                                    .src_b = {25, 26},
                                    .imm = {rng() & 0xffffU, rng() & 0xffffU},
                                });
        AppendInstruction(code, InsSpec{
                                    .op = {0, 15},
                                    .dst = {2, 3},
                                    .src_a = {28, 0},
                                    .src_b = {27, 1},
                                    .imm = {rng() & 0xffffU, rng() & 0xffffU},
                                });
        const u32 body_len = 3 + (rng() % 8);
        for (u32 step = 0; step < body_len; ++step) {
            InsSpec ins{};
            ins.end = (step + 1 == body_len);
            ins.pred_mode = rng() % 16;
            ins.pred_reg = rng() % 4;
            for (u32 lane = 0; lane < 2; ++lane) {
                ins.op[lane] = safe_ops[rng() % safe_ops.size()];
                ins.dst[lane] = (rng() % 5 == 0) ? 24U : (rng() % 4);
                ins.src_a[lane] = (ins.op[lane] == 30) ? 25U : (rng() % 4);
                constexpr std::array<u32, 6> extra_srcs{24, 25, 26, 27, 28, 29};
                ins.src_b[lane] = (rng() % 2 == 0) ? (rng() % 4) : extra_srcs[rng() % 6];
                ins.imm[lane] = (ins.op[lane] == 30) ? (rng() % 64) : (rng() & 0xffffU);
                ins.out_method[lane] = (rng() % 3 == 0) ? (1 + rng() % 6) : 0;
                ins.out_emit[lane] = valid_out[rng() % valid_out.size()];
            }
            AppendInstruction(code, ins);
        }
        // Delay slot after end
        AppendInstruction(code, InsSpec{
                                    .op = {31, 0},
                                    .dst = {24, 0},
                                    .src_a = {25, 0},
                                    .src_b = {2, 3},
                                    .imm = {trial % 32, 0},
                                    .out_emit = {0, 2},
                                });
        std::vector<Word> params(32);
        for (Word& p : params) {
            p = rng();
        }
        CheckEquivalent(code, 0, params, dram, trial + 10);
    }
}
