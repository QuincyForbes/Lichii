// SPDX-License-Identifier: GPL-2.0-or-later
// Isolated little-endian A64 decoded-block experiment.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

enum class Op : uint8_t { Add, Sub, Load, Store, Cbnz, Ret, Svc, Invalid };
struct Insn {
    Op op;
    unsigned rd, rn;
    uint64_t imm;
};

static Insn Decode(uint32_t w, uint64_t pc) {
    const unsigned rd = w & 31, rn = (w >> 5) & 31;
    const unsigned imm = (w >> 10) & 4095;
    if ((w & 0xbf800000) == 0x91000000 && rd != 31 && rn != 31)
        return {(w & (1U << 30)) ? Op::Sub : Op::Add, rd, rn, imm << (12 * ((w >> 22) & 1))};
    if ((w & 0xff800000) == 0xb9000000 && rd != 31 && rn != 31)
        return {(w & (1U << 22)) ? Op::Load : Op::Store, rd, rn, imm * 4};
    if ((w & 0xff000000) == 0xb5000000 && rd != 31) {
        const int offset = static_cast<int>(((w >> 5) & 0x7ffff) ^ 0x40000) - 0x40000;
        return {Op::Cbnz, rd, 0, pc + offset * 4};
    }
    if ((w & 0xfffffc1f) == 0xd65f0000 && rn != 31)
        return {Op::Ret, 0, rn, 0};
    if ((w & 0xffe0001f) == 0xd4000001)
        return {Op::Svc, 0, 0, (w >> 5) & 0xffffU};
    return {Op::Invalid, 0, 0, 0};
}

struct Block {
    uint64_t start;
    unsigned count = 0;
    std::array<Insn, 16> insns;
};

struct Machine {
    std::vector<uint8_t> mem;
    std::array<uint64_t, 31> x{};
    uint64_t pc = 0, retired = 0, decodes = 0, svc = 0;
    std::array<Block, 64> cache{};

    bool Fits(uint64_t addr) const {
        return mem.size() >= 4 && addr <= mem.size() - 4;
    }
    void Fill(Block& block) {
        block.start = pc;
        block.count = 0;
        for (uint64_t addr = pc; block.count < block.insns.size() && Fits(addr); addr += 4) {
            uint32_t word;
            std::memcpy(&word, &mem[addr], 4);
            auto insn = Decode(word, addr);
            block.insns[block.count++] = insn;
            ++decodes;
            if (insn.op != Op::Add && insn.op != Op::Sub && insn.op != Op::Load)
                break;
        }
    }
    void Write(uint64_t addr, uint32_t word) {
        std::memcpy(&mem[addr], &word, 4);
        for (auto& b : cache) {
            if (addr < b.start + 4 * b.count && b.start < addr + 4)
                b.count = 0;
        }
    }
    const char* Run(bool cached, uint64_t budget) {
        while (budget) {
            if ((pc & 3) || !Fits(pc))
                return "fetch";
            auto& b = cache[(pc / 4) % cache.size()];
            if (!cached || !b.count || b.start != pc)
                Fill(b);
            const unsigned count = static_cast<unsigned>(std::min<uint64_t>(budget, b.count));
            for (unsigned i = 0; i < count; ++i) {
                const auto& d = b.insns[i];
                uint64_t next = pc + 4;
                switch (d.op) {
                case Op::Add: x[d.rd] = x[d.rn] + d.imm; break;
                case Op::Sub: x[d.rd] = x[d.rn] - d.imm; break;
                case Op::Cbnz: if (x[d.rd]) next = d.imm; break;
                case Op::Ret: next = x[d.rn]; break;
                case Op::Store:
                case Op::Load: {
                    const uint64_t addr = x[d.rn] + d.imm;
                    if (!Fits(addr))
                        return "data";
                    if (d.op == Op::Store) {
                        Write(addr, static_cast<uint32_t>(x[d.rd]));
                    } else {
                        uint32_t value;
                        std::memcpy(&value, &mem[addr], 4);
                        x[d.rd] = value;
                    }
                    break;
                }
                case Op::Invalid: return "unsupported";
                case Op::Svc: svc = d.imm; ++retired; pc = next; return "svc";
                }
                pc = next;
                ++retired;
            }
            budget -= count;
        }
        return "budget";
    }
};

int main(int argc, char** argv) {
    if (argc != 7 || (std::string(argv[2]) != "cached" && std::string(argv[2]) != "decode")) {
        std::fprintf(stderr, "usage: runner text.bin {cached|decode} budget x0 x1 x2\n");
        return 2;
    }
    std::ifstream file(argv[1], std::ios::binary);
    if (!file)
        return 2;
    Machine m;
    m.mem.assign(std::istreambuf_iterator<char>(file), {});
    uint64_t budget;
    try {
        budget = std::stoull(argv[3]);
        for (unsigned i = 0; i < 3; ++i)
            m.x[i] = std::stoull(argv[4 + i]);
    } catch (const std::exception&) {
        return 2;
    }
    // Test ABI: a top-level RET enters the final word of the supplied image.
    m.x[30] = m.mem.size() - 4;
    const auto start = std::chrono::steady_clock::now();
    const char* stop = m.Run(std::string(argv[2]) == "cached", budget);
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count();
    std::printf("%s %llu %llu %llu %llu %llu %lld\n", stop,
                (unsigned long long)m.pc, (unsigned long long)m.x[0], (unsigned long long)m.svc,
                (unsigned long long)m.retired, (unsigned long long)m.decodes, (long long)ns);
}
