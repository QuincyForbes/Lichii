// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <chrono>
#include <cstring>
#include <vector>

#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <catch2/catch_test_macros.hpp>
#include <fmt/core.h>
#include <oaknut/oaknut.hpp>

#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/interface/exclusive_monitor.h"
#include "dynarmic/interface/optimization_flags.h"
#include "dynarmic/tests/A64/testenv.h"

using namespace Dynarmic;
using namespace oaknut::util;

namespace {

class BenchFastmemEnv final : public A64::UserCallbacks {
public:
    u8* mem = nullptr;
    std::size_t mem_size = 0;
    u64 fault_page_vaddr = ~u64{0};
    std::size_t fault_page_size = 0;
    u8* fault_shadow = nullptr;
    A64::Jit* jit = nullptr;
    u64 svc_calls = 0;
    u64 cntpct_calls = 0;
    u64 callback_reads = 0;
    u64 callback_writes = 0;

    BenchFastmemEnv(u8* mem_, std::size_t mem_size_) : mem(mem_), mem_size(mem_size_) {}

    u8* ResolvePtr(u64 vaddr) {
        if (fault_shadow != nullptr && vaddr >= fault_page_vaddr &&
            vaddr < fault_page_vaddr + fault_page_size) {
            return fault_shadow + (vaddr - fault_page_vaddr);
        }
        ASSERT(vaddr < mem_size);
        return mem + vaddr;
    }

    std::optional<std::uint32_t> MemoryReadCode(u64 vaddr) override {
        if (vaddr + 4 > mem_size) {
            return std::nullopt;
        }
        u32 inst = 0;
        std::memcpy(&inst, mem + vaddr, sizeof(u32));
        return inst;
    }

    u64 MemoryRead(u64 vaddr, std::size_t size) override {
        ++callback_reads;
        u64 val = 0;
        std::memcpy(&val, ResolvePtr(vaddr), size);
        return val;
    }

    Vector MemoryRead128(u64 vaddr) override {
        ++callback_reads;
        Vector val{};
        std::memcpy(val.data(), ResolvePtr(vaddr), sizeof(val));
        return val;
    }

    void MemoryWrite(u64 vaddr, u64 value, std::size_t size) override {
        ++callback_writes;
        std::memcpy(ResolvePtr(vaddr), &value, size);
    }

    void MemoryWrite128(u64 vaddr, Vector value) override {
        ++callback_writes;
        std::memcpy(ResolvePtr(vaddr), value.data(), sizeof(value));
    }

    bool MemoryWriteExclusive8(u64 vaddr, u8 value, u8) override {
        MemoryWrite(vaddr, value, 1);
        return true;
    }
    bool MemoryWriteExclusive16(u64 vaddr, u16 value, u16) override {
        MemoryWrite(vaddr, value, 2);
        return true;
    }
    bool MemoryWriteExclusive32(u64 vaddr, u32 value, u32) override {
        MemoryWrite(vaddr, value, 4);
        return true;
    }
    bool MemoryWriteExclusive64(u64 vaddr, u64 value, u64) override {
        MemoryWrite(vaddr, value, 8);
        return true;
    }
    bool MemoryWriteExclusive128(u64 vaddr, Vector value, Vector) override {
        MemoryWrite128(vaddr, value);
        return true;
    }

    void CallSVC(u32 swi) override {
        ++svc_calls;
        if (swi == 0 || swi == 0x0B) {
            jit->HaltExecution(HaltReason::UserDefined1);
        }
    }

    void ExceptionRaised(u64, A64::Exception) override {
        UNREACHABLE();
    }

    void AddTicks(u64) override {}
    u64 GetTicksRemaining() override { return 1'000'000; }
    u64 GetCNTPCT() override { return ++cntpct_calls * 192; }
};

struct CpuSnapshot {
    std::array<u64, 31> gpr{};
    std::array<Vector, 32> vec{};
    u64 sp = 0;
    u32 nzcv = 0;
    bool operator==(const CpuSnapshot&) const = default;
};

CpuSnapshot CaptureSnapshot(const A64::Jit& jit) {
    return CpuSnapshot{
        .gpr = jit.GetRegisters(),
        .vec = jit.GetVectors(),
        .sp = jit.GetSP(),
        .nzcv = jit.GetPstate(),
    };
}

}  // namespace

TEST_CASE("A64: oversized 512 MiB code_cache_size clamps cleanly", "[a64][arm64_opt]") {
    A64TestEnv env;
    oaknut::VectorCodeGenerator code{env.code_mem, nullptr};
    code.ADD(W0, W1, W2);
    oaknut::Label end;
    code.l(end);
    code.B(end);

    A64::UserConfig conf{};
    conf.callbacks = &env;
    conf.code_cache_size = 512ULL * 1024 * 1024;
    A64::Jit jit{conf};

    jit.SetRegister(1, 0xFFFF'FFFFULL);
    jit.SetRegister(2, 5);
    jit.SetPC(0);
    env.ticks_left = 2;
    jit.Run();

    // Verify 32-bit wrap zero-extends upper 32 bits of X0.
    REQUIRE(jit.GetRegister(0) == 4);
}

TEST_CASE("A64: fastmem offset folding, shared base, scalar FP, CheckBit, and fault fallback", "[a64][arm64_opt]") {
#ifndef _WIN32
    const std::size_t page_size = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    const std::size_t arena_size = page_size * 16;
    void* raw = mmap(nullptr, arena_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    REQUIRE(raw != MAP_FAILED);

    const std::size_t fault_span = page_size * 2;
    std::vector<u8> shadow_page(fault_span, 0);
    const u64 code_base = 0;
    const u64 data_base = page_size * 4;
    const u64 sp_base = page_size * 8;
    const u64 fault_base = page_size * 12;

    // Protect fault_base so fastmem loads/stores there trap into the fallback path.
    REQUIRE(mprotect(static_cast<u8*>(raw) + fault_base, fault_span, PROT_NONE) == 0);

    const auto run_mode = [&](bool use_fastmem) -> std::pair<CpuSnapshot, std::vector<u8>> {
        std::memset(static_cast<u8*>(raw), 0, fault_base);
        std::fill(shadow_page.begin(), shadow_page.end(), 0);
        for (std::size_t i = 0; i < 512; ++i) {
            static_cast<u8*>(raw)[data_base + i] = static_cast<u8>((i * 13 + 7) & 0xFF);
            static_cast<u8*>(raw)[data_base + 4096 + i] = static_cast<u8>((i * 29 + 3) & 0xFF);
            shadow_page[i] = static_cast<u8>((i * 17 + 11) & 0xFF);
            shadow_page[4096 + (i % 256)] = static_cast<u8>((i * 31 + 5) & 0xFF);
        }
        // Write valid floats at data_base + 0x100 and shadow_page + 0x100
        const std::array<float, 8> init_floats = {1.5f, -2.25f, 3.75f, 4.0f, 0.5f, -1.0f, 2.0f, 8.5f};
        std::memcpy(static_cast<u8*>(raw) + data_base + 0x100, init_floats.data(), sizeof(init_floats));
        std::memcpy(shadow_page.data() + 0x100, init_floats.data(), sizeof(init_floats));

        std::vector<u32> words;
        oaknut::VectorCodeGenerator c{words, nullptr};
        oaknut::Label cbz_taken, tbz_not_taken, loop_top, after_eq;

        // 1. Frame prologue + multiple positive/negative/large offsets off X19 (shared fastmem base)
        c.STP(X29, X30, SP, PRE_INDEXED, -48);
        c.STP(X19, X20, SP, 16);
        c.MOV(X19, X0); // data_base + 256
        c.LDR(X2, X19, 0);
        c.LDR(X3, X19, 8);
        c.LDUR(X4, X19, -16);
        c.LDUR(W5, X19, -4);
        c.LDR(X6, X19, 4104); // > AddSubImm 12-bit unshifted range, exercises fallback large disp
        c.SUB(W7, W5, 1);     // 32-bit zext elision check
        c.STR(W7, X19, 32);
        c.STUR(X2, X19, -32);
        // Store where base == value (X8 stored to [X8, #16])
        c.ADD(X8, X19, 64);
        c.STR(X8, X8, 16);

        // 2. Host call (CNTPCT) while shared base X19 has another memory access afterwards
        c.MRS(X9, oaknut::SystemReg::CNTPCT_EL0);
        c.STR(X3, X19, 40);

        // 3. Scalar FP loads, FMADD, FCSEL, lane reads, and scalar FP stores
        c.LDR(S0, X19);
        c.LDR(S1, X19, 4);
        c.LDR(S2, X19, 8);
        c.FMADD(S3, S0, S1, S2);
        c.CMP(W5, W5); // Z = 1 (EQ)
        c.FCSEL(S4, S3, S1, EQ);
        c.STR(S4, X19, 16);
        c.LDP(S5, S6, X19, 0);
        c.FADD(S7, S5, S6);
        c.STP(S7, S4, X19, 20);

        // 4. SIMD ReadWrite move elimination (chained FMLA)
        c.LDR(Q16, X19, 0);
        c.LDR(Q17, X19, 16);
        c.FMLA(V16.S4(), V17.S4(), V17.S4());
        c.STR(Q16, X19, 48);

        // 5. CBZ / TBZ preserving live NZCV from an earlier CMP!
        c.CMP(X2, X2); // sets Z=1, C=1 (EQ)
        c.MOV(X10, 0);
        c.CBZ(X10, cbz_taken);
        c.MOV(X11, 999);
        c.l(cbz_taken);
        // NZCV from CMP(X2, X2) must still be EQ here!
        c.CSET(X11, EQ);

        c.MOV(X12, 1ULL << 45);
        c.TBZ(X12, 45, tbz_not_taken);
        c.ADD(X11, X11, 10);
        c.l(tbz_not_taken);

        // 6. Faulting fastmem accesses on PROT_NONE page (X1 = fault_base + 256):
        // exercises shared base, negative LDUR/STUR, large +4104 disp, and FPR S-load/store fallbacks!
        c.MOV(X20, X1);
        c.LDR(X13, X20, 0);
        c.LDUR(X14, X20, -16);
        c.LDR(X15, X20, 4104);
        c.LDR(S8, X20, 0);
        c.FADD(S8, S8, S0);
        c.STR(S8, X20, 8);
        c.STUR(X13, X20, -24);

        // 7. Loop with SP adjustment before B.NE (tests nzcv_live_pos across SetSP)
        c.MOV(W16, 4);
        c.l(loop_top);
        c.ADD(X11, X11, 3);
        c.SUBS(W16, W16, 1);
        c.B(NE, loop_top);

        c.CMP(X11, 23);
        c.LDP(X19, X20, SP, 16);
        c.LDP(X29, X30, SP, POST_INDEXED, 48); // SetSP while EQ flag is live!
        c.B(EQ, after_eq);
        c.MOV(X11, 0);
        c.l(after_eq);
        c.SVC(0);

        std::memcpy(static_cast<u8*>(raw) + code_base, words.data(), words.size() * sizeof(u32));

        BenchFastmemEnv env{static_cast<u8*>(raw), arena_size};
        env.fault_page_vaddr = fault_base;
        env.fault_page_size = fault_span;
        env.fault_shadow = shadow_page.data();

        A64::UserConfig conf{};
        conf.callbacks = &env;
        conf.enable_cycle_counting = false;
        conf.wall_clock_cntpct = true;
        conf.unsafe_optimizations = true;
        conf.optimizations |= OptimizationFlag::FastDispatch | OptimizationFlag::ReturnStackBuffer |
                              OptimizationFlag::BlockLinking | OptimizationFlag::Unsafe_IgnoreGlobalMonitor;
        if (use_fastmem) {
            conf.fastmem_pointer = reinterpret_cast<std::uintptr_t>(raw);
            conf.fastmem_address_space_bits = 64;
            conf.recompile_on_fastmem_failure = true;
        }

        A64::Jit jit{conf};
        env.jit = &jit;
        jit.SetRegister(0, data_base + 256);
        jit.SetRegister(1, fault_base + 256);
        jit.SetRegister(19, 0x1111'2222'3333'4444ULL);
        jit.SetRegister(20, 0x5555'6666'7777'8888ULL);
        jit.SetVector(4, {~u64{0}, ~u64{0}});
        jit.SetVector(8, {~u64{0}, ~u64{0}});
        jit.SetSP(sp_base + 0x800);
        jit.SetPC(code_base);

        // Run twice in fastmem mode so recompiled post-fault blocks also run.
        jit.Run();
        const CpuSnapshot snap = CaptureSnapshot(jit);
        std::vector<u8> mem_out(512);
        std::memcpy(mem_out.data(), static_cast<u8*>(raw) + data_base, mem_out.size());
        return {snap, mem_out};
    };

    const auto [ref_snap, ref_mem] = run_mode(false);
    const auto [opt_snap, opt_mem] = run_mode(true);

    REQUIRE(opt_snap.gpr == ref_snap.gpr);
    REQUIRE(opt_snap.vec == ref_snap.vec);
    REQUIRE(opt_snap.sp == ref_snap.sp);
    REQUIRE(opt_snap.nzcv == ref_snap.nzcv);
    REQUIRE(opt_mem == ref_mem);
    REQUIRE(opt_snap.gpr[11] == 23);
    // Upper 96 bits of V4 and V8 must be zeroed by scalar S writes.
    REQUIRE((opt_snap.vec[4][0] >> 32) == 0);
    REQUIRE(opt_snap.vec[4][1] == 0);
    REQUIRE((opt_snap.vec[8][0] >> 32) == 0);
    REQUIRE(opt_snap.vec[8][1] == 0);

    munmap(raw, arena_size);
#endif
}

TEST_CASE("A64: Kokiri-style scene loop & SVC dispatch microbenchmark", "[a64][.][bench]") {
#ifndef _WIN32
    const std::size_t arena_size = 2 * 1024 * 1024;
    void* raw = mmap(nullptr, arena_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    REQUIRE(raw != MAP_FAILED);
    std::memset(raw, 0, arena_size);

    const u64 code_base = 0;
    const u64 svc_stub_pc = 0x1000;
    const u64 data_base = 0x100000;
    const u64 sp_base = 0x180000;

    // Initialize float & integer scene data
    float* fdata = reinterpret_cast<float*>(static_cast<u8*>(raw) + data_base);
    for (int i = 0; i < 64; ++i) {
        fdata[i] = 0.01f * static_cast<float>(i + 1);
    }

    // Scene update loop (single hot basic block: shared-base struct loads/stores, scalar FP
    // loads/FMADD/FCSEL/stores, chained SIMD FMLA, 32-bit zext elision, SUBS+B.NE flag reuse)
    {
        std::vector<u32> words;
        oaknut::VectorCodeGenerator c{words, nullptr};
        oaknut::Label loop;

        c.STP(X29, X30, SP, PRE_INDEXED, -48);
        c.STP(X19, X20, SP, 16);
        c.MOV(X19, X0); // data_base + 128
        c.MOV(W20, W1); // iteration count

        c.l(loop);
        // 8 struct loads/stores off shared base X19 + 32-bit zext elision
        c.LDR(W2, X19, 0);
        c.LDR(W3, X19, 4);
        c.LDR(W4, X19, 8);
        c.LDUR(W5, X19, -8);
        c.ADD(W2, W2, 1);
        c.EOR(W3, W3, W2);
        c.ADD(W4, W4, W3);
        c.SUB(W5, W5, W2);
        c.STR(W2, X19, 0);
        c.STR(W3, X19, 4);
        c.STR(W4, X19, 8);
        c.STUR(W5, X19, -8);

        // Scalar FP transform (LDR S + FMADD S + FADD S + FCSEL S + STR S)
        c.LDR(S0, X19, 16);
        c.LDR(S1, X19, 20);
        c.LDR(S2, X19, 24);
        c.LDR(S3, X19, 28);
        c.FMADD(S4, S0, S1, S2);
        c.FMADD(S5, S4, S3, S1);
        c.CMP(W2, 0);
        c.FCSEL(S6, S5, S4, NE);
        c.STR(S4, X19, 72);
        c.STR(S6, X19, 76);

        // Chained SIMD 4x4 matrix-vector FMLA (ReadWrite move elimination)
        c.LDR(Q7, X19, 32);
        c.LDR(Q16, X19, 48);
        c.FMLA(V7.S4(), V16.S4(), V16.S4());
        c.FMLA(V7.S4(), V16.S4(), V16.S4());
        c.STR(Q7, X19, 80);

        c.SUBS(W20, W20, 1);
        c.B(NE, loop);

        c.LDP(X19, X20, SP, 16);
        c.LDP(X29, X30, SP, POST_INDEXED, 48);
        c.SVC(0);

        std::memcpy(static_cast<u8*>(raw) + code_base, words.data(), words.size() * sizeof(u32));
    }

    // Populate 32,768 cold SVC stubs to size block_entries like a real game (~MBs of block tables)
    constexpr u32 kNumStubs = 32768;
    {
        std::vector<u32> stubs;
        stubs.reserve(kNumStubs * 2);
        oaknut::VectorCodeGenerator c{stubs, nullptr};
        for (u32 i = 0; i < kNumStubs; ++i) {
            c.SVC(0x0B);
            c.RET();
        }
        std::memcpy(static_cast<u8*>(raw) + svc_stub_pc, stubs.data(), stubs.size() * sizeof(u32));
    }

    BenchFastmemEnv env{static_cast<u8*>(raw), arena_size};
    A64::UserConfig conf{};
    conf.callbacks = &env;
    conf.enable_cycle_counting = false;
    conf.wall_clock_cntpct = true;
    conf.unsafe_optimizations = true;
    conf.halt_check_on_back_edges = true;
    conf.fast_dispatch_table_bits = 16;
    conf.optimizations |= OptimizationFlag::FastDispatch | OptimizationFlag::ReturnStackBuffer |
                          OptimizationFlag::BlockLinking | OptimizationFlag::Unsafe_IgnoreGlobalMonitor;
    conf.fastmem_pointer = reinterpret_cast<std::uintptr_t>(raw);
    conf.fastmem_address_space_bits = 64;

    A64::Jit jit{conf};
    env.jit = &jit;

    // Warmup + compile all 32,768 blocks so block_entries is realistically populated
    for (u32 i = 0; i < kNumStubs; ++i) {
        const u64 pc = svc_stub_pc + u64{i} * 8;
        jit.SetPC(pc);
        jit.SetRegister(30, pc);
        jit.Run();
        jit.ClearHalt(HaltReason::UserDefined1);
    }

    // Timed scene loop (2,000,000 iterations)
    constexpr u32 kLoopIters = 2'000'000;
    jit.SetRegister(0, data_base + 128);
    jit.SetRegister(1, kLoopIters);
    jit.SetSP(sp_base);
    jit.SetPC(code_base);
    const auto t0 = std::chrono::steady_clock::now();
    jit.Run();
    const auto t1 = std::chrono::steady_clock::now();
    jit.ClearHalt(HaltReason::UserDefined1);

    // Timed SVC round-trip loop across 8 hot SVC stubs (SleepThread, WaitForAddress, SignalToAddress, ArbitrateLock/Unlock...)
    constexpr u32 kSvcIters = 500'000;
    const auto t2 = std::chrono::steady_clock::now();
    for (u32 i = 0; i < kSvcIters; ++i) {
        const u64 pc = svc_stub_pc + u64{(i & 7) * 4096};
        jit.SetPC(pc);
        jit.SetRegister(30, pc);
        jit.Run();
        jit.ClearHalt(HaltReason::UserDefined1);
    }
    const auto t3 = std::chrono::steady_clock::now();

    const double scene_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double svc_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    fmt::print("[arm64_bench] scene_loop ({} iters): {:.2f} ms ({:.2f} ns/iter) | svc_roundtrip ({} calls): {:.2f} ms ({:.2f} ns/call)\n",
               kLoopIters, scene_ms, (scene_ms * 1e6) / kLoopIters,
               kSvcIters, svc_ms, (svc_ms * 1e6) / kSvcIters);

    munmap(raw, arena_size);
#endif
}

TEST_CASE("A64: transient read-trap load survival and separate store block fault markers", "[a64][arm64_opt]") {
#ifndef _WIN32
    const std::size_t host_page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    const std::size_t arena_size = host_page * 16;
    void* raw = mmap(nullptr, arena_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    REQUIRE(raw != MAP_FAILED);
    std::memset(raw, 0, arena_size);

    const u64 code_base = 0;
    const u64 read_vaddr = host_page * 4;
    const u64 write_vaddr = host_page * 8;

    std::vector<u8> write_shadow(host_page, 0);
    const u64 expected0 = 0x11223344'55667788ULL;
    const u64 expected1 = 0x99AABBCC'DDEEFF00ULL;
    std::memcpy(static_cast<u8*>(raw) + read_vaddr, &expected0, sizeof(expected0));
    std::memcpy(static_cast<u8*>(raw) + read_vaddr + 8, &expected1, sizeof(expected1));

    // Software page table where both read_vaddr and write_vaddr are marked RasterizerCachedMemory
    // (0b111): if plain loads were evicted from fastmem onto the page table (without bitmap),
    // every load would bounce to MemoryRead!
    constexpr u32 kPtBits = 28;
    constexpr u64 kAttrMask = ((u64{1} << 44) - 1) << 12;
    constexpr u8 kSignBit = 55;
    std::vector<u64> pt(std::size_t{1} << (kPtBits - 12), 0);
    const auto map_pt = [&](u64 base_vaddr, u8* backing, u64 low_bits) {
        for (std::size_t off = 0; off < host_page; off += 4096) {
            const u64 page_vaddr = base_vaddr + off;
            const std::uintptr_t delta =
                reinterpret_cast<std::uintptr_t>(backing + off) - static_cast<std::uintptr_t>(page_vaddr);
            pt[page_vaddr >> 12] = (static_cast<u64>(delta) & kAttrMask) | low_bits;
        }
    };
    map_pt(read_vaddr, static_cast<u8*>(raw) + read_vaddr, 0b111);
    map_pt(write_vaddr, write_shadow.data(), 0b111);

    // Block has 2 stores to write_vaddr (which stays PROT_READ forever -> 2 store faults!)
    // AND 2 loads from read_vaddr (which starts PROT_NONE as a transient read trap, disarmed to
    // PROT_READ on its very first MemoryRead callback!).
    {
        std::vector<u32> words;
        oaknut::VectorCodeGenerator c{words, nullptr};
        c.STR(X2, X1, 0);
        c.STR(X3, X1, 8);
        c.LDR(X4, X0, 0);
        c.LDR(X5, X0, 8);
        c.SVC(0);
        std::memcpy(static_cast<u8*>(raw) + code_base, words.data(), words.size() * sizeof(u32));
    }

    REQUIRE(mprotect(static_cast<u8*>(raw) + read_vaddr, host_page, PROT_NONE) == 0);
    REQUIRE(mprotect(static_cast<u8*>(raw) + write_vaddr, host_page, PROT_READ) == 0);

    class TransientTrapEnv final : public A64::UserCallbacks {
    public:
        u8* arena = nullptr;
        std::size_t arena_bytes = 0;
        u64 trap_vaddr = 0;
        std::size_t trap_bytes = 0;
        u64 write_page = 0;
        u8* write_backing = nullptr;
        A64::Jit* jit = nullptr;
        u64 read_cbs = 0;
        u64 write_cbs = 0;

        std::optional<u32> MemoryReadCode(u64 vaddr) override {
            u32 inst = 0;
            std::memcpy(&inst, arena + vaddr, sizeof(u32));
            return inst;
        }
        u64 MemoryRead(u64 vaddr, std::size_t size) override {
            ++read_cbs;
            if (vaddr >= trap_vaddr && vaddr < trap_vaddr + trap_bytes) {
                // Disarm the one-shot read trap just like Core::Memory::DisarmReadTraps.
                const int rc = mprotect(arena + trap_vaddr, trap_bytes, PROT_READ);
                ASSERT(rc == 0);
            }
            u64 val = 0;
            std::memcpy(&val, arena + vaddr, size);
            return val;
        }
        Vector MemoryRead128(u64 vaddr) override {
            return {MemoryRead(vaddr, 8), MemoryRead(vaddr + 8, 8)};
        }
        void MemoryWrite(u64 vaddr, u64 value, std::size_t size) override {
            ++write_cbs;
            std::memcpy(write_backing + (vaddr - write_page), &value, size);
        }
        void MemoryWrite128(u64 vaddr, Vector value) override {
            MemoryWrite(vaddr, value[0], 8);
            MemoryWrite(vaddr + 8, value[1], 8);
        }
        bool MemoryWriteExclusive8(u64, u8, u8) override { return false; }
        bool MemoryWriteExclusive16(u64, u16, u16) override { return false; }
        bool MemoryWriteExclusive32(u64, u32, u32) override { return false; }
        bool MemoryWriteExclusive64(u64, u64, u64) override { return false; }
        bool MemoryWriteExclusive128(u64, Vector, Vector) override { return false; }
        void CallSVC(u32) override { jit->HaltExecution(HaltReason::UserDefined1); }
        void ExceptionRaised(u64, A64::Exception) override { UNREACHABLE(); }
        void AddTicks(u64) override {}
        u64 GetTicksRemaining() override { return 10000; }
        u64 GetCNTPCT() override { return 0; }
    } env;

    env.arena = static_cast<u8*>(raw);
    env.arena_bytes = arena_size;
    env.trap_vaddr = read_vaddr;
    env.trap_bytes = host_page;
    env.write_page = write_vaddr;
    env.write_backing = write_shadow.data();

    A64::UserConfig conf{};
    conf.callbacks = &env;
    conf.enable_cycle_counting = false;
    conf.unsafe_optimizations = true;
    conf.page_table = reinterpret_cast<void**>(pt.data());
    conf.page_table_address_space_bits = kPtBits;
    conf.page_table_pointer_mask = kAttrMask;
    conf.page_table_marked_bit = u8{0};
    conf.page_table_sign_extension = kSignBit;
    conf.silently_mirror_page_table = false;
    conf.absolute_offset_page_table = true;
    conf.fastmem_pointer = reinterpret_cast<std::uintptr_t>(raw);
    conf.fastmem_address_space_bits = 64;
    conf.recompile_on_fastmem_failure = true;
    conf.fastmem_shared_markers = false;
    conf.fastmem_block_after_faults = 2;
    conf.fastmem_load_fault_limit = 8;

    A64::Jit jit{conf};
    env.jit = &jit;

    for (int iter = 0; iter < 101; ++iter) {
        jit.SetPC(code_base);
        jit.SetRegister(0, read_vaddr);
        jit.SetRegister(1, write_vaddr);
        jit.SetRegister(2, 0xAAAA'0000ULL + static_cast<u64>(iter));
        jit.SetRegister(3, 0xBBBB'0000ULL + static_cast<u64>(iter));
        jit.SetRegister(4, 0);
        jit.SetRegister(5, 0);
        jit.Run();
        jit.ClearHalt(HaltReason::UserDefined1);
        REQUIRE(jit.GetRegister(4) == expected0);
        REQUIRE(jit.GetRegister(5) == expected1);
    }

    // Only the very first trapped LDR called MemoryRead; all subsequent 201 loads ran in fastmem!
    REQUIRE(env.read_cbs == 1);
    REQUIRE(env.write_cbs == 202);

    munmap(raw, arena_size);
#endif
}

TEST_CASE("A64: untrapped_read_bitmap_ptr bypasses callbacks on clean RasterizerCachedMemory pages", "[a64][arm64_opt]") {
    constexpr u32 kPtBits = 39;
    constexpr u64 kAttrMask = ((u64{1} << 44) - 1) << 12;
    constexpr u8 kSignBit = 55;
    // Test high addresses within 39-bit space (e.g. 0x70'0000'0000) so negative deltas also get
    // sign-extended if backing is lower in memory.
    constexpr u64 kCachedPage = 0x22'6FF3'0000ULL;
    constexpr u64 kDebugPage = 0x22'6FF3'1000ULL;
    constexpr u64 kUnmappedPage = 0x22'6FF3'2000ULL;

    alignas(4096) std::array<u8, 4096> cached_backing{};
    alignas(4096) std::array<u8, 4096> debug_backing{};
    for (std::size_t i = 0; i < 4096; ++i) {
        cached_backing[i] = static_cast<u8>((i * 7 + 3) & 0xFF);
        debug_backing[i] = static_cast<u8>((i * 11 + 5) & 0xFF);
    }

    for (const bool use_lookup_calls : {false, true}) {
        class PtBitmapEnv final : public A64::UserCallbacks {
        public:
            std::vector<u32> code_words;
            u8* cached_mem = nullptr;
            u8* debug_mem = nullptr;
            u64 cached_base = 0;
            u64 debug_base = 0;
            std::vector<u64>* trap_bitmap = nullptr;
            A64::Jit* jit = nullptr;
            u64 read_cbs = 0;
            u64 write_cbs = 0;

            std::optional<u32> MemoryReadCode(u64 vaddr) override {
                const std::size_t idx = static_cast<std::size_t>(vaddr / 4);
                if (idx < code_words.size()) {
                    return code_words[idx];
                }
                return std::nullopt;
            }
            u64 MemoryRead(u64 vaddr, std::size_t size) override {
                ++read_cbs;
                const u64 page = vaddr & ~u64{0xFFF};
                const u64 off = vaddr & 0xFFF;
                if (page == cached_base) {
                    if (trap_bitmap != nullptr) {
                        const u64 p = page >> 12;
                        (*trap_bitmap)[p >> 6] &= ~(u64{1} << (p & 63));
                    }
                    u64 v = 0;
                    std::memcpy(&v, cached_mem + off, size);
                    return v;
                }
                if (page == debug_base) {
                    u64 v = 0;
                    std::memcpy(&v, debug_mem + off, size);
                    return v;
                }
                return 0xDEAD'BEEF'CAFE'BABEULL;
            }
            Vector MemoryRead128(u64 vaddr) override {
                ++read_cbs;
                const u64 page = vaddr & ~u64{0xFFF};
                const u64 off = vaddr & 0xFFF;
                Vector v{};
                if (page == cached_base) {
                    if (trap_bitmap != nullptr) {
                        const u64 p = page >> 12;
                        (*trap_bitmap)[p >> 6] &= ~(u64{1} << (p & 63));
                    }
                    std::memcpy(v.data(), cached_mem + off, 16);
                }
                return v;
            }
            void MemoryWrite(u64 vaddr, u64 value, std::size_t size) override {
                ++write_cbs;
                if ((vaddr & ~u64{0xFFF}) == cached_base) {
                    std::memcpy(cached_mem + (vaddr & 0xFFF), &value, size);
                }
            }
            void MemoryWrite128(u64 vaddr, Vector value) override {
                ++write_cbs;
                if ((vaddr & ~u64{0xFFF}) == cached_base) {
                    std::memcpy(cached_mem + (vaddr & 0xFFF), value.data(), 16);
                }
            }
            bool MemoryWriteExclusive8(u64, u8, u8) override { return false; }
            bool MemoryWriteExclusive16(u64, u16, u16) override { return false; }
            bool MemoryWriteExclusive32(u64, u32, u32) override { return false; }
            bool MemoryWriteExclusive64(u64, u64, u64) override { return false; }
            bool MemoryWriteExclusive128(u64, Vector, Vector) override { return false; }
            void CallSVC(u32) override { jit->HaltExecution(HaltReason::UserDefined1); }
            void ExceptionRaised(u64, A64::Exception) override { UNREACHABLE(); }
            void AddTicks(u64) override {}
            u64 GetTicksRemaining() override { return 10000; }
            u64 GetCNTPCT() override { return 0; }
        } env;

        env.cached_mem = cached_backing.data();
        env.debug_mem = debug_backing.data();
        env.cached_base = kCachedPage;
        env.debug_base = kDebugPage;

        //Sparse-friendly test: allocate page table via mmap (MAP_NORESERVE) for 39-bit space
        std::vector<u64> bitmap((std::size_t{1} << (39 - 12)) / 64, 0);
        env.trap_bitmap = &bitmap;

#ifndef _WIN32
        const std::size_t pt_bytes = (std::size_t{1} << (kPtBits - 12)) * sizeof(u64);
        void* pt_raw = mmap(nullptr, pt_bytes, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANON, -1, 0);
        REQUIRE(pt_raw != MAP_FAILED);
        auto* pt = static_cast<u64*>(pt_raw);
#else
        std::vector<u64> pt_vec(std::size_t{1} << (kPtBits - 12), 0);
        auto* pt = pt_vec.data();
#endif
        const auto encode_pt = [&](u64 page_vaddr, u8* host_ptr, u64 attr) {
            const std::uintptr_t delta =
                reinterpret_cast<std::uintptr_t>(host_ptr) - static_cast<std::uintptr_t>(page_vaddr);
            pt[page_vaddr >> 12] = (static_cast<u64>(delta) & kAttrMask) | attr;
        };
        encode_pt(kCachedPage, cached_backing.data(), 0b111); // RasterizerCachedMemory (marked=1, type=3)
        encode_pt(kDebugPage, debug_backing.data(), 0b101);   // DebugMemory (marked=1, type=2)
        pt[kUnmappedPage >> 12] = 0;                          // Unmapped

        // Program 0 (PC = 0): 8/16/32/64/128-bit reads + 64-bit write on X0
        // Program 1 (PC = 0x40): loop reading X1 words starting at X0 (100 iterations)
        {
            oaknut::VectorCodeGenerator c{env.code_words, nullptr};
            c.LDRB(W2, X0, 0);
            c.LDRH(W3, X0, 2);
            c.LDR(W4, X0, 4);
            c.LDR(X5, X0, 8);
            c.LDR(Q0, X0, 16);
            c.STR(X5, X0, 64);
            c.SVC(0);
            while (env.code_words.size() < 16) {
                c.NOP();
            }
            oaknut::Label loop;
            c.MOV(X2, 0);
            c.l(loop);
            c.LDR(X3, X0);
            c.ADD(X2, X2, X3);
            c.ADD(X0, X0, 8);
            c.SUBS(X1, X1, 1);
            c.B(NE, loop);
            c.SVC(0);
        }

        std::uintptr_t trap_state = 0;

        A64::UserConfig conf{};
        conf.callbacks = &env;
        conf.enable_cycle_counting = false;
        conf.unsafe_optimizations = true;
        conf.page_table = reinterpret_cast<void**>(pt);
        conf.page_table_address_space_bits = kPtBits;
        conf.page_table_pointer_mask = kAttrMask;
        conf.page_table_marked_bit = u8{0};
        conf.page_table_sign_extension = kSignBit;
        conf.untrapped_read_bitmap_ptr = reinterpret_cast<std::uintptr_t>(&trap_state);
        conf.silently_mirror_page_table = false;
        conf.absolute_offset_page_table = true;
        conf.detect_misaligned_access_via_page_table = 16 | 32 | 64 | 128;
        conf.only_detect_misalignment_via_page_table_on_page_boundary = true;
        conf.page_table_lookup_calls = use_lookup_calls;

        A64::Jit jit{conf};
        env.jit = &jit;

        // 1. State == 0 (disabled): all 5 reads + 1 write hit callbacks.
        trap_state = 0;
        jit.SetPC(0);
        jit.SetRegister(0, kCachedPage);
        jit.Run();
        jit.ClearHalt(HaltReason::UserDefined1);
        REQUIRE(env.read_cbs == 5);
        REQUIRE(env.write_cbs == 1);
        const u64 ref_x5 = jit.GetRegister(5);
        const Vector ref_q0 = jit.GetVector(0);

        // 2. State == 1 (enabled, zero traps armed anywhere): 0 read callbacks on
        //    RasterizerCachedMemory, while the store STILL hits MemoryWrite!
        env.read_cbs = 0;
        env.write_cbs = 0;
        trap_state = 1;
        jit.SetPC(0);
        jit.SetRegister(0, kCachedPage);
        jit.SetRegister(5, 0);
        jit.SetVector(0, {0, 0});
        jit.Run();
        jit.ClearHalt(HaltReason::UserDefined1);
        REQUIRE(env.read_cbs == 0);
        REQUIRE(env.write_cbs == 1);
        REQUIRE(jit.GetRegister(5) == ref_x5);
        REQUIRE(jit.GetVector(0) == ref_q0);

        // 3. DebugMemory (0b101) and Unmapped (0b000) MUST still call MemoryRead even when state == 1!
        env.read_cbs = 0;
        jit.SetPC(0);
        jit.SetRegister(0, kDebugPage);
        jit.Run();
        jit.ClearHalt(HaltReason::UserDefined1);
        REQUIRE(env.read_cbs == 5);

        env.read_cbs = 0;
        jit.SetPC(0);
        jit.SetRegister(0, kUnmappedPage);
        jit.Run();
        jit.ClearHalt(HaltReason::UserDefined1);
        REQUIRE(env.read_cbs == 5);

        // 4. State == bitmap pointer with trap armed on kCachedPage:
        //    100 sequential u64 reads on kCachedPage -> 1st read hits MemoryRead (which disarms
        //    the bit), remaining 99 reads execute inline with 0 callbacks!
        const u64 page_idx = kCachedPage >> 12;
        bitmap[page_idx >> 6] |= (u64{1} << (page_idx & 63));
        trap_state = reinterpret_cast<std::uintptr_t>(bitmap.data());
        env.read_cbs = 0;

        u64 expected_sum = 0;
        for (std::size_t i = 0; i < 100; ++i) {
            u64 w = 0;
            std::memcpy(&w, cached_backing.data() + i * 8, sizeof(w));
            expected_sum += w;
        }

        jit.SetPC(0x40);
        jit.SetRegister(0, kCachedPage);
        jit.SetRegister(1, 100);
        jit.Run();
        jit.ClearHalt(HaltReason::UserDefined1);
        REQUIRE(env.read_cbs == 1);
        REQUIRE(jit.GetRegister(2) == expected_sum);

#ifndef _WIN32
        munmap(pt_raw, pt_bytes);
#endif
    }
}

TEST_CASE("A64: non-halting CallSVC clears exclusive_state and continues via RSB", "[a64][arm64_opt]") {
    class NonHaltingSvcEnv final : public A64::UserCallbacks {
    public:
        std::vector<u32> code_words;
        std::array<u8, 4096> mem{};
        ExclusiveMonitor monitor{1};
        A64::Jit* jit = nullptr;
        u32 yield_svcs = 0;

        std::optional<u32> MemoryReadCode(u64 vaddr) override {
            const std::size_t idx = static_cast<std::size_t>(vaddr / 4);
            return idx < code_words.size() ? std::optional<u32>{code_words[idx]} : std::nullopt;
        }
        u64 MemoryRead(u64 vaddr, std::size_t size) override {
            u64 v = 0;
            std::memcpy(&v, mem.data() + (vaddr & 0xFFF), size);
            return v;
        }
        Vector MemoryRead128(u64 vaddr) override {
            return {MemoryRead(vaddr, 8), MemoryRead(vaddr + 8, 8)};
        }
        void MemoryWrite(u64 vaddr, u64 value, std::size_t size) override {
            std::memcpy(mem.data() + (vaddr & 0xFFF), &value, size);
        }
        void MemoryWrite128(u64 vaddr, Vector value) override {
            MemoryWrite(vaddr, value[0], 8);
            MemoryWrite(vaddr + 8, value[1], 8);
        }
        bool MemoryWriteExclusive8(u64, u8, u8) override { return true; }
        bool MemoryWriteExclusive16(u64, u16, u16) override { return true; }
        bool MemoryWriteExclusive32(u64, u32, u32) override { return true; }
        bool MemoryWriteExclusive64(u64 vaddr, u64 value, u64 expected) override {
            u64 cur = MemoryRead(vaddr, 8);
            if (cur != expected) {
                return false;
            }
            MemoryWrite(vaddr, value, 8);
            return true;
        }
        bool MemoryWriteExclusive128(u64, Vector, Vector) override { return true; }
        void CallSVC(u32 swi) override {
            if (swi == 0x0B) {
                ++yield_svcs;
                monitor.ClearProcessor(0);
                return; // Stay in JIT without halting!
            }
            jit->HaltExecution(HaltReason::UserDefined1);
        }
        void ExceptionRaised(u64, A64::Exception) override { UNREACHABLE(); }
        void AddTicks(u64) override {}
        u64 GetTicksRemaining() override { return 10000; }
        u64 GetCNTPCT() override { return 0; }
    } env;

    const u64 initial_val = 0x4242'4242ULL;
    std::memcpy(env.mem.data() + 0x100, &initial_val, sizeof(initial_val));

    {
        oaknut::VectorCodeGenerator c{env.code_words, nullptr};
        c.LDXR(X2, X0);         // Acquire exclusive reservation
        c.SVC(0x0B);            // Non-halting yield SVC -> must clear exclusive_state!
        c.STXR(W3, X4, X0);     // Must fail (W3 = 1) because ERET cleared reservation!
        c.LDXR(X5, X0);         // Fresh reservation after SVC
        c.STXR(W6, X4, X0);     // Must succeed (W6 = 0)!
        c.SVC(0);               // Halt
    }

    A64::UserConfig conf{};
    conf.callbacks = &env;
    conf.global_monitor = &env.monitor;
    conf.processor_id = 0;
    conf.enable_cycle_counting = false;
    conf.unsafe_optimizations = true;
    conf.optimizations |= OptimizationFlag::ReturnStackBuffer | OptimizationFlag::Unsafe_IgnoreGlobalMonitor;

    A64::Jit jit{conf};
    env.jit = &jit;
    jit.SetPC(0);
    jit.SetRegister(0, 0x100);
    jit.SetRegister(4, 0x9999'8888ULL);
    jit.Run();

    REQUIRE(env.yield_svcs == 1);
    REQUIRE(jit.GetRegister(2) == initial_val);
    REQUIRE(jit.GetRegister(3) == 1); // First STXR failed across SVC
    REQUIRE(jit.GetRegister(6) == 0); // Second STXR succeeded
    u64 final_val = 0;
    std::memcpy(&final_val, env.mem.data() + 0x100, sizeof(final_val));
    REQUIRE(final_val == 0x9999'8888ULL);
}
