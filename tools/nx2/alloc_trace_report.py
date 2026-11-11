"""Names the stacks of an alloc_trace dump and adds them up by owner.
   hostdiet-sum.py <trace.json> <dir with lichii.exe + lichii.pdb> [--top N] [--stacks] [--grep text]"""
import ctypes, json, re, sys
from ctypes import wintypes

trace = json.load(open(sys.argv[1]))
exe_dir = sys.argv[2]
TOP = 60
for i, a in enumerate(sys.argv):
    if a == "--top":
        TOP = int(sys.argv[i + 1])
STACKS = "--stacks" in sys.argv
GREP = None
for i, a in enumerate(sys.argv):
    if a == "--grep":
        GREP = sys.argv[i + 1]

dbghelp = ctypes.WinDLL("dbghelp.dll")
H = wintypes.HANDLE(0x1234)
dbghelp.SymSetOptions(0x2 | 0x10)  # UNDNAME | LOAD_LINES
dbghelp.SymInitializeW.argtypes = [wintypes.HANDLE, wintypes.LPCWSTR, wintypes.BOOL]
if not dbghelp.SymInitializeW(H, exe_dir, False):
    sys.exit("SymInitialize failed")
dbghelp.SymLoadModuleExW.restype = ctypes.c_uint64
dbghelp.SymLoadModuleExW.argtypes = [wintypes.HANDLE, wintypes.HANDLE, wintypes.LPCWSTR, wintypes.LPCWSTR,
                                     ctypes.c_uint64, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
BASE = trace["module_base"]
if not dbghelp.SymLoadModuleExW(H, None, exe_dir + r"\lichii.exe", None, BASE, 0, None, 0):
    sys.exit("SymLoadModuleEx failed")


class SYMBOL_INFO(ctypes.Structure):
    _fields_ = [("SizeOfStruct", wintypes.ULONG), ("TypeIndex", wintypes.ULONG), ("Reserved", ctypes.c_uint64 * 2),
                ("Index", wintypes.ULONG), ("Size", wintypes.ULONG), ("ModBase", ctypes.c_uint64),
                ("Flags", wintypes.ULONG), ("Value", ctypes.c_uint64), ("Address", ctypes.c_uint64),
                ("Register", wintypes.ULONG), ("Scope", wintypes.ULONG), ("Tag", wintypes.ULONG),
                ("NameLen", wintypes.ULONG), ("MaxNameLen", wintypes.ULONG), ("Name", ctypes.c_char * 512)]


class LINE64(ctypes.Structure):
    _fields_ = [("SizeOfStruct", wintypes.DWORD), ("Key", ctypes.c_void_p), ("LineNumber", wintypes.DWORD),
                ("FileName", ctypes.c_char_p), ("Address", ctypes.c_uint64)]


dbghelp.SymFromAddr.argtypes = [wintypes.HANDLE, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(SYMBOL_INFO)]
dbghelp.SymGetLineFromAddr64.argtypes = [wintypes.HANDLE, ctypes.c_uint64, ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(LINE64)]
dbghelp.SymAddrIncludeInlineTrace.argtypes = [wintypes.HANDLE, ctypes.c_uint64]
dbghelp.SymAddrIncludeInlineTrace.restype = wintypes.DWORD
dbghelp.SymQueryInlineTrace.argtypes = [wintypes.HANDLE, ctypes.c_uint64, wintypes.DWORD, ctypes.c_uint64, ctypes.c_uint64,
                                        ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD)]
dbghelp.SymFromInlineContext.argtypes = [wintypes.HANDLE, ctypes.c_uint64, wintypes.ULONG, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(SYMBOL_INFO)]
dbghelp.SymGetLineFromInlineContext.argtypes = [wintypes.HANDLE, ctypes.c_uint64, wintypes.ULONG, ctypes.c_uint64,
                                                ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(LINE64)]

cache = {}


def short(path):
    if not path:
        return "?"
    p = path.decode(errors="replace").replace("\\", "/")
    i = p.find("/src/")
    if i >= 0:
        return p[i + 5:]
    return "/".join(p.split("/")[-2:])


def names(addr):
    """Innermost first: inlined functions, then the function itself."""
    if addr in cache:
        return cache[addr]
    out = []
    a = addr - 1
    sym = SYMBOL_INFO()
    sym.SizeOfStruct = 88
    sym.MaxNameLen = 511
    disp = ctypes.c_uint64()
    ldisp = wintypes.DWORD()
    line = LINE64()
    line.SizeOfStruct = ctypes.sizeof(LINE64)
    n = dbghelp.SymAddrIncludeInlineTrace(H, a)
    if n:
        ctx = wintypes.DWORD()
        idx = wintypes.DWORD()
        if dbghelp.SymQueryInlineTrace(H, a, 0, a, a, ctypes.byref(ctx), ctypes.byref(idx)):
            for i in range(n):
                if dbghelp.SymFromInlineContext(H, a, ctx.value + i, ctypes.byref(disp), ctypes.byref(sym)):
                    name = sym.Name.decode(errors="replace")
                    where = ""
                    if dbghelp.SymGetLineFromInlineContext(H, a, ctx.value + i, 0, ctypes.byref(ldisp), ctypes.byref(line)):
                        where = f" ({short(line.FileName)}:{line.LineNumber})"
                    out.append(name + where)
    if dbghelp.SymFromAddr(H, a, ctypes.byref(disp), ctypes.byref(sym)):
        name = sym.Name.decode(errors="replace")
        where = ""
        if not n and dbghelp.SymGetLineFromAddr64(H, a, ctypes.byref(ldisp), ctypes.byref(line)):
            where = f" ({short(line.FileName)}:{line.LineNumber})"
        out.append(name + where)
    else:
        out.append(f"{addr:#x}")
    cache[addr] = out
    return out


# Frames that say nothing about the owner.
NOISE = re.compile(r"^(operator new|`anonymous namespace'::TracedVirtualAlloc|TracedVirtualAlloc|std::|`anonymous namespace'::(Allocate|Note|Add)|Allocate|Note|Add\b|boost::container|"
                   r"fmt::|ankerl::|tsl::|__scrt|_Alloc|Common::ScratchBuffer|Common::SlotVector|nlohmann::|"
                   r"Common::VirtualBuffer|boost::intrusive)")

# First match wins; searched over the whole named stack.
OWNERS = [
    ("dynarmic", r"Dynarmic::|dynarmic/"),
    ("guest thread host stacks (fibers)", r"Common::Fiber::"),
    ("texture cache", r"TextureCache|texture_cache|VideoCommon::(Image|Sampler)|Vulkan::(Image|Framebuffer|Sampler)"),
    ("buffer cache", r"BufferCache|buffer_cache|Vulkan::Buffer\b|BufferBase"),
    ("descriptor queues/pools", r"UpdateDescriptorQueue|DescriptorPool|DescriptorAllocator|ResourcePool"),
    ("staging pool", r"StagingBufferPool"),
    ("shader recompiler (IR/SPIR-V emit)", r"Shader::|Sirit::|shader_recompiler"),
    ("pipelines / shader cache", r"PipelineCache|GraphicsPipeline|ComputePipeline|ShaderCache|RenderPassCache|shader_cache"),
    ("vulkan scheduler/command chunks", r"Scheduler|CommandChunk|vk_scheduler"),
    ("vulkan memory allocator", r"Vma|vma|MemoryAllocator|MemoryCommit"),
    ("GPU memory manager page tables", r"Tegra::MemoryManager|MultiLevelPageTable"),
    ("GPU: DMA pusher / engines / channel", r"DmaPusher|Tegra::Engines|Tegra::Control|Maxwell3D|MacroEngine|Tegra::GPU|GPUThread|Host1x|Tegra::"),
    ("device memory manager (SMMU)", r"DeviceMemoryManager|device_memory_manager"),
    ("query cache", r"QueryCache|query_cache"),
    ("video decode / ffmpeg", r"FFmpeg|av_|Nvdec|Decoder|Vp9|H264"),
    ("nvdrv services", r"Service::Nvidia|nvmap|NvMap|Nvnflinger|nvnflinger"),
    ("kernel page tables (Common::PageTable)", r"Common::PageTable|PageTable::Resize"),
    ("Core::Memory", r"Core::Memory::"),
    ("kernel: memory managers / page heaps", r"KMemoryManager|KPageHeap|KPageBitmap|KMemoryLayout|KPageTableManager|KDynamic|KMemoryBlock|KBlockInfo|KSlab|InitializeSlab|KernelCore::Impl::Init"),
    ("kernel: threads/processes/other", r"Kernel::"),
    ("filesystem: NUP/NCA/RomFS/VFS", r"FileSys::|Nup|NUP|RomFS|romfs|Loader::"),
    ("audio", r"AudioCore|audio_core|Cubeb|cubeb|opus|Opus"),
    ("HLE services", r"Service::"),
    ("OoT hooks / nx2 tools", r"Oot|oot_|Nx2|nx2"),
    ("frame generation / upscalers", r"Lsfg|lsfg|FSR|Fsr|DLSS|Dlss|NGX|FrameGen"),
    ("renderer misc (Vulkan::)", r"Vulkan::|vk::|Vk"),
    ("video_core misc", r"VideoCore|VideoCommon|video_core"),
    ("settings/config/frontend/Qt", r"Settings|Config|QtCommon|GMainWindow|GameList|Q[A-Z][a-z]|yuzu/|qt_common|frontend"),
    ("input", r"InputCommon|input_common|hid_core|Core::HID|SDL"),
    ("logging", r"Common::Log"),
    ("common misc", r"Common::|common/"),
    ("core misc", r"Core::|core/"),
]
OWNERS = [(n, re.compile(p)) for n, p in OWNERS]

groups = []
REGIONS = [{"count": 1, "bytes": r["committed"], "weight": r["committed"], "resident": r["resident"],
            "stack": r["stack"], "va": f"[VirtualAlloc {r['reserved'] >> 20} MiB reserved{' code' if r.get('code') else ''}] "}
           for r in trace.get("regions", [])]
for g in trace["groups"] + REGIONS:
    frames = []
    for addr in g["stack"]:
        if BASE <= addr < BASE + (1 << 28):
            frames += names(addr)
        else:
            frames.append("<other module>")
    meaningful = [f for f in frames if not NOISE.match(f)]
    text = g.get("va", "") + " < ".join(meaningful[:10])
    owner = "unknown"
    for name, rx in OWNERS:
        if rx.search(text):
            owner = name
            break
    groups.append((owner, text, g))

MiB = 1 << 20
owners = {}
for owner, text, g in groups:
    o = owners.setdefault(owner, [0, 0, 0, 0])
    o[0] += g["resident"]
    o[1] += g["weight"]
    o[2] += g["count"]
    # Blocks under 16 KiB are counted resident without being asked.
    o[3] += g["resident"] if g["bytes"] / max(g["count"], 1) >= 16384 else 0
tot = [sum(o[i] for o in owners.values()) for i in range(4)]
print(f"{sys.argv[1]}: resident {tot[0] / MiB:.0f} MiB of {tot[1] / MiB:.0f} MiB allocated in {tot[2]} records "
      f"({trace.get('stale', 0)} stale dropped)")
print(f"{'owner':46s} {'resident':>9s} {'allocated':>9s} {'records':>8s}")
for owner, o in sorted(owners.items(), key=lambda kv: -kv[1][0]):
    print(f"{owner:46s} {o[0] / MiB:9.1f} {o[1] / MiB:9.1f} {o[2]:8d}")
if STACKS:
    # Stacks merged by their first 6 meaningful frames.
    merged = {}
    for owner, text, g in groups:
        key = (owner, " < ".join(text.split(" < ")[:6]))
        m = merged.setdefault(key, [0, 0, 0])
        m[0] += g["resident"]
        m[1] += g["weight"]
        m[2] += g["count"]
    rows = sorted(merged.items(), key=lambda kv: -kv[1][0])
    if GREP:
        rows = [r for r in rows if GREP in r[0][0] or GREP in r[0][1]]
    print()
    for (owner, text), m in rows[:TOP]:
        print(f"{m[0] / MiB:8.1f} of {m[1] / MiB:8.1f} MiB in {m[2]:6d} [{owner}] {text}")
