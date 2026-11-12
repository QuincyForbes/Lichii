# Ocarina of Time: reduced textures (less guest memory)

Ocarina of Time (0400486024DEC000) keeps every texture it loads twice: the decompressed
`Tex/<name>.bntx.zs` file in its CPU heap and the image in its GPU-mapped heap. In Kokiri Forest the
textures it read come to 1223 MiB of BNTX data. Dropping the top mip level of each texture of 512
pixels or more cuts guest memory by about 1.2 GiB there (Windows, handheld, Max Resolution off):

| Point | Guest data, game files | Half (1 level) | Quarter (2 levels) |
|---|---|---|---|
| Title | 2882 MiB | 2057-2059 MiB (-824) | 1849 MiB (-1033) |
| Kokiri Forest | 4544 MiB (GPU-mapped 2339, CPU heap 2105) | 3334-3356 MiB (-1188..-1210) | 3025 MiB (-1519) |
| Kokiri Forest, 60 s later | 4562 MiB | 3352-3375 MiB | 3043 MiB |

The picture is softer up close (textures at half or quarter resolution); nothing else changes.

## In the app: Texture quality

Per-game setting **Texture quality** (Qt: Properties > Advanced Graphics; iOS: the game's settings,
Graphics (Advanced)): Full (default), Half or Quarter. It applies at the next game start and changes
no files: when the game starts, the RomFS it is given serves each `*.bntx.zs` file that holds one
plain 2D texture (no array, cube map or 3D texture, at least two mip levels, a known block format)
of 512 pixels or more with its top one or two mip levels dropped (`src/core/file_sys/texture_quality.cpp`,
the same rules and the same bytes as the tool below, `src/core/file_sys/bntx_reduce.cpp`).

- The RomFS needs each file's size before the game starts, so at the first start with a setting the
  first zstd block of every texture is read (header only; Ocarina: 21,658 files, 0.9-4.6 s on the PC,
  15,348 of them reduced). The sizes are kept in `<cache>/texture_quality/` and later starts take
  about 15 ms.
- A reduced file is made when the game reads it (about 0.8 ms per texture on the PC: the original
  is decompressed, the levels are dropped, and the result is written as zstd raw blocks, whose size
  is known from the header alone) and is not kept, except for a file the game reads in parts
  (up to 32 MiB, emptied on memory pressure).
- The log says it is on: `Texture quality Half: 15348 of 21658 .bntx.zs files get 1 top mip level(s)
  dropped ...` at the start, then `Texture quality: N textures reduced so far, ...` at the first
  texture and every 1000th (category Common.Memory).
- `EDEN_NX2_TEXTURE_QUALITY=0/1/2` overrides the setting for tests.
- Works for any game in a NUP container whose textures are such files; the default is Full
  everywhere.

## Making a mod from your own game (PC)

The same result as a LayeredFS mod. Needs Python 3 with `zstandard` (`pip install zstandard`). The
tool reads your `.nup` and writes only the changed textures; never share or upload what it writes.

```sh
python3 tools/nx2/oot_texture_mod.py make "<game>.nup" "<mod folder>/Reduced textures" --min-size 512
```

Copy `Reduced textures` (it contains `romfs/Tex/...`) into the title's mod folder:
`<user dir>/load/0400486024DEC000/` (Windows: `%APPDATA%\eden\load\0400486024DEC000\`; iOS: the app's
`load/0400486024DEC000/` in its Documents). It is 1.7 GiB at the default zstd level 3 and takes
about 2 minutes. `--levels 2` drops two mip levels (about 94 % less per texture, quarter
resolution up close);
`--min-size 1024` keeps the 512-pixel textures. Arrays, cube maps, 3D textures and textures with
one mip level are left as they are.

`oot_texture_mod.py survey "<game>.nup"` lists texture memory by format and size;
`--list <log>` limits either command to the textures one session read (run Lichii with
`EDEN_NX2_NUP_READ_LOG=1`, which logs `NUP first read: #<index> <name> ...` per file).
`tools/nx2/nup_tool.py` lists and extracts the members of any NUP container.
`tools/nx2/texture_quality_vectors.py` prints the tool's results on synthetic textures that the
C++ unit test checks, and with `--compare <dir>` checks the C++ results on real files that the hidden
test case `[.texture_quality_files]` wrote there.
