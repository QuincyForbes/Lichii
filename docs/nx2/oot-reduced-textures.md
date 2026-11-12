# Ocarina of Time: reduced-texture mod (less guest memory)

Ocarina of Time (0400486024DEC000) keeps every texture it loads twice: the decompressed
`Tex/<name>.bntx.zs` file in its CPU heap and the image in its GPU-mapped heap. In Kokiri Forest the
textures it read come to 1223 MiB of BNTX data. Dropping the top mip level of each texture of 512
pixels or more cuts guest memory by about 1.2 GiB there (Windows, handheld, Max Resolution off):

| Point | Guest data, game files | With the mod | Change |
|---|---|---|---|
| Title | 2881 MiB | 2059 MiB | -822 |
| Kokiri Forest | 4548 MiB (GPU-mapped 2342, CPU heap 2106) | 3336-3351 MiB (1711-1727, 1524-1526) | -1197 to -1212 |
| Kokiri Forest, `--levels 2` | 4548 MiB | 3027 MiB (1551, 1376) | -1521 |
| Title, `--levels 2` | 2881 MiB | 1849 MiB | -1032 |

The picture is softer up close (textures at half resolution); nothing else changes. Remove the mod
folder (or switch the add-on off) to get the full textures back.

## Making the mod from your own game

Needs Python 3 with `zstandard` (`pip install zstandard`). The tool reads your `.nup` and writes
only the changed textures; never share or upload what it writes.

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
