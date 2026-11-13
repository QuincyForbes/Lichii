# Ocarina of Time graphics presets

`oot_graphics_preset` (Ocarina of Time's page only; Qt and iOS) edits the game's own parameter
files as the RomFS is built (`core/file_sys/oot_game_params.cpp`, called from `loader/nup.cpp`).
Nothing is written to disk; Console leaves the files untouched. The phone presets also set
`oot_resolution` and `texture_quality` when picked (`Settings::OotGraphicsPresetCompanions`).
Log line: `Ocarina graphics preset <name>: N values in the game's parameter files, ...`
(Common.Memory).

| Preset | Game files | Resolution / textures |
|---|---|---|
| Phone Low | `gsys/ModelSceneConfigParam/Main`: `CubeMapMaxTexNum=1`, `CubeMapMaxTexWidth=64`, `CreateDof=false`, `CreateXluZPrePass=false`, `DepthShadowCascadeNum0=1`, `DepthShadowTexWidth/Height=512`, `StaticSdwWidth=1024`; `agl/BloomParam/MainField`: `Enable=false`; foliage level 1 (half grass density and plant distances) | 50 %, Quarter |
| Phone Medium | `CubeMapMaxTexNum=1`, `CubeMapMaxTexWidth=64`, `CreateDof=false`, `CreateXluZPrePass=false`, `DepthShadowCascadeNum0=2`, `DepthShadowTexWidth/Height=512`, `StaticSdwWidth=2048` | 56 %, Half |
| Console | none (the game as shipped) | unchanged |
| High | `DepthShadowTexWidth/Height=2048` | unchanged |

Default: Phone Low on iPhones (device profile), Console elsewhere.

## Testing switches

- `EDEN_NX2_OOT_GRAPHICS_PRESET=0..3` (Phone Low, Phone Medium, Console, High) overrides the setting.
- `EDEN_NX2_OOT_GFX="<file>:<Key.Sub>=<value>,...;<file2>:..."` adds or replaces single edits.
  `<file>` is a RomFS path prefix (`gsys/ModelSceneConfigParam/Main`) or `<pack>!<member>` for a
  file inside a pack (`Pack/Bootup!System/GameGraphics/ModelLodParam/Default`). Values are written
  as the type the file already has; a key that is missing is logged and skipped.
- `EDEN_NX2_OOT_GFX_SKIP=<key>,...` drops the preset's edits of those keys (A/B one lever).
- `EDEN_NX2_OOT_FOLIAGE=0..2` overrides the foliage level.

## Levers measured (PC phone simulation, Kokiri, 45-75 s)

Handheld, 56 %, Quarter, `EDEN_NX2_CACHE_BUDGET=ios`; "mesh emulated" adds
`EDEN_NX2_FORCE_MESH_FALLBACK=1`, which breaks render passes as the Legacy iPhone driver does. One run per lever;
identical runs differ by about 1 pass, 0.1 ms and 50-100 MiB of images.

| Lever | Render passes | Draws | GPU ms | Images MiB | Picture |
|---|---|---|---|---|---|
| none (native mesh / mesh emulated) | 124 / 175 | 1325 | 3.5 / 4.7 | ~1600 | |
| reflection probe off | -18 | -20 | -0.2 | -110..-200 | no change seen, but 8 texture cache asserts (zero layer stride): left out |
| probe 1 x 64 px | -19 | -70 | -0.2 | -60 | none seen |
| bloom off (`agl` Enable) | -21 | 0 | -0.1 | ~0 | glow around lights weaker |
| depth of field off | -4.5 | 0 | 0 | -40 | none in gameplay |
| translucent z-prepass off | -1.5 | 0 | 0 | -35 | none seen |
| 1 cascade / 512 px cascades | -2 / 0 | 0 | 0 | -30 / -50 | shadows softer up close |
| static shadow 1024 | 0 | 0 | 0 | -64 | none seen |
| static shadow off | 0 | 0 | 0 | -190 | rocks lose baked cliff shadow: left out |
| dynamic shadow off | -16 (emulated) | 0 | -0.2 | -150 | no character/tree shadows: left out |
| 50 % instead of 56 % | -17 (emulated) | -280 | -0.5 | -175 | softer |
| foliage level 2, fog grid half, SSAO off, LOD scale, shadow distance | 0 | 0 | 0 | 0 | |

Broken, never used: `CreateSsocclusion=false` (thousands of "Unimplemented format" asserts),
`CreateShadowPrePass=false` and `ZPrePass=false` (the game stops at the first frame).

## Presets measured (mesh emulated, 2 runs each)

| Preset | GPU ms | Draws | Render passes | Attachments load / store MiB | Images MiB |
|---|---|---|---|---|---|
| Console, 56 %, Quarter (today's phone) | 4.65 | 1338 | 175 | 364 / 415 | 1210 |
| Phone Medium, 56 %, Half | 4.15 | 1363 | 148 | 360 / 409 | 1233 |
| Phone Low, 50 %, Quarter | 3.55 | 928 | 105 | 243 / 281 | 938 |
| High | 4.65 | 1365 | 174 | 363 / 414 | 1125 |
| Phone Low preset at 56 % (no resolution change) | 4.0 | 1298 | 120 | 351 / 396 | 990 |

Native mesh shaders (the New driver), one run each: Console 56 % 3.5 ms, 1332 draws, 123 passes,
attachments 172 / 223 MiB; Phone Low 2.9 ms, 1031 draws, 71 passes, 124 / 163 MiB.

The 12288x12288 D16 image (288 MiB) is not touched by any parameter here and is not drawn per
frame in Kokiri.
