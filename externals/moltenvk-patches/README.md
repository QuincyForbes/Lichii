# MoltenVK with native mesh shaders

`.github/workflows/moltenvk.yml` builds MoltenVK for iOS and macOS from
[KhronosGroup/MoltenVK#2842](https://github.com/KhronosGroup/MoltenVK/pull/2842) (VK_EXT_mesh_shader on Metal
mesh pipelines) at commit `c516eb423516187204c68cd1ed82fb81520315eb`, whose base is MoltenVK `main` at
`a63d486387` (v1.4.2 plus 38 commits; #2842 does not apply to the v1.4.2 tag). The patches here are applied in
order:

- `0001-pr2843-specialization-constant-interface-arrays.patch`: [#2843](https://github.com/KhronosGroup/MoltenVK/pull/2843)
  (`cae0521996`), merged onto #2842; one conflict in `MVKPipeline.mm` resolved (mesh execution model plus
  specialization values).
- `0002-spirv-cross-msl-mesh-fixes.patch`: SPIRV-Cross pinned to `e99dc7a38d`, the merge of
  [SPIRV-Cross#2686](https://github.com/KhronosGroup/SPIRV-Cross/pull/2686) (six MSL mesh/task fixes).

The workflow publishes `MoltenVK-ios.tar` and `MoltenVK-macos.tar` (laid out like the official release tarballs,
with MoltenVK's `LICENSE`) on the release `moltenvk-mesh-<tag>` of this repository. `externals/CMakeLists.txt`
downloads the iOS one when `LICHII_MOLTENVK_MESH=ON` (the default) and checks its SHA-512; the official v1.4.2
framework is embedded next to it as `MoltenVKLegacy.framework`.
