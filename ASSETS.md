# Assets and licenses

| Asset | Source | License |
|---|---|---|
| Human base mesh, shape targets, `default` rig and weights (163 bones incl. `eye.L/R`, lid bones `orbicularis03/04.L/R`) | MPFB2 2.0.18 Blender extension (extensions.blender.org, sha256 `c075b3e9…2dad`) | Code GPLv3; bundled MakeHuman data CC0 |
| Skin `young_caucasian_female` (+ `young_lightskinned_female_diffuse.png`), eyebrows `eyebrow001`, eyelashes `eyelashes01` | MakeHuman 1.1 system assets | CC0 (MakeHuman relicensed all system assets as CC0 in 2020; official pack: `makehuman_system_assets_cc0.zip`) |
| Eyes (sclera, iris, pupil, cornea), all materials not listed above | Built procedurally by `tools/build_scene.py` | MIT (this repo) |

Notes
- The official CC0 asset pack server (files2.makehumancommunity.org) stalled during the build,
  so the files were taken from a GitHub mirror of the MakeHuman 1.1 system assets
  (`dmaugis/makehuman-py3-assets`). Those older copies still carry the pre-2020 AGPL header
  text inside `.mhmat` files; the assets themselves were relicensed CC0 by their copyright
  holders. To rebuild from the official pack, unzip it and pass its folder to
  `tools/build_scene.py --assets`.
- MPFB2 is only needed to rebuild `scene.blend`. The committed `scene.blend` has all textures
  packed (`bpy.ops.file.pack_all()`), so it opens anywhere without MPFB or the asset pack.
- No MakeHuman eye meshes are used: EyeForge replaces them with its own layered eye assembly.
