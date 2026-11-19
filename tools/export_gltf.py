"""Export the head + rig (eye and lid bones) to web/assets/eyeforge.glb and the geometry
the browser physiology needs to web/assets/eyeforge.json.

    blender -b scene.blend --python-exit-code 1 -P tools/export_gltf.py

glTF cannot carry MPFB's node-group skin or the procedural eye shaders, so export-only
Principled materials are swapped in (the .blend is not saved).
"""
import json
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "assets"


def principled(name, color, rough=0.5, image=None, transmission=0.0, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Transmission Weight"].default_value = transmission
    b.inputs["Alpha"].default_value = alpha
    if image is not None:
        t = m.node_tree.nodes.new("ShaderNodeTexImage")
        t.image = image
        m.node_tree.links.new(t.outputs["Color"], b.inputs["Base Color"])
    if alpha < 1 or transmission > 0:
        m.blend_method = "BLEND"
    return m


def main():
    sc = bpy.context.scene
    info = json.loads(sc["eyeforge"])
    skin_img = next((i for i in bpy.data.images if "diffuse" in i.name.lower()), None)  # MakeHuman skin
    swaps = {"EF_Body.body": principled("web_skin", (0.8, 0.62, 0.52), 0.55, skin_img),
             "EF_Sclera": principled("web_sclera", (0.75, 0.7, 0.66), 0.3),
             "EF_Iris": principled("web_iris", tuple(info.get("web_iris", (0.28, 0.14, 0.05))), 0.6),
             "EF_Pupil": principled("web_pupil", (0.01, 0.01, 0.01), 0.9),
             "EF_Cornea": principled("web_cornea", (1, 1, 1), 0.02, alpha=0.08)}
    for o in sc.objects:
        if o.type == "MESH":
            for slot in o.material_slots:
                if slot.material and slot.material.name in swaps:
                    slot.material = swaps[slot.material.name]
    keep = {"EF_Rig", "EF_Body"} | {o.name for o in sc.objects if o.name.startswith("EF_") and o.type == "MESH"} \
        | {o.name for o in sc.objects if o.type == "MESH" and ("brow" in o.name.lower() or "lash" in o.name.lower())}
    bpy.ops.object.select_all(action="DESELECT")
    for o in sc.objects:
        o.select_set(o.name in keep)
    OUT.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(OUT / "eyeforge.glb"), export_format="GLB", use_selection=True,
                              export_apply=True, export_skins=True, export_morph=True,
                              export_animations=False, export_image_format="JPEG", export_jpeg_quality=88)
    web = {k: info[k] for k in ("C_L", "C_R", "M", "F", "U", "R_img", "pupil_depth", "cam_loc", "eye_bones",
                                "lid_bones", "lid_efficiency", "blink_keys", "ipd")}
    web["lens_mm"] = info["lens"]["lens_mm"]
    web["pupil_mesh_radius_mm"] = 1.0
    (OUT / "eyeforge.json").write_text(json.dumps(web, indent=2))
    print("exported", OUT / "eyeforge.glb")


main()
