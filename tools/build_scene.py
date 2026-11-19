"""Rebuild scene.blend reproducibly: MPFB2 human + layered eye assemblies + locked camera.

Needs Blender 4.2 LTS with the MPFB2 extension installed (repository "user_default") and a
MakeHuman CC0 system-asset folder (containing skins/, eyebrows/, eyelashes/).

    blender -b -P tools/build_scene.py -- --assets <makehuman_assets/base> --out scene.blend
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wrapper"))
import calib  # noqa: E402
import physiology as ph  # noqa: E402

from bl_ext.user_default.mpfb.services.humanservice import HumanService  # noqa: E402
from bl_ext.user_default.mpfb.services.locationservice import LocationService  # noqa: E402
from bl_ext.user_default.mpfb.services.targetservice import TargetService  # noqa: E402

MM = 0.001
# Anatomical eye dimensions (mm). The sclera radius comes from the MPFB socket (see README).
CORNEA_R = 7.8          # anterior corneal radius of curvature
LIMBUS_R = 5.85         # horizontal visible iris diameter 11.7 mm
CORNEA_IOR = 1.376
IRIS_BEHIND_LIMBUS = 0.2
IRIS_CONCAVITY = 0.15
PUPIL_IN_FRONT = 0.10   # pupil disk sits this far in front of the iris center
CHAMBER_BACK = 0.4      # back face of the refracting volume, behind the iris center
SCLERA_INSET = 0.25     # keep the sclera just inside the lids' socket sphere
TARGET_IPD = 0.063
W = H = 1024
EYE_BONES = {"L": "eye.L", "R": "eye.R"}
LID_BONES = {"L": ("orbicularis03.L", "orbicularis04.L"), "R": ("orbicularis03.R", "orbicularis04.R")}
LOG: list[str] = []


def log(msg: str) -> None:
    print("[build_scene]", msg)
    LOG.append(msg)


# ----------------------------------------------------------------------------- helpers

def sphere_fit(P: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Algebraic least-squares sphere: |p|^2 = 2 c.p + (r^2 - |c|^2). Returns c, r, max residual."""
    A = np.c_[2 * P, np.ones(len(P))]
    x = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)[0]
    c = x[:3]
    r = math.sqrt(x[3] + c @ c)
    return c, r, float(np.abs(np.linalg.norm(P - c, axis=1) - r).max())


def evaluated_coords(obj, unmasked: bool = False) -> np.ndarray:
    """World-space evaluated vertices. unmasked=True keeps MakeHuman helper vertices so the
    indices match obj.data (only show_viewport is toggled; renders are unaffected)."""
    masks = [m for m in obj.modifiers if m.type == "MASK" and m.show_viewport] if unmasked else []
    for m in masks:
        m.show_viewport = False
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    M = np.array(obj.matrix_world)
    ev.to_mesh_clear()
    for m in masks:
        m.show_viewport = True
    return co @ M[:3, :3].T + M[:3, 3]


def group_indices(obj, name: str, min_w: float = 0.5) -> np.ndarray:
    gi = obj.vertex_groups[name].index
    return np.array([v.index for v in obj.data.vertices
                     if any(g.group == gi and g.weight >= min_w for g in v.groups)])


def lathe(name: str, profile: list[tuple[float, float]], segments: int = 96,
          outward_ref: float | None = None) -> bpy.types.Object:
    """Revolve (r, z) profile points (meters) about local +Z. r == 0 points become poles."""
    bm = bmesh.new()
    rings = []
    for r, z in profile:
        if r < 1e-9:
            rings.append([bm.verts.new((0.0, 0.0, z))])
        else:
            rings.append([bm.verts.new((r * math.cos(a), r * math.sin(a), z))
                          for a in np.linspace(0, 2 * math.pi, segments, endpoint=False)])
    for a, b in zip(rings, rings[1:]):
        if len(a) == 1 and len(b) == 1:
            continue
        for i in range(segments):
            j = (i + 1) % segments
            if len(a) == 1:
                bm.faces.new((a[0], b[i], b[j]))
            elif len(b) == 1:
                bm.faces.new((a[i], a[j], b[0]))
            else:
                bm.faces.new((a[i], a[j], b[j], b[i]))
    bm.normal_update()
    if outward_ref is not None:  # open shells: point normals away from (0, 0, outward_ref)
        for f in bm.faces:
            if f.normal.dot(f.calc_center_median() - Vector((0, 0, outward_ref))) < 0:
                f.normal_flip()
    else:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def node_mat(name: str):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    return m, nt, out


# ----------------------------------------------------------------------------- materials

def mat_sclera():
    m, nt, out = node_mat("EF_Sclera")
    N, L = nt.nodes, nt.links
    bsdf = N.new("ShaderNodeBsdfPrincipled")
    tc = N.new("ShaderNodeTexCoord")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(tc.outputs["Object"], sep.inputs[0])
    # angle from the optical axis -> 0 at the cornea, grows toward the back of the eye
    length = N.new("ShaderNodeVectorMath"); length.operation = "LENGTH"
    L.new(tc.outputs["Object"], length.inputs[0])
    cosang = N.new("ShaderNodeMath"); cosang.operation = "DIVIDE"
    L.new(sep.outputs["Z"], cosang.inputs[0]); L.new(length.outputs["Value"], cosang.inputs[1])
    # veins: thin ridges of a distorted noise, denser toward the equator
    noise = N.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 90.0
    noise.inputs["Detail"].default_value = 6.0
    noise.inputs["Distortion"].default_value = 0.6
    L.new(tc.outputs["Object"], noise.inputs["Vector"])
    ridge = N.new("ShaderNodeMath"); ridge.operation = "PINGPONG"
    ridge.inputs[1].default_value = 0.5
    L.new(noise.outputs["Fac"], ridge.inputs[0])
    vein = N.new("ShaderNodeMapRange")
    vein.inputs["From Min"].default_value = 0.44; vein.inputs["From Max"].default_value = 0.5
    vein.inputs["To Min"].default_value = 0.0; vein.inputs["To Max"].default_value = 1.0
    L.new(ridge.outputs[0], vein.inputs["Value"])
    region = N.new("ShaderNodeMapRange")  # veins fade out near the limbus
    region.inputs["From Min"].default_value = 0.95; region.inputs["From Max"].default_value = 0.75
    L.new(cosang.outputs[0], region.inputs["Value"])
    veinmask = N.new("ShaderNodeMath"); veinmask.operation = "MULTIPLY"
    L.new(vein.outputs[0], veinmask.inputs[0]); L.new(region.outputs[0], veinmask.inputs[1])
    vm = N.new("ShaderNodeMath"); vm.operation = "MULTIPLY"; vm.inputs[1].default_value = 0.8
    L.new(veinmask.outputs[0], vm.inputs[0])
    mix = N.new("ShaderNodeMix"); mix.data_type = "RGBA"
    mix.inputs[6].default_value = (0.72, 0.66, 0.61, 1)     # sclera: off-white, slightly warm
    mix.inputs[7].default_value = (0.55, 0.10, 0.08, 1)     # vessel red
    L.new(vm.outputs[0], mix.inputs[0])
    # limbal darkening ring at the cornea edge
    limb = N.new("ShaderNodeMapRange")
    limb.inputs["From Min"].default_value = 0.955; limb.inputs["From Max"].default_value = 0.925
    limb.inputs["To Min"].default_value = 0.55; limb.inputs["To Max"].default_value = 1.0
    L.new(cosang.outputs[0], limb.inputs["Value"])
    dark = N.new("ShaderNodeMix"); dark.data_type = "RGBA"; dark.blend_type = "MULTIPLY"
    dark.inputs[0].default_value = 1.0
    L.new(mix.outputs[2], dark.inputs[6])
    comb = N.new("ShaderNodeCombineColor")
    for i in range(3):
        L.new(limb.outputs[0], comb.inputs[i])
    L.new(comb.outputs[0], dark.inputs[7])
    L.new(dark.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.35
    bsdf.inputs["Subsurface Weight"].default_value = 0.15
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.25)
    bsdf.inputs["Subsurface Scale"].default_value = 0.002
    bsdf.inputs["Coat Weight"].default_value = 1.0      # wet conjunctiva
    bsdf.inputs["Coat Roughness"].default_value = 0.03
    L.new(bsdf.outputs[0], out.inputs[0])
    return m


def mat_iris(color):
    m, nt, out = node_mat("EF_Iris")
    N, L = nt.nodes, nt.links
    bsdf = N.new("ShaderNodeBsdfPrincipled")
    tc = N.new("ShaderNodeTexCoord")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(tc.outputs["Object"], sep.inputs[0])
    ang = N.new("ShaderNodeMath"); ang.operation = "ARCTAN2"
    L.new(sep.outputs["Y"], ang.inputs[0]); L.new(sep.outputs["X"], ang.inputs[1])
    rad = N.new("ShaderNodeVectorMath"); rad.operation = "LENGTH"
    xy = N.new("ShaderNodeCombineXYZ")
    L.new(sep.outputs["X"], xy.inputs[0]); L.new(sep.outputs["Y"], xy.inputs[1])
    L.new(xy.outputs[0], rad.inputs[0])
    rn = N.new("ShaderNodeMath"); rn.operation = "DIVIDE"; rn.inputs[1].default_value = LIMBUS_R * MM
    L.new(rad.outputs["Value"], rn.inputs[0])
    # radial fibers: noise sampled on (angle * k, radius) -> streaks along the radius
    polar = N.new("ShaderNodeCombineXYZ")
    angk = N.new("ShaderNodeMath"); angk.operation = "MULTIPLY"; angk.inputs[1].default_value = 14.0
    L.new(ang.outputs[0], angk.inputs[0]); L.new(angk.outputs[0], polar.inputs[0])
    radk = N.new("ShaderNodeMath"); radk.operation = "MULTIPLY"; radk.inputs[1].default_value = 1.2
    L.new(rn.outputs[0], radk.inputs[0]); L.new(radk.outputs[0], polar.inputs[1])
    fib = N.new("ShaderNodeTexNoise")
    fib.inputs["Scale"].default_value = 3.0
    fib.inputs["Detail"].default_value = 8.0
    fib.inputs["Roughness"].default_value = 0.65
    L.new(polar.outputs[0], fib.inputs["Vector"])
    fibc = N.new("ShaderNodeMapRange")
    fibc.inputs["From Min"].default_value = 0.3; fibc.inputs["From Max"].default_value = 0.7
    fibc.inputs["To Min"].default_value = 0.35; fibc.inputs["To Max"].default_value = 1.5
    L.new(fib.outputs["Fac"], fibc.inputs["Value"])
    # radial profile: lighter collarette ring, dark limbal ring at the outer edge
    ramp = N.new("ShaderNodeValToRGB")
    els = ramp.color_ramp.elements
    els[0].position = 0.0; els[0].color = (0.5, 0.42, 0.3, 1)
    els[1].position = 1.0; els[1].color = (0.03, 0.03, 0.03, 1)
    for pos, col in ((0.38, (1.9, 1.5, 0.9)), (0.5, (1.0, 1.0, 1.0)), (0.82, (0.75, 0.75, 0.75)),
                     (0.92, (0.22, 0.22, 0.22))):
        e = els.new(pos); e.color = (*col, 1)  # golden collarette, stroma, dark limbal ring
    L.new(rn.outputs[0], ramp.inputs[0])
    base = N.new("ShaderNodeRGB"); base.name = base.label = "IrisColor"
    base.outputs[0].default_value = (*color, 1.0)
    m1 = N.new("ShaderNodeMix"); m1.data_type = "RGBA"; m1.blend_type = "MULTIPLY"
    m1.inputs[0].default_value = 1.0
    L.new(base.outputs[0], m1.inputs[6]); L.new(ramp.outputs[0], m1.inputs[7])
    m2 = N.new("ShaderNodeVectorMath"); m2.operation = "SCALE"
    L.new(m1.outputs[2], m2.inputs[0]); L.new(fibc.outputs[0], m2.inputs["Scale"])
    # crypts: dark voronoi pits in the stroma
    vor = N.new("ShaderNodeTexVoronoi"); vor.feature = "F1"
    vor.inputs["Scale"].default_value = 1.6
    L.new(polar.outputs[0], vor.inputs["Vector"])
    crypt = N.new("ShaderNodeMapRange")
    crypt.inputs["From Min"].default_value = 0.08; crypt.inputs["From Max"].default_value = 0.22
    crypt.inputs["To Min"].default_value = 0.45; crypt.inputs["To Max"].default_value = 1.0
    L.new(vor.outputs["Distance"], crypt.inputs["Value"])
    m3 = N.new("ShaderNodeVectorMath"); m3.operation = "SCALE"
    L.new(m2.outputs[0], m3.inputs[0]); L.new(crypt.outputs[0], m3.inputs["Scale"])
    L.new(m3.outputs[0], bsdf.inputs["Base Color"])
    bump = N.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.25
    bump.inputs["Distance"].default_value = 0.0002
    L.new(fib.outputs["Fac"], bump.inputs["Height"]); L.new(bump.outputs[0], bsdf.inputs["Normal"])
    bsdf.inputs["Roughness"].default_value = 0.6
    bsdf.inputs["Specular IOR Level"].default_value = 0.1  # the cornea provides the specular
    L.new(bsdf.outputs[0], out.inputs[0])
    return m


def mat_pupil():
    m, nt, out = node_mat("EF_Pupil")
    bsdf = nt.nodes.new("ShaderNodeBsdfDiffuse")
    bsdf.inputs["Color"].default_value = (0.004, 0.004, 0.004, 1)
    nt.links.new(bsdf.outputs[0], out.inputs[0])
    return m


def mat_cornea():
    """Clear glass, IOR 1.376. Shadow rays pass straight through so the iris is not blacked
    out by missing caustics (Cycles cannot sample lights through refraction)."""
    m, nt, out = node_mat("EF_Cornea")
    N, L = nt.nodes, nt.links
    glass = N.new("ShaderNodeBsdfGlass")
    glass.inputs["IOR"].default_value = CORNEA_IOR
    glass.inputs["Roughness"].default_value = 0.0
    glass.inputs["Color"].default_value = (1, 1, 1, 1)
    tr = N.new("ShaderNodeBsdfTransparent")
    lp = N.new("ShaderNodeLightPath")
    mix = N.new("ShaderNodeMixShader")
    L.new(lp.outputs["Is Shadow Ray"], mix.inputs[0])
    L.new(glass.outputs[0], mix.inputs[1]); L.new(tr.outputs[0], mix.inputs[2])
    L.new(mix.outputs[0], out.inputs[0])
    return m


def mask_materials():
    """Emission-only materials for the pupil mask passes (see wrapper/masks.py)."""
    mats = {}
    for name, col in (("EF_MaskBlack", (0, 0, 0)), ("EF_MaskPupil.L", (1, 0, 0)),
                      ("EF_MaskPupil.R", (0, 1, 0))):
        m, nt, out = node_mat(name)
        e = nt.nodes.new("ShaderNodeEmission")
        e.inputs["Color"].default_value = (*col, 1)
        e.inputs["Strength"].default_value = 1.0
        nt.links.new(e.outputs[0], out.inputs[0])
        m.use_fake_user = True
        mats[name] = m
    m, nt, out = node_mat("EF_MaskGlass")  # refraction only: no reflections in the mask
    r = nt.nodes.new("ShaderNodeBsdfRefraction")
    r.inputs["IOR"].default_value = CORNEA_IOR
    r.inputs["Roughness"].default_value = 0.0
    nt.links.new(r.outputs[0], out.inputs[0])
    m.use_fake_user = True
    mats["EF_MaskGlass"] = m
    return mats


# ----------------------------------------------------------------------------- build steps

def make_human(assets: Path):
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    macro = {"gender": 0.0, "age": 0.5, "muscle": 0.5, "weight": 0.5, "height": 0.5,
             "proportions": 0.5, "cupsize": 0.5, "firmness": 0.5,
             "race": {"african": 0.0, "asian": 0.0, "caucasian": 1.0}}
    bm = HumanService.create_human(macro_detail_dict=macro)
    bm.name = "EF_Body"
    # widen the eye spacing toward an adult IPD (MakeHuman default female is ~57.7 mm)
    tdir = LocationService.get_mpfb_data("targets")
    names = {}
    for s in "lr":
        names[s] = f"ef-{s}-eye-trans-out"
        TargetService.load_target(bm, os.path.join(tdir, "eyes", f"{s}-eye-trans-out.target.gz"),
                                  weight=0.0, name=names[s])
    keys = bm.data.shape_keys.key_blocks

    def ipd_at(w):
        for s in "lr":
            keys[names[s]].value = w
        c = eye_centers(bm)
        return float(np.linalg.norm(c["L"][0] - c["R"][0]))

    i0, i1 = ipd_at(0.0), ipd_at(1.0)
    w = (TARGET_IPD - i0) / (i1 - i0)
    log(f"IPD: default {i0*1000:.2f} mm, eye-trans-out weight {w:.3f} -> {ipd_at(w)*1000:.2f} mm")
    return bm, macro, w


def eye_centers(bm) -> dict:
    """Sphere-fit the MakeHuman eyeball helper geometry ('helper-l-eye', 'helper-r-eye')."""
    co = evaluated_coords(bm, unmasked=True)
    out = {}
    for s, g in (("L", "helper-l-eye"), ("R", "helper-r-eye")):
        c, r, res = sphere_fit(co[group_indices(bm, g)])
        out[s] = (c, r, res)
    return out


def fix_bones(arm, centers, F):
    """Move eye (and lid) bones so their heads sit exactly on the fitted eyeball centers."""
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    inv = np.array(arm.matrix_world.inverted())
    report = {}
    for s in "LR":
        c_arm = inv[:3, :3] @ centers[s] + inv[:3, 3]
        for name in (EYE_BONES[s], *LID_BONES[s]):
            eb = arm.data.edit_bones[name]
            before = float(np.linalg.norm(np.array(eb.head) - c_arm))
            if before > 1e-6:  # spec requires a fix above 0.2 mm; we always snap exactly
                d = Vector(c_arm) - eb.head
                eb.head += d
                eb.tail += d
            after = float(np.linalg.norm(np.array(eb.head) - c_arm))
            report[name] = {"before_mm": before * 1000, "after_mm": after * 1000}
            log(f"bone {name}: head-to-center {before*1000:.3f} mm -> {after*1000:.4f} mm")
    bpy.ops.object.mode_set(mode="OBJECT")
    return report


def build_eye(s: str, C, R_s: float, frame: Matrix, arm, mats) -> dict:
    """Layered eye in a local frame (+Z = optical axis), parented rigidly to the eye bone."""
    zL = math.sqrt(R_s ** 2 - (LIMBUS_R * MM) ** 2)
    zc = zL - math.sqrt((CORNEA_R * MM) ** 2 - (LIMBUS_R * MM) ** 2)
    apex = zc + CORNEA_R * MM
    zi = zL - IRIS_BEHIND_LIMBUS * MM
    z_pupil = zi - IRIS_CONCAVITY * MM + PUPIL_IN_FRONT * MM
    zb = zi - IRIS_CONCAVITY * MM - CHAMBER_BACK * MM
    thL = math.acos(zL / R_s)
    sclera = lathe(f"EF_Sclera.{s}", [(R_s * math.sin(t), R_s * math.cos(t))
                                      for t in np.linspace(math.pi, thL, 80)], outward_ref=0.0)
    phL = math.asin(LIMBUS_R / CORNEA_R)
    cornea_prof = [(CORNEA_R * MM * math.sin(p), zc + CORNEA_R * MM * math.cos(p))
                   for p in np.linspace(0, phL, 40)]
    cornea = lathe(f"EF_Cornea.{s}", cornea_prof + [(LIMBUS_R * MM, zb), (0.0, zb)])
    r_iris = (LIMBUS_R - 0.05) * MM
    iris = lathe(f"EF_Iris.{s}", [(r, zi - IRIS_CONCAVITY * MM * (1 - (r / r_iris) ** 2))
                                  for r in np.linspace(0, r_iris, 30)])
    pupil = lathe(f"EF_Pupil.{s}", [(0.0, z_pupil), (1.0 * MM, z_pupil)])  # 1 mm radius, scaled
    for obj, m in ((sclera, mats["sclera"]), (cornea, mats["cornea"]), (iris, mats["iris"]),
                   (pupil, mats["pupil"])):
        obj.data.materials.append(m)
        obj.matrix_world = frame
    # lathe() recalculates closed meshes; make sure the iris/pupil disks face forward (+Z)
    for obj in (iris, pupil):
        if obj.data.polygons[0].normal.z < 0:
            obj.data.flip_normals()
    bpy.context.view_layer.update()
    pb = arm.pose.bones[EYE_BONES[s]]
    bone_world = arm.matrix_world @ pb.matrix @ Matrix.Translation((0, pb.bone.length, 0))
    for obj in (sclera, cornea, iris, pupil):
        obj.parent = arm
        obj.parent_type = "BONE"
        obj.parent_bone = EYE_BONES[s]
        obj.matrix_parent_inverse = bone_world.inverted()
        obj.matrix_basis = frame
    bpy.context.view_layer.update()
    err = max((Vector(np.array(o.matrix_world.translation)) - Vector(C)).length
              for o in (sclera, cornea, iris, pupil))
    assert err < 1e-6, f"eye parts drifted from the eyeball center by {err} m"
    return dict(z_limbus=zL, cornea_center=zc, apex=apex, iris_plane=zi, pupil_plane=z_pupil,
                chamber_back=zb, iris_radius=r_iris)


def lid_margin_vertices(bm, centers, F, U, R_img, blink_keys) -> dict:
    """The visible lid margins above/below each pupil: the skin vertices (no helpers) within
    2 mm of the pupil's vertical plane that the MakeHuman eye-closure target moves the most
    (down for the upper lid, up for the lower lid). Used for the lid-aperture output."""
    keys = bm.data.shape_keys.key_blocks
    open_co = evaluated_coords(bm, unmasked=True)
    for k in blink_keys.values():
        keys[k].value = 1.0
    shut_co = evaluated_coords(bm, unmasked=True)
    for k in blink_keys.values():
        keys[k].value = 0.0
    helper = set(group_indices(bm, "HelperGeometry", 0.01).tolist())
    skin = np.array([i for i in range(len(open_co)) if i not in helper])
    dz = (shut_co[skin] - open_co[skin]) @ U
    out = {}
    for s in "LR":
        d = open_co[skin] - centers[s]
        near = (np.abs(d @ R_img) < 2.0 * MM) & (d @ F > 0)
        up = near & (d @ U > 0)
        lo = near & (d @ U < 0)
        res = {"upper": int(skin[up][np.argmin(dz[up])]), "lower": int(skin[lo][np.argmax(dz[lo])])}
        ap = float(np.linalg.norm(open_co[res["upper"]] - open_co[res["lower"]]))
        shut = float((shut_co[res["upper"]] - shut_co[res["lower"]]) @ U)
        log(f"lid margins {s}: vertices {res}, rest aperture {ap*1000:.2f} mm, closed gap {shut*1000:.2f} mm")
        out[s] = res
    return out


def lid_efficiency(arm, bm, centers, lids, R_img) -> dict:
    """MakeHuman's lid bones are only partially weighted, so a bone rotation of a moves the lid
    margin by less than a. Measure k = (margin arc) / (r * a) for a 10 deg test rotation; at runtime
    the bone angle is divided by k so the lid margin itself follows gaze at the requested ratio."""
    a = math.radians(10.0)
    base = evaluated_coords(bm, unmasked=True)
    out = {}
    for which, col in (("upper", 0), ("lower", 1)):
        ks = []
        for s in "LR":
            C = Vector(centers[s])
            pb = arm.pose.bones[LID_BONES[s][col]]
            q = Matrix.Rotation(-a, 4, Vector(R_img))
            pb.matrix = Matrix.Translation(C) @ q @ Matrix.Translation(-C) @ pb.bone.matrix_local
            bpy.context.view_layer.update()
            moved = evaluated_coords(bm, unmasked=True)
            vid = lids[s][which]
            r = float(np.linalg.norm(base[vid] - centers[s]))
            ks.append(float(np.linalg.norm(moved[vid] - base[vid])) / (r * a))
            pb.matrix_basis = Matrix()
            bpy.context.view_layer.update()
        out[which] = float(np.mean(ks))
    log(f"lid bone efficiency (margin arc / bone arc): upper {out['upper']:.3f}, lower {out['lower']:.3f}")
    return out


def add_blink_keys(bm) -> dict:
    """MakeHuman expression units eye-left/right-closure as blink shape keys (value 0..1)."""
    tdir = LocationService.get_mpfb_data("targets")
    names = {}
    for s, side in (("L", "left"), ("R", "right")):
        names[s] = f"ef-blink-{s}"
        TargetService.load_target(bm, os.path.join(tdir, "expression", "units", "caucasian",
                                                   f"eye-{side}-closure.target.gz"), weight=0.0, name=names[s])
    return names


def setup_render(scene, samples: int):
    scene.render.engine = "CYCLES"
    cy = scene.cycles
    cy.device = "CPU"
    cy.samples = samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.02
    cy.seed = 0
    cy.use_animated_seed = False
    cy.use_denoising = True
    cy.denoiser = "OPENIMAGEDENOISE"
    cy.max_bounces = 8
    cy.diffuse_bounces = 3
    cy.transmission_bounces = 8
    cy.transparent_max_bounces = 8
    cy.glossy_bounces = 4
    cy.caustics_reflective = False
    cy.caustics_refractive = False
    scene.render.use_persistent_data = True
    scene.render.resolution_x, scene.render.resolution_y = W, H
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -0.25
    scene.render.threads_mode = "AUTO"


def add_area(name, loc, target, size, energy, color=(1, 1, 1)):
    d = bpy.data.lights.new(name, "AREA")
    d.shape = "DISK"
    d.size = size
    d.energy = energy
    d.color = color
    o = bpy.data.objects.new(name, d)
    bpy.context.scene.collection.objects.link(o)
    o.matrix_world = Matrix.Translation(Vector(loc)) @ Matrix(
        calib.look_at(loc, target, (0, 0, 1))).to_4x4()
    return o


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True, help="MakeHuman system asset dir (skins/, eyebrows/...)")
    ap.add_argument("--out", default=str(ROOT / "scene.blend"))
    ap.add_argument("--samples", type=int, default=48)
    args = ap.parse_args(argv)
    assets = Path(args.assets)
    scene = bpy.context.scene

    bm, macro, trans_w = make_human(assets)
    arm = HumanService.add_builtin_rig(bm, "default")
    arm.name = "EF_Rig"
    HumanService.add_mhclo_asset(str(assets / "eyebrows/eyebrow001/eyebrow001.mhclo"), bm,
                                 asset_type="Eyebrows", subdiv_levels=1)
    HumanService.add_mhclo_asset(str(assets / "eyelashes/eyelashes01/eyelashes01.mhclo"), bm,
                                 asset_type="Eyelashes", subdiv_levels=1)
    HumanService.set_character_skin(str(assets / "skins/young_caucasian_female/young_caucasian_female.mhmat"),
                                    bm, skin_type="ENHANCED_SSS", material_instances=False)
    skin = bpy.data.materials[f"{bm.name}.body"].node_tree.nodes[f"{bm.name}.body"]
    for k, v in {"Pore strength": 0.06, "Pore scale": 4500.0, "colorMixInStrength": 0.12,
                 "SSS strength": 0.3}.items():
        skin.inputs[k].default_value = v
    sub = bm.modifiers.new("EF_Subdiv", "SUBSURF")
    sub.levels, sub.render_levels = 0, 2
    bpy.context.view_layer.update()

    fits = eye_centers(bm)
    C = {s: fits[s][0] for s in "LR"}
    r_fit = float(np.mean([fits[s][1] for s in "LR"]))
    for s in "LR":
        log(f"sphere fit {s}: center {np.round(C[s], 5).tolist()} r {fits[s][1]*1000:.2f} mm "
            f"max residual {fits[s][2]*1000:.3f} mm")
    U = np.array([0.0, 0.0, 1.0])
    F = np.cross(C["L"] - C["R"], U)
    F /= np.linalg.norm(F)
    M = 0.5 * (C["L"] + C["R"])
    co = evaluated_coords(bm)
    head = co[co[:, 2] > M[2] - 0.12]
    assert (M - head.mean(0)) @ F > 0, "forward-axis detection failed (eyes behind head centroid)"
    log(f"forward axis F = {np.round(F, 6).tolist()} (eyes are {((M - head.mean(0)) @ F)*1000:.1f} mm "
        "in front of the head centroid)")
    bone_report = fix_bones(arm, C, F)

    cam_loc = M + 0.40 * F
    R_wc = calib.look_at(cam_loc, M, U)
    R_img = R_wc[:, 0]
    frame = Matrix(np.column_stack([R_img, U, F]).tolist()).to_4x4()
    R_s = r_fit - SCLERA_INSET * MM
    mats = dict(sclera=mat_sclera(), cornea=mat_cornea(), iris=mat_iris(ph.Params().iris_color),
                pupil=mat_pupil())
    mask_materials()
    eye_geo = {}
    for s in "LR":
        fr = frame.copy()
        fr.translation = Vector(C[s])
        eye_geo[s] = build_eye(s, C[s], R_s, fr, arm, mats)
    g0 = eye_geo["L"]
    log(f"eye: sclera R {R_s*1000:.2f} mm, apex {g0['apex']*1000:.2f} mm, pupil plane "
        f"{g0['pupil_plane']*1000:.2f} mm, iris {(g0['apex']-g0['iris_plane'])*1000:.2f} mm behind apex")
    blink_keys = add_blink_keys(bm)
    lids = lid_margin_vertices(bm, C, F, U, R_img, blink_keys)
    lid_eff = lid_efficiency(arm, bm, C, lids, R_img)

    # ---- camera, locked 40 cm in front of the eye midpoint, lens auto-calibrated
    geom = ph.Geometry(C_L=C["L"], C_R=C["R"], F=F, U=U, R_img=R_img, pupil_depth=g0["pupil_plane"])
    roi = np.vstack([evaluated_coords(o) for o in bpy.data.objects
                     if o.type == "MESH" and ("brow" in o.name.lower() or "lash" in o.name.lower())])
    pairs = []
    for h, v, ax in ((-1, 0, 0), (1, 0, 0), (0, 1, 1), (0, -1, 1)):
        c0, c1 = ph.solve(0, 0, geom), ph.solve(h, v, geom)
        for s in "LR":
            pairs.append((c0["eyes"][s]["pupil_center_3d"], c1["eyes"][s]["pupil_center_3d"], ax))
    lens = calib.auto_lens(cam_loc, R_wc, roi, pairs, W)
    assert lens["min_shift_px"] >= 15.0, lens
    log(f"lens: {lens['lens_mm']} mm (framing allows <= {lens['lens_fit_mm']:.1f} mm, "
        f"15 px shift needs >= {lens['lens_min_for_shift_mm']:.1f} mm); predicted min pupil shift "
        f"{lens['min_shift_px']:.1f} px")
    cd = bpy.data.cameras.new("EF_Camera")
    cd.lens = lens["lens_mm"]
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = 36.0
    cd.clip_start = 0.02
    cam = bpy.data.objects.new("EF_Camera", cd)
    scene.collection.objects.link(cam)
    cam.matrix_world = Matrix.Translation(Vector(cam_loc)) @ Matrix(R_wc).to_4x4()
    cam.lock_location = cam.lock_rotation = (True, True, True)
    scene.camera = cam

    # ---- constant lighting: soft key + fill + dim rim, flat dim world
    add_area("EF_Key", M + 0.70 * F + 0.30 * R_img + 0.35 * U, M, 0.16, 15.0, (1.0, 0.97, 0.93))
    add_area("EF_Fill", M + 0.70 * F - 0.50 * R_img + 0.05 * U, M, 0.3, 6.0, (0.93, 0.96, 1.0))
    add_area("EF_Rim", M - 0.25 * F + 0.55 * U, M, 0.3, 6.0)
    world = bpy.data.worlds.new("EF_World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.045, 0.045, 0.05, 1)
    scene.world = world
    setup_render(scene, args.samples)

    info = dict(
        version=1, blender=bpy.app.version_string, mpfb="2.0.18",
        C_L=C["L"].tolist(), C_R=C["R"].tolist(), M=M.tolist(), F=F.tolist(), U=U.tolist(),
        R_img=R_img.tolist(), ipd=float(np.linalg.norm(C["L"] - C["R"])),
        socket_sphere_radius=r_fit, eyeball_radius=R_s, pupil_depth=g0["pupil_plane"],
        eye_geometry_mm={k: v * 1000 for k, v in g0.items()},
        cornea_radius_mm=CORNEA_R, limbus_radius_mm=LIMBUS_R, cornea_ior=CORNEA_IOR,
        cam_loc=cam_loc.tolist(), cam_R_world_from_blendercam=R_wc.tolist(), lens=lens,
        lid_margin_vertices=lids, lid_efficiency=lid_eff, blink_keys=blink_keys, bones=bone_report, macro=macro, eye_trans_out_weight=trans_w,
        eye_bones=EYE_BONES, lid_bones=LID_BONES, build_log=LOG)
    scene["eyeforge"] = json.dumps(info)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.out).resolve()), compress=True)
    print("[build_scene] saved", args.out)


if __name__ == "__main__":
    main()
