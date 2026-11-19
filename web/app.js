import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { solve } from "./physiology.js";

// Blender world (Z up) -> glTF/three (Y up): (x, y, z) -> (x, z, -y)
const toThree = (v) => new THREE.Vector3(v[0], v[2], -v[1]);
const qToThree = (q) => new THREE.Quaternion(q[1], q[3], -q[2], q[0]); // [w,x,y,z] -> three (x,y,z,w)
const san = (n) => THREE.PropertyBinding.sanitizeNodeName(n);

const $ = (id) => document.getElementById(id);
const hud = $("hud");
const view = $("view");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.AgXToneMapping ?? THREE.ACESFilmicToneMapping;
view.prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1b1b20);
scene.add(new THREE.HemisphereLight(0xfff4ec, 0x30303a, 1.6));
const key = new THREE.DirectionalLight(0xffffff, 2.2);
scene.add(key);
const camera = new THREE.PerspectiveCamera(20, 1, 0.01, 10);

const geo = await (await fetch("./assets/eyeforge.json")).json();
const gltf = await new GLTFLoader().loadAsync("./assets/eyeforge.glb");
scene.add(gltf.scene);
scene.updateMatrixWorld(true);

const M = toThree(geo.M), F = toThree(geo.F);
camera.position.copy(M.clone().addScaledVector(F, 0.4));
camera.up.set(0, 1, 0);
camera.lookAt(M);
key.position.copy(M.clone().addScaledVector(F, 0.7).add(new THREE.Vector3(0.3, 0.35, 0)));
key.target.position.copy(M);
scene.add(key.target);

// each driven bone: q_local = parentWorld^-1 * q_world * parentWorld * restLocal (rotation about its head)
const bones = {};
for (const name of [geo.eye_bones.L, geo.eye_bones.R, ...geo.lid_bones.L, ...geo.lid_bones.R]) {
  const b = gltf.scene.getObjectByName(san(name));
  const pw = new THREE.Quaternion();
  b.parent.getWorldQuaternion(pw);
  bones[name] = { obj: b, rest: b.quaternion.clone(), pw, pwInv: pw.clone().invert() };
}
const setWorldRot = (name, qw) => {
  const b = bones[name];
  b.obj.quaternion.copy(b.pwInv).multiply(qw).multiply(b.pw).multiply(b.rest);
};
const pupils = { L: gltf.scene.getObjectByName(san("EF_Pupil.L")), R: gltf.scene.getObjectByName(san("EF_Pupil.R")) };
const R_img = geo.R_img;

const state = { h: 0, v: 0, profile: "physio", pupil: null, blink: 0 };

function apply() {
  const r = solve(state.h, state.v, geo, state.profile, { pupil_mm: state.pupil });
  for (const s of ["L", "R"]) {
    const e = r.eyes[s];
    setWorldRot(geo.eye_bones[s], qToThree(e.quat));
    const [up, lo] = geo.lid_bones[s];
    // blink: upper lid sweeps down ~40 deg, lower up ~8 deg (bone-driven approximation in the browser)
    const upA = -e.upper_lid / geo.lid_efficiency.upper + state.blink * 0.70;
    const loA = -e.lower_lid / geo.lid_efficiency.lower - state.blink * 0.14;
    const axis = toThree(R_img).normalize();
    setWorldRot(up, new THREE.Quaternion().setFromAxisAngle(axis, upA));
    setWorldRot(lo, new THREE.Quaternion().setFromAxisAngle(axis, loA));
    const k = r.pupil_mm / 2 / geo.pupil_mesh_radius_mm;
    pupils[s].scale.set(k, 1, k);  // exported disk lies in the node's local x-z plane
  }
  const deg = (x) => ((x * 180) / Math.PI).toFixed(2);
  hud.textContent =
    `eyes.move(h=${state.h.toFixed(3)}, v=${state.v.toFixed(3)})  [${state.profile}]\n` +
    `vergence ${deg(r.vergence)}°   pupil ${r.pupil_mm.toFixed(2)} mm\n` +
    `torsion L/R ${deg(r.eyes.L.torsion)}° / ${deg(r.eyes.R.torsion)}°\n` +
    `upper lid ${deg(r.eyes.L.upper_lid)}°   lower lid ${deg(r.eyes.L.lower_lid)}°`;
}

function setLens(mode) {
  const lens = mode === "wide" ? 35 : geo.lens_mm;
  const hfov = 2 * Math.atan(18 / lens);
  const a = view.clientWidth / view.clientHeight;
  camera.aspect = a;
  camera.fov = (2 * Math.atan(Math.tan(hfov / 2) / Math.min(a, 1)) * 180) / Math.PI;
  camera.updateProjectionMatrix();
}
function resize() {
  renderer.setSize(view.clientWidth, view.clientHeight, false);
  setLens($("lens").value);
}
addEventListener("resize", resize);
resize();

// ---- UI
const bindRange = (id, fn) => $(id).addEventListener("input", (e) => fn(parseFloat(e.target.value)));
bindRange("h", (x) => { state.h = x; $("hv").textContent = x.toFixed(2); });
bindRange("v", (x) => { state.v = x; $("vv").textContent = x.toFixed(2); });
bindRange("pupil", (x) => { state.pupil = x < 1.5 ? null : x; $("pupilv").textContent = state.pupil ? x.toFixed(1) : "auto"; });
bindRange("gain", (x) => { $("gainv").textContent = x; });
$("profile").addEventListener("change", (e) => (state.profile = e.target.value));
$("lens").addEventListener("change", (e) => setLens(e.target.value));
const setHV = (h, v) => {
  state.h = h; state.v = v;
  $("h").value = h; $("v").value = v;
  $("hv").textContent = h.toFixed(2); $("vv").textContent = v.toFixed(2);
};
$("reset").addEventListener("click", () => setHV(0, 0));
$("seq").addEventListener("click", async () => {
  for (let k = 0; k < 10; k++) {
    const t = (2 * Math.PI * k) / 10;
    setHV(Math.sin(t), Math.cos(t));
    await new Promise((r) => setTimeout(r, 600));
  }
  setHV(0, 0);
});

// ---- webcam gaze via MediaPipe Face Landmarker (iris landmarks vs eye corners / lids)
let landmarker = null, video = $("cam"), center = { x: 0, y: 0 }, smooth = { h: 0, v: 0 }, lastT = -1;
async function startWebcam() {
  const { FilesetResolver, FaceLandmarker } = await import("./vendor/mediapipe/vision_bundle.mjs");
  const fs = await FilesetResolver.forVisionTasks("./vendor/mediapipe/wasm");
  landmarker = await FaceLandmarker.createFromOptions(fs, {
    baseOptions: { modelAssetPath: "./vendor/mediapipe/face_landmarker.task" },
    runningMode: "VIDEO", numFaces: 1,
  });
  video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
  await video.play();
  video.style.display = "block";
  $("calib").hidden = false;
}
function eyeRatios(lm) {
  // per eye: iris position between the corners (x) and between the lids (y), 0..1
  const one = (iris, a, b, up, lo) => ({
    x: (lm[iris].x - Math.min(lm[a].x, lm[b].x)) / Math.abs(lm[a].x - lm[b].x),
    y: (lm[iris].y - lm[up].y) / (lm[lo].y - lm[up].y),
  });
  const r = one(468, 33, 133, 159, 145), l = one(473, 362, 263, 386, 374);
  return { x: (r.x + l.x) / 2, y: (r.y + l.y) / 2 };
}
let raw = { x: 0.5, y: 0.5 };
function trackWebcam() {
  if (!landmarker || video.readyState < 2 || video.currentTime === lastT) return;
  lastT = video.currentTime;
  const res = landmarker.detectForVideo(video, performance.now());
  if (!res.faceLandmarks?.length) return;
  raw = eyeRatios(res.faceLandmarks[0]);
  const g = parseFloat($("gain").value);
  // raw camera image: the user looking to their own left moves the irises to image right -> h > 0
  const h = THREE.MathUtils.clamp(g * (raw.x - 0.5 - center.x), -1, 1);
  const v = THREE.MathUtils.clamp(-g * 1.3 * (raw.y - 0.5 - center.y), -1, 1);
  smooth.h += 0.35 * (h - smooth.h);
  smooth.v += 0.35 * (v - smooth.v);
  setHV(smooth.h, smooth.v);
}
$("calib").addEventListener("click", () => (center = { x: raw.x - 0.5, y: raw.y - 0.5 }));
$("mode").addEventListener("change", async (e) => {
  if (e.target.value === "webcam" && !landmarker) {
    hud.textContent = "starting webcam…";
    try { await startWebcam(); } catch (err) { hud.textContent = `webcam unavailable: ${err.message}`; }
  }
});

// idle blinks every 3-5 s (close 100 ms, open 150 ms)
let nextBlink = performance.now() + 2500;
function blinkValue(now) {
  const t = now - nextBlink;
  if (t < 0) return 0;
  if (t < 100) return t / 100;
  if (t < 250) return 1 - (t - 100) / 150;
  nextBlink = now + 3000 + Math.random() * 2000;
  return 0;
}

renderer.setAnimationLoop((now) => {
  if ($("mode").value === "webcam") trackWebcam();
  state.blink = blinkValue(now);
  apply();
  renderer.render(scene, camera);
});
