// Port of wrapper/physiology.py (same math, same Blender world frame: Z up, meters).
// Quaternions are [w, x, y, z]. Parity with Python is tested by tests/test_web_parity.py.

export const PROFILES = {
  spec: { rotation: "lookat", lids: false, pupil: "fixed", kappa: false, l2: false },
  physio: { rotation: "listing", lids: true, pupil: "near_triad", kappa: false, l2: false },
  physio_full: { rotation: "listing", lids: true, pupil: "near_triad", kappa: true, l2: true },
};

export const DEFAULT_PARAMS = {
  pupil_mm: null, pupil_fixed_mm: 4.0, pupil_base_mm: 4.5, pupil_near_gain: 0.35,
  kappa_deg: 5.0, l2_gain: 0.25, upper_lid_gain: 1.0, lower_lid_gain: 0.3,
};

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => Math.hypot(a[0], a[1], a[2]);
const unit = (a) => { const n = norm(a); return [a[0] / n, a[1] / n, a[2] / n]; };
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

export function qAxisAngle(axis, angle) {
  const n = norm(axis);
  if (n < 1e-15 || Math.abs(angle) < 1e-15) return [1, 0, 0, 0];
  const s = Math.sin(angle / 2) / n;
  return [Math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s];
}

export function qMul(a, b) {
  const [aw, ax, ay, az] = a, [bw, bx, by, bz] = b;
  return [aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
          aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw];
}

export const qConj = (q) => [q[0], -q[1], -q[2], -q[3]];

export function qNormalize(q) {
  const n = Math.hypot(...q);
  const r = q.map((x) => x / n);
  return r[0] >= 0 ? r : r.map((x) => -x);
}

export function qRotate(q, v) {
  const p = qMul(qMul(q, [0, v[0], v[1], v[2]]), qConj(q));
  return [p[1], p[2], p[3]];
}

export function qBetween(a, b) {
  a = unit(a); b = unit(b);
  const ax = cross(a, b), s = norm(ax), c = clamp(dot(a, b), -1, 1);
  if (s < 1e-12) return [1, 0, 0, 0];
  return qNormalize(qAxisAngle(scale(ax, 1 / s), Math.atan2(s, c)));
}

const twist = (q, axis) => 2 * Math.atan2(dot([q[1], q[2], q[3]], unit(axis)), q[0]);

export function target(h, v, g) {
  const M = scale(add(g.C_L, g.C_R), 0.5), r = g.target_range ?? 0.05, d = g.target_dist ?? 0.35;
  return add(add(M, scale(g.F, d)), add(scale(g.R_img, r * clamp(h, -1, 1)), scale(g.U, r * clamp(v, -1, 1))));
}

export function solve(h, v, g, profile = "physio", params = DEFAULT_PARAMS) {
  const prof = PROFILES[profile], p = { ...DEFAULT_PARAMS, ...params };
  const hc = clamp(h, -1, 1), vc = clamp(v, -1, 1);
  const T = target(hc, vc, g);
  const gaze = { L: unit(sub(T, g.C_L)), R: unit(sub(T, g.C_R)) };
  const vergence = Math.acos(clamp(dot(gaze.L, gaze.R), -1, 1));
  const kappa = prof.kappa ? (p.kappa_deg * Math.PI) / 180 : 0;
  const l2 = prof.l2 ? p.l2_gain * vergence : 0;
  const nasal = { L: -1, R: 1 };
  const eyes = {};
  for (const s of ["L", "R"]) {
    const gz = gaze[s];
    const v0 = qRotate(qAxisAngle(g.U, nasal[s] * kappa), g.F);
    const listing = qBetween(v0, gz);
    let q;
    if (prof.rotation === "lookat") {
      const az = Math.atan2(dot(gz, g.R_img), dot(gz, g.F));
      const el = Math.asin(clamp(dot(gz, g.U), -1, 1));
      q = qNormalize(qMul(qAxisAngle(g.U, az), qAxisAngle(g.R_img, -el)));
    } else {
      const qPrim = qAxisAngle(g.U, -nasal[s] * l2);
      q = qNormalize(qMul(qBetween(qRotate(qPrim, v0), gz), qPrim));
    }
    const el = Math.asin(clamp(dot(gz, g.U), -1, 1));
    eyes[s] = {
      quat: q, visual_axis: gz, optical_axis: qRotate(q, g.F), elevation: el,
      torsion: twist(qMul(q, qConj(listing)), gz),
      upper_lid: prof.lids ? p.upper_lid_gain * el : 0, lower_lid: prof.lids ? p.lower_lid_gain * el : 0,
    };
  }
  const dist = norm(sub(T, scale(add(g.C_L, g.C_R), 0.5)));
  let pupil = p.pupil_mm ?? (prof.pupil === "fixed" ? p.pupil_fixed_mm : p.pupil_base_mm - p.pupil_near_gain / dist);
  pupil = clamp(pupil, 1.5, 8.0);
  return { h: hc, v: vc, target: T, vergence, pupil_mm: pupil, eyes };
}
