// Prints JS physiology results for the grid given on stdin (used by tests/test_web_parity.py).
import { solve } from "./physiology.js";
let buf = "";
process.stdin.on("data", (d) => (buf += d));
process.stdin.on("end", () => {
  const { geom, cases } = JSON.parse(buf);
  const out = cases.map(([h, v, prof]) => {
    const r = solve(h, v, geom, prof);
    return { quat_L: r.eyes.L.quat, quat_R: r.eyes.R.quat, torsion_L: r.eyes.L.torsion,
             vergence: r.vergence, pupil_mm: r.pupil_mm, upper_lid_L: r.eyes.L.upper_lid };
  });
  process.stdout.write(JSON.stringify(out));
});
