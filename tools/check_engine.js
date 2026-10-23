// Node check of the explorer engine against scripts/psc_analysis.py results.
// Usage: node tools/check_engine.js
const fs = require("fs");
const { PSC } = require("./psc_engine.js");
const d = JSON.parse(fs.readFileSync("data/psc_explorer.json", "utf8"));
const e = new PSC(d);
const all = (v) => ({ x: v, y: v, z: v });
const out = {};
for (const om of d.omegas) {
  const ref = d.reference[om];
  const lab = e.simulate({ omega: om, T_uK: 5, loops: [
    { cycles: 30, t2: all(12), t1: all(6), hsr1: -4000 }, { cycles: 10, t2: all(13), t1: all(12), hsr1: -6000 }] });
  const b = ref.best;
  const best = e.simulate({ omega: om, T_uK: 5, loops: [
    { cycles: 30, t2: all(b.t2a), t1: all(b.t1a), hsr1: -4000 }, { cycles: 10, t2: all(b.t2b), t1: all(b.t1b), hsr1: -6000 }] });
  out[om] = { lab_js: lab.hist.at(-1).g3, lab_py: ref.lab, best_js: best.hist.at(-1).g3, best_py: ref.best_ground };
}
console.log(JSON.stringify(out, null, 1));
