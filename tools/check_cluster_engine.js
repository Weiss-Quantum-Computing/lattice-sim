// Node check of tools/cluster_engine.js. Reads a JSON file of cases written by
// tests/test_cluster.py (Stim reference values alongside), prints the engine's
// results as JSON. Usage: node tools/check_cluster_engine.js cases.json
"use strict";
const fs = require("fs");
const E = require("./cluster_engine.js");

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// packed phase rule against the per-qubit g function on random rows
function rowmulAgreement(trials, rng) {
  let bad = 0;
  for (let k = 0; k < trials; k++) {
    const n = 1 + Math.floor(rng() * 80);
    const t = new E.Tableau(n);
    for (const a of [0, 1]) for (let q = 0; q < n; q++) {
      t._set(t.xs, a, q, rng() < 0.5 ? 1 : 0); t._set(t.zs, a, q, rng() < 0.5 ? 1 : 0);
    }
    t.r[0] = rng() < 0.5 ? 1 : 0; t.r[1] = rng() < 0.5 ? 1 : 0;
    // keep only commuting pairs (rows of a stabilizer group commute)
    let anti = 0;
    for (let q = 0; q < n; q++) anti ^= (t._get(t.xs, 0, q) & t._get(t.zs, 1, q)) ^ (t._get(t.zs, 0, q) & t._get(t.xs, 1, q));
    if (anti) { k--; continue; }
    const u = t.clone();
    t._rowmul(0, 1); u._rowmulScalar(0, 1);
    let same = t.r[0] === u.r[0];
    for (let w = 0; w < t.W; w++) same = same && t.xs[w] === u.xs[w] && t.zs[w] === u.zs[w];
    if (!same) bad++;
  }
  return bad;
}

const cases = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = { rowmul_mismatches: rowmulAgreement(2000, mulberry32(7)), gates: {}, cases: [] };
for (const name of Object.keys(E.GATES1)) out.gates[name] = E.GATES1[name];
for (const c of cases.cases) {
  const spec = c.spec, res = { name: c.name };
  res.stabilizers = E.siteStabilizers(spec);
  const prog = E.compile(spec);
  const run = E.runTableau(prog, { snapshots: false, rand: mulberry32(1) });
  res.expectations = (c.paulis || []).map((P) => run.final.expectation(P));
  res.entropies = (c.regions || []).map((sites) => run.final.entropy(sites));
  if (c.noise) {
    const ex = E.sampleExact(spec, c.noise, c.shots, mulberry32(11));
    res.exact = ex;
    res.configs = {};
    for (const cfg of c.configs || []) res.configs[cfg] = E.sampleConfig(spec, cfg, c.noise, c.shots, mulberry32(13)).values;
  }
  out.cases.push(res);
}
process.stdout.write(JSON.stringify(out));
