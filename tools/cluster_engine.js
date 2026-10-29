// Stabilizer engine for the cluster-state explorer (tools/cluster_explorer.html).
//
// * Tableau: Aaronson-Gottesman (CHP) tableau with bit-packed rows and
//   destabilizers; the Pauli-product phase uses Stim's word-parallel rule
//   (inplace_right_mul_returning_log_i_scalar), cross-checked against the
//   per-qubit g-function in tools/check_cluster_engine.js.
// * FrameSim: Pauli-frame sampler (Stim's frame simulator): 32 shots per
//   word, random Z at reset and after every measurement, outcome =
//   reference outcome XOR frame X bit.
// * compile(): the experiment sequence (same JSON as
//   lattice_sim/cluster/sequence.py) -> list of ops, step by step.
//
// Semantics follow lattice_sim.cluster.sequence exactly; tests/test_cluster.py
// runs this file under node against Stim.
"use strict";

// --------------------------------------------------------------- single-qubit Cliffords
// Conjugation tables U P U^dag for P = X, Y, Z (index 1, 2, 3): [sign, image].
const X = 1, Y = 2, Z = 3;
const GATES1 = {
  I: [[1, X], [1, Y], [1, Z]],
  H: [[1, Z], [-1, Y], [1, X]],
  S: [[1, Y], [-1, X], [1, Z]],
  S_DAG: [[-1, Y], [1, X], [1, Z]],
  X: [[1, X], [-1, Y], [-1, Z]],
  Y: [[-1, X], [1, Y], [-1, Z]],
  Z: [[-1, X], [-1, Y], [1, Z]],
  SQRT_X: [[1, X], [1, Z], [-1, Y]],
  SQRT_X_DAG: [[1, X], [-1, Z], [1, Y]],
  SQRT_Y: [[-1, Z], [1, Y], [1, X]],
  SQRT_Y_DAG: [[1, Z], [1, Y], [-1, X]],
};
const PULSE = { // [phase_q][q] -> gate, R_phi(q pi/2) up to a global phase (sequence._PULSE)
  0: { 1: "SQRT_X", 2: "X", 3: "SQRT_X_DAG" }, 1: { 1: "SQRT_Y", 2: "Y", 3: "SQRT_Y_DAG" },
  2: { 1: "SQRT_X_DAG", 2: "X", 3: "SQRT_X" }, 3: { 1: "SQRT_Y_DAG", 2: "Y", 3: "SQRT_Y" },
};
const RZ = { 1: "S", 2: "Z", 3: "S_DAG" };
const LETTER = ["I", "X", "Y", "Z"];
// pauli code from bits: x,z -> 0 I, 1 X, 2 Y, 3 Z
const CODE = (x, z) => (x ? (z ? Y : X) : (z ? Z : 0));
const BITS = [[0, 0], [1, 0], [1, 1], [0, 1]];

function popcount(v) {
  v = v - ((v >>> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  return (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

// --------------------------------------------------------------- tableau
class Tableau {
  constructor(n) {
    this.n = n;
    this.W = Math.max(1, Math.ceil(n / 32));
    const rows = 2 * n + 1;
    this.xs = new Uint32Array(rows * this.W);
    this.zs = new Uint32Array(rows * this.W);
    this.r = new Uint8Array(rows);
    for (let i = 0; i < n; i++) {
      this._set(this.xs, i, i, 1);          // destabilizer X_i
      this._set(this.zs, n + i, i, 1);      // stabilizer Z_i
    }
  }
  clone() {
    const t = Object.create(Tableau.prototype);
    t.n = this.n; t.W = this.W;
    t.xs = this.xs.slice(); t.zs = this.zs.slice(); t.r = this.r.slice();
    return t;
  }
  _get(a, row, q) { return (a[row * this.W + (q >>> 5)] >>> (q & 31)) & 1; }
  _set(a, row, q, v) {
    const k = row * this.W + (q >>> 5), m = 1 << (q & 31);
    if (v) a[k] |= m; else a[k] &= ~m;
  }
  pauli(row, q) { return CODE(this._get(this.xs, row, q), this._get(this.zs, row, q)); }

  gate1(name, q) {
    const tab = GATES1[name];
    if (!tab) throw new Error("unknown gate " + name);
    const w = q >>> 5, m = 1 << (q & 31), W = this.W;
    for (let row = 0; row < 2 * this.n; row++) {
      const k = row * W + w;
      const x = (this.xs[k] & m) !== 0, z = (this.zs[k] & m) !== 0;
      const p = CODE(x, z);
      if (!p) continue;
      const [s, img] = tab[p - 1];
      if (s < 0) this.r[row] ^= 1;
      const [nx, nz] = BITS[img];
      if (nx) this.xs[k] |= m; else this.xs[k] &= ~m;
      if (nz) this.zs[k] |= m; else this.zs[k] &= ~m;
    }
  }
  cz(a, b) {
    const W = this.W, wa = a >>> 5, ma = 1 << (a & 31), wb = b >>> 5, mb = 1 << (b & 31);
    for (let row = 0; row < 2 * this.n; row++) {
      const o = row * W;
      const xa = (this.xs[o + wa] & ma) !== 0, za = (this.zs[o + wa] & ma) !== 0;
      const xb = (this.xs[o + wb] & mb) !== 0, zb = (this.zs[o + wb] & mb) !== 0;
      if (xa && xb && (za !== zb)) this.r[row] ^= 1;
      if (xb) this.zs[o + wa] ^= ma;
      if (xa) this.zs[o + wb] ^= mb;
    }
  }
  // log_i phase of P_h * P_i (both Hermitian); h <- h * i
  _rowmul(h, i) {
    const W = this.W, oh = h * W, oi = i * W;
    let cnt1 = 0, cnt2 = 0, s1 = 0, s2 = 0;
    for (let w = 0; w < W; w++) {
      const x1 = this.xs[oh + w], z1 = this.zs[oh + w], x2 = this.xs[oi + w], z2 = this.zs[oi + w];
      const nx1 = x1 ^ x2, nz1 = z1 ^ z2;
      const x1z2 = x1 & z2;
      const anti = (x2 & z1) ^ x1z2;
      cnt2 = (cnt2 ^ ((cnt1 ^ nx1 ^ nz1 ^ x1z2) & anti)) >>> 0;
      cnt1 = (cnt1 ^ anti) >>> 0;
      this.xs[oh + w] = nx1; this.zs[oh + w] = nz1;
      s1 += popcount(cnt1); s2 += popcount(cnt2);
      cnt1 = 0; cnt2 = 0;
    }
    const logi = (s1 + 2 * s2) & 3;          // P_h P_i = i^logi * (+-) P_new
    // sign of the product: (-1)^(r_h + r_i) * i^logi; logi is 0 or 2 for commuting rows
    this.r[h] = (this.r[h] ^ this.r[i] ^ (logi >>> 1)) & 1;
    return logi;
  }
  // reference: per-qubit g function (Aaronson & Gottesman 2004)
  _rowmulScalar(h, i) {
    let sum = 2 * this.r[h] + 2 * this.r[i];
    for (let q = 0; q < this.n; q++) {
      const x1 = this._get(this.xs, i, q), z1 = this._get(this.zs, i, q);
      const x2 = this._get(this.xs, h, q), z2 = this._get(this.zs, h, q);
      let g = 0;
      if (x1 && z1) g = z2 - x2;
      else if (x1 && !z1) g = z2 * (2 * x2 - 1);
      else if (!x1 && z1) g = x2 * (1 - 2 * z2);
      sum += g;
      this._set(this.xs, h, q, x1 ^ x2); this._set(this.zs, h, q, z1 ^ z2);
    }
    sum = ((sum % 4) + 4) % 4;
    this.r[h] = sum === 2 ? 1 : 0;
  }
  _copyRow(dst, src) {
    const W = this.W;
    this.xs.copyWithin(dst * W, src * W, src * W + W);
    this.zs.copyWithin(dst * W, src * W, src * W + W);
    this.r[dst] = this.r[src];
  }
  _clearRow(row) {
    const W = this.W;
    this.xs.fill(0, row * W, row * W + W); this.zs.fill(0, row * W, row * W + W); this.r[row] = 0;
  }
  // Z measurement; forced: 0/1 to choose a random outcome (else Math.random). Returns {bit, random}.
  measureZ(q, rand) {
    const n = this.n, W = this.W, w = q >>> 5, m = 1 << (q & 31);
    let p = -1;
    for (let row = n; row < 2 * n; row++) if (this.xs[row * W + w] & m) { p = row; break; }
    if (p >= 0) {
      for (let row = 0; row < 2 * n; row++)
        if (row !== p && (this.xs[row * W + w] & m)) this._rowmul(row, p);
      this._copyRow(p - n, p);
      this._clearRow(p);
      const bit = (rand !== undefined ? rand() : Math.random()) < 0.5 ? 0 : 1;
      this.r[p] = bit;
      this.zs[p * W + w] |= m;
      return { bit, random: true };
    }
    const s = 2 * n;
    this._clearRow(s);
    for (let i = 0; i < n; i++) if (this.xs[i * W + w] & m) this._rowmul(s, i + n);
    return { bit: this.r[s], random: false };
  }
  measure(q, basis, rand) {
    if (basis === "X") { this.gate1("H", q); const o = this.measureZ(q, rand); this.gate1("H", q); return o; }
    if (basis === "Y") { this.gate1("SQRT_X", q); const o = this.measureZ(q, rand); this.gate1("SQRT_X_DAG", q); return o; }
    return this.measureZ(q, rand);
  }
  reset(q, rand) { const o = this.measureZ(q, rand); if (o.bit) this.gate1("X", q); }

  // stabilizer row of qubit i (image of Z_i while no measurement has happened)
  row(i) {
    const ops = {};
    for (let q = 0; q < this.n; q++) { const p = this.pauli(this.n + i, q); if (p) ops[q] = LETTER[p]; }
    return { sign: this.r[this.n + i] ? -1 : 1, ops };
  }
  // <P> for P = {sign, ops: {q: "X"|"Y"|"Z"}}: +1, -1 or 0
  expectation(P) {
    const n = this.n, W = this.W, s = 2 * n;
    const px = new Uint32Array(W), pz = new Uint32Array(W);
    for (const [q, L] of Object.entries(P.ops)) {
      const c = LETTER.indexOf(L), [bx, bz] = BITS[c], qi = +q;
      if (bx) px[qi >>> 5] |= 1 << (qi & 31);
      if (bz) pz[qi >>> 5] |= 1 << (qi & 31);
    }
    const anti = (row) => {
      let a = 0;
      for (let w = 0; w < W; w++) a ^= popcount(((this.xs[row * W + w] & pz[w]) ^ (this.zs[row * W + w] & px[w])) >>> 0);
      return a & 1;
    };
    for (let row = n; row < 2 * n; row++) if (anti(row)) return 0;
    this._clearRow(s);
    for (let i = 0; i < n; i++) if (anti(i)) this._rowmul(s, i + n);
    for (let w = 0; w < W; w++) if (this.xs[s * W + w] !== px[w] || this.zs[s * W + w] !== pz[w]) throw new Error("expectation: product mismatch");
    const sign = (P.sign || 1) < 0 ? 1 : 0;
    return this.r[s] === sign ? 1 : -1;
  }
  // entanglement entropy (bits) of the reduced state on `sites`: rank(G|A) - |A|
  entropy(sites) {
    const k = sites.length;
    if (!k) return 0;
    const n = this.n, C = 2 * k, CW = Math.ceil(C / 32);
    const M = new Uint32Array(n * CW);
    for (let r = 0; r < n; r++) {
      const row = n + r;
      for (let j = 0; j < k; j++) {
        const q = sites[j];
        if (this._get(this.xs, row, q)) M[r * CW + (j >>> 5)] |= 1 << (j & 31);
        if (this._get(this.zs, row, q)) M[r * CW + ((j + k) >>> 5)] |= 1 << ((j + k) & 31);
      }
    }
    return gf2Rank(M, n, C, CW) - k;
  }
}

function gf2Rank(M, rows, cols, CW) {
  let rank = 0;
  for (let c = 0; c < cols && rank < rows; c++) {
    const w = c >>> 5, m = 1 << (c & 31);
    let piv = -1;
    for (let r = rank; r < rows; r++) if (M[r * CW + w] & m) { piv = r; break; }
    if (piv < 0) continue;
    if (piv !== rank) for (let j = 0; j < CW; j++) { const t = M[piv * CW + j]; M[piv * CW + j] = M[rank * CW + j]; M[rank * CW + j] = t; }
    for (let r = 0; r < rows; r++)
      if (r !== rank && (M[r * CW + w] & m)) for (let j = w; j < CW; j++) M[r * CW + j] ^= M[rank * CW + j];
    rank++;
  }
  return rank;
}

// --------------------------------------------------------------- frames
class FrameSim {
  constructor(n, shots, rng) {
    this.n = n; this.shots = shots; this.S = Math.ceil(shots / 32);
    this.fx = new Uint32Array(n * this.S); this.fz = new Uint32Array(n * this.S);
    this.rng = rng || Math.random;
    this.lastMask = (shots % 32) ? ((1 << (shots % 32)) - 1) >>> 0 : 0xffffffff;
    for (let q = 0; q < n; q++) this._randomize(this.fz, q);
  }
  _randomize(a, q) {
    const o = q * this.S;
    for (let s = 0; s < this.S; s++) a[o + s] ^= ((this.rng() * 4294967296) >>> 0);
  }
  // Bernoulli(p) mask over all shots, geometric skipping
  mask(p) {
    const m = new Uint32Array(this.S);
    if (p <= 0) return m;
    if (p >= 1) { m.fill(0xffffffff); return m; }
    const l = Math.log1p(-p);
    let i = Math.floor(Math.log(1 - this.rng()) / l);
    while (i < this.shots) { m[i >>> 5] |= 1 << (i & 31); i += 1 + Math.floor(Math.log(1 - this.rng()) / l); }
    return m;
  }
  gate1(name, q) {
    const o = q * this.S, S = this.S, fx = this.fx, fz = this.fz;
    switch (name) {
      case "H": case "SQRT_Y": case "SQRT_Y_DAG":
        for (let s = 0; s < S; s++) { const t = fx[o + s]; fx[o + s] = fz[o + s]; fz[o + s] = t; } break;
      case "S": case "S_DAG": for (let s = 0; s < S; s++) fz[o + s] ^= fx[o + s]; break;
      case "SQRT_X": case "SQRT_X_DAG": for (let s = 0; s < S; s++) fx[o + s] ^= fz[o + s]; break;
      case "X": case "Y": case "Z": case "I": break;
      default: throw new Error("unknown gate " + name);
    }
  }
  cz(a, b) {
    const oa = a * this.S, ob = b * this.S;
    for (let s = 0; s < this.S; s++) { this.fz[oa + s] ^= this.fx[ob + s]; this.fz[ob + s] ^= this.fx[oa + s]; }
  }
  xError(q, p) { const m = this.mask(p), o = q * this.S; for (let s = 0; s < this.S; s++) this.fx[o + s] ^= m[s]; }
  zError(q, p) { const m = this.mask(p), o = q * this.S; for (let s = 0; s < this.S; s++) this.fz[o + s] ^= m[s]; }
  zzError(a, b, p) {
    const m = this.mask(p), oa = a * this.S, ob = b * this.S;
    for (let s = 0; s < this.S; s++) { this.fz[oa + s] ^= m[s]; this.fz[ob + s] ^= m[s]; }
  }
  depolarize1(q, p) {
    const m = this.mask(p), o = q * this.S;
    for (let s = 0; s < this.S; s++) {
      let v = m[s];
      while (v) {
        const b = v & -v; v ^= b;
        const k = 1 + Math.floor(this.rng() * 3);   // 1 X, 2 Y, 3 Z
        const [bx, bz] = BITS[k];
        if (bx) this.fx[o + s] ^= b;
        if (bz) this.fz[o + s] ^= b;
      }
    }
  }
  depolarize2(a, b, p) {
    const m = this.mask(p), oa = a * this.S, ob = b * this.S;
    for (let s = 0; s < this.S; s++) {
      let v = m[s];
      while (v) {
        const bit = v & -v; v ^= bit;
        const k = 1 + Math.floor(this.rng() * 15), pa = k >> 2, pb = k & 3;
        const [ax, az] = BITS[pa], [bx, bz] = BITS[pb];
        if (ax) this.fx[oa + s] ^= bit; if (az) this.fz[oa + s] ^= bit;
        if (bx) this.fx[ob + s] ^= bit; if (bz) this.fz[ob + s] ^= bit;
      }
    }
  }
  // returns flip mask of a Z-basis measurement, then randomizes Z
  measureZ(q, pFlip) {
    const o = q * this.S, out = this.fx.slice(o, o + this.S);
    if (pFlip) { const m = this.mask(pFlip); for (let s = 0; s < this.S; s++) out[s] ^= m[s]; }
    this._randomize(this.fz, q);
    return out;
  }
  measure(q, basis, pFlip) {
    if (basis === "X") { this.gate1("H", q); const o = this.measureZ(q, pFlip); this.gate1("H", q); return o; }
    if (basis === "Y") { this.gate1("SQRT_X", q); const o = this.measureZ(q, pFlip); this.gate1("SQRT_X_DAG", q); return o; }
    return this.measureZ(q, pFlip);
  }
  reset(q) { this.fx.fill(0, q * this.S, (q + 1) * this.S); this.fz.fill(0, q * this.S, (q + 1) * this.S); this._randomize(this.fz, q); }
  // anticommutation mask of the frame with a Pauli {ops}
  anticommute(P) {
    const out = new Uint32Array(this.S);
    for (const [q, L] of Object.entries(P.ops)) {
      const [bx, bz] = BITS[LETTER.indexOf(L)], o = +q * this.S;
      for (let s = 0; s < this.S; s++) out[s] ^= (bx ? this.fz[o + s] : 0) ^ (bz ? this.fx[o + s] : 0);
    }
    return out;
  }
}

// --------------------------------------------------------------- geometry
function regionOf(spec) {
  const [nx, ny, nz] = spec.shape;
  return {
    nx, ny, nz, n: nx * ny * nz,
    index: (x, y, z) => x + nx * (y + ny * z),
    coords: (i) => [i % nx, Math.floor(i / nx) % ny, Math.floor(i / (nx * ny))],
  };
}
function bondsOf(reg, occ, axis) {
  const ax = "xyz".indexOf(axis), out = [], dims = [reg.nx, reg.ny, reg.nz];
  for (let r = 0; r < reg.n; r++) {
    if (!occ[r]) continue;
    const c = reg.coords(r); c[ax] += 1;
    if (c[ax] < dims[ax]) { const s = reg.index(c[0], c[1], c[2]); if (occ[s]) out.push([r, s]); }
  }
  return out;
}

// --------------------------------------------------------------- noise
const IDEAL = { prep: 0, pulse: 0, addressed: 0, spectator: 0, echo_contrast: 1, phase_offset: 0, phase_sigma: 0, leak: 0, readout: 0 };
function zzProbability(nz) {
  return 0.5 * (1 - Math.exp(-(nz.phase_sigma ** 2) / 8) * Math.cos(nz.phase_offset / 2));
}

// --------------------------------------------------------------- compile
// ops: {t:"g1", g, q} | {t:"cz", a, b} | {t:"m", q, basis, p} | {t:"r", q}
//      | noise {t:"X"|"Z"|"D1", q, p} | {t:"ZZ"|"D2", a, b, p}
// steps[k] = {from, to, label, kind, bonds, occAfter}
function compile(spec, noise) {
  const nz = Object.assign({}, IDEAL, noise || {});
  const reg = regionOf(spec);
  let occ = spec.occupancy.map((v) => !!v);
  const ops = [], steps = [];
  const present = () => occ.map((v, i) => (v ? i : -1)).filter((i) => i >= 0);
  const sitesOf = (st) => (st.sites == null ? present() : st.sites.map(Number).filter((i) => occ[i]));
  for (const q of present()) if (nz.prep) ops.push({ t: "X", q, p: nz.prep });
  const prepTo = ops.length;
  const single = (st, tg, addressed) => {
    if (addressed) {
      if (nz.addressed) for (const q of tg) ops.push({ t: "D1", q, p: nz.addressed });
      if (nz.spectator) { const set = new Set(tg); for (const q of present()) if (!set.has(q)) ops.push({ t: "Z", q, p: nz.spectator }); }
    } else if (nz.pulse) for (const q of tg) ops.push({ t: "D1", q, p: nz.pulse });
  };
  for (const st of spec.steps) {
    const from = ops.length;
    let bonds = null;
    if (st.op === "pulse") {
      const q = ((st.q % 4) + 4) % 4, p = (((st.phase_q || 0) % 4) + 4) % 4, tg = sitesOf(st);
      if (q) { for (const i of tg) ops.push({ t: "g1", g: PULSE[p][q], q: i }); single(st, tg, st.sites != null); }
    } else if (st.op === "rz") {
      const q = ((st.q % 4) + 4) % 4, tg = sitesOf(st);
      if (q) { for (const i of tg) ops.push({ t: "g1", g: RZ[q], q: i }); single(st, tg, true); }
    } else if (st.op === "collide") {
      const mode = st.mode || "echo";
      bonds = {};
      for (const a of st.axes) bonds[a] = bondsOf(reg, occ, a);
      const all = [].concat(...st.axes.split("").map((a) => bonds[a]));
      const pres = present();
      if (mode === "echo") {
        for (const [r, s] of all) { ops.push({ t: "cz", a: r, b: s }, { t: "g1", g: "S_DAG", q: r }, { t: "g1", g: "S_DAG", q: s }); }
        for (const q of pres) ops.push({ t: "g1", g: "X", q });
        if (nz.pulse) for (const q of pres) ops.push({ t: "D1", q, p: nz.pulse });
        const pzz = zzProbability(nz);
        if (pzz) for (const [r, s] of all) ops.push({ t: "ZZ", a: r, b: s, p: pzz });
      } else if (mode === "single") {
        for (const [r, s] of all) ops.push({ t: "g1", g: "X", q: s }, { t: "cz", a: r, b: s }, { t: "g1", g: "X", q: s });
        const p = zzProbability(nz);
        if (p) for (const [r, s] of all) ops.push({ t: "Z", q: r, p }, { t: "Z", q: s, p }, { t: "ZZ", a: r, b: s, p });
      } else throw new Error("unknown collision mode " + mode);
      if (nz.leak) for (const [r, s] of all) ops.push({ t: "D2", a: r, b: s, p: nz.leak });
      if (nz.echo_contrast < 1) for (const q of pres) ops.push({ t: "Z", q, p: (1 - nz.echo_contrast) / 2 });
    } else if (st.op === "measure") {
      const tg = sitesOf(st);
      for (const q of tg) {
        const basis = st.bases ? (st.bases[q] || st.bases[String(q)] || "Z") : (st.basis || "Z");
        ops.push({ t: "m", q, basis, p: nz.readout });
      }
    } else if (st.op === "clear") {
      const tg = sitesOf(st);
      for (const q of tg) ops.push({ t: "r", q });
      occ = occ.slice(); for (const q of tg) occ[q] = false;
    } else throw new Error("unknown step " + st.op);
    steps.push({ from, to: ops.length, op: st.op, bonds, occAfter: occ.slice() });
  }
  return { n: reg.n, reg, ops, steps, prepTo, occ0: spec.occupancy.map((v) => !!v) };
}

// --------------------------------------------------------------- running
// Noiseless tableau run with a snapshot after every step; measurement records.
function runTableau(prog, opts) {
  opts = opts || {};
  const rand = opts.rand || Math.random;
  const t = new Tableau(prog.n);
  const snaps = [t.clone()], records = [];
  let k = 0;
  for (const st of prog.steps) {
    for (; k < st.to; k++) {
      const o = prog.ops[k];
      if (o.t === "g1") t.gate1(o.g, o.q);
      else if (o.t === "cz") t.cz(o.a, o.b);
      else if (o.t === "m") { const res = t.measure(o.q, o.basis, rand); records.push({ q: o.q, basis: o.basis, bit: res.bit, random: res.random, op: k }); }
      else if (o.t === "r") t.reset(o.q, rand);
    }
    if (opts.snapshots !== false) snaps.push(t.clone());
  }
  return { final: t, snaps, records };
}

// Pauli-frame sampling of the whole program. Returns per-measurement shot bits
// (reference XOR flips) as Uint32Array masks, and the frame object at `stopAt` ops.
function runFrames(prog, shots, opts) {
  opts = opts || {};
  const rng = opts.rng || Math.random;
  const ref = opts.reference || runTableau(prog, { snapshots: false, rand: rng }).records;
  const f = new FrameSim(prog.n, shots, rng);
  const meas = [];
  let mi = 0;
  const stop = opts.stopAt == null ? prog.ops.length : opts.stopAt;
  for (let k = 0; k < stop; k++) {
    const o = prog.ops[k];
    switch (o.t) {
      case "g1": f.gate1(o.g, o.q); break;
      case "cz": f.cz(o.a, o.b); break;
      case "X": f.xError(o.q, o.p); break;
      case "Z": f.zError(o.q, o.p); break;
      case "D1": f.depolarize1(o.q, o.p); break;
      case "ZZ": f.zzError(o.a, o.b, o.p); break;
      case "D2": f.depolarize2(o.a, o.b, o.p); break;
      case "m": {
        const flips = f.measure(o.q, o.basis, o.p), r = ref[mi++];
        if (r.bit) for (let s = 0; s < f.S; s++) flips[s] = ~flips[s] >>> 0;
        meas.push({ q: o.q, basis: o.basis, bits: flips });
        break;
      }
      case "r": f.reset(o.q); break;
      default: throw new Error("op " + o.t);
    }
  }
  return { frames: f, meas };
}

// index of the first non-unitary step (measure / clear), or steps.length
function unitaryStepCount(spec) {
  const k = spec.steps.findIndex((s) => s.op === "measure" || s.op === "clear");
  return k < 0 ? spec.steps.length : k;
}
function prefixSpec(spec, k) { return Object.assign({}, spec, { steps: spec.steps.slice(0, k) }); }

// <S_i> per site and the fidelity, from the frame at the end of the unitary prefix
function sampleExact(spec, noise, shots, rng) {
  const pre = prefixSpec(spec, unitaryStepCount(spec));
  const prog = compile(pre, noise);
  const tab = runTableau(prog, { snapshots: false }).final;
  const { frames } = runFrames(prog, shots, { rng, reference: [] });
  const any = new Uint32Array(frames.S), means = {};
  for (let i = 0; i < prog.n; i++) {
    if (!pre.occupancy[i]) continue;
    const a = frames.anticommute(tab.row(i));
    let c = 0;
    for (let s = 0; s < frames.S; s++) { const v = s === frames.S - 1 ? a[s] & frames.lastMask : a[s]; c += popcount(v); any[s] |= a[s]; }
    means[i] = 1 - (2 * c) / shots;
  }
  let bad = 0;
  for (let s = 0; s < frames.S; s++) bad += popcount(s === frames.S - 1 ? any[s] & frames.lastMask : any[s]);
  const vals = Object.values(means);
  const bound = 1 - 0.5 * vals.reduce((acc, m) => acc + (1 - m), 0);
  return { means, fidelity: 1 - bad / shots, bound };
}

// --------------------------------------------------------------- configurations
function parityOf(reg, i) { const c = reg.coords(i); return (c[0] + c[1] + c[2]) & 1; }
function configBases(spec, config, stabRows) {
  const reg = regionOf(spec), [kind, sub] = config.split("-"), cp = sub === "A" ? 0 : 1, out = {};
  for (let i = 0; i < reg.n; i++) {
    if (!spec.occupancy[i]) continue;
    if (parityOf(reg, i) !== cp) out[i] = "Z";
    else if (kind === "centre") out[i] = (stabRows[i] && stabRows[i].ops[i]) || "Z";
    else out[i] = kind[0];
  }
  return out;
}
function measurableSites(stabRows, bases) {
  return Object.keys(stabRows).map(Number).filter((i) =>
    Object.entries(stabRows[i].ops).every(([q, L]) => bases[q] === L)).sort((a, b) => a - b);
}
function siteStabilizers(spec) {
  const pre = prefixSpec(spec, unitaryStepCount(spec));
  const t = runTableau(compile(pre), { snapshots: false }).final, out = {};
  for (let i = 0; i < t.n; i++) if (spec.occupancy[i]) out[i] = t.row(i);
  return out;
}
// sampled <S_i> from shot records in a configuration: {site: [mean, stderr]}
function sampleConfig(spec, config, noise, shots, rng) {
  const pre = prefixSpec(spec, unitaryStepCount(spec));
  const stabs = siteStabilizers(spec), bases = configBases(spec, config, stabs), ok = measurableSites(stabs, bases);
  const order = Object.keys(bases).map(Number).sort((a, b) => a - b);
  const seq = Object.assign({}, pre, { steps: pre.steps.concat([{ op: "measure", bases, sites: order }]) });
  const prog = compile(seq, noise);
  const { meas } = runFrames(prog, shots, { rng });
  const col = {}; meas.forEach((m, k) => { col[m.q] = k; });
  const S = Math.ceil(shots / 32), out = {};
  for (const i of ok) {
    const par = new Uint32Array(S);
    for (const q of Object.keys(stabs[i].ops)) { const b = meas[col[q]].bits; for (let s = 0; s < S; s++) par[s] ^= b[s]; }
    let c = 0;
    for (let s = 0; s < S; s++) c += popcount(s === S - 1 && shots % 32 ? par[s] & (((1 << (shots % 32)) - 1) >>> 0) : par[s]);
    const pm = c / shots, mean = stabs[i].sign * (1 - 2 * pm);
    out[i] = [mean, 2 * Math.sqrt(Math.max(pm * (1 - pm), 1e-12) / shots)];
  }
  return { values: out, bases, measurable: ok };
}

// --------------------------------------------------------------- Pauli propagation
// Propagate a Pauli {q: letter} inserted after step `from` through the noiseless
// unitary ops; returns the Pauli after each later step (measurements stop it).
function propagate(prog, fromStep, pauli) {
  const f = new FrameSim(prog.n, 1, () => 0);
  f.fz.fill(0);
  for (const [q, L] of Object.entries(pauli)) { const [bx, bz] = BITS[LETTER.indexOf(L)]; if (bx) f.fx[+q] = 1; if (bz) f.fz[+q] = 1; }
  const out = [];
  const read = () => { const o = {}; for (let q = 0; q < prog.n; q++) { const c = CODE(f.fx[q] & 1, f.fz[q] & 1); if (c) o[q] = LETTER[c]; } return o; };
  for (let k = fromStep; k < prog.steps.length; k++) {
    const st = prog.steps[k], flipped = [];
    for (let j = st.from; j < st.to; j++) {
      const o = prog.ops[j];
      if (o.t === "g1") f.gate1(o.g, o.q);
      else if (o.t === "cz") f.cz(o.a, o.b);
      else if (o.t === "m") {
        const basisC = { X: [1, 0], Y: [1, 1], Z: [0, 1] }[o.basis];
        const anti = ((f.fx[o.q] & basisC[1]) ^ (f.fz[o.q] & basisC[0])) & 1;
        if (anti) flipped.push(o.q);
        f.fx[o.q] = 0; f.fz[o.q] = 0;     // the measured qubit no longer carries the error
      } else if (o.t === "r") { f.fx[o.q] = 0; f.fz[o.q] = 0; }
    }
    out.push({ pauli: read(), flippedOutcomes: flipped });
  }
  return out;
}

const api = { Tableau, FrameSim, compile, runTableau, runFrames, sampleExact, sampleConfig, siteStabilizers,
  configBases, measurableSites, propagate, regionOf, bondsOf, zzProbability, unitaryStepCount, prefixSpec,
  GATES1, PULSE, RZ, LETTER, IDEAL, popcount };
if (typeof module !== "undefined" && module.exports) module.exports = api;
if (typeof window !== "undefined") window.ClusterEngine = api;
