// Projection-sideband-cooling engine for the explorer: the factorized model of
// lattice_sim.cooling.psc.simulate_factorized, run on the exported tables
// (scripts/psc_export_explorer.py). Works in the browser and in Node.
(function (root) {
  "use strict";
  const H = 6.62607015e-34, KB = 1.380649e-23;

  function decodeU16(b64) {
    let bytes;
    if (typeof atob === "function") {
      const bin = atob(b64);
      bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    } else {
      bytes = new Uint8Array(Buffer.from(b64, "base64"));
    }
    const n = bytes.length / 2, out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = (bytes[2 * i] | (bytes[2 * i + 1] << 8)) / 65535;
    return out;
  }

  class PSC {
    constructor(data) {
      this.d = data;
      this.N = data.N;
      this.B = data.band;
      this.cache = new Map();
      this.heat = {};
      for (const a of data.axes) this.heat[a] = data.heat[a];
    }
    key(om, axis, th, kind, hsr) { return `${om}_${axis}_${Number(th).toFixed(1)}_${kind}_${hsr}`; }
    pulse(om, axis, th, kind, hsr) {
      const k = this.key(om, axis, th, kind, hsr);
      if (!this.cache.has(k)) {
        const raw = this.d.pulses[k];
        if (!raw) throw new Error("no pulse table " + k);
        this.cache.set(k, decodeU16(raw));
      }
      return this.cache.get(k);
    }
    // out_block(j) += M[blk][j, i] * v[i], stored as [blk, d, i] = M[i+d, i]
    applyBlock(m, blk, v, out) {
      const N = this.N, W = 2 * this.B + 1, base = blk * W * N;
      for (let k = 0; k < W; k++) {
        const d = k - this.B, off = base + k * N;
        for (let i = 0; i < N; i++) {
          const j = i + d;
          if (j >= 0 && j < N) out[j] += m[off + i] * v[i];
        }
      }
    }
    // one cooling step on one axis: returns {stay, pumped} for input p (|4,-4> levels)
    step(om, axis, t2, t1, hsr1, p) {
      const N = this.N;
      const m2 = this.pulse(om, axis, t2, 2, -6000), m1 = this.pulse(om, axis, t1, 1, hsr1);
      const vu = new Float64Array(N), vl = new Float64Array(N);
      this.applyBlock(m2, 0, p, vu);       // uu
      this.applyBlock(m2, 2, p, vl);       // lu
      const wu = new Float64Array(N), wl = new Float64Array(N);
      this.applyBlock(m1, 0, vu, wu);
      this.applyBlock(m1, 1, vl, wu);      // ul
      this.applyBlock(m1, 2, vu, wl);
      this.applyBlock(m1, 3, vl, wl);      // ll
      return { stay: wu, pumped: wl };
    }
    heatApply(axis, v) {
      const N = this.N, M = this.heat[axis], out = new Float64Array(N);
      for (let j = 0; j < N; j++) { let s = 0; const row = M[j]; for (let i = 0; i < N; i++) s += row[i] * v[i]; out[j] = s; }
      return out;
    }
    thermal(axis, T_uK) {
      const N = this.N, f = this.d.offsets[axis], out = new Float64Array(N);
      if (T_uK <= 0) { out[0] = 1; return out; }
      const r = Math.exp(-H * f / (KB * T_uK * 1e-6));
      // normalized over the infinite ladder: the tail above N counts as never cooled
      for (let n = 0; n < N; n++) out[n] = (1 - r) * Math.pow(r, n);
      return out;
    }
    // cfg: {omega, T_uK, dark, loops: [{cycles, t2:{x,y,z}, t1:{x,y,z}, hsr1}]}
    simulate(cfg) {
      const axes = this.d.axes, dark = cfg.dark || 0;
      const p = {};
      for (const a of axes) p[a] = this.thermal(a, cfg.T_uK);
      const hist = [cfg.light ? { g3: this.d.axes.reduce((g, a) => g * p[a][0], 1) } : this.snapshot(p)];
      let opEvents = 0;
      for (const loop of cfg.loops) {
        for (let c = 0; c < loop.cycles; c++) {
          for (const a of axes) {
            const { stay, pumped } = this.step(cfg.omega, a, loop.t2[a], loop.t1[a], loop.hsr1, p[a]);
            let frac = 0, sStay = 0;
            for (let i = 0; i < this.N; i++) { frac += pumped[i]; sStay += stay[i]; }
            const leak = dark * sStay;
            opEvents += frac + leak;
            const hp = this.heatApply(a, pumped), hs = dark > 0 ? this.heatApply(a, stay) : null;
            const nw = new Float64Array(this.N);
            for (let i = 0; i < this.N; i++) nw[i] = (1 - dark) * stay[i] + hp[i] + (hs ? dark * hs[i] : 0);
            for (const b of axes) {
              if (b === a) continue;
              const hb = this.heatApply(b, p[b]);
              for (let i = 0; i < this.N; i++) p[b][i] = (1 - frac - leak) * p[b][i] + (frac + leak) * hb[i];
            }
            p[a] = nw;
          }
          hist.push(cfg.light ? { g3: this.d.axes.reduce((g, a) => g * p[a][0], 1) } : this.snapshot(p));
        }
      }
      return { hist, final: p, opEvents };
    }
    snapshot(p) {
      const s = { ground: {}, nbar: {}, dist: {} };
      let g = 1;
      for (const a of this.d.axes) {
        let nb = 0;
        for (let i = 0; i < this.N; i++) nb += i * p[a][i];
        s.ground[a] = p[a][0]; s.nbar[a] = nb; g *= p[a][0];
        s.dist[a] = Array.from(p[a]);
      }
      s.g3 = g;
      return s;
    }
    // per-level outcome of one step: clean (n -> n-1 without OP), pumped, stuck
    outcomes(om, axis, t2, t1, hsr1) {
      const N = this.N, res = { clean: [], pumped: [], stuck: [], down2: [] };
      for (let n = 0; n < N; n++) {
        const p = new Float64Array(N); p[n] = 1;
        const { stay, pumped } = this.step(om, axis, t2, t1, hsr1, p);
        let s = 0; for (let i = 0; i < N; i++) s += pumped[i];
        res.clean.push(n > 0 ? stay[n - 1] : 0);
        res.stuck.push(stay[n]);
        res.pumped.push(s);
      }
      return res;
    }
    // single-pulse transfer probabilities per level
    pulseTransfer(om, axis, th, kind, hsr) {
      const N = this.N, W = 2 * this.B + 1, m = this.pulse(om, axis, th, kind, hsr);
      const fwd = [], back = [];
      for (let n = 0; n < N; n++) {
        // |4,-4,n> -> |3,-3,n-kind| : block lu (2), d = -kind
        const k = this.B - kind;
        fwd.push(n - kind >= 0 ? m[2 * W * N + k * N + n] : 0);
        // |3,-3,n> -> |4,-4,n+kind| : block ul (1), d = +kind
        const k2 = this.B + kind;
        back.push(n + kind < N ? m[1 * W * N + k2 * N + n] : 0);
      }
      return { fwd, back };
    }
  }
  root.PSC = PSC;
  if (typeof module !== "undefined") module.exports = { PSC };
})(typeof window !== "undefined" ? window : globalThis);
