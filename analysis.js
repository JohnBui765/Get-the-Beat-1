/* Echo Stems analysis: the measurements behind the mixer's honest levels.

   - Loudness to ITU-R BS.1770-4 (K-weighting, 400 ms blocks with 75% overlap, gating at -70 LUFS and -10 LU).
     Because every fader and pan setting is a fixed linear mix of the six layers, the loudness of ANY setting can be
     read from one set of numbers measured once per song: the K-weighted cross-products of the twelve layer channels
     in each 100 ms stretch (a "Gram matrix"). That makes level match exact and instant.
   - Exact peaks of a mix: each 256-sample block gets an upper bound from the layers' own peaks; only blocks whose
     bound could beat the loudest sample found so far are mixed sample by sample.
   - Reference levels for the gentle clean-up in Solo.

   Loaded by the engine worker (importScripts), the page (<script>) and the Node tests. */
(function (root) {
  'use strict';

  /* ---------- K-weighting (Brecht De Man's derivation, which reproduces the ITU coefficients at 48 kHz) ---------- */
  function kCoefs(fs) {
    let K = Math.tan(Math.PI * 1681.9744509555319 / fs);
    const Q1 = 0.7071752369554193, Vh = Math.pow(10, 3.99984385397 / 20), Vb = Math.pow(Vh, 0.499666774155);
    let a0 = 1 + K / Q1 + K * K;
    const shelf = [(Vh + Vb * K / Q1 + K * K) / a0, 2 * (K * K - Vh) / a0, (Vh - Vb * K / Q1 + K * K) / a0, 2 * (K * K - 1) / a0, (1 - K / Q1 + K * K) / a0];
    K = Math.tan(Math.PI * 38.13547087613982 / fs);
    const Q2 = 0.5003270373253953;
    a0 = 1 + K / Q2 + K * K;
    const hp = [1, -2, 1, 2 * (K * K - 1) / a0, (1 - K / Q2 + K * K) / a0];
    return [shelf, hp];              // each [b0, b1, b2, a1, a2], a0 = 1
  }

  // A running two-stage K filter (direct form I); state lives in the object so a long signal can be fed in pieces.
  function makeK(fs) {
    const [s, h] = kCoefs(fs);
    return { s, h, x1: 0, x2: 0, y1: 0, y2: 0, u1: 0, u2: 0, z1: 0, z2: 0 };
  }
  // Filters src[from..from+len) into dst[0..len).
  function runK(f, src, from, len, dst) {
    const s = f.s, h = f.h;
    let x1 = f.x1, x2 = f.x2, y1 = f.y1, y2 = f.y2, z1 = f.z1, z2 = f.z2;
    const sb0 = s[0], sb1 = s[1], sb2 = s[2], sa1 = s[3], sa2 = s[4];
    const ha1 = h[3], ha2 = h[4];
    for (let i = 0; i < len; i++) {
      const x = src[from + i];
      const y = sb0 * x + sb1 * x1 + sb2 * x2 - sa1 * y1 - sa2 * y2;
      x2 = x1; x1 = x;
      const z = y - 2 * y1 + y2 - ha1 * z1 - ha2 * z2;   // high-pass numerator is [1, -2, 1] applied to y
      y2 = y1; y1 = y;
      z2 = z1; z1 = z;
      dst[i] = z;
    }
    f.x1 = x1; f.x2 = x2; f.y1 = y1; f.y2 = y2; f.z1 = z1; f.z2 = z2;
  }

  const subLen = (fs) => Math.round(fs * 0.1);           // 100 ms: a quarter of a 400 ms gating block

  // Integrated loudness from the mean squares of each 400 ms block (sum over channels, all channel weights 1).
  function gate(z) {
    const n = z.length;
    let sum = 0, cnt = 0;
    for (let j = 0; j < n; j++) if (z[j] > 0 && -0.691 + 10 * Math.log10(z[j]) > -70) { sum += z[j]; cnt++; }
    if (!cnt) return -Infinity;
    const rel = -0.691 + 10 * Math.log10(sum / cnt) - 10;
    sum = 0; cnt = 0;
    for (let j = 0; j < n; j++) {
      if (!(z[j] > 0)) continue;
      const l = -0.691 + 10 * Math.log10(z[j]);
      if (l > -70 && l > rel) { sum += z[j]; cnt++; }
    }
    return cnt ? -0.691 + 10 * Math.log10(sum / cnt) : -Infinity;
  }

  // Loudness of a stereo signal measured directly (used by the tests to check the Gram method).
  function loudnessDirect(L, R, fs) {
    const sb = subLen(fs), nSub = Math.floor(L.length / sb);
    if (nSub < 4) return -Infinity;
    const e = new Float64Array(nSub), tmp = new Float64Array(sb);
    for (const ch of [L, R]) {
      const f = makeK(fs);
      for (let j = 0; j < nSub; j++) {
        runK(f, ch, j * sb, sb, tmp);
        let s = 0; for (let i = 0; i < sb; i++) s += tmp[i] * tmp[i];
        e[j] += s;
      }
    }
    const z = new Float64Array(nSub - 3);
    for (let j = 0; j < z.length; j++) z[j] = (e[j] + e[j + 1] + e[j + 2] + e[j + 3]) / (4 * sb);
    return gate(z);
  }

  /* ---------- Gram matrices ---------- */
  // Pairs (a, b) with a <= b, in a fixed order; NP = n(n+1)/2.
  function pairs(n) {
    const A = [], B = [];
    for (let a = 0; a < n; a++) for (let b = a; b < n; b++) { A.push(a); B.push(b); }
    return { A: Int32Array.from(A), B: Int32Array.from(B), NP: A.length };
  }

  /* K-weighted cross-products of the given signals (twelve: L and R of each layer) per 100 ms stretch.
     Returns { sub: Float64Array(nSub * NP), nSub, nSig, sb }. onProgress(fraction) is called now and then. */
  function gramSub(sigs, fs, onProgress) {
    const nSig = sigs.length, N = sigs[0].length, sb = subLen(fs), nSub = Math.floor(N / sb);
    const { A, B, NP } = pairs(nSig);
    const sub = new Float64Array(Math.max(0, nSub) * NP);
    const filters = sigs.map(() => makeK(fs));
    const tmp = sigs.map(() => new Float64Array(sb));
    for (let j = 0; j < nSub; j++) {
      for (let a = 0; a < nSig; a++) runK(filters[a], sigs[a], j * sb, sb, tmp[a]);
      const o = j * NP;
      for (let p = 0; p < NP; p++) {
        const x = tmp[A[p]], y = tmp[B[p]];
        let s = 0;
        for (let i = 0; i < sb; i++) s += x[i] * y[i];
        sub[o + p] = s;
      }
      if (onProgress && (j & 63) === 0) onProgress(j / nSub);
    }
    return { sub, nSub, nSig, sb };
  }

  // Sums four consecutive 100 ms stretches into each 400 ms gating block.
  function blockGrams(g) {
    const NP = (g.nSig * (g.nSig + 1)) / 2, nb = Math.max(0, g.nSub - 3);
    const out = new Float64Array(nb * NP);
    for (let j = 0; j < nb; j++) {
      const o = j * NP;
      for (let q = 0; q < 4; q++) { const s = (j + q) * NP; for (let p = 0; p < NP; p++) out[o + p] += g.sub[s + p]; }
    }
    return { blocks: out, nb, NP, nSig: g.nSig, sb: g.sb };
  }

  // Integrated loudness of the mix whose left output is sum(cL[a] * signal a) and right output sum(cR[a] * signal a).
  function loudnessFromGram(bg, cL, cR) {
    const { blocks, nb, NP, nSig, sb } = bg;
    if (!nb) return -Infinity;
    const { A, B } = pairs(nSig);
    const w = new Float64Array(NP);
    for (let p = 0; p < NP; p++) {
      const a = A[p], b = B[p], m = a === b ? 1 : 2;
      w[p] = m * (cL[a] * cL[b] + cR[a] * cR[b]);
    }
    const z = new Float64Array(nb), norm = 1 / (4 * sb);
    for (let j = 0; j < nb; j++) {
      let s = 0;
      const o = j * NP;
      for (let p = 0; p < NP; p++) s += w[p] * blocks[o + p];
      z[j] = s * norm;
    }
    return gate(z);
  }

  // Whole-song K-weighted totals (sum over every 100 ms stretch), used for the pan law's correlation.
  function gramTotals(g) {
    const NP = (g.nSig * (g.nSig + 1)) / 2, tot = new Float64Array(NP);
    for (let j = 0; j < g.nSub; j++) { const o = j * NP; for (let p = 0; p < NP; p++) tot[p] += g.sub[o + p]; }
    return tot;
  }
  function pairIdx(nSig, a, b) {           // position of pair (a, b), a <= b, in the fixed order
    if (a > b) { const t = a; a = b; b = t; }
    return a * nSig - (a * (a - 1)) / 2 + (b - a);
  }

  /* ---------- peaks ---------- */
  const PEAK_BLOCK = 64;                    // samples per block for the peak search (1.5 ms)
  function blockMaxima(L, R, B) {
    B = B || PEAK_BLOCK;
    const n = L.length, nb = Math.ceil(n / B), mL = new Float32Array(nb), mR = new Float32Array(nb);
    for (let b = 0; b < nb; b++) {
      const e = Math.min(n, (b + 1) * B);
      let x = 0, y = 0;
      for (let i = b * B; i < e; i++) {
        const l = L[i] < 0 ? -L[i] : L[i], r = R[i] < 0 ? -R[i] : R[i];
        if (l > x) x = l;
        if (r > y) y = r;
      }
      mL[b] = x; mR[b] = y;
    }
    return { L: mL, R: mR, B, nb };
  }

  // How far the original strays from the sum of the layers in each block (about 1e-7 for this app's layers, which are
  // made to add up to the song). It keeps the second bound below rigorous.
  function residualMaxima(orig, layers, B) {
    B = B || PEAK_BLOCK;
    const n = orig[0].length, nb = Math.ceil(n / B), m = new Float32Array(nb);
    for (let b = 0; b < nb; b++) {
      const e = Math.min(n, (b + 1) * B);
      let x = 0;
      for (let t = b * B; t < e; t++) {
        let l = orig[0][t], r = orig[1][t];
        for (let i = 0; i < layers.length; i++) { l -= layers[i][0][t]; r -= layers[i][1][t]; }
        l = Math.abs(l); r = Math.abs(r);
        if (l > x) x = l;
        if (r > x) x = r;
      }
      m[b] = x;
    }
    return m;
  }

  /* Exact loudest sample of: out = orig + sum_i D_i * layer_i, where D_i = [d00, d01, d10, d11] (row-major 2x2)
     takes layer i's (left, right) to the change it makes in (left, right).
     src = { orig: [L, R], layers: [[L, R] x6], bmOrig, bmLayers, resMax }. D = Float64Array(24).
     Each block gets the smaller of two upper bounds:
       change form:  |orig| + sum |D_i| * |layer_i|                (tight when little changes)
       layer form:   |orig - sum layers| + sum |D_i + I| * |layer_i|  (tight for solos and big cuts)
     The async version pauses now and then so the page stays smooth; isStale() lets a newer request abandon it. */
  function boundsFor(src, D) {
    const nb = src.bmOrig.nb, ub = new Float64Array(nb), nl = src.layers.length;
    const act = [];
    for (let i = 0; i < nl; i++) {
      const o = i * 4;
      if (D[o] || D[o + 1] || D[o + 2] || D[o + 3]) act.push(i);
    }
    const oL = src.bmOrig.L, oR = src.bmOrig.R, res = src.resMax;
    const P = new Float64Array(nl * 4);
    for (let i = 0; i < nl; i++) {
      P[4 * i] = Math.abs(D[4 * i] + 1); P[4 * i + 1] = Math.abs(D[4 * i + 1]);
      P[4 * i + 2] = Math.abs(D[4 * i + 2]); P[4 * i + 3] = Math.abs(D[4 * i + 3] + 1);
    }
    for (let b = 0; b < nb; b++) {
      let uL = oL[b], uR = oR[b];
      for (const i of act) {
        const o = i * 4, mL = src.bmLayers[i].L[b], mR = src.bmLayers[i].R[b];
        uL += Math.abs(D[o]) * mL + Math.abs(D[o + 1]) * mR;
        uR += Math.abs(D[o + 2]) * mL + Math.abs(D[o + 3]) * mR;
      }
      let u = uL > uR ? uL : uR;
      if (res) {
        let pL = res[b], pR = res[b];
        for (let i = 0; i < nl; i++) {
          const o = i * 4, mL = src.bmLayers[i].L[b], mR = src.bmLayers[i].R[b];
          pL += P[o] * mL + P[o + 1] * mR;
          pR += P[o + 2] * mL + P[o + 3] * mR;
        }
        const p = pL > pR ? pL : pR;
        if (p < u) u = p;
      }
      ub[b] = u;
    }
    return { ub, act };
  }
  function blockPeak(src, D, act, b) {
    const B = src.bmOrig.B, n = src.orig[0].length, s = b * B, e = Math.min(n, s + B);
    const oL = src.orig[0], oR = src.orig[1];
    let m = 0;
    for (let t = s; t < e; t++) {
      let l = oL[t], r = oR[t];
      for (let k = 0; k < act.length; k++) {
        const i = act[k], o = i * 4, x = src.layers[i][0][t], y = src.layers[i][1][t];
        l += D[o] * x + D[o + 1] * y;
        r += D[o + 2] * x + D[o + 3] * y;
      }
      if (l < 0) l = -l;
      if (r < 0) r = -r;
      if (l > m) m = l;
      if (r > m) m = r;
    }
    return m;
  }
  // Mixes the block with the highest bound first; then only blocks whose bound beats that exact peak are candidates,
  // and only those are sorted (usually a handful).
  function* peakSearch(src, D) {
    const { ub, act } = boundsFor(src, D);
    const nb = ub.length;
    if (!nb) return { peak: 0, bound: 0, evaluated: 0, blocks: 0 };
    let top = 0;
    for (let b = 1; b < nb; b++) if (ub[b] > ub[top]) top = b;
    let best = blockPeak(src, D, act, top), evaluated = 1;
    const cand = [];
    for (let b = 0; b < nb; b++) if (ub[b] > best && b !== top) cand.push(b);
    cand.sort((a, b) => ub[b] - ub[a]);
    for (let k = 0; k < cand.length; k++) {
      const b = cand[k];
      if (ub[b] <= best) break;
      const m = blockPeak(src, D, act, b);
      evaluated++;
      if (m > best) best = m;
      if ((evaluated & 63) === 0) yield;
    }
    return { peak: best, bound: ub[top], evaluated, blocks: nb };
  }
  function exactPeak(src, D) {
    const it = peakSearch(src, D);
    for (;;) { const r = it.next(); if (r.done) return r.value; }
  }
  async function exactPeakAsync(src, D, isStale, budgetMs) {
    const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
    const budget = budgetMs || 8;
    const it = peakSearch(src, D);
    let t0 = now();
    for (;;) {
      const r = it.next();
      if (r.done) return r.value;
      if (now() - t0 > budget) {
        await new Promise((res) => setTimeout(res, 0));
        if (isStale && isStale()) return null;
        t0 = now();
      }
    }
  }

  /* All the tables the peak search needs, in one pass: each signal's loudest sample per block, and how far the
     original strays from the sum of the layers. Fill blocks [b0, b1) at a time, so a page can do it in slices. */
  function peakTablesInit(n, nl, B) {
    B = B || PEAK_BLOCK;
    const nb = Math.ceil(n / B), mk = () => ({ L: new Float32Array(nb), R: new Float32Array(nb), B, nb });
    const layers = [];
    for (let i = 0; i < nl; i++) layers.push(mk());
    return { B, nb, n, bmOrig: mk(), bmLayers: layers, resMax: new Float32Array(nb) };
  }
  function peakTablesFill(T, orig, layers, b0, b1) {
    const B = T.B, n = T.n, nl = layers.length, oL = orig[0], oR = orig[1];
    for (let b = b0; b < b1; b++) {
      const s = b * B, e = Math.min(n, s + B);
      for (let i = 0; i < nl; i++) {
        const L = layers[i][0], R = layers[i][1];
        let x = 0, y = 0;
        for (let t = s; t < e; t++) {
          const l = L[t] < 0 ? -L[t] : L[t], r = R[t] < 0 ? -R[t] : R[t];
          if (l > x) x = l;
          if (r > y) y = r;
        }
        T.bmLayers[i].L[b] = x; T.bmLayers[i].R[b] = y;
      }
      let x = 0, y = 0, z = 0;
      for (let t = s; t < e; t++) {
        let l = oL[t], r = oR[t];
        const al = l < 0 ? -l : l, ar = r < 0 ? -r : r;
        if (al > x) x = al;
        if (ar > y) y = ar;
        for (let i = 0; i < nl; i++) { l -= layers[i][0][t]; r -= layers[i][1][t]; }
        if (l < 0) l = -l;
        if (r < 0) r = -r;
        if (l > z) z = l;
        if (r > z) z = r;
      }
      T.bmOrig.L[b] = x; T.bmOrig.R[b] = y; T.resMax[b] = z;
    }
  }
  function peakTables(orig, layers, B) {
    const T = peakTablesInit(orig[0].length, layers.length, B);
    peakTablesFill(T, orig, layers, 0, T.nb);
    return T;
  }
  async function peakTablesAsync(orig, layers, B, budgetMs) {
    const T = peakTablesInit(orig[0].length, layers.length, B), step = 4096;
    for (let b = 0; b < T.nb; b += step) {
      peakTablesFill(T, orig, layers, b, Math.min(T.nb, b + step));
      await new Promise((r) => setTimeout(r, 0));
    }
    return T;
  }

  /* ---------- clean-up reference ---------- */
  // A layer's "playing" level: the 90th percentile of its 50 ms peaks, counting only stretches that aren't silent.
  function playingLevel(L, R, fs) {
    const B = Math.round(fs * 0.05), n = L.length, nb = Math.floor(n / B);
    const p = [];
    for (let b = 0; b < nb; b++) {
      let m = 0;
      for (let i = b * B, e = i + B; i < e; i++) { const l = Math.abs(L[i]), r = Math.abs(R[i]); if (l > m) m = l; if (r > m) m = r; }
      if (m > 1e-5) p.push(m);
    }
    if (!p.length) return 0;
    p.sort((a, b) => a - b);
    return p[Math.min(p.length - 1, Math.floor(p.length * 0.9))];
  }

  /* ---------- fingerprints of saved layers (FNV-1a over the 24-bit sample values) ---------- */
  function hashStart() { return 0x811c9dc5 | 0; }
  function hashInts(h, a, n) {
    for (let i = 0; i < n; i++) {
      const v = a[i];
      h = Math.imul(h ^ (v & 255), 16777619);
      h = Math.imul(h ^ ((v >> 8) & 255), 16777619);
      h = Math.imul(h ^ ((v >> 16) & 255), 16777619);
    }
    return h;
  }
  // The same fingerprint from decoded sound: v = round(x * 2^23 / scale), which is exact for 24-bit values.
  function* hashSteps(L, R, scale, out) {
    const k = 8388608 / scale, B = 65536, buf = new Int32Array(B * 2);
    let h = hashStart();
    for (let i = 0; i < L.length; i += B) {
      const m = Math.min(B, L.length - i);
      for (let j = 0; j < m; j++) {
        let a = Math.round(L[i + j] * k), b = Math.round(R[i + j] * k);
        a = a > 8388607 ? 8388607 : a < -8388608 ? -8388608 : a;
        b = b > 8388607 ? 8388607 : b < -8388608 ? -8388608 : b;
        buf[2 * j] = a; buf[2 * j + 1] = b;
      }
      h = hashInts(h, buf, 2 * m);
      yield;
    }
    out.h = h >>> 0;
  }
  function hashFloats(L, R, scale) { const out = {}; for (const _ of hashSteps(L, R, scale, out)) { /* run */ } return out.h; }
  async function hashFloatsAsync(L, R, scale, budgetMs) {
    const out = {}, budget = budgetMs || 12;
    let t0 = performance.now();
    for (const _ of hashSteps(L, R, scale, out)) {
      if (performance.now() - t0 > budget) { await new Promise((r) => setTimeout(r, 0)); t0 = performance.now(); }
    }
    return out.h;
  }

  const api = { kCoefs, makeK, runK, gate, loudnessDirect, pairs, pairIdx, gramSub, blockGrams, loudnessFromGram, gramTotals, blockMaxima, residualMaxima, peakTables, peakTablesAsync, exactPeak, exactPeakAsync, playingLevel, hashStart, hashInts, hashFloats, hashFloatsAsync, PEAK_BLOCK, subLen };
  root.ESAnalysis = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : self);
