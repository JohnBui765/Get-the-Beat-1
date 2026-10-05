/* Echo Stems mixing engine.

   Everything you hear is the ORIGINAL recording plus, for each layer you changed, the difference you asked for:
       out = master * ( original + sum over layers of ( P_i * layer_i  -  layer_i ) )
   where P_i is a 2x2 matrix holding that layer's fader gain and left-right position. Because the six layers add up to
   the original exactly, this is the same as playing the six layers at their new settings, but when a fader sits at
   0 dB its layer contributes exactly nothing, so with every fader at 0 dB you hear the original bit for bit.

   All sums are done in 64-bit floating point and written out once as 32-bit. Fader and position changes glide over
   about 30 ms, play, pause and jumps fade over 5 ms, and repeating the song crossfades its end into its start, so
   nothing ever clicks. No limiter or compressor touches the sound: the page lowers the master level only as far as
   the peaks need (see analysis.js).

   This one file is both the AudioWorklet (registered as "es-mixer") and the maths the page uses to save exactly what
   you hear; the tests load it in Node. */
(function (root) {
  'use strict';
  const NL = 6;
  const SR = 44100;

  function dbToGain(db) { return db === -Infinity || db <= -120 ? 0 : Math.pow(10, db / 20); }

  // Left-right position p in [-1, 1] as a 2x2 matrix (row-major, from (left, right) in to (left, right) out).
  // Moving left folds some of the right channel into the left and quietens the right; centre is the identity.
  function panMatrix(p) {
    if (!p) return [1, 0, 0, 1];
    const th = Math.abs(p) * Math.PI / 2, s = Math.sin(th), c = Math.cos(th);
    return p < 0 ? [1, s, 0, c] : [c, 0, s, 1];
  }
  // Folding one channel into the other makes a layer louder when its channels are alike (rho near 1). This factor
  // undoes that, so a layer keeps its loudness as it moves: rho = 2 * cross / (left energy + right energy), measured
  // K-weighted over the whole song.
  function panKeep(p, rho) {
    if (!p) return 1;
    const s = Math.sin(Math.abs(p) * Math.PI / 2);
    return 1 / Math.sqrt(Math.max(0.25, 1 + s * (rho || 0)));
  }

  /* From the mixer's settings to the six play matrices.
     mix = { gain: dB[6] (-Infinity allowed), mute: bool[6], solo: bool[6], pan: [-1..1][6], bedDb }.
     When any layer is soloed, the other layers play as "room tone": their own setting lowered by bedDb. */
  function playMatrices(mix, rho) {
    const P = new Float64Array(NL * 4);
    const anySolo = mix.solo.some(Boolean), bed = dbToGain(mix.bedDb);
    for (let i = 0; i < NL; i++) {
      const base = mix.mute[i] ? 0 : dbToGain(mix.gain[i]);
      const g = anySolo ? (mix.solo[i] ? base : base * bed) : base;
      const A = panMatrix(mix.pan[i]), k = g * panKeep(mix.pan[i], rho ? rho[i] : 0);
      for (let j = 0; j < 4; j++) P[4 * i + j] = k * A[j];
    }
    return P;
  }
  function isIdentity(P, i) { const o = 4 * i; return P[o] === 1 && P[o + 1] === 0 && P[o + 2] === 0 && P[o + 3] === 1; }
  function allIdentity(P) { for (let i = 0; i < NL; i++) if (!isIdentity(P, i)) return false; return true; }
  // The change each layer makes: D_i = P_i - I (for the peak search).
  function changeMatrices(P) { const D = Float64Array.from(P); for (let i = 0; i < NL; i++) { D[4 * i] -= 1; D[4 * i + 3] -= 1; } return D; }
  // Loudness coefficients over the twelve layer channels (L0, R0, L1, R1, ...).
  function loudnessVectors(P) {
    const cL = new Float64Array(2 * NL), cR = new Float64Array(2 * NL);
    for (let i = 0; i < NL; i++) { cL[2 * i] = P[4 * i]; cL[2 * i + 1] = P[4 * i + 1]; cR[2 * i] = P[4 * i + 2]; cR[2 * i + 1] = P[4 * i + 3]; }
    return { cL, cR };
  }

  /* Gentle clean-up for soloed layers: when a layer drops well below its own playing level (stray sound from other
     instruments between its phrases), it is lowered gradually, by at most 15 dB, and opens again within a millisecond
     when the instrument returns. It lowers; it never cuts. */
  const EXP = { below: 30, range: 15, hold: 0.05, close: 0.08, open: 0.001, release: 0.1 };
  function expanderParams(levels) {
    const floor = Math.pow(10, -EXP.range / 20);
    return {
      thr: Float64Array.from(levels, (v) => (v > 0 ? v * Math.pow(10, -EXP.below / 20) : 0)),
      floor,
      rel: Math.exp(-1 / (EXP.release * SR)),
      openC: 1 - Math.exp(-1 / (EXP.open * SR)),
      closeC: 1 - Math.exp(-1 / (EXP.close * SR)),
      holdN: Math.round(EXP.hold * SR),
    };
  }
  function expanderState() { return { env: new Float64Array(NL), gain: new Float64Array(NL).fill(1), hold: new Int32Array(NL) }; }
  // Advances layer i's clean-up by one sample with input (x, y); returns its gain.
  function expanderStep(prm, st, i, x, y) {
    const a = x < 0 ? -x : x, b = y < 0 ? -y : y, m = a > b ? a : b;
    let env = st.env[i];
    env = m > env ? m : env * prm.rel;
    st.env[i] = env;
    const thr = prm.thr[i];
    let want = thr > 0 ? env / thr : 1;
    if (want > 1) want = 1;
    if (want < prm.floor) want = prm.floor;
    let g = st.gain[i];
    if (want >= g) { g += (want - g) * prm.openC; st.hold[i] = prm.holdN; }
    else if (st.hold[i] > 0) st.hold[i]--;
    else g += (want - g) * prm.closeC;
    st.gain[i] = g;
    return g;
  }

  /* Mixes frames [from, to) of src = { oL, oR, lay: [[L, R] x6] } with fixed play matrices P and master gain into
     outL / outR (from index 0). expand = { on: bool[6], prm, st } or null. Used to save exactly what you hear. */
  function renderMix(src, P, master, expand, from, to, outL, outR) {
    const act = [];
    for (let i = 0; i < NL; i++) if (!isIdentity(P, i) || (expand && expand.on[i])) act.push(i);
    const oL = src.oL, oR = src.oR;
    for (let t = from, k = 0; t < to; t++, k++) {
      let l = oL[t], r = oR[t];
      for (let a = 0; a < act.length; a++) {
        const i = act[a], o = 4 * i, x = src.lay[i][0][t], y = src.lay[i][1][t];
        const e = expand && expand.on[i] ? expanderStep(expand.prm, expand.st, i, x, y) : 1;
        l += e * (P[o] * x + P[o + 1] * y) - x;
        r += e * (P[o + 2] * x + P[o + 3] * y) - y;
      }
      outL[k] = master * l;
      outR[k] = master * r;
    }
  }

  const api = { NL, SR, dbToGain, panMatrix, panKeep, playMatrices, isIdentity, allIdentity, changeMatrices, loudnessVectors, EXP, expanderParams, expanderState, expanderStep, renderMix };
  root.ESMix = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;

  /* ================= the AudioWorklet ================= */
  if (typeof registerProcessor !== 'function' || typeof AudioWorkletProcessor === 'undefined') return;

  const FADE = Math.round(0.005 * SR);          // 5 ms: play, pause, jumps, the end of the song, repeat crossfade
  const GLIDE = Math.round(0.03 * SR);          // 30 ms: fader, position and master changes
  const SWITCH = Math.round(0.01 * SR);         // 10 ms: switching between two versions (A/B)

  class Mixer extends AudioWorkletProcessor {
    constructor() {
      super();
      this.sets = new Map();
      this.active = null;
      this.pos = 0;
      this.playing = false;
      this.repeat = true;
      this.tg = 0; this.tTarget = 0; this.tStep = 0; this.tLeft = 0; this.after = null;   // transport fade
      this.P = new Float64Array(NL * 4); for (let i = 0; i < NL; i++) { this.P[4 * i] = 1; this.P[4 * i + 3] = 1; }
      this.Pt = Float64Array.from(this.P); this.Pi = new Float64Array(NL * 4); this.pLeft = 0;
      this.m = 1; this.mt = 1; this.mi = 0; this.mLeft = 0;
      this.skip = new Uint8Array(NL).fill(1);
      this.expOn = new Uint8Array(NL); this.prm = null; this.st = expanderState(); this.expAny = false;
      this.xf = null;                             // { from: set, k } while switching versions
      this.q = 0;
      this.port.onmessage = (e) => this.onMsg(e.data || {});
    }

    onMsg(m) {
      switch (m.type) {
        case 'load': {
          this.sets.set(m.id, { n: m.n, oL: m.oL, oR: m.oR, lay: m.lay });
          if (this.active === null) this.active = m.id;
          this.port.postMessage({ type: 'loaded', id: m.id });
          break;
        }
        // the same, in pieces (the original, then each layer), so the page never stalls handing over a long song
        case 'load-begin': { this.pending = this.pending || new Map(); this.pending.set(m.id, { n: m.n, parts: [] }); break; }
        case 'load-part': { const p = this.pending && this.pending.get(m.id); if (p) p.parts[m.k] = [m.L, m.R]; break; }
        case 'load-end': {
          const p = this.pending && this.pending.get(m.id);
          if (!p) break;
          this.pending.delete(m.id);
          this.sets.set(m.id, { n: p.n, oL: p.parts[0][0], oR: p.parts[0][1], lay: p.parts.slice(1) });
          if (this.active === null) this.active = m.id;
          this.port.postMessage({ type: 'loaded', id: m.id });
          break;
        }
        case 'drop': {
          this.sets.delete(m.id);
          if (this.active === m.id) { this.active = null; this.playing = false; this.tg = 0; this.tLeft = 0; this.pos = 0; this.xf = null; }
          break;
        }
        case 'use': {
          const to = this.sets.get(m.id);
          if (!to || m.id === this.active) break;
          const from = this.sets.get(this.active);
          this.active = m.id;
          if (this.pos >= to.n) this.pos = 0;
          this.xf = this.playing && from ? { from, k: 0 } : null;
          break;
        }
        case 'coefs': {
          const ramp = m.ramp === 0 ? 0 : (m.ramp || GLIDE);
          if (m.P) {
            this.Pt.set(m.P);
            if (ramp === 0) { this.P.set(m.P); this.pLeft = 0; }
            else { for (let j = 0; j < this.P.length; j++) this.Pi[j] = (this.Pt[j] - this.P[j]) / ramp; this.pLeft = ramp; }
          }
          if (typeof m.master === 'number') {
            const mr = typeof m.masterRamp === 'number' ? m.masterRamp : ramp;
            this.mt = m.master;
            if (mr === 0) { this.m = m.master; this.mLeft = 0; }
            else { this.mi = (this.mt - this.m) / mr; this.mLeft = mr; }
          }
          this.updateSkip();
          break;
        }
        case 'expander': {
          this.prm = m.levels ? expanderParams(m.levels) : this.prm;
          const was = this.expOn.slice();
          for (let i = 0; i < NL; i++) this.expOn[i] = m.on && m.on[i] ? 1 : 0;
          for (let i = 0; i < NL; i++) if (this.expOn[i] && !was[i]) { this.st.gain[i] = 1; this.st.env[i] = 0; this.st.hold[i] = 0; }
          this.expAny = !!this.prm && this.expOn.some(Boolean);
          this.updateSkip();
          break;
        }
        case 'play': {
          const set = this.sets.get(this.active);
          if (!set) break;
          if (typeof m.frame === 'number') this.pos = Math.max(0, Math.min(set.n - 1, m.frame | 0));
          if (this.pos >= set.n - 1) this.pos = 0;
          if (!this.playing) { this.playing = true; this.tg = 0; }
          this.after = null;
          this.fadeTo(1, FADE);
          this.report(true);
          break;
        }
        case 'pause': {
          if (!this.playing) break;
          const a = this.after;
          if (a && (a.pause || a.end)) break;                       // already stopping
          this.fadeTo(0, FADE, { pause: true, seekTo: a && typeof a.seek === 'number' ? a.seek : undefined });
          break;
        }
        case 'seek': {
          const set = this.sets.get(this.active);
          if (!set) break;
          const f = Math.max(0, Math.min(set.n - 1, m.frame | 0));
          const a = this.after;
          if (!this.playing) { this.pos = f; this.report(true); }
          else if (a && (a.pause || a.end)) { a.seekTo = f; a.end = false; a.pause = true; }   // stopping: land there
          else this.fadeTo(0, FADE, { seek: f });
          break;
        }
        case 'repeat': this.repeat = !!m.on; break;
        case 'where': this.report(true); break;
      }
    }

    updateSkip() {
      for (let i = 0; i < NL; i++) {
        const o = 4 * i, P = this.P, T = this.Pt;
        const still = this.pLeft === 0 || (T[o] === P[o] && T[o + 1] === P[o + 1] && T[o + 2] === P[o + 2] && T[o + 3] === P[o + 3]);
        this.skip[i] = still && isIdentity(P, i) && !(this.expAny && this.expOn[i]) ? 1 : 0;
      }
    }

    fadeTo(target, len, after) {
      this.tTarget = target;
      this.tLeft = len;
      this.tStep = (target - this.tg) / len;
      this.after = after || null;
    }

    fadeDone() {
      const a = this.after;
      this.after = null;
      if (!a) return;
      if (a.pause || a.end) {
        this.playing = false;
        if (a.end) { this.pos = 0; this.port.postMessage({ type: 'ended' }); }
        if (typeof a.seekTo === 'number') this.pos = a.seekTo;
        this.report(true);
      } else if (typeof a.seek === 'number') {
        this.pos = a.seek;
        this.fadeTo(1, FADE);
        this.report(true);
      }
    }

    report(force) {
      if (force || (++this.q & 7) === 0) this.port.postMessage({ type: 'pos', frame: this.pos, time: currentTime, playing: this.playing });
    }

    // One mixed sample of `set` at frame t into this.l / this.r (gains applied by the caller).
    mixAt(set, t, expand) {
      let l = set.oL[t], r = set.oR[t];
      const P = this.P;
      for (let i = 0; i < NL; i++) {
        if (this.skip[i]) continue;
        const o = 4 * i, x = set.lay[i][0][t], y = set.lay[i][1][t];
        const e = expand && this.expOn[i] ? expanderStep(this.prm, this.st, i, x, y) : (this.expOn[i] && this.prm ? this.st.gain[i] : 1);
        l += e * (P[o] * x + P[o + 1] * y) - x;
        r += e * (P[o + 2] * x + P[o + 3] * y) - y;
      }
      this.l = l; this.r = r;
    }

    process(inputs, outputs) {
      const out = outputs[0];
      const oL = out[0], oR = out[1] || out[0], n = oL.length;
      const set = this.sets.get(this.active);
      if (!set || !this.playing) { oL.fill(0); if (oR !== oL) oR.fill(0); return true; }
      for (let f = 0; f < n; f++) {
        if (!this.playing) { oL[f] = 0; oR[f] = 0; continue; }
        if (this.pLeft > 0) {
          const P = this.P, Pi = this.Pi;
          for (let j = 0; j < P.length; j++) P[j] += Pi[j];
          if (--this.pLeft === 0) { P.set(this.Pt); this.updateSkip(); }
          else if (f === 0) this.updateSkip();
        }
        if (this.mLeft > 0) { this.m += this.mi; if (--this.mLeft === 0) this.m = this.mt; }
        const end = set.n;
        // reaching the end: fade out (no repeat) or crossfade into the start (repeat)
        if (!this.repeat && this.pos >= end - FADE && !(this.after && this.after.end) && this.tTarget !== 0) this.fadeTo(0, Math.max(1, end - this.pos), { end: true });
        let l, r;
        const wrapAt = end - FADE;
        if (this.repeat && end > 4 * FADE && this.pos >= wrapAt) {
          const k = this.pos - wrapAt, a = (k + 0.5) / FADE;
          this.mixAt(set, this.pos, true);
          l = (1 - a) * this.l; r = (1 - a) * this.r;
          this.mixAt(set, k, false);
          l += a * this.l; r += a * this.r;
        } else if (this.pos < end) {
          this.mixAt(set, this.pos, true);
          l = this.l; r = this.r;
        } else { l = 0; r = 0; }
        if (this.xf) {                              // switching versions: fade the old one out underneath
          const a = (this.xf.k + 0.5) / SWITCH, t = Math.min(this.pos, this.xf.from.n - 1);
          this.mixAt(this.xf.from, t, false);
          l = a * l + (1 - a) * this.l; r = a * r + (1 - a) * this.r;
          if (++this.xf.k >= SWITCH) this.xf = null;
        }
        const g = this.m * this.tg;
        oL[f] = g * l; oR[f] = g * r;
        this.pos++;
        if (this.repeat && this.pos >= end) this.pos = end > 4 * FADE ? FADE : 0;     // the start was heard in the crossfade
        else if (this.pos > end) this.pos = end;
        // after moving on, so a jump made when a fade ends is exactly where the next sample comes from
        if (this.tLeft > 0) { this.tg += this.tStep; if (--this.tLeft === 0) { this.tg = this.tTarget; this.fadeDone(); } }
      }
      this.report(false);
      return true;
    }
  }
  registerProcessor('es-mixer', Mixer);
})(typeof globalThis !== 'undefined' ? globalThis : self);
