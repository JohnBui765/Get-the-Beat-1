/* Echo Stems engine. Runs in a background worker so the page stays smooth while a song is separated.

   What it does:
   - keeps the separation model (the "engine", a 336 MB file) in the browser's storage on this computer,
     downloading it once from Hugging Face or taking it from a file you choose;
   - loads it onto the graphics card (WebGPU) or, if asked, the processor (WebAssembly);
   - cuts stereo 44.1 kHz audio into 4-second chunks, turns each chunk into a spectrogram, lets the model split it
     into six layers, hands whatever the model assigned to no layer to the layer that dominates at that moment and
     pitch (so the six layers always add up to the song exactly), turns the layers back into sound and blends the
     chunks together again;
   - at High and Maximum quality, hears every moment two or four times in overlapping chunks (Maximum also listens to
     every other chunk with left and right swapped) and averages them, trusting each listen most where the moment sat
     near the middle of its chunk; where two listens disagree, it notes how much, for the shading in the mixer.

   Model: BS-RoFormer SW, six layers (bass, drums, other, vocals, guitar, piano), exported to ONNX by
   elicwhite: https://huggingface.co/elicwhite/bs-roformer-sw-6stem-onnx
   Its contract: inputs spec_real / spec_imag [1, 2, 1025, 345], outputs out_spec_real / out_spec_imag
   [1, 6, 2, 1025, 345]; STFT n_fft 2048, hop 512, periodic Hann window, centred with reflection at the edges,
   not normalised. The spectrogram maths below reproduces torch.stft / torch.istft with those settings.

   The same file is loaded by the test scripts in Node (it then exports CORE and does nothing else). */
'use strict';

const CORE = (() => {
  const SR = 44100;
  const CHUNK = 176400;                     // 4 s: the length the model was traced at
  const OVERLAP = 44100;                    // 1 s shared by neighbouring chunks
  const STEP = CHUNK - OVERLAP;
  const NFFT = 2048, HOP = 512, PAD = NFFT / 2;
  const F = NFFT / 2 + 1;                   // 1025 frequency bins
  const T = Math.floor(CHUNK / HOP) + 1;    // 345 frames
  const SPEC = 2 * F * T;                   // values in one input tensor (2 channels)
  const STEMS = ['bass', 'drums', 'other', 'vocals', 'guitar', 'piano'];
  const NSTEM = STEMS.length;
  const PADDED = (T - 1) * HOP + NFFT;      // length of the overlap-add buffer before trimming

  // ---- tables ----
  const win = new Float64Array(NFFT);       // periodic Hann (torch.hann_window default)
  const winOverN = new Float64Array(NFFT);  // window / N: the inverse FFT's 1/N folded into the synthesis window
  for (let i = 0; i < NFFT; i++) {
    win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / NFFT);
    winOverN[i] = win[i] / NFFT;
  }
  const invEnv = new Float64Array(PADDED);  // 1 / (sum of squared windows), as torch.istft divides by it
  {
    const env = new Float64Array(PADDED);
    for (let t = 0; t < T; t++) {
      const o = t * HOP;
      for (let i = 0; i < NFFT; i++) env[o + i] += win[i] * win[i];
    }
    for (let n = 0; n < PADDED; n++) invEnv[n] = env[n] > 1e-11 ? 1 / env[n] : 0;
  }
  const BITS = Math.round(Math.log2(NFFT));
  const rev = new Uint32Array(NFFT);
  for (let i = 0; i < NFFT; i++) {
    let r = 0;
    for (let b = 0, x = i; b < BITS; b++, x >>= 1) r = (r << 1) | (x & 1);
    rev[i] = r;
  }
  const cosT = new Float64Array(NFFT / 2), sinT = new Float64Array(NFFT / 2);
  for (let k = 0; k < NFFT / 2; k++) {
    cosT[k] = Math.cos((2 * Math.PI * k) / NFFT);
    sinT[k] = Math.sin((2 * Math.PI * k) / NFFT);
  }

  // In-place forward FFT of length NFFT, X[k] = sum x[n] e^(-2*pi*i*k*n/N).
  function fft(re, im) {
    for (let i = 0; i < NFFT; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let size = 2; size <= NFFT; size <<= 1) {
      const half = size >> 1, stride = NFFT / size;
      for (let start = 0; start < NFFT; start += size) {
        for (let k = 0, tw = 0; k < half; k++, tw += stride) {
          const wr = cosT[tw], wi = -sinT[tw];
          const a = start + k, b = a + half;
          const xr = re[b] * wr - im[b] * wi;
          const xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr; im[b] = im[a] - xi;
          re[a] += xr; im[a] += xi;
        }
      }
    }
  }

  const sRe = new Float64Array(NFFT), sIm = new Float64Array(NFFT);

  // Spectrogram of one stereo chunk (L, R: Float32Array(CHUNK)) into outRe / outIm laid out as [channel][bin][frame].
  // Both channels share one complex FFT: z = left + i*right, then the two spectra are separated again.
  function stft(L, R, outRe, outIm) {
    const last = CHUNK - 1;
    for (let t = 0; t < T; t++) {
      const s0 = t * HOP - PAD;
      for (let i = 0; i < NFFT; i++) {
        let j = s0 + i;
        if (j < 0) j = -j; else if (j > last) j = 2 * last - j;     // reflection at the chunk edges
        const w = win[i];
        sRe[i] = L[j] * w;
        sIm[i] = R[j] * w;
      }
      fft(sRe, sIm);
      for (let k = 0; k < F; k++) {
        const m = (NFFT - k) & (NFFT - 1);
        const zr = sRe[k], zi = sIm[k], mr = sRe[m], mi = sIm[m];
        const iL = k * T + t, iR = (F + k) * T + t;
        outRe[iL] = 0.5 * (zr + mr); outIm[iL] = 0.5 * (zi - mi);
        outRe[iR] = 0.5 * (zi + mi); outIm[iR] = 0.5 * (mr - zr);
      }
    }
  }

  const accL = new Float64Array(PADDED), accR = new Float64Array(PADDED);

  // Sound of one layer (index s) from the model's output [layer][channel][bin][frame] into outL / outR.
  // Both channels share one inverse FFT: Z = X_left + i*X_right gives left in the real part, right in the imaginary.
  function istftStem(oRe, oIm, s, outL, outR) {
    accL.fill(0); accR.fill(0);
    const bL = (s * 2) * F * T, bR = (s * 2 + 1) * F * T;
    const NY = F - 1;
    for (let t = 0; t < T; t++) {
      // DC and Nyquist bins are real for a real signal (irfft ignores their imaginary parts)
      sRe[0] = oRe[bL + t]; sIm[0] = -oRe[bR + t];
      sRe[NY] = oRe[bL + NY * T + t]; sIm[NY] = -oRe[bR + NY * T + t];
      for (let k = 1; k < NY; k++) {
        const off = k * T + t;
        const a = oRe[bL + off], b = oIm[bL + off], c = oRe[bR + off], d = oIm[bR + off];
        // stored conjugated, so the forward FFT below acts as an inverse FFT
        sRe[k] = a - d; sIm[k] = -(b + c);
        sRe[NFFT - k] = a + d; sIm[NFFT - k] = b - c;
      }
      fft(sRe, sIm);
      const o = t * HOP;
      for (let i = 0; i < NFFT; i++) {
        const w = winOverN[i];
        accL[o + i] += sRe[i] * w;
        accR[o + i] -= sIm[i] * w;
      }
    }
    for (let i = 0; i < CHUNK; i++) {
      const n = PAD + i, g = invEnv[n];
      outL[i] = accL[n] * g;
      outR[i] = accR[n] * g;
    }
  }

  /* How much of the music each chunk hears.
     standard: chunks every 3 s (each moment heard about 1.3 times), blended over the shared second, as in 1.0 and 1.1.
     high:     chunks every 2 s, so every moment is heard twice (the engine's own settings file asks for this).
     maximum:  chunks every second, every other one with left and right swapped: four listens per moment.
     High and Maximum weight each listen by a raised-cosine curve that peaks mid-chunk, where the engine hears the
     most music on both sides; at the very start and end of the song the outer half of the chunk counts fully. */
  const QUALITIES = {
    standard: { id: 'standard', step: CHUNK - OVERLAP, hann: false, mirror: false },
    high: { id: 'high', step: CHUNK / 2, hann: true, mirror: false },
    maximum: { id: 'maximum', step: CHUNK / 4, hann: true, mirror: true },
  };
  const qualityOf = (q) => QUALITIES[q] || QUALITIES.standard;
  function chunkCount(N, step) { return N <= CHUNK ? 1 : Math.ceil((N - CHUNK) / step) + 1; }
  function segmentCount(N, quality) { return chunkCount(N, qualityOf(quality).step); }

  const hann = new Float32Array(CHUNK);
  for (let i = 0; i < CHUNK; i++) { const s = Math.sin((Math.PI * i) / CHUNK); hann[i] = s * s; }
  // The weight a chunk's own sample i gets, for chunk k of K.
  function weightAt(q, i, k, K) {
    if (K === 1) return 1;
    if (q.hann) {
      if ((k === 0 && i < CHUNK / 2) || (k === K - 1 && i >= CHUNK / 2)) return 1;
      return hann[i];
    }
    const O = CHUNK - q.step;
    if (k > 0 && i < O) return i / O;
    if (k < K - 1 && i >= CHUNK - O) return 1 - (i - (CHUNK - O)) / O;
    return 1;
  }

  // Copies the chunk starting at `start` into cL / cR, padding with silence past the end. Returns its real length.
  function fillChunk(L, R, start, cL, cR) {
    const len = Math.max(0, Math.min(CHUNK, L.length - start));
    cL.set(L.subarray(start, start + len)); cR.set(R.subarray(start, start + len));
    if (len < CHUNK) { cL.fill(0, len); cR.fill(0, len); }
    return len;
  }

  /* Hands out what the model left over. In every frequency bin of every frame, the mixture X minus the six layers'
     sum is shared among the layers in proportion to each layer's energy there:  S_i += |S_i|^2 / sum|S_j|^2 * (X - sum S_j).
     Afterwards the layers add up to X exactly. Of all the ways to make them add up, this one changes each layer least
     relative to its own strength, so it cannot move the six layers, taken together, further from the real instruments.
     Also checks every number the graphics card sent back. Adds the leftover and mixture energy to `stats`. */
  function project(xRe, xIm, oRe, oIm, stats) {
    const FT = F * T;
    let eR = 0, eX = 0;
    for (let c = 0; c < 2; c++) {
      for (let j = 0; j < FT; j++) {
        const xi = c * FT + j, xr = xRe[xi], xm = xIm[xi];
        let sr = 0, sm = 0, den = 0;
        for (let s = 0; s < NSTEM; s++) {
          const o = (s * 2 + c) * FT + j, a = oRe[o], b = oIm[o];
          sr += a; sm += b; den += a * a + b * b;
        }
        if (!Number.isFinite(den) || !Number.isFinite(sr + sm)) throw new Error('The graphics card sent back numbers that aren’t valid, so the separation was stopped.');
        const rr = xr - sr, rm = xm - sm;
        eR += rr * rr + rm * rm; eX += xr * xr + xm * xm;
        if (den > 1e-24) {
          const inv = 1 / den;
          for (let s = 0; s < NSTEM; s++) {
            const o = (s * 2 + c) * FT + j, a = oRe[o], b = oIm[o], w = (a * a + b * b) * inv;
            oRe[o] = a + w * rr; oIm[o] = b + w * rm;
          }
        } else {                                   // no layer claims this bin: share it equally
          for (let s = 0; s < NSTEM; s++) { const o = (s * 2 + c) * FT + j; oRe[o] += rr / NSTEM; oIm[o] += rm / NSTEM; }
        }
      }
    }
    stats.leftover += eR; stats.mixture += eX;
  }

  function energy(a) { let e = 0; for (let i = 0; i < a.length; i += 7) e += a[i] * a[i]; return e; }

  // Disagreement between two listens is summed in quarter-second blocks of the song.
  const DIS_BLOCK = SR / 4;

  // True when the model's output is usable: finite numbers, and not all silence when the input wasn't silent.
  function outputLooksValid(re, im, inputEnergy) {
    let peak = 0;
    for (let i = 0; i < re.length; i += 97) {
      const a = re[i], b = im[i];
      if (!Number.isFinite(a) || !Number.isFinite(b)) return false;
      const m = Math.abs(a) + Math.abs(b);
      if (m > peak) peak = m;
    }
    return inputEnergy < 1e-6 || peak > 0;
  }

  /* Separates stereo audio into the six layers.
     run(re, im) must return a promise of { re, im } (the model's two outputs as Float32Arrays).
     The next chunk's spectrogram and the previous chunk's layers are worked out while the model runs. */
  async function separate({ L, R, run, onProgress, isCancelled, check, quality }) {
    const q = qualityOf(quality);
    const N = L.length, nSeg = chunkCount(N, q.step);
    const outL = STEMS.map(() => new Float32Array(N)), outR = STEMS.map(() => new Float32Array(N));
    const wsum = new Float32Array(N);
    const cL = new Float32Array(CHUNK), cR = new Float32Array(CHUNK);
    // this chunk's layers and the previous chunk's, kept to compare the two listens where they overlap
    let cur = { L: STEMS.map(() => new Float32Array(CHUNK)), R: STEMS.map(() => new Float32Array(CHUNK)), start: 0, seg: -1 };
    let prev = { L: STEMS.map(() => new Float32Array(CHUNK)), R: STEMS.map(() => new Float32Array(CHUNK)), start: 0, seg: -1 };
    const sets = [0, 1].map(() => ({ re: new Float32Array(SPEC), im: new Float32Array(SPEC), start: 0, len: 0, e: 0, mirror: false }));
    const nDis = Math.ceil(N / DIS_BLOCK);
    const dis = q.hann && nSeg > 1 ? { D: new Float64Array(NSTEM * nDis), E: new Float64Array(NSTEM * nDis), X: new Float64Array(nDis) } : null;
    const stats = { leftover: 0, mixture: 0 };
    const segMs = [];
    let dspMs = 0, pending = null;

    const prepare = (seg) => {
      const st = sets[seg & 1];
      st.start = seg * q.step;
      st.mirror = q.mirror && (seg & 1) === 1;
      st.len = st.mirror ? fillChunk(R, L, st.start, cL, cR) : fillChunk(L, R, st.start, cL, cR);
      st.e = energy(cL) + energy(cR);
      stft(cL, cR, st.re, st.im);
    };

    const compare = () => {                       // the overlap of the previous chunk and this one
      const a = Math.max(cur.start, 0), b = Math.min(prev.start + CHUNK, N);
      const D = dis.D, E = dis.E, X = dis.X;
      for (let n = a; n < b; n++) {
        const i1 = n - prev.start, i2 = n - cur.start;
        const w = weightAt(q, i1, prev.seg, nSeg) * weightAt(q, i2, cur.seg, nSeg);
        if (!(w > 0)) continue;
        const blk = (n / DIS_BLOCK) | 0, xl = L[n], xr = R[n];
        X[blk] += w * (xl * xl + xr * xr);
        for (let s = 0; s < NSTEM; s++) {
          const pl = prev.L[s][i1], pr = prev.R[s][i1], ql = cur.L[s][i2], qr = cur.R[s][i2];
          const dl = ql - pl, dr = qr - pr, ml = 0.5 * (ql + pl), mr = 0.5 * (qr + pr);
          D[s * nDis + blk] += w * (dl * dl + dr * dr);
          E[s * nDis + blk] += w * (ml * ml + mr * mr);
        }
      }
    };

    const finish = (p) => {
      project(p.set.re, p.set.im, p.re, p.im, stats);
      for (let s = 0; s < NSTEM; s++) {
        if (p.mirror) istftStem(p.re, p.im, s, cur.R[s], cur.L[s]);     // swapped back
        else istftStem(p.re, p.im, s, cur.L[s], cur.R[s]);
      }
      cur.start = p.start; cur.seg = p.seg;
      for (let i = 0; i < p.len; i++) {
        const g = p.start + i;
        if (g >= N) break;
        const w = weightAt(q, i, p.seg, nSeg);
        if (w === 0) continue;
        wsum[g] += w;
        for (let s = 0; s < NSTEM; s++) { outL[s][g] += w * cur.L[s][i]; outR[s][g] += w * cur.R[s][i]; }
      }
      if (dis && p.seg > 0) compare();
      const t = prev; prev = cur; cur = t;
    };

    const t0 = now();
    let td = now();
    prepare(0);
    dspMs += now() - td;
    for (let seg = 0; seg < nSeg; seg++) {
      if (isCancelled && isCancelled()) throw cancelledError();
      const ts = now();
      const set = sets[seg & 1];
      const runP = run(set.re, set.im);
      td = now();
      if (pending) { finish(pending); pending = null; }
      if (seg + 1 < nSeg) prepare(seg + 1);
      dspMs += now() - td;
      const out = await runP;
      if (seg === 0 && !outputLooksValid(out.re, out.im, set.e)) {
        throw new Error('The graphics card sent back silence instead of music, so the separation was stopped.');
      }
      if (check) check(seg);
      pending = { re: out.re, im: out.im, set, start: set.start, len: set.len, mirror: set.mirror, seg };
      segMs.push(now() - ts);
      if (onProgress) onProgress({ done: seg + 1, total: nSeg, elapsedMs: now() - t0, segMs: segMs[segMs.length - 1] });
    }
    td = now();
    if (pending) finish(pending);
    for (let g = 0; g < N; g++) {                 // turn the weighted sums into weighted averages
      const w = wsum[g];
      if (w === 1) continue;
      const inv = w > 0 ? 1 / w : 0;
      for (let s = 0; s < NSTEM; s++) { outL[s][g] *= inv; outR[s][g] *= inv; }
    }
    let disagreement = null;
    if (dis) {
      const data = new Float32Array(NSTEM * nDis);
      for (let s = 0; s < NSTEM; s++) {
        for (let b = 0; b < nDis; b++) {
          const d = dis.D[s * nDis + b], e = dis.E[s * nDis + b], x = dis.X[b];
          data[s * nDis + b] = x > 0 || e > 0 ? 10 * Math.log10((d + 1e-30) / (2 * (e + 0.003 * x) + 1e-30)) : NaN;
        }
      }
      disagreement = { block: DIS_BLOCK, nb: nDis, data };
    }
    dspMs += now() - td;
    const leftoverDb = stats.mixture > 0 ? 10 * Math.log10((stats.leftover + 1e-30) / stats.mixture) : -Infinity;
    return { outL, outR, nSeg, segMs, dspMs, totalMs: now() - t0, quality: q.id, disagreement, leftoverDb };
  }

  function now() { return (typeof performance !== 'undefined' ? performance : Date).now(); }
  function cancelledError() { const e = new Error('Stopped'); e.name = 'Cancelled'; return e; }

  return { SR, CHUNK, OVERLAP, STEP, NFFT, HOP, F, T, SPEC, STEMS, QUALITIES, fft, stft, istftStem, chunkCount, segmentCount, weightAt, fillChunk, project, outputLooksValid, energy, separate, DIS_BLOCK };
})();

if (typeof module !== 'undefined' && module.exports) {
  module.exports = CORE;
} else {
  /* ================= worker side ================= */
  // Which ONNX Runtime this worker uses is set by the page's first message ({ type: 'init', rt }), not by the worker's
  // address: when the service worker serves this file from its cache, the address loses its ?rt= part.
  // Each speed-check set-up runs in a fresh worker, so only one engine sits on the graphics card at a time.
  importScripts('analysis.js');                // loudness measurements, shared with the page
  const CDN130 = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
  const RUNTIMES = {
    jsep124: { script: 'ort-1.24.3/ort.min.js', wasm: new URL('ort-1.24.3/', self.location).href },   // saved with the app
    wgpu130: { script: CDN130 + 'ort.webgpu.min.js', wasm: CDN130 },                                  // newer runtime, from a public CDN
  };
  let RT = 'jsep124';
  const REPO = 'elicwhite/bs-roformer-sw-6stem-onnx';
  const REV = 'a744f80957374e1735ad70fa122670b7961da8cc';           // pinned upload, so the file can never change under us
  const MODELS = {
    f16: { file: 'bs_roformer_sw_6stem_fp16.onnx', bytes: 352778874 },
    f32: { file: 'bs_roformer_sw_6stem_fp32.onnx', bytes: 701010313 },
  };
  const MODEL_CACHE = 'es-model';                                     // never cleared by app updates (see sw.js)
  const keyFor = (kind) => new URL('engine-store/' + MODELS[kind].file, self.location).href;
  const urlsFor = (kind) => [
    `https://huggingface.co/${REPO}/resolve/${REV}/${MODELS[kind].file}`,
    `https://huggingface.co/${REPO}/resolve/main/${MODELS[kind].file}`,
  ];

  // ONNX Runtime reports problems (for example steps it moves to the processor) on the console; keep the latest lines
  // so the speed check can include them in its report.
  const logs = [];
  for (const level of ['log', 'info', 'warn', 'error']) {
    const orig = console[level] ? console[level].bind(console) : () => {};
    console[level] = (...a) => {
      try {
        const s = a.map((x) => (typeof x === 'string' ? x : x && x.message ? x.message : String(x))).join(' ').trim();
        if (s) { logs.push(level + ': ' + s.slice(0, 300)); if (logs.length > 60) logs.shift(); }
      } catch (e) { /* never let logging break the engine */ }
      orig(...a);
    };
  }

  const post = (m, tr) => self.postMessage(m, tr || []);
  const now = () => performance.now();
  let dlAbort = null, cancel = false;
  let ortLoaded = false, current = null;       // current = { key, s, runner }
  let device = null, gpuErrors = 0, gpuError = '', gpuLost = '';

  function loadOrt() {
    if (ortLoaded) return;
    importScripts(RUNTIMES[RT].script);
    const ort = self.ort;
    ort.env.wasm.wasmPaths = RUNTIMES[RT].wasm;
    ort.env.wasm.numThreads = 1;           // GitHub Pages can't enable the shared memory that threads need
    ort.env.wasm.proxy = false;
    ort.env.logLevel = 'warning';
    try { ort.env.webgpu.powerPreference = 'high-performance'; } catch (e) { /* option not in this runtime */ }
    ortLoaded = true;
  }

  async function status() {
    const cache = await caches.open(MODEL_CACHE);
    const cached = {};
    for (const kind of Object.keys(MODELS)) {
      const r = await cache.match(keyFor(kind));
      cached[kind] = r ? (Number(r.headers.get('content-length')) || 1) : 0;
    }
    post({ type: 'status', cached });
  }

  async function download(kind) {
    const m = MODELS[kind];
    const ac = new AbortController();
    dlAbort = ac;
    let res = null, lastErr = null;
    for (const url of urlsFor(kind)) {
      try {
        const r = await fetch(url, { signal: ac.signal, cache: 'no-store' });
        if (r.ok && r.body) { res = r; break; }
        lastErr = new Error('the server answered ' + r.status);
      } catch (e) {
        if (ac.signal.aborted) throw cancelled();
        lastErr = e;
      }
    }
    if (!res) throw new Error('The engine could not be downloaded (' + ((lastErr && lastErr.message) || 'no connection') + '). Check the internet connection and try again.');
    const header = Number(res.headers.get('content-length')) || 0;
    const total = header || m.bytes;
    const reader = res.body.getReader();
    const parts = [];
    let got = 0, lastPost = 0;
    const t0 = now();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        parts.push(value);
        got += value.byteLength;
        const t = now();
        if (t - lastPost > 250) { lastPost = t; post({ type: 'dl-progress', kind, got, total, secs: (t - t0) / 1000 }); }
      }
    } catch (e) {
      if (ac.signal.aborted) throw cancelled();
      throw new Error('The download was interrupted (' + e.message + '). Try again; it starts from the beginning.');
    }
    if (header && got !== header) {
      throw new Error('The download ended early (' + mb(got) + ' of ' + mb(total) + '). Try again.');
    }
    if (Math.abs(got - m.bytes) > m.bytes * 0.02) {
      throw new Error('The file that arrived (' + mb(got) + ') isn’t the engine (' + mb(m.bytes) + ' expected). Try again later.');
    }
    post({ type: 'dl-progress', kind, got, total, secs: (now() - t0) / 1000, saving: true });
    const blob = new Blob(parts, { type: 'application/octet-stream' });
    parts.length = 0;
    const cache = await caches.open(MODEL_CACHE);
    await cache.put(keyFor(kind), new Response(blob, { headers: { 'content-type': 'application/octet-stream', 'content-length': String(blob.size) } }));
    dlAbort = null;
    post({ type: 'dl-done', kind, bytes: got });
  }

  async function importFile(file) {
    const kind = file.size >= 500e6 ? 'f32' : 'f16';
    const cache = await caches.open(MODEL_CACHE);
    await cache.put(keyFor(kind), new Response(file, { headers: { 'content-type': 'application/octet-stream', 'content-length': String(file.size) } }));
    if (current && current.key.startsWith(kind + '/')) await dropSession();
    post({ type: 'imported', kind, bytes: file.size });
  }

  async function removeModel(kind) {
    if (current && current.key.startsWith(kind + '/')) await dropSession();
    const cache = await caches.open(MODEL_CACHE);
    await cache.delete(keyFor(kind));
    post({ type: 'removed', kind });
  }

  async function exportModel(kind) {
    const cache = await caches.open(MODEL_CACHE);
    const r = await cache.match(keyFor(kind));
    if (!r) throw new Error('There is no saved engine to copy.');
    post({ type: 'exported', kind, file: MODELS[kind].file, blob: await r.blob() });
  }

  async function dropSession() {
    if (!current) return;
    try { current.runner.dispose(); } catch (e) { /* buffers already gone */ }
    try { await current.s.release(); } catch (e) { /* already gone */ }
    current = null;
  }

  function watch(dev) {
    if (!dev || dev === device || typeof dev.addEventListener !== 'function') return;
    device = dev;
    dev.addEventListener('uncapturederror', (e) => {
      gpuErrors++;
      if (!gpuError) gpuError = (e && e.error && e.error.message) || 'unknown error';
    });
    if (dev.lost) dev.lost.then((info) => { gpuLost = (info && (info.message || info.reason)) || 'the graphics card stopped responding'; });
  }

  async function adapterInfo() {
    try {
      const i = device && device.adapterInfo;
      if (i) return { vendor: i.vendor || '', architecture: i.architecture || '', description: i.description || '', device: i.device || '' };
    } catch (e) { /* not available */ }
    return null;
  }

  // How each chunk reaches the model and comes back.
  // Normal: the data goes in and out through ONNX Runtime. Replay ("graph capture"): the inputs and outputs live in
  // fixed buffers on the graphics card, so after the first run ONNX Runtime can replay the recorded work in one go.
  async function makeRunner(s, backend, cfg) {
    const ort = self.ort, dims = [1, 2, CORE.F, CORE.T];
    if (!(backend === 'webgpu' && cfg.capture)) {
      return {
        run: async (re, im) => {
          const out = await s.run({ spec_real: new ort.Tensor('float32', re, dims), spec_imag: new ort.Tensor('float32', im, dims) });
          const res = { re: out.out_spec_real.data.slice(), im: out.out_spec_imag.data.slice() };
          for (const k of Object.keys(out)) { try { out[k].dispose(); } catch (e) { /* cpu tensor */ } }
          return res;
        },
        dispose: () => {},
      };
    }
    const dev = device;
    if (!dev || typeof dev.createBuffer !== 'function') throw new Error('this runtime doesn\u2019t share its graphics-card handle, which replay mode needs');
    const U = self.GPUBufferUsage;
    const inBytes = CORE.SPEC * 4;
    const inRe = dev.createBuffer({ size: inBytes, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    const inIm = dev.createBuffer({ size: inBytes, usage: U.STORAGE | U.COPY_DST | U.COPY_SRC });
    const feeds = {
      spec_real: ort.Tensor.fromGpuBuffer(inRe, { dataType: 'float32', dims }),
      spec_imag: ort.Tensor.fromGpuBuffer(inIm, { dataType: 'float32', dims }),
    };
    // The outputs are left to ONNX Runtime and released after each read. Pre-allocated output buffers broke the newer
    // runtime's replay after six runs in testing; this way passed 40 runs on both runtimes.
    return {
      run: async (re, im) => {
        dev.queue.writeBuffer(inRe, 0, re);
        dev.queue.writeBuffer(inIm, 0, im);
        const out = await s.run(feeds);
        try {
          return { re: await out.out_spec_real.getData(), im: await out.out_spec_imag.getData() };
        } finally {
          for (const k of Object.keys(out)) { try { out[k].dispose(); } catch (e) { /* already released */ } }
        }
      },
      dispose: () => { for (const b of [inRe, inIm]) { try { b.destroy(); } catch (e) { /* gone */ } } },
    };
  }

  const cfgKey = (kind, backend, cfg) => [kind, backend, cfg.opt || 'disabled', cfg.capture ? 'replay' : 'plain'].join('/');

  async function getSession(kind, backend, cfg) {
    cfg = cfg || {};
    const key = cfgKey(kind, backend, cfg);
    if (current && current.key === key && !gpuLost) return { s: current.s, runner: current.runner, loadMs: 0, fresh: false };
    await dropSession();
    if (gpuLost) { device = null; gpuLost = ''; }
    loadOrt();
    const t0 = now();
    const cache = await caches.open(MODEL_CACHE);
    const r = await cache.match(keyFor(kind));
    if (!r) throw new Error('The engine isn’t saved on this computer yet. Download it first.');
    let bytes = new Uint8Array(await r.arrayBuffer());
    const opts = backend === 'webgpu'
      ? Object.assign({ executionProviders: ['webgpu'], graphOptimizationLevel: cfg.opt || 'disabled', logSeverityLevel: 2 },
          cfg.capture ? { enableGraphCapture: true, preferredOutputLocation: 'gpu-buffer' } : {})
      : { executionProviders: ['wasm'], logSeverityLevel: 2 };
    let s;
    try {
      s = await self.ort.InferenceSession.create(bytes, opts);
    } catch (e) {
      throw new Error((backend === 'webgpu' ? 'The graphics card couldn’t load the engine' : 'The processor couldn’t load the engine') + ' (' + e.message + ').');
    }
    bytes = null;
    const want = ['spec_real', 'spec_imag', 'out_spec_real', 'out_spec_imag'];
    const have = s.inputNames.concat(s.outputNames);
    if (!want.every((n) => have.includes(n))) {
      try { await s.release(); } catch (e) { /* ignore */ }
      throw new Error('That file isn’t the separation engine this app expects (BS-RoFormer SW, 6 layers).');
    }
    if (backend === 'webgpu') {
      try { watch(await self.ort.env.webgpu.device); } catch (e) { /* no device handle in this runtime */ }
    }
    let runner;
    try { runner = await makeRunner(s, backend, cfg); }
    catch (e) { try { await s.release(); } catch (x) { /* ignore */ } throw new Error('Replay mode couldn’t start (' + e.message + ').'); }
    current = { key, s, runner };
    return { s, runner, loadMs: now() - t0, fresh: true };
  }

  function checkGpu() {
    if (gpuLost) throw new Error('The graphics card stopped responding (' + gpuLost + '). Try again, or run on the processor.');
    if (gpuErrors) throw new Error('The graphics card reported an error (' + gpuError + ').');
  }

  async function separate(msg) {
    cancel = false;
    gpuErrors = 0; gpuError = '';
    const { L, R, kind, backend } = msg;
    const quality = msg.quality || 'standard';
    const cfg = backend === 'webgpu' ? (msg.cfg || {}) : {};
    post({ type: 'stage', stage: 'loading' });
    const g = await getSession(kind, backend, cfg);
    post({ type: 'stage', stage: 'running', total: CORE.segmentCount(L.length, quality) });
    const r = await CORE.separate({
      L, R, run: g.runner.run, check: checkGpu, quality,
      isCancelled: () => cancel,
      onProgress: (p) => post({ type: 'progress', ...p }),
    });
    // Measurements the mixer needs: the layers' K-weighted cross-products (exact loudness for any fader setting)
    // and each layer's playing level (for the clean-up in Solo).
    post({ type: 'stage', stage: 'measuring' });
    const tm = now();
    const flat = [];
    for (let i = 0; i < CORE.STEMS.length; i++) flat.push(r.outL[i], r.outR[i]);
    const gram = self.ESAnalysis.gramSub(flat, CORE.SR, (f) => post({ type: 'measure-progress', f }));
    const levels = CORE.STEMS.map((_, i) => self.ESAnalysis.playingLevel(r.outL[i], r.outR[i], CORE.SR));
    // the peak search's tables (each layer's loudest sample per 1.5 ms), worked out here rather than on the page
    const layers = CORE.STEMS.map((_, i) => [r.outL[i], r.outR[i]]);
    const peaks = self.ESAnalysis.peakTables([L, R], layers);
    const measureMs = now() - tm;
    const stems = CORE.STEMS.map((name, i) => ({ name, L: r.outL[i], R: r.outR[i] }));
    const transfer = [gram.sub.buffer, peaks.bmOrig.L.buffer, peaks.bmOrig.R.buffer, peaks.resMax.buffer];
    for (const t of peaks.bmLayers) transfer.push(t.L.buffer, t.R.buffer);
    for (const st of stems) transfer.push(st.L.buffer, st.R.buffer);
    if (r.disagreement) transfer.push(r.disagreement.data.buffer);
    post({
      type: 'result', stems,
      analysis: { gram: { sub: gram.sub, nSub: gram.nSub, nSig: gram.nSig, sb: gram.sb }, levels, disagreement: r.disagreement, leftoverDb: r.leftoverDb, peaks },
      timing: { loadMs: g.loadMs, freshSession: g.fresh, totalMs: r.totalMs, segMs: r.segMs, dspMs: r.dspMs, measureMs, nSeg: r.nSeg, quality: r.quality, backend, kind, cfg, runtime: RT, adapter: backend === 'webgpu' ? await adapterInfo() : null },
    }, transfer);
  }

  // Speed check: load the engine with one set-up, run the same 4-second chunk a few times, report the times and the result.
  async function bench(msg) {
    const { kind, cfg, L, R, runs } = msg;
    const maxMs = msg.maxMs || 60000;
    logs.length = 0; gpuErrors = 0; gpuError = '';
    post({ type: 'bench-stage', stage: 'loading' });
    const g = await getSession(kind, 'webgpu', cfg);
    const cL = new Float32Array(CORE.CHUNK), cR = new Float32Array(CORE.CHUNK);
    CORE.fillChunk(L, R, 0, cL, cR);
    const re = new Float32Array(CORE.SPEC), im = new Float32Array(CORE.SPEC);
    let t = now();
    CORE.stft(cL, cR, re, im);
    const stftMs = now() - t;
    post({ type: 'bench-stage', stage: 'running' });
    const times = [];
    let out = null;
    const tStart = now();
    for (let i = 0; i < runs; i++) {
      t = now();
      out = await g.runner.run(re, im);
      times.push(now() - t);
      checkGpu();
      if (!CORE.outputLooksValid(out.re, out.im, 1)) throw new Error('the graphics card sent back silence or invalid numbers');
      post({ type: 'bench-progress', run: i + 1, runs, ms: times[i] });
      if (i >= 1 && now() - tStart > maxMs) break;
    }
    const sL = new Float32Array(CORE.CHUNK), sR = new Float32Array(CORE.CHUNK);
    t = now();
    for (let s = 0; s < CORE.STEMS.length; s++) CORE.istftStem(out.re, out.im, s, sL, sR);
    const istftMs = now() - t;
    const res = { re: out.re.slice(), im: out.im.slice() };
    post({ type: 'bench-result', loadMs: g.loadMs, times, stftMs, istftMs, runtime: RT, adapter: await adapterInfo(), logs: logs.slice(-12), re: res.re, im: res.im }, [res.re.buffer, res.im.buffer]);
  }

  function cancelled() { const e = new Error('Stopped'); e.name = 'Cancelled'; return e; }
  function mb(n) { return n > 0 && n < 1048576 ? 'under 1 MB' : Math.round(n / 1048576) + ' MB'; }

  self.onmessage = (e) => {
    const m = e.data || {};
    const job = {
      init: () => { if (!ortLoaded && RUNTIMES[m.rt]) RT = m.rt; },
      status: () => status(),
      download: () => download(m.kind),
      'cancel-download': () => { if (dlAbort) dlAbort.abort(); },
      import: () => importFile(m.file),
      remove: () => removeModel(m.kind),
      export: () => exportModel(m.kind),
      separate: () => separate(m),
      bench: () => bench(m),
      cancel: () => { cancel = true; },
    }[m.type];
    if (!job) return;
    Promise.resolve().then(job).catch((err) => {
      if (m.type === 'download') dlAbort = null;
      post({ type: 'error', for: m.type, cancelled: err && err.name === 'Cancelled', message: (err && err.message) || String(err), logs: m.type === 'bench' ? logs.slice(-12) : undefined });
    });
  };
}
