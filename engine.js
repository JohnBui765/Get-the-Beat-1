/* Echo Stems engine. Runs in a background worker so the page stays smooth while a song is separated.

   What it does:
   - keeps the separation model (the "engine", a 336 MB file) in the browser's storage on this computer,
     downloading it once from Hugging Face or taking it from a file you choose;
   - loads it onto the graphics card (WebGPU) or, if asked, the processor (WebAssembly);
   - cuts stereo 44.1 kHz audio into 4-second chunks that overlap by 1 second, turns each chunk into a
     spectrogram, lets the model split it into six layers, turns the layers back into sound and blends the
     chunks together again.

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

  function segmentCount(N) { return Math.max(1, Math.ceil((N - OVERLAP) / STEP)); }

  // Copies the chunk starting at `start` into cL / cR, padding with silence past the end. Returns its real length.
  function fillChunk(L, R, start, cL, cR) {
    const len = Math.max(0, Math.min(CHUNK, L.length - start));
    cL.set(L.subarray(start, start + len)); cR.set(R.subarray(start, start + len));
    if (len < CHUNK) { cL.fill(0, len); cR.fill(0, len); }
    return len;
  }

  // Adds one separated chunk into the full-length layer, fading in and out across the shared second.
  function addSegment(dstL, dstR, srcL, srcR, start, len, seg, nSeg) {
    const N = dstL.length, fadeOutFrom = CHUNK - OVERLAP;
    for (let i = 0; i < len; i++) {
      const g = start + i;
      if (g >= N) break;
      let w = 1;
      if (seg > 0 && i < OVERLAP) w = i / OVERLAP;
      if (seg < nSeg - 1 && i >= fadeOutFrom) w = 1 - (i - fadeOutFrom) / OVERLAP;
      dstL[g] += srcL[i] * w;
      dstR[g] += srcR[i] * w;
    }
  }

  function energy(a) { let e = 0; for (let i = 0; i < a.length; i += 7) e += a[i] * a[i]; return e; }

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
  async function separate({ L, R, run, onProgress, isCancelled, check }) {
    const N = L.length, nSeg = segmentCount(N);
    const outL = STEMS.map(() => new Float32Array(N)), outR = STEMS.map(() => new Float32Array(N));
    const cL = new Float32Array(CHUNK), cR = new Float32Array(CHUNK);
    const sL = new Float32Array(CHUNK), sR = new Float32Array(CHUNK);
    const sets = [0, 1].map(() => ({ re: new Float32Array(SPEC), im: new Float32Array(SPEC), start: 0, len: 0, e: 0 }));
    const segMs = [];
    let dspMs = 0, pending = null;

    const prepare = (seg) => {
      const st = sets[seg & 1];
      st.start = seg * STEP;
      st.len = fillChunk(L, R, st.start, cL, cR);
      st.e = energy(cL) + energy(cR);
      stft(cL, cR, st.re, st.im);
    };
    const finish = (p) => {
      for (let s = 0; s < NSTEM; s++) {
        istftStem(p.re, p.im, s, sL, sR);
        addSegment(outL[s], outR[s], sL, sR, p.start, p.len, p.seg, nSeg);
      }
    };

    const t0 = now();
    let td = now();
    prepare(0);
    dspMs += now() - td;
    for (let seg = 0; seg < nSeg; seg++) {
      if (isCancelled && isCancelled()) throw cancelledError();
      const ts = now();
      const cur = sets[seg & 1];
      const runP = run(cur.re, cur.im);
      td = now();
      if (pending) { finish(pending); pending = null; }
      if (seg + 1 < nSeg) prepare(seg + 1);
      dspMs += now() - td;
      const out = await runP;
      if (seg === 0 && !outputLooksValid(out.re, out.im, cur.e)) {
        throw new Error('The graphics card sent back silence instead of music, so the test was stopped.');
      }
      if (check) check(seg);
      pending = { re: out.re, im: out.im, start: cur.start, len: cur.len, seg };
      segMs.push(now() - ts);
      if (onProgress) onProgress({ done: seg + 1, total: nSeg, elapsedMs: now() - t0, segMs: segMs[segMs.length - 1] });
    }
    td = now();
    if (pending) finish(pending);
    dspMs += now() - td;
    return { outL, outR, nSeg, segMs, dspMs, totalMs: now() - t0 };
  }

  function now() { return (typeof performance !== 'undefined' ? performance : Date).now(); }
  function cancelledError() { const e = new Error('Stopped'); e.name = 'Cancelled'; return e; }

  return { SR, CHUNK, OVERLAP, STEP, NFFT, HOP, F, T, SPEC, STEMS, fft, stft, istftStem, segmentCount, fillChunk, addSegment, separate };
})();

if (typeof module !== 'undefined' && module.exports) {
  module.exports = CORE;
} else {
  /* ================= worker side ================= */
  const ORT_DIR = 'ort-1.24.3/';
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

  const post = (m, tr) => self.postMessage(m, tr || []);
  let dlAbort = null, cancel = false;
  let ortLoaded = false, session = null, sessionKey = '';
  let device = null, gpuErrors = 0, gpuError = '', gpuLost = '';

  function loadOrt() {
    if (ortLoaded) return;
    importScripts(ORT_DIR + 'ort.min.js');
    const ort = self.ort;
    ort.env.wasm.wasmPaths = new URL(ORT_DIR, self.location).href;
    ort.env.wasm.numThreads = 1;           // GitHub Pages can't enable the shared memory that threads need
    ort.env.wasm.proxy = false;
    ort.env.logLevel = 'error';
    try { ort.env.webgpu.powerPreference = 'high-performance'; } catch (e) { /* older option name */ }
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
    const t0 = performance.now();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        parts.push(value);
        got += value.byteLength;
        const t = performance.now();
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
      throw new Error('The file that arrived (' + mb(got) + ') isn\u2019t the engine (' + mb(m.bytes) + ' expected). Try again later.');
    }
    post({ type: 'dl-progress', kind, got, total, secs: (performance.now() - t0) / 1000, saving: true });
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
    if (sessionKey.startsWith(kind + '/')) await dropSession();
    post({ type: 'imported', kind, bytes: file.size });
  }

  async function removeModel(kind) {
    if (sessionKey.startsWith(kind + '/')) await dropSession();
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
    if (session) { try { await session.release(); } catch (e) { /* already gone */ } }
    session = null; sessionKey = '';
  }

  function watch(dev) {
    if (!dev || dev === device) return;
    device = dev;
    dev.addEventListener('uncapturederror', (e) => {
      gpuErrors++;
      if (!gpuError) gpuError = (e && e.error && e.error.message) || 'unknown error';
    });
    dev.lost.then((info) => { gpuLost = (info && (info.message || info.reason)) || 'the graphics card stopped responding'; });
  }

  async function adapterInfo() {
    try {
      const dev = device || (session && sessionKey.endsWith('/webgpu') ? await self.ort.env.webgpu.device : null);
      const i = dev && dev.adapterInfo;
      if (i) return { vendor: i.vendor || '', architecture: i.architecture || '', description: i.description || '', device: i.device || '' };
    } catch (e) { /* not available */ }
    return null;
  }

  async function getSession(kind, backend) {
    const key = kind + '/' + backend;
    if (session && sessionKey === key && !gpuLost) return { s: session, loadMs: 0, fresh: false };
    await dropSession();
    if (gpuLost) { device = null; gpuLost = ''; }
    loadOrt();
    const t0 = performance.now();
    const cache = await caches.open(MODEL_CACHE);
    const r = await cache.match(keyFor(kind));
    if (!r) throw new Error('The engine isn’t saved on this computer yet. Download it first.');
    const bytes = new Uint8Array(await r.arrayBuffer());
    const opts = backend === 'webgpu'
      ? { executionProviders: ['webgpu'], graphOptimizationLevel: 'disabled' }
      : { executionProviders: ['wasm'] };
    let s;
    try {
      s = await self.ort.InferenceSession.create(bytes, opts);
    } catch (e) {
      throw new Error((backend === 'webgpu' ? 'The graphics card couldn’t load the engine' : 'The processor couldn’t load the engine') + ' (' + e.message + ').');
    }
    const want = ['spec_real', 'spec_imag', 'out_spec_real', 'out_spec_imag'];
    const have = s.inputNames.concat(s.outputNames);
    if (!want.every((n) => have.includes(n))) {
      try { await s.release(); } catch (e) { /* ignore */ }
      throw new Error('That file isn’t the separation engine this app expects (BS-RoFormer SW, 6 layers).');
    }
    if (backend === 'webgpu') {
      try { watch(await self.ort.env.webgpu.device); } catch (e) { /* no device handle in this version */ }
    }
    session = s; sessionKey = key;
    return { s, loadMs: performance.now() - t0, fresh: true };
  }

  async function separate(msg) {
    cancel = false;
    gpuErrors = 0; gpuError = '';
    const { L, R, kind, backend } = msg;
    post({ type: 'stage', stage: 'loading' });
    const { s, loadMs, fresh } = await getSession(kind, backend);
    post({ type: 'stage', stage: 'running', total: CORE.segmentCount(L.length) });
    const ort = self.ort, dims = [1, 2, CORE.F, CORE.T];
    const run = async (re, im) => {
      const feeds = { spec_real: new ort.Tensor('float32', re, dims), spec_imag: new ort.Tensor('float32', im, dims) };
      const out = await s.run(feeds);
      const res = { re: out.out_spec_real.data.slice(), im: out.out_spec_imag.data.slice() };
      for (const k of Object.keys(out)) { try { out[k].dispose(); } catch (e) { /* cpu tensor */ } }
      return res;
    };
    const check = () => {
      if (gpuLost) throw new Error('The graphics card stopped responding (' + gpuLost + '). Try again, or run on the processor.');
      if (gpuErrors) throw new Error('The graphics card reported an error (' + gpuError + '). Try again, or run on the processor.');
    };
    const r = await CORE.separate({
      L, R, run, check,
      isCancelled: () => cancel,
      onProgress: (p) => post({ type: 'progress', ...p }),
    });
    const stems = CORE.STEMS.map((name, i) => ({ name, L: r.outL[i], R: r.outR[i] }));
    const transfer = [];
    for (const st of stems) transfer.push(st.L.buffer, st.R.buffer);
    post({
      type: 'result', stems,
      timing: { loadMs, freshSession: fresh, totalMs: r.totalMs, segMs: r.segMs, dspMs: r.dspMs, nSeg: r.nSeg, backend, kind, adapter: backend === 'webgpu' ? await adapterInfo() : null },
    }, transfer);
  }

  function cancelled() { const e = new Error('Stopped'); e.name = 'Cancelled'; return e; }
  function mb(n) { return n > 0 && n < 1048576 ? 'under 1 MB' : Math.round(n / 1048576) + ' MB'; }

  self.onmessage = (e) => {
    const m = e.data || {};
    const job = {
      status: () => status(),
      download: () => download(m.kind),
      'cancel-download': () => { if (dlAbort) dlAbort.abort(); },
      import: () => importFile(m.file),
      remove: () => removeModel(m.kind),
      export: () => exportModel(m.kind),
      separate: () => separate(m),
      cancel: () => { cancel = true; },
    }[m.type];
    if (!job) return;
    Promise.resolve().then(job).catch((err) => {
      if (m.type === 'download') dlAbort = null;
      post({ type: 'error', for: m.type, cancelled: err && err.name === 'Cancelled', message: (err && err.message) || String(err) });
    });
  };
}
