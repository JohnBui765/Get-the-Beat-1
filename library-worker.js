/* Echo Stems library worker: packs a layer losslessly (FLAC, 24-bit) for the song library, off the page's thread.

   Each layer is scaled by a power of two only if it would otherwise exceed full scale (so nothing ever clips), turned
   into 24-bit whole numbers, and encoded with libFLAC's verify mode on: libFLAC decodes every frame it writes and
   compares it with the input, so a bad file can't be saved silently. A fingerprint of the 24-bit numbers is kept with
   the song; when the song is reopened the page recomputes it from the decoded sound and checks they match.

   FLAC encoder: libFLAC 1.3 (Xiph.Org, BSD licence) compiled to WebAssembly by libflac.js (MIT), in flac/. */
'use strict';
self.FLAC_SCRIPT_LOCATION = 'flac/';
importScripts('flac/libflac.min.wasm.js', 'analysis.js');

const ready = new Promise((resolve) => {
  if (self.Flac.isReady()) resolve();
  else self.Flac.on('ready', () => resolve());
});

async function encode({ id, L, R, sr }) {
  await ready;
  const Flac = self.Flac;
  const n = L.length;
  let peak = 0;
  for (let i = 0; i < n; i++) { const a = Math.abs(L[i]), b = Math.abs(R[i]); if (a > peak) peak = a; if (b > peak) peak = b; }
  const scale = peak > 1 ? Math.pow(2, Math.ceil(Math.log2(peak))) : 1;
  const k = 8388608 / scale;
  const t0 = performance.now();
  const enc = Flac.create_libflac_encoder(sr, 2, 24, 5, n, true, 0);
  if (!enc) throw new Error('The FLAC encoder couldn’t start.');
  const parts = [];
  const st = Flac.init_encoder_stream(enc, (data) => { parts.push(data.slice()); });
  if (st !== 0) { Flac.FLAC__stream_encoder_delete(enc); throw new Error('The FLAC encoder couldn’t start (' + st + ').'); }
  const B = 65536, buf = new Int32Array(B * 2);
  let h = self.ESAnalysis.hashStart();
  for (let i = 0; i < n; i += B) {
    const m = Math.min(B, n - i);
    for (let j = 0; j < m; j++) {
      let a = Math.round(L[i + j] * k), b = Math.round(R[i + j] * k);
      a = a > 8388607 ? 8388607 : a < -8388608 ? -8388608 : a;
      b = b > 8388607 ? 8388607 : b < -8388608 ? -8388608 : b;
      buf[2 * j] = a; buf[2 * j + 1] = b;
    }
    h = self.ESAnalysis.hashInts(h, buf, 2 * m);
    if (!Flac.FLAC__stream_encoder_process_interleaved(enc, m === B ? buf : buf.subarray(0, 2 * m), m)) {
      const state = Flac.FLAC__stream_encoder_get_state(enc);
      Flac.FLAC__stream_encoder_delete(enc);
      throw new Error('The FLAC encoder stopped (state ' + state + ').');
    }
    if ((i / B) % 16 === 0) self.postMessage({ type: 'progress', id, f: i / n });
  }
  const okDone = Flac.FLAC__stream_encoder_finish(enc);
  Flac.FLAC__stream_encoder_delete(enc);
  if (!okDone) throw new Error('The FLAC encoder’s own check found a problem, so this layer wasn’t saved.');
  let bytes = 0;
  for (const p of parts) bytes += p.length;
  const out = new Uint8Array(bytes);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  self.postMessage({ type: 'encoded', id, flac: out.buffer, hash: h >>> 0, scale, n, ms: performance.now() - t0 }, [out.buffer]);
}

self.onmessage = (e) => {
  const m = e.data || {};
  if (m.type !== 'encode') return;
  encode(m).catch((err) => self.postMessage({ type: 'error', id: m.id, message: (err && err.message) || String(err) }));
};
