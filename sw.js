// Echo Stems service worker: keeps the app, its mixing engine, the FLAC library code, the typeface and the separation
// runtime available with no internet.
// The separation engine itself (the 336 MB model) is saved separately in the "es-model" cache by engine.js,
// and this file never deletes it, so updating the app doesn't make you download the engine again.
// Bump CACHE whenever you upload changed files, so computers pick them up.
// The "es-" prefix keeps these caches apart from the Echo Loop apps on the same site.
const CACHE = 'es-app-v3';
const RUNTIME = 'es-ort-1.24.3';
// The newer runtime the speed check may choose comes from a public CDN; once fetched it is kept here for offline use.
const CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
const CDN_CACHE = 'es-cdn-ort-1.30.0';
const SHELL = [
  './', './index.html', './engine.js', './analysis.js', './mixer-worklet.js', './library-worker.js', './manifest.webmanifest',
  './icon-180.png', './icon-192.png', './icon-512.png',
  './flac/libflac.min.wasm.js', './flac/libflac.min.wasm.wasm',
  './fonts/archivo-latin-wdth-normal.woff2', './fonts/archivo-latin-ext-wdth-normal.woff2', './fonts/archivo-vietnamese-wdth-normal.woff2',
];
const ORT = ['./ort-1.24.3/ort.min.js', './ort-1.24.3/ort-wasm-simd-threaded.jsep.mjs', './ort-1.24.3/ort-wasm-simd-threaded.jsep.wasm'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    Promise.all([
      // straight from the server, never from the browser's own short-term cache, so an update can't mix old and new files
      caches.open(CACHE).then((c) => c.addAll(SHELL.map((u) => new Request(u, { cache: 'reload' })))),
      caches.open(RUNTIME).then((c) => c.addAll(ORT.map((u) => new Request(u, { cache: 'reload' })))),
    ]).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys
        .filter((k) => (k.startsWith('es-app-') && k !== CACHE) || (k.startsWith('es-ort-') && k !== RUNTIME) || (k.startsWith('es-cdn-') && k !== CDN_CACHE))
        .map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET' || req.headers.has('range')) return;
  const url = new URL(req.url);
  if (req.url.startsWith(CDN)) {                                  // newer runtime: from the cache, or fetched once and kept
    event.respondWith(
      caches.open(CDN_CACHE).then((c) => c.match(req.url).then((hit) => hit || fetch(req.url, { mode: 'cors', credentials: 'omit' }).then((res) => {
        if (res && res.ok) c.put(req.url, res.clone()).catch(() => {});
        return res;
      })))
    );
    return;
  }
  if (url.origin !== self.location.origin) return;                 // the engine download from Hugging Face goes straight through
  const scope = new URL(self.registration.scope).pathname;
  if (!url.pathname.startsWith(scope)) return;

  // The engine runtime never changes within a version folder: serve it from the cache, no refresh.
  if (url.pathname.startsWith(scope + 'ort-')) {
    event.respondWith(
      caches.open(RUNTIME).then((c) => c.match(req, { ignoreSearch: true }).then((hit) => hit || fetch(req).then((res) => {
        if (res && res.ok && res.type === 'basic') c.put(req, res.clone()).catch(() => {});
        return res;
      })))
    );
    return;
  }

  // The app itself: serve from the cache first (instant, works offline), refresh the cache in the background.
  let saved = Promise.resolve();
  const network = fetch(req.url, { cache: 'no-cache', credentials: 'same-origin' }).then((res) => {
    if (res && res.ok && res.type === 'basic' && !res.redirected) {
      const copy = res.clone();
      saved = caches.open(CACHE).then((c) => c.put(req, copy)).catch(() => {});
    }
    return res;
  });
  event.waitUntil(network.then(() => saved, () => {}));
  event.respondWith(
    caches.open(CACHE)
      .then((c) => c.match(req, { ignoreSearch: true }))
      .then((hit) => hit || network.catch(() => (req.mode === 'navigate' ? caches.open(CACHE).then((c) => c.match('./index.html')) : undefined)))
      .then((res) => res || new Response('Offline', { status: 503, statusText: 'Offline' }))
  );
});
