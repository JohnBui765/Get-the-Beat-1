# Echo Stems 1.3

A desktop music app for playing, mixing and studying music. Echo Stems splits a song into six layers (vocals, guitar, piano, other, bass and drums) on your own computer, then lets you mix them: raise the instrument you're learning, hear two instruments alone together, take the singer out and play along. Nothing is uploaded, and once the engine is saved it works with the internet unplugged.

## What's new in 1.3

**Faders you can hear.** In 1.2, *Level match* reset the overall loudness to the original's after every move, so it turned everything down by as much as your boost turned a layer up. With one layer soloed that cancelled the boost completely: moving the piano from 0 to +5.9 dB changed what you heard by 0.0 dB. In the full mix the peak guard did much the same, because a loudly mastered song has no room above its own peaks: in a test song a +5.9 dB piano came through as about +2 dB while everything else dropped about 4 dB, so the song got quieter rather than the piano louder.

1.3 replaces Level match with **headroom** (top right):

- Everything you hear, your mix and the Original alike, plays a fixed amount below the song's own level: **6 dB** by default, or *None* or *12 dB*. It's an exact halving (a quarter at 12 dB), so the song stays bit for bit, only quieter. Turn your speakers up once to make up for it.
- Your faders act on top of that at face value: **+6 dB on a layer is +6 dB in your ears, and every other layer stays exactly where it was.**
- The master moves only when a boost would push the peaks past the original's own. Then the whole mix is lowered just enough, the **Master** readout turns mustard and the line above the lanes says by how much. With 6 dB of headroom that happens only past about +6 dB, the mustard end of the fader; with 12 dB, hardly ever.
- A quick safety bound keeps every peak under the original's even in the middle of a drag, before the exact peak search has finished.

**Before and after for one layer.** Click a layer's dB number (or press **A**) to hear that layer at 0 dB for a moment; the number turns teal and struck through, and the fader keeps your setting. Click again (or press A) to hear your setting. A works on the layer you touched last, or on the one whose control has focus. Moving the fader ends the comparison.

**Waveforms at the size you hear them.** Every lane is drawn to scale against the original's loudest moment, so the headroom shows as space above and below the waveforms, a boost grows into it and a cut shrinks. A thin outline marks where the layer would be at 0 dB, so each change shows as the gap between the two. If the peak guard lowers the mix, every lane shrinks together.

**Saved files at the song's own level.** *Save what you're hearing* and *Save the beat* keep your balance exactly and leave the headroom out, so the file is as loud as the original, lowered only if a boost would push its peaks past the original's.

**Smaller fixes.** In Solo, the line above the lanes says "nothing else" when every other layer is muted (1.2 still promised room tone), and names the muted ones when only some are. The report adds the headroom and the loudness as heard.

## What's new in 1.2

**A mixer built so boosts stay natural.** You always hear the original recording, plus only the change you ask for. With every fader at 0 dB you hear the song bit for bit; a boost adds a copy of that layer locked to the original sample by sample, so its tone, room and stereo position stay the original's.

- **Faders from silence to +12 dB**, each with a printed scale. The band from 0 to +6 dB (teal) is where boosts sound cleanest; above +6 dB (mustard) the engine's flaws start to show. Double-click a fader for 0 dB; arrow keys move it in 0.5 dB steps (Shift: 0.1 dB).
- **S and M** solo and mute each layer. Solo several layers to hear them together.
- **Left-right position** for every layer. It keeps the layer's loudness as it moves. Double-click to centre.
- **Original** plays the song exactly as recorded without losing your mix; click it again to return. **Beat** mutes the vocals and resets the rest; **Reset** puts every layer back to 0 dB. Both can be undone.
- **Room tone under Solo**: when you solo, the rest of the song plays 20 dB below (adjustable, or silent), like room tone under film dialogue. It hides the seams.
- **Clean up between phrases** (optional, in Solo): gently lowers stray sound from other instruments while a soloed layer rests, by at most 15 dB. It never cuts.

**Honest levels.** Peaks never go above the original's own peaks: if a boost would push them higher, the whole mix is lowered just enough. Nothing is ever squashed by a limiter. (1.2's *Level match* is gone in 1.3; see above.)

**Layers that add up to the song exactly.** Whatever the engine assigns to no layer is handed to the layer that dominates at that moment and pitch. Played together at 0 dB, the six saved layers rebuild the song.

**Three qualities.** *Standard* hears each moment about 1.3 times (as in 1.1). *High* (recommended) hears each moment twice and averages, trusting each listen most where the moment sat mid-chunk; this is what the engine's own settings ask for. *Maximum* listens four times, half of them with left and right swapped. On your computer a 4-minute song takes about 7, 10½ and 21 minutes. The app shows its estimate for each song before you start.

**Uncertain passages.** At High and Maximum, stretches where the engine's listens disagreed are shaded in mustard on that layer: a boost may sound rough there. The shading is a first calibration; the report includes the numbers so it can be tuned.

**A library.** Whole songs are kept automatically, losslessly (24-bit FLAC, checked by the encoder as it writes), with your mix. They reopen in seconds without the internet, and every layer is checked against its fingerprint. Parts (30 or 60 s) are kept only if you click **Keep in library**. A 4-minute song takes about 200 MB.

**Compare versions.** Separate the same part again at another quality and switch between the versions (or press V) while it plays.

**Saving.** *Save what you're hearing*, *Save the beat* and *Save every layer*, as 24-bit WAV (with dither) or 32-bit float WAV for recording software. *Save every layer* asks for a folder and writes all six files there.

**The engine settings** (download, backup, speed check) now live behind the gear button at the top right.

## Updating from version 1.2

1. In your `echo-stems` repository, click **Add file > Upload files** and drag in `index.html`, `sw.js` and `README.md` from this folder. (Dragging in everything works too: GitHub leaves the unchanged files as they are.)
   - changed: `index.html`, `sw.js`, `README.md`
   - unchanged: everything else, including `engine.js`, `analysis.js`, `mixer-worklet.js`, `library-worker.js` and the `ort-1.24.3`, `flac` and `fonts` folders
2. Click **Commit changes**, wait a minute, then open the app once while online. Wait a few seconds, close it and open it again. Version 1.3 shows "1.3" next to the name.
3. Everything now plays 6 dB quieter at first (the headroom): turn your speakers up once.

Your library, your mixes, the engine you downloaded and your settings all stay as they are. Coming from 1.1? Drag in everything from this folder at once, including the folders.

## Put it online for the first time (GitHub Pages)

1. On the same GitHub account as Echo Loop, create a new public repository, for example `echo-stems`.
2. Upload everything in this folder, including the `ort-1.24.3`, `flac` and `fonts` folders. The largest file is 24 MB, under GitHub's 25 MB limit for uploads from a browser.
3. In the repository, open **Settings > Pages** and choose **Deploy from a branch > main > / (root)**. Save.
4. After a minute, open `https://<your-username>.github.io/echo-stems/` in Chrome or Edge on your desktop.
5. Click **Install app** at the top right to give it its own window and a Start-menu entry.

## Using it

1. **Get the engine** (first time only): the home screen offers **Download the engine (336 MB, once)**.
2. **Add a song**: drop an MP3, M4A, WAV or FLAC anywhere, or click **Add a song**. For the cleanest layers use the best copy you have (WAV, FLAC or MP3 at 320 kbps).
3. **Choose the part and the quality**, then **Separate**. Whole songs go into the library when they're done.
4. **Mix.** Space plays and pauses, Home goes to the start, the left and right arrows jump 5 s (Shift: 1 s), O compares with the original, A compares the layer you touched last with 0 dB, B sets the beat, V switches versions. Click a waveform to jump there.
5. **Copy report** puts the numbers on the clipboard: paste them into our chat with a sentence about how it sounds.

## Good to know

- **Memory.** While a 4-minute song is open, Echo Stems uses about 1.2 GB of memory: the mixing engine keeps its own copy of every layer so it never waits for the page. Songs up to 8 minutes can be separated whole.
- **Where things are kept.** The engine and the library are stored by Chrome for your github.io address, separately from Echo Loop. Clearing browsing data for that site removes them. In Settings, **Keep my library safe** asks Chrome not to clear it when the disk runs low.
- **The cleanest sound path.** Echo Stems mixes at 44.1 kHz in 32-bit floating point. For no rate conversion at all, open Windows Settings > System > Sound, choose your output device and set Format to 24 bit, 44100 Hz; switch Audio enhancements off there too.
- **Use the web address.** Double-clicking `index.html` in a folder doesn't work: browsers don't run the background engine or the mixing engine from a file opened that way.
- **Updating later.** If you change any file, also change `es-app-v4` to `es-app-v5` (and so on) at the top of `sw.js`.

## Files

| File | Purpose |
|---|---|
| `index.html` | The app: library, separation, mixer, settings |
| `engine.js` | The background engine: download and storage, spectrograms, the three qualities, handing out the leftover, the disagreement map |
| `analysis.js` | Loudness to ITU-R BS.1770, the exact peak search, fingerprints |
| `mixer-worklet.js` | The mixing engine that plays what you hear, and the maths that saves exactly the same |
| `library-worker.js` | Packs layers losslessly (FLAC) for the library |
| `sw.js` | Keeps the app available offline |
| `manifest.webmanifest` | App name, icon and colours used when it's installed |
| `ort-1.24.3/` | ONNX Runtime Web 1.24.3 by Microsoft (MIT licence), which runs the engine on the graphics card |
| `flac/` | libFLAC by Xiph.Org (BSD licence), compiled to WebAssembly by libflac.js (MIT); see `flac/LICENSE.txt` |
| `fonts/` | Archivo by Omnibus-Type (SIL Open Font License); see `fonts/OFL.txt` |
| `icon-180.png`, `icon-192.png`, `icon-512.png`, `make_icons.py` | App icons and the script that draws them |

## Credits

- **Separation model:** BS-RoFormer SW, six layers. The BS-RoFormer design comes from researchers at ByteDance (2023). The browser-ready ONNX version is by elicwhite on Hugging Face (`elicwhite/bs-roformer-sw-6stem-onnx`, MIT). The weights were rehosted by jarredou; who originally trained them is not documented.
- **Runtime:** ONNX Runtime Web by Microsoft, MIT licence (see `ort-1.24.3/LICENSE.txt`). Version 1.30.0, used by the "Newer runtime" speed setting, is fetched from jsDelivr when it is tested.
- **Lossless library:** libFLAC (Xiph.Org Foundation, BSD) via libflac.js 5.6.0 (MIT).
- **Typeface:** Archivo (Omnibus-Type, SIL Open Font License 1.1).
