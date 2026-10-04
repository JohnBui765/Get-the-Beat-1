# Echo Stems (engine test)

The first stage of your vocal remover. Echo Stems splits a song into six layers (vocals, guitar, piano, other, bass and drums) on your own computer. Nothing is uploaded: the song never leaves your PC, and once the engine is saved it works with the internet unplugged.

This test version answers two questions before the full app is built:

- **How fast is it on your graphics card?** It times every part of the job and estimates how long a whole song would take.
- **How good is the beat?** You hear the original, the beat (the original with the vocals taken out), and every layer on its own.

## Put it online (GitHub Pages)

1. On the **same GitHub account** as Echo Loop, create a new public repository, for example `echo-stems`.
2. Upload **everything in this folder, including the `ort-1.24.3` folder**. On GitHub's upload page, drag the whole folder's contents in at once so the `ort-1.24.3` folder keeps its name and its three files. The largest file is 24 MB, under GitHub's 25 MB limit for uploads from a browser.
3. In the repository, open **Settings → Pages** and choose **Deploy from a branch → main → / (root)**. Save.
4. After a minute, open `https://<your-username>.github.io/echo-stems/` in **Chrome** or **Edge** on your desktop.
5. Click **Install app** at the top right if you want it in its own window with a Start-menu entry.

## Run the test

1. **Get the engine.** Click **Download the engine (336 MB, once)**. It comes from Hugging Face and is saved in Chrome on this computer. If your graphics card lacks 16-bit maths, the app asks for the larger 32-bit engine (669 MB) instead.
2. **Choose a song.** Drop an MP3 (or M4A, WAV, FLAC) onto the page. Drag the highlighted 30-second part to a stretch **with singing**, or press **Listen to this part** to check. You can also pick 60 seconds or the whole song.
3. **Separate.** The first chunk is slower while the graphics card gets ready; the rest run at full speed.
4. **Listen.** **Beat** is the original with the vocals taken out; **Original** is the song as it was. Click a layer's name to take it out too (for example the guitar, to play that part yourself), and click the headphones to hear one layer alone. Space plays and pauses.
5. **Send me the numbers.** Click **Copy report** and paste it into our chat, with a sentence about how the beat sounds.

### What to listen for

- **In the beat:** is there any ghost of the singer left? Do the drums and cymbals stay crisp? Is the bass still full?
- **In the vocals on their own:** do instruments leak into the voice? A little leakage is normal; a lot means the engine was unsure about that song.

## Good to know

- **Works offline.** After the first visit and the engine download, the badge says **Works offline**. From then on you can unplug the internet and the app still opens and separates songs.
- **Where the engine lives.** It is saved in Chrome's storage for your github.io address. Clearing browsing data for that site removes it, and the app would need to download it again. **Save a backup copy** stores the file wherever you choose; **load it from a file** puts it back without downloading.
- **Use the web address.** Double-clicking `index.html` in a folder doesn't work for this app: browsers don't allow the background engine to run in a file opened that way.
- **Processor option.** **Run on: Processor** exists only to compare. It is many times slower than the graphics card.
- **Echo Loop apps on the same site.** Echo Stems keeps its own storage, so it never touches your Echo Loop recordings or flags.
- **Updating later.** If you change any file, also change `es-app-v1` to `es-app-v2` (and so on) at the top of `sw.js`. The engine stays saved through updates.

## Files

| File | Purpose |
|---|---|
| `index.html` | The page: the three steps, the result and the layer mixer |
| `engine.js` | The background engine: download and storage, spectrograms, running the model, joining the chunks |
| `sw.js` | Keeps the app and its runtime available offline |
| `manifest.webmanifest` | App name, icon and colours used when it's installed |
| `ort-1.24.3/` | ONNX Runtime Web 1.24.3 by Microsoft (MIT licence), which runs the model on the graphics card |
| `icon-180.png`, `icon-192.png`, `icon-512.png` | App icons |
| `make_icons.py` | Script that draws the icons (only needed if you want to change them) |

## Credits

- **Separation model:** BS-RoFormer SW, six layers. The BS-RoFormer design comes from researchers at ByteDance (2023). The browser-ready ONNX version is by elicwhite on Hugging Face (`elicwhite/bs-roformer-sw-6stem-onnx`, MIT). The weights were rehosted by jarredou; who originally trained them is not documented.
- **Runtime:** ONNX Runtime Web by Microsoft, MIT licence (see `ort-1.24.3/LICENSE.txt`).
