"""Echo Stems icon: the Echo Loop family's loop ring on a teal-to-indigo tile, around a stack of layer bars.
The top bar (the vocals) is only an outline: taken out of the song, the rest of the stack stays solid."""
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = os.path.dirname(os.path.abspath(__file__))
S = 2048
C = S / 2

a = np.array([13, 148, 136], dtype=float)   # teal-600
b = np.array([55, 48, 163], dtype=float)    # indigo-700
yy, xx = np.mgrid[0:S, 0:S]
t = ((xx + yy) / (2 * S - 2))[..., None]
img = Image.fromarray((a * (1 - t) + b * t).astype(np.uint8), "RGB").convert("RGBA")

glow = Image.new("L", (S, S), 0)
ImageDraw.Draw(glow).ellipse([-S * 0.25, -S * 0.35, S * 0.75, S * 0.55], fill=50)
glow = glow.filter(ImageFilter.GaussianBlur(S * 0.08))
img = Image.composite(Image.new("RGBA", (S, S), (255, 255, 255, 255)), img, glow)

d = ImageDraw.Draw(img)
white = (255, 255, 255, 255)

# loop ring with arrowhead (same motif as Echo Loop, Echo Loop+ and Echo Loop+ Desktop)
R, W = 0.30 * S, 0.064 * S
start, end = -40, 245
d.arc([C - R - W / 2, C - R - W / 2, C + R + W / 2, C + R + W / 2], start=start, end=end, fill=white, width=int(W))
sx, sy = C + R * math.cos(math.radians(start)), C + R * math.sin(math.radians(start))
d.ellipse([sx - W / 2, sy - W / 2, sx + W / 2, sy + W / 2], fill=white)
th = math.radians(end)
P = (C + R * math.cos(th), C + R * math.sin(th))
T = (-math.sin(th), math.cos(th))
N = (math.cos(th), math.sin(th))
L, Hw = 0.10 * S, 0.072 * S
tip = (P[0] + L * T[0], P[1] + L * T[1])
b1 = (P[0] + Hw * N[0] - 0.01 * S * T[0], P[1] + Hw * N[1] - 0.01 * S * T[1])
b2 = (P[0] - Hw * N[0] - 0.01 * S * T[0], P[1] - Hw * N[1] - 0.01 * S * T[1])
d.polygon([tip, b1, b2], fill=white)

# four stacked layers; the top one (vocals) is an outline only
bh, gap = 0.052 * S, 0.030 * S
widths = [0.30, 0.34, 0.28, 0.32]
total = len(widths) * bh + (len(widths) - 1) * gap
y = C - total / 2
for i, wf in enumerate(widths):
    w = wf * S
    box = [C - w / 2, y, C + w / 2, y + bh]
    if i == 0:
        d.rounded_rectangle(box, radius=bh / 2, outline=white, width=int(0.014 * S))
    else:
        d.rounded_rectangle(box, radius=bh / 2, fill=white)
    y += bh + gap

img = img.convert("RGB")
for size, name in [(512, "icon-512.png"), (192, "icon-192.png"), (180, "icon-180.png")]:
    img.resize((size, size), Image.LANCZOS).save(os.path.join(OUT, name), optimize=True)
print("icons written to", OUT)
