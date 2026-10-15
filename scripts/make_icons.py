"""Draws the PWA / home-screen icons into frontend/static/icons/.

Run by hand when the artwork changes (it is not part of the build and needs Pillow, which the
project itself does not):

    python scripts/make_icons.py

The icons are drawn, not scaled from tiles/GL.png (90 px, too small), at 4x and reduced so the
edges are smooth. The artwork is a board seen from above: the circle, its red diamond and a
flower on the centre point. It is "maskable": everything important sits inside the central 80%.
"""
import os
import sys

from PIL import Image, ImageDraw

BG = (22, 21, 18)            # --bg (dark theme)
CREAM = (239, 233, 220)      # --garden-white
RED = (184, 69, 60)          # --garden-red
ORANGE = (224, 122, 47)
OUT = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static', 'icons')


def draw(size):
    big = size * 4
    img = Image.new('RGB', (big, big), BG)
    d = ImageDraw.Draw(img)
    c = big / 2
    r = big * 0.36                                   # the board circle, inside the 80% safe zone
    d.ellipse([c - r, c - r, c + r, c + r], fill=CREAM)
    k = r * 0.82
    d.polygon([(c, c - k), (c + k, c), (c, c + k), (c - k, c)], fill=RED)
    k2 = r * 0.42
    d.polygon([(c, c - k2), (c + k2, c), (c, c + k2), (c - k2, c)], fill=CREAM)
    f = r * 0.24                                      # the flower: petals around a centre
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        d.ellipse([c + dx * f * 0.8 - f * 0.62, c + dy * f * 0.8 - f * 0.62,
                   c + dx * f * 0.8 + f * 0.62, c + dy * f * 0.8 + f * 0.62], fill=ORANGE)
    d.ellipse([c - f * 0.5, c - f * 0.5, c + f * 0.5, c + f * 0.5], fill=(255, 224, 160))
    return img.resize((size, size), Image.LANCZOS)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for name, size in (('icon-192.png', 192), ('icon-512.png', 512), ('apple-touch-icon.png', 180)):
        draw(size).save(os.path.join(OUT, name), optimize=True)
        print('wrote', name, file=sys.stderr)
