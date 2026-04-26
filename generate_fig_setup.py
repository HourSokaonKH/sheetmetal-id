#!/usr/bin/env python3
"""
Generate Figure 3.2: Experimental setup composite from lab photos.
  (a) Full setup — UTM + camera + lighting
  (b) Close-up of UTM grips with specimen and camera
  (c) Camera LCD showing recording settings
  (d) Camera-to-specimen distance measurement

Reads: lab_photo_A.png, lab_photo_B.png, lab_photo_C.png, lab_photo_D.png
Produces: output/fig_experimental_setup.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)


def load_and_crop(filename, crop_frac=None):
    """Load image and optionally crop by fractional bounds (left, top, right, bottom)."""
    img = Image.open(os.path.join(BASE_DIR, filename))
    if crop_frac:
        w, h = img.size
        box = (int(crop_frac[0] * w), int(crop_frac[1] * h),
               int(crop_frac[2] * w), int(crop_frac[3] * h))
        img = img.crop(box)
    return np.array(img)


# Load and crop images
# A: Full setup — crop slightly to remove empty edges
img_a = load_and_crop('lab_photo_A.png', (0.02, 0.0, 0.98, 0.95))

# B: Close-up of grips — crop to focus on UTM + camera
img_b = load_and_crop('lab_photo_B.png', (0.05, 0.0, 0.95, 0.95))

# C: Camera LCD — crop to focus on the camera back/LCD
img_c = load_and_crop('lab_photo_C.png', (0.05, 0.05, 0.95, 0.95))

# D: Distance measurement — crop to show tape measure
img_d = load_and_crop('lab_photo_D.png', (0.0, 0.05, 1.0, 0.95))

# Create figure: (a) large on top, (b)(c)(d) as three panels below
fig = plt.figure(figsize=(14, 12))

# Top row: (a) full setup — wide panel
ax_a = fig.add_axes([0.02, 0.42, 0.52, 0.56])
ax_a.imshow(img_a)
ax_a.set_xticks([])
ax_a.set_yticks([])
ax_a.set_title('(a) Full experimental setup', fontsize=12, fontweight='bold', pad=6)

# Add annotations on photo A
h_a, w_a = img_a.shape[:2]
# UTM label
ax_a.annotate('UTM', xy=(w_a * 0.50, h_a * 0.10),
              fontsize=10, color='white', fontweight='bold',
              ha='center', va='center',
              bbox=dict(facecolor='black', alpha=0.7, pad=3, edgecolor='none'))
# Camera label
ax_a.annotate('DIC Camera', xy=(w_a * 0.50, h_a * 0.60),
              fontsize=10, color='white', fontweight='bold',
              ha='center', va='center',
              bbox=dict(facecolor='black', alpha=0.7, pad=3, edgecolor='none'))
# Light labels
ax_a.annotate('LED Light', xy=(w_a * 0.15, h_a * 0.15),
              fontsize=9, color='white', fontweight='bold',
              ha='center', va='center',
              bbox=dict(facecolor='black', alpha=0.7, pad=2, edgecolor='none'))
ax_a.annotate('LED Light', xy=(w_a * 0.85, h_a * 0.15),
              fontsize=9, color='white', fontweight='bold',
              ha='center', va='center',
              bbox=dict(facecolor='black', alpha=0.7, pad=2, edgecolor='none'))

# Top right: (b) close-up grips + camera
ax_b = fig.add_axes([0.56, 0.42, 0.42, 0.56])
ax_b.imshow(img_b)
ax_b.set_xticks([])
ax_b.set_yticks([])
ax_b.set_title('(b) UTM grips with specimen', fontsize=12, fontweight='bold', pad=6)

h_b, w_b = img_b.shape[:2]
# Annotate specimen and grips
ax_b.annotate('Specimen', xy=(w_b * 0.35, h_b * 0.50),
              fontsize=10, color='white', fontweight='bold',
              ha='center', va='center',
              bbox=dict(facecolor='black', alpha=0.7, pad=3, edgecolor='none'))
ax_b.annotate('Upper grip', xy=(w_b * 0.35, h_b * 0.15),
              fontsize=9, color='white', fontweight='bold',
              ha='center', va='center',
              bbox=dict(facecolor='black', alpha=0.7, pad=2, edgecolor='none'))
ax_b.annotate('Lower grip', xy=(w_b * 0.35, h_b * 0.85),
              fontsize=9, color='white', fontweight='bold',
              ha='center', va='center',
              bbox=dict(facecolor='black', alpha=0.7, pad=2, edgecolor='none'))
ax_b.annotate('Canon EOS R6', xy=(w_b * 0.80, h_b * 0.40),
              fontsize=9, color='white', fontweight='bold',
              ha='center', va='center',
              bbox=dict(facecolor='black', alpha=0.7, pad=2, edgecolor='none'))

# Bottom left: (c) camera LCD
ax_c = fig.add_axes([0.02, 0.02, 0.47, 0.38])
ax_c.imshow(img_c)
ax_c.set_xticks([])
ax_c.set_yticks([])
ax_c.set_title('(c) Camera recording: 4K video, 1/320s, f/11, ISO 1600',
               fontsize=11, fontweight='bold', pad=6)

# Bottom right: (d) distance measurement
ax_d = fig.add_axes([0.51, 0.02, 0.47, 0.38])
ax_d.imshow(img_d)
ax_d.set_xticks([])
ax_d.set_yticks([])
ax_d.set_title('(d) Camera-to-specimen distance',
               fontsize=11, fontweight='bold', pad=6)

out_path = os.path.join(OUT_DIR, 'fig_experimental_setup.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_path}")
plt.close()
