#!/usr/bin/env python3
"""
Process img_0001.png for Figure 3.3: Speckle pattern close-up.
Crops to gauge section, adds scale bar and annotations.
Produces: output/fig_speckle_pattern.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'output')

# Load raw image
img = Image.open(os.path.join(BASE_DIR, 'img_0001_raw.png'))
w, h = img.size  # 1000 x 2160

# Crop to gauge section (center region, avoiding grips)
# Grips are at top ~0-400px and bottom ~1760-2160px
# Gauge section roughly: x=250-750, y=550-1600
crop_left = 200
crop_right = 800
crop_top = 500
crop_bottom = 1650
gauge = img.crop((crop_left, crop_top, crop_right, crop_bottom))

# Also get a zoomed-in inset for speckle detail
inset_left = 350
inset_right = 650
inset_top = 900
inset_bottom = 1200
inset = img.crop((inset_left, inset_top, inset_right, inset_bottom))

# Create figure with main image + inset
fig = plt.figure(figsize=(8, 10))

# Main gauge section image
ax_main = fig.add_axes([0.05, 0.02, 0.55, 0.96])
ax_main.imshow(np.array(gauge))
ax_main.set_xticks([])
ax_main.set_yticks([])
ax_main.set_title('(a) Gauge section', fontsize=12, pad=8)

# Scale bar on main image
# Specimen gauge width = 20 mm, cropped width = 600 px
# So 1 mm ≈ 30 px
px_per_mm = (crop_right - crop_left) / 20.0  # 600/20 = 30 px/mm
bar_mm = 5  # 5 mm scale bar
bar_px = bar_mm * px_per_mm

gauge_h = crop_bottom - crop_top
gauge_w = crop_right - crop_left

# Position scale bar at bottom-left
bar_x = 40
bar_y = gauge_h - 60
ax_main.plot([bar_x, bar_x + bar_px], [bar_y, bar_y], 'w-', lw=3)
ax_main.plot([bar_x, bar_x], [bar_y - 10, bar_y + 10], 'w-', lw=2)
ax_main.plot([bar_x + bar_px, bar_x + bar_px], [bar_y - 10, bar_y + 10], 'w-', lw=2)
ax_main.text(bar_x + bar_px / 2, bar_y - 25, f'{bar_mm} mm',
             color='white', fontsize=11, ha='center', fontweight='bold',
             bbox=dict(facecolor='black', alpha=0.5, pad=2, edgecolor='none'))

# Draw rectangle showing inset region
rect_x = inset_left - crop_left
rect_y = inset_top - crop_top
rect_w = inset_right - inset_left
rect_h = inset_bottom - inset_top
rect = mpatches.Rectangle((rect_x, rect_y), rect_w, rect_h,
                           linewidth=2, edgecolor='#FF5722', facecolor='none')
ax_main.add_patch(rect)

# Zoomed inset
ax_inset = fig.add_axes([0.62, 0.45, 0.36, 0.36])
ax_inset.imshow(np.array(inset))
ax_inset.set_xticks([])
ax_inset.set_yticks([])
ax_inset.set_title('(b) Speckle detail', fontsize=12, pad=8)
for spine in ax_inset.spines.values():
    spine.set_color('#FF5722')
    spine.set_linewidth(2)

# Scale bar on inset
# Inset covers 300px = 10mm → 30 px/mm
inset_bar_mm = 2
inset_bar_px = inset_bar_mm * px_per_mm
inset_h = inset_bottom - inset_top
inset_w = inset_right - inset_left
ix, iy = 20, inset_h - 40
ax_inset.plot([ix, ix + inset_bar_px], [iy, iy], 'w-', lw=3)
ax_inset.plot([ix, ix], [iy - 8, iy + 8], 'w-', lw=2)
ax_inset.plot([ix + inset_bar_px, ix + inset_bar_px], [iy - 8, iy + 8], 'w-', lw=2)
ax_inset.text(ix + inset_bar_px / 2, iy - 18, f'{inset_bar_mm} mm',
              color='white', fontsize=10, ha='center', fontweight='bold',
              bbox=dict(facecolor='black', alpha=0.5, pad=2, edgecolor='none'))

# Connection lines from rectangle to inset
fig.patches.append(mpatches.ConnectionPatch(
    xyA=(rect_x + rect_w, rect_y), coordsA=ax_main.transData,
    xyB=(0, 0), coordsB=ax_inset.transAxes,
    color='#FF5722', lw=1, ls='--', alpha=0.7))
fig.patches.append(mpatches.ConnectionPatch(
    xyA=(rect_x + rect_w, rect_y + rect_h), coordsA=ax_main.transData,
    xyB=(0, 1), coordsB=ax_inset.transAxes,
    color='#FF5722', lw=1, ls='--', alpha=0.7))

# Annotation text
ax_note = fig.add_axes([0.62, 0.08, 0.36, 0.3])
ax_note.axis('off')
note_text = (
    "Speckle pattern details:\n"
    "• White spray paint base coat\n"
    "• Black speckle overspray\n"
    "• Speckle size: ~3–5 pixels\n"
    "• Coverage: ~50% black/white\n"
    "• Applied to gauge section\n"
    "  for DIC strain measurement\n\n"
    "Specimen: SGCC JIS G 3302\n"
    "Gauge: 20 × 80 mm\n"
    "Thickness: 1.5 mm"
)
ax_note.text(0.05, 0.95, note_text, transform=ax_note.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(facecolor='#F5F5F5', edgecolor='#BDBDBD',
                       boxstyle='round,pad=0.5'))

out_path = os.path.join(OUT_DIR, 'fig_speckle_pattern.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_path}")
plt.close()
