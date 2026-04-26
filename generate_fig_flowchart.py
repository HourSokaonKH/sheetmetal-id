#!/usr/bin/env python3
"""
Generate Figure 3.4: Image extraction and DIC analysis flowchart.
Produces output/fig_dic_workflow_flowchart.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 4.5)
ax.axis('off')

# Style
box_kw = dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', edgecolor='#1565C0', lw=1.5)
box_kw2 = dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9', edgecolor='#2E7D32', lw=1.5)
box_kw3 = dict(boxstyle='round,pad=0.4', facecolor='#FFF3E0', edgecolor='#E65100', lw=1.5)
arrow_kw = dict(arrowstyle='->', color='#37474F', lw=2,
                connectionstyle='arc3,rad=0')

# Boxes — top row (main pipeline)
boxes = [
    (1.2, 3.3, '4K Video\n(25 fps)', box_kw),
    (3.5, 3.3, 'FFMPEG\nExtract 1 fps', box_kw),
    (5.8, 3.3, 'Image Sequence\n(PNG)', box_kw),
    (8.1, 3.3, 'Ufreckles\nFEM-DIC', box_kw2),
    (10.4, 3.3, 'Strain Fields\n(Eyy, Exx, Exy)', box_kw3),
]

for x, y, txt, kw in boxes:
    ax.text(x, y, txt, ha='center', va='center', fontsize=10,
            fontweight='bold', bbox=kw)

# Arrows between top boxes
for i in range(len(boxes) - 1):
    x1 = boxes[i][0] + 0.85
    x2 = boxes[i+1][0] - 0.85
    y = 3.3
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=arrow_kw)

# Bottom row — parallel data path
boxes_bot = [
    (1.2, 1.3, 'UTM Machine\n(1 Hz)', box_kw),
    (3.5, 1.3, 'Load–Time\nData', box_kw),
    (5.8, 1.3, 'Engineering\nStress (σ = F/A₀)', box_kw),
]

for x, y, txt, kw in boxes_bot:
    ax.text(x, y, txt, ha='center', va='center', fontsize=10,
            fontweight='bold', bbox=kw)

# Arrows between bottom boxes
for i in range(len(boxes_bot) - 1):
    x1 = boxes_bot[i][0] + 0.85
    x2 = boxes_bot[i+1][0] - 0.85
    y = 1.3
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=arrow_kw)

# Merge box
ax.text(8.1, 1.3, 'Combine\nσ(t) + ε(t)', ha='center', va='center',
        fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F3E5F5',
                  edgecolor='#6A1B9A', lw=1.5))

# Final output
ax.text(10.4, 1.3, 'Stress–Strain\nCurves (CSV)', ha='center', va='center',
        fontsize=10, fontweight='bold', bbox=box_kw3)

# Arrow from stress to merge
ax.annotate('', xy=(8.1 - 0.85, 1.3), xytext=(5.8 + 0.85, 1.3),
            arrowprops=arrow_kw)

# Arrow from merge to output
ax.annotate('', xy=(10.4 - 0.85, 1.3), xytext=(8.1 + 0.85, 1.3),
            arrowprops=arrow_kw)

# Arrow from strain fields down to merge
ax.annotate('', xy=(8.1 + 0.3, 1.3 + 0.65), xytext=(10.4, 3.3 - 0.55),
            arrowprops=dict(arrowstyle='->', color='#37474F', lw=2,
                            connectionstyle='arc3,rad=0.3'))

# Sync label
ax.text(1.2, 2.3, '1:1 temporal\nsynchronization',
        ha='center', va='center', fontsize=8, fontstyle='italic',
        color='#616161')
ax.annotate('', xy=(1.2, 1.85), xytext=(1.2, 2.75),
            arrowprops=dict(arrowstyle='<->', color='#9E9E9E', lw=1))

# Tool labels (small text)
ax.text(3.5, 3.9, 'ffmpeg -vf fps=1', ha='center', fontsize=7,
        fontstyle='italic', color='#616161')
ax.text(8.1, 3.9, 'Python/MATLAB', ha='center', fontsize=7,
        fontstyle='italic', color='#616161')

plt.tight_layout()
import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output',
                   'fig_dic_workflow_flowchart.png')
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {out}")
plt.close()
