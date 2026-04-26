# 1. Data acquisition protocol

To reproduce these methods on your own material, follow the protocol below
**exactly**. Deviating from it (different specimen geometry, different
acquisition rate, different DIC subset) is fine, but you will need to update
the corresponding constants in `material_constants.json` and the
`HALF_WIDTH`, `THICKNESS`, `HALF_GAUGE` values at the top of the Abaqus
scripts.

---

## 1.1 Specimen

| Item              | Value                                  |
| ----------------- | -------------------------------------- |
| Standard          | DIN 50125 - H 20×80 dog-bone           |
| Material          | SGCC JIS G 3302 (galvanised steel)     |
| Sheet thickness   | 1.50 ± 0.02 mm (measure each specimen) |
| Gauge length L₀   | 80 mm                                  |
| Gauge width w₀    | 20 mm                                  |
| Cutting angles    | 0°, 45°, 90° to rolling direction      |
| Replicates        | **3 specimens per angle** (= 9 total)  |

Cutting must be done by **water-jet** or **wire-EDM** to avoid heat-affected
zones near the gauge section. Hand-grind any burrs perpendicular to the
loading axis, then degrease with isopropanol.

## 1.2 Speckle pattern (for DIC)

| Item              | Value                                  |
| ----------------- | -------------------------------------- |
| Base coat         | Matte white acrylic, ≤ 30 μm thick     |
| Speckle           | Black acrylic, **0.30 – 0.40 mm** dot   |
| Speckle density   | 50 – 70 % black coverage               |
| Application       | Stencil + airbrush at 1 bar, 15 cm     |
| Drying            | 24 h, room temperature                 |

A reference speckle photograph is included in the paper
(`figures/fig_speckle_pattern.png`). The MATLAB script `extract_strains_*deg.m`
expects ≥ 4 useable gauge zones per specimen.

## 1.3 UTM (load) acquisition

| Parameter           | Value                                |
| ------------------- | ------------------------------------ |
| Machine class       | ISO 7500-1 class 1                   |
| Load cell           | 50 kN (≥ 5× expected peak load)      |
| Crosshead speed     | **2 mm/min** (quasi-static, ε̇ ≈ 4·10⁻⁴ s⁻¹) |
| Output rate         | **1 Hz**                             |
| Output channels     | Time, Load (kgf), Displacement (mm)  |

The UTM should export an ASCII file shaped like `examples/UTM_00_01.txt`. The
header keys (`Lo`, `So`, `Customer`, …) are tolerated by the reader; only the
`-----------Curve----------` block is parsed.

## 1.4 DIC acquisition

| Parameter           | Value                                |
| ------------------- | ------------------------------------ |
| Camera              | Canon EOS R6 Mark II (24.2 MP) or equivalent |
| Lens                | Canon RF 100 mm F2.8L Macro          |
| Recording           | **4K UHD video, 25 fps**             |
| Stand-off           | 600 mm (gauge fills 60 % of frame)   |
| Lighting            | 2× CRI > 95 LED panels, 45° diffuse  |
| Synchronisation     | Audible UTM trigger + clapper        |

**Frame extraction** to PNG is done with FFmpeg, with a centred 1000×2160
crop that contains the entire gauge length:

```bash
ffmpeg -i video.mp4 -vf "fps=5,crop=1000:2160:(iw-ow)/2:(ih-oh)/2" img_%04d.png
```

`fps=5` produces ~5 images per UTM data point (1 Hz), giving comfortable
phase margin during the post-synchronisation step.

## 1.5 DIC computation (Ufreckles)

The pipeline assumes that you have run [Ufreckles](https://github.com/jrethore/Ufreckles)
on the extracted PNG sequence with the following settings:

| Parameter         | Value                                |
| ----------------- | ------------------------------------ |
| Subset (element)  | 32 × 32 px                           |
| Step (overlap)    | 16 px (50 % overlap)                 |
| Shape function    | Q4 bilinear                          |
| Correlation       | ZNCC                                 |
| Strain calculation| Polynomial fit, half-width = 1 element |
| Reference image   | `img_0001.png` (un-deformed)         |

The result is a single binary file `<specimen>.res` (MATLAB `-mat`) that
holds the displacement field for every node and frame. The exact structure
expected by the MATLAB scripts is documented in
[02_data_format.md](02_data_format.md).

## 1.6 Specimen ↔ file naming

```
<angle>-<replicate>            angle ∈ {00, 45, 90};  replicate ∈ {01, 02, 03}
```

Examples: `00-01`, `45-03`, `90-02`. **Use these names everywhere** — the
scripts auto-discover specimens by glob.

## 1.7 Where to put the files

After all the steps above, your workspace should contain:

```
raw_data/
├── UTM_00_01.txt        ← UTM exports (rename if needed)
├── UTM_00_02.txt
├── …
├── UTM_90_03.txt
├── 00-01/
│   ├── 00-01.res        ← Ufreckles binary
│   ├── 00-01-gage-01.csv  ← Ufreckles "Strain gage" exports
│   ├── …                  (8 zones recommended)
│   └── img_0001.png … img_NNNN.png   (kept for traceability, not parsed)
├── 00-02/
│   …
└── 90-03/
```

Bundled `examples/` shows exactly this layout for **specimen 00-01**.
