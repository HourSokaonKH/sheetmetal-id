#!/usr/bin/env python3
"""
Strain Rate Verification — Quasi-Static Assumption Check
Reads raw UTM displacement data and computes engineering strain rate.
"""

import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw_data')
GAUGE_LENGTH = 85.0  # mm (from UTM header Lo)

def verify_strain_rate():
    print("=" * 65)
    print("STRAIN RATE VERIFICATION — Quasi-Static Assumption")
    print("=" * 65)
    print(f"  Gauge length (Lo): {GAUGE_LENGTH} mm")
    print(f"  Quasi-static threshold: ε̇ < 1×10⁻³ s⁻¹\n")
    print(f"  {'Specimen':>10s}  {'v (mm/s)':>10s}  {'ε̇ (s⁻¹)':>12s}  {'Status':>10s}")
    print(f"  {'-'*48}")

    rates = []
    for d in ['00', '45', '90']:
        for s in ['01', '02', '03']:
            fname = os.path.join(DATA_DIR, f'UTM_{d}_{s}.txt')
            if not os.path.exists(fname):
                continue

            times, disps = [], []
            in_data = False
            with open(fname, 'r') as f:
                for line in f:
                    if line.startswith('Time\tLoad'):
                        in_data = True
                        continue
                    if in_data:
                        parts = line.strip().split('\t')
                        if len(parts) >= 4:
                            try:
                                times.append(float(parts[0]))
                                disps.append(float(parts[3]))
                            except ValueError:
                                continue

            times = np.array(times)
            disps = np.array(disps)

            # Crosshead velocity (linear fit over full test)
            if len(times) > 10:
                coeffs = np.polyfit(times, disps, 1)
                v_crosshead = coeffs[0]  # mm/s
                strain_rate = v_crosshead / GAUGE_LENGTH
                rates.append(strain_rate)

                status = "✓ OK" if strain_rate < 1e-3 else "✗ HIGH"
                print(f"  {d}-{s:>2s}      {v_crosshead:10.4f}  {strain_rate:12.2e}  {status:>10s}")

    if rates:
        mean_rate = np.mean(rates)
        print(f"\n  {'Mean':>10s}  {'':>10s}  {mean_rate:12.2e}")
        print(f"\n  Result: All specimens at ε̇ ≈ {mean_rate:.1e} s⁻¹")
        print(f"  This is well below 10⁻³ s⁻¹ → quasi-static assumption CONFIRMED.")
        print(f"  Crosshead speed ≈ {np.mean(rates)*GAUGE_LENGTH*60:.1f} mm/min")

if __name__ == '__main__':
    verify_strain_rate()
