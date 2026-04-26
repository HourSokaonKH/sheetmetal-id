#!/usr/bin/env python3
"""
Diagnostic script to analyze R² fitting issues and test improved hardening laws.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import os, warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
E_MPA = 200000.0

# ============================================================================
# Load functions (same as data_processing.py)
# ============================================================================
def load_dic_data(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    delimiter = ';' if ';' in lines[7] else ','
    data_lines = lines[7:]
    steps, eyy, exx, exy = [], [], [], []
    for line in data_lines:
        line = line.strip()
        if not line: continue
        parts = line.split(delimiter)
        if len(parts) >= 5:
            try:
                steps.append(int(parts[1]))
                eyy.append(float(parts[2]))
                exx.append(float(parts[3]))
                exy.append(float(parts[4]))
            except: continue
    return pd.DataFrame({'Step': steps, 'Eyy': eyy, 'Exx': exx, 'Exy': exy})

def load_stress_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = ['Time', 'Load', 'Stress', 'Strain']
    return df

# ============================================================================
# Hardening laws
# ============================================================================
def swift(ep, K, e0, n):
    return K * (e0 + ep)**n

def voce(ep, sy, Q, b):
    return sy + Q * (1 - np.exp(-b * ep))

def modified_voce(ep, sy, Q1, b1, Q2, b2):
    """Modified Voce with 2 exponential terms"""
    return sy + Q1*(1-np.exp(-b1*ep)) + Q2*(1-np.exp(-b2*ep))

def swift_voce(ep, alpha, K, e0, n, sy, Q, b):
    """Weighted combination: alpha*Swift + (1-alpha)*Voce"""
    return alpha * swift(ep,K,e0,n) + (1-alpha) * voce(ep,sy,Q,b)

def hockett_sherby(ep, sigma_s, sigma_y, m, p):
    """Hockett-Sherby: sigma = sigma_s - (sigma_s - sigma_y)*exp(-m * ep^p)"""
    return sigma_s - (sigma_s - sigma_y)*np.exp(-m * ep**p)

def modified_swift(ep, K, e0, n, Q, b):
    """Modified Swift: sigma = K*(e0+ep)^n + Q*(1-exp(-b*ep))"""
    return K*(e0+ep)**n + Q*(1-np.exp(-b*ep))

def ludwik(ep, sy, K, n):
    """Ludwik: sigma = sigma_y + K * ep^n"""
    return sy + K * ep**n

# ============================================================================
# MAIN DIAGNOSTIC
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("HARDENING FIT DIAGNOSTIC: Why is R² low?")
    print("=" * 70)

    directions = ['00', '45', '90']
    specimens = ['01', '02', '03']
    
    # Store per-specimen data
    all_data = {}
    for d in directions:
        for s in specimens:
            sf = os.path.join(DATA_DIR, f'stress-{d}-{s}.csv')
            if not os.path.exists(sf): continue
            df = load_stress_data(sf)
            eng_s = df['Stress'].values
            eng_e = df['Strain'].values
            
            # Truncate at UTS
            uts_idx = np.argmax(eng_s)
            eng_s = eng_s[1:uts_idx+1]  # skip zero
            eng_e = eng_e[1:uts_idx+1]
            
            # True conversion
            true_s = eng_s * (1 + eng_e)
            true_e = np.log(1 + eng_e)
            
            # Plastic strain
            ep = true_e - true_s / E_MPA
            mask = ep >= 0
            ep = ep[mask]
            ts = true_s[mask]
            
            all_data[f'{d}_{s}'] = {'ep': ep, 'ts': ts, 'n_points': len(ep)}
    
    # ============================================================================
    # DIAGNOSTIC 1: Per-specimen vs pooled R²
    # ============================================================================
    print("\n" + "="*70)
    print("DIAGNOSTIC 1: Per-specimen vs. Pooled fitting (Voce law)")
    print("="*70)
    
    for d in directions:
        # Pooled
        ep_all = np.concatenate([all_data[f'{d}_{s}']['ep'] for s in specimens if f'{d}_{s}' in all_data])
        ts_all = np.concatenate([all_data[f'{d}_{s}']['ts'] for s in specimens if f'{d}_{s}' in all_data])
        sort_idx = np.argsort(ep_all)
        ep_all, ts_all = ep_all[sort_idx], ts_all[sort_idx]
        
        try:
            popt, _ = curve_fit(voce, ep_all, ts_all, p0=[250,200,10], 
                                bounds=([50,10,0.1],[1000,1000,200]), maxfev=10000)
            res = ts_all - voce(ep_all, *popt)
            r2_pooled = 1 - np.sum(res**2)/np.sum((ts_all-np.mean(ts_all))**2)
        except:
            r2_pooled = -1
        
        print(f"\n--- {int(d)}° Direction ---")
        print(f"  POOLED (all 3 specimens): R² = {r2_pooled:.4f}")
        
        for s in specimens:
            k = f'{d}_{s}'
            if k not in all_data: continue
            ep, ts = all_data[k]['ep'], all_data[k]['ts']
            try:
                popt, _ = curve_fit(voce, ep, ts, p0=[250,200,10],
                                    bounds=([50,10,0.1],[1000,1000,200]), maxfev=10000)
                res = ts - voce(ep, *popt)
                r2 = 1 - np.sum(res**2)/np.sum((ts-np.mean(ts))**2)
                print(f"  Specimen {s}: R² = {r2:.4f}, σy={popt[0]:.1f}, Q={popt[1]:.1f}, b={popt[2]:.2f}")
            except:
                print(f"  Specimen {s}: fitting failed")
    
    # ============================================================================
    # DIAGNOSTIC 2: Plastic region filter (ep > 0.002 vs ep > 0)
    # ============================================================================
    print("\n" + "="*70)
    print("DIAGNOSTIC 2: Effect of plastic strain threshold")
    print("="*70)
    
    for d in directions:
        ep_all = np.concatenate([all_data[f'{d}_{s}']['ep'] for s in specimens if f'{d}_{s}' in all_data])
        ts_all = np.concatenate([all_data[f'{d}_{s}']['ts'] for s in specimens if f'{d}_{s}' in all_data])
        sort_idx = np.argsort(ep_all)
        ep_all, ts_all = ep_all[sort_idx], ts_all[sort_idx]
        
        print(f"\n--- {int(d)}° Direction ---")
        print(f"  Plastic strain range: [{ep_all.min():.6f}, {ep_all.max():.4f}]")
        print(f"  Stress range: [{ts_all.min():.1f}, {ts_all.max():.1f}] MPa")
        print(f"  Total data points: {len(ep_all)}")
        
        for threshold in [0.0, 0.001, 0.002, 0.005, 0.01]:
            mask = ep_all >= threshold
            if np.sum(mask) < 20: continue
            ep_f, ts_f = ep_all[mask], ts_all[mask]
            try:
                popt, _ = curve_fit(voce, ep_f, ts_f, p0=[250,200,10],
                                    bounds=([50,10,0.1],[1000,1000,200]), maxfev=10000)
                res = ts_f - voce(ep_f, *popt)
                r2 = 1 - np.sum(res**2)/np.sum((ts_f-np.mean(ts_f))**2)
                print(f"  Threshold εp > {threshold:.3f}: N={np.sum(mask):4d}, R²={r2:.4f}")
            except:
                print(f"  Threshold εp > {threshold:.3f}: fitting failed")
    
    # ============================================================================
    # DIAGNOSTIC 3: Compare ALL hardening law models (pooled, ep > 0.002)
    # ============================================================================
    print("\n" + "="*70)
    print("DIAGNOSTIC 3: Compare hardening law models (εp > 0.002)")
    print("="*70)
    
    summary_results = {}
    
    for d in directions:
        ep_all = np.concatenate([all_data[f'{d}_{s}']['ep'] for s in specimens if f'{d}_{s}' in all_data])
        ts_all = np.concatenate([all_data[f'{d}_{s}']['ts'] for s in specimens if f'{d}_{s}' in all_data])
        sort_idx = np.argsort(ep_all)
        ep_all, ts_all = ep_all[sort_idx], ts_all[sort_idx]
        
        # Filter plastic region
        mask = ep_all >= 0.002
        ep_f, ts_f = ep_all[mask], ts_all[mask]
        
        print(f"\n--- {int(d)}° Direction (N={len(ep_f)} points, εp > 0.002) ---")
        
        results_d = {}
        
        # 1. Swift
        try:
            popt, _ = curve_fit(swift, ep_f, ts_f, p0=[600,0.005,0.2],
                                bounds=([100,1e-6,0.01],[2000,0.5,1.0]), maxfev=10000)
            res = ts_f - swift(ep_f, *popt)
            r2 = 1 - np.sum(res**2)/np.sum((ts_f-np.mean(ts_f))**2)
            results_d['Swift'] = {'r2': r2, 'params': popt, 'names': ['K','e0','n']}
            print(f"  Swift:          R² = {r2:.4f}  K={popt[0]:.1f}, e0={popt[1]:.5f}, n={popt[2]:.4f}")
        except Exception as e:
            print(f"  Swift: failed ({e})")
        
        # 2. Voce
        try:
            popt, _ = curve_fit(voce, ep_f, ts_f, p0=[250,200,10],
                                bounds=([50,10,0.1],[1000,1000,200]), maxfev=10000)
            res = ts_f - voce(ep_f, *popt)
            r2 = 1 - np.sum(res**2)/np.sum((ts_f-np.mean(ts_f))**2)
            results_d['Voce'] = {'r2': r2, 'params': popt, 'names': ['sy','Q','b']}
            print(f"  Voce:           R² = {r2:.4f}  σy={popt[0]:.1f}, Q={popt[1]:.1f}, b={popt[2]:.2f}")
        except Exception as e:
            print(f"  Voce: failed ({e})")
        
        # 3. Modified Voce (2 terms)
        try:
            popt, _ = curve_fit(modified_voce, ep_f, ts_f, p0=[250,100,20,100,2],
                                bounds=([50,1,0.1,1,0.01],[1000,500,200,500,50]), maxfev=20000)
            res = ts_f - modified_voce(ep_f, *popt)
            r2 = 1 - np.sum(res**2)/np.sum((ts_f-np.mean(ts_f))**2)
            results_d['Mod. Voce (2-term)'] = {'r2': r2, 'params': popt, 'names': ['sy','Q1','b1','Q2','b2']}
            print(f"  Mod. Voce (2t): R² = {r2:.4f}  σy={popt[0]:.1f}, Q1={popt[1]:.1f}, b1={popt[2]:.2f}, Q2={popt[3]:.1f}, b2={popt[4]:.2f}")
        except Exception as e:
            print(f"  Mod. Voce (2t): failed ({e})")
        
        # 4. Swift-Voce combined
        try:
            popt, _ = curve_fit(swift_voce, ep_f, ts_f, p0=[0.5,600,0.005,0.2,250,200,10],
                                bounds=([0,100,1e-6,0.01,50,10,0.1],[1,2000,0.5,1.0,1000,1000,200]),
                                maxfev=20000)
            res = ts_f - swift_voce(ep_f, *popt)
            r2 = 1 - np.sum(res**2)/np.sum((ts_f-np.mean(ts_f))**2)
            results_d['Swift-Voce'] = {'r2': r2, 'params': popt, 'names': ['α','K','e0','n','sy','Q','b']}
            print(f"  Swift-Voce:     R² = {r2:.4f}  α={popt[0]:.3f}")
        except Exception as e:
            print(f"  Swift-Voce: failed ({e})")
        
        # 5. Hockett-Sherby
        try:
            popt, _ = curve_fit(hockett_sherby, ep_f, ts_f, p0=[500,250,5,0.5],
                                bounds=([200,50,0.01,0.01],[1500,500,100,5]), maxfev=20000)
            res = ts_f - hockett_sherby(ep_f, *popt)
            r2 = 1 - np.sum(res**2)/np.sum((ts_f-np.mean(ts_f))**2)
            results_d['Hockett-Sherby'] = {'r2': r2, 'params': popt, 'names': ['σs','σy','m','p']}
            print(f"  Hockett-Sherby: R² = {r2:.4f}  σs={popt[0]:.1f}, σy={popt[1]:.1f}, m={popt[2]:.2f}, p={popt[3]:.3f}")
        except Exception as e:
            print(f"  Hockett-Sherby: failed ({e})")
    
        # 6. Ludwik
        try:
            popt, _ = curve_fit(ludwik, ep_f, ts_f, p0=[250,500,0.3],
                                bounds=([50,10,0.01],[1000,2000,1.0]), maxfev=10000)
            res = ts_f - ludwik(ep_f, *popt)
            r2 = 1 - np.sum(res**2)/np.sum((ts_f-np.mean(ts_f))**2)
            results_d['Ludwik'] = {'r2': r2, 'params': popt, 'names': ['σy','K','n']}
            print(f"  Ludwik:         R² = {r2:.4f}  σy={popt[0]:.1f}, K={popt[1]:.1f}, n={popt[2]:.4f}")
        except Exception as e:
            print(f"  Ludwik: failed ({e})")
        
        # 7. Modified Swift (Swift + exponential term)
        try:
            popt, _ = curve_fit(modified_swift, ep_f, ts_f, p0=[400,0.005,0.2,100,5],
                                bounds=([50,1e-6,0.01,1,0.01],[2000,0.5,1.0,500,100]), maxfev=20000)
            res = ts_f - modified_swift(ep_f, *popt)
            r2 = 1 - np.sum(res**2)/np.sum((ts_f-np.mean(ts_f))**2)
            results_d['Mod. Swift'] = {'r2': r2, 'params': popt, 'names': ['K','e0','n','Q','b']}
            print(f"  Mod. Swift:     R² = {r2:.4f}  K={popt[0]:.1f}, e0={popt[1]:.5f}, n={popt[2]:.4f}, Q={popt[3]:.1f}, b={popt[4]:.2f}")
        except Exception as e:
            print(f"  Mod. Swift: failed ({e})")
        
        summary_results[d] = results_d
    
    # ============================================================================
    # DIAGNOSTIC 4: Per-specimen fitting (best model)
    # ============================================================================
    print("\n" + "="*70)
    print("DIAGNOSTIC 4: Per-specimen fitting with Modified Voce (2-term)")
    print("="*70)
    
    for d in directions:
        print(f"\n--- {int(d)}° Direction ---")
        for s in specimens:
            k = f'{d}_{s}'
            if k not in all_data: continue
            ep, ts = all_data[k]['ep'], all_data[k]['ts']
            mask = ep >= 0.002
            ep_f, ts_f = ep[mask], ts[mask]
            
            # Voce per specimen
            try:
                popt_v, _ = curve_fit(voce, ep_f, ts_f, p0=[250,200,10],
                                      bounds=([50,10,0.1],[1000,1000,200]), maxfev=10000)
                r2_v = 1 - np.sum((ts_f-voce(ep_f,*popt_v))**2)/np.sum((ts_f-np.mean(ts_f))**2)
            except: r2_v = -1
            
            # Modified Voce per specimen
            try:
                popt_mv, _ = curve_fit(modified_voce, ep_f, ts_f, p0=[250,100,20,100,2],
                                       bounds=([50,1,0.1,1,0.01],[1000,500,200,500,50]), maxfev=20000)
                r2_mv = 1 - np.sum((ts_f-modified_voce(ep_f,*popt_mv))**2)/np.sum((ts_f-np.mean(ts_f))**2)
            except: r2_mv = -1
            
            print(f"  Spec {s} (N={len(ep_f)}): Voce R²={r2_v:.4f}, Mod.Voce R²={r2_mv:.4f}")
    
    # ============================================================================
    # DIAGNOSTIC 5: Scatter analysis - inter-specimen variance
    # ============================================================================
    print("\n" + "="*70)
    print("DIAGNOSTIC 5: Inter-specimen scatter analysis")
    print("="*70)
    
    for d in directions:
        stresses_at_strains = {}
        for s in specimens:
            k = f'{d}_{s}'
            if k not in all_data: continue
            ep, ts = all_data[k]['ep'], all_data[k]['ts']
            for target_ep in [0.01, 0.05, 0.10, 0.15, 0.20]:
                idx = np.argmin(np.abs(ep - target_ep))
                if abs(ep[idx] - target_ep) < 0.005:
                    if target_ep not in stresses_at_strains:
                        stresses_at_strains[target_ep] = []
                    stresses_at_strains[target_ep].append(ts[idx])
        
        print(f"\n--- {int(d)}° Direction ---")
        for target_ep in sorted(stresses_at_strains.keys()):
            vals = stresses_at_strains[target_ep]
            if len(vals) >= 2:
                mean_s = np.mean(vals)
                std_s = np.std(vals)
                cov = std_s/mean_s*100
                print(f"  εp={target_ep:.2f}: σ = {mean_s:.1f} ± {std_s:.1f} MPa (CoV={cov:.1f}%)")
    
    # ============================================================================
    # PLOT: Comparison of all hardening models
    # ============================================================================
    print("\n\nGenerating comparison plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    model_colors = {
        'Swift': '#e41a1c', 'Voce': '#377eb8', 'Mod. Voce (2-term)': '#4daf4a',
        'Swift-Voce': '#984ea3', 'Hockett-Sherby': '#ff7f00', 
        'Ludwik': '#a65628', 'Mod. Swift': '#f781bf'
    }
    
    for idx, d in enumerate(directions):
        ax = axes[idx]
        
        ep_all = np.concatenate([all_data[f'{d}_{s}']['ep'] for s in specimens if f'{d}_{s}' in all_data])
        ts_all = np.concatenate([all_data[f'{d}_{s}']['ts'] for s in specimens if f'{d}_{s}' in all_data])
        mask = ep_all >= 0.002
        ep_f, ts_f = ep_all[mask], ts_all[mask]
        
        ax.scatter(ep_f, ts_f, s=1, alpha=0.15, color='gray', label='Experimental')
        
        ep_plot = np.linspace(0.002, ep_f.max(), 300)
        
        if d in summary_results:
            for model_name, info in summary_results[d].items():
                p = info['params']
                r2 = info['r2']
                color = model_colors.get(model_name, 'black')
                
                if model_name == 'Swift': y = swift(ep_plot, *p)
                elif model_name == 'Voce': y = voce(ep_plot, *p)
                elif model_name == 'Mod. Voce (2-term)': y = modified_voce(ep_plot, *p)
                elif model_name == 'Swift-Voce': y = swift_voce(ep_plot, *p)
                elif model_name == 'Hockett-Sherby': y = hockett_sherby(ep_plot, *p)
                elif model_name == 'Ludwik': y = ludwik(ep_plot, *p)
                elif model_name == 'Mod. Swift': y = modified_swift(ep_plot, *p)
                else: continue
                
                ax.plot(ep_plot, y, color=color, linewidth=1.5, 
                        label=f'{model_name} (R²={r2:.3f})')
        
        ax.set_xlabel('Plastic Strain', fontsize=12)
        ax.set_ylabel('True Stress (MPa)', fontsize=12)
        ax.set_title(f'{int(d)}° Direction', fontsize=14)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_hardening_model_comparison.png'), dpi=300)
    plt.close()
    print("  Saved: fig_hardening_model_comparison.png")
    
    # ============================================================================
    # PLOT: Per-specimen fits
    # ============================================================================
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, d in enumerate(directions):
        for j, s in enumerate(specimens):
            ax = axes[idx][j]
            k = f'{d}_{s}'
            if k not in all_data: continue
            ep, ts = all_data[k]['ep'], all_data[k]['ts']
            mask = ep >= 0.002
            ep_f, ts_f = ep[mask], ts[mask]
            
            ax.scatter(ep_f, ts_f, s=3, alpha=0.3, color='gray')
            ep_plot = np.linspace(0.002, ep_f.max(), 200)
            
            # Voce
            try:
                popt, _ = curve_fit(voce, ep_f, ts_f, p0=[250,200,10],
                                    bounds=([50,10,0.1],[1000,1000,200]), maxfev=10000)
                r2 = 1 - np.sum((ts_f-voce(ep_f,*popt))**2)/np.sum((ts_f-np.mean(ts_f))**2)
                ax.plot(ep_plot, voce(ep_plot, *popt), 'b-', lw=2, label=f'Voce R²={r2:.3f}')
            except: pass
            
            # Modified Voce
            try:
                popt, _ = curve_fit(modified_voce, ep_f, ts_f, p0=[250,100,20,100,2],
                                    bounds=([50,1,0.1,1,0.01],[1000,500,200,500,50]), maxfev=20000)
                r2 = 1 - np.sum((ts_f-modified_voce(ep_f,*popt))**2)/np.sum((ts_f-np.mean(ts_f))**2)
                ax.plot(ep_plot, modified_voce(ep_plot, *popt), 'g--', lw=2, label=f'Mod.Voce R²={r2:.3f}')
            except: pass
            
            ax.set_title(f'{int(d)}°-Spec.{s}', fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Plastic Strain')
            ax.set_ylabel('True Stress (MPa)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_per_specimen_fits.png'), dpi=300)
    plt.close()
    print("  Saved: fig_per_specimen_fits.png")
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
