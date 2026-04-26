%% compute_anisotropy.m
%  Compute Normal and Planar Anisotropy from Directional r-Values
%
%  Run AFTER extract_strains_00deg.m, extract_strains_45deg.m,
%  and extract_strains_90deg.m, which save r0_result.mat, r45_result.mat,
%  and r90_result.mat respectively.
%
%  Computes:
%    r̄  = (r₀ + 2r₄₅ + r₉₀) / 4     — normal anisotropy
%    Δr = (r₀ - 2r₄₅ + r₉₀) / 2     — planar anisotropy
%
%  Reference: Hill (1948), ISO 10113:2020

clear; clc;

%% ============== LOAD RESULTS ==============
base = pwd;

if ~exist(fullfile(base, 'r0_result.mat'), 'file') || ...
   ~exist(fullfile(base, 'r45_result.mat'), 'file') || ...
   ~exist(fullfile(base, 'r90_result.mat'), 'file')
    error(['Missing result files. Run extract_strains_00deg.m, ' ...
           'extract_strains_45deg.m, and extract_strains_90deg.m first.']);
end

d0  = load(fullfile(base, 'r0_result.mat'));
d45 = load(fullfile(base, 'r45_result.mat'));
d90 = load(fullfile(base, 'r90_result.mat'));

r0  = d0.r0_final;
r45 = d45.r45_final;
r90 = d90.r90_final;

fprintf('============================================================\n');
fprintf('  PLASTIC ANISOTROPY SUMMARY — SGCC JIS G 3302 (1.5 mm)\n');
fprintf('============================================================\n\n');

%% ============== PER-DIRECTION RESULTS ==============
fprintf('  Direction     r-value   Method\n');
fprintf('  ─────────────────────────────────\n');
fprintf('  0°  (RD)      %.3f     Robust weighted\n', r0);
fprintf('  45°           %.3f     Robust weighted\n', r45);
fprintf('  90° (TD)      %.3f     Robust weighted\n', r90);

%% ============== NORMAL ANISOTROPY ==============
r_bar = (r0 + 2*r45 + r90) / 4;

fprintf('\n  ─────────────────────────────────\n');
fprintf('  Normal anisotropy:\n');
fprintf('    r̄ = (r₀ + 2r₄₅ + r₉₀) / 4\n');
fprintf('    r̄ = (%.3f + 2×%.3f + %.3f) / 4\n', r0, r45, r90);
fprintf('    r̄ = %.3f\n', r_bar);
fprintf('\n');

if r_bar > 1.0
    fprintf('    Interpretation: r̄ > 1 → Favorable for deep drawing\n');
    fprintf('    (sheet resists thinning — good formability)\n');
elseif r_bar > 0.7
    fprintf('    Interpretation: r̄ ≈ 0.7–1.0 → Moderate formability\n');
    fprintf('    (typical for commercial cold-rolled steel)\n');
else
    fprintf('    Interpretation: r̄ < 0.7 → Poor deep-drawing performance\n');
    fprintf('    (sheet prone to thinning)\n');
end

%% ============== PLANAR ANISOTROPY ==============
delta_r = (r0 - 2*r45 + r90) / 2;

fprintf('\n  ─────────────────────────────────\n');
fprintf('  Planar anisotropy:\n');
fprintf('    Δr = (r₀ − 2r₄₅ + r₉₀) / 2\n');
fprintf('    Δr = (%.3f − 2×%.3f + %.3f) / 2\n', r0, r45, r90);
fprintf('    Δr = %.3f\n', delta_r);
fprintf('\n');

if abs(delta_r) < 0.1
    fprintf('    Interpretation: |Δr| < 0.1 → Nearly isotropic in-plane\n');
    fprintf('    (minimal earing in cup drawing)\n');
elseif abs(delta_r) < 0.3
    fprintf('    Interpretation: |Δr| ≈ 0.1–0.3 → Moderate earing tendency\n');
elseif delta_r > 0
    fprintf('    Interpretation: Δr > 0.3 → Ears at 0°/90° in cup drawing\n');
else
    fprintf('    Interpretation: Δr < −0.3 → Ears at 45° in cup drawing\n');
end

%% ============== HILL48 PARAMETERS ==============
fprintf('\n  ─────────────────────────────────\n');
fprintf('  Hill''48 yield criterion parameters:\n');

% Hill48: F, G, H, N from r-values
% G + H = 1 (normalization to σ_RD)
% H/G = r0  →  H = r0/(1+r0),  G = 1/(1+r0)
% F/G = r0/r90  →  F = r0/((1+r0)*r90)
% N from r45: N = (r45+0.5)*(2*F+2*G) / (2*(1+r0)) ... simplified:
%   N = ((r0+r90)*(1+2*r45)) / (2*r90*(1+r0))

H = r0 / (1 + r0);
G = 1 / (1 + r0);
F = r0 / ((1 + r0) * r90);
N = ((r0 + r90) * (1 + 2*r45)) / (2 * r90 * (1 + r0));

fprintf('    F = %.4f\n', F);
fprintf('    G = %.4f\n', G);
fprintf('    H = %.4f\n', H);
fprintf('    N = %.4f\n', N);
fprintf('    (Normalized: G + H = 1)\n');

% Verification
fprintf('\n  Verification:\n');
fprintf('    r₀  = H/G = %.3f  (input: %.3f)\n', H/G, r0);
fprintf('    r₉₀ = H/F = %.3f  (input: %.3f)\n', H/F, r90);
r45_check = (2*N - F - G) / (2*(F + G));
fprintf('    r₄₅ = (2N−F−G)/(2(F+G)) = %.3f  (input: %.3f)\n', r45_check, r45);

%% ============== FINAL TABLE ==============
fprintf('\n\n');
fprintf('============================================================\n');
fprintf('  FINAL RESULTS TABLE (for thesis)\n');
fprintf('============================================================\n');
fprintf('  ┌──────────┬─────────┬──────────────────────────────────┐\n');
fprintf('  │ Parameter│  Value  │ Description                      │\n');
fprintf('  ├──────────┼─────────┼──────────────────────────────────┤\n');
fprintf('  │ r₀       │  %.3f  │ Lankford coeff. — rolling dir.   │\n', r0);
fprintf('  │ r₄₅      │  %.3f  │ Lankford coeff. — 45° to RD     │\n', r45);
fprintf('  │ r₉₀      │  %.3f  │ Lankford coeff. — transverse dir.│\n', r90);
fprintf('  │ r̄        │  %.3f  │ Normal anisotropy                │\n', r_bar);
fprintf('  │ Δr       │ %+.3f  │ Planar anisotropy                │\n', delta_r);
fprintf('  ├──────────┼─────────┼──────────────────────────────────┤\n');
fprintf('  │ F        │  %.4f │ Hill''48 parameter                 │\n', F);
fprintf('  │ G        │  %.4f │ Hill''48 parameter                 │\n', G);
fprintf('  │ H        │  %.4f │ Hill''48 parameter                 │\n', H);
fprintf('  │ N        │  %.4f │ Hill''48 parameter                 │\n', N);
fprintf('  └──────────┴─────────┴──────────────────────────────────┘\n');
fprintf('============================================================\n');

% Save final results
save(fullfile(base, 'anisotropy_results.mat'), ...
    'r0', 'r45', 'r90', 'r_bar', 'delta_r', 'F', 'G', 'H', 'N');
fprintf('\n  All results saved to anisotropy_results.mat\n');
