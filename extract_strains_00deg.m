%% extract_strains_00deg.m
%  Multi-Zone Strain Extraction and R-Value Analysis — 0° Direction
%  Specimens: 00-01, 00-02, 00-03
%
%  This script extracts Exx, Eyy, Exy from UFreckles DIC results
%  for multiple rectangular gauge zones and computes the Lankford
%  r-value (r₀) from the slope of transverse vs. longitudinal strain.
%
%  Reference: Lankford, Snyder & Bauscher (1950)
%             ISO 10113:2020
%
%  Input:  UFreckles result files (.res) for each specimen
%  Output: CSV strain histories, r-value summary, plots

clear; clc; close all;

%% ============== CONFIGURATION ==============
direction = '00';
specimens = {'00-01', '00-02', '00-03'};
base_dir = fullfile(pwd, 'raw_data');

% Strain range for r-value regression [min, max] (small/linearized strain)
% Use uniform deformation range — before necking / after initial elastic
eyy_min = 0.02;
eyy_max = 0.10;

% Summary storage
r_summary = [];

%% ============== PROCESS EACH SPECIMEN ==============
for s = 1:length(specimens)
    specimen_name = specimens{s};
    data_dir = fullfile(base_dir, specimen_name);
    res_file = fullfile(data_dir, [specimen_name, '.res']);
    
    fprintf('\n========================================\n');
    fprintf('  Processing: %s (0° direction)\n', specimen_name);
    fprintf('========================================\n');
    
    % --- Check files exist ---
    if ~exist(res_file, 'file')
        warning('Result file not found: %s. Skipping.', res_file);
        continue;
    end
    
    % --- Load full displacement field ---
    fprintf('Loading %s...\n', res_file);
    full = load(res_file, '-mat');
    
    Nnod = length(full.xo);
    Nframes = size(full.U, 2);
    fprintf('  Nodes: %d, Elements: %d, Frames: %d\n', Nnod, size(full.conn,1), Nframes);
    
    % Blocked DOF layout: U = [Ux_all; Uy_all]
    Ux = full.U(1:Nnod, :);
    Uy = full.U(Nnod+1:end, :);
    
    % ROI offset (image pixels -> mesh coordinates)
    roi = full.param.roi;
    x_offset = roi(1);
    y_offset = roi(3);
    fprintf('  ROI offset: x=%d, y=%d\n', x_offset, y_offset);
    
    % --- Load zone definitions from gage files ---
    gage_files = dir(fullfile(data_dir, [specimen_name, '-gage-*.res']));
    if isempty(gage_files)
        warning('No gage files found for %s. Skipping.', specimen_name);
        continue;
    end
    
    Nzones = length(gage_files);
    zones = zeros(Nzones, 4);
    fprintf('  Loading %d zone definitions...\n', Nzones);
    for k = 1:Nzones
        g = load(fullfile(data_dir, gage_files(k).name), '-mat');
        zones(k,:) = [min(g.gage(:,1)) - x_offset, max(g.gage(:,1)) - x_offset, ...
                       min(g.gage(:,2)) - y_offset, max(g.gage(:,2)) - y_offset];
        fprintf('    Zone %d: x=[%.0f, %.0f], y=[%.0f, %.0f]\n', ...
            k, zones(k,1), zones(k,2), zones(k,3), zones(k,4));
    end
    
    % --- Compute strains at all element centers ---
    conn = full.conn;
    Nelt = size(conn, 1);
    xc = mean(full.xo(conn), 2);
    yc = mean(full.yo(conn), 2);
    
    Exx_all = zeros(Nelt, Nframes);
    Eyy_all = zeros(Nelt, Nframes);
    Exy_all = zeros(Nelt, Nframes);
    
    % Q4 shape function derivatives at center (xi=0, eta=0)
    dNdxi  = [-1, 1, 1, -1] / 4;
    dNdeta = [-1, -1, 1, 1] / 4;
    
    for e = 1:Nelt
        nodes = conn(e, :);
        xe = full.xo(nodes);
        ye = full.yo(nodes);
        
        J11 = dNdxi * xe;  J12 = dNdxi * ye;
        J21 = dNdeta * xe; J22 = dNdeta * ye;
        detJ = J11*J22 - J12*J21;
        
        dNdx = ( J22 * dNdxi - J12 * dNdeta) / detJ;
        dNdy = (-J21 * dNdxi + J11 * dNdeta) / detJ;
        
        ux_e = Ux(nodes, :);
        uy_e = Uy(nodes, :);
        
        Exx_all(e, :) = dNdx * ux_e;
        Eyy_all(e, :) = dNdy * uy_e;
        Exy_all(e, :) = 0.5 * (dNdy * ux_e + dNdx * uy_e);
    end
    
    % --- Extract zone averages ---
    results = struct();
    for z = 1:Nzones
        mask = (xc >= zones(z,1)) & (xc <= zones(z,2)) & ...
               (yc >= zones(z,3)) & (yc <= zones(z,4));
        n_elts = sum(mask);
        
        if n_elts == 0
            warning('  Zone %d: no elements found. Skipping.', z);
            results(z).Exx = []; results(z).Eyy = []; results(z).Exy = [];
            results(z).n_elts = 0;
            continue;
        end
        
        results(z).Exx = mean(Exx_all(mask, :), 1);
        results(z).Eyy = mean(Eyy_all(mask, :), 1);
        results(z).Exy = mean(Exy_all(mask, :), 1);
        results(z).Eyy_std = std(Eyy_all(mask, :), [], 1);
        results(z).n_elts = n_elts;
        fprintf('    Zone %d: %d elements, Eyy_final=%.4f\n', z, n_elts, results(z).Eyy(end));
    end
    
    % --- Export CSV files ---
    output_dir = fullfile(data_dir, 'strain_export');
    if ~exist(output_dir, 'dir'), mkdir(output_dir); end
    
    % Per-zone CSV
    for z = 1:Nzones
        if isempty(results(z).Exx), continue; end
        T = table((1:Nframes)', results(z).Exx', results(z).Eyy', results(z).Exy', ...
            'VariableNames', {'Frame', 'Exx', 'Eyy', 'Exy'});
        fname = fullfile(output_dir, sprintf('%s-zone%02d-strains.csv', specimen_name, z));
        writetable(T, fname);
    end
    
    % Combined CSV
    headers = {'Frame'};
    data_matrix = (1:Nframes)';
    for z = 1:Nzones
        if isempty(results(z).Exx), continue; end
        headers = [headers, {sprintf('Z%d_Exx',z)}, {sprintf('Z%d_Eyy',z)}, {sprintf('Z%d_Exy',z)}];
        data_matrix = [data_matrix, results(z).Exx', results(z).Eyy', results(z).Exy'];
    end
    T_all = array2table(data_matrix, 'VariableNames', headers);
    writetable(T_all, fullfile(output_dir, sprintf('%s-all-zones-strains.csv', specimen_name)));
    fprintf('  CSV files exported to: %s\n', output_dir);
    
    % --- Compute r-value per zone (linear regression in uniform range) ---
    fprintf('\n  R-value analysis (Eyy range: [%.2f, %.2f]):\n', eyy_min, eyy_max);
    fprintf('  %-8s  %-10s  %-10s  %-10s  %-8s  %-10s  %-12s\n', ...
        'Zone', 'Slope', 'R²', 'r-value', 'N_pts', 'CV_Eyy', 'Quality');
    
    specimen_r = [];
    for z = 1:Nzones
        if isempty(results(z).Exx), continue; end
        
        valid = results(z).Eyy > eyy_min & results(z).Eyy < eyy_max;
        n_pts = sum(valid);
        if n_pts < 10
            fprintf('  Zone %d: insufficient data points (%d). Skipping.\n', z, n_pts);
            continue;
        end
        
        eyy_v = results(z).Eyy(valid);
        exx_v = results(z).Exx(valid);
        
        % Linear regression
        p = polyfit(eyy_v, exx_v, 1);
        slope = p(1);
        
        % R² goodness of fit
        exx_pred = polyval(p, eyy_v);
        SS_res = sum((exx_v - exx_pred).^2);
        SS_tot = sum((exx_v - mean(exx_v)).^2);
        R2 = 1 - SS_res / SS_tot;
        
        % Lankford r-value: r = -slope / (1 + slope)
        r_val = -slope / (1 + slope);
        
        % Strain uniformity: CV = std/mean of Eyy across elements in zone
        % High CV indicates strain localization (necking/fracture nearby)
        cv_eyy = mean(results(z).Eyy_std(valid) ./ results(z).Eyy(valid));
        
        % Quality: R² > 0.98 AND CV < 15% = uniform deformation
        if R2 > 0.98 && cv_eyy < 0.15
            quality = 'GOOD';
        elseif cv_eyy >= 0.15
            quality = 'LOCALIZED';
        else
            quality = 'LOW R2';
        end
        
        fprintf('  Zone %-2d   %-10.4f  %-10.4f  %-10.3f  %-8d  %5.1f%%      %-12s\n', ...
            z, slope, R2, r_val, n_pts, cv_eyy*100, quality);
        specimen_r = [specimen_r; z, slope, R2, r_val, n_pts, cv_eyy];
    end
    
    % Store specimen summary (GOOD zones: R² > 0.98 AND CV < 15%)
    if ~isempty(specimen_r)
        good = specimen_r(:, 3) > 0.98 & specimen_r(:, 6) < 0.15;
        if any(good)
            r_mean = mean(specimen_r(good, 4));
            r_std  = std(specimen_r(good, 4));
        else
            r_mean = mean(specimen_r(:, 4));
            r_std  = std(specimen_r(:, 4));
        end
        r_summary = [r_summary; s, r_mean, r_std, sum(good)];
        fprintf('\n  >>> %s: r₀ = %.3f ± %.3f (from %d reliable zones)\n', ...
            specimen_name, r_mean, r_std, sum(good));
    end
    
    % --- Plot per specimen ---
    figure('Name', sprintf('%s — 0° Strain Fields', specimen_name), ...
           'Position', [50+200*(s-1), 100, 500, 700]);
    
    % Eyy vs Frame
    subplot(3,1,1); hold on;
    for z = 1:Nzones
        if isempty(results(z).Eyy), continue; end
        plot(results(z).Eyy, 'DisplayName', sprintf('Z%d (%d elts)', z, results(z).n_elts));
    end
    xlabel('Frame'); ylabel('\epsilon_{yy}');
    title(sprintf('%s — Eyy (longitudinal)', specimen_name));
    legend('Location', 'best'); grid on;
    
    % Exx vs Eyy (for r-value)
    subplot(3,1,2); hold on;
    for z = 1:Nzones
        if isempty(results(z).Exx), continue; end
        plot(results(z).Eyy, results(z).Exx, 'DisplayName', sprintf('Z%d', z));
    end
    xlabel('\epsilon_{yy}'); ylabel('\epsilon_{xx}');
    title(sprintf('%s — Transverse vs Longitudinal Strain', specimen_name));
    xline([eyy_min, eyy_max], '--k', {'fit start', 'fit end'});
    legend('Location', 'best'); grid on;
    
    % Exy vs Frame
    subplot(3,1,3); hold on;
    for z = 1:Nzones
        if isempty(results(z).Exy), continue; end
        plot(results(z).Exy, 'DisplayName', sprintf('Z%d', z));
    end
    xlabel('Frame'); ylabel('\epsilon_{xy}');
    title(sprintf('%s — Exy (shear)', specimen_name));
    legend('Location', 'best'); grid on;
    
    sgtitle(sprintf('%s — 0° Direction Multi-Zone Analysis', specimen_name));
    
    % Clean up large variables for next specimen
    clear full Ux Uy Exx_all Eyy_all Exy_all results;
end

%% ============== DIRECTION SUMMARY ==============
fprintf('\n\n');
fprintf('============================================\n');
fprintf('  0° DIRECTION — R-VALUE SUMMARY (r₀)\n');
fprintf('============================================\n');
fprintf('  %-12s  %-10s  %-10s  %-12s\n', 'Specimen', 'r₀', '± std', 'Good zones');
for i = 1:size(r_summary, 1)
    fprintf('  %-12s  %-10.3f  %-10.3f  %-12d\n', ...
        specimens{r_summary(i,1)}, r_summary(i,2), r_summary(i,3), r_summary(i,4));
end

% --- Weighted pooling (weight = number of GOOD zones) ---
weights = r_summary(:, 4);
if sum(weights) == 0
    error('No GOOD zones found across all specimens. Cannot compute weighted r-value.');
end
r0_weighted = sum(r_summary(:, 2) .* weights) / sum(weights);

% --- IQR outlier rejection on per-specimen means ---
r_vals = r_summary(:, 2);
if length(r_vals) >= 3
    Q1 = quantile(r_vals, 0.25); Q3 = quantile(r_vals, 0.75);
    IQR = Q3 - Q1;
    inlier = r_vals >= (Q1 - 1.5*IQR) & r_vals <= (Q3 + 1.5*IQR);
else
    inlier = true(size(r_vals));
end
r0_robust = sum(r_vals(inlier) .* weights(inlier)) / sum(weights(inlier));
r0_std_robust = std(r_vals(inlier));
n_outliers = sum(~inlier);

fprintf('  ──────────────────────────────────────────\n');
fprintf('  Simple mean:    r₀ = %.3f ± %.3f\n', mean(r_vals), std(r_vals));
fprintf('  Weighted mean:  r₀ = %.3f  (by N_good zones)\n', r0_weighted);
if n_outliers > 0
    fprintf('  Robust (IQR):   r₀ = %.3f ± %.3f  (%d outlier removed)\n', ...
        r0_robust, r0_std_robust, n_outliers);
else
    fprintf('  Robust (IQR):   r₀ = %.3f ± %.3f  (no outliers)\n', ...
        r0_robust, r0_std_robust);
end
fprintf('============================================\n');

% Save for external use (e.g., compute_anisotropy.m)
r0_final = r0_robust;
save(fullfile(pwd, 'r0_result.mat'), 'r0_final', 'r_summary', 'r0_weighted', 'r0_robust');
fprintf('  Result saved to r0_result.mat\n');
