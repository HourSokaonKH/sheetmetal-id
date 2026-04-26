%% zone_convergence_study.m
%  Zone Convergence Study — R-Value Sensitivity to Zone Count
%
%  Investigates how the number of DIC extraction zones affects the
%  computed Lankford r-value. Tests: 2, 4, 6, 8, 12, 16 zones.
%
%  This validates the 8-zone configuration used in the main analysis.
%  A converged r-value with increasing zone count confirms that the
%  spatial sampling is sufficient.
%
%  Input:  UFreckles .res files from raw_data/
%  Output: Convergence plots, tabular summary
%
%  Author: PhD Candidate
%  Date:   2026

clear; clc; close all;

%% ============== CONFIGURATION ==============
base_dir = fullfile(pwd, 'raw_data');
output_dir = fullfile(pwd, 'output');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

% Zone counts to test (columns × rows arrangement)
zone_configs = struct( ...
    'nz', {2, 4, 6, 8, 12, 16}, ...
    'ncols', {1, 2, 2, 2, 3, 4}, ...
    'nrows', {2, 2, 3, 4, 4, 4} );

% All specimens
specimens = { ...
    '00-01', '00-02', '00-03', ...
    '45-01', '45-02', '45-03', ...
    '90-01', '90-02', '90-03'};
directions = {'00', '45', '90'};

% R-value regression range
eyy_min = 0.02;
eyy_max = 0.10;

% Quality thresholds
R2_THRESHOLD = 0.98;
CV_THRESHOLD = 0.15;

% Gauge region (fraction of ROI to use as gauge area)
% Central 60% width, middle 80% height — avoids grips/edges
GAUGE_X_FRAC = [0.20, 0.80];  % fraction of ROI width
GAUGE_Y_FRAC = [0.10, 0.90];  % fraction of ROI height

%% ============== PRECOMPUTE ELEMENT STRAINS ==============
% Load and compute strains for all specimens once
fprintf('Loading and computing element-level strains...\n');
specimen_data = struct();

for si = 1:length(specimens)
    sname = specimens{si};
    data_dir = fullfile(base_dir, sname);
    res_file = fullfile(data_dir, [sname, '.res']);
    
    if ~exist(res_file, 'file')
        warning('Result file not found: %s. Skipping.', res_file);
        continue;
    end
    
    fprintf('  Loading %s... ', sname);
    full = load(res_file, '-mat');
    
    Nnod = length(full.xo);
    Nframes = size(full.U, 2);
    
    Ux = full.U(1:Nnod, :);
    Uy = full.U(Nnod+1:end, :);
    
    % ROI offset
    roi = full.param.roi;
    x_offset = roi(1);
    y_offset = roi(3);
    
    % Element centers (in mesh coordinates)
    conn = full.conn;
    Nelt = size(conn, 1);
    xc = mean(full.xo(conn), 2);
    yc = mean(full.yo(conn), 2);
    
    % Compute strains at element centers
    dNdxi  = [-1, 1, 1, -1] / 4;
    dNdeta = [-1, -1, 1, 1] / 4;
    
    Exx = zeros(Nelt, Nframes);
    Eyy = zeros(Nelt, Nframes);
    Exy = zeros(Nelt, Nframes);
    
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
        
        Exx(e, :) = dNdx * ux_e;
        Eyy(e, :) = dNdy * uy_e;
        Exy(e, :) = 0.5 * (dNdy * ux_e + dNdx * uy_e);
    end
    
    % Gauge region bounds
    xmin_g = min(xc) + (max(xc) - min(xc)) * GAUGE_X_FRAC(1);
    xmax_g = min(xc) + (max(xc) - min(xc)) * GAUGE_X_FRAC(2);
    ymin_g = min(yc) + (max(yc) - min(yc)) * GAUGE_Y_FRAC(1);
    ymax_g = min(yc) + (max(yc) - min(yc)) * GAUGE_Y_FRAC(2);
    
    specimen_data(si).name = sname;
    specimen_data(si).xc = xc;
    specimen_data(si).yc = yc;
    specimen_data(si).Exx = Exx;
    specimen_data(si).Eyy = Eyy;
    specimen_data(si).Exy = Exy;
    specimen_data(si).Nframes = Nframes;
    specimen_data(si).gauge = [xmin_g, xmax_g, ymin_g, ymax_g];
    
    fprintf('%d elements, %d frames\n', Nelt, Nframes);
end

Nspec = length(specimens);

%% ============== RUN CONVERGENCE STUDY ==============
fprintf('\n========================================\n');
fprintf('  ZONE CONVERGENCE STUDY\n');
fprintf('========================================\n');

Nconfigs = length(zone_configs);
% Results: r_values(specimen, config), n_good(specimen, config)
r_values = nan(Nspec, Nconfigs);
r_stds   = nan(Nspec, Nconfigs);
n_good   = zeros(Nspec, Nconfigs);
n_total  = zeros(Nspec, Nconfigs);

for ci = 1:Nconfigs
    nz = zone_configs(ci).nz;
    ncols = zone_configs(ci).ncols;
    nrows = zone_configs(ci).nrows;
    
    fprintf('\n--- Config: %d zones (%dx%d) ---\n', nz, ncols, nrows);
    
    for si = 1:Nspec
        if ~isfield(specimen_data, 'name') || isempty(specimen_data(si).name)
            continue;
        end
        
        xc = specimen_data(si).xc;
        yc = specimen_data(si).yc;
        Exx = specimen_data(si).Exx;
        Eyy = specimen_data(si).Eyy;
        gauge = specimen_data(si).gauge;
        Nframes = specimen_data(si).Nframes;
        
        % Divide gauge region into ncols × nrows zones
        x_edges = linspace(gauge(1), gauge(2), ncols + 1);
        y_edges = linspace(gauge(3), gauge(4), nrows + 1);
        
        zone_r = [];
        n_good_z = 0;
        
        for ic = 1:ncols
            for ir = 1:nrows
                mask = (xc >= x_edges(ic)) & (xc < x_edges(ic+1)) & ...
                       (yc >= y_edges(ir)) & (yc < y_edges(ir+1));
                n_elts = sum(mask);
                
                if n_elts < 5, continue; end
                
                % Zone-averaged strains
                eyy_z = mean(Eyy(mask, :), 1);
                exx_z = mean(Exx(mask, :), 1);
                eyy_std_z = std(Eyy(mask, :), [], 1);
                
                % Select uniform deformation range
                valid = eyy_z > eyy_min & eyy_z < eyy_max;
                n_pts = sum(valid);
                if n_pts < 10, continue; end
                
                eyy_v = eyy_z(valid);
                exx_v = exx_z(valid);
                
                % Linear regression
                p = polyfit(eyy_v, exx_v, 1);
                slope = p(1);
                
                exx_pred = polyval(p, eyy_v);
                SS_res = sum((exx_v - exx_pred).^2);
                SS_tot = sum((exx_v - mean(exx_v)).^2);
                if SS_tot == 0, continue; end
                R2 = 1 - SS_res / SS_tot;
                
                r_val = -slope / (1 + slope);
                
                cv = mean(eyy_std_z(valid) ./ max(eyy_z(valid), 1e-10));
                
                if R2 > R2_THRESHOLD && cv < CV_THRESHOLD
                    zone_r(end+1) = r_val; %#ok<SAGROW>
                    n_good_z = n_good_z + 1;
                end
            end
        end
        
        n_total(si, ci) = nz;
        n_good(si, ci) = n_good_z;
        
        if ~isempty(zone_r)
            r_values(si, ci) = mean(zone_r);
            r_stds(si, ci) = std(zone_r);
        end
        
        fprintf('  %s: r=%.4f (%d/%d GOOD)\n', ...
            specimen_data(si).name, r_values(si,ci), n_good_z, nz);
    end
end

%% ============== COMPUTE DIRECTION AVERAGES ==============
dir_idx = struct('d00', 1:3, 'd45', 4:6, 'd90', 7:9);
dir_names = {'0°', '45°', '90°'};
dir_fields = {'d00', 'd45', 'd90'};

dir_r = nan(3, Nconfigs);      % direction × config
dir_r_std = nan(3, Nconfigs);

for di = 1:3
    idx = dir_idx.(dir_fields{di});
    for ci = 1:Nconfigs
        vals = r_values(idx, ci);
        vals = vals(~isnan(vals));
        if ~isempty(vals)
            % Weight by number of good zones
            wts = n_good(idx, ci);
            wts = wts(~isnan(r_values(idx, ci)));
            if sum(wts) > 0
                dir_r(di, ci) = sum(vals .* wts) / sum(wts);
            else
                dir_r(di, ci) = mean(vals);
            end
            dir_r_std(di, ci) = std(vals);
        end
    end
end

%% ============== CONVERGENCE PLOTS ==============
nz_list = [zone_configs.nz];
colors3 = {'b', 'r', 'g'};
markers = {'o', 's', '^'};

% Figure 1: Per-direction r-value convergence
figure('Position', [100, 100, 800, 500]);
hold on;
for di = 1:3
    errorbar(nz_list, dir_r(di,:), dir_r_std(di,:), ['-', markers{di}], ...
        'Color', colors3{di}, 'MarkerFaceColor', colors3{di}, ...
        'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', dir_names{di});
end
xlabel('Number of Zones');
ylabel('Weighted Mean r-Value');
title('Lankford r-Value Convergence with Zone Count');
legend('Location', 'best');
grid on;
set(gca, 'XTick', nz_list);
hold off;
saveas(gcf, fullfile(output_dir, 'fig_zone_convergence_r_value.png'));
fprintf('\nSaved: fig_zone_convergence_r_value.png\n');

% Figure 2: Normal and planar anisotropy convergence
r_bar = (dir_r(1,:) + 2*dir_r(2,:) + dir_r(3,:)) / 4;
delta_r = (dir_r(1,:) - 2*dir_r(2,:) + dir_r(3,:)) / 2;

figure('Position', [100, 100, 800, 500]);
subplot(1,2,1);
plot(nz_list, r_bar, 'ko-', 'LineWidth', 2, 'MarkerFaceColor', 'k', 'MarkerSize', 8);
xlabel('Number of Zones');
ylabel('Normal Anisotropy r-bar');
title('Normal Anisotropy Convergence');
grid on;
set(gca, 'XTick', nz_list);

subplot(1,2,2);
plot(nz_list, delta_r, 'ko-', 'LineWidth', 2, 'MarkerFaceColor', 'k', 'MarkerSize', 8);
xlabel('Number of Zones');
ylabel('Planar Anisotropy Dr');
title('Planar Anisotropy Convergence');
grid on;
set(gca, 'XTick', nz_list);
sgtitle('Anisotropy Parameter Convergence');
saveas(gcf, fullfile(output_dir, 'fig_zone_convergence_anisotropy.png'));
fprintf('Saved: fig_zone_convergence_anisotropy.png\n');

% Figure 3: Good zone fraction vs zone count
figure('Position', [100, 100, 800, 500]);
hold on;
for di = 1:3
    idx = dir_idx.(dir_fields{di});
    frac = mean(n_good(idx, :) ./ n_total(idx, :), 1);
    plot(nz_list, frac * 100, ['-', markers{di}], ...
        'Color', colors3{di}, 'MarkerFaceColor', colors3{di}, ...
        'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', dir_names{di});
end
xlabel('Number of Zones');
ylabel('Good Zone Fraction (%)');
title('Quality Pass Rate vs Zone Count');
legend('Location', 'best');
grid on;
ylim([0, 105]);
set(gca, 'XTick', nz_list);
hold off;
saveas(gcf, fullfile(output_dir, 'fig_zone_convergence_quality.png'));
fprintf('Saved: fig_zone_convergence_quality.png\n');

% Figure 4: Per-specimen convergence (detailed)
figure('Position', [100, 100, 1200, 800]);
spec_colors = lines(9);
for si = 1:Nspec
    if ~isfield(specimen_data, 'name') || isempty(specimen_data(si).name)
        continue;
    end
    subplot(3, 3, si);
    plot(nz_list, r_values(si, :), 'ko-', 'LineWidth', 1.5, ...
        'MarkerFaceColor', spec_colors(si,:), 'MarkerSize', 6);
    if ~all(isnan(r_stds(si,:)))
        hold on;
        errorbar(nz_list, r_values(si,:), r_stds(si,:), 'k.', 'LineWidth', 1);
        hold off;
    end
    xlabel('Zones');
    ylabel('r');
    title(specimen_data(si).name);
    grid on;
    set(gca, 'XTick', nz_list);
end
sgtitle('Per-Specimen r-Value Convergence', 'FontSize', 14);
saveas(gcf, fullfile(output_dir, 'fig_zone_convergence_per_specimen.png'));
fprintf('Saved: fig_zone_convergence_per_specimen.png\n');

%% ============== SUMMARY TABLE ==============
fprintf('\n========================================\n');
fprintf('  CONVERGENCE SUMMARY\n');
fprintf('========================================\n\n');

fprintf('%-10s', 'Zones');
for ci = 1:Nconfigs
    fprintf('  %8d', zone_configs(ci).nz);
end
fprintf('\n%s\n', repmat('-', 1, 10 + Nconfigs*10));

for di = 1:3
    fprintf('%-10s', ['r_', dir_names{di}]);
    for ci = 1:Nconfigs
        fprintf('  %8.4f', dir_r(di, ci));
    end
    fprintf('\n');
end

fprintf('%-10s', 'r_bar');
for ci = 1:Nconfigs
    fprintf('  %8.4f', r_bar(ci));
end
fprintf('\n');

fprintf('%-10s', 'delta_r');
for ci = 1:Nconfigs
    fprintf('  %8.4f', delta_r(ci));
end
fprintf('\n');

% Check convergence: |r(N) - r(N-1)| / r(N) < threshold
fprintf('\n--- Relative change from previous zone count ---\n');
for di = 1:3
    fprintf('%s: ', dir_names{di});
    for ci = 2:Nconfigs
        if ~isnan(dir_r(di,ci)) && ~isnan(dir_r(di,ci-1)) && dir_r(di,ci) ~= 0
            rel_change = abs(dir_r(di,ci) - dir_r(di,ci-1)) / abs(dir_r(di,ci)) * 100;
            fprintf('  %.2f%%', rel_change);
        else
            fprintf('  N/A   ');
        end
    end
    fprintf('\n');
end

fprintf('\n✓ If change < 2%% between consecutive configs, the zone count is sufficient.\n');
fprintf('✓ The 8-zone configuration is validated if 6→8 and 8→12 changes are small.\n');

fprintf('\nDone.\n');
