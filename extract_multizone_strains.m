%% extract_multizone_strains.m
%  Extract Exx, Eyy, Exy from UFreckles .res files for multiple zones
%  No GUI needed — works directly from displacement field
%
%  Usage:
%    1. Set specimen_name and data_dir below
%    2. Define zones as [xmin, xmax, ymin, ymax] in pixel coordinates
%       OR load zone definitions from existing gage files
%    3. Run script — outputs CSV files with strain histories
%
%  Author: Auto-generated for PhD research
%  Date: 2026-04-18

clear; clc; close all;

%% ============== USER SETTINGS ==============
specimen_name = '45-02';
data_dir = fullfile(pwd, 'raw_data', specimen_name);

% Load the full DIC result
res_file = fullfile(data_dir, [specimen_name, '.res']);
if ~isfile(res_file)
    error('Result file not found: %s', res_file);
end
fprintf('Loading %s...\n', res_file);
full = load(res_file, '-mat');

Nnod = length(full.xo);
Nframes = size(full.U, 2);
fprintf('Nodes: %d, Elements: %d, Frames: %d\n', Nnod, size(full.conn,1), Nframes);

% Extract blocked DOFs
Ux = full.U(1:Nnod, :);       % [Nnodes x Nframes]
Uy = full.U(Nnod+1:end, :);   % [Nnodes x Nframes]

%% ============== DEFINE ZONES ==============
% Load zone definitions from existing gage files
% Gage format: 5x2 closed polygon [x, y] vertices in IMAGE pixel coords
% Must convert to mesh coords: mesh = pixel - roi_origin
roi = full.param.roi;  % [xmin, xmax, ymin, ymax] in image pixels
x_offset = roi(1);     % 405
y_offset = roi(3);     % 725
fprintf('ROI offset: x=%d, y=%d\n', x_offset, y_offset);

gage_files = dir(fullfile(data_dir, [specimen_name, '-gage-*.res']));
if ~isempty(gage_files)
    fprintf('\nFound %d existing gage files. Loading zone definitions...\n', length(gage_files));
    zones = zeros(length(gage_files), 4);  % [xmin, xmax, ymin, ymax] in MESH coords
    for k = 1:length(gage_files)
        g = load(fullfile(data_dir, gage_files(k).name), '-mat');
        % gage is 5x2 polygon: columns = [x, y] in IMAGE pixels
        xmin_z = min(g.gage(:,1)) - x_offset;
        xmax_z = max(g.gage(:,1)) - x_offset;
        ymin_z = min(g.gage(:,2)) - y_offset;
        ymax_z = max(g.gage(:,2)) - y_offset;
        zones(k,:) = [xmin_z, xmax_z, ymin_z, ymax_z];
        fprintf('  Zone %d: x=[%.0f, %.0f], y=[%.0f, %.0f] (mesh coords)\n', k, xmin_z, xmax_z, ymin_z, ymax_z);
    end
    Nzones = size(zones, 1);
else
    % Option B: Define zones manually [xmin, xmax, ymin, ymax] in PIXEL coords
    % These will be converted to mesh coords by subtracting ROI offset
    fprintf('\nNo gage files found. Using manual zone definitions.\n');
    zones_pixel = [
        420, 480, 300, 380;   % Zone 1: upper gauge
        420, 480, 400, 480;   % Zone 2: upper-mid
        420, 480, 500, 580;   % Zone 3: center
        420, 480, 600, 680;   % Zone 4: lower-mid
        420, 480, 700, 780;   % Zone 5: lower gauge
    ];
    % Convert pixel coords to mesh coords
    zones = zones_pixel;
    zones(:, [1 2]) = zones_pixel(:, [1 2]) - x_offset;
    zones(:, [3 4]) = zones_pixel(:, [3 4]) - y_offset;
    Nzones = size(zones, 1);
end

%% ============== COMPUTE ELEMENT CENTROIDS ==============
conn = full.conn;
Nelt = size(conn, 1);

% Element centroids
xc = mean(full.xo(conn), 2);
yc = mean(full.yo(conn), 2);

%% ============== STRAIN COMPUTATION (Q4 at element center) ==============
fprintf('\nComputing strains for all elements...\n');

% Preallocate strain arrays [Nelt x Nframes]
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
    
    % Jacobian at element center
    J11 = dNdxi  * xe;
    J12 = dNdxi  * ye;
    J21 = dNdeta * xe;
    J22 = dNdeta * ye;
    detJ = J11*J22 - J12*J21;
    
    % Shape function derivatives in physical coords
    dNdx = ( J22 * dNdxi - J12 * dNdeta) / detJ;
    dNdy = (-J21 * dNdxi + J11 * dNdeta) / detJ;
    
    % Displacement at element nodes [4 x Nframes]
    ux_e = Ux(nodes, :);
    uy_e = Uy(nodes, :);
    
    % Displacement gradients [1 x Nframes]
    duxdx = dNdx * ux_e;
    duxdy = dNdy * ux_e;
    duydx = dNdx * uy_e;
    duydy = dNdy * uy_e;
    
    % Small (linearized) strain
    Exx_all(e, :) = duxdx;
    Eyy_all(e, :) = duydy;
    Exy_all(e, :) = 0.5 * (duxdy + duydx);
end

fprintf('Strain computation complete.\n');

%% ============== EXTRACT ZONE AVERAGES ==============
fprintf('\nExtracting zone averages...\n');

% Results storage
results = struct();

for z = 1:Nzones
    % Zone boundaries [xmin, xmax, ymin, ymax]
    xmin = zones(z, 1); xmax = zones(z, 2);
    ymin = zones(z, 3); ymax = zones(z, 4);
    
    % Find elements inside zone
    mask = (xc >= xmin) & (xc <= xmax) & (yc >= ymin) & (yc <= ymax);
    n_elts = sum(mask);
    
    if n_elts == 0
        warning('Zone %d: no elements found in [%.0f,%.0f]x[%.0f,%.0f]. Skipping.', ...
            z, xmin, xmax, ymin, ymax);
        continue;
    end
    
    fprintf('  Zone %d: [%.0f,%.0f]x[%.0f,%.0f] -> %d elements\n', ...
        z, xmin, xmax, ymin, ymax, n_elts);
    
    % Average strain over zone elements
    results(z).Exx = mean(Exx_all(mask, :), 1);   % [1 x Nframes]
    results(z).Eyy = mean(Eyy_all(mask, :), 1);
    results(z).Exy = mean(Exy_all(mask, :), 1);
    results(z).n_elts = n_elts;
    results(z).zone = [xmin, xmax, ymin, ymax];
end

%% ============== EXPORT TO CSV ==============
output_dir = fullfile(data_dir, 'strain_export');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

for z = 1:length(results)
    if isempty(results(z).Exx), continue; end
    
    % Create table: Frame, Exx, Eyy, Exy
    T = table((1:Nframes)', results(z).Exx', results(z).Eyy', results(z).Exy', ...
        'VariableNames', {'Frame', 'Exx', 'Eyy', 'Exy'});
    
    fname = fullfile(output_dir, sprintf('%s-zone%02d-strains.csv', specimen_name, z));
    writetable(T, fname);
    fprintf('  Exported: %s\n', fname);
end

%% ============== ALSO EXPORT COMBINED FILE ==============
% All zones side by side
headers = {'Frame'};
data_matrix = (1:Nframes)';
for z = 1:length(results)
    if isempty(results(z).Exx), continue; end
    headers{end+1} = sprintf('Z%d_Exx', z);
    headers{end+1} = sprintf('Z%d_Eyy', z);
    headers{end+1} = sprintf('Z%d_Exy', z);
    data_matrix = [data_matrix, results(z).Exx', results(z).Eyy', results(z).Exy'];
end
T_all = array2table(data_matrix, 'VariableNames', headers);
fname_all = fullfile(output_dir, sprintf('%s-all-zones-strains.csv', specimen_name));
writetable(T_all, fname_all);
fprintf('\nCombined file: %s\n', fname_all);

%% ============== PLOT ==============
figure('Name', [specimen_name, ' Multi-Zone Strains'], 'Position', [100,100,1400,500]);

subplot(1,3,1); hold on;
for z = 1:length(results)
    if isempty(results(z).Exx), continue; end
    plot(results(z).Exx, 'DisplayName', sprintf('Zone %d', z));
end
xlabel('Frame'); ylabel('\epsilon_{xx}'); title('Exx (transverse)');
legend('Location', 'best'); grid on;

subplot(1,3,2); hold on;
for z = 1:length(results)
    if isempty(results(z).Eyy), continue; end
    plot(results(z).Eyy, 'DisplayName', sprintf('Zone %d', z));
end
xlabel('Frame'); ylabel('\epsilon_{yy}'); title('Eyy (longitudinal)');
legend('Location', 'best'); grid on;

subplot(1,3,3); hold on;
for z = 1:length(results)
    if isempty(results(z).Exy), continue; end
    plot(results(z).Exy, 'DisplayName', sprintf('Zone %d', z));
end
xlabel('Frame'); ylabel('\epsilon_{xy}'); title('Exy (shear)');
legend('Location', 'best'); grid on;

sgtitle(sprintf('%s — Multi-Zone Strain Extraction (%d zones)', specimen_name, length(results)));

%% ============== R-VALUE FROM ZONE DATA ==============
figure('Name', [specimen_name, ' R-value per zone'], 'Position', [100,650,800,400]);
hold on;
for z = 1:length(results)
    if isempty(results(z).Exx), continue; end
    % r = -Exx / (Exx + Eyy) = -Exx / Ezz_approx
    Ezz = results(z).Exx + results(z).Eyy;  % through-thickness (volume conservation)
    r_inst = -results(z).Exx ./ Ezz;
    
    % Only plot where strains are significant (avoid noise at start)
    valid = abs(results(z).Eyy) > 0.005;
    plot(results(z).Eyy(valid), r_inst(valid), 'DisplayName', sprintf('Zone %d', z));
end
xlabel('\epsilon_{yy}'); ylabel('r-value');
title(sprintf('%s — Instantaneous R-value vs Longitudinal Strain', specimen_name));
legend('Location', 'best'); grid on;

fprintf('\n=== Done! ===\n');
