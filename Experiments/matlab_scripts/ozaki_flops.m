%% GFLOPS plotting using column indices
%
% Column format:
%   Column 1 = device
%   Column 2 = log2(K)
%   Column 3 = no_of_cols
%   Column 4 = runtime_impl
%   Column 5 = runtime_baseline1   <-- ignored
%   Column 6 = runtime_baseline2
%
% no_of_cols:
%   -1 = single-core
%    2 = 2-core
%    4 = 4-core
%    8 = 8-core
%
% The -1 value is plotted as 0 on the x-axis.

clear;
clc;
close all;

path = "..\measurements\";
plotsPath = "..\plots\";
filename = 'ozaki_perf_tests';

deviceMap = containers.Map( ...
    ["npu1", "npu2"], ...
    ["XDNA1", "XDNA2"]);

%% Load data
% Replace with your actual filename
T = readtable(fullfile(path,filename) + ".dat", 'ReadVariableNames', false);

%% Extract columns by index

device       = string(T{:,1});
log2K        = T{:,2};
no_of_cols   = T{:,3};
runtime_impl  = T{:,4};
runtime_base2 = T{:,6};

%% Parameters

devices = unique(device, 'stable');
log2K_values = 7:12;

% Plot -1 (single-core) as 0
x_values = [0 1 2 4 8];

% One color for each matrix size
colors = lines(length(log2K_values));

%% Create one figure per device

for d = 1:length(devices)

    current_device = devices(d);
    all_gflops = [];

    fig = figure('Color', 'w');
    fig.WindowState = 'maximized';
    hold on;
    grid on;
    box on;

    for k = 1:length(log2K_values)

        current_log2K = log2K_values(k);

        %% Select data for this device and matrix size

        idx = device == current_device & ...
              log2K == current_log2K;

        % Extract selected data
        x = no_of_cols(idx);
        
        K = 2^current_log2K;
        
        % Runtime is in microseconds
        y_impl  = (2 * K^3) ./ (1e3 * runtime_impl(idx));
        y_base2 = (2 * K^3) ./ (1e3 * runtime_base2(idx));
        
        % Store GFLOPS
        all_gflops = [all_gflops; y_impl; y_base2];
        
        % Map configurations to equally spaced x positions
        x_plot = zeros(size(x));
        
        x_plot(x ==  0) = 0;   % Single-core
        x_plot(x ==  1) = 2;
        x_plot(x ==  2) = 3;
        x_plot(x ==  4) = 4;
        x_plot(x ==  8) = 5;
        
        % Sort by configuration
        [x_plot, order] = sort(x_plot);
        
        y_impl  = y_impl(order);
        y_base2 = y_base2(order);
        
        % Plot implementation
        plot(x_plot, y_impl, ...
            '-o', ...
            'Color', colors(k,:), ...
            'LineWidth', 1.8, ...
            'MarkerSize', 6, ...
            'DisplayName', sprintf('Oz-AIE, K = 2^{%d}', current_log2K));
        
        % Plot baseline 2
        % plot(x_plot, y_base2, ...
        %     '--s', ...
        %     'Color', colors(k,:), ...
        %     'LineWidth', 1.5, ...
        %     'MarkerSize', 5, ...
        %     'DisplayName', sprintf('BLIS, K = 2^{%d}', current_log2K));



    end

    %% Formatting

    xlabel('# of columns of AIE array', 'FontSize', 12);
    ylabel('GFLOPS', 'FontSize', 12);

    title(sprintf('GFLOPS - %s', deviceMap(current_device)), ...
        'FontSize', 14);

    xlim([-0.25 5.25]);

    labels = {'Single-core','', '1', '2', '4', '8'};
    xticklabels(labels);

    if deviceMap(current_device)=="XDNA1"
        xticks(0:4);
    else
        xticks(0:5);
    end    
    

    xlim([0 6.25]);

    lgd = legend('Location', 'southeast');
    lgd.FontSize = 12;
    lgd.Box = 'on';
    lgd.ItemTokenSize = [10, 8];
    %lgd.Position(2) = lgd.Position(2) - 0.1;

    set(gca, 'FontSize', 12);

    
    %% Logarithmic Y-axis

    % set(gca, 'YScale', 'log');
    % 
    % % Remove invalid values
    % valid = isfinite(all_gflops) & all_gflops > 0;
    % all_gflops = all_gflops(valid);
    % 
    % % Determine range from actual GFLOPS values
    % y_min = 10^floor(log10(min(all_gflops)));
    % y_max = 10^ceil(log10(max(all_gflops)));
    % 
    % ylim([y_min y_max]);
    % 
    % % Major ticks at powers of 10
    % yticks(10.^(floor(log10(min(all_gflops))) : ...
    %         ceil(log10(max(all_gflops)))));

    ylim([0 16]);

    hold off;

    %% ============================================================
    %  Save as vector PDF for LaTeX
    % =============================================================

    exportgraphics(gcf, ...
        fullfile(plotsPath, filename + "_" + current_device + ".pdf"), ...
        'ContentType', 'vector', ...
        'BackgroundColor', 'white');

    exportgraphics(gcf, ...
        fullfile(plotsPath, filename + "_" + current_device + ".png"), ...
        'ContentType', 'vector', ...
        'BackgroundColor', 'white');


end

