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

T = readtable(fullfile(path,filename) + ".dat", ...
    'ReadVariableNames', false);

%% Extract columns

device        = string(T{:,1});
log2K         = T{:,2};
no_of_cols    = T{:,3};
runtime_impl  = T{:,4};
runtime_base1 = T{:,5};
runtime_base2 = T{:,6};

%% Parameters

devices = unique(device, 'stable');
log2K_values = 7:12;

% Number of columns / configurations
x_values = [0 1 2 4 8];
x_labels = {'Single-core vs naive DGEMM' 'Single-core', '1', '2', '4', '8'};

colors = [
    0.60 0.60 0.60;   % Single-core vs plain DGEMM single-core
    0.30 0.30 0.30;   % Single-core
    0.20 0.50 0.80;   % 1 columns
    0.20 0.70 0.40;   % 2 columns
    0.90 0.50 0.10;   % 4 columns
    0.70 0.25 0.70    % 8 columns
];

%% Create one figure per device

for d = 1:length(devices)

    current_device = devices(d);

    fig = figure('Color', 'w');
    fig.WindowState = 'maximized';

    hold on;
    grid on;
    box on;

    %% Build speedup matrix
    %
    % Rows    = matrix sizes
    % Columns = number of columns / configurations
    %
    % speedup_matrix(k,c) =
    %   speedup for K = 2^log2K_values(k)
    %   and configuration c

    speedup_matrix = nan(length(log2K_values), length(x_values));
    speedup_matrix2 = nan(length(log2K_values), 1);

    for k = 1:length(log2K_values)

        current_log2K = log2K_values(k);

        for c = 1:length(x_values)

            current_cols = x_values(c);

            idx = device == current_device & ...
                  log2K == current_log2K & ...
                  no_of_cols == current_cols;

            
            idx_base = device == current_device & log2K == current_log2K;

            if any(idx)

                % Speedup = baseline2 runtime / implementation runtime
                speedup_matrix(k,c) = ...
                    runtime_base2(find(idx,1)) / ...
                    runtime_impl(find(idx,1));
            end

            if (current_cols==0)
                idx = device == current_device & ...
                      log2K == current_log2K & ...
                      no_of_cols == current_cols;

                if any(idx)
                    % Speedup = baseline1 runtime / implementation runtime
                        speedup_matrix2(k,1) = ...
                            runtime_base1(find(idx,1)) / ...
                            runtime_impl(find(idx,1));
                end
            end

        end

    end

    %% Plot grouped bars
    %
    % Each group = one matrix size
    % Each bar   = one number-of-columns configuration
    speedup_matrix = [speedup_matrix2, speedup_matrix];

    b = bar(speedup_matrix, 'grouped');

    % Apply colors according to number of columns
    for c = 1:size(speedup_matrix, 2)
        b(c).FaceColor = colors(c,:);
        b(c).EdgeColor = 'none';
    end

    %% Formatting

    xlabel('$K$ : matmul($K$, $K$, $K$)', ...
        'Interpreter', 'latex','Interpreter', 'latex', 'FontSize', 12);
    ylabel('Speedup (base-2)', 'FontSize', 12);

    title(sprintf('Speedup - %s', deviceMap(current_device)), ...
        'FontSize', 14);

    % Matrix size labels
    xticks(1:length(log2K_values));

    xticklabels(arrayfun( ...
        @(k) sprintf('$2^{%d}$', k), ...
        log2K_values, ...
        'UniformOutput', false));

    set(gca, 'TickLabelInterpreter', 'latex');

    xlim([0.5 length(log2K_values)+0.5]);

   %% Log2 Y-axis

    set(gca, 'YScale', 'log');
    
    % Collect valid speedup values
    valid_speedups = speedup_matrix( ...
        isfinite(speedup_matrix) & speedup_matrix > 0);
    
    % Determine power-of-two range
    y_min_exp = floor(log2(min(valid_speedups)));
    y_max_exp = 2;
    
    %% Major ticks: powers of 2
    
    exponents = y_min_exp:y_max_exp;
    y_ticks = 2.^exponents;
    
    yticks(y_ticks);
    
    yticklabels(arrayfun( ...
        @(e) sprintf('$2^{%d}$', e), ...
        exponents, ...
        'UniformOutput', false));
    
    set(gca, 'TickLabelInterpreter', 'latex');
    
    %% Minor ticks: one line halfway between each power of 2
    
    % One minor tick at 2^(n + 1/2) = sqrt(2) * 2^n
    minor_ticks = 2.^((y_min_exp:y_max_exp-1) + 0.5);
    
    ax = gca;
    
    ax.YMinorTick = 'on';
    ax.YAxis.MinorTickValues = minor_ticks;
    ax.YMinorGrid = 'on';
    
    % Optional: make minor grid lines lighter than major grid lines
    ax.GridAlpha = 0.25;
    ax.MinorGridAlpha = 0.15;
    
    %% Axis limits
    
    ylim([2^y_min_exp, 2^y_max_exp]);
    
    %% 1x baseline
    
    yline(1, '--k', '1x baseline', ...
        'LineWidth', 1.2, ...
        'LabelHorizontalAlignment', 'left');


    %% Legend

    legend_labels = string(x_labels);
    legend_labels(3:end) = legend_labels(3:end) + " cols";

    lgd = legend(b, legend_labels, ...
        'Location', 'northeast');

    lgd.FontSize = 14;
    lgd.Box = 'on';
    lgd.ItemTokenSize = [10, 8];

    set(gca, 'FontSize', 14);

    %% Save as vector PDF

    exportgraphics(gcf, ...
        fullfile(plotsPath, ...
        filename + "_speedup_" + current_device + ".pdf"), ...
        'ContentType', 'vector', ...
        'BackgroundColor', 'white');

    exportgraphics(gcf, ...
        fullfile(plotsPath, ...
        filename + "_speedup_" + current_device + ".png"), ...
        'ContentType', 'vector', ...
        'BackgroundColor', 'white');

    hold off;

end
