clear;
clc;
close all;

path = "..\measurements\";
plotsPath = "..\plots\";
filename = 'ozaki_perf_tests_pinned';

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
num_blis_threads = T{:,4};
npu_runtime   = T{:,5};
blis_runtime  = T{:,7};

%% Parameters

devices = unique(device, 'stable');
log2K_values = 7:12;

% Number of columns / configurations
x_values = [0 1 2 4];
x_labels_orig = {'Single-core', '1 cols', '2 cols', '4 cols'};

colors_base = [
    0.30 0.30 0.30;   % Single-core
    0.20 0.50 0.80;   % 1 columns
    0.20 0.70 0.40;   % 2 columns
    0.90 0.50 0.10;   % 4 columns   
    0.70 0.25 0.70    % 8 columns
];

%% Create one figure per device

for d = 1:length(devices)

    current_device = devices(d);

    if current_device == "npu1"
        max_cores = 8;
        max_cols = 4;
        bar_cols = [-1 0 1 2 4 4];
        x_labels = {'Single-core vs All-core BLIS', 'Single-core', '1 cols', '2 cols', ...
                    'Whole-array vs All-core BLIS', 'Whole-array vs Single-core BLIS'};
        colors = colors_base(1:end-1, :);
        colors = [ 
            0.6 0.6 0.6; 
            colors
        ];
        colors(end+1, :) = [0.9 0.0 0.0];
    else
        max_cores = 16;
        max_cols = 8;
        bar_cols = [-1 0 1 2 4 8 8];
        x_labels = {'Single-core vs All-core BLIS', 'Single-core', '1 cols', '2 cols', ...
                    '4 cols', 'Whole-array vs All-core BLIS', 'Whole-array vs Single-core BLIS'};
        colors = colors_base;
        colors = [ 
            0.6 0.6 0.6; 
            colors
        ];
        colors(end+1, :) = [0.9 0.0 0.0];
    end

    

    fig = figure('Color', 'w');
    fig.WindowState = 'maximized';

    hold on;
    grid on;
    box on;

    %% Build speedup matrix
    %
    % Rows    = matrix sizes
    % Columns = configurations
    %
    % speedup_matrix(k,c) =
    %   speedup for K = 2^log2K_values(k)
    %   and configuration c

    speedup_matrix = nan(length(log2K_values), length(bar_cols));

    for k = 1:length(log2K_values)

        current_log2K = log2K_values(k);

        % Select rows for this device and matrix size
        idx_group = device == current_device & ...
                    log2K == current_log2K;

        group_rows = find(idx_group);

        % Single bars for non-maximum column counts
        for c = 2:length(bar_cols)-2

            current_cols = bar_cols(c);

            idx = device == current_device & ...
                  log2K == current_log2K & ...
                  no_of_cols == current_cols;

            row_idx = find(idx);

            if ~isempty(row_idx)
                speedup_matrix(k,c) = ...
                    blis_runtime(row_idx(1)) / ...
                    npu_runtime(row_idx(1));
            end

        end

        % Two separate bars for the maximum column count
        idx_max = group_rows(no_of_cols(group_rows) == max_cols);

        [~, order] = sort(num_blis_threads(idx_max), 'descend');
        idx_max = idx_max(order);

        for j = 1:min(length(idx_max), 2)
            speedup_matrix(k, end-2+j) = ...
                blis_runtime(idx_max(j)) / npu_runtime(idx_max(j));
        end

        % One separate bars for the single-core agains all-core baseline
        idx_min = group_rows(no_of_cols(group_rows) == 0 & num_blis_threads(group_rows) == max_cores);

        if (idx_min)
            speedup_matrix(k, 1) = ...
                    blis_runtime(idx_min(1)) / npu_runtime(idx_min(1));
        end

    end

    %% Plot grouped bars

    b = bar(speedup_matrix, 'grouped');

    % Apply colors according to number of columns
    for c = 1:size(speedup_matrix, 2)
        

        b(c).FaceColor = colors(c,:);
        b(c).EdgeColor = 'none';
    end

    %% Formatting

    xlabel('$K$ : matmul($K$, $K$, $K$)', ...
        'Interpreter', 'latex', 'FontSize', 12);

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

    if isempty(valid_speedups)
        error('No positive, finite speedup values for device %s.', ...
            current_device);
    end

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

    %% Minor ticks

    minor_ticks = 2.^((y_min_exp:y_max_exp-1) + 0.5);

    ax = gca;

    ax.YMinorTick = 'on';
    ax.YAxis.MinorTickValues = minor_ticks;
    ax.YMinorGrid = 'on';

    ax.GridAlpha = 0.25;
    ax.MinorGridAlpha = 0.15;

    %% Axis limits

    ylim([2^y_min_exp, 2^y_max_exp]);

    %% 1x baseline

    yline(1, '--k', '1x baseline', ...
        'LineWidth', 1.2, ...
        'LabelHorizontalAlignment', 'left');

    %% Legend

    lgd = legend(b, x_labels, ...
        'Location', 'northeast');

    lgd.FontSize = 13;
    lgd.Box = 'on';
    lgd.ItemTokenSize = [10, 8];

    % pos = lgd.Position;
    % pos(1) = pos(1) - 0.2;  % Move legend up
    % lgd.Position = pos;

    set(gca, 'FontSize', 14);

    %% Save as vector PDF and PNG

    exportgraphics(gcf, ...
        fullfile(plotsPath, ...
        filename + "_speedup_" + current_device + ".pdf"), ...
        'ContentType', 'vector', ...
        'BackgroundColor', 'white');

    exportgraphics(gcf, ...
        fullfile(plotsPath, ...
        filename + "_speedup_" + current_device + ".png"), ...
        'BackgroundColor', 'white');

    hold off;

end
