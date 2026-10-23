%% Efficiency of implementation scaling

clear;
clc;
close all;

path = "..\measurements\";
plotsPath = "..\plots\";
filename = 'ozaki_perf_tests';

deviceMap = containers.Map( ...
    ["npu1", "npu2"], ...
    ["XDNA1", "XDNA2"]);

%% Parallel scaling efficiency

%% Read data

file = fullfile(path, [filename '.dat']);

data = readtable(file, ...
    'ReadVariableNames', false, ...
    'Delimiter', {' ', '\t'}, ...
    'MultipleDelimsAsOne', true);

data.Properties.VariableNames = { ...
    'device', ...
    'log2K', ...
    'parallelism_factor', ...
    'runtime_impl', ...
    'runtime_base1', ...
    'runtime_base2' ...
};

%% Extract columns

device              = data.device;
log2K               = data.log2K;
parallelism_factor  = data.parallelism_factor;
runtime_impl        = data.runtime_impl;

device = string(data.device);

%% Effective parallelism factor

p_eff = parallelism_factor * 4;
p_eff(parallelism_factor == 0) = 1;

%% Calculate efficiency

efficiency = NaN(height(data), 1);

devices = unique(device);

for d = 1:length(devices)

    device_idx = device == devices(d);
    K_values = unique(log2K(device_idx));
    current_device = devices(d);

    for k = 1:length(K_values)

        idx = device_idx & log2K == K_values(k);

        % Implementation's own single-core runtime
        idx_single = idx & parallelism_factor == 0;

        if ~any(idx_single)
            warning('No single-core runtime for device=%s, log2(K)=%g', ...
                devices(d), K_values(k));
            continue;
        end

        T_single = runtime_impl(find(idx_single, 1));

        % Efficiency
        efficiency(idx) = ...
            T_single ./ (runtime_impl(idx) .* p_eff(idx));

    end
end

%% Plot one figure per device

for d = 1:length(devices)

    fig = figure(d);
    fig.WindowState = 'maximized';
    clf;
    hold on;
    grid on;
    box on;

    device_idx = device == devices(d);
    K_values = unique(log2K(device_idx));
     current_device = devices(d);

    for k = 1:length(K_values)

        idx = device_idx & log2K == K_values(k);

        % FIX 1: Skip plotting if no single-core baseline exists
        idx_single = idx & parallelism_factor == 0;

        if ~any(idx_single)
            continue;
        end

        [p_sorted, order] = sort(parallelism_factor(idx));

        eff_sorted = efficiency(idx);
        eff_sorted = eff_sorted(order);

        % Display positions:
        %
        % parallelism factor    display position
        %          0                    0
        %       dummy (0.5)             1
        %          1                    2
        %          2                    3
        %          4                    4
        %          8                    5
        %
        % The dummy position represents 2 cores visually,
        % but there is NO corresponding data point.

        x_plot = zeros(size(p_sorted));

        x_plot(p_sorted == 0) = 0;
        x_plot(p_sorted == 1) = 2;
        x_plot(p_sorted == 2) = 3;
        x_plot(p_sorted == 4) = 4;
        x_plot(p_sorted == 8) = 5;

        plot(x_plot, eff_sorted, '-o', ...
            'LineWidth', 1.5, ...
            'MarkerSize', 5, ...
            'DisplayName', sprintf('Oz-AIE, K = 2^{%d}', K_values(k)));

    end

    % Ideal efficiency
    yline(1, '--k', 'Ideal efficiency', ...
        'LineWidth', 1.5, ...
        'HandleVisibility', 'off');

    xlabel('# of columns of AIE array', 'FontSize', 12);
    ylabel('Parallel Efficiency', 'FontSize', 12);

    title(sprintf('Parallel efficiency - %s', deviceMap(current_device)), ...
        'FontSize', 14);

    % Equal spacing, with an unlabeled dummy tick between
    % single-core and parallelism factor = 1
    xticks(0:5);
    xticklabels({'Single-core', '', '1', '2', '4', '8'});

    if current_device=="npu1"
        xticks(0:4);
    else
        xticks(0:5);
    end    


    xlim([0 5.25]);



    ylim([0.0 1.1]);

    lgd = legend('Location', 'best');
    lgd.FontSize = 12;
    lgd.Box = 'on';

    set(gca, 'FontSize', 12);

    hold off;

    %% ============================================================
    %  Save as vector PDF for LaTeX
    % =============================================================

    exportgraphics(gcf, ...
        fullfile(plotsPath, filename + "_efficiency_" + current_device + ".pdf"), ...
        'ContentType', 'vector', ...
        'BackgroundColor', 'white');

    exportgraphics(gcf, ...
        fullfile(plotsPath, filename + "_efficiency_" + current_device + ".png"), ...
        'ContentType', 'vector', ...
        'BackgroundColor', 'white');

end




