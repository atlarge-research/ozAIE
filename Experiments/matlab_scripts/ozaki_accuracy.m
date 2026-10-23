%% Plot relative errors from .dat file

clear;
clc;
close all;

%% File name
path = "..\measurements\";
plotsPath = "..\plots\";
filename = "ozaki_accuracy_tests";

deviceMap = containers.Map( ...
    ["npu1", "npu2"], ...
    ["XDNA1", "XDNA2"]);

%% Read data
%
% Columns:
% 1  device
% 2  core_mode
% 3  fi
% 4  log2(K)
% 5  dummy1
% 6  dummy2
% 7  relativeError_impl
% 8  relativeError2
% 9  relativeError3
% 10 ignored
%

fid = fopen(path + filename + ".dat", 'r');

if fid == -1
    error('Could not open file: %s', filename);
end

C = textscan(fid, ...
    '%s %s %f %f %f %f %f %f %f %f', ...
    'Delimiter', {' ', '\t'}, ...
    'MultipleDelimsAsOne', true, ...
    'CollectOutput', false);

fclose(fid);

%% Extract columns
device    = C{1};       % string
core_mode = C{2};       % string

fi        = C{3};
log2K     = C{4};

log2K_values = unique(log2K, 'sorted');

dummy1    = C{5};       %#ok<NASGU>
dummy2    = C{6};       %#ok<NASGU>

err_impl  = C{7};
err2      = C{8};
err3      = C{9};

% C{10} is intentionally ignored

%% Check that small numbers were read correctly
fprintf('First few relative errors:\n');
fprintf('Oz-AIE: %.17e\n', err_impl(1:min(5,end)));
fprintf('Naive DGEMM: %.17e\n', err2(1:min(5,end)));
fprintf('BLIS: %.17e\n', err3(1:min(5,end)));

%% Unique values
devices = unique(device);
fis     = unique(fi);
modes   = unique(core_mode);

fprintf('\nDevices:\n');
disp(devices);

fprintf('fi values:\n');
disp(fis);

fprintf('Core modes:\n');
disp(modes);

%% Colors
color_impl = [0.0000 0.4470 0.7410];   % Blue
color_err2 = [0.8500 0.3250 0.0980];  % Orange
color_err3 = [0.4660 0.6740 0.1880];  % Green

%% Create one figure per device
for d = 1:length(devices)

    thisDevice = devices{d};

    fig = figure('Name', sprintf('Device %s', deviceMap(string(thisDevice))), ...
           'NumberTitle', 'off', ...
           'Color', 'w');
    fig.WindowState = 'maximized';

    %% Four subplots, one for each fi
    for f = 1:length(fis)

        thisFi = fis(f);

        subplot(2, 2, f);
        hold on;
        grid on;
        box on;

        %% Data for this device and fi
        idx = strcmp(device, thisDevice) & fi == thisFi;

        %% =========================================================
        % relativeError_impl
        % Two lines: one per core mode
        % ==========================================================

        for m = 1:length(modes)

            thisMode = modes{m};

            idx_mode = idx & strcmp(core_mode, thisMode);

            x = log2K(idx_mode);
            y = err_impl(idx_mode);

            [x, order] = sort(x);
            y = y(order);

            if m == 1
                lineStyle = '-';
            else
                lineStyle = '--';
            end

            semilogy(x, y, ...
                'Color', color_impl, ...
                'LineStyle', lineStyle, ...
                'LineWidth', 1.8, ...
                'Marker', 'o', ...
                'MarkerSize', 5, ...
                'DisplayName', sprintf('Oz-AIE, %s', thisMode));
        end

        %% =========================================================
        % relativeError2
        % One line
        % ==========================================================

        x = log2K(idx);
        y = err2(idx);

        [x, order] = sort(x);
        y = y(order);

        % Avoid plotting duplicate values if err2 occurs once per core mode
        [x_unique, ia] = unique(x);
        y_unique = y(ia);

        semilogy(x_unique, y_unique, ...
            'Color', color_err2, ...
            'LineStyle', '-', ...
            'LineWidth', 1.8, ...
            'Marker', 's', ...
            'MarkerSize', 5, ...
            'DisplayName', 'naive DGEMM');

        %% =========================================================
        % relativeError3
        % Two lines: one per core mode
        % ==========================================================

        for m = 1:length(modes)

            thisMode = modes{m};

            idx_mode = idx & strcmp(core_mode, thisMode);

            x = log2K(idx_mode);
            y = err3(idx_mode);

            [x, order] = sort(x);
            y = y(order);

            if m == 1
                lineStyle = '-';
            else
                lineStyle = '--';
            end

            semilogy(x, y, ...
                'Color', color_err3, ...
                'LineStyle', lineStyle, ...
                'LineWidth', 1.8, ...
                'Marker', '^', ...
                'MarkerSize', 5, ...
                'DisplayName', sprintf('BLIS, %s', thisMode));
        end

        %% Axes
        xlabel('$K$ : matmul($K$, $K$, $K$)', ...
        'Interpreter', 'latex','Interpreter', 'latex', 'FontSize', 12);
        ylabel('maximum relative error');

        title(sprintf('$\\phi = %g$', thisFi), ...
            'Interpreter', 'latex');

        xlim([7 12]);
        xticks(7:12);

        xticklabels(arrayfun( ...
            @(k) sprintf('$2^{%d}$', k), ...
            log2K_values, ...
            'UniformOutput', false));

        set(gca, 'TickLabelInterpreter', 'latex');
        set(gca, 'YScale', 'log');
        set(gca, 'FontSize', 12);

        ylim([1e-17 1e-7]);
        yticks(10.^(-17:-8));
        
        ax = gca;
        ax.YMinorTick = 'on';
        ax.YAxis.MinorTickValues = reshape(10.^(-17:-8) .* [2; 5], 1, []);

        if (f==4)
            legend('Location', 'southeast');
        end

        hold off;
    end

    %% Overall figure title
    sgtitle("Maximum Relative Error of Oz-AIE accuracy configurations on " + deviceMap(string(thisDevice)), ...
            'FontSize', 15, ...
            'FontWeight', 'bold');


    %% ============================================================
    %  Save as vector (PDF) and PNG 
    % =============================================================

    exportgraphics(gcf, ...
        fullfile(plotsPath, filename + "_" + thisDevice + ".pdf"), ...
        'ContentType', 'vector', ...
        'BackgroundColor', 'white');

    exportgraphics(gcf, ...
        fullfile(plotsPath, filename + "_" + thisDevice + ".png"), ...
        'ContentType', 'vector', ...
        'BackgroundColor', 'white');

end
