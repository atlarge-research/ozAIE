%% ============================================================
%  PATH AND FILENAME
% =============================================================

path = "..\measurements\";
plotsPath = "..\plots\";
filename = 'ozaki_cycles';
devices = ["npu1","npu2"];

for device = devices

    %% ============================================================
    %  Load data
    % =============================================================

    data = readtable(fullfile(path, [filename '.dat']), ...
                     'NumHeaderLines', 0);

    % Column 1 contains the device name (text)
    data = data(strcmp(string(data{:,1}), string(device)), :);


    %% ============================================================
    %  Define columns
    % =============================================================

    plotCol  = 2;              % Determines the 4 subplots
    groupCol = 3;              % Determines the bar groups

    barCols = [6 8 10 12 14 16];     % Columns plotted as bars


    %% ============================================================
    %  Extract relevant columns as arrays, NOT tables
    % =============================================================

    plotValues  = data{:, plotCol};
    groupValues = data{:, groupCol};

    % Ensure they are numeric
    plotValues  = double(plotValues);
    groupValues = double(groupValues);

    plotCategories  = unique(plotValues);
    groupCategories = unique(groupValues);


    %% ============================================================
    %  Colors for the four bar types
    % =============================================================

    colors = [
        0.3010 0.7450 0.9330    % Light blue
        0.0000 0.2000 0.5000    % Dark blue
        0.8500 0.3250 0.3250    % Light red
        0.6000 0.0000 0.0000    % Dark red
        0.0000 0.4470 0.7410    % Blue
         0.4660 0.6740 0.1880    % Green
    ];


    %% ============================================================
    %  Create figure and tiled layout
    % =============================================================

    fig = figure('Color','w', ...
           'Units','inches', ...
           'Position',[1 1 7 5]);
    fig.WindowState = "maximized";

    t = tiledlayout(2,2, ...
        'TileSpacing','compact', ...
        'Padding','compact');


    %% ============================================================
    %  Create the four plots
    % ============================================================
    
    ax = gobjects(length(plotCategories),1);
    yMax = 0;
    
    for i = 1:length(plotCategories)
    
        % Rows belonging to this subplot
        idxPlot = (plotValues == plotCategories(i));
    
        % Rows = group categories
        % Columns = bar columns
        Y = NaN(length(groupCategories), length(barCols));
    
        for j = 1:length(groupCategories)
    
            idx = idxPlot & ...
                  (groupValues == groupCategories(j));
    
            rows = data(idx,:);
    
            if ~isempty(rows)
    
                % Extract numeric bar columns from table
                barData = rows{:, barCols};
    
                % If multiple rows exist, take their mean
                Y(j,:) = mean(barData, 1);
            end
        end
    
        % Update global maximum
        yMax = max(yMax, max(Y(:), [], 'omitnan'));
    
        %% Create tile
    
        ax(i) = nexttile;
    
        b = bar(Y, 'grouped');
    
        %% Apply colors
    
        for k = 1:length(barCols)
            b(k).FaceColor = colors(k,:);
        end
    
        %% Formatting
    
        title(sprintf('Bits per slice = %g', plotCategories(i)));
    
        xticks(1:length(groupCategories));
        xticklabels(string(groupCategories));
    
        grid on;
        box on;
    
    end


    %% ============================================================
    %  Same Y-axis for all four plots
    % ============================================================
    
    linkaxes(ax, 'y');
    
    % Start at zero and leave 10% headroom above maximum
    % ylim(ax(1), [0, 1.10*yMax]);
    ylim(ax(1), [0, 20000]);
    
    for i = 1:length(ax)
        ax(i).YAxis.Exponent = 0;
    end

    %% ============================================================
    %  Shared X and Y labels
    % =============================================================

    xlabel(t, 'Mantissa length (l_m)');
    ylabel(t, 'Cycles');


    %% ============================================================
    %  Shared legend
    % =============================================================

    lgd = legend(b, ...
        {'Max exp A', 'Max exp B', 'Splits A', 'Splits B', 'Mul', 'Sum'}, ...
        'Orientation', 'horizontal');

    lgd.Layout.Tile = 'north';


    %% ============================================================
    %  Font
    % =============================================================

    set(findall(gcf,'-property','FontSize'), ...
        'FontSize', 12);


    %% ============================================================
    %  Save as vector PDF for LaTeX
    % =============================================================

    exportgraphics(gcf, ...
        fullfile(plotsPath, filename + "_" + device + ".pdf"), ...
        'ContentType', 'vector', ...
        'BackgroundColor', 'white');

    exportgraphics(gcf, ...
        fullfile(plotsPath, filename + "_" + device + ".png"), ...
        'ContentType', 'vector', ...
        'BackgroundColor', 'white');

end

