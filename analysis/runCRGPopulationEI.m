% runCRGPopulationEI.m - Population E/I analysis of contrast reversing gratings
%   Paper reference: Figure 4F, Figure 6C (E/I ratio across light levels)
%   Requires: main.m to be run first (sets up summaryFolder)
%
%   Analyzes population-level E/I cross-correlation and temporal offset
%   across cell types and bar sizes from CRG data.

%% Population level summary EI analysis of CRG
clc; clear barID
CloseAllFiguresExceptGUI;

% Load the data into a table
load(fullfile(summaryFolder, 'EISummary.mat'));
sumTable = struct2table(EISummary);
sumTable.cellType = categorical(sumTable.cellType);

% Get unique cell types
cellTypes = unique(sumTable.cellType);

% Loop through each cell type
for c = 1:numel(cellTypes)
    subTable = sumTable(sumTable.cellType == cellTypes(c), {'date', 'cellID', 'barList', 'lags', 'eiCorr'});
    for cellIdx = 1:height(subTable)
        cellData = subTable(cellIdx, :);
        figure('Name', sprintf('Cell Type: %s, Cell ID: %s', char(cellTypes(c)), cellData.cellID{1}), 'Color', 'w');
        hold on;
        for barIdx = 1:numel(cellData.barList{1})
            lags = cellData.lags(1, :);
            corrVals = cellData.eiCorr{1}(barIdx, :);
            plot(lags, corrVals, 'LineWidth', 2, 'DisplayName', sprintf('Bar Size: %d', cellData.barList{1}(barIdx)));
        end
        xlabel('Lag (ms)');
        ylabel('Cross-correlation');
        title(sprintf('Cross-correlation for Cell ID: %s, Cell Type: %s, Date: %s', cellData.cellID{1}, char(cellTypes(c)), cellData.date{1}));
        legend('show');
        hold off;
    end
end

%% Overlay population cross-correlation
figure1 = figure;
hold on;
figure2 = figure;
hold on;

colors = [[0.5 0.5 0.5]; [0.9 0.1 0.1]];

for c = 1:numel(cellTypes)
    subTable = sumTable(sumTable.cellType == cellTypes(c), {'date', 'cellID', 'barList', 'lags', 'eiCorr','meanEIOffset'});
    if c == 1
        plotList = [4 5 5 5 9];
        delayList = [4 6 1 4 8];
    else
        plotList = [3 2 2 3 5 4 3 3 1 3];
        delayList = [3 3 4 4 4 3 3 5 3 3];
    end
    tpTime = zeros(1, height(subTable));
    for cellIdx = 1:height(subTable)
        cellData = subTable(cellIdx, :);
        lags = cellData.lags;
        corrVals = cellData.eiCorr{1}(plotList(cellIdx), :);
        figure(figure1);
        plot(lags, corrVals, 'LineWidth', 2, 'Color', colors(c, :));
        tpTime(cellIdx) = -cellData.meanEIOffset{1}(delayList(cellIdx));
    end
    figure(figure2);
    scatter(c*ones(1, height(subTable))+0.1*rand(1,height(subTable)), tpTime, 150, colors(c, :),'filled');
    xlim([0.5 2.5]);
end

figure(figure1);
xlabel('Lag (ms)');
ylabel('Cross-correlation');
xlim([-200, 200]);

figure(figure2);
xlabel('Cell Index');
ylabel('Temporal Offset (ms)');
title('Scatter Plot of Time Points');
set(gca,'xtick',[ 1 2],'xticklabel',{'OffS','OffT'})

%% Repeated EI analysis (single bar size)
clc; clear barID
CloseAllFiguresExceptGUI;
load(fullfile(summaryFolder, 'EISummary.mat'));
sumTable = struct2table(EISummary);
sumTable.cellType = categorical(sumTable.cellType);
cellTypes = unique(sumTable.cellType);

for c = 1:numel(cellTypes)
    subTable = sumTable(sumTable.cellType == cellTypes(c), {'date', 'cellID', 'barList', 'lags', 'eiCorr'});
    for cellIdx = 1:height(subTable)
        cellData = subTable(cellIdx, :);
        figure('Name', sprintf('Cell Type: %s, Cell ID: %s', char(cellTypes(c)), cellData.cellID{1}), 'Color', 'w');
        hold on;
        for barIdx = 1:numel(cellData.barList{1})
            lags = cellData.lags(1, :);
            corrVals = cellData.eiCorr{1}(barIdx, :);
            plot(lags, corrVals, 'LineWidth', 2, 'DisplayName', sprintf('Bar Size: %d', cellData.barList{1}(barIdx)));
        end
        xlabel('Lag (ms)');
        ylabel('Cross-correlation');
        title(sprintf('Cross-correlation for Cell ID: %s, Cell Type: %s, Date: %s', cellData.cellID{1}, char(cellTypes(c)), cellData.date{1}));
        legend('show');
        hold off;
    end
end

%% Overlay population for a single bar size
figure1 = figure;
hold on;
figure2 = figure;
hold on;
colors = [[0.5 0.5 0.5]; [0.9 0.1 0.1]];

for c = 1:numel(cellTypes)
    subTable = sumTable(sumTable.cellType == cellTypes(c), {'date', 'cellID', 'barList', 'lags', 'eiCorr','meanEIOffset'});
    if c == 1
        plotList = [4 5 5 5 9];
        delayList = [4 6 1 4 8];
    else
        plotList = [3 2 2 3 5 4 3 3 1 3];
        delayList = [3 3 4 4 4 3 3 5 3 3];
    end
    tpTime = zeros(1, height(subTable));
    for cellIdx = 1:height(subTable)
        cellData = subTable(cellIdx, :);
        lags = cellData.lags;
        corrVals = cellData.eiCorr{1}(plotList(cellIdx), :);
        figure(figure1);
        plot(lags, corrVals, 'LineWidth', 2, 'Color', colors(c, :));
        tpTime(cellIdx) = -cellData.meanEIOffset{1}(delayList(cellIdx));
    end
    figure(figure2);
    scatter(c*ones(1, height(subTable))+0.1*rand(1,height(subTable)), tpTime, 150, colors(c, :),'filled');
    xlim([0.5 2.5]);
end

figure(figure1);
xlabel('Lag (ms)');
ylabel('Cross-correlation');
xlim([-200, 200]);
title('Correlation Curves (Single Bar Size)');

figure(figure2);
xlabel('Cell Type');
ylabel('Temporal Offset (ms)');
title('Scatter Plot of Time Points (Single Bar Size)');
set(gca,'xtick',[1 2],'xticklabel',{'OffS','OffT'})

%% Compare across bar sizes (population average per bar size)
targetBarList = [10, 20, 40, 80];
nBars = numel(targetBarList);
barColors = lines(nBars);

figure3 = figure('Name', 'Population Correlation by Bar Size', 'Color', 'w');
figure4 = figure('Name', 'Temporal Offset by Bar Size', 'Color', 'w');
hold on;

for c = 1:numel(cellTypes)
    subTable = sumTable(sumTable.cellType == cellTypes(c), {'date', 'cellID', 'barList', 'lags', 'eiCorr', 'meanEIOffset'});
    nCells = height(subTable);
    lags = subTable.lags(1, :);
    nLags = numel(lags);
    corrMatrix = NaN(nCells, nLags, nBars);
    offsetMatrix = NaN(nCells, nBars);
    for cellIdx = 1:nCells
        cellData = subTable(cellIdx, :);
        cellBarList = cellData.barList{1};
        for barIdx = 1:nBars
            barMatch = find(cellBarList == targetBarList(barIdx), 1);
            if ~isempty(barMatch)
                corrMatrix(cellIdx, :, barIdx) = cellData.eiCorr{1}(barMatch, :);
                offsetMatrix(cellIdx, barIdx) = -cellData.meanEIOffset{1}(barMatch);
            end
        end
    end
    figure(figure3);
    subplot(1, numel(cellTypes), c);
    hold on;
    plotHandles = gobjects(nBars, 1);
    for barIdx = 1:nBars
        meanCorr = nanmean(corrMatrix(:, :, barIdx), 1);
        semCorr = nanstd(corrMatrix(:, :, barIdx), 0, 1) / sqrt(sum(~isnan(corrMatrix(:, 1, barIdx))));
        errIdx = 1:20:nLags;
        plotHandles(barIdx) = errorbar(lags(errIdx), meanCorr(errIdx), semCorr(errIdx), ...
            'o-', 'Color', barColors(barIdx, :), 'LineWidth', 2, 'CapSize', 4, ...
            'MarkerFaceColor', barColors(barIdx, :), 'MarkerSize', 4);
    end
    xlabel('Lag (ms)');
    ylabel('Cross-correlation');
    xlim([-200, 200]);
    title(sprintf('%s (n=%d)', char(cellTypes(c)), nCells));
    legend(plotHandles, arrayfun(@(x) sprintf('%d um', x), targetBarList, 'UniformOutput', false), 'Location', 'best');
    hold off;

    figure(figure4);
    for barIdx = 1:nBars
        validIdx = ~isnan(offsetMatrix(:, barIdx));
        validOffsets = offsetMatrix(validIdx, barIdx);
        nValid = sum(validIdx);
        xPos = (c - 1) * (nBars + 1) + barIdx;
        scatter(xPos * ones(1, nValid) + 0.1 * rand(1, nValid), validOffsets, 150, barColors(barIdx, :), 'filled');
    end
end

figure(figure3);
sgtitle('Population Cross-correlation by Bar Size');

figure(figure4);
xlabel('Bar Size');
ylabel('Temporal Offset (ms)');
title('Temporal Offset by Bar Size');
xTickPos = [];
xTickLabels = {};
for c = 1:numel(cellTypes)
    for barIdx = 1:nBars
        xTickPos = [xTickPos, (c - 1) * (nBars + 1) + barIdx];
        xTickLabels = [xTickLabels, sprintf('%d', targetBarList(barIdx))];
    end
end
set(gca, 'xtick', xTickPos, 'xticklabel', xTickLabels);
xlim([0, numel(cellTypes) * (nBars + 1)]);
ax = gca;
for c = 1:numel(cellTypes)
    centerX = (c - 1) * (nBars + 1) + (nBars + 1) / 2;
    text(centerX, ax.YLim(1) - 0.1 * diff(ax.YLim), char(cellTypes(c)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
end
