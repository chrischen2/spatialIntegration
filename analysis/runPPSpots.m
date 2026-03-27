% runPPSpots.m - Paired-pulse spot analysis
%   Paper reference: Figure 7A-H (paired-pulse facilitation, intervals)
%   Requires: main.m to be run first (sets up listSorted, summaryFolder)
%
%   Analyzes paired-pulse spot experiments with variable mean luminance
%   and variable pulse intervals. Includes drug conditions, population
%   visualization of pulse ratios, and interval-dependent recovery.

%% Create GUI for PP spots
rigSplit = @(listSorted)splitOnRigs(listSorted);
rigSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, rigSplit);

cellTypeSplit = @(listSorted)splitOnCellType(listSorted);
cellTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, cellTypeSplit);

dateSplit = @(listSorted)splitOnExperimentDate(listSorted);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, dateSplit);

tree = riekesuite.analysis.buildTree(listSorted,{cellTypeSplit_java, dateSplit_java,rigSplit_java,'protocolSettings(psth)', ...
    'cell.label','protocolSettings(epochGroup:label)'});
gui = epochTreeGUI(tree);

%% Analyze PP spot variable mean with drug
clc;
clear ppSpotsMean ppSpotsMeanDrug
paras.saveCell =0;
CloseAllFiguresExceptGUI;
paras.psthSigma  = 10;
paras.spikeTh    = 1.2;
paras.sampleRate = 1e4;

selectedNodes = gui.getSelectedEpochTreeNodes;
if isempty(selectedNodes)
    error('No nodes selected from the epoch tree!');
end

stimTime      = selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');
preTime       = selectedNodes{1}.epochList.firstValue.protocolSettings('preTime');
tailTime      = selectedNodes{1}.epochList.firstValue.protocolSettings('tailTime');
flashDuration = selectedNodes{1}.epochList.firstValue.protocolSettings('flashDuration');
paras.flashContrast = selectedNodes{1}.epochList.firstValue.protocolSettings('flashContrast');
paras.psth          = selectedNodes{1}.epochList.firstValue.protocolSettings('psth');
paras.cellType=selectedNodes{1}.epochList.firstValue.protocolSettings('source:type');

timeToPts = @(x) x/1e3 * paras.sampleRate;
paras.stimPts   = timeToPts(stimTime);
paras.prePts    = timeToPts(preTime);
paras.tailPts   = timeToPts(tailTime);
paras.flashPts  = timeToPts(flashDuration);

stats = analyzePPSpotsMean(selectedNodes, paras);

% Save stats for population analysis if enabled
if paras.saveCell
    if numel(selectedNodes) > 1
        matFileName = fullfile(summaryFolder, 'ppSpotsMeanDrug.mat');
        try
            load(matFileName, 'ppSpotsMeanDrug');
        catch
            ppSpotsMeanDrug = [];
        end
        if ~exist('ppSpotsMeanDrug', 'var') || ~isstruct(ppSpotsMeanDrug)
            ppSpotsMeanDrug = struct('cellID', {},'nodeName', {}, 'amp1', {}, 'amp2', {}, 'mTrace', {});
        end
        numCells = numel(ppSpotsMeanDrug);
        currentCellID = numCells + 1;
        for j = 1:length(stats.nodes)
            nodeStats = stats.nodes{j};
            ppSpotsMeanDrug(numCells+j) = struct(...
                'cellID',  currentCellID, ...
                'nodeName', selectedNodes{j}.splitValue, ...
                'amp1',    nodeStats.amp1, ...
                'amp2',    nodeStats.amp2, ...
                'mTrace',  nodeStats.mTrace ...
            );
        end
        save(matFileName, 'ppSpotsMeanDrug');
    else
        if selectedNodes{1}.parent.parent.splitValue == 0
            matFileName = fullfile(summaryFolder, 'ppSpotsMean.mat');
        else
            matFileName = fullfile(summaryFolder, 'ppSpotsMeanSpike.mat');
        end
        try
            load(matFileName, 'ppSpotsMean');
        catch
            ppSpotsMean = [];
        end
        if ~exist('ppSpotsMean', 'var') || ~isstruct(ppSpotsMean)
            ppSpotsMean = struct('nodeName', {}, 'amp1', {}, 'amp2', {}, 'mTrace', {},'cellType',{});
        end
        numCells = numel(ppSpotsMean);
        nodeStats = stats.nodes{1};
        firstIntervalIdx = 1;
        amp1FirstInterval = nodeStats.amp1(:, firstIntervalIdx);
        amp2FirstInterval = nodeStats.amp2(:, firstIntervalIdx);
        mTraceFirstInterval = squeeze(nodeStats.mTrace(:, firstIntervalIdx, :));
        ppSpotsMean(numCells+1) = struct(...
            'nodeName', selectedNodes{1}.splitValue, ...
            'amp1',    amp1FirstInterval, ...
            'amp2',    amp2FirstInterval, ...
            'mTrace',  mTraceFirstInterval, ...
            'cellType', paras.cellType ...
            );
        save(matFileName, 'ppSpotsMean');
    end
end

%% Control Group Visualization
close all;
controlFile = fullfile(summaryFolder, 'ppSpotsMean.mat');
if ~exist(controlFile, 'file')
    controlFile = fullfile(summaryFolder, 'ppSpotsMeanSpike.mat');
end

if exist(controlFile, 'file')
    load(controlFile, 'ppSpotsMean');
    numEntries = numel(ppSpotsMean);
    if numEntries == 0
        error('No data found in control summary file.');
    end

    allCellTypes = {ppSpotsMean.cellType};
    uniqueCellTypes = unique(allCellTypes);

    for ctIdx = 1:2
        ct = uniqueCellTypes{ctIdx};
        cellTypeIdx = find(strcmp(allCellTypes, ct));
        nCells = numel(cellTypeIdx);
        contrastCounts = arrayfun(@(x) length(x.amp1), ppSpotsMean(cellTypeIdx));
        maxContrasts = max(contrastCounts);
        switch maxContrasts
            case 3
                contrastArray = [0, 0.3, 0.6] * 100;
            case 4
                contrastArray = [0, 0.3, 0.6, 0.9] * 100;
            otherwise
                error('Unexpected maximum number of contrasts: %d', maxContrasts);
        end
        pr = [];
        for i = 1:numel(cellTypeIdx)
            entryIdx = cellTypeIdx(i);
            pulseRatio = ppSpotsMean(entryIdx).amp2 ./ ppSpotsMean(entryIdx).amp1;
            pulseRatio = pulseRatio(:)';
            pulseRatio=pulseRatio(:,1:3);
            contrastArray=contrastArray(1:3);
            pr = [pr; pulseRatio];
        end

        figure('Color', 'w');
        hold on;
        colors = lines(size(pr, 1));
        data = pr';
        groups = contrastArray;
        groupName = arrayfun(@(x) sprintf('%g%%', x), groups, 'UniformOutput', false);
        linedScatterplot(data, groups, groupName, 'Pulse Ratios Across Contrasts', ...
            arrayfun(@(x) sprintf('Cell %d', x), 1:size(pr, 1), 'UniformOutput', false));
        xlabel('Contrast Index');
        ylabel('Pulse Ratio');
        title('Pulse Ratios Across Contrasts');
        legend('Location', 'Best');
        hold off;
    end
else
    warning('Control summary file not found: %s', controlFile);
end

%% Drug/Wash Group Visualization
drugFile = fullfile(summaryFolder, 'ppSpotsMeanDrug.mat');
if exist(drugFile, 'file')
    load(drugFile, 'ppSpotsMeanDrug');
    numEntries = numel(ppSpotsMeanDrug);
    if numEntries == 0
        error('No data found in drug summary file.');
    end

    allCellIDs = [ppSpotsMeanDrug.cellID];
    uniqueCellIDs = unique(allCellIDs);
    nCells = numel(uniqueCellIDs);
    contrastArray = [0, 0.3, 0.6] * 100;
    nContrasts = numel(contrastArray);
    allNodeNames = {ppSpotsMeanDrug.nodeName};
    uniqueNodeNames = unique(allNodeNames, 'stable');
    nConds = numel(uniqueNodeNames);
    pulseRatioCells = cell(1, nContrasts);
    for c = 1:nContrasts
        pulseRatioCells{c} = NaN(nConds, nCells);
    end
    for cellIdx = 1:nCells
        cellID = uniqueCellIDs(cellIdx);
        idxEntries = find([ppSpotsMeanDrug.cellID] == cellID);
        for condIdx = 1:nConds
            condName = uniqueNodeNames{condIdx};
            idxCond = idxEntries(strcmp({ppSpotsMeanDrug(idxEntries).nodeName}, condName));
            if ~isempty(idxCond)
                for c = 1:nContrasts
                    prVals = [];
                    for j = 1:length(idxCond)
                        entry = ppSpotsMeanDrug(idxCond(j));
                        pr = entry.amp2(c) / entry.amp1(c);
                        prVals(end+1) = pr;
                    end
                    pulseRatioCells{c}(condIdx, cellIdx) = mean(prVals);
                end
            end
        end
    end
    totalSubplots = nContrasts + 1;
    figure;
    for c = 1:nContrasts
        subplot(totalSubplots, 1, c);
        hold on;
        for cellIdx = 1:nCells
            yVals = pulseRatioCells{c}(:, cellIdx);
            if all(isnan(yVals))
                continue;
            end
            plot(1:nConds, yVals, '-', 'Color', [0.8, 0.8, 0.8], 'LineWidth', 1);
        end
        meanVals = nanmean(pulseRatioCells{c}, 2);
        semVals  = nanstd(pulseRatioCells{c}, 0, 2) ./ sqrt(sum(~isnan(pulseRatioCells{c}),2));
        errorbar(1:nConds, meanVals, semVals, '-ok', 'LineWidth', 3, 'MarkerSize', 10, 'MarkerFaceColor', 'k');
        set(gca, 'XTick', 1:nConds, 'XTickLabel', uniqueNodeNames);
        ylabel('Pulse Ratio');
        title(sprintf('Drug Conditions - Contrast = %.0f%%', contrastArray(c)));
        box on;
        hold off;
    end
    subplot(totalSubplots, 1, totalSubplots);
    hold on;
    colors = lines(nContrasts);
    overlayHandles = gobjects(nContrasts, 1);
    legendLabels = cell(1, nContrasts);
    for c = 1:nContrasts
        meanVals = nanmean(pulseRatioCells{c}, 2);
        semVals = nanstd(pulseRatioCells{c}, 0, 2) ./ sqrt(sum(~isnan(pulseRatioCells{c}),2));
        overlayHandles(c) = errorbar(1:nConds, meanVals, semVals, '-o', 'Color', colors(c,:), 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', colors(c,:));
        legendLabels{c} = sprintf('Contrast = %.0f%%', contrastArray(c));
    end
    set(gca, 'XTick', 1:nConds, 'XTickLabel', uniqueNodeNames);
    xlabel('Condition');
    ylabel('Pulse Ratio');
    title('Overlay of Contrast Errorbars for Drug Conditions');
    legend(overlayHandles, legendLabels, 'Location', 'Best');
    box on;
    hold off;
else
    warning('Drug summary file not found: %s', drugFile);
end

%% Analyze PP spot variable intervals
clc;
clear ppSpotsInterval stats paras
paras.saveCell=1;
paras.filterRes=0;
CloseAllFiguresExceptGUI;
paras.psthSigma=10;
paras.spikeTh=1.2;
paras.sampleRate=1e4;
selectedNodes = gui.getSelectedEpochTreeNodes;
paras.rmRep=[   ];
if isempty(selectedNodes)
    error('No nodes selected from the epoch tree!');
end
stimTime=selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');
preTime=selectedNodes{1}.epochList.firstValue.protocolSettings('preTime');
tailTime=selectedNodes{1}.epochList.firstValue.protocolSettings('tailTime');
flashDuration=selectedNodes{1}.epochList.firstValue.protocolSettings('flashDuration');
paras.flashContrast=selectedNodes{1}.epochList.firstValue.protocolSettings('flashContrast');
paras.stepContrast=selectedNodes{1}.epochList.firstValue.protocolSettings('stepContrast');
paras.cellType=shortCellType(selectedNodes{1}.epochList.firstValue.protocolSettings('source:type'));
paras.psth=selectedNodes{1}.epochList.firstValue.protocolSettings('psth');
timeToPts=@(x) x/1e3*paras.sampleRate;
paras.stimPts=timeToPts(stimTime);
paras.prePts=timeToPts(preTime);
paras.tailPts=timeToPts(tailTime);
paras.flashPts=timeToPts(flashDuration);
[stats, paras]=analyzePPSpotsIntervals(selectedNodes{1},paras);
paras.cellType='OffT';

% Save stats for population analysis
if paras.saveCell
    numCells=0;
    try
        load(fullfile(summaryFolder, 'ppSpotsIntervalInh.mat'));
        numCells=numel(ppSpotsInterval);
    end
    ppSpotsInterval(numCells+1)=struct('uniqueCellID', numCells+1, 'cellType', paras.cellType, 'intervalArray',stats.intervalArray, 'amp1',stats.amp1, 'amp2',...
        stats.amp2, 'peakTime1', stats.peakTime1, 'peakTime2', stats.peakTime2,'secondPulseBaselines',stats.secondPulseBaselines,'recType', ...
        paras.recType,'meanTrace', stats.mTrace,'stimTrace', stats.stimTrace);
    save(fullfile(summaryFolder, 'ppSpotsIntervalNew.mat'),'ppSpotsInterval');
end

%% Visualize the population data
load(fullfile(summaryFolder, 'ppSpotsIntervalNew.mat'));

categorizeCells = @(cellType, recType) find(arrayfun(@(x) strcmpi(x.cellType, cellType) && strcmpi(x.recType, recType), ppSpotsInterval));
excCells = categorizeCells('OffT', 'exc');
spikeCells = categorizeCells('OffT', 'spike');
inhCells = categorizeCells('OffT', 'inh');

fprintf('- Excitatory cells: %d\n', length(excCells));
fprintf('- Spike recordings: %d\n', length(spikeCells));
fprintf('- Inhibitory cells: %d\n', length(inhCells));

% Helper function to compute stats
function [means, sems, counts, allIntervals] = computeStats(ppSpotsInterval, cellIdxs, valueFcn)
    allIntervals = unique(cell2mat(arrayfun(@(x) x.intervalArray, ppSpotsInterval(cellIdxs), 'UniformOutput', false)));
    allIntervals = sort(allIntervals);
    means = zeros(size(allIntervals));
    sems = zeros(size(allIntervals));
    counts = zeros(size(allIntervals));
    for intIdx = 1:length(allIntervals)
        currentInterval = allIntervals(intIdx);
        values = [];
        for cellIdx = cellIdxs
            intervalIdx = find(ppSpotsInterval(cellIdx).intervalArray == currentInterval);
            if ~isempty(intervalIdx)
                values = [values, valueFcn(ppSpotsInterval(cellIdx), intervalIdx)];
            end
        end
        if ~isempty(values)
            means(intIdx) = mean(values);
            sems(intIdx) = std(values) / sqrt(length(values));
            counts(intIdx) = length(values);
        end
    end
end

[excRatioMeans, excRatioSEMs, excRatioCounts, excIntervals] = computeStats(ppSpotsInterval, excCells, ...
    @(cell, idx) cell.amp2(idx) / cell.amp1(idx));
[excBaselineMeans, excBaselineSEMs, excBaselineCounts, ~] = computeStats(ppSpotsInterval, excCells, ...
    @(cell, idx) cell.secondPulseBaselines(idx));
[spikeRatioMeans, spikeRatioSEMs, spikeRatioCounts, spikeIntervals] = computeStats(ppSpotsInterval, spikeCells, ...
    @(cell, idx) cell.amp2(idx) / cell.amp1(idx));
if ~isempty(inhCells)
    [inhRatioMeans, inhRatioSEMs, inhRatioCounts, inhIntervals] = computeStats(ppSpotsInterval, inhCells, ...
        @(cell, idx) cell.amp2(idx) / cell.amp1(idx));
    [inhBaselineMeans, inhBaselineSEMs, inhBaselineCounts, ~] = computeStats(ppSpotsInterval, inhCells, ...
        @(cell, idx) cell.secondPulseBaselines(idx));
end

% Plot excitatory ratio
if ~isempty(excCells)
    figure;
    hold on;
    errorbar(excIntervals, excRatioMeans, excRatioSEMs, 'o-', 'LineWidth', 2, 'DisplayName', 'Exc Ratio');
    title('Excitatory Cells: Amplitude Ratio (amp2/amp1)');
    xlabel('Interval (ms)');
    ylabel('Adjusted Ratio');
    grid on;
    arrayfun(@(x, y, n) text(x, y + 0.05, ['n=', num2str(n)]), excIntervals, excRatioMeans + excRatioSEMs, excRatioCounts);
end

% Plot excitatory baseline
if ~isempty(excCells)
    figure;
    hold on;
    errorbar(excIntervals, excBaselineMeans, excBaselineSEMs, 'o-', 'LineWidth', 2, 'DisplayName', 'Baseline');
    title('Excitatory Cells: Second Pulse Baseline');
    xlabel('Interval (ms)');
    ylabel('Baseline Value');
    grid on;
    arrayfun(@(x, y, n) text(x, y + 0.05, ['n=', num2str(n)]), excIntervals, excBaselineMeans + excBaselineSEMs, excBaselineCounts);
end

% Plot spike ratio
if ~isempty(spikeCells)
    figure;
    hold on;
    errorbar(spikeIntervals, spikeRatioMeans, spikeRatioSEMs, 'o-', 'LineWidth', 2, 'Color', 'r', 'DisplayName', 'Spike Ratio');
    title('Spike Recordings: Amplitude Ratio (amp2/amp1)');
    xlabel('Interval (ms)');
    ylabel('Ratio');
    grid on;
    arrayfun(@(x, y, n) text(x, y + 0.05, ['n=', num2str(n)], 'Color', 'r'), spikeIntervals, spikeRatioMeans + spikeRatioSEMs, spikeRatioCounts);
end

% Plot inhibitory ratio
if ~isempty(inhCells)
    figure;
    hold on;
    errorbar(inhIntervals, inhRatioMeans, inhRatioSEMs, 'o-', 'LineWidth', 2, 'Color', [0.5 0 0.5], 'DisplayName', 'Inh Ratio');
    title('Inhibitory Cells: Amplitude Ratio (amp2/amp1)');
    xlabel('Interval (ms)');
    ylabel('Ratio');
    grid on;
    arrayfun(@(x, y, n) text(x, y + 0.05, ['n=', num2str(n)], 'Color', [0.5 0 0.5]), inhIntervals, inhRatioMeans + inhRatioSEMs, inhRatioCounts);
end

% Plot inhibitory baseline
if ~isempty(inhCells)
    figure;
    hold on;
    errorbar(inhIntervals, inhBaselineMeans, inhBaselineSEMs, 'o-', 'LineWidth', 2, 'Color', [0.5 0 0.5], 'DisplayName', 'Inh Baseline');
    title('Inhibitory Cells: Second Pulse Baseline');
    xlabel('Interval (ms)');
    ylabel('Baseline Value');
    grid on;
    arrayfun(@(x, y, n) text(x, y + 0.05, ['n=', num2str(n)], 'Color', [0.5 0 0.5]), inhIntervals, inhBaselineMeans + inhBaselineSEMs, inhBaselineCounts);
end

% Combined comparison
figure;
hold on;
if ~isempty(excCells)
    errorbar(excIntervals, excRatioMeans, excRatioSEMs, 'o-', 'LineWidth', 2, 'Color', 'b', 'DisplayName', 'Exc');
end
if ~isempty(spikeCells)
    errorbar(spikeIntervals, spikeRatioMeans, spikeRatioSEMs, 'o-', 'LineWidth', 2, 'Color', 'r', 'DisplayName', 'Spike');
end
if ~isempty(inhCells)
    errorbar(inhIntervals, inhRatioMeans, inhRatioSEMs, 'o-', 'LineWidth', 2, 'Color', [0.5 0 0.5], 'DisplayName', 'Inh');
end
xlabel('Interval (ms)');
ylabel('amp2/amp1 Ratio');
title('Paired-Pulse Recovery: All Recording Types');
legend('show', 'Location', 'best');
grid on;
