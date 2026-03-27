% this script anayze the pair pulse relieve experiment 
clearvars; close all; clc;
% define plot color sequence, axis fonts
import auimodel.*
import vuidocument.*
loader = edu.washington.rieke.Analysis.getEntityLoader();
treeFactory = edu.washington.rieke.Analysis.getEpochTreeFactory();
listFactory = edu.washington.rieke.Analysis.getEpochListFactory();
newList=listFactory.create;
ovaExportFolder='/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/';
dataFolder='/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/';

% list = loader.loadEpochList([ovaExportFolder 'PPSpotWithVariableMean.mat'], dataFolder);
list = loader.loadEpochList([ovaExportFolder 'pprInh.mat'], dataFolder);
% list = loader.loadEpochList([ovaExportFolder 'PPGratingWithVariableMean.mat'], dataFolder);
% list = loader.loadEpochList([ovaExportFolder 'PPGratingWithVariableInterval.mat'], dataFolder);


for i = 1:list.length
    try
        list.elements(i).setProtocolSetting('user:startDate',datestr((list.elements(i).startDate)'));
    catch 
        fprintf('%s  %i\n', 'fail to format', i);
    end
end
listSorted = list.sortedBy('protocolSettings(user:startDate)'); % list sorted chronologically


%% PP spot variable mean/interval  split data and create GUI
rigSplit = @(listSorted)splitOnRigs(listSorted);
rigSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, rigSplit);

cellTypeSplit = @(listSorted)splitOnCellType(listSorted);
cellTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, cellTypeSplit);

dateSplit = @(listSorted)splitOnExperimentDate(listSorted);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, dateSplit);

tree = riekesuite.analysis.buildTree(listSorted,{cellTypeSplit_java, dateSplit_java,rigSplit_java,'protocolSettings(psth)', ...
    'cell.label','protocolSettings(epochGroup:label)'});

% tree = riekesuite.analysis.buildTree(listSorted,{cellTypeSplit_java, dateSplit_java,rigSplit_java,'protocolSettings(psth)', ...
%     'cell.label','protocolSettings(stepContrast)'});

gui = epochTreeGUI(tree);



%% analyze PP spot variable mean with drug 
clc;
clear ppSpotsMean ppSpotsMeanDrug
% paras.rmRep = [1:3]; % removes rows 2 and 4
paras.saveCell =0;
CloseAllFiguresExceptGUI;
paras.psthSigma  = 10;
paras.spikeTh    = 1.2;
paras.sampleRate = 1e4;

selectedNodes = gui.getSelectedEpochTreeNodes;
if isempty(selectedNodes)
    error('No nodes selected from the epoch tree!');
end

% Get protocol settings from the first selected node
stimTime      = selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');
preTime       = selectedNodes{1}.epochList.firstValue.protocolSettings('preTime');
tailTime      = selectedNodes{1}.epochList.firstValue.protocolSettings('tailTime');
flashDuration = selectedNodes{1}.epochList.firstValue.protocolSettings('flashDuration');
paras.flashContrast = selectedNodes{1}.epochList.firstValue.protocolSettings('flashContrast');
paras.psth          = selectedNodes{1}.epochList.firstValue.protocolSettings('psth');
paras.cellType=selectedNodes{1}.epochList.firstValue.protocolSettings('source:type'); 
% Optionally, define paras.rmRep to remove specific trials. For example:

timeToPts = @(x) x/1e3 * paras.sampleRate;
paras.stimPts   = timeToPts(stimTime);
paras.prePts    = timeToPts(preTime);
paras.tailPts   = timeToPts(tailTime);
paras.flashPts  = timeToPts(flashDuration);


% Call the updated analyzePPSpotsMean function (which now uses paras.rmRep and outputs a structure with nodes)
stats = analyzePPSpotsMean(selectedNodes, paras);

% save the stats for population analysis if enabled
if paras.saveCell
    if numel(selectedNodes) > 1
        % File name for multiple nodes (drug analysis)
        matFileName = '/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/ppSpotsMeanDrug.mat';
        % Attempt to load existing data.
        try
            load(matFileName, 'ppSpotsMeanDrug');
        catch
            ppSpotsMeanDrug = [];
        end
        % If loaded variable is not a struct array, reinitialize.
        if ~exist('ppSpotsMeanDrug', 'var') || ~isstruct(ppSpotsMeanDrug)
            ppSpotsMeanDrug = struct('cellID', {},'nodeName', {}, 'amp1', {}, 'amp2', {}, 'mTrace', {});
        end
        % Determine the number of entries already saved.
        numCells = numel(ppSpotsMeanDrug);
        % Use the next index as the cell ID.
        currentCellID = numCells + 1;
        % Append each node's analysis results.
        for j = 1:length(stats.nodes)
            nodeStats = stats.nodes{j};
            ppSpotsMeanDrug(numCells+j) = struct(...
                'cellID',  currentCellID, ...                          % Unique cell identifier
                'nodeName', selectedNodes{j}.splitValue, ...  % Node name/identifier
                'amp1',    nodeStats.amp1, ...                % Amplitude 1 for each contrast
                'amp2',    nodeStats.amp2, ...                % Amplitude 2 for each contrast
                'mTrace',  nodeStats.mTrace ...               % Mean trace for each contrast
            );
        end
        % Save to file.
        save(matFileName, 'ppSpotsMeanDrug');
    else
        % Optionally, handle the single-node saving here.
        % For example, save to a separate file (e.g., ppSpotsMean.mat)
        % Determine which file to save to based on the parent property of selectedNodes{1}
        if selectedNodes{1}.parent.parent.splitValue == 0
            matFileName = '/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/ppSpotsMean.mat';
        else
            matFileName = '/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/ppSpotsMeanSpike.mat';
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
        % For a single node, store the same fields.
        nodeStats = stats.nodes{1};
        % Extract data for the first interval only.
        firstIntervalIdx = 1; % Assuming the first interval is always at index 1.
        amp1FirstInterval = nodeStats.amp1(:, firstIntervalIdx); % Amplitude 1 for the first interval
        amp2FirstInterval = nodeStats.amp2(:, firstIntervalIdx); % Amplitude 2 for the first interval
        mTraceFirstInterval = squeeze(nodeStats.mTrace(:, firstIntervalIdx, :)); % Mean trace for the first interval

        % Append the new structure entry, now including the cellType field.
        ppSpotsMean(numCells+1) = struct(...
            'nodeName', selectedNodes{1}.splitValue, ...  % node identifier/name
            'amp1',    amp1FirstInterval, ...            % amplitude 1 for each contrast (first interval only)
            'amp2',    amp2FirstInterval, ...            % amplitude 2 for each contrast (first interval only)
            'mTrace',  mTraceFirstInterval, ...          % mean trace for each contrast (first interval only)
            'cellType', paras.cellType ...               % cell type information
            );

        save(matFileName, 'ppSpotsMean');
    end
end

%% ---------------- Control Group Visualization ----------------
% Determine which control file exists: use ppSpotsMean.mat if available,
% otherwise fall back to the spike‐based summary ppSpotsMeanSpike.mat.
close all;
controlFile = '/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/ppSpotsMean.mat';
if ~exist(controlFile, 'file')
    controlFile = '/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/ppSpotsMeanSpike.mat';
end

if exist(controlFile, 'file')
    load(controlFile, 'ppSpotsMean');  % expected variable: ppSpotsMean
    numEntries = numel(ppSpotsMean);
    if numEntries == 0
        error('No data found in control summary file.');
    end
    
    % Group control entries by cell type.
    allCellTypes = {ppSpotsMean.cellType};
    uniqueCellTypes = unique(allCellTypes);
    
    for ctIdx = 1:2
        ct = uniqueCellTypes{ctIdx};
        % Get the indices of all entries for this cell type.
        cellTypeIdx = find(strcmp(allCellTypes, ct));
        nCells = numel(cellTypeIdx);
        
        % Determine the maximum number of contrasts among these cells.
        contrastCounts = arrayfun(@(x) length(x.amp1), ppSpotsMean(cellTypeIdx));
        maxContrasts = max(contrastCounts);
        
        % Define the common contrast array based on the maximum number.
        switch maxContrasts
            case 3
                contrastArray = [0, 0.3, 0.6] * 100;  % in %
            case 4
                contrastArray = [0, 0.3, 0.6, 0.9] * 100;  % in %
            otherwise
                error('Unexpected maximum number of contrasts: %d', maxContrasts);
        end
        pr = []; % Initialize pulse ratio array

        % Slice the pulseRatio data using cellTypeIdx
        for i = 1:numel(cellTypeIdx)
            entryIdx = cellTypeIdx(i);
            pulseRatio = ppSpotsMean(entryIdx).amp2 ./ ppSpotsMean(entryIdx).amp1; % Calculate pulse ratio
            pulseRatio = pulseRatio(:)'; % Ensure pulseRatio is a row vector
            pulseRatio=pulseRatio(:,1:3);
            contrastArray=contrastArray(1:3);
            pr = [pr; pulseRatio]; % Concatenate pulse ratios

            % if length(pulseRatio) < nContrasts
            %     % Pad with NaNs to match the number of contrasts
            %     tpr = [pulseRatio, NaN(1, nContrasts - length(pulseRatio))];
            %     pr = [pr; tpr]; % Concatenate pulse ratios
            % else
            %     pr = [pr; pulseRatio]; % Concatenate pulse ratios
            % end
        end


        % Visualization using linedScatterplot
        figure('Color', 'w');
        hold on;

        % Generate distinct colors for each cell
        colors = lines(size(pr, 1)); % Rows represent cells

        % Prepare data for all cells
        data = pr'; % Transpose so that columns represent contrast groups
        groups = contrastArray; % Use contrast values directly as group names
        groupName = arrayfun(@(x) sprintf('%g%%', x), groups, 'UniformOutput', false); % Format group names as percentages

        % Call the linedScatterplot function
        linedScatterplot(data, groups, groupName, 'Pulse Ratios Across Contrasts', ...
            arrayfun(@(x) sprintf('Cell %d', x), 1:size(pr, 1), 'UniformOutput', false));

        % Add labels and title
        xlabel('Contrast Index');
        ylabel('Pulse Ratio');
        title('Pulse Ratios Across Contrasts');
        legend('Location', 'Best');
        hold off;
    end
else
    warning('Control summary file not found: %s', controlFile);
end

%% ---------------- Drug/Wash Group Visualization ----------------
drugFile = '/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/ppSpotsMeanDrug.mat';
if exist(drugFile, 'file')
    load(drugFile, 'ppSpotsMeanDrug');  % expected variable: ppSpotsMeanDrug
    numEntries = numel(ppSpotsMeanDrug);
    if numEntries == 0
        error('No data found in drug summary file.');
    end
    
    % Aggregate entries by cellID so that all nodes from the same cell are combined.
    allCellIDs = [ppSpotsMeanDrug.cellID];
    uniqueCellIDs = unique(allCellIDs);
    nCells = numel(uniqueCellIDs);
    
    % For drug conditions the contrast array is fixed: [0, 0.3, 0.6] (in %).
    contrastArray = [0, 0.3, 0.6] * 100;  
    nContrasts = numel(contrastArray);
    
    % Determine condition names from nodeName field (e.g. 'Control', 'Drug', 'Wash').
    allNodeNames = {ppSpotsMeanDrug.nodeName};
    uniqueNodeNames = unique(allNodeNames, 'stable');  % preserve desired order
    nConds = numel(uniqueNodeNames);
    
    % Create a cell array to store pulse ratio data for each contrast.
    % Each element is a matrix [nConds x nCells], where each column is one cell.
    pulseRatioCells = cell(1, nContrasts);
    for c = 1:nContrasts
        pulseRatioCells{c} = NaN(nConds, nCells);
    end
    
    % Loop over each unique cell.
    for cellIdx = 1:nCells
        cellID = uniqueCellIDs(cellIdx);
        idxEntries = find([ppSpotsMeanDrug.cellID] == cellID);
        % For each condition (nodeName), average pulse ratio if multiple entries exist.
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
    
    % Create a figure with one subplot per contrast PLUS an additional overlay subplot.
    totalSubplots = nContrasts + 1;
    figure;
    for c = 1:nContrasts
        subplot(totalSubplots, 1, c);
        hold on;
        % Plot individual cell traces for this contrast as light-gray lines.
        for cellIdx = 1:nCells
            yVals = pulseRatioCells{c}(:, cellIdx);
            if all(isnan(yVals))
                continue;
            end
            plot(1:nConds, yVals, '-', 'Color', [0.8, 0.8, 0.8], 'LineWidth', 1);
        end
        % Compute mean and SEM across cells (ignoring NaNs) for each condition.
        meanVals = nanmean(pulseRatioCells{c}, 2);
        semVals  = nanstd(pulseRatioCells{c}, 0, 2) ./ sqrt(sum(~isnan(pulseRatioCells{c}),2));
        
        % Overlay the group mean with error bars as a thick black line.
        errorbar(1:nConds, meanVals, semVals, '-ok', 'LineWidth', 3, 'MarkerSize', 10, 'MarkerFaceColor', 'k');
        set(gca, 'XTick', 1:nConds, 'XTickLabel', uniqueNodeNames);
        ylabel('Pulse Ratio');
        title(sprintf('Drug Conditions - Contrast = %.0f%%', contrastArray(c)));
        box on;
        hold off;
    end
    
    % --- Additional Overlay Subplot: errorbars for all contrasts on one axis ---
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


%% analyze PP spot variable intervals 
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
% Optionally, define paras.rmRep to remove specific trials. For example:
% paras.rmRep = [2, 4]; % removes rows 2 and 4
timeToPts=@(x) x/1e3*paras.sampleRate;
paras.stimPts=timeToPts(stimTime);
paras.prePts=timeToPts(preTime);
paras.tailPts=timeToPts(tailTime);
paras.flashPts=timeToPts(flashDuration);
[stats, paras]=analyzePPSpotsIntervals(selectedNodes{1},paras);
paras.cellType='OffT';
% save the stats for population analysis
if paras.saveCell
    numCells=0;
    try
        load('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/ppSpotsIntervalInh.mat');
        numCells=numel(ppSpotsInterval);
    end
    ppSpotsInterval(numCells+1)=struct('uniqueCellID', numCells+1, 'cellType', paras.cellType, 'intervalArray',stats.intervalArray, 'amp1',stats.amp1, 'amp2',...
        stats.amp2, 'peakTime1', stats.peakTime1, 'peakTime2', stats.peakTime2,'secondPulseBaselines',stats.secondPulseBaselines,'recType', ...
        paras.recType,'meanTrace', stats.mTrace,'stimTrace', stats.stimTrace);
    save('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/ppSpotsIntervalNew.mat','ppSpotsInterval');
end


%% visualize the population data
% Load data
load('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/ppSpotsIntervalNew.mat');

% Helper function to categorize cells by type and recType
categorizeCells = @(cellType, recType) find(arrayfun(@(x) strcmpi(x.cellType, cellType) && strcmpi(x.recType, recType), ppSpotsInterval));

excCells = categorizeCells('OffT', 'exc');
spikeCells = categorizeCells('OffT', 'spike');
inhCells = categorizeCells('OffT', 'inh');

fprintf('- Excitatory cells: %d\n', length(excCells));
fprintf('- Spike recordings: %d\n', length(spikeCells));
fprintf('- Inhibitory cells: %d\n', length(inhCells));

% Helper function to compute stats for a given cell set and value extractor
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

% Compute stats for excitatory cells
[excRatioMeans, excRatioSEMs, excRatioCounts, excIntervals] = computeStats(ppSpotsInterval, excCells, ...
    @(cell, idx) cell.amp2(idx) / cell.amp1(idx));
[excBaselineMeans, excBaselineSEMs, excBaselineCounts, ~] = computeStats(ppSpotsInterval, excCells, ...
    @(cell, idx) cell.secondPulseBaselines(idx));

% Compute stats for spike cells
[spikeRatioMeans, spikeRatioSEMs, spikeRatioCounts, spikeIntervals] = computeStats(ppSpotsInterval, spikeCells, ...
    @(cell, idx) cell.amp2(idx) / cell.amp1(idx));

% Compute stats for inhibitory cells
if ~isempty(inhCells)
    [inhRatioMeans, inhRatioSEMs, inhRatioCounts, inhIntervals] = computeStats(ppSpotsInterval, inhCells, ...
        @(cell, idx) cell.amp2(idx) / cell.amp1(idx));
    [inhBaselineMeans, inhBaselineSEMs, inhBaselineCounts, ~] = computeStats(ppSpotsInterval, inhCells, ...
        @(cell, idx) cell.secondPulseBaselines(idx));
end

% Plot 1: Adjusted amp2/amp1 ratio for excitatory cells
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

% Plot 2: Baseline values for excitatory cells
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

% Plot 3: Unadjusted amp2/amp1 ratios for spike recordings
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

% Plot 4: Amplitude ratio for inhibitory cells
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

% Plot 5: Baseline values for inhibitory cells
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

% Plot 6: Combined comparison - all three recording types
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
%% split for gratings with varialbe means 
% intervals
cellTypeSplit = @(listSorted)splitOnCellType(listSorted);
cellTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, cellTypeSplit);

dateSplit = @(listSorted)splitOnExperimentDate(listSorted);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, dateSplit);

tree = riekesuite.analysis.buildTree(listSorted,{cellTypeSplit_java, dateSplit_java, 'cell.label','protocolSettings(grateContrast)', ...
    'protocolSettings(psth)'});

gui = epochTreeGUI(tree);


%% analyze  grating with varialbe mean
clc;
clear ppGratingsMean 
paras.saveCell=0;
CloseAllFiguresExceptGUI;
paras.psthSigma=10;
paras.spikeTh=1.2;
paras.sampleRate=1e4;
selectedNodes = gui.getSelectedEpochTreeNodes;
stimTime=selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');
preTime=selectedNodes{1}.epochList.firstValue.protocolSettings('preTime');
tailTime=selectedNodes{1}.epochList.firstValue.protocolSettings('tailTime');
flashDuration=selectedNodes{1}.epochList.firstValue.protocolSettings('grateDuration');
paras.flashContrast=selectedNodes{1}.epochList.firstValue.protocolSettings('grateContrast');

pulseIntervals=selectedNodes{1}.epochList.firstValue.protocolSettings('pulseIntervals');
paras.psth=selectedNodes{1}.epochList.firstValue.protocolSettings('psth');
timeToPts=@(x) x/1e3*paras.sampleRate;
paras.stimPts=timeToPts(stimTime);
paras.prePts=timeToPts(preTime);
paras.tailPts=timeToPts(tailTime);
paras.flashPts=timeToPts(flashDuration);
paras.intervalPts=timeToPts(pulseIntervals);
stats=analyzePPGratingsMean(selectedNodes{1},paras);

% save the stats for population analysis
if paras.saveCell
    numCells=0;
    try
        load('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/ppGratingsMean.mat');
        numCells=numel(ppGratingsMean);
    end
    ppGratingsMean(numCells+1)=struct('contrastArray',stats.contrastArray, 'ratio1',stats.ratio1, 'ratio2',stats.ratio2, 'amp1', stats.amp1, ...
        'amp2', stats.amp2, 'amp3', stats.amp3,'amp4', stats.amp4); 
    save('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/ppGratingsMean.mat','ppGratingsMean');
end

%% plot the summary for pp gratings mean 
load('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/ppGratingsMeanSpike.mat','ppGratingsMean');

for i=1:size(ppGratingsMean,2)
    tp1(i,:)=ppGratingsMean(i).amp2(1:2) ;
    tp2(i,:)=ppGratingsMean(i).amp4(1:2);
end
figure; hold all;
errorbar(ppGratingsMean(1).contrastArray(1:2)*100, mean(tp1), std(tp1)/sqrt(size(tp1,1))); 
errorbar(ppGratingsMean(1).contrastArray(1:2)*100, mean(tp2), std(tp2)/sqrt(size(tp2,1))); 
initFig(gca,'step Contrast', 'amplitude');

%% analyze  grating with varialbe intervals
clc;
clear ppGratingsInterval 
paras.saveCell=0;
CloseAllFiguresExceptGUI;
paras.psthSigma=10;
paras.spikeTh=1.2;
paras.sampleRate=1e4;
selectedNodes = gui.getSelectedEpochTreeNodes;
stimTime=selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');
preTime=selectedNodes{1}.epochList.firstValue.protocolSettings('preTime');
tailTime=selectedNodes{1}.epochList.firstValue.protocolSettings('tailTime');
flashDuration=selectedNodes{1}.epochList.firstValue.protocolSettings('grateDuration');
paras.flashContrast=selectedNodes{1}.epochList.firstValue.protocolSettings('grateContrast');

paras.stepContrast=selectedNodes{1}.epochList.firstValue.protocolSettings('stepContrast');
paras.psth=selectedNodes{1}.epochList.firstValue.protocolSettings('psth');
timeToPts=@(x) x/1e3*paras.sampleRate;
paras.stimPts=timeToPts(stimTime);
paras.prePts=timeToPts(preTime);
paras.tailPts=timeToPts(tailTime);
paras.flashPts=timeToPts(flashDuration);
stats=analyzePPGratingsInterval(selectedNodes{1},paras);

% save the stats for population analysis
if paras.saveCell
    numCells=0;
    try
        load('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/ppGratingsInterval.mat');
        numCells=numel(ppGratingsInterval);
    end
    ppGratingsInterval(numCells+1)=struct('contrastArray',stats.contrastArray, 'ratio1',stats.ratio1, 'ratio2',stats.ratio2, 'amp1', stats.amp1, ...
        'amp2', stats.amp2, 'amp3', stats.amp3,'amp4', stats.amp4); 
    save('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/ppGratingsInterval.mat','ppGratingsInterval');
end