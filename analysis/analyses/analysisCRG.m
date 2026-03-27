% analysisCRG.m - Contrast reversing grating analysis
%   Paper reference: Figure 4 (subunit spatial tuning, F2), Figure 4F/6C (E/I ratio)
%   Requires: main.m to be run first (sets up listSorted, gui, summaryFolder)
%
%   Analyzes contrast-reversing grating responses via F2 analysis, computes
%   E/I balance, and population-level cross-correlation summaries.

%% Analyze contrast reversing grating
selectedNodes = gui.getSelectedEpochTreeNodes;
paras.spikeTh=1.5;
clc; CloseAllFiguresExceptGUI;
paras.spikeTag=0;
paras.psthSigma=10;
paras.stimTime=selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');
paras.preTime=selectedNodes{1}.epochList.firstValue.protocolSettings('preTime');
paras.tailTime=selectedNodes{1}.epochList.firstValue.protocolSettings('tailTime');
paras.sampleRate=selectedNodes{1}.epochList.firstValue.protocolSettings('sampleRate');
paras.tempFreq=selectedNodes{1}.epochList.firstValue.protocolSettings('temporalFrequency');
paras.meanIntensity=selectedNodes{1}.epochList.firstValue.protocolSettings('backgroundIntensity');
timeToPts=@(x) x/1e3*paras.sampleRate;
paras.prePts=timeToPts(paras.preTime);
paras.stimPts=timeToPts(paras.stimTime);
paras.tailPts=timeToPts(paras.tailTime);
paras.CoMTh=0.5;
[output]=analyzeContrastReversingGrating(selectedNodes, paras);
fprintf('%s , %f \n', 'temporal freq::',paras.tempFreq);

%% Save E/I amplitude across light levels (OffT only)
clc; clear LightLevelEISummary;
numCells=0;
example=1;
meanLuminance=input ('enter the mean luminance:');
expDate=datestr(selectedNodes{1}.epochList.elements(1).startDate','yyyy/mm/dd');
cellLabel=selectedNodes{1}.epochList.firstValue.protocolSettings('source:label');
try
    load(fullfile(summaryFolder, 'LightLevelEISummary.mat'));
    numCells=numel(LightLevelEISummary);
end
LightLevelEISummary(numCells+1)=struct('exampleCell',example, 'date',expDate,'cellID',cellLabel, 'barList',output{1}.barList, ...
    'meanEIOffset',output{1}.meanEIOffset, 'eiOffset',output{1}.eiOffset,'eiRatio',output{1}.eiRatio,'lags', output{1}.lags,'eiCorr',output{1}.eiCorr);
save(fullfile(summaryFolder, 'LightLevelEISummary.mat'),'LightLevelEISummary');

%% Save cell for CRG population summary
clc; clear CRGSummary;
output.onlineAnalysis
meanLuminance=input ('enter the mean luminance:');
cellType=shortCellType(selectedNodes{1}.epochList.firstValue.protocolSettings('source:type'));
recType=output.onlineAnalysis;
numCells=0;
try
    load(fullfile(summaryFolder, 'contrastReversingGrating.mat'));
    numCells=numel(CRGSummary);
end
CRGSummary(numCells+1)=struct('date',output.expDate,'cellID',output.cellLabel,'cellType', cellType,'recType',recType,'tempFreq', paras.tempFreq,'barList',output.barList, ...
    'F2',output.F2,'sinoF2',output.sinoF2,'suppression',output.suppress,'subUnitSize', output.subUnitSize,'meanLum',meanLuminance);
save(fullfile(summaryFolder, 'contrastReversingGrating.mat'),'CRGSummary');
fprintf('%s \n', '---new cell data saved---');

%% Save EI summary
clc; clear EISummary;
numCells=0;
example=1;
expDate=datestr(selectedNodes{1}.epochList.elements(1).startDate','yyyy/mm/dd');
cellLabel=selectedNodes{1}.epochList.firstValue.protocolSettings('source:label');
cellType=shortCellType(selectedNodes{1}.epochList.firstValue.protocolSettings('source:type'));
try
    load(fullfile(summaryFolder, 'EISummary.mat'));
    numCells=numel(EISummary);
end
EISummary(numCells+1)=struct('exampleCell',example, 'date',expDate,'cellID',cellLabel,'cellType', cellType,'barList',output{1}.barList, ...
    'meanEIOffset',output{1}.meanEIOffset, 'eiOffset',output{1}.eiOffset,'eiRatio',output{1}.eiRatio,'lags', output{1}.lags,'eiCorr',output{1}.eiCorr);
save(fullfile(summaryFolder, 'EISummary.mat'),'EISummary');

%% Save spike/exc/inh summary
clc; clear SeiSummary;
numCells=0;
try
    load(fullfile(summaryFolder, 'CRGEIanalysis.mat'));
    numCells=numel(SeiSummary);
end
SeiSummary(numCells+1)=struct('tempFreq',paras.tempFreq,'EI',output{1}.EiRatio,'phaseDiff',output{1}.phaseDiff,'normSpike',output{1}.normSpike);
save(fullfile(summaryFolder, 'CRGEIanalysis.mat'),'SeiSummary');

%% Population level summary analysis of contrast reversing gratings
clc; clear barID
load(fullfile(summaryFolder, 'contrastReversingGrating.mat'));
sumTable=struct2table(CRGSummary);
sumTable.recType=categorical(sumTable.recType); sumTable.cellType=categorical(sumTable.cellType);
subunit=varfun( @(x) mean(x), sumTable(sumTable.meanLum==100 & sumTable.recType=='extracellular',:), 'GroupingVariables', {'cellType','meanLum','recType'},...
    'InputVariables',{'subUnitSize','suppression'},'outputformat','table');
CloseAllFiguresExceptGUI;
figure('color','w','position',[200 200 900 900]);
cellTypes=unique(sumTable.cellType);
cellTypes={'OffT','OffS','OnT','OnS'};
for c=1:numel(cellTypes)
    subTable=sumTable(sumTable.cellType==cellTypes(c) & sumTable.recType=='extracellular' & sumTable.meanLum==100,{'cellID','barList','F2','sinoF2','suppression'});
    validRows = false(size(subTable, 1), 1);
    for i = 1:size(subTable, 1)
        if any(subTable.barList{i} >= 120) && subTable.F2{i}(end) > 0.6
            validRows(i) = true;
        end
    end
    subTable = subTable(validRows, :);
    ax(c)=subplot(3,2,c); hold all;
    for i=1:size(subTable,1)
        plot(cell2mat(subTable.barList(i)), cell2mat(subTable.F2(i)),'color','k','linewidth',0.5);
    end
    [G,barID{c}] = findgroups(cat(1,subTable.barList{:}));
    barMean.(char(cellTypes(c)))=splitapply(@mean,cat(1,subTable.F2{:}),G);
    barErr.(char(cellTypes(c)))=splitapply(@(x) std(x)/sqrt(numel(x)),cat(1,subTable.F2{:}),G);
    errorbar(barID{c}, barMean.(char(cellTypes(c))),barErr.(char(cellTypes(c))),'r','linewidth',3);
    title(char(cellTypes(c)));
end

% Overlay different recTypes for OffT
colors=pmkmp(10,'Isol');
figure('color','w','position',[200 200 900 900]);
recTypes=unique(sumTable.recType);
clear barID
for r=1:numel(recTypes)
    subTable2=sumTable(sumTable.cellType=='OffT' & sumTable.recType==recTypes(r) & sumTable.meanLum==100,{'cellID','barList','F2'});
    hold all;
    [G,barID{r}] = findgroups(cat(1,subTable2.barList{:}));
    barMean.(char(recTypes(r)))=splitapply(@mean,cat(1,subTable2.F2{:}),G);
    scalor=max(barMean.(char(recTypes(r)))); barMean.(char(recTypes(r)))=barMean.(char(recTypes(r)))/scalor;
    barErr.(char(recTypes(r)))=splitapply(@(x) std(x)/sqrt(numel(x)),cat(1,subTable2.F2{:}),G);  barErr.(char(recTypes(r)))=barErr.(char(recTypes(r)))/scalor;
    errorbar(barID{r}, barMean.(char(recTypes(r))),barErr.(char(recTypes(r))),'color',colors(r,:),'linewidth',3);
    title(char(recTypes(r)));
end
legend(char(recTypes)); legend boxoff;

% E/I locality index analysis
recTypes={'exc','inh'};
for r=1:numel(recTypes)
    subTable=sumTable(sumTable.cellType=='OffT' & sumTable.recType==recTypes{r} & sumTable.meanLum==100,{'cellID','barList','F2','sinoF2','suppression'});
    validRows = false(size(subTable, 1), 1);
    if r==1
        for i = 1:size(subTable, 1)
            if any(subTable.barList{i} >= 120) && subTable.F2{i}(end) > 0.7
                validRows(i) = true;
            end
        end
    else
        for i = 1:size(subTable, 1)
            if any(subTable.barList{i} >= 120) && subTable.F2{i}(end) <0.9
                validRows(i) = true;
            end
        end
    end
    subTable = subTable(validRows, :);
    ax(c)=subplot(3,2,c); hold all;
    for i=1:size(subTable,1)
        plot(cell2mat(subTable.barList(i)), cell2mat(subTable.F2(i)),'color','k','linewidth',0.5);
    end
    [G,barID{c}] = findgroups(cat(1,subTable.barList{:}));
    barMean.(char(cellTypes(c)))=splitapply(@mean,cat(1,subTable.F2{:}),G);
    barErr.(char(cellTypes(c)))=splitapply(@(x) std(x)/sqrt(numel(x)),cat(1,subTable.F2{:}),G);
    errorbar(barID{c}, barMean.(char(cellTypes(c))),barErr.(char(cellTypes(c))),'r','linewidth',3);
    title(char(cellTypes(c)));

    localityIndex = [];
    ax(r) = subplot(3, 2, r);
    hold all;
    for i = 1:size(subTable, 1)
        barList = subTable.barList{i};
        F2 = subTable.F2{i};
        if any(barList >= 120) && any(barList == 40)
            F2_160 = F2(end);
            F2_40 = F2(barList == 40);
            localityIndex = [localityIndex F2_160];
            plot(barList, F2, 'color', 'k', 'linewidth', 0.5);
        end
    end
    localityIndices{r} = localityIndex;
    title(char(recTypes{r}));
end

% Scatter plot of locality index
figure;
hold all;
xPositions = [];
jitterAmount=0.1;
for r = 1:numel(recTypes)
    jitteredX = r + (rand(size(localityIndices{r})) - 0.5) * jitterAmount;
    scatter(jitteredX, localityIndices{r}, 50, 'filled', 'DisplayName', recTypes{r});
end
for r = 1:numel(recTypes)
    meanValue = mean(localityIndices{r});
    stderrValue = std(localityIndices{r}) / sqrt(numel(localityIndices{r}));
    errorbar(r, meanValue, stderrValue, 'k', 'LineWidth', 2, 'CapSize', 10);
end
xlim([0.5, numel(recTypes) + 0.5]);
xticks(1:numel(recTypes));
xticklabels(recTypes);
ylabel('Locality Index (F2_{160} / F2_{40})');
legend show;
title('Locality Index for Excitatory and Inhibitory Cells');
hold off;

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
