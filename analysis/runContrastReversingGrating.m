% runContrastReversingGrating.m - Contrast reversing grating analysis
%   Paper reference: Figure 4 (subunit spatial tuning, F2 analysis)
%   Requires: main.m to be run first (sets up listSorted, summaryFolder)
%
%   Analyzes contrast-reversing grating responses to characterize subunit
%   spatial structure via F2 (second harmonic) analysis. Computes E/I
%   balance and population summaries across cell types.

%% Create GUI for contrast reversing grating
cellTypeSplit = @(listSorted)splitOnCellType(listSorted);
cellTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, cellTypeSplit);

dateSplit = @(listSorted)splitOnExperimentDate(listSorted);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, dateSplit);

ndfSplit = @(listSorted)splitOnNDFs(listSorted);
ndfSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, ndfSplit);

brightnessSplit = @(listSorted)splitOnDeviceBrightNess(listSorted);
brightnessSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, brightnessSplit);

tree = riekesuite.analysis.buildTree(listSorted,{cellTypeSplit_java,dateSplit_java, 'cell.label', 'protocolSettings(epochGroup:label)',...
    brightnessSplit_java, ndfSplit_java,'protocolSettings(onlineAnalysis)'});
gui = epochTreeGUI(tree);

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
