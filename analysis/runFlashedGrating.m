% runFlashedGrating.m - Flashed grating analysis
%   Paper reference: Figure 3 (flashed grating onset/offset), Figure 5 (pharmacology)
%   Requires: main.m to be run first (sets up listSorted, gui, summaryFolder)
%
%   Analyzes flashed grating responses (onset and offset) across bar sizes.
%   Includes drug experiment saving and population-level summaries with
%   onset/offset area sum, amplitude, and I/E ratio analysis.

%% Analyze flashed gratings
clc; selectedNodes = gui.getSelectedEpochTreeNodes;
CloseAllFiguresExceptGUI;
paras.spikeTh=1.2;
paras.spikeTag=0;
paras.psthSigma=10;
paras.spikeoffset=0;
paras.wcoffset=100;
preTime=selectedNodes{1}.epochList.firstValue.protocolSettings('preTime');
stimTime=selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');
tailTime=selectedNodes{1}.epochList.firstValue.protocolSettings('tailTime');
paras.sampleRate=selectedNodes{1}.epochList.firstValue.protocolSettings('sampleRate');
timeToPts=@(x) x/1e3*paras.sampleRate;
paras.prePts=timeToPts(preTime); paras.stimPts=timeToPts(stimTime); paras.tailPts=timeToPts(tailTime);
[f,stats]=analyzeFlashGrating(selectedNodes,paras); ax=gca(f(1));
fprintf('%s %d %s %d \n','preTime-- ',preTime, ' --stimTime-- ', stimTime);

%% Save cell data for population analysis
clc; clear flashGSummary;
stats.onlineAnalysis
meanLuminance=input('enter the mean luminance:');
numCells=0;
try
    load(fullfile(summaryFolder, 'flashedGrating.mat'));
    numCells=numel(flashGSummary);
end
flashGSummary(numCells+1)=struct('date',stats.expDate,...
    'cellID',stats.cellLabel,...
    'cellType', stats.cellType,...
    'recType',stats.onlineAnalysis,...
    'barList',stats.barList, ...
    'OnsetResponse',stats.onset,...
    'OffsetResponse',stats.offset,...
    'baselineResponse', stats.baseline,...
    'OffsetBaseline', stats.offset_baseline,...
    'OnsetAmplitude', stats.peakOnset,...
    'OffsetAmplitude', stats.peakOffset,...
    'meanLum',meanLuminance);
save(fullfile(summaryFolder, 'flashedGrating.mat'),'flashGSummary');
fprintf('%s \n', '---new cell data saved---');

%% Save drug experiment data
clc; clear flashGDrugSummary;
meanLuminance=100;
stats.onlineAnalysis
numCells=0;
drugUsed=input ('enter the drug used:','s');
try
    load(fullfile(summaryFolder, 'flashedGratingDrug.mat'));
    numCells=numel(flashGDrugSummary);
end
flashGDrugSummary(numCells+1)=struct('date',stats.expDate,'cellID',stats.cellLabel,'cellType', stats.cellType,'onlineAnalysis',stats.onlineAnalysis,'barList',stats.barList, ...
    'OnsetResponse',stats.onset,'OffsetResponse',stats.offset,'OnsetAmp',stats.peakOnset,'OffsetAmp',stats.peakOffset,'baselineResponse', stats.baseline,'meanLum',meanLuminance,'drugUsed',drugUsed);
save(fullfile(summaryFolder, 'flashedGratingDrug.mat'),'flashGDrugSummary');
fprintf('%s \n', '---new cell data saved---');

%% Population level summary analysis of flashed gratings
clc; load(fullfile(summaryFolder, 'flashedGratingOld.mat'));
sumTable=struct2table(flashGSummary);
sumTable.recType=categorical(sumTable.recType); sumTable.cellType=categorical(sumTable.cellType);
CloseAllFiguresExceptGUI;
figure('color','w','position',[200 200 900 900]);
cellTypes=unique(sumTable.cellType);
recTypeInsp='extracellular';
lightLevelInsp=100;
normalizeResponses=false;

% Figure 1: Onset Response (Area Sum)
for c=1:numel(cellTypes)
    resTable=sumTable(sumTable.cellType==cellTypes(c) & sumTable.recType==recTypeInsp& ...
        (sumTable.meanLum==lightLevelInsp ),{'cellType','barList','baselineResponse','OnsetResponse','OffsetResponse'});
    try
        ax(c)=subplot(3,2,c); hold all;
        allBars = [];
        allResponses = {};
        for i=1:size(resTable,1)
            barList = cell2mat(resTable.barList(i));
            onsetResp = cell2mat(resTable.OnsetResponse(i));
            if normalizeResponses
                maxAbsValue = max(abs(onsetResp));
                if maxAbsValue > 1e-6
                    onsetResp = onsetResp / maxAbsValue;
                end
            end
            allResponses{i} = onsetResp;
            allBars = [allBars, barList];
            plot(barList, onsetResp, 'color', [0 0 0 0.3], 'linewidth', 0.5);
        end
        [G, barID{c}] = findgroups(allBars);
        allConcatenatedResponses = cat(2, allResponses{:});
        barMean.(char(cellTypes(c))) = splitapply(@mean, allConcatenatedResponses, G);
        barErr.(char(cellTypes(c))) = splitapply(@(x) std(x)/sqrt(numel(x)), allConcatenatedResponses, G);
        errorbar(barID{c}, barMean.(char(cellTypes(c))), barErr.(char(cellTypes(c))), 'r', 'linewidth', 3);
        title(char(cellTypes(c)));
    end
    if normalizeResponses
        st=sgtitle('Onset Area Sum (Normalized)');
    else
        st=sgtitle('Onset Area Sum');
    end
    set(st,'fontsize',26);
end

% Figure 2: Offset Response (Area Sum)
figure('color','w','position',[200 200 900 900]);
for c=1:numel(cellTypes)
    resTable=sumTable(sumTable.cellType==cellTypes(c) & sumTable.recType==recTypeInsp & ...
        sumTable.meanLum==lightLevelInsp,{'cellType','barList','baselineResponse','OnsetResponse','OffsetResponse'});
    try
        subplot(3,2,c); hold all;
        allBars = [];
        allResponses = {};
        for i=1:size(resTable,1)
            barList = cell2mat(resTable.barList(i));
            offsetResp = cell2mat(resTable.OffsetResponse(i));
            if normalizeResponses
                maxAbsValue = max(abs(offsetResp));
                if maxAbsValue > 1e-6
                    offsetResp = offsetResp / maxAbsValue;
                end
            end
            allResponses{i} = offsetResp;
            allBars = [allBars, barList];
            plot(barList, offsetResp, 'color', [0 0 0 0.3], 'linewidth', 0.5);
        end
        [G, barID{c}] = findgroups(allBars);
        allConcatenatedResponses = cat(2, allResponses{:});
        barMean.(char(cellTypes(c))) = splitapply(@mean, allConcatenatedResponses, G);
        barErr.(char(cellTypes(c))) = splitapply(@(x) std(x)/sqrt(numel(x)), allConcatenatedResponses, G);
        errorbar(barID{c}, barMean.(char(cellTypes(c))), barErr.(char(cellTypes(c))), 'r', 'linewidth', 3);
        title(char(cellTypes(c)));
    end
    if normalizeResponses
        st=sgtitle('Offset Area Sum (Normalized)');
    else
        st=sgtitle('Offset Area Sum');
    end
    set(st,'fontsize',26);
end

try
    % Figure 3: Onset Amplitude
    figure('color','w','position',[200 200 900 900]);
    for c=1:numel(cellTypes)
        resTable=sumTable(sumTable.cellType==cellTypes(c) & sumTable.recType==recTypeInsp& ...
            (sumTable.meanLum==lightLevelInsp ),{'cellType','barList','OnsetAmplitude','OffsetAmplitude'});
        try
            subplot(3,2,c); hold all;
            allBars = [];
            allResponses = {};
            for i=1:size(resTable,1)
                barList = cell2mat(resTable.barList(i));
                onsetAmp = cell2mat(resTable.OnsetAmplitude(i));
                if normalizeResponses
                    maxAbsValue = max(abs(onsetAmp));
                    if maxAbsValue > 1e-6
                        onsetAmp = onsetAmp / maxAbsValue;
                    end
                end
                allResponses{i} = onsetAmp;
                allBars = [allBars, barList];
                plot(barList, onsetAmp, 'color', [0 0 0 0.3], 'linewidth', 0.5);
            end
            [G, barID{c}] = findgroups(allBars);
            allConcatenatedResponses = cat(2, allResponses{:});
            barMean.(char(cellTypes(c))) = splitapply(@mean, allConcatenatedResponses, G);
            barErr.(char(cellTypes(c))) = splitapply(@(x) std(x)/sqrt(numel(x)), allConcatenatedResponses, G);
            errorbar(barID{c}, barMean.(char(cellTypes(c))), barErr.(char(cellTypes(c))), 'r', 'linewidth', 3);
            title(char(cellTypes(c)));
        end
        if normalizeResponses
            st=sgtitle('Onset Amplitude (Normalized)');
        else
            st=sgtitle('Onset Amplitude');
        end
        set(st,'fontsize',26);
    end

    % Figure 4: Offset Amplitude
    figure('color','w','position',[200 200 900 900]);
    for c=1:numel(cellTypes)
        resTable=sumTable(sumTable.cellType==cellTypes(c) & sumTable.recType==recTypeInsp & ...
            sumTable.meanLum==lightLevelInsp,{'cellType','barList','OnsetAmplitude','OffsetAmplitude'});
        try
            subplot(3,2,c); hold all;
            allBars = [];
            allResponses = {};
            for i=1:size(resTable,1)
                barList = cell2mat(resTable.barList(i));
                offsetAmp = cell2mat(resTable.OffsetAmplitude(i));
                if normalizeResponses
                    maxAbsValue = max(abs(offsetAmp));
                    if maxAbsValue > 1e-6
                        offsetAmp = offsetAmp / maxAbsValue;
                    end
                end
                allResponses{i} = offsetAmp;
                allBars = [allBars, barList];
                plot(barList, offsetAmp, 'color', [0 0 0 0.3], 'linewidth', 0.5);
            end
            [G, barID{c}] = findgroups(allBars);
            allConcatenatedResponses = cat(2, allResponses{:});
            barMean.(char(cellTypes(c))) = splitapply(@mean, allConcatenatedResponses, G);
            barErr.(char(cellTypes(c))) = splitapply(@(x) std(x)/sqrt(numel(x)), allConcatenatedResponses, G);
            errorbar(barID{c}, barMean.(char(cellTypes(c))), barErr.(char(cellTypes(c))), 'r', 'linewidth', 3);
            title(char(cellTypes(c)));
        end
        if normalizeResponses
            st=sgtitle('Offset Amplitude (Normalized)');
        else
            st=sgtitle('Offset Amplitude');
        end
        set(st,'fontsize',26);
    end
end

% Figure 5: Scatter plot of area sum onset vs offset
f1=figure('color','w','position',[300 300 600 600]);
barToInspect=80; cellTypeInsp='OffT';
resTable=sumTable(sumTable.cellType==cellTypeInsp & sumTable.recType==recTypeInsp & ...
    sumTable.meanLum==lightLevelInsp,{'cellType','barList','baselineResponse','OnsetResponse','OffsetResponse'});
barLists=cat(2,resTable.barList{:}); res.onset=cat(2,resTable.OnsetResponse{:}); res.offset=cat(2,resTable.OffsetResponse{:});
res.onset=res.onset(barLists==barToInspect);   res.offset=res.offset(barLists==barToInspect);
meanOnset = mean(res.onset);
meanOffset = mean(res.offset);
semOnset = std(res.onset)/sqrt(numel(res.onset));
semOffset = std(res.offset)/sqrt(numel(res.offset));
hold all;
scatter(res.onset, res.offset, 100, 'k', 'filled', 'MarkerFaceAlpha', 0.7);
h = errorbar(meanOnset, meanOffset, semOffset, semOffset, semOnset, semOnset, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'LineWidth', 2);
h.CapSize = 10;
xline(0, 'k--');
yline(0, 'k--');
xmin = min(min(res.onset), 0) * 1.1;
xmax = max(max(res.onset), 0) * 1.1;
ymin = min(min(res.offset), 0) * 1.1;
ymax = max(max(res.offset), 0) * 1.1;
axisLim = [min(xmin, ymin), max(xmax, ymax)];
plot(axisLim, axisLim, 'k--');
axis equal;
xlim(axisLim);
ylim(axisLim);
title([char(cellTypeInsp), ' Area Sum, Bar Width = ', num2str(barToInspect)], 'FontSize', 14);
hold off;
setAxes(f1);
initFig(gca(f1), 'Onset Area Sum', 'Offset Area Sum');

try
    % Figure 6: Scatter plot of amplitude onset vs offset
    f2=figure('color','w','position',[300 300 600 600]);
    resTable=sumTable(sumTable.cellType==cellTypeInsp & sumTable.recType==recTypeInsp & ...
        sumTable.meanLum==lightLevelInsp,{'cellType','barList','OnsetAmplitude','OffsetAmplitude'});
    barLists=cat(2,resTable.barList{:}); res.amp_onset=cat(2,resTable.OnsetAmplitude{:}); res.amp_offset=cat(2,resTable.OffsetAmplitude{:});
    res.amp_onset=res.amp_onset(barLists==barToInspect);   res.amp_offset=res.amp_offset(barLists==barToInspect);
    meanOnsetAmp = mean(res.amp_onset);
    meanOffsetAmp = mean(res.amp_offset);
    semOnsetAmp = std(res.amp_onset)/sqrt(numel(res.amp_onset));
    semOffsetAmp = std(res.amp_offset)/sqrt(numel(res.amp_offset));
    hold all;
    scatter(res.amp_onset, res.amp_offset, 100, 'k', 'filled', 'MarkerFaceAlpha', 0.7);
    h = errorbar(meanOnsetAmp, meanOffsetAmp, semOffsetAmp, semOffsetAmp, semOnsetAmp, semOnsetAmp, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'LineWidth', 2);
    h.CapSize = 10;
    xline(0, 'k--');
    yline(0, 'k--');
    xmin = min(min(res.amp_onset), 0) * 1.1;
    xmax = max(max(res.amp_onset), 0) * 1.1;
    ymin = min(min(res.amp_offset), 0) * 1.1;
    ymax = max(max(res.amp_offset), 0) * 1.1;
    axisLim = [min(xmin, ymin), max(xmax, ymax)];
    plot(axisLim, axisLim, 'k--');
    axis equal;
    xlim(axisLim);
    ylim(axisLim);
    title([char(cellTypeInsp), ' Amplitude, Bar Width = ', num2str(barToInspect)], 'FontSize', 14);
    hold off;
    setAxes(f2);
    initFig(gca(f2), 'Onset Amplitude', 'Offset Amplitude');
end

% Figure 7: I/E ratio analysis
IETable=sumTable(sumTable.cellType=='OffT' & sumTable.recType~='extracellular' & ...
    sumTable.meanLum==100,{'date','cellID','recType','barList','OnsetResponse','OffsetResponse'});
[G,ID]=findgroups(IETable(:,{'date', 'cellID'})); gList=unique(G);
figure('color','w','position',[400 400 800 400]);
allBarSizes = [];
for i=1:size(IETable,1)
    barSizes = IETable.barList{i};
    allBarSizes = [allBarSizes, barSizes];
end
uniqueBarSizes = unique(allBarSizes);
ieRatio.onset = cell(length(uniqueBarSizes), 1);
ieRatio.offset = cell(length(uniqueBarSizes), 1);
for i=1:length(uniqueBarSizes)
    ieRatio.onset{i} = [];
    ieRatio.offset{i} = [];
end
for i=1:numel(gList)
    gIndex=find(gList(i)==G);
    if numel(gIndex)==2
        excIdx = find(strcmp(IETable.recType(gIndex), 'exc'));
        inhIdx = find(strcmp(IETable.recType(gIndex), 'inh'));
        if ~isempty(excIdx) && ~isempty(inhIdx)
            excIdx = gIndex(excIdx);
            inhIdx = gIndex(inhIdx);
            barList_exc = IETable.barList{excIdx};
            barList_inh = IETable.barList{inhIdx};
            onset_exc = IETable.OnsetResponse{excIdx};
            onset_inh = IETable.OnsetResponse{inhIdx};
            offset_exc = IETable.OffsetResponse{excIdx};
            offset_inh = IETable.OffsetResponse{inhIdx};
            for j=1:length(uniqueBarSizes)
                barSize = uniqueBarSizes(j);
                exc_barIdx = find(barList_exc == barSize);
                inh_barIdx = find(barList_inh == barSize);
                if ~isempty(exc_barIdx) && ~isempty(inh_barIdx)
                    for e_idx = 1:length(exc_barIdx)
                        for i_idx = 1:length(inh_barIdx)
                            excVal = onset_exc(exc_barIdx(e_idx));
                            inhVal = onset_inh(inh_barIdx(i_idx));
                            if abs(excVal + inhVal) > 1e-6
                                onsetRatio = (inhVal - excVal) / (inhVal + excVal);
                                ieRatio.onset{j} = [ieRatio.onset{j}, onsetRatio];
                            end
                            excVal = offset_exc(exc_barIdx(e_idx));
                            inhVal = offset_inh(inh_barIdx(i_idx));
                            if abs(excVal + inhVal) > 1e-6
                                offsetRatio = (inhVal - excVal) / (inhVal + excVal);
                                ieRatio.offset{j} = [ieRatio.offset{j}, offsetRatio];
                            end
                        end
                    end
                end
            end
        end
    end
end
onset_means = zeros(length(uniqueBarSizes), 1);
onset_sems = zeros(length(uniqueBarSizes), 1);
offset_means = zeros(length(uniqueBarSizes), 1);
offset_sems = zeros(length(uniqueBarSizes), 1);
validSizes = [];
for i=1:length(uniqueBarSizes)
    if ~isempty(ieRatio.onset{i}) && ~isempty(ieRatio.offset{i})
        onset_means(i) = mean(ieRatio.onset{i});
        onset_sems(i) = std(ieRatio.onset{i}) / sqrt(length(ieRatio.onset{i}));
        offset_means(i) = mean(ieRatio.offset{i});
        offset_sems(i) = std(ieRatio.offset{i}) / sqrt(length(ieRatio.offset{i}));
        validSizes = [validSizes, uniqueBarSizes(i)];
    end
end
subplot(1,2,1);
errorbar(uniqueBarSizes, onset_means, onset_sems, 'o-', 'LineWidth', 2, 'MarkerFaceColor', 'auto');
hold on;
yline(0, 'k--');
xlabel('Bar Size');
ylabel('I-E / I+E Ratio');
title('Onset IE Ratio');
xlim([min(uniqueBarSizes)*0.8, max(uniqueBarSizes)*1.2]);
ylim([-1.1, 1.1]);
grid on;

subplot(1,2,2);
errorbar(uniqueBarSizes, offset_means, offset_sems, 'o-', 'LineWidth', 2, 'MarkerFaceColor', 'auto');
hold on;
yline(0, 'k--');
xlabel('Bar Size');
ylabel('I-E / I+E Ratio');
title('Offset IE Ratio');
xlim([min(uniqueBarSizes)*0.8, max(uniqueBarSizes)*1.2]);
ylim([-1.1, 1.1]);
grid on;
sgtitle('Inhibition/Excitation Balance Across Bar Sizes', 'FontSize', 16);

barInsp=80;
count=0;
for i=1:numel(gList)
    gIndex=find(gList(i)==G);
    if numel(gIndex)==2
        count=count+1;
        barList=cell2mat(IETable(gIndex(1),:).barList);  barIndex=find(barList==barInsp);
        tp.(char(IETable(gIndex(1),:).recType)).onset= cell2mat(IETable(gIndex(1),:).OnsetResponse);
        tp.(char(IETable(gIndex(2),:).recType)).onset=cell2mat(IETable(gIndex(2),:).OnsetResponse);
        rt= (tp.inh.onset-tp.exc.onset)./(tp.inh.onset+tp.exc.onset);
        ratio.onset(count)=rt(barIndex);
        tp.(char(IETable(gIndex(1),:).recType)).offset= cell2mat(IETable(gIndex(1),:).OffsetResponse);
        tp.(char(IETable(gIndex(2),:).recType)).offset=cell2mat(IETable(gIndex(2),:).OffsetResponse);
        rt=(tp.inh.offset-tp.exc.offset)./(tp.inh.offset+tp.exc.offset); ratio.offset(count)=rt(barIndex);
    end
end
