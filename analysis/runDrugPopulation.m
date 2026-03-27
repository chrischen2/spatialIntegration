% runDrugPopulation.m - Drug population analysis
%   Paper reference: Figure 5 (APB/LY, strychnine pharmacology)
%   Requires: main.m to be run first (sets up summaryFolder)
%
%   Analyzes population-level drug effects (LY, strychnine) on flashed
%   grating responses. Compares control vs drug conditions for onset/offset
%   responses and amplitudes across bar sizes.

%% Analyze drug population data
clc; clear barMean barErr
load(fullfile(summaryFolder, 'flashedGratingDrugOld.mat'));
sumTable=struct2table(flashGDrugSummary);
sumTable.onlineAnalysis=categorical(sumTable.onlineAnalysis);
sumTable.cellType=categorical(sumTable.cellType);
sumTable.drugUsed=categorical(sumTable.drugUsed);
CloseAllFiguresExceptGUI;

% Parameters to adjust
cellTypeToInsp='OffT';
recTypeInsp='exc';
drugInsp='LY';
barInsp=80;
conds={'control', drugInsp};

% Get relevant data
resTable=sumTable(sumTable.cellType==cellTypeToInsp & sumTable.onlineAnalysis==recTypeInsp & ...
    (sumTable.drugUsed==drugInsp | sumTable.drugUsed=='control'),...
    {'date','cellID','barList','OnsetResponse','OffsetResponse','OnsetAmp','OffsetAmp','drugUsed'});

[G,~]=findgroups(resTable(:,{'date','cellID'}));
gList=unique(G);
drugG=[];

% Figure 1: Paired cell responses for barInsp (onset response)
f2=figure('color','w','position',[700 200 500 500]);
subplot(2,2,1); hold all;
title(['Onset Response - ' num2str(barInsp) ' deg bar']);
for i=1:numel(gList)
    temp=find(gList(i)==G);
    if numel(temp)==2
        if resTable.drugUsed(temp(1))~=resTable.drugUsed(temp(2))
            drugG=[drugG;find(gList(i)==G)];
            xplot=[find(ismember(conds,'control')); find(ismember(conds,drugInsp))];
            bars=cat(2,resTable.barList{temp});
            resps=cat(2,resTable.OnsetResponse{temp});
            plot(xplot,-resps(bars==barInsp),'-ko','markersize',15);
        end
    end
end
set(gca,'xtick',[1 2],'xticklabel',{'control', drugInsp});
xlim([0.7 2.3]);
if strcmp(recTypeInsp,'exc')
    ylabel('Response Amplitude');
else
    ylabel('Spike Rate (spikes/s)');
end

% Paired cell responses (offset)
subplot(2,2,2); hold all;
title(['Offset Response - ' num2str(barInsp) ' deg bar']);
for i=1:numel(gList)
    temp=find(gList(i)==G);
    if numel(temp)==2
        if resTable.drugUsed(temp(1))~=resTable.drugUsed(temp(2))
            xplot=[find(ismember(conds,'control')); find(ismember(conds,drugInsp))];
            bars=cat(2,resTable.barList{temp});
            resps=cat(2,resTable.OffsetResponse{temp});
            plot(xplot,-resps(bars==barInsp),'-ko','markersize',15);
        end
    end
end
set(gca,'xtick',[1 2],'xticklabel',{'control', drugInsp});
xlim([0.7 2.3]);
if strcmp(recTypeInsp,'exc')
    ylabel('Response Amplitude');
else
    ylabel('Spike Rate (spikes/s)');
end

% Paired peak amplitudes (onset)
subplot(2,2,3); hold all;
title('Onset Amplitude');
for i=1:numel(gList)
    temp=find(gList(i)==G);
    if numel(temp)==2
        if resTable.drugUsed(temp(1))~=resTable.drugUsed(temp(2))
            xplot=[find(ismember(conds,'control')); find(ismember(conds,drugInsp))];
            onsetAmps = zeros(1, length(temp));
            for j = 1:length(temp)
                if iscell(resTable.OnsetAmp)
                    ampValue = resTable.OnsetAmp{temp(j)};
                    if iscell(ampValue)
                        ampValue = ampValue{1};
                    elseif length(ampValue) > 1
                        ampValue = ampValue(1);
                    end
                    onsetAmps(j) = ampValue;
                else
                    onsetAmps(j) = resTable.OnsetAmp(temp(j));
                end
            end
            if strcmp(recTypeInsp,'exc')
                plot(xplot, -onsetAmps, '-ko', 'markersize', 15);
            else
                plot(xplot, onsetAmps, '-ko', 'markersize', 15);
            end
        end
    end
end
set(gca,'xtick',[1 2],'xticklabel',{'control', drugInsp});
xlim([0.7 2.3]);
if strcmp(recTypeInsp,'exc')
    ylabel('Peak Amplitude');
else
    ylabel('Peak Spike Rate (spikes/s)');
end

% Paired peak amplitudes (offset)
subplot(2,2,4); hold all;
title('Offset Amplitude');
for i=1:numel(gList)
    temp=find(gList(i)==G);
    if numel(temp)==2
        if resTable.drugUsed(temp(1))~=resTable.drugUsed(temp(2))
            xplot=[find(ismember(conds,'control')); find(ismember(conds,drugInsp))];
            offsetAmps = zeros(1, length(temp));
            for j = 1:length(temp)
                if iscell(resTable.OffsetAmp)
                    ampValue = resTable.OffsetAmp{temp(j)};
                    if iscell(ampValue)
                        ampValue = ampValue{1};
                    elseif length(ampValue) > 1
                        ampValue = ampValue(1);
                    end
                    offsetAmps(j) = ampValue;
                else
                    offsetAmps(j) = resTable.OffsetAmp(temp(j));
                end
            end
            if strcmp(recTypeInsp,'exc')
                plot(xplot, -offsetAmps, '-ko', 'markersize', 15);
            else
                plot(xplot, offsetAmps, '-ko', 'markersize', 15);
            end
        end
    end
end
set(gca,'xtick',[1 2],'xticklabel',{'control', drugInsp});
xlim([0.7 2.3]);
if strcmp(recTypeInsp,'exc')
    ylabel('Peak Amplitude');
else
    ylabel('Peak Spike Rate (spikes/s)');
end
sgtitle([cellTypeToInsp ' cells - ' drugInsp ' vs control'], 'FontSize', 14);

% Use filtered data for subsequent analysis
resTable=resTable(drugG,:);

% Figure 2: Size tuning curves
f1=figure('color','w','position',[200 200 1000 500]);
subplot(2,2,1); hold all;
title('Onset Response by Bar Size');
for i=1:numel(conds)
    tempTable=resTable(resTable.drugUsed==conds{i},:);
    [G,barID] = findgroups(cat(2,tempTable.barList{:}));
    barMean=splitapply(@mean,cat(2,tempTable.OnsetResponse{:}),G);
    barErr=splitapply(@(x) std(x)/sqrt(numel(x)),cat(2,tempTable.OnsetResponse{:}),G);
    if strcmp(recTypeInsp,'exc')
        barMean=-barMean;
        barErr=-barErr;
    end
    errorbar(barID, barMean, barErr, 'linewidth', 3);
end
xlabel('Bar Size (deg)');
if strcmp(recTypeInsp,'exc')
    ylabel('Response Amplitude');
else
    ylabel('Spike Rate (spikes/s)');
end
legend(conds); legend boxoff;

subplot(2,2,2); hold all;
title('Offset Response by Bar Size');
for i=1:numel(conds)
    tempTable=resTable(resTable.drugUsed==conds{i},:);
    [G,barID] = findgroups(cat(2,tempTable.barList{:}));
    barMean=splitapply(@mean,cat(2,tempTable.OffsetResponse{:}),G);
    barErr=splitapply(@(x) std(x)/sqrt(numel(x)),cat(2,tempTable.OffsetResponse{:}),G);
    if strcmp(recTypeInsp,'exc')
        barMean=-barMean;
        barErr=-barErr;
    end
    errorbar(barID, barMean, barErr, 'linewidth', 3);
end
xlabel('Bar Size (deg)');
ylabel('Response (spikes/s)');
legend(conds); legend boxoff;

% Onset amplitude by cell
subplot(2,2,3); hold all;
title('Onset Amplitude by Cell');
cellIDs = unique(resTable.cellID);
cellX = 1:length(cellIDs);
amplitudes = zeros(length(cellIDs), 2);
for i = 1:length(cellIDs)
    cellData = resTable(strcmp(resTable.cellID, cellIDs{i}), :);
    for j = 1:numel(conds)
        condData = cellData(cellData.drugUsed == conds{j}, :);
        if ~isempty(condData)
            if iscell(condData.OnsetAmp)
                ampValue = condData.OnsetAmp{1};
                if iscell(ampValue)
                    ampValue = ampValue{1};
                elseif length(ampValue) > 1
                    ampValue = ampValue(1);
                end
            else
                ampValue = condData.OnsetAmp;
            end
            if strcmp(recTypeInsp,'exc')
                amplitudes(i, j) = -ampValue;
            else
                amplitudes(i, j) = ampValue;
            end
        end
    end
end
bar(cellX, amplitudes);
set(gca, 'XTick', cellX, 'XTickLabel', cellIDs);
xtickangle(45);
legend(conds); legend boxoff;
ylabel('Amplitude (spikes/s)');

% Offset amplitude by cell
subplot(2,2,4); hold all;
title('Offset Amplitude by Cell');
amplitudes = zeros(length(cellIDs), 2);
for i = 1:length(cellIDs)
    cellData = resTable(strcmp(resTable.cellID, cellIDs{i}), :);
    for j = 1:numel(conds)
        condData = cellData(cellData.drugUsed == conds{j}, :);
        if ~isempty(condData)
            if iscell(condData.OffsetAmp)
                ampValue = condData.OffsetAmp{1};
                if iscell(ampValue)
                    ampValue = ampValue{1};
                elseif length(ampValue) > 1
                    ampValue = ampValue(1);
                end
            else
                ampValue = condData.OffsetAmp;
            end
            if strcmp(recTypeInsp,'exc')
                amplitudes(i, j) = -ampValue;
            else
                amplitudes(i, j) = ampValue;
            end
        end
    end
end
bar(cellX, amplitudes);
set(gca, 'XTick', cellX, 'XTickLabel', cellIDs);
xtickangle(45);
legend(conds); legend boxoff;
ylabel('Amplitude (spikes/s)');
sgtitle([cellTypeToInsp ' cells - ' drugInsp ' vs control'], 'FontSize', 14);

%% Comparison across multiple bar sizes with significance testing
f3 = figure('color','w','position',[200 700 1000 500]);
allBarSizes = [];
for i = 1:size(resTable,1)
    allBarSizes = [allBarSizes, resTable.barList{i}];
end
uniqueBarSizes = unique(allBarSizes);
if length(uniqueBarSizes) > 8
    uniqueBarSizes = uniqueBarSizes(uniqueBarSizes >= 20);
    if length(uniqueBarSizes) > 8
        uniqueBarSizes = uniqueBarSizes(1:2:end);
    end
end

subplot(1,2,1); hold all;
title('Onset Response by Bar Size');
ctrlOnsetMeans = zeros(size(uniqueBarSizes));
ctrlOnsetSEMs = zeros(size(uniqueBarSizes));
drugOnsetMeans = zeros(size(uniqueBarSizes));
drugOnsetSEMs = zeros(size(uniqueBarSizes));
pValues = zeros(size(uniqueBarSizes));
for i = 1:length(uniqueBarSizes)
    currentBar = uniqueBarSizes(i);
    ctrlData = [];
    drugData = [];
    for j = 1:size(resTable,1)
        bars = resTable.barList{j};
        resps = resTable.OnsetResponse{j};
        barIdx = bars == currentBar;
        if ~isempty(barIdx) && any(barIdx)
            if resTable.drugUsed(j) == 'control'
                if strcmp(recTypeInsp,'exc')
                    ctrlData = [ctrlData; -resps(barIdx)];
                else
                    ctrlData = [ctrlData; resps(barIdx)];
                end
            else
                if strcmp(recTypeInsp,'exc')
                    drugData = [drugData; -resps(barIdx)];
                else
                    drugData = [drugData; resps(barIdx)];
                end
            end
        end
    end
    ctrlOnsetMeans(i) = mean(ctrlData);
    ctrlOnsetSEMs(i) = std(ctrlData) / sqrt(length(ctrlData));
    drugOnsetMeans(i) = mean(drugData);
    drugOnsetSEMs(i) = std(drugData) / sqrt(length(drugData));
    if length(ctrlData) > 1 && length(drugData) > 1
        [~, pValues(i)] = ttest2(ctrlData, drugData);
    else
        pValues(i) = NaN;
    end
end
errorbar(uniqueBarSizes, ctrlOnsetMeans, ctrlOnsetSEMs, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
errorbar(uniqueBarSizes, drugOnsetMeans, drugOnsetSEMs, 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
for i = 1:length(uniqueBarSizes)
    if ~isnan(pValues(i)) && pValues(i) < 0.05
        yPos = max([ctrlOnsetMeans(i) + ctrlOnsetSEMs(i), drugOnsetMeans(i) + drugOnsetSEMs(i)]) * 1.1;
        text(uniqueBarSizes(i), yPos, '*', 'FontSize', 16, 'HorizontalAlignment', 'center');
    end
end
xlabel('Bar Size (deg)');
if strcmp(recTypeInsp,'exc')
    ylabel('Onset Response Amplitude');
else
    ylabel('Onset Spike Rate (spikes/s)');
end
legend('Control', drugInsp, 'Location', 'best');
legend boxoff;

subplot(1,2,2); hold all;
title('Offset Response by Bar Size');
ctrlOffsetMeans = zeros(size(uniqueBarSizes));
ctrlOffsetSEMs = zeros(size(uniqueBarSizes));
drugOffsetMeans = zeros(size(uniqueBarSizes));
drugOffsetSEMs = zeros(size(uniqueBarSizes));
pValues = zeros(size(uniqueBarSizes));
for i = 1:length(uniqueBarSizes)
    currentBar = uniqueBarSizes(i);
    ctrlData = [];
    drugData = [];
    for j = 1:size(resTable,1)
        bars = resTable.barList{j};
        resps = resTable.OffsetResponse{j};
        barIdx = bars == currentBar;
        if ~isempty(barIdx) && any(barIdx)
            if resTable.drugUsed(j) == 'control'
                if strcmp(recTypeInsp,'exc')
                    ctrlData = [ctrlData; -resps(barIdx)];
                else
                    ctrlData = [ctrlData; resps(barIdx)];
                end
            else
                if strcmp(recTypeInsp,'exc')
                    drugData = [drugData; -resps(barIdx)];
                else
                    drugData = [drugData; resps(barIdx)];
                end
            end
        end
    end
    ctrlOffsetMeans(i) = mean(ctrlData);
    ctrlOffsetSEMs(i) = std(ctrlData) / sqrt(length(ctrlData));
    drugOffsetMeans(i) = mean(drugData);
    drugOffsetSEMs(i) = std(drugData) / sqrt(length(drugData));
    if length(ctrlData) > 1 && length(drugData) > 1
        [~, pValues(i)] = ttest2(ctrlData, drugData);
    else
        pValues(i) = NaN;
    end
end
errorbar(uniqueBarSizes, ctrlOffsetMeans, ctrlOffsetSEMs, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
errorbar(uniqueBarSizes, drugOffsetMeans, drugOffsetSEMs, 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
for i = 1:length(uniqueBarSizes)
    if ~isnan(pValues(i)) && pValues(i) < 0.05
        yPos = max([ctrlOffsetMeans(i) + ctrlOffsetSEMs(i), drugOffsetMeans(i) + drugOffsetSEMs(i)]) * 1.1;
        text(uniqueBarSizes(i), yPos, '*', 'FontSize', 16, 'HorizontalAlignment', 'center');
    end
end
xlabel('Bar Size (deg)');
if strcmp(recTypeInsp,'exc')
    ylabel('Offset Response Amplitude');
else
    ylabel('Offset Spike Rate (spikes/s)');
end
legend('Control', drugInsp, 'Location', 'best');
legend boxoff;
sgtitle([cellTypeToInsp ' cells - ' drugInsp ' vs control across bar sizes'], 'FontSize', 14);
