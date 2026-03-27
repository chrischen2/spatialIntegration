% runLinearDisc.m - Linear equivalent disc analysis
%   Paper reference: Figures 1-2 (NLI, natural image patches vs discs)
%   Requires: main.m to be run first (sets up listSorted, gui, summaryFolder)
%
%   Analyzes linear-equivalent disc responses to compute the Nonlinearity
%   Index (NLI) for onset and offset, comparing natural image patches with
%   uniform discs. Includes population summaries, drug experiments, and
%   cluster analysis.

%% Create GUI for Linear Disc / Flashed Grating
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

%% Analyze linear disc
selectedNodes = gui.getSelectedEpochTreeNodes;
paras.spikeTh=1.6;
clc; CloseAllFiguresExceptGUI;
paras.spikeTag=0;
paras.psthSigma=20;
paras.rmreps=[  ];
preTime=selectedNodes{1}.epochList.firstValue.protocolSettings('preTime');
stimTime=selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');
tailTime=selectedNodes{1}.epochList.firstValue.protocolSettings('tailTime');
paras.sampleRate=selectedNodes{1}.epochList.firstValue.protocolSettings('sampleRate');
paras.wcoffset=100;
paras.spikeoffset=300;
paras.nSamples=30;
paras.plotChoice='multiple';
timeToPts=@(x) x/1e3*paras.sampleRate;
paras.prePts=timeToPts(preTime); paras.stimPts=timeToPts(stimTime); paras.tailPts=timeToPts(tailTime);
output=analyzeLinearDisc(selectedNodes,paras);

%% Save cell information for population analysis
clc; meanLuminance=input ('enter the mean luminance:');

load(fullfile(summaryFolder, 'linearEqvDiscNew.mat'));
for i = 1:length(linearDiscSummary)
    if ~isfield(linearDiscSummary(i), 'respOnset')
        linearDiscSummary(i).respOnset = NaN;
    end
    if ~isfield(linearDiscSummary(i), 'respOffset')
        linearDiscSummary(i).respOffset = NaN;
    end
end

foundMatch = false;
for i = 1:length(linearDiscSummary)
    if isequal(linearDiscSummary(i).date, output.expDate) && ...
            isequal(linearDiscSummary(i).cellID, output.cellLabel) && ...
            isequal(linearDiscSummary(i).onlineAnalysis, output.onlineAnalysis) && ...
            isequal(linearDiscSummary(i).cellType, output.cellType) && ...
            isequal(linearDiscSummary(i).meanLum, meanLuminance)
        linearDiscSummary(i).respOnset = [mean(output.stats.(output.onlineAnalysis).image.onset.mean) ...
            mean(output.stats.(output.onlineAnalysis).disc.onset.mean)];
        linearDiscSummary(i).respOffset = [mean(output.stats.(output.onlineAnalysis).image.offset.mean) ...
            mean(output.stats.(output.onlineAnalysis).disc.offset.mean)];
        linearDiscSummary(i).OnsetNLI= output.NLI.(output.onlineAnalysis).onset;
        linearDiscSummary(i).OffsetNLI= output.NLI.(output.onlineAnalysis).offset;
        foundMatch = true;
        break;
    end
end

if ~foundMatch
    newEntry = struct(...
        'date', output.expDate,...
        'cellID', output.cellLabel,...
        'cellType', output.cellType,...
        'onlineAnalysis', output.onlineAnalysis,...
        'meanOnsetNLI', mean(output.NLI.(output.onlineAnalysis).onset),...
        'meanOffsetNLI', mean(output.NLI.(output.onlineAnalysis).offset),...
        'OnsetNLI', output.NLI.(output.onlineAnalysis).onset,...
        'OffsetNLI', output.NLI.(output.onlineAnalysis).offset,...
        'meanLum', meanLuminance,...
        'respOnset', [mean(output.stats.(output.onlineAnalysis).image.onset.mean) ...
        mean(output.stats.(output.onlineAnalysis).disc.onset.mean)], ...
        'respOffset', [mean(output.stats.(output.onlineAnalysis).image.offset.mean) ...
        mean(output.stats.(output.onlineAnalysis).disc.offset.mean)] ...
        );
    linearDiscSummary(end+1) = newEntry;
    fprintf('Added new entry: %s, %s\n', output.expDate, output.cellLabel);
end
save(fullfile(summaryFolder, 'linearEqvDiscOld.mat'), 'linearDiscSummary');

%% Save analysis information for drug experiments
clc;  meanLuminance=input ('enter the mean luminance:');  numCells=0;
drugUsed=input ('enter the drug used:','s');
try
    load(fullfile(summaryFolder, 'linearEqvDiscDrug.mat'));
    numCells=numel(linearDiscDrugSummary);
end
linearDiscDrugSummary(numCells+1)=struct('date',output.expDate,'cellID',output.cellLabel,'cellType', output.cellType,'onlineAnalysis',output.onlineAnalysis,...
    'meanOnsetNLI',mean(output.NLI.onset),'meanOffsetNLI',mean(output.NLI.offset),'OnsetNLI',output.NLI.onset,'OffsetNLI',output.NLI.offset,'meanLum',meanLuminance, ...
    'drugUsed',drugUsed);
save(fullfile(summaryFolder, 'linearEqvDiscDrug.mat'),'linearDiscDrugSummary');
fprintf('%s \n', '---new cell data saved---');

%% Population level summary analysis of linear disc (OLD)
clc; load(fullfile(summaryFolder, 'linearEqvDiscOld.mat'));
discTable=struct2table(linearDiscSummary);
discTable.onlineAnalysis=categorical(discTable.onlineAnalysis); discTable.cellType=categorical(discTable.cellType);
CloseAllFiguresExceptGUI;
cellTypeToInsp=categorical({'OffT','OffS','OnT'}); recTypeToInsp='inh';
subTable=discTable(discTable.onlineAnalysis==recTypeToInsp & discTable.cellType=='OffT' & (discTable.meanLum==100 | discTable.meanLum==10), ...,
    {'cellType','meanOnsetNLI','meanOffsetNLI','onlineAnalysis','meanLum'});
[G,gName] = findgroups(subTable.cellType);
cellMean=splitapply(@mean,subTable.meanOnsetNLI,G);
cellErr=splitapply(@(x) std(x)/sqrt(numel(x)),subTable.meanOnsetNLI,G);
figure; scatterWithMeanAndError(G,subTable.meanOnsetNLI,cellMean,cellErr,string(gName),1);

[G,gName] = findgroups(subTable.cellType);
cellMean=splitapply(@mean,subTable.meanOffsetNLI,G);
cellErr=splitapply(@(x) std(x)/sqrt(numel(x)),subTable.meanOffsetNLI,G);
figure; scatterWithMeanAndError(G,subTable.meanOffsetNLI,cellMean,cellErr,string(gName),1);

% Cumulative NLI plots
f1=figure('color','w','position',[100 300 700 700]);  hold all;
for c=1:numel(cellTypeToInsp)
    cumRes=discTable(discTable.cellType==cellTypeToInsp(c)&discTable.onlineAnalysis==recTypeToInsp&...
        discTable.meanLum==100,{'cellType','OnsetNLI','OffsetNLI'});
    cp=cdfplot(cat(2,cumRes.OnsetNLI{:})); set(cp,'linewidth',2);
end
plot([0 0],[0, 1],'--k');
grid off;  title('All fixation histogram');  legend(cellTypeToInsp,'location','SE'); legend boxoff;
initFig(gca(f1),'Onset NLI','cumulative fraction'); setAxes(f1);  xlim([-1 1]);

f2=figure('color','w','position',[100 300 700 700]);  hold all;
for c=1
    cumRes=discTable(discTable.cellType==cellTypeToInsp(c)&discTable.onlineAnalysis==recTypeToInsp& discTable.meanLum==100,{'cellType','OnsetNLI','OffsetNLI'});
    cp=cdfplot(cat(2,cumRes.OffsetNLI{:})); set(cp,'linewidth',2);
end
plot([0 0],[0, 1],'--k');
grid off;  title('All fixation histogram');
initFig(gca(f2),'Offset NLI','cumulative fraction'); setAxes(f2);  xlim([-1 1]);

f2=figure('color','w','position',[100 300 700 700]);  hold all;
cumRes=discTable(discTable.cellType==cellTypeToInsp(c)&discTable.onlineAnalysis==recTypeToInsp& discTable.meanLum==100,{'cellType','OnsetNLI','OffsetNLI'});
cp=cdfplot(cat(2,cumRes.OnsetNLI{:})); set(cp,'linewidth',2);
cumRes=discTable(discTable.cellType==cellTypeToInsp(c)&discTable.onlineAnalysis==recTypeToInsp& discTable.meanLum==100,{'cellType','OnsetNLI','OffsetNLI'});
cp=cdfplot(cat(2,cumRes.OffsetNLI{:})); set(cp,'linewidth',2);
plot([0 0],[0, 1],'--k');
grid off;  title('All fixation histogram');  legend('onset','offset'); legend boxoff;
initFig(gca(f2),'Inh NLI','cumulative fraction'); setAxes(f2);  xlim([-1 1]);

% Luminance dependency of OffT NLI
lumTable=discTable(discTable.cellType=='OffT'& discTable.onlineAnalysis=='extracellular',...
    {'date','cellID','OnsetNLI','OffsetNLI','meanOnsetNLI','meanOffsetNLI','meanLum'});
meanLumList=unique(lumTable.meanLum);
f3=figure('color','w','position',[100 300 700 700]);  hold all;
for m=1:numel(meanLumList)
    cumRes=lumTable(lumTable.meanLum==meanLumList(m),{'OnsetNLI','OffsetNLI'});
    cp=cdfplot(cat(2,cumRes.OnsetNLI{:})); set(cp,'linewidth',2);
end
plot([0 0],[0, 1],'--k');
grid off;  title('All fixation histogram');  legend(split(num2str(meanLumList')),'location','SE'); legend boxoff;
initFig(gca(f3),'Onset NLI','cumulative fraction'); setAxes(f3);  xlim([-1 1]);

f4=figure('color','w','position',[100 300 700 700]);  hold all;
for m=1:numel(meanLumList)
    cumRes=lumTable(lumTable.meanLum==meanLumList(m),{'OnsetNLI','OffsetNLI'});
    cp=cdfplot(cat(2,cumRes.OnsetNLI{:})); set(cp,'linewidth',2);
end
plot([0 0],[0, 1],'--k');
grid off;  title('All fixation histogram');  legend(split(num2str(meanLumList')),'location','SE'); legend boxoff;
initFig(gca(f4),'Offset NLI','cumulative fraction'); setAxes(f4); xlim([-1 1]);

% Paired luminance comparison
lumTable=lumTable(lumTable.meanLum==100 | lumTable.meanLum==1000,:);
[G,ID]=findgroups(lumTable(:,{'date', 'cellID'})); gList=unique(G);
f5=figure('color','w','position',[100 300 700 700]);  hold all;
for i=1: numel(gList)
    rows=find(G==gList(i));
    if numel(rows)==2
        plot(log(lumTable.meanLum(rows)),lumTable.meanOnsetNLI(rows),'-ko','markersize',15);
    end
end
set(gca,'xtick', log(meanLumList),'xticklabels',split(num2str(meanLumList'))); xlabel('Mean luminance'); ylabel('Onset NLI');

% Overlay Exc vs Inh
overlayTable=discTable(discTable.cellType=='OffT' & discTable.meanLum==100 & (discTable.onlineAnalysis=='exc'| discTable.onlineAnalysis=='inh'), ...,
    {'date','cellID','OnsetNLI','onlineAnalysis'});
overlayTable([1 2 5 6 7 10 11 12 17 18 23 24 25],:)=[];
[G,~]=findgroups(overlayTable(:,{'date', 'cellID'})); gList=unique(G);
figure('color','w','position',[100 300 700 700]);  hold all;
for i=2: numel(gList)
    rows=find(G==gList(i));
    if numel(rows)==2
        tempTable=overlayTable(rows,:);
        excInd=find(tempTable.onlineAnalysis=='exc'); inhInd=find(tempTable.onlineAnalysis=='inh');
        scatter(tempTable.OnsetNLI{excInd}, tempTable.OnsetNLI{inhInd});
    end
end

%% Population level summary analysis of linear disc (NEW)
clc; load(fullfile(summaryFolder, 'olderSummary', 'linearEqvDiscOld.mat'));
discTable = struct2table(linearDiscSummary);
discTable.onlineAnalysis = categorical(discTable.onlineAnalysis);
discTable.cellType = categorical(discTable.cellType);
CloseAllFiguresExceptGUI;

cellTypeToInsp = categorical({'OffS','OffT'});
recTypeToInsp = 'exc';
subTable = discTable(discTable.onlineAnalysis==recTypeToInsp & discTable.meanLum==100, :);
[G, gName] = findgroups(subTable.cellType);

figure('Position', [100 100 1200 800]);

subplot(2,2,1);
onsetNLIs = cellfun(@median, subTable.OnsetNLI);
cellMean = splitapply(@median, onsetNLIs, G);
cellErr = splitapply(@(x) std(x)/sqrt(numel(x)), onsetNLIs, G);
scatterWithMeanAndError(G, onsetNLIs, cellMean, cellErr, string(gName), 1);
title('Mean Onset NLI');
ylabel('NLI');

subplot(2,2,2);
offsetNLIs = cellfun(@median, subTable.OffsetNLI);
cellMean = splitapply(@median, offsetNLIs, G);
cellErr = splitapply(@(x) std(x)/sqrt(numel(x)), offsetNLIs, G);
scatterWithMeanAndError(G, offsetNLIs, cellMean, cellErr, string(gName), 1);
title('Mean Offset NLI');
ylabel('NLI');

subplot(2,2,3);
hold all;
for c = 1:numel(cellTypeToInsp)
    cumRes = subTable(subTable.cellType==cellTypeToInsp(c), :);
    onsetNLIs = cat(2, cumRes.OnsetNLI{:});
    [f,x] = ecdf(onsetNLIs);
    plot(x, f, 'LineWidth', 2);
end
plot([0 0], [0, 1], '--k');
title('Cumulative Onset NLI');
xlabel('NLI');
ylabel('Cumulative Fraction');
legend(cellTypeToInsp, 'Location', 'SE');
xlim([-1 1]);

subplot(2,2,4);
hold all;
for c = 1:numel(cellTypeToInsp)
    cumRes = subTable(subTable.cellType==cellTypeToInsp(c), :);
    offsetNLIs = cat(2, cumRes.OffsetNLI{:});
    [f,x] = ecdf(offsetNLIs);
    plot(x, f, 'LineWidth', 2);
end
plot([0 0], [0, 1], '--k');
title('Cumulative Offset NLI');
xlabel('NLI');
ylabel('Cumulative Fraction');
legend(cellTypeToInsp, 'Location', 'SE');
xlim([-1 1]);

% Response comparisons
figure('Position', [100 100 1200 400]);
colors = pmkmp(numel(cellTypeToInsp),'IsoL');

subplot(1,2,1);
hold all;
h = zeros(numel(cellTypeToInsp), 1);
for c = 1:numel(cellTypeToInsp)
    cellData = subTable(subTable.cellType==cellTypeToInsp(c), :);
    scatter(cellData.respOnset(:,1), cellData.respOnset(:,2), 100, colors(c,:), 'filled', 'MarkerFaceAlpha', 0.6);
    meanX = mean(cellData.respOnset(:,1));
    meanY = mean(cellData.respOnset(:,2));
    semX = std(cellData.respOnset(:,1))/sqrt(height(cellData));
    semY = std(cellData.respOnset(:,2))/sqrt(height(cellData));
    h(c) = errorbar(meanX, meanY, semY, semY, semX, semX, 'Color', colors(c,:), 'LineWidth', 3);
end
plot([0 max([xlim ylim])], [0 max([xlim ylim])], '--k');
xlabel('Image Response');
ylabel('Disc Response');
title('Onset Responses');
legend(h, cellTypeToInsp, 'Location', 'SE','fontsize',24); legend boxoff;
axis square;

subplot(1,2,2);
hold all;
h = zeros(numel(cellTypeToInsp), 1);
for c = 1:numel(cellTypeToInsp)
    cellData = subTable(subTable.cellType==cellTypeToInsp(c), :);
    maxOnsetPerCell = max(cellData.respOnset, [], 2);
    normRespOffset = cellData.respOffset ./ maxOnsetPerCell;
    scatter(normRespOffset(:,1), normRespOffset(:,2), 100, colors(c,:), 'filled', 'MarkerFaceAlpha', 0.6);
    meanX = mean(normRespOffset(:,1));
    meanY = mean(normRespOffset(:,2));
    semX = std(normRespOffset(:,1))/sqrt(height(cellData));
    semY = std(normRespOffset(:,2))/sqrt(height(cellData));
    h(c) = errorbar(meanX, meanY, semY, semY, semX, semX, 'Color', colors(c,:), 'LineWidth', 3);
end
plot([0 1], [0 1], '--k');
xlabel('Normalized Image Response');
ylabel('Normalized Disc Response');
title('Offset Responses (Normalized to Max Onset)');
legend(h, cellTypeToInsp, 'Location', 'SE','fontsize',24); legend boxoff;
axis square;
xlim([0 1]);
ylim([0 1]);

% Combined onset/offset comparison
figure('Position', [100 100 600 600]);
hold all;
h_onset = zeros(numel(cellTypeToInsp), 1);
h_offset = zeros(numel(cellTypeToInsp), 1);
for c = 1:numel(cellTypeToInsp)
    cellData = subTable(subTable.cellType==cellTypeToInsp(c), :);
    scatter(cellData.respOnset(:,1), cellData.respOnset(:,2), 50, colors(c,:), 'LineWidth', 1.5, 'MarkerFaceAlpha', 0.2);
    scatter(cellData.respOffset(:,1), cellData.respOffset(:,2), 50, colors(c,:), 'filled', 'MarkerFaceAlpha', 0.2);
    meanX = mean(cellData.respOnset(:,1));
    meanY = mean(cellData.respOnset(:,2));
    semX = std(cellData.respOnset(:,1))/sqrt(height(cellData));
    semY = std(cellData.respOnset(:,2))/sqrt(height(cellData));
    errorbar(meanX, meanY, semY, semY, semX, semX, 'Color', colors(c,:), 'LineWidth', 2);
    h_onset(c) = scatter(meanX, meanY, 200, colors(c,:), 'LineWidth', 3);
    meanX = mean(cellData.respOffset(:,1));
    meanY = mean(cellData.respOffset(:,2));
    semX = std(cellData.respOffset(:,1))/sqrt(height(cellData));
    semY = std(cellData.respOffset(:,2))/sqrt(height(cellData));
    errorbar(meanX, meanY, semY, semY, semX, semX, 'Color', colors(c,:), 'LineWidth', 2);
    h_offset(c) = scatter(meanX, meanY, 200, colors(c,:), 'filled');
end
plot([0 max([xlim ylim])], [0 max([xlim ylim])], '--k');
xlabel('Image Response');
ylabel('Disc Response');
title('Onset (hollow) and Offset (filled) Responses');
cellTypeNames = categories(cellTypeToInsp);
legend_entries = {};
h_combined = [];
for i = 1:numel(cellTypeNames)
    h_combined = [h_combined; h_onset(i); h_offset(i)];
    legend_entries{end+1} = [cellTypeNames{i} ' onset'];
    legend_entries{end+1} = [cellTypeNames{i} ' offset'];
end
legend(h_combined, legend_entries, 'Location', 'SE', 'fontsize', 24);
axis square;

%% Population linear disc drug
clc; CloseAllFiguresExceptGUI;
load(fullfile(summaryFolder, 'linearEqvDiscDrug.mat'));
discDrugTable=struct2table(linearDiscDrugSummary);
discDrugTable.onlineAnalysis=categorical(discDrugTable.onlineAnalysis); discDrugTable.cellType=...
    categorical(discDrugTable.cellType); discDrugTable.drugUsed=categorical(discDrugTable.drugUsed);

subTable=discDrugTable(discDrugTable.meanLum==100,{'meanOnsetNLI','drugUsed','date','cellID'});
[G,ID]=findgroups(subTable(:,{'date','cellID'})); gList=unique(G);
f5=figure('color','w','position',[100 300 700 700]);  hold all;
gNames=categorical({'LY','control','str'});
for i=1: numel(gList)
    rows=find(G==gList(i));
    if numel(rows)==2
        tp=subTable.drugUsed(rows);
        xplot=[find(ismember(gNames, tp(1))) find(ismember(gNames, tp(2)))];
        plot(xplot,subTable.meanOnsetNLI(rows),'-ko','markersize',15);
    end
end
set(gca,'xtick',[1 2 3],'xticklabel',gNames); xlabel(''); ylabel('NLI'); ax=gca;

%% Population linear disc inhibitory cluster analysis
clc; CloseAllFiguresExceptGUI;
load(fullfile(summaryFolder, 'linearEqvDiscClusterAnalysis.mat'));
clusterTable=struct2table(excInhClusterSummary);
colors=pmkmp(size(clusterTable,1),'IsoL');
excNLI=zeros(size(clusterTable,1),3); inhNLI=zeros(size(clusterTable,1),3);
sf=figure('position',[50 250 700 700]); hold all;
sf2=figure('position',[50 250 700 700]); hold all;
sf3=figure('position',[50 250 700 700]); hold all;
mcolors=[[0.1 0.1 0.1]; [0.4 0.4 0.4]; [0.7 0.7 0.7]];
for c=1:size(clusterTable,1)
    excDiff=clusterTable.ampStats(c).exc.image.onset.mean-clusterTable.ampStats(c).exc.disc.onset.mean;
    inhDiff=clusterTable.ampStats(c).inh.image.onset.mean-clusterTable.ampStats(c).inh.disc.onset.mean;
    for cluster=1:3
        excNLI(c,cluster)=mean(clusterTable.NLI(c).exc.onset(clusterTable.clusterIndex(c,:)==cluster));
        inhNLI(c,cluster)=mean(clusterTable.NLI(c).inh.onset(clusterTable.clusterIndex(c,:)==cluster));
        h2(cluster)=scatter(gca(sf2),clusterTable.deltaEqvInt(c).exc(clusterTable.clusterIndex(c,:)==cluster), ...
            excDiff(clusterTable.clusterIndex(c,:)==cluster),150,mcolors(cluster,:),'filled');
        h3(cluster)=scatter(gca(sf3),clusterTable.deltaEqvInt(c).inh(clusterTable.clusterIndex(c,:)==cluster), ...
            inhDiff(clusterTable.clusterIndex(c,:)==cluster),150,mcolors(cluster,:),'filled');
    end
    scalor=max(clusterTable.ampStats(c).inh.image.onset.mean);
    scatter(gca(sf),clusterTable.ampStats(c).inh.image.onset.mean/scalor, clusterTable.ampStats(c).exc.image.onset.mean/scalor,100,colors(c,:),'filled');
    [p,s]=polyfit(clusterTable.ampStats(c).inh.image.onset.mean/scalor, clusterTable.ampStats(c).exc.image.onset.mean/scalor,1);
    fitV=polyval(p,clusterTable.ampStats(c).inh.image.onset.mean/scalor);
    plot(gca(sf),clusterTable.ampStats(c).inh.image.onset.mean/scalor, fitV,'--','color',colors(c,:),'linewidth',2);
end

figure('position',[50 250 900 400]);
subplot(1,2,1); hold all;
scatter(excNLI(:,1),excNLI(:,3),200,'filled'); errorbar(mean(excNLI(:,1)), mean(excNLI(:,3)),ste(excNLI(:,3)),ste(excNLI(:,3)),ste(excNLI(:,1)),ste(excNLI(:,1)),'o');
xlim([-0.7 0.3]); refline(1,0); axis equal; title('excitation')
subplot(1,2,2); hold all;
scatter(inhNLI(:,1),inhNLI(:,3),200,'filled'); errorbar(mean(inhNLI(:,1)), mean(inhNLI(:,3)),ste(inhNLI(:,3)),ste(inhNLI(:,3)),ste(inhNLI(:,1)),ste(inhNLI(:,1)),'o');
xlim([0 1]); refline(1,0);  axis equal;  title('inhibition');
s=sgtitle('NLI'); set(s,'fontsize',25);
