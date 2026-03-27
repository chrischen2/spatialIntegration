clearvars; close all; clc;
% define plot color sequence, axis fonts
import auimodel.*
import vuidocument.*
loader = edu.washington.rieke.Analysis.getEntityLoader();
treeFactory = edu.washington.rieke.Analysis.getEpochTreeFactory();
listFactory = edu.washington.rieke.Analysis.getEpochListFactory();
newList=listFactory.create;
ovaExportFolder='/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/fromFred/';
dataFolder='/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/fromFred/';
list = loader.loadEpochList([ovaExportFolder 'LinearEqvDisc.mat'], dataFolder);

for i = 1:list.length
    try
        list.elements(i).setProtocolSetting('user:startDate',datestr((list.elements(i).startDate)'));
    catch
        fprintf('%s  %i\n', 'fail to format', i);
    end
end
listSorted = list.sortedBy('protocolSettings(user:startDate)'); % list sorted chronologically

%% expanding spot split data and create GUI

cellTypeSplit = @(listSorted)splitOnCellType(listSorted);
cellTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, cellTypeSplit);

dateSplit = @(listSorted)splitOnExperimentDate(listSorted);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, dateSplit);

ndfSplit = @(listSorted)splitOnNDFs(listSorted);
ndfSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, ndfSplit);

brightnessSplit = @(listSorted)splitOnDeviceBrightNess(listSorted);
brightnessSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, brightnessSplit);

tree = riekesuite.analysis.buildTree(listSorted,{cellTypeSplit_java, dateSplit_java, 'cell.label','protocolSettings(epochGroup:label)',...
    brightnessSplit_java, ndfSplit_java,'protocolSettings(onlineAnalysis)'});
gui = epochTreeGUI(tree);

%% expanding spots analyzing
clc;
CloseAllFiguresExceptGUI;
paras.psthSigma=20;
paras.spikeTh=1.2;
paras.sampleRate=1e4;
paras.spikeTag=0;
paras.wcOffset=500;
paras.spikeOffset=500;
paras.plotOffset=10;
selectedNodes = gui.getSelectedEpochTreeNodes;
stimTime=selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');
preTime=selectedNodes{1}.epochList.firstValue.protocolSettings('preTime');
timeToPts=@(x) x/1e3*paras.sampleRate;
paras.stimPts=timeToPts(stimTime);
paras.prePts=timeToPts(preTime);
paras.backgroundIntensity=selectedNodes{1}.epochList.firstValue.protocolSettings('backgroundIntensity');
% selectedInd=getSelectedIndex(selectedNodes{1}.epochList);
[ax,output,onlineAnalysis]=analyzeExpandingSpots(selectedNodes,paras);
output.minRes
%% save cell info for expanding spots population summary
clc; clear expSpotSummary;
onlineAnalysis
meanLuminance=input ('enter the mean luminance:');
cellType=shortCellType(selectedNodes{1}.epochList.firstValue.protocolSettings('source:type'));
recType=onlineAnalysis;
expDate=datestr(selectedNodes{1}.epochList.elements(1).startDate','yyyy/mm/dd');
cellLabel=selectedNodes{1}.epochList.firstValue.protocolSettings('source:label');
numCells=0;
try
    load('/Volumes/GoogleDrive/My Drive/projects/spatialIntegration/summary/expandingSpots.mat');
    numCells=numel(expSpotSummary);
end
expSpotSummary(numCells+1)=struct('date',expDate,'cellID',cellLabel,'cellType', cellType,'recType',recType,'spotList',output.spotList, ...
    'normRes',output.normRes,'sigmaC',output.model.sigmaC,'sigmaS',output.model.sigmaS,'Kc',output.model.Kc,'kS',output.model.Ks,'meanLum',meanLuminance);
save('/Volumes/GoogleDrive/My Drive/projects/spatialIntegration/summary/expandingSpots.mat','expSpotSummary');
fprintf('%s \n', '---new cell data saved---');

%% population level summary analysis of expanding spots
clc; clear sumTable spotMean spotErr spotID
load('/Users/chrischen/Dropbox/research/projects/spatialIntegration/summary/expandingSpots.mat');
sumTable=struct2table(expSpotSummary);
sumTable.recType=categorical(sumTable.recType); sumTable.cellType=categorical(sumTable.cellType);
CloseAllFiguresExceptGUI;
cellTypes=unique(sumTable.cellType);
f=figure('color','w','position',[50 100 1800 900]);
for c=1:numel(cellTypes)
    ax(c)=subplot(2,ceil(numel(cellTypes)/2),c);   hold all;
    try
        spotRes=sumTable(sumTable.cellType==cellTypes(c) & sumTable.recType=='extracellular'& sumTable.meanLum==100,{'spotList','normRes','sigmaC'});
        for i=1:size(spotRes,1)
            plot(cell2mat(spotRes.spotList(i)), cell2mat(spotRes.normRes(i)),'linewidth',2,'color','k');
        end
        % legend(cellstr(num2str((1:size(spotRes,1))', 'trial %-d')),'fontsize',15); legend boxoff;
        % compute the mean and error for each bar size
        [G,spotID{c}] = findgroups(cat(2,spotRes.spotList{:}));
        spotMean.(char(cellTypes(c)))=splitapply(@mean,cat(2,spotRes.normRes{:}),G);
        spotErr.(char(cellTypes(c)))=splitapply(@(x) std(x)/sqrt(numel(x)),cat(2,spotRes.normRes{:}),G);
        errorbar(spotID{c}, spotMean.(char(cellTypes(c))),spotErr.(char(cellTypes(c))),'r','linewidth',3);
        plot(spotID, spotMean.(char(cellTypes(c))),'r','linewidth',3);
        title(char(cellTypes(c))); xlabel('spot size');  ylabel('Norm Response'); setAxes(f);
        hold off;
    end
end
% overlay of cell types
f=figure('color','w','position',[50 100 600 600]); hold all;
for c=1:numel(cellTypes)
    try
        spotRes=sumTable(sumTable.cellType==cellTypes(c) & sumTable.recType=='extracellular'& sumTable.meanLum==100,{'spotList','normRes','sigmaC'});

        % legend(cellstr(num2str((1:size(spotRes,1))', 'trial %-d')),'fontsize',15); legend boxoff;
        % compute the mean and error for each bar size
        [G,spotID{c}] = findgroups(cat(2,spotRes.spotList{:}));
        spotMean.(char(cellTypes(c)))=splitapply(@mean,cat(2,spotRes.normRes{:}),G);
        spotErr.(char(cellTypes(c)))=splitapply(@(x) std(x)/sqrt(numel(x)),cat(2,spotRes.normRes{:}),G);
        errorbar(spotID{c}, spotMean.(char(cellTypes(c))),spotErr.(char(cellTypes(c))),'linewidth',3);
        plot(spotID, spotMean.(char(cellTypes(c))),'r','linewidth',3);
    end
end
title(string(cellTypes(c))); xlabel('spot size');  ylabel('Norm Response'); setAxes(f);


rfSize=varfun( @(x) mean(x), sumTable, 'GroupingVariables', {'cellType','meanLum'},...
    'InputVariables',{'sigmaC'},'outputformat','table');

% cell mean
rfTable=sumTable( sumTable.recType=='extracellular'& sumTable.meanLum==100,{'cellType','sigmaC'});
[G,gName]=findgroups(rfTable(:,{'cellType'})); gList=unique(G);
cellMean=splitapply(@mean,rfTable.sigmaC,G);
cellErr=splitapply(@(x) std(x)/sqrt(numel(x)),rfTable.sigmaC,G);
figure; ax=scatterWithMeanAndError(G,rfTable.sigmaC,cellMean,cellErr,cellstr(gName.cellType),1);

% cell mean
a2Table=sumTable((sumTable.cellType=='A2' | sumTable.cellType=='ACs') &sumTable.recType=='currentClamp',:);
[G,~]=findgroups(a2Table(:,{'date', 'cellID'})); gList=unique(G);
f5=figure('color','w','position',[100 300 700 700]);  hold all;
for i=1: numel(gList)
    rows=find(G==gList(i));
    try
        plot(log(a2Table.meanLum(rows)),2*a2Table.sigmaC(rows),'-ko','markersize',15);

    end
end
xlabel('Mean luminance'); ylabel('RF diameter');
lumList=unique(a2Table.meanLum);
for i=1:numel(lumList)
    scMean(i)=2*mean(a2Table.sigmaC(a2Table.meanLum==lumList(i)));
    scErr(i)=2*ste(a2Table.sigmaC(a2Table.meanLum==lumList(i)));
end
errorbar(log(lumList),scMean, scErr);

%% create GUI for contrast spots
cellTypeSplit = @(listSorted)splitOnCellType(listSorted);
cellTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, cellTypeSplit);

dateSplit = @(listSorted)splitOnExperimentDate(listSorted);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, dateSplit);

ndfSplit = @(listSorted)splitOnNDFs(listSorted);
ndfSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, ndfSplit);

brightnessSplit = @(listSorted)splitOnDeviceBrightNess(listSorted);
brightnessSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, brightnessSplit);

tree = riekesuite.analysis.buildTree(listSorted,{cellTypeSplit_java,dateSplit_java, 'cell.label','protocolSettings(onlineAnalysis)',...
    brightnessSplit_java, ndfSplit_java});
gui = epochTreeGUI(tree);

%% analyze contrast spots
clc;
CloseAllFiguresExceptGUI;
paras.psthSigma=20;
paras.spikeTh=1.2;
paras.spikeTag=0;
paras.sampleRate=1e4;
selectedNodes = gui.getSelectedEpochTreeNodes;
stimTime=selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');
preTime=selectedNodes{1}.epochList.firstValue.protocolSettings('preTime');
timeToPts=@(x) x/1e3*paras.sampleRate;
paras.stimPts=timeToPts(stimTime);
paras.prePts=timeToPts(preTime);
paras.wcoffset=0;
% selectedInd=getSelectedIndex(selectedNodes{1}.epochList);
output=analyzeContrastSpots(selectedNodes,paras);
allFilters{end+1} = output{1}.temporalFilter;
%% save cell info for contrast spots population summary
clc; clear expSpotSummary;
onlineAnalysis=selectedNodes{1}.parent.parent.splitValue
meanLuminance=input ('enter the mean luminance:');
cellType=shortCellType(selectedNodes{1}.epochList.firstValue.protocolSettings('source:type'));
expDate=datestr(selectedNodes{1}.epochList.elements(1).startDate','yyyy/mm/dd');
cellLabel=selectedNodes{1}.epochList.firstValue.protocolSettings('source:label');
numCells=0;
try
    load('/Volumes/GoogleDrive/My Drive/projects/spatialIntegration/summary/contrastSpots.mat');
    numCells=numel(contrastSpotSummary);
end
contrastSpotSummary(numCells+1)=struct('date',expDate,'cellID',cellLabel,'cellType', cellType,'onlineAnalysis',onlineAnalysis,...
    'contrastList',contrastArray, 'resList',resArray, 'meanLum',meanLuminance);
save('/Volumes/GoogleDrive/My Drive/projects/spatialIntegration/summary/contrastSpots.mat','contrastSpotSummary');
fprintf('%s \n', '---new cell data saved---');

%% load all cell data for contrast spots
clc; clear sumTable
load('/Users/chrischen/Dropbox/research/projects/spatialIntegration/summary/contrastSpots.mat');
sumTable=struct2table(contrastSpotSummary);
sumTable.onlineAnalysis=categorical(sumTable.onlineAnalysis); sumTable.cellType=categorical(sumTable.cellType);
recInsp='exc'; meanLum=100;
if strcmp(recInsp,'exc')
    sumTable.resList=cellfun(@(x) x/min(x),sumTable.resList,'UniformOutput',false);
else
    sumTable.resList=cellfun(@(x) x/max(x),sumTable.resList,'UniformOutput',false);
end
CloseAllFiguresExceptGUI;
cellTypes=unique(sumTable.cellType);


for c=1:2
    f=figure('color','w','position',[50 100 600 600]);
    %     ax(c)=subplot(numel(cellTypes),1,c);
    hold all;
    try
        spotRes=sumTable(sumTable.cellType==cellTypes(c) & sumTable.onlineAnalysis==recInsp & sumTable.meanLum==meanLum,{'contrastList','resList'});
        %                         spotRes=spotRes([2 3 6 7],:);
        % spotRes=spotRes([ 2  4  ],:);
        for i=1:size(spotRes,1)
            plot(cell2mat(spotRes.contrastList(i)), cell2mat(spotRes.resList(i)),'linewidth',2);
        end

        % legend(cellstr(num2str((1:size(spotRes,1))', 'trial %-d')),'fontsize',15); legend boxoff;
        % compute the mean and error for each bar size
        [G,spotID{c}] = findgroups(cat(2,spotRes.contrastList{:}));
        spotMean.(string(cellTypes(c)))=splitapply(@mean,cat(2,spotRes.resList{:}),G);
        spotErr.(string(cellTypes(c)))=splitapply(@(x) std(x)/sqrt(numel(x)),cat(2,spotRes.resList{:}),G);
        errorbar(spotID{c}, spotMean.(string(cellTypes(c))),spotErr.(string(cellTypes(c))),'r','linewidth',3);
        if strcmp(recInsp,'exc')
            resFit=fitCRF_sigmoid(spotID{c},spotMean.(string(cellTypes(c))),[-0.5 0,1,0]);
        else
            resFit=fitCRF_sigmoid(spotID{c},spotMean.(string(cellTypes(c))),[0.5 0,1,0]);
        end
        % plot(spotID{c},sigmoidCRF(spotID{c},resFit.k,resFit.c0,resFit.amp,resFit.yOff),'k','linewidth',2);
        % plot(barID, barMean.(string(cellTypes(c))),'r','linewidth',3);
        title(cellTypes(c) ); xlabel('contrast');  ylabel('Int Response'); setAxes(f);
        hold off;
        contrasts=spotID{c}; contrastResponses=spotMean.(string(cellTypes(c)));
        save(['/Users/chrischen/Dropbox/research/projects/spatialIntegration/summary/contrastFunctions/' char(cellTypes(c)) recInsp '.mat'], ...
            'contrasts', 'contrastResponses');
    end
end

%% save CRS I/E ratio
clc; clear IERatio;
meanLuminance=input ('enter the mean luminance:');
numCells=0;
try
    load('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/IERatio.mat');
    numCells=numel(IERatioSummary);
end
IERatioSummary(numCells+1)=struct('IERatio',output{1}.ieRatio, 'meanLum',meanLuminance);
save('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/IERatio.mat','IERatioSummary');
fprintf('%s \n', '---new cell data saved---');


%% contrast reversing grating split data and create GUI
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

%% analyze contrast reversing grating
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

%% save E/I amplitude in 100 versus 1000 light levels, for OffT only
clc; clear LightLevelEISummary;
numCells=0;
example=1;
meanLuminance=input ('enter the mean luminance:');
expDate=datestr(selectedNodes{1}.epochList.elements(1).startDate','yyyy/mm/dd');
cellLabel=selectedNodes{1}.epochList.firstValue.protocolSettings('source:label');
try
    load('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/LightLevelEISummary.mat');
    numCells=numel(LightLevelEISummary);
end
LightLevelEISummary(numCells+1)=struct('exampleCell',example, 'date',expDate,'cellID',cellLabel, 'barList',output{1}.barList, ...
    'meanEIOffset',output{1}.meanEIOffset, 'eiOffset',output{1}.eiOffset,'eiRatio',output{1}.eiRatio,'lags', output{1}.lags,'eiCorr',output{1}.eiCorr);
save('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/LightLevelEISummary.mat','LightLevelEISummary');

%% save particular cell for population summary
clc; clear CRGSummary;
output.onlineAnalysis
meanLuminance=input ('enter the mean luminance:');
cellType=shortCellType(selectedNodes{1}.epochList.firstValue.protocolSettings('source:type'));
recType=output.onlineAnalysis;
numCells=0;
try
    load('/Volumes/GoogleDrive/My Drive/projects/spatialIntegration/summary/contrastReversingGrating.mat');
    numCells=numel(CRGSummary);
end
CRGSummary(numCells+1)=struct('date',output.expDate,'cellID',output.cellLabel,'cellType', cellType,'recType',recType,'tempFreq', paras.tempFreq,'barList',output.barList, ...
    'F2',output.F2,'sinoF2',output.sinoF2,'suppression',output.suppress,'subUnitSize', output.subUnitSize,'meanLum',meanLuminance);
save('/Volumes/GoogleDrive/My Drive/projects/spatialIntegration/summary/contrastReversingGrating.mat','CRGSummary');
fprintf('%s \n', '---new cell data saved---');

%% save EI summary
clc; clear EISummary;
numCells=0;
example=1;
expDate=datestr(selectedNodes{1}.epochList.elements(1).startDate','yyyy/mm/dd');
cellLabel=selectedNodes{1}.epochList.firstValue.protocolSettings('source:label');
cellType=shortCellType(selectedNodes{1}.epochList.firstValue.protocolSettings('source:type'));
% cellType='OffS'
try
    load('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/EISummary.mat');
    numCells=numel(EISummary);
end
EISummary(numCells+1)=struct('exampleCell',example, 'date',expDate,'cellID',cellLabel,'cellType', cellType,'barList',output{1}.barList, ...
    'meanEIOffset',output{1}.meanEIOffset, 'eiOffset',output{1}.eiOffset,'eiRatio',output{1}.eiRatio,'lags', output{1}.lags,'eiCorr',output{1}.eiCorr);
save('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/EISummary.mat','EISummary');

%% save spike/exc/inh summary
clc; clear SeiSummary;
numCells=0;
try
    load('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegrationsummary/CRGEIanalysis.mat');
    numCells=numel(SeiSummary);
end
SeiSummary(numCells+1)=struct('tempFreq',paras.tempFreq,'EI',output{1}.EiRatio,'phaseDiff',output{1}.phaseDiff,'normSpike',output{1}.normSpike);
save('/Volumes/GoogleDrive/My Drive/projects/spatialIntegration/summary/CRGEIanalysis.mat','SeiSummary');

%% population level summary analysis of contrast reversing gratings
clc; clear barID
load('/Users/chrischen/Dropbox/research/projects/spatialIntegration/summary/contrastReversingGrating.mat');
sumTable=struct2table(CRGSummary);
sumTable.recType=categorical(sumTable.recType); sumTable.cellType=categorical(sumTable.cellType);
% average subunit size of given luminance given certain cell type
subunit=varfun( @(x) mean(x), sumTable(sumTable.meanLum==100 & sumTable.recType=='extracellular',:), 'GroupingVariables', {'cellType','meanLum','recType'},...
    'InputVariables',{'subUnitSize','suppression'},'outputformat','table');
CloseAllFiguresExceptGUI;
figure('color','w','position',[200 200 900 900]);
cellTypes=unique(sumTable.cellType);
cellTypes={'OffT','OffS','OnT','OnS'};
for c=1:numel(cellTypes)
    subTable=sumTable(sumTable.cellType==cellTypes(c) & sumTable.recType=='extracellular' & sumTable.meanLum==100,{'cellID','barList','F2','sinoF2','suppression'});
    % Further filter the subTable based on bar size and suppression
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
    % compute the mean and error for each bar size
    [G,barID{c}] = findgroups(cat(1,subTable.barList{:}));
    barMean.(char(cellTypes(c)))=splitapply(@mean,cat(1,subTable.F2{:}),G);
    barErr.(char(cellTypes(c)))=splitapply(@(x) std(x)/sqrt(numel(x)),cat(1,subTable.F2{:}),G);
    errorbar(barID{c}, barMean.(char(cellTypes(c))),barErr.(char(cellTypes(c))),'r','linewidth',3);
    title(char(cellTypes(c)));
end

% overlay different recTypes for offT
colors=pmkmp(10,'Isol');
figure('color','w','position',[200 200 900 900]);
recTypes=unique(sumTable.recType);
clear barID
for r=1:numel(recTypes)
    subTable2=sumTable(sumTable.cellType=='OffT' & sumTable.recType==recTypes(r) & sumTable.meanLum==100,{'cellID','barList','F2'});
    hold all;
    %     for i=1:size(f2Table,1)
    %         plot(cell2mat(f2Table.barList(i)), cell2mat(f2Table.F2(i)),'linewidth',0.5);
    %     end
    % compute the mean and error for each bar size
    [G,barID{r}] = findgroups(cat(1,subTable2.barList{:}));
    barMean.(char(recTypes(r)))=splitapply(@mean,cat(1,subTable2.F2{:}),G);
    scalor=max(barMean.(char(recTypes(r)))); barMean.(char(recTypes(r)))=barMean.(char(recTypes(r)))/scalor;
    barErr.(char(recTypes(r)))=splitapply(@(x) std(x)/sqrt(numel(x)),cat(1,subTable2.F2{:}),G);  barErr.(char(recTypes(r)))=barErr.(char(recTypes(r)))/scalor;
    errorbar(barID{r}, barMean.(char(recTypes(r))),barErr.(char(recTypes(r))),'color',colors(r,:),'linewidth',3);
    title(char(recTypes(r)));
end
legend(char(recTypes)); legend boxoff;


recTypes={'exc','inh'};
for r=1:numel(recTypes)
    subTable=sumTable(sumTable.cellType=='OffT' & sumTable.recType==recTypes{r} & sumTable.meanLum==100,{'cellID','barList','F2','sinoF2','suppression'});
    % Further filter the subTable based on bar size and suppression
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
    % compute the mean and error for each bar size
    [G,barID{c}] = findgroups(cat(1,subTable.barList{:}));
    barMean.(char(cellTypes(c)))=splitapply(@mean,cat(1,subTable.F2{:}),G);
    barErr.(char(cellTypes(c)))=splitapply(@(x) std(x)/sqrt(numel(x)),cat(1,subTable.F2{:}),G);
    errorbar(barID{c}, barMean.(char(cellTypes(c))),barErr.(char(cellTypes(c))),'r','linewidth',3);
    title(char(cellTypes(c)));

    localityIndex = []; % Initialize locality index array
    ax(r) = subplot(3, 2, r);
    hold all;

    for i = 1:size(subTable, 1)
        barList = subTable.barList{i};
        F2 = subTable.F2{i};

        % Check if both bar sizes 160 and 40 are present
        if any(barList >= 120) && any(barList == 40)
            F2_160 = F2(end);
            F2_40 = F2(barList == 40);
            localityIndex = [localityIndex F2_160]; % Compute locality index
            plot(barList, F2, 'color', 'k', 'linewidth', 0.5);
        end
    end

    localityIndices{r} = localityIndex;

    title(char(recTypes{r}));

end
% Plot scatter plot of locality index for both excitatory and inhibitory types
figure;
hold all;
xPositions = [];
jitterAmount=0.1;
for r = 1:numel(recTypes)
    % Add jitter to x-positions
    jitteredX = r + (rand(size(localityIndices{r})) - 0.5) * jitterAmount;
    scatter(jitteredX, localityIndices{r}, 50, 'filled', 'DisplayName', recTypes{r});
end

% Calculate mean and standard error
for r = 1:numel(recTypes)
    meanValue = mean(localityIndices{r});
    stderrValue = std(localityIndices{r}) / sqrt(numel(localityIndices{r}));
    % Plot the mean and error bar
    errorbar(r, meanValue, stderrValue, 'k', 'LineWidth', 2, 'CapSize', 10);
end

xlim([0.5, numel(recTypes) + 0.5]);
xticks(1:numel(recTypes));
xticklabels(recTypes);
ylabel('Locality Index (F2_{160} / F2_{40})');
legend show;
title('Locality Index for Excitatory and Inhibitory Cells');
hold off;

%% population level summary EI analysis of CRG
clc; clear barID
CloseAllFiguresExceptGUI;

% Load the data into a table
load('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/EISummary.mat');
sumTable = struct2table(EISummary);
sumTable.cellType = categorical(sumTable.cellType);

% Get unique cell types
cellTypes = unique(sumTable.cellType);

% Loop through each cell type
for c = 1:numel(cellTypes)
    % Extract subset of the table for the current cell type
    subTable = sumTable(sumTable.cellType == cellTypes(c), {'date', 'cellID', 'barList', 'lags', 'eiCorr'});

    % Loop through each cell in the current cell type
    for cellIdx = 1:height(subTable)
        cellData = subTable(cellIdx, :);

        % Create a new figure for each cell
        figure('Name', sprintf('Cell Type: %s, Cell ID: %s', char(cellTypes(c)), cellData.cellID{1}), 'Color', 'w');
        hold on;

        % Plot correlation over lags for each bar size
        for barIdx = 1:numel(cellData.barList{1})
            lags = cellData.lags(1, :); % Assuming lags are stored in rows corresponding to bar sizes
            corrVals = cellData.eiCorr{1}(barIdx, :); % Assuming correlations are stored similarly

            % Plot the correlation curve
            plot(lags, corrVals, 'LineWidth', 2, 'DisplayName', sprintf('Bar Size: %d', cellData.barList{1}(barIdx)));
        end

        % Add labels and title
        xlabel('Lag (ms)');
        ylabel('Cross-correlation');
        title(sprintf('Cross-correlation for Cell ID: %s, Cell Type: %s, Date: %s', cellData.cellID{1}, char(cellTypes(c)), cellData.date{1}));
        legend('show');
        hold off;
    end
end

% now let's overlay the population
% Loop through each cell type
% Create a new figure for each cell
% Initialize figures
figure1 = figure;
hold on;
figure2 = figure;
hold on;

colors = [[0.5 0.5 0.5]; [0.9 0.1 0.1]];

for c = 1:numel(cellTypes)
    % Extract subset of the table for the current cell type
    subTable = sumTable(sumTable.cellType == cellTypes(c), {'date', 'cellID', 'barList', 'lags', 'eiCorr','meanEIOffset'});

    if c == 1
        plotList = [4 5 5 5 9];
        delayList = [4 6 1 4 8];
    else
        plotList = [3 2 2 3 5 4 3 3 1 3];
        delayList = [3 3 4 4 4 3 3 5 3 3];
    end

    % Loop through each cell in the current cell type
    tpTime = zeros(1, height(subTable)); % Preallocate tpTime for efficiency

    for cellIdx = 1:height(subTable)
        cellData = subTable(cellIdx, :);

        lags = cellData.lags; % Assuming lags are stored in rows corresponding to bar sizes
        corrVals = cellData.eiCorr{1}(plotList(cellIdx), :); % Assuming correlations are stored similarly

        % Plot the correlation curve
        figure(figure1); % Switch to figure1
        plot(lags, corrVals, 'LineWidth', 2, 'Color', colors(c, :));

        tpTime(cellIdx) = -cellData.meanEIOffset{1}(delayList(cellIdx));
    end

    % Plot the scatter plot
    figure(figure2); % Switch to figure2
    scatter(c*ones(1, height(subTable))+0.1*rand(1,height(subTable)), tpTime, 150, colors(c, :),'filled');

    xlim([0.5 2.5]);
end

% Labels and titles for the figures
figure(figure1);
xlabel('Lags');
ylabel('Correlation');
title('Correlation Curves');
% Add labels and title
xlabel('Lag (ms)');
ylabel('Cross-correlation');
xlim([-200, 200]);


figure(figure2);
xlabel('Cell Index');
ylabel('Temporal Offset (ms)');
title('Scatter Plot of Time Points');
set(gca,'xtick',[ 1 2],'xticklabel',{'OffS','OffT'})

%% population level summary EI analysis of CRG
clc; clear barID
CloseAllFiguresExceptGUI;
% Load the data into a table
load('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/EISummary.mat');
sumTable = struct2table(EISummary);
sumTable.cellType = categorical(sumTable.cellType);
% Get unique cell types
cellTypes = unique(sumTable.cellType);

% Loop through each cell type
for c = 1:numel(cellTypes)
    % Extract subset of the table for the current cell type
    subTable = sumTable(sumTable.cellType == cellTypes(c), {'date', 'cellID', 'barList', 'lags', 'eiCorr'});
    % Loop through each cell in the current cell type
    for cellIdx = 1:height(subTable)
        cellData = subTable(cellIdx, :);
        % Create a new figure for each cell
        figure('Name', sprintf('Cell Type: %s, Cell ID: %s', char(cellTypes(c)), cellData.cellID{1}), 'Color', 'w');
        hold on;
        % Plot correlation over lags for each bar size
        for barIdx = 1:numel(cellData.barList{1})
            lags = cellData.lags(1, :); % Assuming lags are stored in rows corresponding to bar sizes
            corrVals = cellData.eiCorr{1}(barIdx, :); % Assuming correlations are stored similarly
            % Plot the correlation curve
            plot(lags, corrVals, 'LineWidth', 2, 'DisplayName', sprintf('Bar Size: %d', cellData.barList{1}(barIdx)));
        end
        % Add labels and title
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
    % Extract subset of the table for the current cell type
    subTable = sumTable(sumTable.cellType == cellTypes(c), {'date', 'cellID', 'barList', 'lags', 'eiCorr','meanEIOffset'});
    if c == 1
        plotList = [4 5 5 5 9];
        delayList = [4 6 1 4 8];
    else
        plotList = [3 2 2 3 5 4 3 3 1 3];
        delayList = [3 3 4 4 4 3 3 5 3 3];
    end
    % Loop through each cell in the current cell type
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

%% New figures: Compare across bar sizes (population average per bar size)
% Define the bar sizes we want to analyze
targetBarList = [10, 20, 40, 80];
nBars = numel(targetBarList);

% Generate a colormap for bar sizes
barColors = lines(nBars);

% Figure 3: Mean correlation curves across cells for each bar size
figure3 = figure('Name', 'Population Correlation by Bar Size', 'Color', 'w');

% Figure 4: Temporal offset across bar sizes
figure4 = figure('Name', 'Temporal Offset by Bar Size', 'Color', 'w');
hold on;

for c = 1:numel(cellTypes)
    subTable = sumTable(sumTable.cellType == cellTypes(c), {'date', 'cellID', 'barList', 'lags', 'eiCorr', 'meanEIOffset'});
    nCells = height(subTable);
    
    % Get lags from first cell (assuming consistent across cells)
    lags = subTable.lags(1, :);
    nLags = numel(lags);
    
    % Preallocate matrices: rows = cells, cols = lags, depth = bar sizes
    corrMatrix = NaN(nCells, nLags, nBars);
    offsetMatrix = NaN(nCells, nBars);
    
    for cellIdx = 1:nCells
        cellData = subTable(cellIdx, :);
        cellBarList = cellData.barList{1};
        
        for barIdx = 1:nBars
            % Find this bar size in the cell's bar list
            barMatch = find(cellBarList == targetBarList(barIdx), 1);
            if ~isempty(barMatch)
                corrMatrix(cellIdx, :, barIdx) = cellData.eiCorr{1}(barMatch, :);
                offsetMatrix(cellIdx, barIdx) = -cellData.meanEIOffset{1}(barMatch);
            end
        end
    end
    
% Plot mean correlation for each bar size
% Plot mean correlation for each bar size
figure(figure3);
subplot(1, numel(cellTypes), c);
hold on;
plotHandles = gobjects(nBars, 1);
for barIdx = 1:nBars
    meanCorr = nanmean(corrMatrix(:, :, barIdx), 1);
    semCorr = nanstd(corrMatrix(:, :, barIdx), 0, 1) / sqrt(sum(~isnan(corrMatrix(:, 1, barIdx))));
    
    % Plot mean with error bars (every 20th point to avoid clutter)
    errIdx = 1:20:nLags;
    plotHandles(barIdx) = errorbar(lags(errIdx), meanCorr(errIdx), semCorr(errIdx), ...
        'o-', 'Color', barColors(barIdx, :), 'LineWidth', 2, 'CapSize', 4, ...
        'MarkerFaceColor', barColors(barIdx, :), 'MarkerSize', 4);
end
xlabel('Lag (ms)');
ylabel('Cross-correlation');
xlim([-200, 200]);
title(sprintf('%s (n=%d)', char(cellTypes(c)), nCells));
legend(plotHandles, arrayfun(@(x) sprintf('%d µm', x), targetBarList, 'UniformOutput', false), 'Location', 'best');
hold off;
    
    % Plot temporal offset for each bar size (similar to original figure2 style)
    figure(figure4);
    for barIdx = 1:nBars
        validIdx = ~isnan(offsetMatrix(:, barIdx));
        validOffsets = offsetMatrix(validIdx, barIdx);
        nValid = sum(validIdx);
        xPos = (c - 1) * (nBars + 1) + barIdx; % Group by cell type with spacing
        scatter(xPos * ones(1, nValid) + 0.1 * rand(1, nValid), validOffsets, 150, barColors(barIdx, :), 'filled');
    end
end

% Format figure 3
figure(figure3);
sgtitle('Population Cross-correlation by Bar Size');

% Format figure 4
figure(figure4);
xlabel('Bar Size');
ylabel('Temporal Offset (ms)');
title('Temporal Offset by Bar Size');
% Create x-tick positions and labels
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

% Add cell type labels
ax = gca;
for c = 1:numel(cellTypes)
    centerX = (c - 1) * (nBars + 1) + (nBars + 1) / 2;
    text(centerX, ax.YLim(1) - 0.1 * diff(ax.YLim), char(cellTypes(c)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12);
end


%% split for Linear Disc  or flashed Grating (same as the one for contrast reversing grating)
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

%% analyze linear Disc
selectedNodes = gui.getSelectedEpochTreeNodes;
% selectedNodes{2}=selectedNodes{1};
paras.spikeTh=1.6;
clc; CloseAllFiguresExceptGUI;
paras.spikeTag=0;
paras.psthSigma=20;
paras.rmreps=[  ];
preTime=selectedNodes{1}.epochList.firstValue.protocolSettings('preTime');
stimTime=selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');
tailTime=selectedNodes{1}.epochList.firstValue.protocolSettings('tailTime');
paras.sampleRate=selectedNodes{1}.epochList.firstValue.protocolSettings('sampleRate');
paras.wcoffset=100;  % pts
paras.spikeoffset=300;  % ptsm
paras.nSamples=30;
paras.plotChoice='multiple';  % Options: 'maxmin', 'multiple'
timeToPts=@(x) x/1e3*paras.sampleRate;
paras.prePts=timeToPts(preTime); paras.stimPts=timeToPts(stimTime); paras.tailPts=timeToPts(tailTime);
output=analyzeLinearDisc(selectedNodes,paras);

%% save cell information for population analysis
% cluster 1 linear,  cluster 3 nonlinear; cluster in between,  swap if not

% output.cellType='OffS';
clc; meanLuminance=input ('enter the mean luminance:');

load('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/linearEqvDiscNew.mat');
% First add new fields to all entries with NaN default values
for i = 1:length(linearDiscSummary)
    if ~isfield(linearDiscSummary(i), 'respOnset')
        linearDiscSummary(i).respOnset = NaN;
    end
    if ~isfield(linearDiscSummary(i), 'respOffset')
        linearDiscSummary(i).respOffset = NaN;
    end
end

% Try to find matching entry
foundMatch = false;
for i = 1:length(linearDiscSummary)
    if isequal(linearDiscSummary(i).date, output.expDate) && ...
            isequal(linearDiscSummary(i).cellID, output.cellLabel) && ...
            isequal(linearDiscSummary(i).onlineAnalysis, output.onlineAnalysis) && ...
            isequal(linearDiscSummary(i).cellType, output.cellType) && ...
            isequal(linearDiscSummary(i).meanLum, meanLuminance)

        % Found matching entry - update only the new fields
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

% If no match found, create new entry
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
save('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/linearEqvDiscOld.mat', 'linearDiscSummary');

% excInhClusterSummary(numCells+1)=struct('date',output.expDate,'cellID',output.cellLabel,'ampStats', output.stats,'NLI',output.NLI,...
%     'clusterIndex',output.clusterIndex,'clusterSummary',output.clusterSummary,'deltaEqvInt',output.deltaEqvInt);
% save('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/linearEqvDiscClusterAnalysis.mat','excInhClusterSummary');

%% save analysis information for drug experiments
clc;  meanLuminance=input ('enter the mean luminance:');  numCells=0;
drugUsed=input ('enter the drug used:','s');
try
    load('/Volumes/GoogleDrive/My Drive/projects/spatialIntegration/summary/linearEqvDiscDrug.mat');
    numCells=numel(linearDiscDrugSummary);
end
linearDiscDrugSummary(numCells+1)=struct('date',output.expDate,'cellID',output.cellLabel,'cellType', output.cellType,'onlineAnalysis',output.onlineAnalysis,...
    'meanOnsetNLI',mean(output.NLI.onset),'meanOffsetNLI',mean(output.NLI.offset),'OnsetNLI',output.NLI.onset,'OffsetNLI',output.NLI.offset,'meanLum',meanLuminance, ...
    'drugUsed',drugUsed);
save('/Volumes/GoogleDrive/My Drive/projects/spatialIntegration/summary/linearEqvDiscDrug.mat','linearDiscDrugSummary');
fprintf('%s \n', '---new cell data saved---');
%% OLD CODE FOR OLD SUMMARY population level summary analysis of linear disc
clc; load('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/linearEqvDiscOld.mat');
discTable=struct2table(linearDiscSummary);
discTable.onlineAnalysis=categorical(discTable.onlineAnalysis); discTable.cellType=categorical(discTable.cellType);
CloseAllFiguresExceptGUI;
cellTypeToInsp=categorical({'OffT','OffS','OnT'}); recTypeToInsp='inh';
subTable=discTable(discTable.onlineAnalysis==recTypeToInsp & discTable.cellType=='OffT' & (discTable.meanLum==100 | discTable.meanLum==10), ...,
    {'cellType','meanOnsetNLI','meanOffsetNLI','onlineAnalysis','meanLum'});
% ax1=categoryBoxplot(subTable.meanOnsetNLI,findgroups(subTable.cellType),unique(subTable.cellType),'meanOnset NLI');
% ax2=categoryBoxplot(subTable.meanOffsetNLI,findgroups(subTable.cellType),unique(subTable.cellType),'meanOffset NLI');
[G,gName] = findgroups(subTable.cellType);
cellMean=splitapply(@mean,subTable.meanOnsetNLI,G);
cellErr=splitapply(@(x) std(x)/sqrt(numel(x)),subTable.meanOnsetNLI,G);
figure; scatterWithMeanAndError(G,subTable.meanOnsetNLI,cellMean,cellErr,string(gName),1);


[G,gName] = findgroups(subTable.cellType);
cellMean=splitapply(@mean,subTable.meanOffsetNLI,G);
cellErr=splitapply(@(x) std(x)/sqrt(numel(x)),subTable.meanOffsetNLI,G);
figure; scatterWithMeanAndError(G,subTable.meanOffsetNLI,cellMean,cellErr,string(gName),1);

% cellTypeToInsp='OffT';
% cellTable=discTable(discTable.cellType==cellTypeToInsp & discTable.meanLum==100,{'onlineAnalysis','meanOnsetNLI','meanOffsetNLI'});
% ax3=categoryBoxplot(cellTable.meanOnsetNLI,findgroups(cellTable.onlineAnalysis),unique(cellTable.onlineAnalysis),[ cellTypeToInsp ' Onset']);
% ax4=categoryBoxplot(cellTable.meanOffsetNLI,findgroups(cellTable.onlineAnalysis),unique(cellTable.onlineAnalysis),[cellTypeToInsp ' Offset']);
% create cumulative plots for Nonlinearity Index for various cell types

% cellTypes=unique(discTable.cellType);
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


% plot the luminance dependency of offT cell
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

% cell mean
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

%%%%%%%%% overlay Exc. vs . Inh
overlayTable=discTable(discTable.cellType=='OffT' & discTable.meanLum==100 & (discTable.onlineAnalysis=='exc'| discTable.onlineAnalysis=='inh'), ...,
    {'date','cellID','OnsetNLI','onlineAnalysis'});
overlayTable([1 2 5 6 7 10 11 12 17 18 23 24 25],:)=[];

% cell mean
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

%% population level summary analysis of linear disc
clc; load('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/olderSummary/linearEqvDiscOld.mat');
discTable = struct2table(linearDiscSummary);
discTable.onlineAnalysis = categorical(discTable.onlineAnalysis);
discTable.cellType = categorical(discTable.cellType);
CloseAllFiguresExceptGUI;

% Set analysis parameters
cellTypeToInsp = categorical({'OffS','OffT'});
recTypeToInsp = 'exc';

% Filter data for analysis
subTable = discTable(discTable.onlineAnalysis==recTypeToInsp & discTable.meanLum==100, :);

% Get unique groups
[G, gName] = findgroups(subTable.cellType);

% Create figure for NLI comparisons
figure('Position', [100 100 1200 800]);

% Plot mean OnsetNLI
subplot(2,2,1);
onsetNLIs = cellfun(@median, subTable.OnsetNLI); % Compute mean for each cell
cellMean = splitapply(@median, onsetNLIs, G);
cellErr = splitapply(@(x) std(x)/sqrt(numel(x)), onsetNLIs, G);
scatterWithMeanAndError(G, onsetNLIs, cellMean, cellErr, string(gName), 1);
title('Mean Onset NLI');
ylabel('NLI');

% Plot mean OffsetNLI
subplot(2,2,2);
offsetNLIs = cellfun(@median, subTable.OffsetNLI); % Compute mean for each cell
cellMean = splitapply(@median, offsetNLIs, G);
cellErr = splitapply(@(x) std(x)/sqrt(numel(x)), offsetNLIs, G);
scatterWithMeanAndError(G, offsetNLIs, cellMean, cellErr, string(gName), 1);
title('Mean Offset NLI');
ylabel('NLI');

% Cumulative plots for Onset NLI
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

% Cumulative plots for Offset NLI
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

% Create figure for response comparisons
figure('Position', [100 100 1200 400]);

% Get color map for cell types
colors = pmkmp(numel(cellTypeToInsp),'IsoL');

% Scatter plot for onset responses
subplot(1,2,1);
hold all;
h = zeros(numel(cellTypeToInsp), 1); % Handle array for legend
for c = 1:numel(cellTypeToInsp)
    cellData = subTable(subTable.cellType==cellTypeToInsp(c), :);
    scatter(cellData.respOnset(:,1), cellData.respOnset(:,2), 100, colors(c,:), 'filled', 'MarkerFaceAlpha', 0.6);

    % Calculate mean and SEM for this cell type
    meanX = mean(cellData.respOnset(:,1));
    meanY = mean(cellData.respOnset(:,2));
    semX = std(cellData.respOnset(:,1))/sqrt(height(cellData));
    semY = std(cellData.respOnset(:,2))/sqrt(height(cellData));

    % Plot error bars with matching color and save handle for legend
    h(c) = errorbar(meanX, meanY, semY, semY, semX, semX, 'Color', colors(c,:), 'LineWidth', 3);
end
plot([0 max([xlim ylim])], [0 max([xlim ylim])], '--k');
xlabel('Image Response');
ylabel('Disc Response');
title('Onset Responses');
legend(h, cellTypeToInsp, 'Location', 'SE','fontsize',24); legend boxoff;
axis square;

% Scatter plot for normalized offset responses
subplot(1,2,2);
hold all;
h = zeros(numel(cellTypeToInsp), 1); % Handle array for legend
for c = 1:numel(cellTypeToInsp)
    cellData = subTable(subTable.cellType==cellTypeToInsp(c), :);

    % Get max onset response for each cell
    maxOnsetPerCell = max(cellData.respOnset, [], 2);

    % Normalize offset responses by max onset
    normRespOffset = cellData.respOffset ./ maxOnsetPerCell;

    scatter(normRespOffset(:,1), normRespOffset(:,2), 100, colors(c,:), 'filled', 'MarkerFaceAlpha', 0.6);

    % Calculate mean and SEM for normalized data
    meanX = mean(normRespOffset(:,1));
    meanY = mean(normRespOffset(:,2));
    semX = std(normRespOffset(:,1))/sqrt(height(cellData));
    semY = std(normRespOffset(:,2))/sqrt(height(cellData));

    % Plot error bars with matching color and save handle for legend
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


% another way of visualization
% Create new figure for combined onset/offset comparison
figure('Position', [100 100 600 600]);
hold all;


% Handles for legend
h_onset = zeros(numel(cellTypeToInsp), 1);
h_offset = zeros(numel(cellTypeToInsp), 1);

for c = 1:numel(cellTypeToInsp)
    cellData = subTable(subTable.cellType==cellTypeToInsp(c), :);

    % Plot individual points onset (hollow)
    scatter(cellData.respOnset(:,1), cellData.respOnset(:,2), 50, colors(c,:), 'LineWidth', 1.5, 'MarkerFaceAlpha', 0.2);

    % Plot individual points offset (filled)
    scatter(cellData.respOffset(:,1), cellData.respOffset(:,2), 50, colors(c,:), 'filled', 'MarkerFaceAlpha', 0.2);

    % Calculate and plot mean & SEM for onset
    meanX = mean(cellData.respOnset(:,1));
    meanY = mean(cellData.respOnset(:,2));
    semX = std(cellData.respOnset(:,1))/sqrt(height(cellData));
    semY = std(cellData.respOnset(:,2))/sqrt(height(cellData));
    % Plot error bars
    errorbar(meanX, meanY, semY, semY, semX, semX, 'Color', colors(c,:), 'LineWidth', 2);
    % Plot mean point with larger hollow marker
    h_onset(c) = scatter(meanX, meanY, 200, colors(c,:), 'LineWidth', 3);

    % Calculate and plot mean & SEM for offset
    meanX = mean(cellData.respOffset(:,1));
    meanY = mean(cellData.respOffset(:,2));
    semX = std(cellData.respOffset(:,1))/sqrt(height(cellData));
    semY = std(cellData.respOffset(:,2))/sqrt(height(cellData));
    % Plot error bars
    errorbar(meanX, meanY, semY, semY, semX, semX, 'Color', colors(c,:), 'LineWidth', 2);
    % Plot mean point with larger filled marker
    h_offset(c) = scatter(meanX, meanY, 200, colors(c,:), 'filled');
end

plot([0 max([xlim ylim])], [0 max([xlim ylim])], '--k');
xlabel('Image Response');
ylabel('Disc Response');
title('Onset (hollow) and Offset (filled) Responses');

% Create legend entries correctly from categorical array
cellTypeNames = categories(cellTypeToInsp); % Get cell type names
legend_entries = {};
h_combined = [];

% Build handles and legend entries in matching order
for i = 1:numel(cellTypeNames)
    h_combined = [h_combined; h_onset(i); h_offset(i)];
    legend_entries{end+1} = [cellTypeNames{i} ' onset'];
    legend_entries{end+1} = [cellTypeNames{i} ' offset'];
end

legend(h_combined, legend_entries, 'Location', 'SE', 'fontsize', 24);
axis square;


%% population linear disc drug
clc; CloseAllFiguresExceptGUI;
load('/Volumes/GoogleDrive/My Drive/projects/spatialIntegration/summary/linearEqvDiscDrug.mat');
discDrugTable=struct2table(linearDiscDrugSummary);
discDrugTable.onlineAnalysis=categorical(discDrugTable.onlineAnalysis); discDrugTable.cellType=...
    categorical(discDrugTable.cellType); discDrugTable.drugUsed=categorical(discDrugTable.drugUsed);

subTable=discDrugTable(discDrugTable.meanLum==100,{'meanOnsetNLI','drugUsed','date','cellID'});
% cell mean
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
% set(gca,'xtick', log(meanLumList),'xticklabels',split(num2str(meanLumList'))); xlabel('Mean luminance'); ylabel('Onset NLI');

%% population linear disc inh cluster analysis
clc; CloseAllFiguresExceptGUI;
load('/Volumes/GoogleDrive/My Drive/projects/spatialIntegration/summary/linearEqvDiscClusterAnalysis.mat');
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
    %     legend(gca(sf2), h2, {'cluster 1', 'cluster 2','cluster 3'});  legend boxoff;
    %     legend(gca(sf3), h3, {'cluster 1', 'cluster 2','cluster 3'});  legend boxoff;
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

%% analyze flashed gratings
clc; selectedNodes = gui.getSelectedEpochTreeNodes;
CloseAllFiguresExceptGUI;
paras.spikeTh=1.2;
paras.spikeTag=0;
paras.psthSigma=10;
paras.spikeoffset=0;
paras.wcoffset=100;  % pts
preTime=selectedNodes{1}.epochList.firstValue.protocolSettings('preTime');
stimTime=selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');
tailTime=selectedNodes{1}.epochList.firstValue.protocolSettings('tailTime');
paras.sampleRate=selectedNodes{1}.epochList.firstValue.protocolSettings('sampleRate');
timeToPts=@(x) x/1e3*paras.sampleRate;
paras.prePts=timeToPts(preTime); paras.stimPts=timeToPts(stimTime); paras.tailPts=timeToPts(tailTime);
[f,stats]=analyzeFlashGrating(selectedNodes,paras); ax=gca(f(1));
fprintf('%s %d %s %d \n','preTime-- ',preTime, ' --stimTime-- ', stimTime);
%% save cell data for population analysis, for flashed grating
clc; clear flashGSummary;
stats.onlineAnalysis
meanLuminance=input('enter the mean luminance:');
numCells=0;
try
    load('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/flashedGrating.mat');
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

save('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/flashedGrating.mat','flashGSummary');
fprintf('%s \n', '---new cell data saved---');

%% flash grate summary visualization
clc; clear flashGDrugSummary; % meanLuminance=input ('enter the mean luminance:');
meanLuminance=100;
stats.onlineAnalysis
numCells=0;
drugUsed=input ('enter the drug used:','s');
try
    load('/Volumes/GoogleDrive/My Drive/projects/spatialIntegration/summary/flashedGratingDrug.mat');
    numCells=numel(flashGDrugSummary);
end
flashGDrugSummary(numCells+1)=struct('date',stats.expDate,'cellID',stats.cellLabel,'cellType', stats.cellType,'onlineAnalysis',stats.onlineAnalysis,'barList',stats.barList, ...
    'OnsetResponse',stats.onset,'OffsetResponse',stats.offset,'OnsetAmp',stats.peakOnset,'OffsetAmp',stats.peakOffset,'baselineResponse', stats.baseline,'meanLum',meanLuminance,'drugUsed',drugUsed);
save('/Volumes/GoogleDrive/My Drive/projects/spatialIntegration/summary/flashedGratingDrug.mat','flashGDrugSummary');
fprintf('%s \n', '---new cell data saved---');

%% population level summary analysis of flashed gratings

clc; load('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/flashedGratingOld.mat');
sumTable=struct2table(flashGSummary);
sumTable.recType=categorical(sumTable.recType); sumTable.cellType=categorical(sumTable.cellType);
CloseAllFiguresExceptGUI;
figure('color','w','position',[200 200 900 900]);
cellTypes=unique(sumTable.cellType);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
recTypeInsp='extracellular';
lightLevelInsp=100;
normalizeResponses=false; % Set to false to use raw (unnormalized) values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Figure 1: Onset Response (Area Sum)
for c=1:numel(cellTypes)
    resTable=sumTable(sumTable.cellType==cellTypes(c) & sumTable.recType==recTypeInsp& ...
        (sumTable.meanLum==lightLevelInsp ),{'cellType','barList','baselineResponse','OnsetResponse','OffsetResponse'});
    try
        ax(c)=subplot(3,2,c); hold all;

        % Collect all bar sizes and responses
        allBars = [];
        allResponses = {};

        % First collect all bar sizes and responses, normalizing if needed
        for i=1:size(resTable,1)
            barList = cell2mat(resTable.barList(i));
            onsetResp = cell2mat(resTable.OnsetResponse(i));

            % Normalize if requested
            if normalizeResponses
                maxAbsValue = max(abs(onsetResp));
                if maxAbsValue > 1e-6
                    onsetResp = onsetResp / maxAbsValue;
                end
            end

            % Store response and bar list
            allResponses{i} = onsetResp;
            allBars = [allBars, barList];

            % Plot individual cell responses
            if strcmp(recTypeInsp,'exc')
                plot(barList, onsetResp, 'color', [0 0 0 0.3], 'linewidth', 0.5);
            else
                plot(barList, onsetResp, 'color', [0 0 0 0.3], 'linewidth', 0.5);
            end
        end

        % Compute the mean and error for each bar size
        [G, barID{c}] = findgroups(allBars);

        % Concatenate all responses
        allConcatenatedResponses = cat(2, allResponses{:});

        % Calculate mean and standard error
        barMean.(char(cellTypes(c))) = splitapply(@mean, allConcatenatedResponses, G);
        barErr.(char(cellTypes(c))) = splitapply(@(x) std(x)/sqrt(numel(x)), allConcatenatedResponses, G);

        % Plot mean with error bars
        errorbar(barID{c}, barMean.(char(cellTypes(c))), barErr.(char(cellTypes(c))), 'r', 'linewidth', 3);

        title(char(cellTypes(c)));
    end

    % Update title based on normalization
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

        % Collect all bar sizes and responses
        allBars = [];
        allResponses = {};

        % First collect all bar sizes and responses, normalizing if needed
        for i=1:size(resTable,1)
            barList = cell2mat(resTable.barList(i));
            offsetResp = cell2mat(resTable.OffsetResponse(i));

            % Normalize if requested
            if normalizeResponses
                maxAbsValue = max(abs(offsetResp));
                if maxAbsValue > 1e-6
                    offsetResp = offsetResp / maxAbsValue;
                end
            end

            % Store response and bar list
            allResponses{i} = offsetResp;
            allBars = [allBars, barList];

            % Plot individual cell responses
            plot(barList, offsetResp, 'color', [0 0 0 0.3], 'linewidth', 0.5);
        end

        % Compute the mean and error for each bar size
        [G, barID{c}] = findgroups(allBars);

        % Concatenate all responses
        allConcatenatedResponses = cat(2, allResponses{:});

        % Calculate mean and standard error
        barMean.(char(cellTypes(c))) = splitapply(@mean, allConcatenatedResponses, G);
        barErr.(char(cellTypes(c))) = splitapply(@(x) std(x)/sqrt(numel(x)), allConcatenatedResponses, G);

        % Plot mean with error bars
        errorbar(barID{c}, barMean.(char(cellTypes(c))), barErr.(char(cellTypes(c))), 'r', 'linewidth', 3);

        title(char(cellTypes(c)));
    end

    % Update title based on normalization
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

            % Collect all bar sizes and responses
            allBars = [];
            allResponses = {};

            % First collect all bar sizes and responses, normalizing if needed
            for i=1:size(resTable,1)
                barList = cell2mat(resTable.barList(i));
                onsetAmp = cell2mat(resTable.OnsetAmplitude(i));

                % Normalize if requested
                if normalizeResponses
                    maxAbsValue = max(abs(onsetAmp));
                    if maxAbsValue > 1e-6
                        onsetAmp = onsetAmp / maxAbsValue;
                    end
                end

                % Store response and bar list
                allResponses{i} = onsetAmp;
                allBars = [allBars, barList];

                % Plot individual cell responses
                if strcmp(recTypeInsp,'exc')
                    plot(barList, onsetAmp, 'color', [0 0 0 0.3], 'linewidth', 0.5);
                else
                    plot(barList, onsetAmp, 'color', [0 0 0 0.3], 'linewidth', 0.5);
                end
            end

            % Compute the mean and error for each bar size
            [G, barID{c}] = findgroups(allBars);

            % Concatenate all responses
            allConcatenatedResponses = cat(2, allResponses{:});

            % Calculate mean and standard error
            barMean.(char(cellTypes(c))) = splitapply(@mean, allConcatenatedResponses, G);
            barErr.(char(cellTypes(c))) = splitapply(@(x) std(x)/sqrt(numel(x)), allConcatenatedResponses, G);

            % Plot mean with error bars
            errorbar(barID{c}, barMean.(char(cellTypes(c))), barErr.(char(cellTypes(c))), 'r', 'linewidth', 3);

            title(char(cellTypes(c)));
        end

        % Update title based on normalization
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

            % Collect all bar sizes and responses
            allBars = [];
            allResponses = {};

            % First collect all bar sizes and responses, normalizing if needed
            for i=1:size(resTable,1)
                barList = cell2mat(resTable.barList(i));
                offsetAmp = cell2mat(resTable.OffsetAmplitude(i));

                % Normalize if requested
                if normalizeResponses
                    maxAbsValue = max(abs(offsetAmp));
                    if maxAbsValue > 1e-6
                        offsetAmp = offsetAmp / maxAbsValue;
                    end
                end

                % Store response and bar list
                allResponses{i} = offsetAmp;
                allBars = [allBars, barList];

                % Plot individual cell responses
                plot(barList, offsetAmp, 'color', [0 0 0 0.3], 'linewidth', 0.5);
            end

            % Compute the mean and error for each bar size
            [G, barID{c}] = findgroups(allBars);

            % Concatenate all responses
            allConcatenatedResponses = cat(2, allResponses{:});

            % Calculate mean and standard error
            barMean.(char(cellTypes(c))) = splitapply(@mean, allConcatenatedResponses, G);
            barErr.(char(cellTypes(c))) = splitapply(@(x) std(x)/sqrt(numel(x)), allConcatenatedResponses, G);

            % Plot mean with error bars
            errorbar(barID{c}, barMean.(char(cellTypes(c))), barErr.(char(cellTypes(c))), 'r', 'linewidth', 3);

            title(char(cellTypes(c)));
        end

        % Update title based on normalization
        if normalizeResponses
            st=sgtitle('Offset Amplitude (Normalized)');
        else
            st=sgtitle('Offset Amplitude');
        end
        set(st,'fontsize',26);
    end

end
% Figure 5: Scatter plot of area sum onset vs offset (always raw values)
f1=figure('color','w','position',[300 300 600 600]);
barToInspect=80; cellTypeInsp='OffT';
resTable=sumTable(sumTable.cellType==cellTypeInsp & sumTable.recType==recTypeInsp & ...
    sumTable.meanLum==lightLevelInsp,{'cellType','barList','baselineResponse','OnsetResponse','OffsetResponse'});
barLists=cat(2,resTable.barList{:}); res.onset=cat(2,resTable.OnsetResponse{:}); res.offset=cat(2,resTable.OffsetResponse{:});
res.onset=res.onset(barLists==barToInspect);   res.offset=res.offset(barLists==barToInspect);

% Calculate mean and standard error
meanOnset = mean(res.onset);
meanOffset = mean(res.offset);
semOnset = std(res.onset)/sqrt(numel(res.onset));
semOffset = std(res.offset)/sqrt(numel(res.offset));

hold all;
% Plot individual cell responses
scatter(res.onset, res.offset, 100, 'k', 'filled', 'MarkerFaceAlpha', 0.7);

% Plot mean with error bars
h = errorbar(meanOnset, meanOffset, semOffset, semOffset, semOnset, semOnset, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'LineWidth', 2);
h.CapSize = 10;

% Add reference lines
xline(0, 'k--');
yline(0, 'k--');

% Add identity line
xmin = min(min(res.onset), 0) * 1.1;
xmax = max(max(res.onset), 0) * 1.1;
ymin = min(min(res.offset), 0) * 1.1;
ymax = max(max(res.offset), 0) * 1.1;
axisLim = [min(xmin, ymin), max(xmax, ymax)];
plot(axisLim, axisLim, 'k--');

% Set axes limits
axis equal;
xlim(axisLim);
ylim(axisLim);

% Add title and labels
title([char(cellTypeInsp), ' Area Sum, Bar Width = ', num2str(barToInspect)], 'FontSize', 14);
hold off;
setAxes(f1);
initFig(gca(f1), 'Onset Area Sum', 'Offset Area Sum');

try
    % Figure 6: Scatter plot of amplitude onset vs offset (always raw values)
    f2=figure('color','w','position',[300 300 600 600]);
    resTable=sumTable(sumTable.cellType==cellTypeInsp & sumTable.recType==recTypeInsp & ...
        sumTable.meanLum==lightLevelInsp,{'cellType','barList','OnsetAmplitude','OffsetAmplitude'});
    barLists=cat(2,resTable.barList{:}); res.amp_onset=cat(2,resTable.OnsetAmplitude{:}); res.amp_offset=cat(2,resTable.OffsetAmplitude{:});
    res.amp_onset=res.amp_onset(barLists==barToInspect);   res.amp_offset=res.amp_offset(barLists==barToInspect);

    % Calculate mean and standard error
    meanOnsetAmp = mean(res.amp_onset);
    meanOffsetAmp = mean(res.amp_offset);
    semOnsetAmp = std(res.amp_onset)/sqrt(numel(res.amp_onset));
    semOffsetAmp = std(res.amp_offset)/sqrt(numel(res.amp_offset));

    hold all;
    % Plot individual cell responses
    scatter(res.amp_onset, res.amp_offset, 100, 'k', 'filled', 'MarkerFaceAlpha', 0.7);

    % Plot mean with error bars
    h = errorbar(meanOnsetAmp, meanOffsetAmp, semOffsetAmp, semOffsetAmp, semOnsetAmp, semOnsetAmp, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'LineWidth', 2);
    h.CapSize = 10;

    % Add reference lines
    xline(0, 'k--');
    yline(0, 'k--');

    % Add identity line
    xmin = min(min(res.amp_onset), 0) * 1.1;
    xmax = max(max(res.amp_onset), 0) * 1.1;
    ymin = min(min(res.amp_offset), 0) * 1.1;
    ymax = max(max(res.amp_offset), 0) * 1.1;
    axisLim = [min(xmin, ymin), max(xmax, ymax)];
    plot(axisLim, axisLim, 'k--');

    % Set axes limits
    axis equal;
    xlim(axisLim);
    ylim(axisLim);

    % Add title and labels
    title([char(cellTypeInsp), ' Amplitude, Bar Width = ', num2str(barToInspect)], 'FontSize', 14);
    hold off;
    setAxes(f2);
    initFig(gca(f2), 'Onset Amplitude', 'Offset Amplitude');
end
% Figure 7: I/E ratio analysis
IETable=sumTable(sumTable.cellType=='OffT' & sumTable.recType~='extracellular' & ...
    sumTable.meanLum==100,{'date','cellID','recType','barList','OnsetResponse','OffsetResponse'});
[G,ID]=findgroups(IETable(:,{'date', 'cellID'})); gList=unique(G);

% Create a figure for visualizing I/E ratio across bar sizes
figure('color','w','position',[400 400 800 400]);

% Prepare data structures for storing IE ratios for each bar size
allBarSizes = [];
for i=1:size(IETable,1)
    % Directly access the cell array content
    barSizes = IETable.barList{i};
    allBarSizes = [allBarSizes, barSizes];
end
uniqueBarSizes = unique(allBarSizes);

% Initialize data structures
ieRatio.onset = cell(length(uniqueBarSizes), 1);
ieRatio.offset = cell(length(uniqueBarSizes), 1);
for i=1:length(uniqueBarSizes)
    ieRatio.onset{i} = [];
    ieRatio.offset{i} = [];
end

% Calculate IE ratios for all paired cells across all bar sizes
for i=1:numel(gList)
    gIndex=find(gList(i)==G);
    if numel(gIndex)==2 % Must have both exc and inh recordings for this cell
        % Get the bar lists and responses for both recordings
        excIdx = find(strcmp(IETable.recType(gIndex), 'exc'));
        inhIdx = find(strcmp(IETable.recType(gIndex), 'inh'));

        if ~isempty(excIdx) && ~isempty(inhIdx)
            excIdx = gIndex(excIdx);
            inhIdx = gIndex(inhIdx);

            % Get bar lists
            barList_exc = IETable.barList{excIdx};
            barList_inh = IETable.barList{inhIdx};

            % Get onset and offset responses
            onset_exc = IETable.OnsetResponse{excIdx};
            onset_inh = IETable.OnsetResponse{inhIdx};
            offset_exc = IETable.OffsetResponse{excIdx};
            offset_inh = IETable.OffsetResponse{inhIdx};

            % Calculate IE ratio for each bar size
            for j=1:length(uniqueBarSizes)
                barSize = uniqueBarSizes(j);

                % Find indices for this bar size in both recordings
                exc_barIdx = find(barList_exc == barSize);
                inh_barIdx = find(barList_inh == barSize);

                % Print some debug info for first few iterations
                if i <= 3 && j <= 3
                    fprintf('Cell %d, Bar size %d: ', i, barSize);
                    fprintf('exc indices: %s, inh indices: %s\n', mat2str(exc_barIdx), mat2str(inh_barIdx));

                    if ~isempty(exc_barIdx) && ~isempty(inh_barIdx)
                        fprintf('  exc onset: %f, inh onset: %f\n', onset_exc(exc_barIdx(1)), onset_inh(inh_barIdx(1)));
                        fprintf('  exc offset: %f, inh offset: %f\n', offset_exc(exc_barIdx(1)), offset_inh(inh_barIdx(1)));
                    end
                end

                % Only calculate ratio if we have data for this bar size in both recordings
                if ~isempty(exc_barIdx) && ~isempty(inh_barIdx)
                    % For onset - handle each value in the arrays separately
                    for e_idx = 1:length(exc_barIdx)
                        for i_idx = 1:length(inh_barIdx)
                            % Get the actual values
                            excVal = onset_exc(exc_barIdx(e_idx));
                            inhVal = onset_inh(inh_barIdx(i_idx));

                            % Check for division by zero
                            if abs(excVal + inhVal) > 1e-6
                                % Calculate onset IE ratio
                                onsetRatio = (inhVal - excVal) / (inhVal + excVal);
                                ieRatio.onset{j} = [ieRatio.onset{j}, onsetRatio];
                            end

                            % Calculate offset ratio similarly
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

% Calculate mean and SEM for IE ratios at each bar size
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

% Plot the IE ratios
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

% Continue with specific bar size analysis as in the original code
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


%% analyze drug population data
clc; clear barMean barErr
load('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/flashedGratingDrugOld.mat');
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

% Leave those with two (control, drug) reps in the grouping
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
% Set appropriate y-label based on recording type
if strcmp(recTypeInsp,'exc')
    ylabel('Response Amplitude');
else
    ylabel('Spike Rate (spikes/s)');
end

% Figure 1: Paired cell responses for barInsp (offset response)
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
% Set appropriate y-label based on recording type
if strcmp(recTypeInsp,'exc')
    ylabel('Response Amplitude');
else
    ylabel('Spike Rate (spikes/s)');
end

% Figure 1: Paired cell peak amplitudes (onset)
subplot(2,2,3); hold all;
title('Onset Amplitude');
for i=1:numel(gList)
    temp=find(gList(i)==G);
    if numel(temp)==2
        if resTable.drugUsed(temp(1))~=resTable.drugUsed(temp(2))
            xplot=[find(ismember(conds,'control')); find(ismember(conds,drugInsp))];
            % Extract amplitude values (handling complex data structures)
            onsetAmps = zeros(1, length(temp));
            for j = 1:length(temp)
                if iscell(resTable.OnsetAmp)
                    ampValue = resTable.OnsetAmp{temp(j)};
                    % Handle case where ampValue is itself a cell array or vector
                    if iscell(ampValue)
                        ampValue = ampValue{1}; % Get first element if nested cell
                    elseif length(ampValue) > 1
                        ampValue = ampValue(1); % Get first element if vector
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
% Set appropriate y-label based on recording type
if strcmp(recTypeInsp,'exc')
    ylabel('Peak Amplitude');
else
    ylabel('Peak Spike Rate (spikes/s)');
end

% Figure 1: Paired cell peak amplitudes (offset)
subplot(2,2,4); hold all;
title('Offset Amplitude');
for i=1:numel(gList)
    temp=find(gList(i)==G);
    if numel(temp)==2
        if resTable.drugUsed(temp(1))~=resTable.drugUsed(temp(2))
            xplot=[find(ismember(conds,'control')); find(ismember(conds,drugInsp))];
            % Extract amplitude values (handling complex data structures)
            offsetAmps = zeros(1, length(temp));
            for j = 1:length(temp)
                if iscell(resTable.OffsetAmp)
                    ampValue = resTable.OffsetAmp{temp(j)};
                    % Handle case where ampValue is itself a cell array or vector
                    if iscell(ampValue)
                        ampValue = ampValue{1}; % Get first element if nested cell
                    elseif length(ampValue) > 1
                        ampValue = ampValue(1); % Get first element if vector
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
% Set appropriate y-label based on recording type
if strcmp(recTypeInsp,'exc')
    ylabel('Peak Amplitude');
else
    ylabel('Peak Spike Rate (spikes/s)');
end

sgtitle([cellTypeToInsp ' cells - ' drugInsp ' vs control'], 'FontSize', 14);

% Use filtered data for subsequent analysis
resTable=resTable(drugG,:);

% Figure 2: Size tuning curves - Onset Response
f1=figure('color','w','position',[200 200 1000 500]); 

% Onset response by bar size
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
% Set appropriate y-label based on recording type
if strcmp(recTypeInsp,'exc')
    ylabel('Response Amplitude');
else
    ylabel('Spike Rate (spikes/s)');
end
legend(conds); legend boxoff;

% Offset response by bar size
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

% Onset amplitude
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
                % Handle case where ampValue is itself a cell array or vector
                if iscell(ampValue)
                    ampValue = ampValue{1}; % Get first element if nested cell
                elseif length(ampValue) > 1
                    ampValue = ampValue(1); % Get first element if vector
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

% Offset amplitude
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
                % Handle case where ampValue is itself a cell array or vector
                if iscell(ampValue)
                    ampValue = ampValue{1}; % Get first element if nested cell
                elseif length(ampValue) > 1
                    ampValue = ampValue(1); % Get first element if vector
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

% Figure 3: Comparison across multiple bar sizes
f3 = figure('color','w','position',[200 700 1000 500]);

% Get all bar sizes available in the data
allBarSizes = [];
for i = 1:size(resTable,1)
    allBarSizes = [allBarSizes, resTable.barList{i}];
end
uniqueBarSizes = unique(allBarSizes);

% Skip very small bar sizes if there are many
if length(uniqueBarSizes) > 8
    % Get a representative subset
    uniqueBarSizes = uniqueBarSizes(uniqueBarSizes >= 20);
    if length(uniqueBarSizes) > 8
        uniqueBarSizes = uniqueBarSizes(1:2:end);
    end
end

% Subplot for Onset Response
subplot(1,2,1); hold all;
title('Onset Response by Bar Size');

% For each bar size, get control and drug data
ctrlOnsetMeans = zeros(size(uniqueBarSizes));
ctrlOnsetSEMs = zeros(size(uniqueBarSizes));
drugOnsetMeans = zeros(size(uniqueBarSizes));
drugOnsetSEMs = zeros(size(uniqueBarSizes));
pValues = zeros(size(uniqueBarSizes));

for i = 1:length(uniqueBarSizes)
    currentBar = uniqueBarSizes(i);
    ctrlData = [];
    drugData = [];
    
    % Collect responses for this bar size
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
    
    % Calculate means and SEMs
    ctrlOnsetMeans(i) = mean(ctrlData);
    ctrlOnsetSEMs(i) = std(ctrlData) / sqrt(length(ctrlData));
    drugOnsetMeans(i) = mean(drugData);
    drugOnsetSEMs(i) = std(drugData) / sqrt(length(drugData));
    
    % Perform t-test if enough data
    if length(ctrlData) > 1 && length(drugData) > 1
        [~, pValues(i)] = ttest2(ctrlData, drugData);
    else
        pValues(i) = NaN;
    end
end

% Plot error bars for control and drug
errorbar(uniqueBarSizes, ctrlOnsetMeans, ctrlOnsetSEMs, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
errorbar(uniqueBarSizes, drugOnsetMeans, drugOnsetSEMs, 'ro-', 'LineWidth', 2, 'MarkerSize', 8);

% Add significance markers
for i = 1:length(uniqueBarSizes)
    if ~isnan(pValues(i)) && pValues(i) < 0.05
        yPos = max([ctrlOnsetMeans(i) + ctrlOnsetSEMs(i), drugOnsetMeans(i) + drugOnsetSEMs(i)]) * 1.1;
        text(uniqueBarSizes(i), yPos, '*', 'FontSize', 16, 'HorizontalAlignment', 'center');
    end
end

xlabel('Bar Size (deg)');
% Set appropriate y-label based on recording type
if strcmp(recTypeInsp,'exc')
    ylabel('Onset Response Amplitude');
else
    ylabel('Onset Spike Rate (spikes/s)');
end
legend('Control', drugInsp, 'Location', 'best');
legend boxoff;

% Subplot for Offset Response
subplot(1,2,2); hold all;
title('Offset Response by Bar Size');

% For each bar size, get control and drug data
ctrlOffsetMeans = zeros(size(uniqueBarSizes));
ctrlOffsetSEMs = zeros(size(uniqueBarSizes));
drugOffsetMeans = zeros(size(uniqueBarSizes));
drugOffsetSEMs = zeros(size(uniqueBarSizes));
pValues = zeros(size(uniqueBarSizes));

for i = 1:length(uniqueBarSizes)
    currentBar = uniqueBarSizes(i);
    ctrlData = [];
    drugData = [];
    
    % Collect responses for this bar size
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
    
    % Calculate means and SEMs
    ctrlOffsetMeans(i) = mean(ctrlData);
    ctrlOffsetSEMs(i) = std(ctrlData) / sqrt(length(ctrlData));
    drugOffsetMeans(i) = mean(drugData);
    drugOffsetSEMs(i) = std(drugData) / sqrt(length(drugData));
    
    % Perform t-test if enough data
    if length(ctrlData) > 1 && length(drugData) > 1
        [~, pValues(i)] = ttest2(ctrlData, drugData);
    else
        pValues(i) = NaN;
    end
end

% Plot error bars for control and drug
errorbar(uniqueBarSizes, ctrlOffsetMeans, ctrlOffsetSEMs, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
errorbar(uniqueBarSizes, drugOffsetMeans, drugOffsetSEMs, 'ro-', 'LineWidth', 2, 'MarkerSize', 8);

% Add significance markers
for i = 1:length(uniqueBarSizes)
    if ~isnan(pValues(i)) && pValues(i) < 0.05
        yPos = max([ctrlOffsetMeans(i) + ctrlOffsetSEMs(i), drugOffsetMeans(i) + drugOffsetSEMs(i)]) * 1.1;
        text(uniqueBarSizes(i), yPos, '*', 'FontSize', 16, 'HorizontalAlignment', 'center');
    end
end

xlabel('Bar Size (deg)');
% Set appropriate y-label based on recording type
if strcmp(recTypeInsp,'exc')
    ylabel('Offset Response Amplitude');
else
    ylabel('Offset Spike Rate (spikes/s)');
end
legend('Control', drugInsp, 'Location', 'best');
legend boxoff;

sgtitle([cellTypeToInsp ' cells - ' drugInsp ' vs control across bar sizes'], 'FontSize', 14);
%% split for split centering
cellTypeSplit = @(listSorted)splitOnCellType(listSorted);
cellTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, cellTypeSplit);

dateSplit = @(listSorted)splitOnExperimentDate(listSorted);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, dateSplit);

splitSplit = @(listSorted)splitOnSplitField(listSorted);
splitSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, splitSplit);

ndfSplit = @(listSorted)splitOnNDFs(listSorted);
ndfSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, ndfSplit);

brightnessSplit = @(listSorted)splitOnDeviceBrightNess(listSorted);
brightnessSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, brightnessSplit);

tree = riekesuite.analysis.buildTree(listSorted,{cellTypeSplit_java,dateSplit_java, 'cell.label', 'protocolSettings(epochGroup:label)',...
    splitSplit_java,'protocolSettings(onlineAnalysis)',brightnessSplit_java, ndfSplit_java});
gui = epochTreeGUI(tree);
%% analyze split field centering protocol;
clc;
selectedNodes = gui.getSelectedEpochTreeNodes;
paras.spikeTh=1.2;
CloseAllFiguresExceptGUI;
paras.spikeTag=0;
paras.psthSigma=10;
paras.wcOffset=1300;
stimTime=selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');
preTime=selectedNodes{1}.epochList.firstValue.protocolSettings('preTime');
tailTime=selectedNodes{1}.epochList.firstValue.protocolSettings('tailTime');
paras.sampleRate=selectedNodes{1}.epochList.firstValue.protocolSettings('sampleRate');
paras.tempFreq=selectedNodes{1}.epochList.firstValue.protocolSettings('temporalFrequency');
paras.meanIntensity=selectedNodes{1}.epochList.firstValue.protocolSettings('backgroundIntensity');
timeToPts=@(x) x/1e3*paras.sampleRate;
paras.prePts=timeToPts(preTime);
paras.stimPts=timeToPts(stimTime);
[ax,output]=analyzeSplitFieldCentering(selectedNodes, paras);
output.sinoF2

%% save stats for population analysis
clc; clear splitCenterSummary;
output.recType
meanLuminance=input ('enter the mean luminance:');
cellType=shortCellType(selectedNodes{1}.epochList.firstValue.protocolSettings('source:type'));
cellType='A2';
expDate=datestr(selectedNodes{1}.epochList.elements(1).startDate','yyyy/mm/dd');
cellLabel=selectedNodes{1}.epochList.firstValue.protocolSettings('source:label');
pattern=selectedNodes{1}.parent.parent.parent.splitValue;
numCells=0;

try
    load('/Volumes/GoogleDrive/My Drive/projects/spatialIntegration/summary/splitFieldCentering.mat');
    numCells=numel(splitCenterSummary);
end
% note, two nodes data can be saved same time, control then drug node
splitCenterSummary(numCells+1)=struct('date',expDate,'cellID',cellLabel,'cellType', cellType,'recType',output.recType,'responseAmp',output.modulation.positive, ...
    'rectIndex',output.rectIndex,'meanLum',meanLuminance,'pattern',pattern);
save('/Volumes/GoogleDrive/My Drive/projects/spatialIntegration/summary/splitFieldCentering.mat','splitCenterSummary');
fprintf('%s \n', '---new cell data saved---');

%% save stats for population drug exepriments analysis
clc; clear splitCenterSummary;
output.recType
meanLuminance=input ('enter the mean luminance:');
drugUsed=input ('enter the drug used:','s');
cellType=shortCellType(selectedNodes{1}.epochList.firstValue.protocolSettings('source:type'));
cellType='A2';
expDate=datestr(selectedNodes{1}.epochList.elements(1).startDate','yyyy/mm/dd');
cellLabel=selectedNodes{1}.epochList.firstValue.protocolSettings('source:label');
pattern=selectedNodes{1}.parent.parent.parent.splitValue;
numCells=0;
% compute the drug sensitive percentage
drugSensPerc=-(output.modulation.positive(2)-output.modulation.positive(1))/output.modulation.positive(1)
try
    load('/Volumes/GoogleDrive/My Drive/projects/spatialIntegration/summary/splitFieldCentering.mat');
    numCells=numel(splitCenterDrugSummary);
end
% note, two nodes data can be saved same time, control then drug node
splitCenterDrugSummary(numCells+1)=struct('date',expDate,'cellID',cellLabel,'cellType', cellType,'recType',output.recType,'responseAmp',output.modulation.positive, ...
    'rectIndex',output.rectIndex,'drugUsed',drugUsed,'meanLum',meanLuminance,'DrugSensPerc',drugSensPerc,'pattern',pattern);
save('/Volumes/GoogleDrive/My Drive/projects/spatialIntegration/summary/splitFieldCenteringDrug.mat','splitCenterSummary');
fprintf('%s \n', '---new cell data saved---');


%% create summary plots for splitCentering protocol without drugs
clc; CloseAllFiguresExceptGUI; clear f2Ratio grps scTable;
load('/Users/chrischen/Dropbox/research/projects/spatialIntegration/summary/splitFieldCentering.mat');
scTable=struct2table(splitCenterSummary);
scTable.recType=categorical(scTable.recType);  scTable.cellType=categorical(scTable.cellType);  scTable.pattern=categorical(scTable.pattern);
scTable=scTable(scTable.recType=='exc'& scTable.meanLum==100,:);
lumList=unique(scTable.meanLum); count=0;
for i=1:numel(lumList)
    subTable=scTable(scTable.meanLum==lumList(i),:);
    [G,ID]=findgroups(subTable(:,{'date','cellID'})); gList=unique(G);
    for j=1:numel(gList)
        if numel(find(gList(j)==G))==2
            f2Ratio{i}(j)=subTable(gList(j)==G & subTable.pattern=='split-field',:).responseAmp./ ...
                subTable(gList(j)==G & subTable.pattern=='full-field',:).responseAmp;
            count=count+1; grps(count)=i;
        end
    end
end
figure;
scatterWithMeanAndError(grps,cat(2,f2Ratio{:}),cellfun(@mean, f2Ratio),cellfun(@ste, f2Ratio),{'10','100'},1);

%% create summary plots for splitCentering protocol with APB drugs
clc; CloseAllFiguresExceptGUI;
load('/Users/chrischen/Dropbox/research/projects/spatialIntegration/summary/splitFieldCenteringDrug.mat');
scTable=struct2table(splitCenterDrugSummary);
scTable.recType=categorical(scTable.recType);  scTable.cellType=categorical(scTable.cellType);  scTable.pattern=categorical(scTable.pattern);
% rectification index for exc
drugTable=scTable(scTable.recType=='exc' & scTable.pattern=='split-field'  ,{'DrugSensPerc', 'meanLum','cellType'});
cellTypes=unique(drugTable.cellType);
figure; ax=axes; hold all;
for c=2:numel(cellTypes)
    subTable=drugTable(drugTable.cellType==cellTypes(c),:);
    %     ax=categoryBoxplot(subTable.DrugSensPerc,findgroups(subTable.meanLum),{'10','100','1000'},[cellTypes(c) ' '  'LY sensitive perc']);
    [G,gName] = findgroups(subTable.meanLum);
    cellMean=splitapply(@mean,subTable.DrugSensPerc,G);
    cellErr=splitapply(@(x) std(x)/sqrt(numel(x)),subTable.DrugSensPerc,G);
    scatterWithMeanAndError(G,subTable.DrugSensPerc*100,cellMean*100,cellErr*100,string(gName),0);
end


%% analyze the AII and bipolar data
cellTypeSplit = @(listSorted)splitOnCellType(listSorted);
cellTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, cellTypeSplit);

dateSplit = @(listSorted)splitOnExperimentDate(listSorted);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, dateSplit);

protocolSplit = @(listSorted)splitOnShortProtocolID(listSorted);
ProtocolSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, protocolSplit);

ndfSplit = @(listSorted)splitOnNDFs(listSorted);
ndfSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, ndfSplit);

brightnessSplit = @(listSorted)splitOnDeviceBrightNess(listSorted);
brightnessSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, brightnessSplit);

recordingTypeSplit = @(listSorted)splitOnRecordingType(listSorted);
recordingTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, recordingTypeSplit);

tree = riekesuite.analysis.buildTree(listSorted,{cellTypeSplit_java,dateSplit_java, 'cell.label','protocolSettings(epochGroup:label)',...
    ProtocolSplit_java, brightnessSplit_java,ndfSplit_java,recordingTypeSplit_java});
gui = epochTreeGUI(tree);

%% variable Mean Noise Protocol

cellTypeSplit = @(listSorted)splitOnCellType(listSorted);
cellTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, cellTypeSplit);

dateSplit = @(listSorted)splitOnExperimentDate(listSorted);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, dateSplit);

ndfSplit = @(listSorted)splitOnNDFs(listSorted);
ndfSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, ndfSplit);

tree = riekesuite.analysis.buildTree(listSorted,{cellTypeSplit_java,dateSplit_java, 'cell.label',ndfSplit_java});
gui = epochTreeGUI(tree);

%% set up parameters and analyze the noise protocol for one long LED epochs
clear paras tpStats lgds
clc; CloseAllFiguresExceptGUI;
paras.psthSigma=10;
selectedNodes = gui.getSelectedEpochTreeNodes;
paras.sampleRate=selectedNodes{1}.epochList.firstValue.protocolSettings('sampleRate');
paras.spikeTh=1.2;
paras.spikeTag=1;
paras.baseRange=[0.001 0.05]; % seconds
paras.preTime=0; paras.tailTime=0;
paras.plotTrace=1;
paras.downsample=10;
%%%%%%%%%%%%%% compute some basic infos.
paras.led=selectedNodes{1}.epochList.firstValue.protocolSettings('led'); paras.led=regexprep(paras.led,' ','_');
paras.frequencyCutoff=selectedNodes{1}.epochList.firstValue.protocolSettings('frequencyCutoff');
paras.numberOfFilters=selectedNodes{1}.epochList.firstValue.protocolSettings('numberOfFilters');
paras.lowerLimit=selectedNodes{1}.epochList.firstValue.protocolSettings(['stimulus:' paras.led,':lowerLimit']);
paras.upperLimit=selectedNodes{1}.epochList.firstValue.protocolSettings(['stimulus:' paras.led,':upperLimit'])+3.3;
paras.contrast=selectedNodes{1}.epochList.firstValue.protocolSettings('Contrast');
paras.stdv=selectedNodes{1}.epochList.firstValue.protocolSettings('stdv');
paras.lMean=selectedNodes{1}.epochList.firstValue.protocolSettings('lightMean');
paras.stimTime=selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');
for node=1:numel(selectedNodes)
    [stats] = analyzeLongSingleEpochNoise(selectedNodes{node},paras);
    tpStats{node}=stats;
    lgds{node}=['FW' num2str(selectedNodes{node}.epochList.firstValue.protocolSettings('background:FilterWheel:NDF'))];
end
if numel(selectedNodes)>1
    colors=pmkmp(numel(selectedNodes),'IsoL');
    figure('position',[50 50 600 800]);
    subplot(2,1,1); hold all;
    for node=1:numel(selectedNodes)
        plot(tpStats{node}.lnModel.filterTimeStamps,tpStats{node}.lnModel.filter,'color',colors(node,:),'linewidth',3);
    end
    legend(lgds); legend boxoff; title('Filter');
    subplot(2,1,2); hold all;
    for node=1:numel(selectedNodes)
        scatter(tpStats{node}.normLNModel.nlX,tpStats{node}.normLNModel.nlY,20,colors(node,:),'filled');
    end
    legend(lgds); legend boxoff; title('Filter');
end

