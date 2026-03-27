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
list = loader.loadEpochList([ovaExportFolder 'LinearEqvDiscChris.mat'], dataFolder);

for i = 1:list.length
    list.elements(i).setProtocolSetting('user:startDate',datestr((list.elements(i).startDate)'));
end
listSorted = list.sortedBy('protocolSettings(user:startDate)'); % list sorted chronologically

%% expanding spot split data and create GUI  

cellTypeSplit = @(listSorted)splitOnCellType(listSorted);
cellTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, cellTypeSplit);

dateSplit = @(listSorted)splitOnExperimentDate(listSorted);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, dateSplit);

recordingTypeSplit = @(listSorted)splitOnRecordingType(listSorted);
recordingTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, recordingTypeSplit);

ndfSplit = @(listSorted)splitOnNDFs(listSorted);
ndfSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, ndfSplit);

brightnessSplit = @(listSorted)splitOnDeviceBrightNess(listSorted);
brightnessSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, brightnessSplit);

tree = riekesuite.analysis.buildTree(listSorted,{cellTypeSplit_java,dateSplit_java, 'cell.label','protocolSettings(epochGroup:label)',...
    brightnessSplit_java, ndfSplit_java, recordingTypeSplit_java});
gui = epochTreeGUI(tree);

%% expanding spots analyzing
clc; 
CloseAllFiguresExceptGUI;
paras.psthSigma=20;
paras.spikeTh=1.2;
paras.sampleRate=1e4;
paras.spikeTag=1;
selectedNodes = gui.getSelectedEpochTreeNodes;
stimTime=selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');
preTime=selectedNodes{1}.epochList.firstValue.protocolSettings('preTime');
timeToPts=@(x) x/1e3*paras.sampleRate;
paras.stimPts=timeToPts(stimTime);
paras.prePts=timeToPts(preTime);
paras.backgroundIntensity=selectedNodes{1}.epochList.firstValue.protocolSettings('backgroundIntensity');
% selectedInd=getSelectedIndex(selectedNodes{1}.epochList);
[ax,output,onlineAnalysis]=analyzeExpandingSpots(selectedNodes{1},paras);

%% save cell info for expanding spots population summary 
clc; clear expSpotSummary;
onlineAnalysis
meanLuminance=input ('enter the mean luminance:');
cellType=shortCellType(selectedNodes{1}.epochList.firstValue.protocolSettings('source:type'));
recType=onlineAnalysis;
% recType='exc';
expDate=datestr(selectedNodes{1}.epochList.elements(1).startDate','yyyy/mm/dd'); 
cellLabel=selectedNodes{1}.epochList.firstValue.protocolSettings('source:label');
numCells=0; 
try
    load('/Users/chrischen/research/projects/spatialIntegration/summary/expandingSpots.mat');
    numCells=numel(expSpotSummary);
end
expSpotSummary(numCells+1)=struct('date',expDate,'cellID',cellLabel,'cellType', cellType,'recType',recType,'spotList',output.spotList, ...
    'normRes',output.normRes,'sigmaC',output.model.sigmaC,'sigmaS',output.model.sigmaS,'Kc',output.model.Kc,'kS',output.model.Ks,'meanLum',meanLuminance);
save('/Users/chrischen/research/projects/spatialIntegration/summary/expandingSpots.mat','expSpotSummary');
fprintf('%s \n', '---new cell data saved---');

%% population level summary analysis of expanding spots 
clc; clear sumTable
load('/Users/chrischen/research/projects/spatialIntegration/summary/expandingSpots.mat');
sumTable=struct2table(expSpotSummary);
sumTable.recType=categorical(sumTable.recType); sumTable.cellType=categorical(sumTable.cellType);
CloseAllFiguresExceptGUI;
cellTypes=unique(sumTable.cellType);
f=figure('color','w','position',[50 100 1800 900]);
for c=1:numel(cellTypes)
    ax(c)=subplot(2,ceil(numel(cellTypes)/2),c);   hold all;
    try
        spotRes=sumTable(sumTable.cellType==cellTypes(c) & sumTable.recType=='exc'& sumTable.meanLum==100,{'spotList','normRes'});
        for i=1:size(spotRes,1)
            plot(cell2mat(spotRes.spotList(i)), cell2mat(spotRes.normRes(i)),'linewidth',2,'color','k');
        end
        % legend(cellstr(num2str((1:size(spotRes,1))', 'trial %-d')),'fontsize',15); legend boxoff;
        % compute the mean and error for each bar size
        [G,spotID{c}] = findgroups(cat(2,spotRes.spotList{:}));
        spotMean.(string(cellTypes(c)))=splitapply(@mean,cat(2,spotRes.normRes{:}),G);
        spotErr.(string(cellTypes(c)))=splitapply(@(x) std(x)/sqrt(numel(x)),cat(2,spotRes.normRes{:}),G);
        errorbar(spotID{c}, spotMean.(string(cellTypes(c))),spotErr.(string(cellTypes(c))),'r','linewidth',3);
        % plot(barID, barMean.(string(cellTypes(c))),'r','linewidth',3);
        title(cellTypes(c) ); xlabel('spot size');  ylabel('Norm Response'); setAxes(f);
        hold off;
    end
end

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
paras.wcoffset=500;
% selectedInd=getSelectedIndex(selectedNodes{1}.epochList);
[ax,contrastArray,resArray,onlineAnalysis]=analyzeContrastSpots(selectedNodes{1},paras);

%% save cell info for contrast spots population summary 
clc; clear expSpotSummary;
onlineAnalysis
meanLuminance=input ('enter the mean luminance:'); 
cellType=shortCellType(selectedNodes{1}.epochList.firstValue.protocolSettings('source:type'));
recType=onlineAnalysis;
expDate=datestr(selectedNodes{1}.epochList.elements(1).startDate','yyyy/mm/dd'); 
cellLabel=selectedNodes{1}.epochList.firstValue.protocolSettings('source:label');
numCells=0; 
try
    load('/Users/chrischen/research/projects/spatialIntegration/summary/expandingSpots.mat');
    numCells=numel(contrastSpotSummary);
end
contrastSpotSummary(numCells+1)=struct('date',expDate,'cellID',cellLabel,'cellType', cellType,'recType',recType,...
    'contrastList',contrastArray, 'resList',resArray, 'meanLum',meanLuminance);
save('/Users/chrischen/research/projects/spatialIntegration/summary/contrastSpots.mat','contrastSpotSummary');
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
paras.spikeTh=1.2;
clc; CloseAllFiguresExceptGUI;
paras.spikeTag=0;
paras.psthSigma=10;

stimTime=selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');
preTime=selectedNodes{1}.epochList.firstValue.protocolSettings('preTime');
paras.sampleRate=selectedNodes{1}.epochList.firstValue.protocolSettings('sampleRate');
paras.tempFreq=selectedNodes{1}.epochList.firstValue.protocolSettings('temporalFrequency');
paras.meanIntensity=selectedNodes{1}.epochList.firstValue.protocolSettings('backgroundIntensity');
timeToPts=@(x) x/1e3*paras.sampleRate;
paras.prePts=timeToPts(preTime);
paras.stimPts=timeToPts(stimTime);
[ax,output,onlineAnalysis]=analyzeContrastReversingGrating(selectedNodes, paras);
fprintf('%s , %f \n', 'temporal freq::',paras.tempFreq);
%% save particular cell for population summary 
clc; clear CRGSummary;
onlineAnalysis
meanLuminance=input ('enter the mean luminance:');
cellType=shortCellType(selectedNodes{1}.epochList.firstValue.protocolSettings('source:type'));
recType=onlineAnalysis;
numCells=0; 
try
    load('/Users/chrischen/research/projects/spatialIntegration/summary/contrastReversingGrating.mat');
    numCells=numel(CRGSummary);
end
CRGSummary(numCells+1)=struct('cellID',numCells+1,'cellType', cellType,'recType',recType,'barList',output.barList, ...
    'F2',output.F2,'suppression',output.suppress,'subUnitSize', output.subUnitSize,'meanLum',meanLuminance);
save('/Users/chrischen/research/projects/spatialIntegration/summary/contrastReversingGrating.mat','CRGSummary');
fprintf('%s \n', '---new cell data saved---');
%% population level summary analysis of contrast reversing gratings 
clc; clear barID
load('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/contrastReversingGrating.mat');
sumTable=struct2table(CRGSummary);
sumTable.recType=categorical(sumTable.recType); sumTable.cellType=categorical(sumTable.cellType);
% average subunit size of given luminance given certain cell type 
subunit=varfun( @(x) mean(x), sumTable, 'GroupingVariables', {'cellType','meanLum','recType'},...
    'InputVariables',{'subUnitSize','suppression'},'outputformat','table');


figure('color','w','position',[200 200 900 900]); 
cellTypes=unique(sumTable.cellType);
for c=1:numel(cellTypes)
    f2Table=sumTable(sumTable.cellType==cellTypes(c) & sumTable.recType=='extracellular'& sumTable.meanLum==100,{'barList','F2'});
    subplot(2,2,c); hold all;
    for i=1:size(f2Table,1)
        plot(cell2mat(f2Table.barList(i)), cell2mat(f2Table.F2(i)),'color','k','linewidth',0.5);
    end
    % compute the mean and error for each bar size
    [G,barID{c}] = findgroups(cat(1,f2Table.barList{:}));
    barMean.(string(cellTypes(c)))=splitapply(@mean,cat(1,f2Table.F2{:}),G);
    barErr.(string(cellTypes(c)))=splitapply(@(x) std(x)/sqrt(numel(x)),cat(1,f2Table.F2{:}),G);
    errorbar(barID{c}, barMean.(string(cellTypes(c))),barErr.(string(cellTypes(c))),'r','linewidth',3);
    title(cellTypes(c));
end

figure('color','w','position',[1200 300 700 700]);  hold all;
ccolors='krbg';
for c=1:numel(cellTypes)
    scalor=max(barMean.(string(cellTypes(c))));
    errorbar(barID{c}, barMean.(string(cellTypes(c)))/scalor,barErr.(string(cellTypes(c)))/scalor,'color',ccolors(c),'linewidth',3);
end
legend(cellTypes); legend boxoff; ylim([0 1]);

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
paras.spikeTh=1.2;
clc; CloseAllFiguresExceptGUI;
paras.spikeTag=0;
paras.psthSigma=20;
paras.rmreps=[ ];
preTime=selectedNodes{1}.epochList.firstValue.protocolSettings('preTime');
stimTime=selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');
tailTime=selectedNodes{1}.epochList.firstValue.protocolSettings('tailTime');
paras.sampleRate=selectedNodes{1}.epochList.firstValue.protocolSettings('sampleRate');
paras.backgroundIntensity=selectedNodes{1}.epochList.firstValue.protocolSettings('backgroundIntensity');
paras.onlineAnalysis=selectedNodes{1}.epochList.firstValue.protocolSettings('onlineAnalysis');
paras.wcoffset=700;  % pts
paras.spikeoffset=400;  % pts
timeToPts=@(x) x/1e3*paras.sampleRate;
paras.prePts=timeToPts(preTime); paras.stimPts=timeToPts(stimTime); paras.tailPts=timeToPts(tailTime);
[output]=analyzeLinearDisc(selectedNodes,paras); 

%% save cell information for population analysis 
clc; 
output.cellType='OffS';
% meanLuminance=input ('enter the mean luminance:'); 
meanLuminance=100;
cellType=shortCellType(selectedNodes{1}.epochList.firstValue.protocolSettings('source:type'));
expDate=datestr(selectedNodes{1}.epochList.elements(1).startDate','yyyy/mm/dd'); 
cellLabel=selectedNodes{1}.epochList.firstValue.protocolSettings('source:label');
numCells=0; 
try
    load('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/linearEqvDiscNew.mat');
    numCells=numel(linearDiscSummary);
end
linearDiscSummary(numCells+1)=struct('date',expDate,'cellID',cellLabel,'cellType', output.cellType,'onlineAnalysis',paras.onlineAnalysis,'meanOnsetNLI',mean(output.NLI.onset), ...
    'meanOffsetNLI',mean(output.NLI.offset),'OnsetNLI',output.NLI.onset, ...
    'OffsetNLI',output.NLI.offset,'meanLum',meanLuminance,'respOnset',[mean(output.stats.image.onset.mean) ...
    mean(output.stats.disc.onset.mean)], 'respOffset',[mean(output.stats.image.offset.mean) ...
    mean(output.stats.disc.offset.mean)]);
save('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/linearEqvDiscNew.mat','linearDiscSummary');
fprintf('%s \n', '---new cell data saved---');

%% population level summary analysis of linear disc
clc;
recType='inh';
load('/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/summary/linearEqvDiscOld.mat');
discTable=struct2table(linearDiscSummary);
discTable.onlineAnalysis=categorical(discTable.onlineAnalysis); discTable.cellType=categorical(discTable.cellType);
% Compute meanOnsetNLI and meanOffsetNLI on the fly from arrays
% and apply sanity check for offsetNLI based on respOffset values
for i = 1:height(discTable)
    % Compute meanOnsetNLI on the fly
    if ~isempty(discTable.OnsetNLI{i})
        discTable.meanOnsetNLI(i) = mean(discTable.OnsetNLI{i});
    else
        discTable.meanOnsetNLI(i) = NaN;
    end
    
    % Apply sanity check for offsetNLI based on respOffset values
    % if ~isempty(discTable.OffsetNLI{i})
    %     % Check if respOffset values are < 3 for extracellular recordings
    %     if strcmp(char(discTable.onlineAnalysis(i)), 'exc') && ...
    %        all(discTable.respOffset(i,:) < 5)  % offsetResp is mx2 matrix
    %         discTable.meanOffsetNLI(i) = 0;
    %         discTable.OffsetNLI{i} = zeros(size(discTable.OffsetNLI{i})); % Set all offsetNLI to 0
    %     else
    %         discTable.meanOffsetNLI(i) = mean(discTable.OffsetNLI{i});
    %     end
    % else
    %     discTable.meanOffsetNLI(i) = NaN;
    % end
end

CloseAllFiguresExceptGUI;
cellTypes=categorical({'OffS','OffT'});
% Filter the table to only include the specified cell types
subTable=discTable(discTable.onlineAnalysis==recType & discTable.meanLum==100 & ismember(discTable.cellType, cellTypes),{'cellType','meanOnsetNLI','meanOffsetNLI'});
% ax1=categoryBoxplot(subTable.meanOnsetNLI,findgroups(subTable.cellType),unique(subTable.cellType),'meanOnset NLI');
% ax2=categoryBoxplot(subTable.meanOffsetNLI,findgroups(subTable.cellType),unique(subTable.cellType),'meanOffset NLI');

% First, calculate mean and error values for each cell type
cellTypes = unique(subTable.cellType);
groups = findgroups(subTable.cellType);

% Calculate mean values for each group
meanOnsetNLI = splitapply(@mean, subTable.meanOnsetNLI, groups);
meanOffsetNLI = splitapply(@mean, subTable.meanOffsetNLI, groups);

% Calculate standard error for each group
stdErrOnsetNLI = splitapply(@(x) std(x)/sqrt(length(x)), subTable.meanOnsetNLI, groups);
stdErrOffsetNLI = splitapply(@(x) std(x)/sqrt(length(x)), subTable.meanOffsetNLI, groups);

% Create the figures
figure;
subplot(1,2,1);
ax1 = scatterWithMeanAndError(groups, subTable.meanOnsetNLI, meanOnsetNLI, stdErrOnsetNLI, cellTypes, true);
title('Mean Onset NLI');
ylabel('NLI');

subplot(1,2,2);
ax2 = scatterWithMeanAndError(groups, subTable.meanOffsetNLI, meanOffsetNLI, stdErrOffsetNLI, cellTypes, true);
title('Mean Offset NLI');
ylabel('NLI');

cellTypeToInsp='OffT';
cellTable=discTable(discTable.cellType==cellTypeToInsp & discTable.meanLum==100,{'onlineAnalysis','meanOnsetNLI','meanOffsetNLI'});
ax3=categoryBoxplot(cellTable.meanOnsetNLI,findgroups(cellTable.onlineAnalysis),unique(cellTable.onlineAnalysis),[ cellTypeToInsp ' Onset']);
ax4=categoryBoxplot(cellTable.meanOffsetNLI,findgroups(cellTable.onlineAnalysis),unique(cellTable.onlineAnalysis),[cellTypeToInsp ' Offset']);

% create cumulative plots for Nonlinearity Index for various cell types
% cellTypes=unique(discTable.cellType);
cellTypes=categorical({'OffS','OffT'});
f1=figure('color','w','position',[100 300 700 700]);  hold all;
for c=1:numel(cellTypes)
    cumRes=discTable(discTable.cellType==cellTypes(c)&discTable.onlineAnalysis==recType&...
        discTable.meanLum==100,{'cellType','OnsetNLI','OffsetNLI'});
    cp=cdfplot(cat(2,cumRes.OnsetNLI{:})); set(cp,'linewidth',2);
end
grid off;  title('All fixation histogram');  legend(cellTypes,'location','SE'); legend boxoff;
initFig(gca(f1),'Onset NLI','cumulative fraction'); setAxes(f1);  xlim([-1 1]);

f2=figure('color','w','position',[100 300 700 700]);  hold all;
for c=1:numel(cellTypes)
    try
        cumRes=discTable(discTable.cellType==cellTypes(c)&discTable.onlineAnalysis==recType& discTable.meanLum==100,{'cellType','OnsetNLI','OffsetNLI'});
        cp=cdfplot(cat(2,cumRes.OffsetNLI{:})); set(cp,'linewidth',2);
    end
end
grid off;  title('All fixation histogram');  legend(cellTypes,'location','SE'); legend boxoff;
initFig(gca(f2),'Offset NLI','cumulative fraction'); setAxes(f2);  xlim([-1 1]);

% plot the luminance dependency of offT cell
lumTable=discTable(discTable.cellType=='OffT'& discTable.onlineAnalysis=='extracellular',...
    {'date','cellID','OnsetNLI','OffsetNLI','meanOnsetNLI','meanOffsetNLI','meanLum'});
meanLumList=unique(lumTable.meanLum);
f3=figure('color','w','position',[100 300 700 700]);  hold all;
for m=1:numel(meanLumList)
    cumRes=lumTable(lumTable.meanLum==meanLumList(m),{'OnsetNLI','OffsetNLI'});
    cp=cdfplot(cat(2,cumRes.OnsetNLI{:})); set(cp,'linewidth',2);
end
grid off;  title('All fixation histogram');  legend(split(num2str(meanLumList')),'location','SE'); legend boxoff;
initFig(gca(f3),'Onset NLI','cumulative fraction'); setAxes(f3);  xlim([-1 1]);

f4=figure('color','w','position',[100 300 700 700]);  hold all;
for m=1:numel(meanLumList)
    cumRes=lumTable(lumTable.meanLum==meanLumList(m),{'OnsetNLI','OffsetNLI'});
    cp=cdfplot(cat(2,cumRes.OffsetNLI{:})); set(cp,'linewidth',2);
end
grid off;  title('All fixation histogram');  legend(split(num2str(meanLumList')),'location','SE'); legend boxoff;
initFig(gca(f4),'Offset NLI','cumulative fraction'); setAxes(f4); xlim([-1 1]);

% cell mean
[G,ID]=findgroups(lumTable(:,{'date', 'cellID'})); gList=unique(G); 
f5=figure('color','w','position',[100 300 700 700]);  hold all;  
for i=1: numel(gList)
    rows=find(G==gList(i));
    try
        plot(log(lumTable.meanLum(rows)),lumTable.meanOnsetNLI(rows),'-ko','markersize',15);  
    end
end
set(gca,'xtick', log(meanLumList),'xticklabels',split(num2str(meanLumList'))); xlabel('Mean luminance'); ylabel('Onset NLI'); 
 
%% analyze flashed gratings  
clc; selectedNodes = gui.getSelectedEpochTreeNodes;
CloseAllFiguresExceptGUI;
paras.spikeTh=1.2;
paras.spikeTag=0;
paras.psthSigma=10;
paras.wcoffset=1500;  % pts
preTime=selectedNodes{1}.epochList.firstValue.protocolSettings('preTime');
stimTime=selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');  
tailTime=selectedNodes{1}.epochList.firstValue.protocolSettings('tailTime');
paras.sampleRate=selectedNodes{1}.epochList.firstValue.protocolSettings('sampleRate');
timeToPts=@(x) x/1e3*paras.sampleRate;
paras.prePts=timeToPts(preTime); paras.stimPts=timeToPts(stimTime); paras.tailPts=timeToPts(tailTime);
[f,stats,onlineAnalysis]=analyzeFlashGrating(selectedNodes,paras); ax=gca(f(1));

%% save cell data for population analysis, for flashed grating 
clc; clear flashGSummary;
onlineAnalysis
meanLuminance=input ('enter the mean luminance:');
cellType=shortCellType(selectedNodes{1}.epochList.firstValue.protocolSettings('source:type'));
recType=onlineAnalysis;
expDate=datestr(selectedNodes{1}.epochList.elements(1).startDate','yyyy/mm/dd'); 
cellLabel=selectedNodes{1}.epochList.firstValue.protocolSettings('source:label');
numCells=0; 
try
    load('/Users/chrischen/research/projects/spatialIntegration/summary/flashedGrating.mat');
    numCells=numel(flashGSummary);
end
flashGSummary(numCells+1)=struct('date',expDate,'cellID',cellLabel,'cellType', cellType,'recType',recType,'barList',stats.barList, ...
    'OnsetResponse',stats.onset,'OffsetResponse',stats.offset,'baselineResponse', stats.baseline,'meanLum',meanLuminance);
save('/Users/chrischen/research/projects/spatialIntegration/summary/flashedGrating.mat','flashGSummary');
fprintf('%s \n', '---new cell data saved---');

%% population level summary analysis of flashed gratings 
clc;
load('/Users/chrischen/research/projects/spatialIntegration/summary/flashedGrating.mat');
sumTable=struct2table(flashGSummary);
sumTable.recType=categorical(sumTable.recType); sumTable.cellType=categorical(sumTable.cellType);
CloseAllFiguresExceptGUI;
figure('color','w','position',[200 200 900 900]); 
cellTypes=unique(sumTable.cellType);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
recTypeInsp='extracellular';
lightLevelInsp=100; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for c=1:numel(cellTypes)
    resTable=sumTable(sumTable.cellType==cellTypes(c) & sumTable.recType==recTypeInsp& ...
        sumTable.meanLum==lightLevelInsp,{'cellType','barList','baselineResponse','OnsetResponse','OffsetResponse'});
    try
        subplot(2,2,c); hold all;
        for i=1:size(resTable,1)
            plot(cell2mat(resTable.barList(i)), cell2mat(resTable.OnsetResponse(i)),'color','k','linewidth',0.5);
        end
        %compute the mean and error for each bar size
        [G,barID{c}] = findgroups(cat(2,resTable.barList{:}));
        barMean.(string(cellTypes(c)))=splitapply(@mean,cat(2,resTable.OnsetResponse{:}),G);
        barErr.(string(cellTypes(c)))=splitapply(@(x) std(x)/sqrt(numel(x)),cat(2,resTable.OnsetResponse{:}),G);
        errorbar(barID{c}, barMean.(string(cellTypes(c))),barErr.(string(cellTypes(c))),'r','linewidth',3);
        title(cellTypes(c));
    end
    st=sgtitle('Onset'); set(st,'fontsize',26);
end

figure('color','w','position',[200 200 900 900]); 
cellTypes=unique(sumTable.cellType);
for c=1:numel(cellTypes)
    resTable=sumTable(sumTable.cellType==cellTypes(c) & sumTable.recType==recTypeInsp & ...
        sumTable.meanLum==lightLevelInsp,{'cellType','barList','baselineResponse','OnsetResponse','OffsetResponse'});
    try
        subplot(2,2,c); hold all;
        for i=1:size(resTable,1)
            plot(cell2mat(resTable.barList(i)), cell2mat(resTable.OffsetResponse(i)),'color','k','linewidth',0.5);
        end
        %     % compute the mean and error for each bar size
        [G,barID{c}] = findgroups(cat(2,resTable.barList{:}));
        barMean.(string(cellTypes(c)))=splitapply(@mean,cat(2,resTable.OffsetResponse{:}),G);
        barErr.(string(cellTypes(c)))=splitapply(@(x) std(x)/sqrt(numel(x)),cat(2,resTable.OffsetResponse{:}),G);
        errorbar(barID{c}, barMean.(string(cellTypes(c))),barErr.(string(cellTypes(c))),'r','linewidth',3);
        title(cellTypes(c));
    end
    st=sgtitle('Offset'); set(st,'fontsize',26);
end

f=figure('color','w','position',[300 300 600 600]);
barToInspect=80; cellTypeInsp='OffT';
resTable=sumTable(sumTable.cellType==cellTypeInsp & sumTable.recType==recTypeInsp & ...
    sumTable.meanLum==lightLevelInsp,{'cellType','barList','baselineResponse','OnsetResponse','OffsetResponse'});
barLists=cat(2,resTable.barList{:}); res.onset=cat(2,resTable.OnsetResponse{:}); res.offset=cat(2,resTable.OffsetResponse{:});
res.onset=res.onset(barLists==barToInspect);   res.offset=res.offset(barLists==barToInspect);
hold all;
scatterWithError(mean(res.onset),mean(res.offset),std(res.onset)/sqrt(numel(res.onset)-1),std(res.offset)/sqrt(numel(res.offset)-1),1);
scatter(res.onset,res.offset,100,'k','filled'); axis equal;
xline(0); yline(0); hold off; setAxes(f); initFig(gca(f),'Onset','Offset')
 
%compute I/E ratio for onset and offset across bar sizes
IETable=sumTable(sumTable.cellType=='OffT' & sumTable.recType~='extracellular' & ...
    sumTable.meanLum==100,{'date','cellID','recType','barList','OnsetResponse','OffsetResponse'});
[G,ID]=findgroups(IETable(:,{'date', 'cellID'})); gList=unique(G);
count=0; barInsp=80;
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
%% analyze split field centering protocol
selectedNodes = gui.getSelectedEpochTreeNodes;
paras.spikeTh=1.2;
CloseAllFiguresExceptGUI;
paras.spikeTag=0;
paras.psthSigma=10;
paras.wcOffset=0;
stimTime=selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');
preTime=selectedNodes{1}.epochList.firstValue.protocolSettings('preTime');
tailTime=selectedNodes{1}.epochList.firstValue.protocolSettings('tailTime');
paras.sampleRate=selectedNodes{1}.epochList.firstValue.protocolSettings('sampleRate');
paras.tempFreq=selectedNodes{1}.epochList.firstValue.protocolSettings('temporalFrequency');
paras.meanIntensity=selectedNodes{1}.epochList.firstValue.protocolSettings('backgroundIntensity');
timeToPts=@(x) x/1e3*paras.sampleRate;
paras.prePts=timeToPts(preTime);
paras.stimPts=timeToPts(stimTime);
[ax,output, recType]=analyzeSplitFieldCentering(selectedNodes, paras);
output.rectIndex
output.modulation.positive
%% save stats for population analysis/ drug exepriments 
clc; clear splitCenterSummary;
recType
meanLuminance=input ('enter the mean luminance:');
drugUsed=input ('enter the drug used:','s');
cellType=shortCellType(selectedNodes{1}.epochList.firstValue.protocolSettings('source:type'));
expDate=datestr(selectedNodes{1}.epochList.elements(1).startDate','yyyy/mm/dd'); 
cellLabel=selectedNodes{1}.epochList.firstValue.protocolSettings('source:label');
pattern=selectedNodes{1}.parent.parent.parent.splitValue;
numCells=0; 
% compute the drug sensitive percentage 
drugSensPerc=-(output.modulation.positive(2)-output.modulation.positive(1))/output.modulation.positive(1)
try
    load('/Users/chrischen/research/projects/spatialIntegration/summary/splitFieldCenteringDrug.mat');
    numCells=numel(splitCenterDrugSummary);
end
splitCenterDrugSummary(numCells+1)=struct('date',expDate,'cellID',cellLabel,'cellType', cellType,'recType',recType,'responseAmp',output.modulation.positive, ...
    'rectIndex',output.rectIndex,'drugUsed',drugUsed,'meanLum',meanLuminance,'DrugSensPerc',drugSensPerc,'pattern',pattern);
save('/Users/chrischen/research/projects/spatialIntegration/summary/splitFieldCenteringDrug.mat','splitCenterDrugSummary');
fprintf('%s \n', '---new cell data saved---');

%% create summary plots for splitCentering protocol
load('/Users/chrischen/research/projects/spatialIntegration/summary/splitFieldCenteringDrug.mat');
scTable=struct2table(splitCenterDrugSummary);
scTable.recType=categorical(scTable.recType);  scTable.cellType=categorical(scTable.cellType);  scTable.pattern=categorical(scTable.pattern); 
% rectification index for exc 
drugTable=scTable(scTable.recType=='exc' & scTable.pattern=='full-field'  ,{'DrugSensPerc', 'meanLum','cellType'});
CloseAllFiguresExceptGUI;
cellTypes=unique(drugTable.cellType); 
for c=1:numel(cellTypes)
    subTable=drugTable(drugTable.cellType==cellTypes(c),:);
    ax=categoryBoxplot(subTable.DrugSensPerc,findgroups(subTable.meanLum),{'10','100','1000'},[cellTypes(c) ' '  'LY sensitive perc']);
end