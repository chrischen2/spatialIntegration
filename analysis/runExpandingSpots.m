% runExpandingSpots.m - Expanding spot analysis
%   Paper reference: Supplementary Figure 3A (RF center size, DoG model)
%   Requires: main.m to be run first (sets up listSorted, summaryFolder)
%
%   Analyzes expanding spot responses to measure RF center size using a
%   Difference-of-Gaussians model. Computes population summaries across
%   cell types and light levels.

%% Create GUI for expanding spots
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

%% Analyze expanding spots
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
[ax,output,onlineAnalysis]=analyzeExpandingSpots(selectedNodes,paras);
output.minRes

%% Save cell info for expanding spots population summary
clc; clear expSpotSummary;
onlineAnalysis
meanLuminance=input ('enter the mean luminance:');
cellType=shortCellType(selectedNodes{1}.epochList.firstValue.protocolSettings('source:type'));
recType=onlineAnalysis;
expDate=datestr(selectedNodes{1}.epochList.elements(1).startDate','yyyy/mm/dd');
cellLabel=selectedNodes{1}.epochList.firstValue.protocolSettings('source:label');
numCells=0;
try
    load(fullfile(summaryFolder, 'expandingSpots.mat'));
    numCells=numel(expSpotSummary);
end
expSpotSummary(numCells+1)=struct('date',expDate,'cellID',cellLabel,'cellType', cellType,'recType',recType,'spotList',output.spotList, ...
    'normRes',output.normRes,'sigmaC',output.model.sigmaC,'sigmaS',output.model.sigmaS,'Kc',output.model.Kc,'kS',output.model.Ks,'meanLum',meanLuminance);
save(fullfile(summaryFolder, 'expandingSpots.mat'),'expSpotSummary');
fprintf('%s \n', '---new cell data saved---');

%% Population level summary analysis of expanding spots
clc; clear sumTable spotMean spotErr spotID
load(fullfile(summaryFolder, 'expandingSpots.mat'));
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
        [G,spotID{c}] = findgroups(cat(2,spotRes.spotList{:}));
        spotMean.(char(cellTypes(c)))=splitapply(@mean,cat(2,spotRes.normRes{:}),G);
        spotErr.(char(cellTypes(c)))=splitapply(@(x) std(x)/sqrt(numel(x)),cat(2,spotRes.normRes{:}),G);
        errorbar(spotID{c}, spotMean.(char(cellTypes(c))),spotErr.(char(cellTypes(c))),'r','linewidth',3);
        plot(spotID, spotMean.(char(cellTypes(c))),'r','linewidth',3);
        title(char(cellTypes(c))); xlabel('spot size');  ylabel('Norm Response'); setAxes(f);
        hold off;
    end
end

% Overlay of cell types
f=figure('color','w','position',[50 100 600 600]); hold all;
for c=1:numel(cellTypes)
    try
        spotRes=sumTable(sumTable.cellType==cellTypes(c) & sumTable.recType=='extracellular'& sumTable.meanLum==100,{'spotList','normRes','sigmaC'});
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

% Cell mean RF size
rfTable=sumTable( sumTable.recType=='extracellular'& sumTable.meanLum==100,{'cellType','sigmaC'});
[G,gName]=findgroups(rfTable(:,{'cellType'})); gList=unique(G);
cellMean=splitapply(@mean,rfTable.sigmaC,G);
cellErr=splitapply(@(x) std(x)/sqrt(numel(x)),rfTable.sigmaC,G);
figure; ax=scatterWithMeanAndError(G,rfTable.sigmaC,cellMean,cellErr,cellstr(gName.cellType),1);

% A2/AC RF size across light levels
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
