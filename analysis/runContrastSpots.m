% runContrastSpots.m - Contrast spot analysis
%   Paper reference: Figure 6A-B (contrast response functions)
%   Requires: main.m to be run first (sets up listSorted, summaryFolder)
%
%   Analyzes contrast response functions across cell types and computes
%   I/E ratios at different light levels.

%% Create GUI for contrast spots
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

%% Analyze contrast spots
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
output=analyzeContrastSpots(selectedNodes,paras);
allFilters{end+1} = output{1}.temporalFilter;

%% Save cell info for contrast spots population summary
clc; clear expSpotSummary;
onlineAnalysis=selectedNodes{1}.parent.parent.splitValue
meanLuminance=input ('enter the mean luminance:');
cellType=shortCellType(selectedNodes{1}.epochList.firstValue.protocolSettings('source:type'));
expDate=datestr(selectedNodes{1}.epochList.elements(1).startDate','yyyy/mm/dd');
cellLabel=selectedNodes{1}.epochList.firstValue.protocolSettings('source:label');
numCells=0;
try
    load(fullfile(summaryFolder, 'contrastSpots.mat'));
    numCells=numel(contrastSpotSummary);
end
contrastSpotSummary(numCells+1)=struct('date',expDate,'cellID',cellLabel,'cellType', cellType,'onlineAnalysis',onlineAnalysis,...
    'contrastList',contrastArray, 'resList',resArray, 'meanLum',meanLuminance);
save(fullfile(summaryFolder, 'contrastSpots.mat'),'contrastSpotSummary');
fprintf('%s \n', '---new cell data saved---');

%% Load all cell data for contrast spots population summary
clc; clear sumTable
load(fullfile(summaryFolder, 'contrastSpots.mat'));
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
    hold all;
    try
        spotRes=sumTable(sumTable.cellType==cellTypes(c) & sumTable.onlineAnalysis==recInsp & sumTable.meanLum==meanLum,{'contrastList','resList'});
        for i=1:size(spotRes,1)
            plot(cell2mat(spotRes.contrastList(i)), cell2mat(spotRes.resList(i)),'linewidth',2);
        end
        [G,spotID{c}] = findgroups(cat(2,spotRes.contrastList{:}));
        spotMean.(string(cellTypes(c)))=splitapply(@mean,cat(2,spotRes.resList{:}),G);
        spotErr.(string(cellTypes(c)))=splitapply(@(x) std(x)/sqrt(numel(x)),cat(2,spotRes.resList{:}),G);
        errorbar(spotID{c}, spotMean.(string(cellTypes(c))),spotErr.(string(cellTypes(c))),'r','linewidth',3);
        if strcmp(recInsp,'exc')
            resFit=fitCRF_sigmoid(spotID{c},spotMean.(string(cellTypes(c))),[-0.5 0,1,0]);
        else
            resFit=fitCRF_sigmoid(spotID{c},spotMean.(string(cellTypes(c))),[0.5 0,1,0]);
        end
        title(cellTypes(c) ); xlabel('contrast');  ylabel('Int Response'); setAxes(f);
        hold off;
        contrasts=spotID{c}; contrastResponses=spotMean.(string(cellTypes(c)));
        save(fullfile(summaryFolder, 'contrastFunctions', [char(cellTypes(c)) recInsp '.mat']), ...
            'contrasts', 'contrastResponses');
    end
end

%% Save CRS I/E ratio
clc; clear IERatio;
meanLuminance=input ('enter the mean luminance:');
numCells=0;
try
    load(fullfile(summaryFolder, 'IERatio.mat'));
    numCells=numel(IERatioSummary);
end
IERatioSummary(numCells+1)=struct('IERatio',output{1}.ieRatio, 'meanLum',meanLuminance);
save(fullfile(summaryFolder, 'IERatio.mat'),'IERatioSummary');
fprintf('%s \n', '---new cell data saved---');
