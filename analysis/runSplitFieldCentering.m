% runSplitFieldCentering.m - Split-field centering analysis
%   Paper reference: Methods (RF centering procedure)
%   Requires: main.m to be run first (sets up listSorted, summaryFolder)
%
%   Analyzes split-field centering protocol to determine RF center position.
%   Includes population summaries and drug experiment analysis.

%% Create GUI for split centering
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

%% Analyze split field centering protocol
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

%% Save stats for population analysis
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
    load(fullfile(summaryFolder, 'splitFieldCentering.mat'));
    numCells=numel(splitCenterSummary);
end
splitCenterSummary(numCells+1)=struct('date',expDate,'cellID',cellLabel,'cellType', cellType,'recType',output.recType,'responseAmp',output.modulation.positive, ...
    'rectIndex',output.rectIndex,'meanLum',meanLuminance,'pattern',pattern);
save(fullfile(summaryFolder, 'splitFieldCentering.mat'),'splitCenterSummary');
fprintf('%s \n', '---new cell data saved---');

%% Save stats for population drug experiments analysis
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
drugSensPerc=-(output.modulation.positive(2)-output.modulation.positive(1))/output.modulation.positive(1)
try
    load(fullfile(summaryFolder, 'splitFieldCentering.mat'));
    numCells=numel(splitCenterDrugSummary);
end
splitCenterDrugSummary(numCells+1)=struct('date',expDate,'cellID',cellLabel,'cellType', cellType,'recType',output.recType,'responseAmp',output.modulation.positive, ...
    'rectIndex',output.rectIndex,'drugUsed',drugUsed,'meanLum',meanLuminance,'DrugSensPerc',drugSensPerc,'pattern',pattern);
save(fullfile(summaryFolder, 'splitFieldCenteringDrug.mat'),'splitCenterSummary');
fprintf('%s \n', '---new cell data saved---');

%% Create summary plots for splitCentering protocol without drugs
clc; CloseAllFiguresExceptGUI; clear f2Ratio grps scTable;
load(fullfile(summaryFolder, 'splitFieldCentering.mat'));
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

%% Create summary plots for splitCentering protocol with APB drugs
clc; CloseAllFiguresExceptGUI;
load(fullfile(summaryFolder, 'splitFieldCenteringDrug.mat'));
scTable=struct2table(splitCenterDrugSummary);
scTable.recType=categorical(scTable.recType);  scTable.cellType=categorical(scTable.cellType);  scTable.pattern=categorical(scTable.pattern);
drugTable=scTable(scTable.recType=='exc' & scTable.pattern=='split-field'  ,{'DrugSensPerc', 'meanLum','cellType'});
cellTypes=unique(drugTable.cellType);
figure; ax=axes; hold all;
for c=2:numel(cellTypes)
    subTable=drugTable(drugTable.cellType==cellTypes(c),:);
    [G,gName] = findgroups(subTable.meanLum);
    cellMean=splitapply(@mean,subTable.DrugSensPerc,G);
    cellErr=splitapply(@(x) std(x)/sqrt(numel(x)),subTable.DrugSensPerc,G);
    scatterWithMeanAndError(G,subTable.DrugSensPerc*100,cellMean*100,cellErr*100,string(gName),0);
end
