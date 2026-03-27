% analysisSupplementary.m - Supplementary analyses (centering + noise)
%   Paper reference: Methods (RF centering), Supp. Fig 5 (LN model)
%   Requires: main.m to be run first (sets up listSorted, gui, summaryFolder)
%   External dependency: LNNodeModelWrapper / SigmoidNlNode
%     (see https://github.com/chrischen2/cascadeGraph)
%
%   Part 1: Split-field centering protocol for RF center determination.
%   Part 2: AII amacrine / bipolar cell noise analysis with LN models.

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

%% ===== Part 2: Noise Analysis (Supp. Fig 5) =====
% NOTE: Requires a different GUI. Run the noise GUI section in main.m first.

%% Variable Mean Noise Protocol GUI
cellTypeSplit = @(listSorted)splitOnCellType(listSorted);
cellTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, cellTypeSplit);

dateSplit = @(listSorted)splitOnExperimentDate(listSorted);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, dateSplit);

ndfSplit = @(listSorted)splitOnNDFs(listSorted);
ndfSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, ndfSplit);

tree = riekesuite.analysis.buildTree(listSorted,{cellTypeSplit_java,dateSplit_java, 'cell.label',ndfSplit_java});
gui = epochTreeGUI(tree);

%% Analyze noise protocol for long LED epochs
clear paras tpStats lgds
clc; CloseAllFiguresExceptGUI;
paras.psthSigma=10;
selectedNodes = gui.getSelectedEpochTreeNodes;
paras.sampleRate=selectedNodes{1}.epochList.firstValue.protocolSettings('sampleRate');
paras.spikeTh=1.2;
paras.spikeTag=1;
paras.baseRange=[0.001 0.05];
paras.preTime=0; paras.tailTime=0;
paras.plotTrace=1;
paras.downsample=10;
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
