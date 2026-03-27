% runNoiseAnalysis.m - Noise analysis and LN model identification
%   Paper reference: Supplementary Figure 5 (LN model identification)
%   Requires: main.m to be run first (sets up listSorted, summaryFolder)
%   External dependency: LNNodeModelWrapper / SigmoidNlNode
%     (see https://github.com/chrischen2/cascadeGraph)
%
%   Analyzes AII amacrine / bipolar cell noise protocols and computes
%   LN model filters and nonlinearities across light levels.

%% Create GUI for AII and bipolar data
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
