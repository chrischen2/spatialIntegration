% runPPGratings.m - Paired-pulse grating analysis
%   Paper reference: Figure 7I-N (paired-pulse gratings, drug)
%   Requires: main.m to be run first (sets up listSorted, summaryFolder)
%
%   Analyzes paired-pulse grating experiments with variable mean luminance
%   and variable pulse intervals. Computes facilitation ratios across
%   contrast levels.

%% Create GUI for gratings with variable means / intervals
cellTypeSplit = @(listSorted)splitOnCellType(listSorted);
cellTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, cellTypeSplit);

dateSplit = @(listSorted)splitOnExperimentDate(listSorted);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, dateSplit);

tree = riekesuite.analysis.buildTree(listSorted,{cellTypeSplit_java, dateSplit_java, 'cell.label','protocolSettings(grateContrast)', ...
    'protocolSettings(psth)'});
gui = epochTreeGUI(tree);

%% Analyze grating with variable mean
clc;
clear ppGratingsMean
paras.saveCell=0;
CloseAllFiguresExceptGUI;
paras.psthSigma=10;
paras.spikeTh=1.2;
paras.sampleRate=1e4;
selectedNodes = gui.getSelectedEpochTreeNodes;
stimTime=selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');
preTime=selectedNodes{1}.epochList.firstValue.protocolSettings('preTime');
tailTime=selectedNodes{1}.epochList.firstValue.protocolSettings('tailTime');
flashDuration=selectedNodes{1}.epochList.firstValue.protocolSettings('grateDuration');
paras.flashContrast=selectedNodes{1}.epochList.firstValue.protocolSettings('grateContrast');
pulseIntervals=selectedNodes{1}.epochList.firstValue.protocolSettings('pulseIntervals');
paras.psth=selectedNodes{1}.epochList.firstValue.protocolSettings('psth');
timeToPts=@(x) x/1e3*paras.sampleRate;
paras.stimPts=timeToPts(stimTime);
paras.prePts=timeToPts(preTime);
paras.tailPts=timeToPts(tailTime);
paras.flashPts=timeToPts(flashDuration);
paras.intervalPts=timeToPts(pulseIntervals);
stats=analyzePPGratingsMean(selectedNodes{1},paras);

% Save stats for population analysis
if paras.saveCell
    numCells=0;
    try
        load(fullfile(summaryFolder, 'ppGratingsMean.mat'));
        numCells=numel(ppGratingsMean);
    end
    ppGratingsMean(numCells+1)=struct('contrastArray',stats.contrastArray, 'ratio1',stats.ratio1, 'ratio2',stats.ratio2, 'amp1', stats.amp1, ...
        'amp2', stats.amp2, 'amp3', stats.amp3,'amp4', stats.amp4);
    save(fullfile(summaryFolder, 'ppGratingsMean.mat'),'ppGratingsMean');
end

%% Plot summary for PP gratings mean
load(fullfile(summaryFolder, 'ppGratingsMeanSpike.mat'),'ppGratingsMean');

for i=1:size(ppGratingsMean,2)
    tp1(i,:)=ppGratingsMean(i).amp2(1:2) ;
    tp2(i,:)=ppGratingsMean(i).amp4(1:2);
end
figure; hold all;
errorbar(ppGratingsMean(1).contrastArray(1:2)*100, mean(tp1), std(tp1)/sqrt(size(tp1,1)));
errorbar(ppGratingsMean(1).contrastArray(1:2)*100, mean(tp2), std(tp2)/sqrt(size(tp2,1)));
initFig(gca,'step Contrast', 'amplitude');

%% Analyze grating with variable intervals
clc;
clear ppGratingsInterval
paras.saveCell=0;
CloseAllFiguresExceptGUI;
paras.psthSigma=10;
paras.spikeTh=1.2;
paras.sampleRate=1e4;
selectedNodes = gui.getSelectedEpochTreeNodes;
stimTime=selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');
preTime=selectedNodes{1}.epochList.firstValue.protocolSettings('preTime');
tailTime=selectedNodes{1}.epochList.firstValue.protocolSettings('tailTime');
flashDuration=selectedNodes{1}.epochList.firstValue.protocolSettings('grateDuration');
paras.flashContrast=selectedNodes{1}.epochList.firstValue.protocolSettings('grateContrast');
paras.stepContrast=selectedNodes{1}.epochList.firstValue.protocolSettings('stepContrast');
paras.psth=selectedNodes{1}.epochList.firstValue.protocolSettings('psth');
timeToPts=@(x) x/1e3*paras.sampleRate;
paras.stimPts=timeToPts(stimTime);
paras.prePts=timeToPts(preTime);
paras.tailPts=timeToPts(tailTime);
paras.flashPts=timeToPts(flashDuration);
stats=analyzePPGratingsInterval(selectedNodes{1},paras);

% Save stats for population analysis
if paras.saveCell
    numCells=0;
    try
        load(fullfile(summaryFolder, 'ppGratingsInterval.mat'));
        numCells=numel(ppGratingsInterval);
    end
    ppGratingsInterval(numCells+1)=struct('contrastArray',stats.contrastArray, 'ratio1',stats.ratio1, 'ratio2',stats.ratio2, 'amp1', stats.amp1, ...
        'amp2', stats.amp2, 'amp3', stats.amp3,'amp4', stats.amp4);
    save(fullfile(summaryFolder, 'ppGratingsInterval.mat'),'ppGratingsInterval');
end
