clearvars; close all; clc;
% define plot color sequence, axis fonts
import auimodel.*
import vuidocument.*
loader = edu.washington.rieke.Analysis.getEntityLoader();
treeFactory = edu.washington.rieke.Analysis.getEpochTreeFactory();
listFactory = edu.washington.rieke.Analysis.getEpochListFactory();
newList=listFactory.create;
ovaExportFolder='/Users/chrischen/research/projects/spatialIntegration/';
dataFolder='/Users/chrischen/research/projects/spatialIntegration/';
list = loader.loadEpochList([ovaExportFolder 'noiseCorrelationPair.mat'], dataFolder);
for i = 1:list.length
    list.elements(i).setProtocolSetting('user:startDate',datestr((list.elements(i).startDate)'));
end
listSorted = list.sortedBy('protocolSettings(user:startDate)'); % list sorted chronologically


%% split the data and analyze
dateSplit = @(listSorted)splitOnExperimentDate(listSorted);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, dateSplit);


protocolSplit = @(listSorted)splitOnShortProtocolID(listSorted);
ProtocolSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, protocolSplit);


keywordSplit = @(listSorted)splitOnKeywords(listSorted);
keywordSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, keywordSplit);

tree = riekesuite.analysis.buildTree(listSorted,{dateSplit_java,'cell.label',ProtocolSplit_java, keywordSplit_java, ...
    'protocolSettings(lightMean)'});
gui = epochTreeGUI(tree);

%% compute the signal+noise and correlation
clc; CloseAllFiguresExceptGUI;
selectedNodes = gui.getSelectedEpochTreeNodes;
paras.sampleRate=selectedNodes{1}.epochList.firstValue.protocolSettings('sampleRate');
paras.stimTime=selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');
paras.preTime=0; paras.tailTime=0;
paras.frequencyCutoff=selectedNodes{1}.epochList.firstValue.protocolSettings('frequencyCutoff');
paras.numberOfFilters=selectedNodes{1}.epochList.firstValue.protocolSettings('numberOfFilters');
led=regexprep(selectedNodes{1}.epochList.firstValue.protocolSettings('led'),' ','_');
paras.lowerLimit=selectedNodes{1}.epochList.firstValue.protocolSettings(['stimulus:' led,':lowerLimit']);
paras.upperLimit=selectedNodes{1}.epochList.firstValue.protocolSettings(['stimulus:' led,':upperLimit']);
paras.contrast=selectedNodes{1}.epochList.firstValue.protocolSettings('Contrast');

meanRes.c1=zeros(numel(selectedNodes),size(resMat.c1,2));
meanRes.c2=zeros(numel(selectedNodes),size(resMat.c2,2));

for node=1:numel(selectedNodes)
    resMat.c1=riekesuite.getResponseMatrix(selectedNodes{node}.epochList,'Amp1');
    resMat.c2=riekesuite.getResponseMatrix(selectedNodes{node}.epochList,'Amp2');
    for i=1:size(resMat.c1,1)
        resMat.c1(i,:)=smooth(resMat.c1(i,:),50);
        resMat.c2(i,:)=smooth(resMat.c2(i,:),50);
        %adjust the baseline to remove slow drifting
        resMat.c1(i,:)=resMat.c1(i,:)-mean(resMat.c1(i,1:50));
        resMat.c2(i,:)=resMat.c2(i,:)-mean(resMat.c2(i,1:50));
    end
    
    meanRes.c1(node,:)=mean(resMat.c1);
    meanRes.c2(node,:)=mean(resMat.c2);
    residual.c1=resMat.c1-repmat(mean(resMat.c1),size(resMat.c1,1),1);
    residual.c2=resMat.c2-repmat(mean(resMat.c2),size(resMat.c2,1),1);
    %regenerate the noise inputs
    noiseSeeds=selectedNodes{node}.epochList.firstValue.protocolSettings('seed');
    lightMean=selectedNodes{node}.epochList.firstValue.protocolSettings(['stimulus:' led,':mean']);
    lightStDev=selectedNodes{node}.epochList.firstValue.protocolSettings(['stimulus:' led,':stDev']);
    stimulus= createGaussianNoiseStimulus(paras,lightMean,lightStDev,noiseSeeds);
    figure('position',[200 200  1400 600]);
    subplot(3,1,1);
    plot(stimulus,'r'); axis off;
    subplot(3,1,2);
    plot(mean(resMat.c1),'k'); axis off;
    subplot(3,1,3);
    plot(mean(resMat.c2),'k'); axis off;
    sgtitle(['meanLightLevel::',num2str(lightMean)]);
    
    figure('position',[200 200  1400 600]);
    expTrial=randi(size(residual.c1,1));
    plot(residual.c1(expTrial,:),'r'); hold all;  plot(residual.c2(expTrial,:),'k'); axis off; box off;
    sgtitle(['meanLightLevel::',num2str(lightMean)]);
    
    % compute and plot the noise correlation
    figure('position',[200 200  800 600]);  hold all;
    range=[-0.5 0.5]*paras.sampleRate+size(residual.c1,2);
    for i=1:size(residual.c1)
        [c(i,:),lags]=xcov(residual.c1(i,:), residual.c2(i,:),'coeff');
        plot(lags((range(1):range(2)))*1e3/paras.sampleRate,c(i,range(1):range(2)),'k','linewidth',0.3);
    end
    plot(lags((range(1):range(2)))*1e3/paras.sampleRate,mean(c(:,range(1):range(2))),'r','linewidth',4);
    xlabel('Time (ms)');  ylabel('correlation');
    % as sanity check, shuffle the trials and compute the noise correlaiton
    shuffle.c1=residual.c1(randperm(size(residual.c1,1)),:);
    shuffle.c2=residual.c2(randperm(size(residual.c2,1)),:);
    sgtitle(['meanLightLevel::',num2str(lightMean)]);
    
    figure('position',[1000 200  800 600]);  hold all;
    for i=1:size(residual.c1)
        [cf(i,:),lags]=xcov(shuffle.c1(i,:), shuffle.c2(i,:),'coeff');
        plot(lags((range(1):range(2)))*1e3/paras.sampleRate,cf(i,range(1):range(2)),'k','linewidth',0.3);
    end
    plot(lags((range(1):range(2)))*1e3/paras.sampleRate,mean(cf(:,range(1):range(2))),'r','linewidth',4);
    xlabel('Time (ms)');  ylabel('correlation');
    sgtitle(['meanLightLevel::',num2str(lightMean)]);
end

figure('position',[200 200  1400 600]);
for node=1:numel(selectedNodes)
    subplot(numel(selectedNodes),1,node);
    plot(meanRes.c1(node,:)-mean(meanRes.c1(node,1:50)),'k'); hold all;
    plot(meanRes.c2(node,:)-mean(meanRes.c2(node,1:50)),'r');
end
