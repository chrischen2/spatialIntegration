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
list = loader.loadEpochList([ovaExportFolder 'spatialIntegrationAllData.mat'], dataFolder);

for i = 1:list.length
    list.elements(i).setProtocolSetting('user:startDate',datestr((list.elements(i).startDate)'));
end
listSorted = list.sortedBy('protocolSettings(user:startDate)'); % list sorted chronologically

%% split data and create GUI
dateSplit = @(listSorted)splitOnExperimentDate(listSorted);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, dateSplit);

cellTypeSplit = @(listSorted)splitOnCellType(listSorted);
cellTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, cellTypeSplit);

recordingTypeSplit = @(listSorted)splitOnRecordingType(listSorted);
recordingTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, recordingTypeSplit);

protocolSplit = @(listSorted)splitOnShortProtocolID(listSorted);
ProtocolSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, protocolSplit);

keywordSplit = @(listSorted)splitOnKeywords(listSorted);
keywordSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, keywordSplit);

tree = riekesuite.analysis.buildTree(listSorted,{cellTypeSplit_java,dateSplit_java, 'cell.label',ProtocolSplit_java, ...
    keywordSplit_java,'protocolSettings(background:Microdisplay_Stage@localhost:microdisplayBrightness)'});
gui = epochTreeGUI(tree);

%% expanding spots analyzing
clc; 
CloseAllFiguresExceptGUI;
psthSigma=20;
spikeTh=1.5;
sampleRate=1e4;
savePlot=0;
selectedNodes = gui.getSelectedEpochTreeNodes;
% selectedInd=getSelectedIndex(selectedNodes{1}.epochList);
[f,spotSizes,spotRes]=analyzeExpandingSpots(selectedNodes{1},psthSigma,sampleRate,spikeTh);
if savePlot
    exportFigToPDF([dataFolder 'RF fitting'],f(1),300);
    exportFigToPDF([dataFolder 'step PSTH'],f(2),300);
end
fprintf('%s %s\n', 'cell type is ',selectedNodes{1}.epochList.firstValue.protocolSettings.get('source:type'))
%% main part
clear spikeTimes resMat paras response stats meanTrace imageIndex contrastList tfH
selectedNodes = gui.getSelectedEpochTreeNodes;
paras.spikeTh=1.5;
CloseAllFiguresExceptGUI;
paras.spikeTag=0;
paras.psthSigma=25;
paras.showIndividual=0;

stimTime=selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');
preTime=selectedNodes{1}.epochList.firstValue.protocolSettings('preTime');
paras.sampleRate=selectedNodes{1}.epochList.firstValue.protocolSettings('sampleRate');
timeToPts=@(x) x/1e3*paras.sampleRate;
resMat=riekesuite.getResponseMatrix(selectedNodes{1}.epochList,'Amp1');
% flat the baseline for better spike detection
paras.epochRange=1:size(resMat,1);
% paras.epochRange=181:360;
resMat=resMat(paras.epochRange,:);

for i=1:size(resMat,1)
    if strcmp(selectedNodes{1}.parent.splitValue, '[spike]') || paras.spikeTag
        resMat(i,:)=resMat(i,:)-movmedian(resMat(i,:),100);
    else
        resMat(i,:)=smooth(resMat(i,:),100);
    end
end

if strcmp(selectedNodes{1}.parent.splitValue, '[spike]') || paras.spikeTag
    [spikeTimes,~,~,emptyTrial]=SpikeDetectorNew(resMat, 'thresholdSpikeFactor', paras.spikeTh);
    psth=spikeTimeToPSTH(resMat, spikeTimes, paras.psthSigma, paras.sampleRate);
end
paras.maxIntensity=selectedNodes{1}.epochList.firstValue.protocolSettings ...
    ('epoch:Microdisplay_Stage@localhost:white:rodConversionFactor');
fprintf('%s , %f \n', 'max Luminance::', paras.maxIntensity');

imageIndex=zeros(size(resMat,1),1);
stimTag=zeros(size(resMat,1),1);
response.onset=zeros(size(resMat,1),1);
response.offset=zeros(size(resMat,1),1);
contrast=zeros(size(resMat,1),1);
for i=1:size(resMat,1)
    imageIndex(i)=selectedNodes{1}.epochList.elements(paras.epochRange(i)).protocolSettings('imagePatchIndex') ...
        +str2double(selectedNodes{1}.epochList.elements(paras.epochRange(i)).protocolSettings('imageName'))*100;
    tpTag=selectedNodes{1}.epochList.elements(paras.epochRange(i)).protocolSettings('stimulusTag');
    switch tpTag
        case 'image'
            stimTag(i)=1;
        case 'intensity'
            stimTag(i)=2;
    end
    if strcmp(selectedNodes{1}.parent.splitValue, '[spike]') || paras.spikeTag
        response.onset(i)=length(spikeTimes{i}(spikeTimes{i}>timeToPts(preTime) & spikeTimes{i}<timeToPts(preTime+stimTime)));
        response.offset(i)=length(spikeTimes{i}(spikeTimes{i}>timeToPts(preTime+stimTime)));
        %         response.onset(i)= response.onset(i)-length(spikeTimes{i}(spikeTimes{i}<timeToPts(preTime)))*stimTime/preTime;
    else
        response.onset(i)=sum(resMat(i,timeToPts(preTime):timeToPts(preTime+stimTime))-mean2(resMat(:,1:timeToPts(preTime))))/1e4;
        response.offset(i)=sum(resMat(i,timeToPts(preTime+stimTime):end)-mean2(resMat(:,1:timeToPts(preTime))))/1e4;
    end
    % compute the equv contrast
    contrast(i)=selectedNodes{1}.epochList.elements(paras.epochRange(i)).protocolSettings('equivalentIntensity')/ ...
        selectedNodes{1}.epochList.elements(paras.epochRange(i)).protocolSettings('backgroundIntensity') -1;
end

uniquePatches=unique(imageIndex);
repCount=0; rmInd=[];
for i=1:numel(uniquePatches)
    if length(find(imageIndex==uniquePatches(i)))>=4
        repCount=repCount+1;
        stats.image.onset.mean(repCount)=mean(response.onset((imageIndex==uniquePatches(i) & stimTag==1)));
        stats.disc.onset.mean(repCount)=mean(response.onset((imageIndex==uniquePatches(i) & stimTag==2)));
        stats.image.onset.ste(repCount)=ste(response.onset((imageIndex==uniquePatches(i) & stimTag==1)));
        stats.disc.onset.ste(repCount)=ste(response.onset((imageIndex==uniquePatches(i) & stimTag==2)));
        stats.image.offset.mean(repCount)=mean(response.offset((imageIndex==uniquePatches(i) & stimTag==1)));
        stats.disc.offset.mean(repCount)=mean(response.offset((imageIndex==uniquePatches(i) & stimTag==2)));
        stats.image.offset.ste(repCount)=ste(response.offset((imageIndex==uniquePatches(i) & stimTag==1)));
        stats.disc.offset.ste(repCount)=ste(response.offset((imageIndex==uniquePatches(i) & stimTag==2)));
        contrastList(repCount)=median(unique(contrast(imageIndex==uniquePatches(i) & stimTag==1)));
    else
        rmInd=[rmInd i];
    end
end
uniquePatches(rmInd)=[];
%%%% compute distance to unity line
distances=pointToLineDistance([stats.image.onset.mean' stats.disc.onset.mean'], [0 0],[1 1]);
maxInd=find(distances==max(distances)); minInd=find(distances==min(distances)); maxInd=maxInd(1); minInd=minInd(1);
f1=figure('position',[200 200 1600 600],'color','w');
subplot(1,2,1); hold all;
scatterWithError(stats.image.onset.mean,stats.disc.onset.mean,stats.image.onset.ste,stats.disc.onset.ste,1 );
scatter(stats.image.onset.mean(maxInd), stats.disc.onset.mean(maxInd),250,'r','filled');
scatter(stats.image.onset.mean(minInd), stats.disc.onset.mean(minInd),250,'g','filled');
initFig(gca(f1),'response to image', 'response to disc'); setAxes(f1); title('Onset');
subplot(1,2,2);
scatterWithError(stats.image.offset.mean,stats.disc.offset.mean,stats.image.offset.ste,stats.disc.offset.ste,1);
initFig(gca(f1),'response to image', 'response to disc'); setAxes(f1); title('Offset');
scatter(stats.image.offset.mean(maxInd), stats.disc.offset.mean(maxInd),250,'r','filled');
scatter(stats.image.offset.mean(minInd), stats.disc.offset.mean(minInd),250,'g','filled');

tf=figure('position',[200 200 800 700]); hold all;
scatter(contrastList,stats.image.onset.mean,120,'r');
[xMid.image,meanY.image] = averageInBins(contrastList,stats.image.onset.mean,10);
scatter(contrastList,stats.disc.onset.mean,120,'g');
[xMid.disc,meanY.disc] = averageInBins(contrastList,stats.disc.onset.mean,10);
tfH(1)=plot(xMid.image, meanY.image,'r','linewidth',2);
tfH(2)=plot(xMid.disc, meanY.disc,'g','linewidth',2);
% overlay the contrast spot data if exists 
if ~isempty(selectedNodes{1}.parent.parent.parent.childBySplitValue('ContrastResponseSpots')) && exist('spotContrastList','var') == 1  ...
    &&  strcmp(selectedNodes{1}.parent.splitValue, '[spike]') 
    for i=1:numel(condNames)
        tfH(i+2)=plot(spotContrastList,meanResp.(condNames{i}),'linewidth',3,'color',newcolors(i,:));
    end
    legend(tfH, cat(2,{'image','disc'}, condNames));
else
    legend(tfH, {'image','disc'});
end
initFig(gca(tf),'contrast', 'response'); setAxes(tf); title('contrast tuning Onset');


%%% plot the example trace for whole cell recordings
if ~(strcmp(selectedNodes{1}.parent.splitValue, '[spike]') || paras.spikeTag)
    meanTrace.image.max=mean(resMat(imageIndex==uniquePatches(maxInd(1)) & stimTag==1,:));
    meanTrace.image.min=mean(resMat(imageIndex==uniquePatches(minInd(1)) & stimTag==1,:));
    meanTrace.disc.max=mean(resMat(imageIndex==uniquePatches(maxInd(1)) & stimTag==2,:));
    meanTrace.disc.min=mean(resMat(imageIndex==uniquePatches(minInd(1)) & stimTag==2,:));
else
    meanTrace.image.max=mean(psth(imageIndex==uniquePatches(maxInd(1)) & stimTag==1,:));
    meanTrace.image.min=mean(psth(imageIndex==uniquePatches(minInd(1)) & stimTag==1,:));
    meanTrace.disc.max=mean(psth(imageIndex==uniquePatches(maxInd(1)) & stimTag==2,:));
    meanTrace.disc.min=mean(psth(imageIndex==uniquePatches(minInd(1)) & stimTag==2,:));
end
f2=figure('position',[200 200 1600 600],'color','w');
subplot(1,2,1); hold all;
plot(meanTrace.image.max,'g','linewidth',2);
plot(meanTrace.disc.max,'r','linewidth',2);
legend({'image','disc'}); legend boxoff;
subplot(1,2,2); hold all;
plot(meanTrace.image.min,'g','linewidth',2);
plot(meanTrace.disc.min,'r','linewidth',2);
legend({'image','disc'}); legend boxoff; 
setAxes(f2); title('example trace');

% visualize the examplary patches
patchInd.max= find(uniquePatches(maxInd)==imageIndex); patchInd.max=patchInd.max(1);
patchInd.min= find(uniquePatches(minInd)==imageIndex); patchInd.min=patchInd.min(1);
imageName.max=selectedNodes{1}.epochList.elements(patchInd.max).protocolSettings('imageName');
imageName.min=selectedNodes{1}.epochList.elements(patchInd.min).protocolSettings('imageName');
patchLoc.max=convertJavaArrayListMatrix(selectedNodes{1}.epochList.elements(patchInd.max).protocolSettings('currentPatchLocation'));
patchLoc.min=convertJavaArrayListMatrix(selectedNodes{1}.epochList.elements(patchInd.min).protocolSettings('currentPatchLocation'));
RFsigma=selectedNodes{1}.epochList.elements(patchInd.max).protocolSettings('rfSigmaCenter');
apertureDiameter=selectedNodes{1}.epochList.elements(patchInd.max).protocolSettings('apertureDiameter');
[eContrast.max,imagePatch.max ] = loadNatPatch(imageName.max,patchLoc.max, RFsigma,apertureDiameter);
[eContrast.min,imagePatch.min ] = loadNatPatch(imageName.min,patchLoc.min, RFsigma,apertureDiameter);
f2=figure('position',[200 200 800 300]);
subplot(1,2,1); hold all;
imagesc(imagePatch.max);  colormap gray; title('furthest to unity line');
s=size(imagePatch.max);     rectangle('position',[0 0 s(2) s(1)],'edgecolor','r','linewidth',5);
subplot(1,2,2);  hold all;
imagesc(imagePatch.min); colormap gray; title('closest to unity line');
s=size(imagePatch.min);     rectangle('position',[0 0 s(2) s(1)],'edgecolor','g','linewidth',5);

% visualize the mean traces pooled together 
poolID=find(distances>median(distances)); 
traceIDs.image=find(ismember(imageIndex, uniquePatches(poolID)) & stimTag==1);
traceIDs.disc=find(ismember(imageIndex, uniquePatches(poolID)) & stimTag==2);
pooledTrace.image=mean(resMat(traceIDs.image,:));
pooledTrace.disc=mean(resMat(traceIDs.disc,:));
f3=figure('position',[200 200 1000 500],'color','w'); hold all;
plot(pooledTrace.image,'r');
plot(pooledTrace.disc,'g');
legend({'image','disc'}); legend boxoff;
setAxes(f2); title('example trace');
title('Average across patches,furthest half');

% NLI.onset=(stats.image.onset.mean - stats.disc.onset.mean) ./ (stats.image.onset.mean +stats.disc.onset.mean);
% NLI.offset=mean(stats.image.offset.mean - stats.disc.offset.mean) ./ (stats.image.offset.mean +stats.disc.offset.mean);
NLI.onset=normDistance(stats.disc.onset.mean,stats.image.onset.mean);
NLI.offset=normDistance(stats.disc.offset.mean,stats.image.offset.mean);
f4=figure('position',[200 200 1000 500]); hold all;
histData{1}=NLI.onset; histData{2}=NLI.offset;
multiHist(histData',{'Onset','Offset'},20,'xlabels','NLI','ylabels','PDF'); setAxes(f4);


meanWave.image=zeros(numel(uniquePatches),size(resMat,2)); meanWave.disc=meanWave.image;
for i=1:numel(uniquePatches)
    if ~(strcmp(selectedNodes{1}.parent.splitValue, '[spike]') || paras.spikeTag)
        meanWave.image(i,:)=mean(resMat(imageIndex==uniquePatches(i) & stimTag==1,:));
        meanWave.disc(i,:)=mean(resMat(imageIndex==uniquePatches(i) & stimTag==2,:));
    else
        meanWave.image(i,:)=mean(resMat(imageIndex==uniquePatches(i) & stimTag==1,:));
        meanWave.disc(i,:)=mean(resMat(imageIndex==uniquePatches(i) & stimTag==2,:));
    end
end
if paras.showIndividual
    f5=figure('position',[200 200 1000 500]); 
    for i=1:numel(uniquePatches)
        plot(meanWave.image(i,:),'k'); hold all; plot(meanWave.disc(i,:),'r'); legend('image','disc'); legend boxoff;
        setAxes(f5); 
        pause(1); clf(f5);
    end   
end

fprintf('%s %1.2f, %s %1.2f \n', 'MeanNLI Onset::', mean(NLI.onset), '::OffSet::', mean(NLI.offset));
fprintf('%s %1.2f, %s %1.2f \n', 'Median NLI Onset::', median(NLI.onset), '::OffSet::', median(NLI.offset));

%% run this to export plots
cellName= [regexprep(selectedNodes{1}.parent.parent.parent.parent.parent.splitValue,'/','-') '-'  ...
    selectedNodes{1}.parent.parent.parent.splitValue '-' ...
    regexprep(selectedNodes{1}.parent.parent.parent.parent.splitValue,'\','-') '-' ...
    selectedNodes{1}.splitValue '-' selectedNodes{1}.parent.splitValue];
exportFigToPDF([cellName '-scatter-plot'], f1, 300);
exportFigToPDF([cellName '-example-plot'], f2, 300);
exportFigToPDF([cellName '-averagedTraces-plot'], f3, 300);
%% analyze contrast reversing gratings
clear spikeTimes resMat response stats
selectedNodes = gui.getSelectedEpochTreeNodes;
paras.spikeTh=1.5;
CloseAllFiguresExceptGUI;
paras.spikeTag=0;
paras.psthSigma=20;

stimTime=selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');
preTime=selectedNodes{1}.epochList.firstValue.protocolSettings('preTime');
sampleRate=selectedNodes{1}.epochList.firstValue.protocolSettings('sampleRate');
tempFreq=selectedNodes{1}.epochList.firstValue.protocolSettings('temporalFrequency');
timeToPts=@(x) x/1e3*sampleRate;
resMat=riekesuite.getResponseMatrix(selectedNodes{1}.epochList,'Amp1');
paras.epochRange=1:size(resMat,1);
% paras.epochRange=104:141;
resMat=resMat(paras.epochRange,:);

barWidths=zeros(size(resMat,1),1);
for i=1:size(resMat,1)
    if strcmp(selectedNodes{1}.parent.splitValue, '[spike]') || paras.spikeTag 
        resMat(i,:)=resMat(i,:)-movmedian(resMat(i,:),100);
    else
        resMat(i,:)=smooth(resMat(i,:),100);
    end
    barWidths(i)=selectedNodes{1}.epochList.elements(paras.epochRange(i)).protocolSettings('currentBarWidth');
end

if strcmp(selectedNodes{1}.parent.splitValue, '[spike]') || paras.spikeTag 
    resMat=spikeBinary(resMat, paras.spikeTh);
else
    resMat=resMat-repmat(mean(resMat(:,1:timeToPts(preTime)),2),1,size(resMat,2));
end

uniqueBarWidth=unique(barWidths);
F1=zeros(numel(uniqueBarWidth),1); F2=zeros(numel(uniqueBarWidth),1);
meanRes=zeros(numel(uniqueBarWidth),size(resMat,2));

if strcmp(selectedNodes{1}.parent.splitValue, '[spike]')  || paras.spikeTag 
    psth=spikeBinaryToPSTH(resMat, paras.psthSigma, sampleRate);
end

for i=1:numel(uniqueBarWidth)
    barInd=find(barWidths==uniqueBarWidth(i));
    
    if strcmp(selectedNodes{1}.parent.splitValue, '[spike]')  || paras.spikeTag 
        tp=sum(psth(barInd,:),1)/numel(barInd);
        
    else
        tp=sum(resMat(barInd,:),1)/numel(barInd);
    end
    meanRes(i,:)=tp;
    F2(i)=sum(tp(timeToPts(preTime)+1:timeToPts(preTime+stimTime)))/sampleRate;
    [F1(i), F2(i)]=computeF1F2(tp,sampleRate,tempFreq);
    
end
f3=figure('position',[300 500 1000 500],'color','w');
subplot(1,2,2);
h2 = line(uniqueBarWidth, F2/max(F2));
set(h2,'Color','k','LineWidth',2,'Marker','o','markersize',10);
setAxes(f3); initFig(gca(f3),'Bar width (um)','Integrated Response');

% create examplary plot for whole cell data

subplot(1,2,1);
hold all;
for i=1:numel(uniqueBarWidth)
    if strcmp(selectedNodes{1}.parent.splitValue, '[spike]')  || paras.spikeTag 
        plotOffset=100;
    elseif strcmp(selectedNodes{1}.parent.splitValue, '[ext]')
        plotOffset=200;
    else
        plotOffset=300;
    end
    plot(meanRes(i,:)+plotOffset*(i-1));
    tx=text(size(resMat,2)/8,i*plotOffset+mean2(meanRes),['barWidth::',num2str(uniqueBarWidth(i))]);
    set(tx,'fontsize',15,'color','k');
end
output= F2'/max(F2);

paras.maxIntensity=selectedNodes{1}.epochList.firstValue.protocolSettings ...
    ('epoch:Microdisplay_Stage@localhost:white:rodConversionFactor');
fprintf('%s , %f \n', 'max Luminance::', paras.maxIntensity');
%% export figure 4 if necessary
cellName= [regexprep(selectedNodes{1}.parent.parent.parent.parent.parent.splitValue,'/','-') '-'  ...
    selectedNodes{1}.parent.parent.parent.splitValue '-' ...
    regexprep(selectedNodes{1}.parent.parent.parent.parent.splitValue,'\','-') '-' ...
    selectedNodes{1}.splitValue '-' selectedNodes{1}.parent.splitValue];
exportFigToPDF([cellName '-example-F1F2-plot'], f3, 300);


%% analyze flashed gratings versus eqv disc
clc; clear resMat meanRes
selectedNodes = gui.getSelectedEpochTreeNodes;
paras.spikeTh=1.7;
CloseAllFiguresExceptGUI;
paras.spikeTag=0;
paras.psthSigma=10;
paras.sampleRate=10000;
timeToPts=@(x) x/1e3*paras.sampleRate;

stimPts=timeToPts(selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime'));
prePts=timeToPts(selectedNodes{1}.epochList.firstValue.protocolSettings('preTime'));

resMat=riekesuite.getResponseMatrix(selectedNodes{1}.epochList,'Amp1');

barWidths=zeros(size(resMat,1),1);
contrastList=zeros(size(resMat,1),1);
stimTags=zeros(size(resMat,1),1);
for i=1:size(resMat,1)
    if strcmp(selectedNodes{1}.parent.splitValue, '[spike]') || paras.spikeTag
        resMat(i,:)=resMat(i,:)-movmedian(resMat(i,:),100);
    else
        resMat(i,:)=smooth(resMat(i,:),100);
    end
    barWidths(i)=selectedNodes{1}.epochList.elements(i).protocolSettings('currentBarWidth');
    contrastList(i)=selectedNodes{1}.epochList.elements(i).protocolSettings('currentEqvContrast');
    if strcmp(selectedNodes{1}.epochList.elements(i).protocolSettings('currentStimulusTag'),'grate')
        stimTags(i)=1;
    else
        stimTags(i)=2;
    end
end

if strcmp(selectedNodes{1}.parent.splitValue, '[spike]') || paras.spikeTag
    resMat=spikeBinary(resMat, paras.spikeTh);
else
    resMat=resMat-repmat(mean(resMat(:,1:timeToPts(preTime)),2),1,size(resMat,2));
end

barList=unique(barWidths); conList=unique(contrastList); tags=unique(stimTags);
meanRes.onset=zeros(numel(tags),numel(barList),numel(conList));
meanRes.offset=zeros(numel(tags),numel(barList),numel(conList));

if strcmp(selectedNodes{1}.parent.splitValue, '[spike]')  || paras.spikeTag
    psth=spikeBinaryToPSTH(resMat, paras.psthSigma, paras.sampleRate);
end

colors=pmkmp(numel(barList),'IsoL');
f3=figure('position',[100 300 1400 600]);
ax(1)=subplot(1,2,1); hold all;
ax(2)=subplot(1,2,2); hold all;

for j=1:size(meanRes.onset,2)
    for i=1:size(meanRes.onset,1)
        for k=1:size(meanRes.onset,3)
            tpInd=find(tags(i)==stimTags & barList(j)==barWidths & conList(k)==contrastList);
            try 
            meanRes.onset(i,j,k)=sum(resMat(tpInd,prePts+1:prePts+stimPts),2)/numel(tpInd);
            meanRes.offset(i,j,k)=sum(resMat(tpInd,prePts+stimPts+1:end),2)/numel(tpInd);
            catch 
                disp('this comb of stim is not ran');
            end
        end
    end
    scatter(ax(1), squeeze(meanRes.onset(1,j,:)),squeeze(meanRes.onset(2,j,:)),barList(j)*10,'filled');
    scatter(ax(2), squeeze(meanRes.offset(1,j,:)),squeeze(meanRes.offset(2,j,:)),barList(j)*10,'filled');
end
% add unity line 
line([min(meanRes.onset(:)), max(meanRes.onset(:))], [min(meanRes.onset(:)), ...
    max(meanRes.onset(:))], 'Parent', ax(1),'Color','k','Marker','none','LineStyle','--');
line([min(meanRes.offset(:)), max(meanRes.offset(:))], [min(meanRes.offset(:)),...
    max(meanRes.offset(:))], 'Parent', ax(2),'Color','k','Marker','none','LineStyle','--');
title(ax(1), 'onset');  title(ax(2), 'offset'); 
xlabel(ax(1),'response to Grating');  xlabel(ax(2),'response to Grating');
ylabel(ax(1),'response to disc');  ylabel(ax(2),'response to disc');

%% analyze contrast response spots
clc; clear epochTraces contrastResponse spotContrast countrastCounts meanResp figH condNames
selectedNodes = gui.getSelectedEpochTreeNodes;
paras.spikeTh=1.7;
CloseAllFiguresExceptGUI;
paras.spikeTag=0;

stimTime=selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');
preTime=selectedNodes{1}.epochList.firstValue.protocolSettings('preTime');
tailTime=selectedNodes{1}.epochList.firstValue.protocolSettings('tailTime');

numConds=selectedNodes{1}.children.length;  % number of intensity conditions
spotContrastList=convertJavaArrayListMatrix(selectedNodes{1}.epochList.firstValue.protocolSettings('spotContrast'));
for i=1:numConds
    condNames{i}=[selectedNodes{1}.children(i).splitValue  'ContrastSpots'];
    epochTraces.(condNames{i})=riekesuite.getResponseMatrix(selectedNodes{1}.children(i).epochList,'Amp1');
    if strcmp(selectedNodes{1}.splitValue, '[spike]') || paras.spikeTag
        [spikeTimes,~,~,~]=SpikeDetectorNew(epochTraces.(condNames{i}), 'thresholdSpikeFactor', paras.spikeTh);
    end
    countrastCounts.(condNames{i})=zeros(size(spotContrastList));
    meanResp.(condNames{i})=zeros(size(spotContrastList));
    for ii=1:size( epochTraces.(condNames{i}),1)
        currentContrast=selectedNodes{1}.children(i).epochList.elements(ii).protocolSettings('currentSpotContrast');
        spotContrast.(condNames{i})(ii)=currentContrast;
        contrastInd=find(currentContrast==spotContrastList);
        if strcmp(selectedNodes{1}.splitValue, '[spike]') || paras.spikeTag
            epochTraces.(condNames{i})(ii,:)= epochTraces.(condNames{i})(ii,:)-movmedian (epochTraces.(condNames{i})(ii,:),100);
            contrastResponse.(condNames{i})(ii)=length(spikeTimes{ii}(spikeTimes{ii}>timeToPts(preTime) & spikeTimes{ii}<timeToPts(preTime+stimTime)));
        else
            epochTraces.(condNames{i})(ii,:)=smooth( epochTraces.(condNames{i})(ii,:),100);
            contrastResponse.(condNames{i})(ii)=sum(epochTraces.(condNames{i})(ii,timeToPts(preTime):timeToPts(preTime+stimTime)) ...
                -mean(epochTraces.(condNames{i})(ii,1:timeToPts(preTime))))/1e4;
        end
        countrastCounts.(condNames{i})(contrastInd)=countrastCounts.(condNames{i})(contrastInd)+1;
        meanResp.(condNames{i})(contrastInd)=meanResp.(condNames{i})(contrastInd)+contrastResponse.(condNames{i})(ii);
        
    end
    meanResp.(condNames{i})=meanResp.(condNames{i})./countrastCounts.(condNames{i});
end

contrastFig=figure('position',[400 400 800 700]); hold all;
newcolors = [0.83 0.14 0.14
             1.00 0.54 0.00
             0.47 0.25 0.80
             0.25 0.80 0.54];
for i=1:numConds
    scatter(spotContrast.(condNames{i}),contrastResponse.(condNames{i}),120,newcolors(i,:),'filled');
    figH(i)=plot(spotContrastList,meanResp.(condNames{i}),'linewidth',3,'color',newcolors(i,:));
end

setAxes(contrastFig); initFig(gca(contrastFig),'Contrast','Integrated Amplitude'); title('contrast Spots');
legend(figH,condNames); 

%% export figures if necessary
cellName= [regexprep(selectedNodes{1}.parent.parent.parent.parent.splitValue,'/','-') '-'  ...
    selectedNodes{1}.parent.parent.splitValue '-' ...
    regexprep(selectedNodes{1}.parent.parent.parent.splitValue,'\','-') '-' ...
    selectedNodes{1}.splitValue '-' selectedNodes{1}.parent.splitValue];
exportFigToPDF([cellName '-contrastTuning'], f3, 300);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% analyze the new flashed grating protocol (need to split on epoch groups) 
dateSplit = @(listSorted)splitOnExperimentDate(listSorted);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, dateSplit);

cellTypeSplit = @(listSorted)splitOnCellType(listSorted);
cellTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, cellTypeSplit);

recordingTypeSplit = @(listSorted)splitOnRecordingType(listSorted);
recordingTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, recordingTypeSplit);

protocolSplit = @(listSorted)splitOnShortProtocolID(listSorted);
ProtocolSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, protocolSplit);

tree = riekesuite.analysis.buildTree(listSorted,{cellTypeSplit_java,dateSplit_java, 'cell.label',ProtocolSplit_java,...
    'protocolSettings(epochGroup:label)','protocolSettings(onlineAnalysis)',...
    'protocolSettings(background:Microdisplay_Stage@localhost:microdisplayBrightness)'});
gui = epochTreeGUI(tree);
%% load the data and create plots and stats 
clc; 
selectedNodes = gui.getSelectedEpochTreeNodes;
paras.spikeTh=1.7;
CloseAllFiguresExceptGUI;
paras.spikeTag=0;
resMat=riekesuite.getResponseMatrix(selectedNodes{1}.epochList,'Amp1');
barWidths=zeros(size(resMat,1),1);
for i=1:size(resMat,1)
    if strcmp(selectedNodes{1}.parent.splitValue, 'extracellular') || paras.spikeTag
        resMat(i,:)=resMat(i,:)-movmedian(resMat(i,:),100);
    else
        resMat(i,:)=smooth(resMat(i,:),200); 
    end
    barWidths(i)=selectedNodes{1}.epochList.elements(i).protocolSettings('currentBarWidth');
end
resMat=resMat-repmat(mean(resMat(:,1:100),2),1,size(resMat,2));
barList=unique(barWidths);
figure('position',[100 200 600 900]);
meanRes=zeros(numel(barList),size(resMat,2)); hold all; 
for i=1:numel(barList)
    meanRes(i,:)=mean(resMat(barList(i)==barWidths,:));
    plot(meanRes(i,:)+i*100,'k','linewidth',2); tx=text(size(resMat,2)/8,i*420+mean2(meanRes),['barWidth::',num2str(barList(i))]);
    set(tx,'fontsize',15,'color','r');
end


%% split data and create GUI for FlashedGrating
dateSplit = @(listSorted)splitOnExperimentDate(listSorted);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, dateSplit);

cellTypeSplit = @(listSorted)splitOnCellType(listSorted);
cellTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, cellTypeSplit);

ndfSplit = @(listSorted)splitOnNDFs(listSorted);
ndfSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, ndfSplit);


tree = riekesuite.analysis.buildTree(listSorted,{cellTypeSplit_java,dateSplit_java, 'cell.label',...
    'protocolSettings(onlineAnalysis)',ndfSplit_java});
gui = epochTreeGUI(tree);


%% %% analyze flashed gratings versus eqv disc
clc; clear resMat meanTrace meanCount
selectedNodes = gui.getSelectedEpochTreeNodes;
paras.spikeTh=1.5;
CloseAllFiguresExceptGUI;
paras.psthSigma=15;
paras.sampleRate=1e4;
timeToPts=@(x) x/1e3*paras.sampleRate;
paras.offset=0;
paras.basePts=100;
meanInt=selectedNodes{1}.epochList.firstValue.protocolSettings('backgroundIntensity');

for node=1:numel(selectedNodes)
    stimPts=timeToPts(selectedNodes{node}.epochList.firstValue.protocolSettings('stimTime'));
    prePts=timeToPts(selectedNodes{node}.epochList.firstValue.protocolSettings('preTime'));
    tailPts=timeToPts(selectedNodes{node}.epochList.firstValue.protocolSettings('tailTime'));   
    resMat=riekesuite.getResponseMatrix(selectedNodes{node}.epochList,'Amp1');  
    barWidths=zeros(size(resMat,1),1);
    for i=1:size(resMat,1)
        if strcmp(selectedNodes{node}.parent.splitValue, 'extracellular')
            resMat(i,:)=resMat(i,:)-movmedian(resMat(i,:),100);
        else
            resMat(i,:)=smooth(resMat(i,:),100);
            resMat(i,:)=noisyBaseline(resMat(i,:),prePts,paras.basePts);
        end
        barWidths(i)=selectedNodes{node}.epochList.elements(i).protocolSettings('currentBarWidth');
    end
    paras.epochRange=1:size(resMat,1);
%     paras.epochRange=[31:79];
    resMat=resMat(paras.epochRange,:); barWidths=barWidths(paras.epochRange);
    
    if strcmp(selectedNodes{node}.parent.splitValue, 'extracellular')
        resMat=smoothPSTH(resMat, paras.psthSigma,paras.sampleRate,paras.spikeTh);
    end
    
    barList=unique(barWidths);
    
    meanTrace{node}=zeros(numel(barList),size(resMat,2));
    colors=pmkmp(numel(barList),'IsoL');
    f1=figure('position',[100 200 1800 800]);
    for k=1:size(meanTrace{node},1)
        tpInd=find( barList(k)==barWidths);
        meanTrace{node}(k,:)=sum(resMat(tpInd,:))/numel(tpInd);
        subplot(2, ceil(numel(barList)/2),k);
        plot(resMat(tpInd,:)'); hold all;
        plot(meanTrace{node}(k,:),'k','linewidth',2);
        meanCount.onset(k)=(mean(meanTrace{node}(k,prePts:prePts+stimPts+paras.offset-1)));
        meanCount.offset(k)=(mean(meanTrace{node}(k,prePts+stimPts++paras.offset:end)));
    end
     if strcmp(selectedNodes{node}.parent.splitValue, 'extracellular')
         plotOffset=100;
     elseif strcmp(selectedNodes{node}.parent.splitValue, 'exc')
         plotOffset=200;
     else 
         plotOffset=300;
     end
    
    f2=figure('position',[100 200 1800 800]);
    subplot(1,3,1);
    hold all;
    for k=1:size(meanTrace{node},1)
        plot(meanTrace{node}(k,:)+plotOffset*k,'k','linewidth',2);
    end
    title('mean Trace');
    subplot(1,3,2);
    plot(barList, meanCount.onset,'r','linewidth',3);
    set(gca,'xtick',barList); box off; title('Onset');
    
    subplot(1,3,3);
    plot(barList, meanCount.offset,'r','linewidth',3);  box off;  title('Offset');
    s=sgtitle(selectedNodes{node}.parent.splitValue); set(s,'fontsize',40);
end


if numel(selectedNodes)>1
    colors='krb';
    f3=figure('position',[100 200 400 800]);
    hold all;
    for i=1:numel(barList)
        for j=1:numel(selectedNodes)
            
            plot(meanTrace{j}(i,:)+plotOffset*i,'color',colors(j),'linewidth',2);
        end
    end
end
        
%% spit the data for drug application experiments.  ( Glycine, LY+ APB, etc. ) 
dateSplit = @(listSorted)splitOnExperimentDate(listSorted);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, dateSplit);

cellTypeSplit = @(listSorted)splitOnCellType(listSorted);
cellTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, cellTypeSplit);

ndfSplit = @(listSorted)splitOnNDFs(listSorted);
ndfSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, ndfSplit);

protocolSplit = @(listSorted)splitOnShortProtocolID(listSorted);
ProtocolSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, protocolSplit);


tree = riekesuite.analysis.buildTree(listSorted,{cellTypeSplit_java,dateSplit_java, 'cell.label', ProtocolSplit_java, ...
    'protocolSettings(onlineAnalysis)',ndfSplit_java,'protocolSettings(epochGroup:label)'});
gui = epochTreeGUI(tree);

%% analyze contrast reversing gratings for drug experiments
clear  resMat  meanRes
selectedNodes = gui.getSelectedEpochTreeNodes;
paras.spikeTh=1.5;
CloseAllFiguresExceptGUI;
paras.spikeTag=0;
paras.psthSigma=20;

sampleRate=selectedNodes{1}.epochList.firstValue.protocolSettings('sampleRate');
tempFreq=selectedNodes{1}.epochList.firstValue.protocolSettings('temporalFrequency');
timeToPts=@(x) x/1e3*sampleRate;

rawF=figure;
axes1 = axes('Parent',rawF,'FontSize',12,'FontName','arial'); 
for node=1:numel(selectedNodes)
    stimTime=selectedNodes{node}.epochList.firstValue.protocolSettings('stimTime'); 
   
    preTime=selectedNodes{node}.epochList.firstValue.protocolSettings('preTime');
    
    resMat=riekesuite.getResponseMatrix(selectedNodes{node}.epochList,'Amp1');
    paras.epochRange=1:size(resMat,1);
%     paras.epochRange=1:13;
    resMat=resMat(paras.epochRange,:);
    
    barWidths=zeros(size(resMat,1),1);
    for i=1:size(resMat,1)    
        resMat(i,:)=smooth(resMat(i,:),100);
        barWidths(i)=selectedNodes{node}.epochList.elements(paras.epochRange(i)).protocolSettings('currentBarWidth');
    end
    resMat=resMat-repmat(mean(resMat(:,1:timeToPts(preTime)),2),1,size(resMat,2));
    uniqueBarWidth=unique(barWidths);
    F1=zeros(numel(uniqueBarWidth),1); F2=zeros(numel(uniqueBarWidth),1);
    meanRes{node}=zeros(numel(uniqueBarWidth),size(resMat,2));

    for i=1:numel(uniqueBarWidth)
        barInd=find(barWidths==uniqueBarWidth(i));       
        tp=sum(resMat(barInd,:),1)/numel(barInd);  
        plot(axes1,resMat(barInd,:)');  pause; 
        meanRes{node}(i,:)=tp;
        F2(i)=sum(tp(timeToPts(preTime)+1:timeToPts(preTime+stimTime)))/sampleRate;
        [F1(i), F2(i)]=computeF1F2(tp,sampleRate,tempFreq);
        
    end
    f3=figure('position',[300 500 1000 500]);
    t=sgtitle(selectedNodes{node}.splitValue); set(t,'fontsize',30);
    subplot(1,2,2);
    h2 = line(uniqueBarWidth, F2/max(F2));
    set(h2,'Color','r','LineWidth',2,'Marker','o');
    setAxes(f3); initFig(gca(f3),'Bar width (um)','Integrated Amplitude');
    
    % create examplary plot for whole cell data 
    subplot(1,2,1);
    hold all;
    for i=1:numel(uniqueBarWidth)
        if strcmp(selectedNodes{node}.parent.parent.splitValue, 'exc')
            plot(meanRes{node}(i,:)+150*(i-1));
        else
            plot(meanRes{node}(i,:)+150*(i-1));
        end
    end
    output= F2'/max(F2);
end
if numel(selectedNodes)>1
    drugSensitive=meanRes{1}-meanRes{2};
    for i=1:numel(uniqueBarWidth)
        tp=drugSensitive(i,:);
        F2(i)=sum(tp(timeToPts(preTime)+1:timeToPts(preTime+stimTime)))/sampleRate;
        [F1(i), F2(i)]=computeF1F2(tp,sampleRate,tempFreq);    
    end
    f3=figure('position',[300 500 1000 500]);
    t=sgtitle('drug sensitive component'); set(t,'fontsize',30);
    subplot(1,2,2);
    h2 = line(uniqueBarWidth, F2/max(F2));
    set(h2,'Color','r','LineWidth',2,'Marker','o');
    setAxes(f3); initFig(gca(f3),'Bar width (um)','Integrated Amplitude');
    
    % create examplary plot for whole cell data
    subplot(1,2,1);
    hold all;
    for i=1:numel(uniqueBarWidth)
        plot(drugSensitive(i,:)+100*(i-1));
    end
    output= F2'/max(F2);
    
    exaInd=[2 4 6];
    ylims=[min(resMat(:))-20 max(resMat(:))+50];
    figure('position',[300 300 1000 800]);
    for i=1:numel(exaInd)
        subplot(numel(exaInd),1,i); hold all;
        plot(meanRes{1}(exaInd(i),:),'k');
        plot(meanRes{2}(exaInd(i),:),'r');
        plot(drugSensitive(exaInd(i),:),'b');
        legend('pre','drug','drug-sensitive');
        ylim(ylims);
        hold off; title(['bar width::',num2str(uniqueBarWidth(exaInd(i)))]);
    end
end

%% analyze the flash grating drug experiments. 
clc;  clear meanRes
selectedNodes = gui.getSelectedEpochTreeNodes;
paras.spikeTh=1.7;
CloseAllFiguresExceptGUI;
paras.spikeTag=0;

stimTime=selectedNodes{1}.epochList.firstValue.protocolSettings('stimTime');
preTime=selectedNodes{1}.epochList.firstValue.protocolSettings('preTime');
fprintf('%s %d %s %d \n','preTime:',preTime,'stimTime:',stimTime);
for node=1:numel(selectedNodes) 
    resMat=riekesuite.getResponseMatrix(selectedNodes{node}.epochList,'Amp1');
    barWidths=zeros(size(resMat,1),1);
    for i=1:size(resMat,1)
        if strcmp(selectedNodes{node}.parent.parent.splitValue, 'spike') || paras.spikeTag
            resMat(i,:)=resMat(i,:)-movmedian(resMat(i,:),100);
        else
            resMat(i,:)=smooth(resMat(i,:),200);
        end
        barWidths(i)=selectedNodes{node}.epochList.elements(i).protocolSettings('currentBarWidth');
    end
    resMat=resMat-repmat(mean(resMat(:,1:100),2),1,size(resMat,2));
    barList=unique(barWidths);
    
    meanRes{node}=zeros(numel(barList),size(resMat,2));
    for i=1:numel(barList)
        meanRes{node}(i,:)=mean(resMat(barList(i)==barWidths,:));  
    end
end
 
figure('position',[100 200 600 900]); hold all;
for i=1:numel(barList)
    ht(1)=plot(meanRes{1}(i,:)+i*100,'k','linewidth',2);
    try
    ht(2)=plot(meanRes{2}(i,:)+i*100,'r','linewidth',2);
    end
    tx=text(size(resMat,2)/8,i*120+mean2(meanRes{1}),['barWidth::',num2str(barList(i))]);
    set(tx,'fontsize',15,'color','r');
end
legend(ht,{'pre','drug'});
t=title(selectedNodes{node}.splitValue); set(t,'fontsize',30);