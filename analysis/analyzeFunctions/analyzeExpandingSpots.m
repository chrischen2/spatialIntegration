function [ax,output,onlineAnalysis] = analyzeExpandingSpots(selectedNodes,paras)
sizeInsp=[20  40 60  80 120 160 400];
for node=1:numel(selectedNodes)
    resMat=riekesuite.getResponseMatrix(selectedNodes{node}.epochList,'Amp1');
    spotSize=zeros(1, selectedNodes{node}.epochList.length);
    for i=1:length(spotSize)
        spotSize(i)=selectedNodes{node}.epochList.elements(i).protocolSettings('currentSpotSize');
    end
    paras.epochRange=1:size(resMat,1);
    %         if strcmp(selectedNodes{node}.splitValue, 'exc')
%             paras.epochRange=find(mean(resMat,2)<0); 
%         elseif strcmp(selectedNodes{node}.splitValue, 'inh')
%             paras.epochRange=find(mean(resMat,2)>0);
%         end
%     paras.epochRange=121:280;
    resMat=resMat(paras.epochRange,1:10000); spotSize=spotSize(paras.epochRange);
    onlineAnalysis=selectedNodes{node}.splitValue;
    if strcmp(selectedNodes{node}.epochList.firstValue.protocolSettings('epochGroup:pipetteSolution'),'potassium')  % currentClamp
        onlineAnalysis='currentClamp'; 
    else
        if ~strcmp(onlineAnalysis, 'extracellular')
            if mean2(resMat)<0
                onlineAnalysis='exc';  % some recordings did not set the right onlineAnalysis
            else
                onlineAnalysis='inh';
            end
        end
        if strcmp(onlineAnalysis, 'exc')
            resMat=-resMat;
        end
    end
    
    if strcmp(selectedNodes{node}.splitValue, 'extracellular') || paras.spikeTag
        for i=1:size(resMat,1)
            resMat(i,:)=resMat(i,:)-movmedian(resMat(i,:),100);
        end
        [spikeTimes,~,~]=SpikeDetectorNew(resMat, 'thresholdSpikeFactor',paras.spikeTh);
        trace=spikeTimeToPSTH(resMat,spikeTimes,paras.psthSigma, paras.sampleRate);
    else
        for i=1:size(resMat,1)
            resMat(i,:)=smooth(resMat(i,:),100);
        end
        trace=resMat-repmat(mean(resMat(:,1:paras.prePts),2),1,size(resMat,2)); % baseline adjustment
    end
    
    sizeArray{node}=unique(spotSize); resArray{node}=zeros(1,length(sizeArray{node})); errArray{node}=zeros(1,length(sizeArray{node}));
    mTrace{node}=zeros(length(sizeArray{node}), size(trace,2));
    
    for i=1:length(sizeArray{node})
        if strcmp(selectedNodes{node}.splitValue, 'extracellular') || paras.spikeTag
            temp=cellfun(@(x) length(x(x>paras.prePts+paras.spikeOffset & x< paras.prePts+paras.stimPts+paras.spikeOffset)),spikeTimes(sizeArray{node}(i)==spotSize));
            base=cellfun(@(x) length(x(x<paras.prePts+paras.spikeOffset)),spikeTimes(sizeArray{node}(i)==spotSize));
            temp=temp-base*paras.stimPts/paras.prePts;
        else
            temp=mean(trace(sizeArray{node}(i)==spotSize,paras.prePts+paras.wcOffset:paras.prePts+paras.stimPts+paras.wcOffset),2)*paras.stimPts/1e4; % unit pA*s
        end
        resArray{node}(i)=mean(temp);
        errArray{node}(i)=std(temp)/sqrt(length(temp));
        mTrace{node}(i,:)=sum(trace(sizeArray{node}(i)==spotSize,:),1)/numel(temp);
    end
    
    if strcmp(onlineAnalysis,'exc')  % reverse back
        mTrace{node}=-mTrace{node};
    end
    % normalize the curve
    % export the minimal
    minInd=find(sizeArray{node}==40);
    fprintf('%s %d %s %d %s %d \n', 'min spot size ', sizeArray{node}(minInd), ' and min res ', resArray{node}(minInd),' stim Time(ms)', paras.stimPts/10);
    output.minRes=[sizeArray{node}(minInd)  resArray{node}(minInd)  paras.stimPts/10];
    if ~isempty(find(sizeArray{node}==20))
        minInd=find(sizeArray{node}==20);
       fprintf('%s %d %s %d %s %d \n', 'min spot size ', sizeArray{node}(minInd), ' and min res ', resArray{node}(minInd),' stim Time(ms)', paras.stimPts/10);
       output.minRes=[  output.minRes sizeArray{node}(minInd)  resArray{node}(minInd)  paras.stimPts/10];
    end
    scalor=max(abs(resArray{node})); 
    resArray{node}=resArray{node}/scalor; errArray{node}=errArray{node}/scalor;
    % [Kc,sigmaC,Ks,sigmaS]=fitDoG(linspace(min(sizeArray{node}), max(sizeArray{node}), 50), ...,
    %     interp1(sizeArray{node}, resArray{node}, linspace(min(sizeArray{node}), max(sizeArray{node}), 50)), [1 100 1 400]);
    if strcmp(onlineAnalysis, 'extracellular') || paras.spikeTag
        [Kc,sigmaC,Ks,sigmaS,baseF]=fitDoG(spotSize', ...,
            cellfun(@(x) length(x(x>paras.prePts & x< paras.prePts+paras.stimPts)),spikeTimes)/scalor, [1 5 3 500 0]);
    else
        [Kc,sigmaC,Ks,sigmaS,baseF]=fitDoG(spotSize', ...,
            mean(trace(:,paras.prePts+paras.wcOffset:paras.prePts+paras.stimPts+paras.wcOffset),2)*paras.stimPts/(scalor*1e4), [1 5 1 500 0]);
    end
    
    f=figure('color','w','position',[1200 300 800 1200]);
    ax(1)=subplot(2,1,1); hold all;
    errorbar(sizeArray{node}, resArray{node}, errArray{node}); hold all;
    plot(sizeArray{node}, DoG([Kc,sigmaC,Ks,sigmaS,baseF], sizeArray{node}),'linewidth',5,'color','r');
    if strcmp(selectedNodes{node}.splitValue, 'extracellular') || paras.spikeTag
        scatter(spotSize, cellfun(@(x) length(x(x>paras.prePts+paras.wcOffset & x< paras.prePts+paras.stimPts+paras.wcOffset)),spikeTimes)/scalor,100,'markeredgecolor','b','markerfacecolor','b');
    else
        scatter(spotSize,mean(trace(:,paras.prePts+paras.wcOffset:paras.prePts+paras.stimPts+paras.wcOffset),2)*paras.stimPts/(scalor*1e4),100,'markeredgecolor','b','markerfacecolor','b');
    end
    hold off; box off; xlabel('spot Size/um'); 
    ax=setAxes(f(1)); ax.Title.String=['sigmaC' ' ' num2str(sigmaC) ' sigmaS' ' ' num2str(sigmaS)];
    
    ax(2)=subplot(2,1,2);
    hold all;
    for i=1:size(mTrace{node},1)
        plot((1:size(mTrace{node},2))/paras.sampleRate,mTrace{node}(i,:),'linewidth',3);
    end
    setAxes(f); legend(cellstr(num2str(sizeArray{node}', 'Size %-d')),'fontsize',15); legend boxoff;
    modelResult.Kc=Kc; modelResult.Ks=Ks; modelResult.sigmaC=sigmaC; modelResult.sigmaS=sigmaS; modelResult.baseF=baseF;
    output.spotList=sizeArray{node}; output.normRes=resArray{node}; output.model=modelResult;
    
    inds=find(ismember(sizeArray{node},sizeInsp));
%     inds=1:numel(sizeArray{node});
%     figure('color','w','position',[1000 300 500 1200]);
%     hold all;
%     for i=1:numel(inds)
%         plot((1:size(mTrace{node},2))/paras.sampleRate,mTrace{node}(inds(i),:)+paras.plotOffset*(i-1), 'k','linewidth',3);
%     end
    condStr{node}=selectedNodes{node}.parent.splitValue;
end
bgIntensity=selectedNodes{node}.epochList.firstValue.protocolSettings ...
    ('epoch:Microdisplay_Stage@localhost:white:rodConversionFactor')*paras.backgroundIntensity;
fprintf('%s , %f \n', 'background Luminance::',bgIntensity');
if numel(selectedNodes)>1
    figure('color','w','position',[1000 300 600 600]);
    hold all;
    for node=1:numel(selectedNodes)
        errorbar(sizeArray{node}, resArray{node}, errArray{node},'linewidth',2);  
    end
    legend(condStr);   legend boxoff;
    figure('color','w','position',[1000 300 600 1000]);
    hold all;
    colors='krbgc';
    for node=1:numel(selectedNodes)
        for i=1:numel(inds)
            h(node)=plot((1:size(mTrace{node},2))/paras.sampleRate,mTrace{node}(inds(i),:)+paras.plotOffset*(i-1), 'color',colors(node),'linewidth',2);
        end
    end
    legend(h,condStr); legend boxoff;
end

