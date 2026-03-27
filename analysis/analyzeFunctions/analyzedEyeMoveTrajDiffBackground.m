function [stats,paras,mResponse] = analyzedEyeMoveTrajDiffBackground(selectedNodes,paras)
stats.cellType=shortCellType(selectedNodes{1}.epochList.firstValue.protocolSettings('source:type'));
stats.expDate=datestr(selectedNodes{1}.epochList.elements(1).startDate','yyyy/mm/dd');
stats.cellLabel=selectedNodes{1}.epochList.firstValue.protocolSettings('source:label');
paras.startingLum={'low','high'};
timeToPts=@(x) x/1e3*(paras.sampleRate/paras.downsample); prePts=timeToPts(paras.preTime); stimPts=timeToPts(paras.stimTime); 
offsetPts=timeToPts(paras.clipOffset); clipPts=timeToPts(paras.clipOnset);
stats.apertureDiameter=selectedNodes{1}.epochList.firstValue.protocolSettings('apertureDiameter');

% % load information about the movies
% resourcesDir = '/Volumes/GoogleDrive/My Drive/codes/stimulus/chris-package/+edu/+washington/+riekelab/+chris/+resources/';
% paras.currentImageSet = '/VHsubsample_20160105';
% paras.currentStimSet = 'SaccadeLocationsLibrary_20171011';
% load([resourcesDir,paras.currentStimSet,'.mat']);
% fieldName = ['imk', paras.imageName];
% % 
% % %load appropriate image...
% paras.currentStimSet = '/VHsubsample_20160105';
% fileId=fopen([resourcesDir, paras.currentImageSet, '/imk', paras.imageName,'.iml'],'rb','ieee-be');
% img = fread(fileId, [1536,1024], 'uint16');
% img = double(img);
% img = (img./max(img(:))); %rescale s.t. brightest point is maximum monitor level
% img = img.*255; %rescale s.t. brightest point is maximum monitor level
% imageMatrix = uint8(img');
% %          

%  %1) restrict to desired patch mean luminance:
%  imageMean = imageData.(fieldName).imageMean;
%  paras.backgroundIntensity = imageMean;%set the mean to the mean over the image
%  locationMean = imageData.(fieldName).patchMean;
%  
%  if strcmp(paras.patchMean,'all')
%      inds = 1:length(locationMean);
%  elseif strcmp(paras.patchMean,'positive')
%      inds = find((locationMean-imageMean) > 0);
%  elseif strcmp(paras.patchMean,'negative')åç
%      inds = find((locationMean-imageMean) <= 0);
%  end
%  rng(paras.randomSeed); %set random seed for fixation draw
%  drawInd = randsample(inds,1);
%  paras.p0(1) = imageData.(fieldName).location(drawInd,1);
%  paras.p0(2) = imageData.(fieldName).location(drawInd,2);
% 
%  % xTraj, yTraj are in canvas pixels. translate into VH pixels. 
%  paras.xTraj= paras.p0(1)-paras.xTraj.*paras.umPerPixel/3.3;   % vh pixels
%  paras.yTraj= paras.p0(2)-paras.yTraj.*paras.umPerPixel/3.3;   % vh pixels
 
if stats.apertureDiameter==0
    stats.apertureDiameter=800;
end
baseRange=timeToPts(paras.baseRange(1)*1e4):timeToPts(paras.baseRange(2)*1e4);

for node=1:numel(selectedNodes)
    resMat=riekesuite.getResponseMatrix(selectedNodes{node}.epochList,'Amp1');
    paras.epochRange=1:size(resMat,1);
    paras.epochRange(ismember(paras.epochRange,paras.rmRep))=[];
    resMat=resMat(paras.epochRange,:);
    if strcmp(selectedNodes{node}.parent.splitValue, 'extracellular')  || paras.spikeTag
        resMat=arrayfun(@(rowInd) resMat(rowInd,:)-movmedian(resMat(rowInd,:),200),...
            1:size(resMat,1),'UniformOutput',false); resMat=cell2mat(resMat');
        [resMat,~,~,spikeTimes]=smoothPSTH(resMat,paras.psthSigma,paras.sampleRate,paras.spikeTh);
        paras.recType='extracellular';
    else
        if mean2(resMat)<0
            paras.recType='exc';
        else
            paras.recType='inh';
        end
        resMat=smoothMatrix(resMat,50);
%         resMat=baseAdjust(resMat,baseRange,100);
        resMat=resMat-repmat(mean(resMat(:,baseRange),2),1,size(resMat,2));
    end
%     
%     % checking the frame timing over the super long epoch 
%     badFrames=zeros(1,size(resMat,1));
%     for i=1:size(resMat,1)
%         if ~isempty(strfind(selectedNodes{1}.epochList.elements(i).keywords,'badFrameTiming'))
%             badFrames(i)=1;
%         end
%     end
%     if sum(badFrames)~=0
%         fprintf('%s %s\n','bad frames as:epoch ', mat2str(find(badFrames==1)));
%     end
    tempMat=zeros(size(resMat,1), size(resMat,2)/paras.downsample);
    for i=1:size(resMat,1)
        tpRes=resMat(i,:);
        tempMat(i,:)=arrayfun(@(x) mean(tpRes(x:x+paras.downsample-1)),1:paras.downsample:length(tpRes)-paras.downsample+1);
    end
    resMat=tempMat; 
    backgrounds=zeros(1,numel(paras.epochRange));
    for i=1:numel(paras.epochRange)
        backgrounds(i)=selectedNodes{node}.epochList.elements(paras.epochRange(i)).protocolSettings('backgroundScale');
    end
    backgroundList=unique(backgrounds);  %background scale list
    if numel(backgroundList)~=2
        fprintf('%s %d \n', 'number of backgrounds:', numel(backgroundList));
        error('wrong settings, not two backgrounds in the epoch Group');
    end
    clip1=prePts+clipPts+1:prePts+stimPts/2-offsetPts; clip2=prePts+stimPts/2+clipPts+1:prePts+stimPts-offsetPts;
    for i=1:numel(backgroundList)
        response.(paras.startingLum{i})=resMat(backgroundList(i)==backgrounds,:);
        mResponse.(paras.startingLum{i}).full= sum(response.(paras.startingLum{i}),1)/size(response.(paras.startingLum{i}),1);
        mResponse.(paras.startingLum{i}).loop1=mResponse.(paras.startingLum{i}).full(:, clip1);
        mResponse.(paras.startingLum{i}).loop2=mResponse.(paras.startingLum{i}).full(:, clip2);
        if strcmp(paras.recType,'extracellular')
            spt.(paras.startingLum{i})=spikeTimes(backgroundList(i)==backgrounds);
        end
    end
    paras.imageName=selectedNodes{1}.epochList.firstValue.protocolSettings('imageName');
    paras.xTraj=convertJavaArrayList(selectedNodes{1}.epochList.firstValue.protocolSettings('xTraj'));
    paras.yTraj=convertJavaArrayList(selectedNodes{1}.epochList.firstValue.protocolSettings('yTraj'));
    nodeResponse{node}=mResponse;
    xTraj{node}=paras.xTraj; yTraj{node}=paras.yTraj;    
    timePts=(1:size(resMat,2))/paras.sampleRate*paras.downsample;
    lowInd=paras.epochRange(backgroundList(1)==backgrounds);
    highInd=paras.epochRange(backgroundList(2)==backgrounds);
    figure('color','w','position',[100 450 1200 700]);
    subplot(3,1,1); hold all;
    plot(timePts,response.(paras.startingLum{1})');
    box off;
    h = findobj(gca,'Type','line');  legend(h, strsplit(num2str(flip(lowInd)))); legend boxoff
%     plot(timePts,mResponse.(paras.startingLum{1}).full,'r');
    subplot(3,1,2); hold all;
    plot(timePts,response.(paras.startingLum{2})');
    box off;
    h = findobj(gca,'Type','line');  legend(h, strsplit(num2str(flip(highInd))));  legend boxoff
%     plot(timePts,mResponse.(paras.startingLum{2}).full,'k');
    subplot(3,1,3);
    hold all;
    plot(timePts,mResponse.(paras.startingLum{1}).full,'r','linewidth',2);
    plot(timePts,mResponse.(paras.startingLum{2}).full,'k','linewidth',2);legend(paras.startingLum); legend boxoff
%     errorbar(timePts,mResponse.(paras.startingLum{1}).full, std(response.(paras.startingLum{1}))/sqrt(3),'r','linewidth',2);
%     errorbar(timePts,mResponse.(paras.startingLum{2}).full, std(response.(paras.startingLum{2}))/sqrt(3),'k','linewidth',2);legend(paras.startingLum); legend boxoff
    box off;
    
    figure('color','w','position',[100 300 1200 400]);
    for i=1:numel(backgroundList)
        subplot(2,1,i); hold all; 
        plot(mResponse.(paras.startingLum{i}).loop1); 
        plot(mResponse.(paras.startingLum{i}).loop2); 
        legend('loop 1', 'loop 2');
        title(paras.startingLum{i});
    end 
    % compute the PSTH difference for loop1 and loop2. in older version
    % summary, [ 1 2 3 4] , 1 is low vs high loop1, 2 is low vs high loop2.
    % 3 is loop1 vs loop2 low, 4 is loop1 vs loop2 high 
    winLen= floor(numel(mResponse.low.loop1)/paras.nWindow); traceDiff=zeros(1,4*paras.nWindow); 
    traceCorr=zeros(1,4*paras.nWindow);  vpCrossPairDist=zeros(1,6*paras.nWindow); vpWithinDist=zeros(2,6*paras.nWindow);        
    for i=1:paras.nWindow
        tpClip=(i-1)*winLen+1: i*winLen;
        traceDiff(i)=sqrt(sum((mResponse.low.loop1(tpClip)-mResponse.high.loop1(tpClip)).^2))./...
            sqrt(sum((mResponse.low.loop1(tpClip)+mResponse.high.loop1(tpClip)).^2));
        traceDiff(i+paras.nWindow)=sqrt(sum((mResponse.low.loop2(tpClip)-mResponse.high.loop2(tpClip)).^2))./...
            sqrt(sum((mResponse.low.loop2(tpClip)+mResponse.high.loop2(tpClip)).^2));
        traceDiff(i+2*paras.nWindow)=sqrt(sum((mResponse.low.loop1(tpClip)-mResponse.low.loop2(tpClip)).^2))./...
            sqrt(sum((mResponse.low.loop1(tpClip)+mResponse.low.loop2(tpClip)).^2));
        traceDiff(i+3*paras.nWindow)=sqrt(sum((mResponse.high.loop1(tpClip)-mResponse.high.loop2(tpClip)).^2))./...
            sqrt(sum((mResponse.high.loop1(tpClip)+mResponse.high.loop2(tpClip)).^2));    
        % compute the correlation coeff difference for loop1 and loop2.
        tp=corrcoef(mResponse.low.loop1(tpClip),mResponse.high.loop1(tpClip)); traceCorr(i)=tp(1,2);
        tp=corrcoef(mResponse.low.loop2(tpClip),mResponse.high.loop2(tpClip)); traceCorr(i+paras.nWindow)=tp(1,2);
        tp=corrcoef(mResponse.low.loop1(tpClip),mResponse.low.loop2(tpClip)); traceCorr(i+2*paras.nWindow)=tp(1,2);
        tp=corrcoef(mResponse.high.loop1(tpClip),mResponse.high.loop2(tpClip)); traceCorr(i+3*paras.nWindow)=tp(1,2);
        if strcmp(paras.recType,'extracellular')
            % compute the VP distance
            tpSpt1=cellfun(@(x)  x(x>=paras.downsample*(tpClip(1)+prePts+clipPts) & x<=paras.downsample*(tpClip(end)+prePts+clipPts)), spt.low,'UniformOutput',false);
            tpSpt2=cellfun(@(x)  x(x>=paras.downsample*(tpClip(1)+prePts+clipPts) & x<=paras.downsample*(tpClip(end)+prePts+clipPts)), spt.high,'UniformOutput',false);
            temp=ComputePairwiseSpikeDistances(tpSpt1,tpSpt2,paras.qCost(1)); vpCrossPairDist(i)=temp.mean;
            temp=ComputePairwiseSpikeDistances(tpSpt1,tpSpt2,paras.qCost(2)); vpCrossPairDist(i+2*paras.nWindow)=temp.mean;
            temp=ComputePairwiseSpikeDistances(tpSpt1,tpSpt2,paras.qCost(3)); vpCrossPairDist(i+4*paras.nWindow)=temp.mean;

            if numel(tpSpt1)>=2 && numel(tpSpt2)>=2
                temp=withinTrialsSpikeDistance(tpSpt1,paras.qCost(1)); vpWithinDist(1,i)=temp.mean;
                temp=withinTrialsSpikeDistance(tpSpt2,paras.qCost(1)); vpWithinDist(2,i)=temp.mean;
                temp=withinTrialsSpikeDistance(tpSpt1,paras.qCost(2)); vpWithinDist(1,i+2*paras.nWindow)=temp.mean;
                temp=withinTrialsSpikeDistance(tpSpt2,paras.qCost(2)); vpWithinDist(2,i+2*paras.nWindow)=temp.mean;
                temp=withinTrialsSpikeDistance(tpSpt1,paras.qCost(3)); vpWithinDist(1,i+4*paras.nWindow)=temp.mean;
                temp=withinTrialsSpikeDistance(tpSpt2,paras.qCost(3)); vpWithinDist(2,i+4*paras.nWindow)=temp.mean;
            end
            tpSpt1=cellfun(@(x)  x(x>=paras.downsample*(tpClip(1)+prePts+clipPts+stimPts/2) & x<=paras.downsample*(tpClip(end)+prePts+clipPts+stimPts/2)), spt.low,'UniformOutput',false);
            tpSpt2=cellfun(@(x)  x(x>=paras.downsample*(tpClip(1)+prePts+clipPts+stimPts/2) & x<=paras.downsample*(tpClip(end)+prePts+clipPts+stimPts/2)), spt.high,'UniformOutput',false);
            temp=ComputePairwiseSpikeDistances(tpSpt1,tpSpt2,paras.qCost(1)); vpCrossPairDist(i+paras.nWindow)=temp.mean;
            temp=ComputePairwiseSpikeDistances(tpSpt1,tpSpt2,paras.qCost(2)); vpCrossPairDist(i+3*paras.nWindow)=temp.mean;
            temp=ComputePairwiseSpikeDistances(tpSpt1,tpSpt2,paras.qCost(3)); vpCrossPairDist(i+5*paras.nWindow)=temp.mean;
            % compute within trial distance , thus variablity over time
            if numel(tpSpt1)>=2 && numel(tpSpt2)>=2
                temp=withinTrialsSpikeDistance(tpSpt1,paras.qCost(1)); vpWithinDist(1,i+paras.nWindow)=temp.mean;
                temp=withinTrialsSpikeDistance(tpSpt2,paras.qCost(1)); vpWithinDist(2,i+paras.nWindow)=temp.mean;
                temp=withinTrialsSpikeDistance(tpSpt1,paras.qCost(2)); vpWithinDist(1,i+3*paras.nWindow)=temp.mean;
                temp=withinTrialsSpikeDistance(tpSpt2,paras.qCost(2)); vpWithinDist(2,i+3*paras.nWindow)=temp.mean;
                temp=withinTrialsSpikeDistance(tpSpt1,paras.qCost(3)); vpWithinDist(1,i+5*paras.nWindow)=temp.mean;
                temp=withinTrialsSpikeDistance(tpSpt2,paras.qCost(3)); vpWithinDist(2,i+5*paras.nWindow)=temp.mean;
            end
        end
    end
    if  ~strcmp(paras.recType,'extracellular')
        vpCrossPairDist=[];   vpWithinDist=[];
    else
        if numel(tpSpt1)<2 || numel(tpSpt2)<2
            vpWithinDist=[];
        end
    end
    stats.traceDiff=traceDiff;
    stats.traceCorr=traceCorr; 
    stats.vpCrossPairDist=vpCrossPairDist; 
    stats.vpWithinDist=vpWithinDist;
    
    % visualize the raster plot of spike times
    if strcmp(paras.recType,'extracellular')
        try
            figure('color','w','position',[-1900 0 1800 900]);
            subplot(5,1,[1 2]);
            plotSpikeRaster(spt.low,'PlotType','vertline'); axis off;
            subplot(5,1,[3 4]);
            plotSpikeRaster(spt.high,'PlotType','vertline'); axis off;
            subplot(5,1,5);
            hold all;
            plot(timePts,mResponse.low.full,'k','linewidth',2);
            plot(timePts,mResponse.high.full,'r','linewidth',2); legend('low','high'); legend boxoff;
            xlim([timePts(1) timePts(end)]);
        end
    end
end

if numel(selectedNodes)>1
    figure('color','w','position',[20 20 1800 900]);
    colors=pmkmp(3,'IsoL');
    for i=1:numel(backgroundList)
        subplot(numel(backgroundList),1,i); hold all; 
        for node=1:numel(selectedNodes)
             plot(timePts,nodeResponse{node}.(paras.startingLum{i}).full,'color',colors(node,:),'linewidth',2);
             nodeLgds{node}=selectedNodes{node}.epochList.firstValue.protocolSettings('source:type');
        end
        title(['starting background' ' ' paras.startingLum{i}]);  legend(nodeLgds); legend boxoff;
    end
end

