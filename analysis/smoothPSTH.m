function [psth,spikeBinary,emptyTrial,spikeTimes] = smoothPSTH(resMat,psthSigma, sampleRate,th)
%for smoothed PSTH...
filterSigma = (psthSigma / 1e3) * sampleRate; % 5 msec -> datapoints
newFilt = gaussFilter1D(filterSigma);
[spikeTimes,~,~,emptyTrial]=SpikeDetectorNew(resMat, 'thresholdSpikeFactor', th,'CheckDetection', false);

spikeBinary= zeros(numel(spikeTimes),size(resMat,2));
psth= zeros(numel(spikeTimes),size(resMat,2));
for j = 1:size(spikeBinary,1)
    if size(resMat,1)>1
        spikeBinary(j,spikeTimes{j} )=1;
        spikeTimes{j}=spikeTimes{j}/sampleRate;
    else
        spikeBinary(j,spikeTimes)=1;
        spimeTimes=spikeTimes/sampleRate;
    end
    psth(j,:) =  sampleRate*conv(spikeBinary(j,:),newFilt.amp,'same');
end
% [~, sInd]=rmoutliers(std(psth,[],2));
% if size(spikeTimes,2)<3 && size(spikeTimes,2)>1% if trials smaller than 4, thens how spike detection
%     figure('position',[50 100 1800 300]);
%     plot(resMat(1,:),'k'); hold all;
%     scatter(spikeTimes{1}, resMat(1,spikeTimes{1}));
% end

