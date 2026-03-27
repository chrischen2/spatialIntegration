function [psth] = spikeTimeToPSTH(resMat,spikeTimes,psthSigma, sampleRate)
%for smoothed PSTH...
filterSigma = (psthSigma / 1e3) * sampleRate; % 5 msec -> datapoints
newFilt = gaussFilter1D(filterSigma);
spikeBinary= zeros(size(resMat));
psth= zeros(size(resMat));
for j = 1:size(spikeBinary,1)
    if size(resMat,1)>1
        spikeBinary(j,spikeTimes{j})=1;
    else
        spikeBinary(j,spikeTimes)=1;
    end
    psth(j,:) =  sampleRate*conv(spikeBinary(j,:),newFilt.amp,'same');
end
