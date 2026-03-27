function [psth, emptyTrial] = spikeTimesToPSTH(spikeTimes, epoch_len, maxTime, psthSigma, sampleRate)
% For smoothed PSTH...
% input would be M by 1 cell array
filterSigma = (psthSigma / 1e3) * sampleRate; % 5 msec -> datapoints
newFilt = gaussFilter1D(filterSigma);

numTrials = numel(spikeTimes);
spikeBinary = cell(1, numTrials);
psth = cell(1, numTrials);
emptyTrial = false(1, numTrials);
edges=0:epoch_len:maxTime; 
for j = 1:numTrials
    spikeBinary{j} = zeros(1, maxTime);
    epoch_spike_count=histcounts(spikeTimes{j}, edges);
    if numel(spikeTimes{j})<20 || numel(spikeTimes{j})>400 || std(epoch_spike_count)/mean(epoch_spike_count)>1 ...
            || sum(epoch_spike_count<2)>=5
        emptyTrial(j) = true;
    else
        spikeTimes{j}=ceil(spikeTimes{j}); spikeTimes{j}(spikeTimes{j}==0)=1;  spikeTimes{j}(spikeTimes{j}>maxTime)=maxTime;
        spikeBinary{j}(spikeTimes{j}) = 1;

    end
    psth{j} = sampleRate * conv(spikeBinary{j}, newFilt.amp, 'same');
end

psth = cell2mat(psth');

% [~, sInd]=rmoutliers(mean(psth,2));
% emptyTrial(sInd) = true;
