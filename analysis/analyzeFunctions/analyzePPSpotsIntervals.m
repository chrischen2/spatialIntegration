function  [stats,paras] = analyzePPSpotsIntervals(selectedNode,paras)
resMat=riekesuite.getResponseMatrix(selectedNode.epochList,'Amp1');
currentInterval = zeros(1, selectedNode.epochList.length);

for i = 1:length(currentInterval)
    currentInterval(i) = selectedNode.epochList.elements(i).protocolSettings('currentInterval');
end

paras.epochRange=1:size(resMat,1);

if ~paras.psth
    % paras.epochRange=find(mean(resMat,2)<0); paras.recType='exc'; paras.offset=0;
    paras.epochRange=find(mean(resMat,2)>0); paras.recType='inh'; paras.offset=paras.flashPts; 
else 
    paras.recType='spike'; paras.offset=0;
end
paras.epochRange(paras.rmRep)=[];
resMat=resMat(paras.epochRange,:);
currentInterval = currentInterval(paras.epochRange);

% Apply bandpass filter between 1 and 60 Hz
% Design bandpass filter
if paras.filterRes
    lowFreq = 1;   % Lower cutoff frequency (Hz)
    highFreq = 60; % Upper cutoff frequency (Hz)
    nyquist = paras.sampleRate/2;
    [b, a] = butter(3, [lowFreq/nyquist, highFreq/nyquist], 'bandpass');

    % Apply filter to each trace
    filteredResMat = zeros(size(resMat));
    for i = 1:size(resMat, 1)
        filteredResMat(i, :) = filtfilt(b, a, double(resMat(i, :)));
    end
    % Replace original matrix with filtered matrix
    resMat = filteredResMat;
end

if paras.psth
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

intervalArray = unique(currentInterval);
mTrace = zeros(length(intervalArray), size(trace, 2));
stimTrace = ones(length(intervalArray), size(trace, 2));
timeToPts=@(x,sampleRate) x/1e3*sampleRate; 
for i = 1:length(intervalArray)
    mTrace(i, :) = mean(trace(intervalArray(i) == currentInterval, :), 1);
    stimTrace(i, paras.prePts+1:paras.prePts+paras.flashPts) = 1 + paras.flashContrast;
    stimTrace(i, paras.prePts+paras.flashPts+1:paras.prePts+paras.flashPts+timeToPts(intervalArray(i), paras.sampleRate)) = 1 + paras.stepContrast;
    stimTrace(i, paras.prePts+paras.flashPts+timeToPts(intervalArray(i), paras.sampleRate)+1:paras.prePts+paras.flashPts*2+ ...
        timeToPts(intervalArray(i), paras.sampleRate)) = 1 + paras.flashContrast;
end

peakTime1 = zeros(1, length(intervalArray));
peakTime2 = zeros(1, length(intervalArray));
peakIndex1 = zeros(1, length(intervalArray));  % Store the actual indices for later use
peakIndex2 = zeros(1, length(intervalArray));  % Store the actual indices for later use
secondPulseBaselines = zeros(length(intervalArray), 1);

for i = 1:length(intervalArray)
    % Pulse 1 amplitude and peak time
    window1 = paras.prePts+paras.offset:paras.prePts+paras.flashPts+paras.offset;
    % Pulse 2 amplitude and peak time
    window2 = paras.offset+paras.prePts+paras.flashPts+timeToPts(intervalArray(i), paras.sampleRate):paras.prePts+paras.flashPts*2+timeToPts(intervalArray(i), paras.sampleRate)+paras.offset;
 
    if strcmp(paras.recType,'exc')
        % For non-PSTH case, we don't subtract any baseline for finding peaks
        [amp1(i), peakIndex1(i)] = min(mTrace(i, window1));
        [amp2(i), peakIndex2(i)] = min(mTrace(i, window2));  % Removed baseline subtraction
    else
        baseline = mean(mTrace(i, 1:paras.prePts));
        [amp1(i), peakIndex1(i)] = max(mTrace(i, window1) - baseline);
        [amp2(i), peakIndex2(i)] = max(mTrace(i, window2) - baseline);
    end

    % Calculate relative peak times within their windows
    peakTime1(i) = (peakIndex1(i) - 1) / length(window1)*1e3;
    peakTime2(i) = (peakIndex2(i) - 1) / length(window2)*1e3;
    
    % Adjust peak indices to be relative to the full trace
    peakIndex1(i) = peakIndex1(i) + window1(1) - 1;
    peakIndex2(i) = peakIndex2(i) + window2(1) - 1;

        % Get the actual peak position in the full trace for the second peak
    actual_peak_pos = peakIndex2(i);
    
    % Calculate baseline window positions relative to the second peak
    % Use (2nd_peak_pts-800:2nd_peak_pts-500) as requested
    baseline_start = window2(1);
    baseline_end = window2(1)+300;
    
    % Make sure we don't go out of bounds
    baseline_end = min(baseline_end, size(mTrace, 2));
    
    % Compute baseline as mean of points in the specified range
    secondPulseBaselines(i) = mean(mTrace(i, baseline_start:baseline_end));
    amp2(i)=amp2(i)-secondPulseBaselines(i);
end

f=figure('color','w','position',[200 300 600 1000]);
% Full trace plot
ax(1) = subplot(6, 1, 1);
hold all;
colors = pmkmp(numel(intervalArray), 'IsoL');
for i = 1:size(mTrace, 1)
    plot((1:size(mTrace, 2))/paras.sampleRate, mTrace(i, :), 'color', colors(i, :), 'linewidth', 3);
end
initFig(ax(1), 'Time (s)', 'Response');

% Stimulus plot
ax(2) = subplot(6, 1, 2);
hold all;
for i = 1:size(mTrace, 1)
    plot((1:size(stimTrace, 2))/paras.sampleRate, 100*stimTrace(i, :), 'color', colors(i, :), 'linewidth', 3);
end
initFig(ax(2), 'Time (s)', 'Isom');

% First pulse zoom
ax(3) = subplot(6, 1, 3);
hold all;
for i = 1:size(mTrace, 1)
    plot(mTrace(i, paras.prePts+1+paras.offset:paras.prePts+paras.flashPts+paras.offset), 'color', colors(i, :), 'linewidth', 3);
end
initFig(ax(3), 'Time (samples)', 'Response');

% Second pulse zoom
ax(4) = subplot(6, 1, 4);
hold all;
for i = 1:size(mTrace, 1)
    plot(mTrace(i,paras.offset+paras.prePts+paras.flashPts+timeToPts(intervalArray(i), paras.sampleRate)+1:paras.offset+paras.prePts+paras.flashPts*2+ ...
        timeToPts(intervalArray(i), paras.sampleRate)), 'color', colors(i, :), 'linewidth', 3);
end
initFig(ax(4), 'Time (samples)', 'Response');

% Amplitude ratio vs interval
ax(5) = subplot(6, 1, 5);
hold all;
plot(intervalArray, amp2./amp1, 's-', 'LineWidth', 2);
initFig(ax(5), 'Interval (s)', 'Pulse ratio (P2/P1)');
title('Amplitude Ratio vs Interval');


% Add the new subplot for baseline vs amplitude
ax(6) = subplot(6, 1, 6);
hold all;
scatter(intervalArray,secondPulseBaselines);
initFig(ax(6), 'Baseline', 'Second Pulse Amplitude');
title('Second Pulse Amplitude over time');

% Save output into stats structure
stats = struct();
stats.intervalArray = intervalArray;
stats.amp1 = amp1;
stats.amp2 = amp2;
stats.peakTime1 = peakTime1;
stats.peakTime2 = peakTime2;
stats.secondPulseBaselines = secondPulseBaselines;
stats.mTrace = mTrace;
stats.stimTrace = stimTrace;

end

