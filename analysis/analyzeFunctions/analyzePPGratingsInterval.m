function stats = analyzePPGratingsInterval(selectedNode, paras)
    resMat = riekesuite.getResponseMatrix(selectedNode.epochList, 'Amp1');
    currentInterval = zeros(1, selectedNode.epochList.length);
    for i = 1:length(currentInterval)
        currentInterval(i) = selectedNode.epochList.elements(i).protocolSettings('currentInterval');
    end
    paras.epochRange = 1:size(resMat, 1);
    if ~paras.psth
        paras.epochRange = find(mean(resMat, 2) < 0);
    end
    resMat = resMat(paras.epochRange, :);
    currentInterval = currentInterval(paras.epochRange);

    if selectedNode.splitValue
        for i = 1:size(resMat, 1)
            resMat(i, :) = resMat(i, :) - movmedian(resMat(i, :), 100);
        end
        [spikeTimes, ~, ~] = SpikeDetectorNew(resMat, 'thresholdSpikeFactor', paras.spikeTh);
        trace = spikeTimeToPSTH(resMat, spikeTimes, paras.psthSigma, paras.sampleRate);
    else
        for i = 1:size(resMat, 1)
            resMat(i, :) = smooth(resMat(i, :), 100);
        end
        trace = resMat - repmat(mean(resMat(:, 1:paras.prePts), 2), 1, size(resMat, 2)); % baseline adjustment
    end

    intervalArray = unique(currentInterval);
    mTrace = zeros(length(intervalArray), size(trace, 2));
    stimTrace = ones(length(intervalArray), size(trace, 2));
    for i = 1:length(intervalArray)
        intervalIndex = find(intervalArray(i) == currentInterval);
        mTrace(i, :) = mean(trace(intervalIndex, :), 1);
        stimTrace(i, paras.prePts+1:paras.prePts+paras.flashPts) = 1 + 0.01;
        stimTrace(i, paras.prePts+paras.flashPts+1:paras.prePts+paras.flashPts+timeToPts(intervalArray(i), paras.sampleRate)) = 1 + paras.stepContrast;
        stimTrace(i, paras.prePts+paras.flashPts+timeToPts(intervalArray(i), paras.sampleRate)+1:paras.prePts+paras.flashPts*2+timeToPts(intervalArray(i), paras.sampleRate)) = 1 + 0.01;
    end

    % Calculate amplitudes and peak times for four pulses
    amp1 = zeros(1, length(intervalArray));
    amp2 = zeros(1, length(intervalArray));
    amp3 = zeros(1, length(intervalArray));
    amp4 = zeros(1, length(intervalArray));
    peakTime1 = zeros(1, length(intervalArray));
    peakTime2 = zeros(1, length(intervalArray));
    peakTime3 = zeros(1, length(intervalArray));
    peakTime4 = zeros(1, length(intervalArray));

    for i = 1:length(intervalArray)
        % Pulse 1 amplitude and peak time
        window1 = paras.prePts:paras.prePts+paras.flashPts;
        % Pulse 2 amplitude and peak time
        window2 = paras.prePts+paras.flashPts:paras.prePts+paras.flashPts*2;
        % Pulse 3 amplitude and peak time
        window3 = paras.prePts+timeToPts(intervalArray(i), paras.sampleRate)+paras.flashPts:paras.prePts+timeToPts(intervalArray(i), paras.sampleRate)+paras.flashPts*2;
        % Pulse 4 amplitude and peak time
        window4 = paras.prePts+timeToPts(intervalArray(i), paras.sampleRate)+paras.flashPts*2:paras.prePts+timeToPts(intervalArray(i), paras.sampleRate)+paras.flashPts*3;

        if paras.psth == 1
            baseline = mean(mTrace(i, 1:paras.prePts));
            [amp1(i), peakIndex1] = max(mTrace(i, window1) - baseline);
            [amp2(i), peakIndex2] = max(mTrace(i, window2) - baseline);
            [amp3(i), peakIndex3] = max(mTrace(i, window3) - baseline);
            [amp4(i), peakIndex4] = max(mTrace(i, window4) - baseline);
        else
            [amp1(i), peakIndex1] = min(mTrace(i, window1));
            [amp2(i), peakIndex2] = min(mTrace(i, window2));
            [amp3(i), peakIndex3] = min(mTrace(i, window3));
            [amp4(i), peakIndex4] = min(mTrace(i, window4));
        end
        
        % Calculate relative peak times within their windows
        peakTime1(i) = (peakIndex1 - 1) / length(window1);
        peakTime2(i) = (peakIndex2 - 1) / length(window2);
        peakTime3(i) = (peakIndex3 - 1) / length(window3);
        peakTime4(i) = (peakIndex4 - 1) / length(window4);
    end

    % Use absolute values for amplitudes
    amp1 = abs(amp1);
    amp2 = abs(amp2);
    amp3 = abs(amp3);
    amp4 = abs(amp4);

    % Calculate ratios
    ratio1 = amp1 ./ amp2;
    ratio2 = amp3 ./ amp4;

    % Plotting
    f = figure('color', 'w', 'position', [200 300 600 1200]);
    
    % Full trace plot
    ax(1) = subplot(6, 1, 1);
    hold all;
    colors = pmkmp(numel(intervalArray), 'IsoL');
    for i = 1:size(mTrace, 1)
        plot((1:size(mTrace, 2))/paras.sampleRate, mTrace(i, :), 'color', colors(i, :), 'linewidth', 3);
    end
    initFig(ax(1), 'Time (s)', 'Response');
    title('Full Trace');

    % Stimulus plot
    ax(2) = subplot(6, 1, 2);
    hold all;
    for i = 1:size(mTrace, 1)
        plot((1:size(stimTrace, 2))/paras.sampleRate, stimTrace(i, :), 'color', colors(i, :), 'linewidth', 3);
    end
    initFig(ax(2), 'Time (s)', 'Contrast');
    title('Stimulus');

    % First segment zoom
    ax(3) = subplot(6, 1, 3);
    hold all;
    for i = 1:size(mTrace, 1)
        plot(mTrace(i, paras.prePts:paras.prePts+paras.flashPts*2-1), ...
            'color', colors(i, :), 'linewidth', 3);
    end
    initFig(ax(3), 'Time (s)', 'Response');
    title('First Segment Zoom');

    % Second segment zoom
    ax(4) = subplot(6, 1, 4);
    hold all;
    for i = 1:size(mTrace, 1)
        secondSegmentStart = paras.prePts + timeToPts(intervalArray(i), paras.sampleRate) + paras.flashPts;
        plot(mTrace(i, secondSegmentStart:secondSegmentStart+paras.flashPts*2-1), ...
            'color', colors(i, :), 'linewidth', 3);
    end
    initFig(ax(4), 'Time (s)', 'Response');
    title('Second Segment Zoom');
    % Amplitude ratios vs interval
    ax(5) = subplot(6, 1, 5);
    hold all;
    plot(intervalArray, ratio1, 'o-', 'LineWidth', 2, 'DisplayName', 'Pulse 1/2');
    plot(intervalArray, ratio2, 's-', 'LineWidth', 2, 'DisplayName', 'Pulse 3/4');
    initFig(ax(5), 'Interval (s)', 'Amplitude Ratio');
    title('Amplitude Ratios vs Interval');
    legend('Location', 'Best');

    % Peak time vs interval
    ax(6) = subplot(6, 1, 6);
    hold all;
    plot(intervalArray, peakTime1, 'o-', 'LineWidth', 2, 'DisplayName', 'Pulse 1');
    plot(intervalArray, peakTime2, 's-', 'LineWidth', 2, 'DisplayName', 'Pulse 2');
    plot(intervalArray, peakTime3, '^-', 'LineWidth', 2, 'DisplayName', 'Pulse 3');
    plot(intervalArray, peakTime4, 'd-', 'LineWidth', 2, 'DisplayName', 'Pulse 4');
    initFig(ax(6), 'Interval (s)', 'Relative Peak Time');
    title('Peak Time vs Interval');
    legend('Location', 'Best');

    % Save output into stats structure
    stats = struct();
    stats.intervalArray = intervalArray;
    stats.amp1 = amp1;
    stats.amp2 = amp2;
    stats.amp3 = amp3;
    stats.amp4 = amp4;
    stats.peakTime1 = peakTime1;
    stats.peakTime2 = peakTime2;
    stats.peakTime3 = peakTime3;
    stats.peakTime4 = peakTime4;
    stats.ratio1 = ratio1;
    stats.ratio2 = ratio2;
    stats.mTrace = mTrace;
    stats.stimTrace = stimTrace;
end



function pts = timeToPts(time, sampleRate)
pts = round(time * sampleRate/1e3);
end