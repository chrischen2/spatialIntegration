function  stats = analyzePPGratingsMean(selectedNode,paras)
resMat=riekesuite.getResponseMatrix(selectedNode.epochList,'Amp1');
stepContrst=zeros(1, selectedNode.epochList.length);
for i=1:length(stepContrst)
    stepContrst(i)=selectedNode.epochList.elements(i).protocolSettings('currentStepContrast');
end
paras.epochRange=1:size(resMat,1);
if ~paras.psth
    paras.epochRange=find(mean(resMat,2)<0);
end
%  paras.epochRange=[1:22 24:45];
resMat=resMat(paras.epochRange,:); stepContrst=stepContrst(paras.epochRange);
if selectedNode.splitValue
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

contrastArray=unique(stepContrst);
mTrace=zeros(length(contrastArray), size(trace,2));
stimTrace=ones(length(contrastArray), size(trace,2));
for i=1:length(contrastArray)
    stepIndex=find(contrastArray(i)==stepContrst);
    mTrace(i,:)=sum(trace(stepIndex,:))/numel(stepIndex);
    stimTrace(i,paras.prePts+1:paras.prePts+paras.flashPts)= 1;
    stimTrace(i,paras.prePts+paras.flashPts+1:paras.prePts+paras.flashPts+paras.intervalPts)=1+contrastArray(i);
    stimTrace(i,paras.prePts+paras.flashPts+paras.intervalPts+1:paras.prePts+paras.flashPts*2+paras.intervalPts)=1;
end


    % Calculate amplitudes for four pulses
    amp1 = zeros(1, length(contrastArray));
    amp2 = zeros(1, length(contrastArray));
    amp3 = zeros(1, length(contrastArray));
    amp4 = zeros(1, length(contrastArray));
    for i = 1:length(contrastArray)
        % Pulse 1 amplitude
        window1 = paras.prePts:paras.prePts+paras.flashPts;
        % Pulse 2 amplitude
        window2 = paras.prePts+paras.flashPts:paras.prePts+paras.flashPts*2;
        % Pulse 3 amplitude
        window3 = paras.prePts+paras.intervalPts+paras.flashPts:paras.prePts+paras.intervalPts+paras.flashPts*2;
        % Pulse 4 amplitude
        window4 = paras.prePts+paras.intervalPts+paras.flashPts*2:paras.prePts+paras.intervalPts+paras.flashPts*3;

        if paras.psth == 1
            baseline = mean(mTrace(i, 1:paras.prePts));
            amp1(i) = max(mTrace(i, window1))-baseline;
            amp2(i) = max(mTrace(i, window2))-baseline;
            amp3(i) = max(mTrace(i, window3))-baseline;
            amp4(i) = max(mTrace(i, window4))-baseline;
        else
            amp1(i) = min(mTrace(i, window1));
            amp2(i) = min(mTrace(i, window2));
            amp3(i) = min(mTrace(i, window3));
            amp4(i) = min(mTrace(i, window4));
        end
    end
 % Use absolute values for amplitudes
    amp1 = abs(amp1);
    amp2 = abs(amp2);
    amp3 = abs(amp3);
    amp4 = abs(amp4);
    % Calculate ratios
    ratio1 = amp1 ./ amp2;
    ratio2 = amp3 ./ amp4;

    f = figure('color', 'w', 'position', [200 300 600 1000]);
    
    % Full trace plot
    ax(1) = subplot(5,1,1);
    hold all;
    colors = pmkmp(numel(contrastArray), 'IsoL');
    for i = 1:size(mTrace, 1)
        plot((1:size(mTrace,2))/paras.sampleRate, mTrace(i,:), 'color', colors(i,:), 'linewidth', 3);
    end
    initFig(ax(1), 'Time', 'pA');
    title('Full Trace');

    % Stimulus plot
    ax(2) = subplot(5,1,2);
    hold all;
    for i = 1:size(mTrace,1)
        plot((1:size(stimTrace,2))/paras.sampleRate, stimTrace(i,:), 'color', colors(i,:), 'linewidth', 3);
    end
    initFig(ax(2), 'Time', 'Contrast');
    title('Stimulus');

    % First segment zoom
    ax(3) = subplot(5,1,3);
    hold all;
    for i = 1:size(mTrace,1)
        plot((paras.prePts:paras.prePts+paras.flashPts*2)/paras.sampleRate, ...
             mTrace(i, paras.prePts:paras.prePts+paras.flashPts*2), ...
             'color', colors(i,:), 'linewidth', 3);
    end
    initFig(ax(3), 'Time', 'pA');
    title('First Segment Zoom');

    % Second segment zoom
    ax(4) = subplot(5,1,4);
    hold all;
    for i = 1:size(mTrace,1)
        plot((paras.prePts+paras.intervalPts+paras.flashPts:paras.prePts+paras.intervalPts+paras.flashPts*3)/paras.sampleRate, ...
             mTrace(i, paras.prePts+paras.intervalPts+paras.flashPts:paras.prePts+paras.intervalPts+paras.flashPts*3), ...
             'color', colors(i,:), 'linewidth', 3);
    end
    initFig(ax(4), 'Time', 'pA');
    title('Second Segment Zoom');

% Amplitude ratios vs contrast
    ax(5) = subplot(5,1,5);
    hold all;
    plot(abs(contrastArray), ratio1, 'o-', 'LineWidth', 2, 'DisplayName', 'Pulse 1/2');
    plot(abs(contrastArray), ratio2, 's-', 'LineWidth', 2, 'DisplayName', 'Pulse 3/4');
    xlabel('Contrast');
    ylabel('Amplitude Ratio');
    title('Amplitude Ratios vs Contrast');
    legend('Location', 'Best');

    % Save output into stats structure
    stats = struct();
    stats.contrastArray = contrastArray;
    stats.amp1 = amp1;
    stats.amp2 = amp2;
    stats.amp3 = amp3;
    stats.amp4 = amp4;
    stats.ratio1 = ratio1;  % Add ratio1 to stats
    stats.ratio2 = ratio2;  % Add ratio2 to stats
    stats.mTrace = mTrace;
    stats.stimTrace = stimTrace;
end