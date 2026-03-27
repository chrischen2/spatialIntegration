function stats = analyzePPSpotsMean(selectedNodes, paras)
% analyzePPSpotsMean processes one or more nodes (in a cell array) and returns a stats structure.
% Each element of selectedNodes is a structure equivalent to the original selectedNode.
%
% The function calculates response matrices, processes the traces according to the parameters,
% computes mean traces and amplitude values, and then – if more than one node is provided –
% creates additional figures to overlay the mean traces (one subplot per flash contrast)
% and to compare amplitude ratios across nodes for each contrast.
%
% Usage:
%   stats = analyzePPSpotsMean({node1, node2, ...}, paras)

% If not a cell, wrap into a cell array for consistency.
if ~iscell(selectedNodes)
    selectedNodes = {selectedNodes};
end

nNodes = numel(selectedNodes);
stats = struct();
stats.nodes = cell(nNodes,1);
allContrastArray = [];
allIntervalArray = [];

% Process each node individually.
for j = 1:nNodes
    node = selectedNodes{j};
    
    % Get Response Matrix and Contrast Values
    resMat = riekesuite.getResponseMatrix(node.epochList, 'Amp1');
    stepContrst = zeros(1, node.epochList.length);
    pulseIntervals = zeros(1, node.epochList.length);
    for i = 1:length(stepContrst)
        stepContrst(i) = node.epochList.elements(i).protocolSettings('currentStepContrast');
        pulseIntervals(i) = node.epochList.elements(i).protocolSettings('pulseIntervals');
    end
    
    paras.epochRange = 1:size(resMat, 1);
    if ~paras.psth
        paras.epochRange = find(mean(resMat, 2) < 0);
    end
    resMat = resMat(paras.epochRange, :);
    stepContrst = stepContrst(paras.epochRange);
    pulseIntervals = pulseIntervals(paras.epochRange);

    % Remove Specific Trials if Requested using paras.rmRep
    if isfield(paras, 'rmRep') && ~isempty(paras.rmRep)
        % Remove specified row indices.
        resMat(paras.rmRep, :) = [];
        stepContrst(paras.rmRep) = [];
        pulseIntervals(paras.rmRep) = [];
    end

    % Process the Trace According to Node Settings
    if node.parent.parent.splitValue
        % If splitValue is truthy, subtract a moving median and use a spike detection method.
        for i = 1:size(resMat, 1)
            resMat(i, :) = resMat(i, :) - movmedian(resMat(i, :), 100);
        end
        [spikeTimes, ~, ~] = SpikeDetectorNew(resMat, 'thresholdSpikeFactor', paras.spikeTh);
        trace = spikeTimeToPSTH(resMat, spikeTimes, paras.psthSigma, paras.sampleRate);
    else
        % Otherwise, smooth the trace and perform baseline adjustment.
        for i = 1:size(resMat, 1)
            resMat(i, :) = smooth(resMat(i, :), 100);
        end
        trace = resMat - repmat(mean(resMat(:, 1:paras.prePts), 2), 1, size(resMat, 2));
    end
    
    % Compute Mean Trace for Each Flash Contrast and Build Stimulus Trace
    contrastArray = unique(stepContrst);
    intervalArray = unique(pulseIntervals);
    % Store (or update) the contrast array for later multi-node plotting. It is assumed that the
    % contrastArray is the same for all nodes. If not, an intersection is used.
    if j == 1
        allContrastArray = contrastArray;
        allIntervalArray = intervalArray;
    else
        if ~isequal(allContrastArray, contrastArray)
            warning('Contrast arrays differ between nodes. Using intersection of contrast values.');
            allContrastArray = intersect(allContrastArray, contrastArray);
        end
        if ~isequal(allIntervalArray, intervalArray)
            warning('Interval arrays differ between nodes. Using intersection of interval values.');
            allIntervalArray = intersect(allIntervalArray, intervalArray);
        end
    end
    
    mTrace = zeros(length(contrastArray), length(intervalArray), size(trace, 2));
    stimTrace = ones(length(contrastArray), length(intervalArray), size(trace, 2));
    for c = 1:length(contrastArray)
        for i = 1:length(intervalArray)
            stepIndex = find(contrastArray(c) == stepContrst & intervalArray(i) == pulseIntervals);
            % Compute mean trace instead of sum/numel
            mTrace(c, i, :) = mean(trace(stepIndex, :), 1);
            stimTrace(c, i, paras.prePts+1 : paras.prePts+paras.flashPts) = 1 + paras.flashContrast;
            stimTrace(c, i, paras.prePts+paras.flashPts+1 : paras.prePts+paras.flashPts+intervalArray(i)) = 1 + contrastArray(c);
            stimTrace(c, i, paras.prePts+paras.flashPts+intervalArray(i)+1 : paras.prePts+paras.flashPts*2+intervalArray(i)) = 1 + paras.flashContrast;
        end
    end
    % Calculate Amplitudes for Pulse 1 and Pulse 2, then the Amplitude Ratio
    amp1 = zeros(length(contrastArray), length(intervalArray));
    amp2 = zeros(length(contrastArray), length(intervalArray));
    for c = 1:length(contrastArray)
        for i = 1:length(intervalArray)
            window1 = paras.prePts : paras.prePts+paras.flashPts;
            if paras.psth == 1
                amp1(c, i) = max(mTrace(c, i, window1));
            else
                amp1(c, i) = min(mTrace(c, i, window1));
            end
            
            window2 = paras.prePts+paras.flashPts+intervalArray(i) : paras.prePts+paras.flashPts*2+intervalArray(i);
            if paras.psth == 1
                amp2(c, i) = max(mTrace(c, i, window2));
            else
                amp2(c, i) = min(mTrace(c, i, window2));
            end
        end
    end
    % Use absolute values so that amplitudes are positive.
    amp1 = abs(amp1);
    amp2 = abs(amp2);
    pulseRatio = amp2 ./ amp1;
    
    % Save Statistics for This Node
    nodeStats = struct();
    nodeStats.splitValue = node.splitValue; % used as node name in aggregated figures
    nodeStats.contrastArray = contrastArray;
    nodeStats.intervalArray = intervalArray;
    nodeStats.mTrace = mTrace;
    nodeStats.stimTrace = stimTrace;
    nodeStats.amp1 = amp1;
    nodeStats.amp2 = amp2;
    nodeStats.pulseRatio = pulseRatio;
    nodeStats.traceTime = (1:size(trace, 2)) / paras.sampleRate;  
    stats.nodes{j} = nodeStats;

    % (Optional) Individual Figure for This Node: if desired you can uncomment this section

    f = figure('color', 'w', 'position', [200 300 800 1200]);

    % Overlay plots for multiple pulse intervals for each contrast
    ax(1) = subplot(3, 1, 1);
    hold on;
    colors = pmkmp(max(2,numel(intervalArray)), 'IsoL');
    for c = 1
        for i = 1:length(intervalArray)
            plot((1:size(mTrace, 3)) / paras.sampleRate, squeeze(mTrace(c, i, :)), ...
                'color', colors(i, :), 'linewidth', 2, 'DisplayName', sprintf('Interval: %g', intervalArray(i)));
        end
        title(sprintf('Overlay for Contrast: %g', contrastArray(c)));
        xlabel('Time (s)');
        ylabel('Mean Trace (pA)');
        legend('show');
    end
    initFig(ax(1), 'Time', 'pA');

    % Overlay plots for multiple contrasts for each pulse interval
    ax(2) = subplot(3, 1, 2);
    hold on;
    colors = pmkmp(numel(contrastArray), 'IsoL');
    for i = 1
        for c = 1:length(contrastArray)
            plot((1:size(mTrace, 3)) / paras.sampleRate, squeeze(mTrace(c, i, :)), ...
                'color', colors(c, :), 'linewidth', 2, 'DisplayName', sprintf('Contrast: %g', contrastArray(c)));
        end
        title(sprintf('Overlay for Interval: %g', intervalArray(i)));
        xlabel('Time (s)');
        ylabel('Mean Trace (pA)');
        legend('show');
    end
    initFig(ax(2), 'Time', 'pA');

    % Plot pulse ratios
    ax(3) = subplot(3, 1, 3);
    hold on;
    for c = 1:length(contrastArray)
        plot(intervalArray, pulseRatio(c, :), 's-', 'LineWidth', 2, ...
            'DisplayName', sprintf('Contrast: %g', contrastArray(c)));
    end
    ylabel('Amplitude Ratio');
    xlabel('Pulse Interval');
    title('Amplitude Ratio vs Pulse Interval');
    legend('Location', 'Best');
    initFig(ax(6), 'Interval', 'Pulse Ratio');

end
    
 

% If there is more than one node, create aggregated figures.
if nNodes > 1
    %---- Overlay Figure: For each flash contrast and interval, overlay the mean traces from all nodes ----%
    fOverlay = figure('color', 'w', 'position', [100 100 800 600]);
    nContrast = numel(allContrastArray);
    nInterval = numel(allIntervalArray);
    % Use a loop to create one subplot per contrast value.
    for c = 1:nContrast
        for i = 1:nInterval
            subplot(nContrast, nInterval, (c-1)*nInterval + i);
            hold on;
            for j = 1:nNodes
                nodeStats = stats.nodes{j};
                % Find the index corresponding to the current contrast and interval in the node's arrays.
                idxContrast = find(nodeStats.contrastArray == allContrastArray(c));
                idxInterval = find(nodeStats.intervalArray == allIntervalArray(i));
                if ~isempty(idxContrast) && ~isempty(idxInterval)
                    % Plot the mean trace of the current contrast and interval.
                    plot(nodeStats.traceTime, squeeze(nodeStats.mTrace(idxContrast, idxInterval, :)), 'LineWidth', 2, ...
                         'DisplayName', num2str(nodeStats.splitValue));
                end
            end
            title(sprintf('Contrast: %g, Interval: %g', allContrastArray(c), allIntervalArray(i)));
            xlabel('Time (s)');
            ylabel('Mean Trace (pA)');
            legend('show');
            hold off;
        end
    end
end


