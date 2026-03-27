function output = analyzeContrastSpots(selectedNodes, paras)
% Input:
% selectedNodes - array of nodes
% paras - structure with parameters

% Initialize figure for overlay plotting
f = figure('color', 'w', 'position', [200 300 1000 750]);
ax(1) = subplot(2, 2, 1);
hold(ax(1), 'on');
ax(2) = subplot(2, 2, 2);
hold(ax(2), 'on');
ax(3) = subplot(2, 2, [3, 4]);
hold(ax(3), 'on');

colors = ['r', 'k', 'b'];
baseline = zeros(1, length(selectedNodes));

for nodeIdx = 1:length(selectedNodes)
    selectedNode = selectedNodes{nodeIdx};
    resMat = riekesuite.getResponseMatrix(selectedNode.epochList, 'Amp1');

    baselineValues = mean(resMat(:, 1:paras.prePts), 2);

    isExcitatory = mean(baselineValues) < 0;
    if strcmp(selectedNode.parent.parent.splitValue, 'exc')
        paras.epochRange = find(baselineValues < 0);
    elseif strcmp(selectedNode.parent.parent.splitValue, 'inh')
        paras.epochRange = find(baselineValues > 0);
    else
        error('Unknown splitValue. Expected ''exc'' or ''inh''.');
    end

    resMat = resMat(paras.epochRange, :);
    baseline(nodeIdx) = mean(mean(resMat(:, 1:paras.prePts), 2));

    spotContrast = zeros(1, selectedNode.epochList.length);
    for i = 1:length(spotContrast)
        spotContrast(i) = selectedNode.epochList.elements(i).protocolSettings('currentSpotContrast');
    end

    paras.epochRange = 1:size(resMat, 1);
    spotContrast = spotContrast(paras.epochRange);
    contrastArray = unique(spotContrast);

    if strcmp(selectedNode.epochList.firstValue.protocolSettings('onlineAnalysis'), 'extracellular') || paras.spikeTag
        for i = 1:size(resMat, 1)
            resMat(i, :) = resMat(i, :) - movmedian(resMat(i, :), 100);
        end
        [spikeTimes, ~, ~] = SpikeDetectorNew(resMat, 'thresholdSpikeFactor', paras.spikeTh);
        trace = spikeTimeToPSTH(resMat, spikeTimes, paras.psthSigma, paras.sampleRate);
    else
        for i = 1:size(resMat, 1)
            resMat(i, :) = smooth(resMat(i, :), 500);
        end
        % trace=resMat;
        trace = resMat - repmat(mean(resMat(:, 1:paras.prePts), 2), 1, size(resMat, 2));
    end

    resArray = zeros(1, length(contrastArray));
    mTraceArray = zeros(length(contrastArray), size(resMat, 2));

    for i = 1:length(contrastArray)
        contrastMask = contrastArray(i) == spotContrast;

        if strcmp(selectedNode.epochList.firstValue.protocolSettings('onlineAnalysis'), 'extracellular') || paras.spikeTag
            temp = cellfun(@(x) length(x(x > paras.prePts & x < paras.prePts + paras.stimPts)), spikeTimes(contrastMask));
            resArray(i) = mean(temp);
        else
            % Compute amplitude instead of area
            traceSegment = trace(contrastMask, paras.prePts:paras.prePts + paras.stimPts);
            [maxAbsValue, maxIdx] = max(abs(traceSegment), [], 2);
            signs = sign(traceSegment(sub2ind(size(traceSegment), (1:size(traceSegment,1))', maxIdx)));
            amplitude = maxAbsValue .* signs;
            resArray(i) = mean(amplitude);  % This line should now work correctly
        end

        % Normalize the contrast tuning curve
              % Determine normalization factor
        if isExcitatory
            normIndex = find(contrastArray == -0.9);
        else
            normIndex = find(contrastArray == 0.9);
        end
  
            normFactor = resArray(normIndex);
 
                normalizedResArray = resArray / normFactor;

 
        if sum(contrastMask) > 1
            mTraceArray(i, :) = mean(trace(contrastMask, :), 1);
        else
            mTraceArray(i, :) = trace(contrastMask, :);
        end
    end

    nodeColor = colors(mod(nodeIdx-1, length(colors)) + 1);
    for i = 1:size(mTraceArray, 1)
        plot(ax(1), (1:size(mTraceArray, 2))/paras.sampleRate, mTraceArray(i, :), ...
            'linewidth', 1, 'color', nodeColor, 'DisplayName', '');
    end

    plot(ax(2), contrastArray, normalizedResArray, '-o', 'linewidth', 2, 'DisplayName', ['Node ' num2str(selectedNode.splitValue)]);

    % Compute differentiation of each row in mTraceArray
    diffTraceArray = diff(mTraceArray, 1, 2) * paras.sampleRate;

    % Select only negative contrasts
    if isExcitatory
        negativeContrastMask = contrastArray < -0.2;
    else
        negativeContrastMask = contrastArray > 0.2;
    end

    negativeContrastDiffTraces = diffTraceArray(negativeContrastMask, :);

    % Normalize each negative contrast trace individually based on polarity
    normalizedNegativeContrastDiffTraces = zeros(size(negativeContrastDiffTraces));
    for i = 1:size(negativeContrastDiffTraces, 1)
        if isExcitatory
            % For excitatory traces (negative deflections), use the minimum value
            minValue = min(negativeContrastDiffTraces(i, paras.prePts:paras.prePts+paras.stimPts-1));
            normalizedNegativeContrastDiffTraces(i, :) = negativeContrastDiffTraces(i, :) / abs(minValue);
        else
            % For inhibitory traces (positive deflections), use the maximum value
            maxValue = max(negativeContrastDiffTraces(i, paras.prePts:paras.prePts+paras.stimPts-1));
            normalizedNegativeContrastDiffTraces(i, :) = negativeContrastDiffTraces(i, :) / abs(maxValue);
        end
    end


    timeAxis = (1:size(diffTraceArray, 2)) / paras.sampleRate;

    % Plot normalized differentiated traces for negative contrasts
    for i = 1:size(normalizedNegativeContrastDiffTraces, 1)
        plot(ax(3), timeAxis, normalizedNegativeContrastDiffTraces(i, :), 'linewidth', 1, 'color', 'b', 'DisplayName', '');
    end

    % Compute and plot mean normalized differentiated trace for negative contrasts
    meanNormalizedNegativeContrastDiffTrace = mean(normalizedNegativeContrastDiffTraces, 1);
    temporalFilter = meanNormalizedNegativeContrastDiffTrace(paras.prePts:paras.prePts+paras.stimPts-1);
    % Time vector for the temporal filter
    t = (0:length(temporalFilter)-1) / paras.sampleRate;

    % Fit biphasic EPSC-like curve to the temporal filter
    [fitResult, gof, fitInfo] = fitBiphasicEPSCCurve(t, temporalFilter, isExcitatory);

    % Extract parameters from the fit (if needed)
    % Extract parameters from the fit
    A1 = fitResult.A1;
    tau1 = fitResult.tau1;
    A2 = fitResult.A2;
    tau2 = fitResult.tau2;
    t0 = fitResult.t0;


    % Generate fitted curve
    % Generate fitted curve
    fittedCurve = biphasicEPSCfunction(fitResult, t);

    % Plot the original temporal filter and the fitted curve
    plot(ax(3), timeAxis(paras.prePts:paras.prePts+paras.stimPts-1), temporalFilter, 'linewidth', 2, 'color', nodeColor, 'DisplayName', ['Node ' num2str(selectedNode.splitValue) ' (Data)']);
    plot(ax(3), timeAxis(paras.prePts:paras.prePts+paras.stimPts-1), fittedCurve, 'linewidth', 2, 'color', nodeColor, 'linestyle', '--', 'DisplayName', ['Node ' num2str(selectedNode.splitValue) ' (Fit)']);

    % Store contrast tuning nonlinearity in the output structure
    output{nodeIdx}.contrastArray = contrastArray;
    output{nodeIdx}.rawResArray = resArray;
    output{nodeIdx}.normalizedResArray = normalizedResArray;

    output{nodeIdx}.diffTraceArray = diffTraceArray;
    output{nodeIdx}.normalizedNegativeContrastDiffTraces = normalizedNegativeContrastDiffTraces;
    output{nodeIdx}.meanNormalizedNegativeContrastDiffTrace = meanNormalizedNegativeContrastDiffTrace;
    output{nodeIdx}.temporalFilter = temporalFilter;
    % Store fit results and diagnostics
    % Store the new fit results and parameters
    output{nodeIdx}.EPSCfit.A1 = A1;
    output{nodeIdx}.EPSCfit.tau1 = tau1;
    output{nodeIdx}.EPSCfit.A2 = A2;
    output{nodeIdx}.EPSCfit.tau2 = tau2;
    output{nodeIdx}.EPSCfit.t0 = t0;
    output{nodeIdx}.EPSCfit.gof = gof;
    output{nodeIdx}.EPSCfit.fitInfo = fitInfo;
    output{nodeIdx}.EPSCfit.isExcitatory = isExcitatory;

end

if length(selectedNodes) == 2
    if baseline(1) < 0 && baseline(2) > 0
        excIdx = 1;
        inhIdx = 2;
    elseif baseline(2) < 0 && baseline(1) > 0
        excIdx = 2;
        inhIdx = 1;
    else
        error('Both nodes have the same sign for baseline. Cannot assign Exc/Inh.');
    end

    maxExc = max(abs(output{excIdx}.resArray));
    maxInh = max(abs(output{inhIdx}.resArray));

    ieRatio = maxInh / maxExc;
    output{1}.ieRatio = ieRatio;
    output{2}.ieRatio = ieRatio;

    disp(['I/E Ratio: ', num2str(ieRatio)]);
end

legend(ax(1), 'off');
initFig(ax(1), 'Time (s)', 'pA');
legend(ax(2), 'show');
initFig(ax(2), 'Contrast', 'Integrated Response (pC)');
box(ax(2), 'off');
legend(ax(3), 'show');
initFig(ax(3), 'Time (s)', 'Normalized Rate of Change (Negative Contrasts)');
box(ax(3), 'off');
setAxes(f);

hold(ax(1), 'off');
hold(ax(2), 'off');
hold(ax(3), 'off');

end

function [fitResult, gof, fitInfo] = fitBiphasicEPSCCurve(t, y, isExcitatory)
% Define the biphasic EPSC function using difference of two alpha functions
EPSCfun = @(A1, tau1, A2, tau2, t0, t) ...
    (A1 * ((t - t0)/tau1) .* exp(1 - (t - t0)/tau1) - ...
    A2 * ((t - t0)/tau2) .* exp(1 - (t - t0)/tau2)) .* (t >= t0);

% Set up fittype and options
ft = fittype(EPSCfun, 'independent', 't', 'dependent', 'y');
opts = fitoptions('Method', 'NonlinearLeastSquares', 'Display', 'Off');

% Clean up data: remove NaNs and Infs
validIdx = isfinite(t) & isfinite(y);
t = t(validIdx);
y = y(validIdx);

% Estimate initial values
t0_guess = t(1); % Assuming the response starts at the beginning

if isExcitatory
    % For excitatory traces (negative currents)
    [troughY, troughIdx] = min(y);
    A1_guess = -troughY; % Amplitude should be positive
    tau1_guess = (t(troughIdx) - t0_guess) / 2;

    % Find the overshoot (post-peak positive deflection)
    postPeakRange = troughIdx+1:length(y);
    if ~isempty(postPeakRange)
        [postPeakY, postPeakIdxRel] = max(y(postPeakRange));
        postPeakIdx = postPeakRange(1) + postPeakIdxRel - 1;
        A2_guess = postPeakY;
        tau2_guess = (t(postPeakIdx) - t(troughIdx)) / 2;
    else
        A2_guess = A1_guess / 2; % Default guess
        tau2_guess = tau1_guess * 2;
    end
else
    % For inhibitory traces (positive currents)
    [peakY, peakIdx] = max(y);
    A1_guess = peakY;
    tau1_guess = (t(peakIdx) - t0_guess) / 2;

    % Find the overshoot (post-peak negative deflection)
    postTroughRange = peakIdx+1:length(y);
    if ~isempty(postTroughRange)
        [postTroughY, postTroughIdxRel] = min(y(postTroughRange));
        postTroughIdx = postTroughRange(1) + postTroughIdxRel - 1;
        A2_guess = -postTroughY;
        tau2_guess = (t(postTroughIdx) - t(peakIdx)) / 2;
    else
        A2_guess = A1_guess / 2; % Default guess
        tau2_guess = tau1_guess * 2;
    end
end

% Ensure time constants are positive and reasonable
tau1_guess = max(tau1_guess, t(2) - t(1)); % At least one time step
tau2_guess = max(tau2_guess, t(2) - t(1));

% Set initial values and bounds based on polarity
opts.StartPoint = [A1_guess, tau1_guess, A2_guess, tau2_guess, t0_guess];
opts.Lower = [0, 0, 0, 0, min(t)];
opts.Upper = [Inf, Inf, Inf, Inf, max(t)];

% Perform the fit
[fitResult, gof, fitInfo] = fit(t', y', ft, opts);
end


function y_fit = biphasicEPSCfunction(fitResult, t)
% Reconstruct the biphasic EPSC function from fit parameters
A1 = fitResult.A1;
tau1 = fitResult.tau1;
A2 = fitResult.A2;
tau2 = fitResult.tau2;
t0 = fitResult.t0;

% Compute the EPSC function
y_fit = (A1 * ((t - t0)/tau1) .* exp(1 - (t - t0)/tau1) - ...
    A2 * ((t - t0)/tau2) .* exp(1 - (t - t0)/tau2)) .* (t >= t0);

end

