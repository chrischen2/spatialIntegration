% Spatial-Temporal Receptive Field (STRF) Model for OFF-transient Retinal Ganglion Cells
%
% This code implements a computational model of OFF-transient retinal ganglion 
% cells incorporating spatial and temporal filtering, synaptic depression, and
% nonlinear integration. The model simulates:
%   - Excitatory and inhibitory subunit mosaics
%   - Center-surround organization for inhibitory subunits
%   - Temporal filtering in both pathways
%   - Synaptic depression at excitatory subunits (bipolar cell terminals)
%
% Reference:
%   Authors: Qiang (Chris) Chen
%   Date: December 2024
%
%   For further details on the dynamic synapse mechanism and supporting data,
%   please see Chen, et al. 2025.


%% INITIALIZATION AND PARAMETER SETUP
clearvars; close all; clc;

% Define core model parameters (all units are provided in microns or seconds)
modelParams = struct();
modelParams.MicronsPerPixel       = 2;
modelParams.RFSize_um             = 400;
modelParams.excSubunitSigma_um    = 15;
modelParams.excSubunitSpacing_um  = 30;
modelParams.excPoolingSigma_um    = 50;  % For computing total excitatory RGC response
modelParams.inhSubunitSigma_um    = 12;
modelParams.inhSubunitSpacing_um  = 25;
modelParams.inhSurroundSigma_um   = 20;
modelParams.inhSurroundStrength   = 0.9; % Strength of center-surround inhibition
modelParams.inhPoolingSigma_um    = 50;
modelParams.inhPoolingStrength    = 0.9; % Regulates presynaptic inhibition onto bipolar cells
modelParams.dt                    = 1/1000;
modelParams.K_rec                 = 10;    % Recovery/restocking rate
modelParams.K_rel                 = 5;     % Release rate
modelParams.b                     = 5;     % Scaling factor
modelParams.exc_0                 = 0.3;   % Baseline excitation
modelParams.excRectRatio          = 0.3;   % Rectification factor for excitatory pathway
modelParams.inhRectRatio          = 0;     % Rectification factor for inhibitory pathway
modelParams.beta                  = 10;    % Sensitivity of release rate to inhibition
modelParams.subunitJitter         = 0.2;   % Jitter fraction to reduce aliasing

% Baseline vesicle occupancy calculation
modelParams.n_0 = 1 / (1 + modelParams.b * modelParams.exc_0 * modelParams.K_rel / modelParams.K_rec);

% Define the receptive field size (in pixels)
rfPixelSize = round(modelParams.RFSize_um / modelParams.MicronsPerPixel);
[X, Y] = meshgrid(1:rfPixelSize, 1:rfPixelSize);
centerPixel = rfPixelSize / 2;

%% CONVERT SPATIAL DIMENSIONS FROM MICRONS TO PIXELS
excSubunitSigma_px   = round(modelParams.excSubunitSigma_um / modelParams.MicronsPerPixel);
excSubunitSpacing_px = round(modelParams.excSubunitSpacing_um / modelParams.MicronsPerPixel);
excPoolingSigma_px   = round(modelParams.excPoolingSigma_um / modelParams.MicronsPerPixel);
inhSubunitSigma_px   = round(modelParams.inhSubunitSigma_um / modelParams.MicronsPerPixel);
inhSubunitSpacing_px = round(modelParams.inhSubunitSpacing_um / modelParams.MicronsPerPixel);
inhSurroundSigma_px  = round(modelParams.inhSurroundSigma_um / modelParams.MicronsPerPixel);
inhPoolingSigma_px   = round(modelParams.inhPoolingSigma_um / modelParams.MicronsPerPixel);

% GENERATE JITTERED SUBUNIT MOSAICS
% Excitatory subunits 
[modelParams.excSubunitIndices, excLocations] = createJitteredSubunitMosaic(rfPixelSize, ...
    2 * excSubunitSigma_px, modelParams.subunitJitter );
% Inhibitory subunits 
[modelParams.inhSubunitIndices, inhLocations] = createJitteredSubunitMosaic(rfPixelSize, ...
    2 * inhSubunitSigma_px, modelParams.subunitJitter );

% CREATE SPATIAL FILTERS
% Excitatory subunit filter: normalized Gaussian
modelParams.excSubunitFilter = exp(-((X - centerPixel).^2 + (Y - centerPixel).^2) / (2 * excSubunitSigma_px^2));
modelParams.excSubunitFilter = modelParams.excSubunitFilter / sum(modelParams.excSubunitFilter(:));

% Excitatory pooling filter: normalized Gaussian
modelParams.excPoolingFilter = exp(-((X - centerPixel).^2 + (Y - centerPixel).^2) / (2 * excPoolingSigma_px^2));
modelParams.excPoolingFilter = modelParams.excPoolingFilter / sum(modelParams.excPoolingFilter(:));

% Inhibitory subunit filter: normalized Gaussian
modelParams.inhSubunitFilter = exp(-((X - centerPixel).^2 + (Y - centerPixel).^2) / (2 * inhSubunitSigma_px^2));
modelParams.inhSubunitFilter = modelParams.inhSubunitFilter / sum(modelParams.inhSubunitFilter(:));

% Inhibitory surround filter: normalized and scaled
modelParams.inhSubunitSurround = exp(-((X - centerPixel).^2 + (Y - centerPixel).^2) / (2 * inhSurroundSigma_px^2));
modelParams.inhSubunitSurround = modelParams.inhSurroundStrength * (modelParams.inhSubunitSurround / sum(modelParams.inhSubunitSurround(:)));

% Inhibitory pooling filter (used later for spatial weighting)
modelParams.inhPoolingFilter = exp(-((X - centerPixel).^2 + (Y - centerPixel).^2) / (2 * inhPoolingSigma_px^2));

%% LOAD AND PROCESS TEMPORAL FILTERS
% Load temporal nonlinearity filters from external file
nl = load('OffTNonlinearity.mat', 'exc', 'inh');
downsampleFactor = 10^4 * modelParams.dt;
excTempFilt = smooth(-nl.exc.temporalFilter(:), 5 * downsampleFactor);  % flip to push polarity into temporal filter
inhTempFilt = smooth(nl.inh.temporalFilter(:), 5 * downsampleFactor);  % cross over inhibition

% Downsample and normalize temporal filters
modelParams.excTemporalFilter = excTempFilt(1:downsampleFactor:end)' / abs(min(excTempFilt(1:downsampleFactor:end)));
modelParams.inhTemporalFilter = inhTempFilt(1:downsampleFactor:end)' / abs(max(inhTempFilt(1:downsampleFactor:end)));

%% VISUALIZATION OF SPATIAL AND TEMPORAL FILTERS
figure('Position', [100 100 1500 1000]);

% --- Top Row: Spatial Filters and Cross Sections ---
subplot(2,4,1);
imagesc(modelParams.excSubunitFilter);
axis image; colorbar;
title('Excitatory Subunit Filter');
xlabel('Pixels'); ylabel('Pixels');
hold on;
plot([1, size(modelParams.excSubunitFilter, 2)], [round(centerPixel), round(centerPixel)], 'r--');

subplot(2,4,2);
plot(modelParams.excSubunitFilter(round(centerPixel), :), 'r-', 'LineWidth', 2);
grid on;
title('Excitatory Filter Cross Section');
xlabel('Position (pixels)'); ylabel('Filter Value');

subplot(2,4,3);
inhCenterSurround = modelParams.inhSubunitFilter - modelParams.inhSubunitSurround;
imagesc(inhCenterSurround);
axis image; colorbar;
title('Inhibitory Center-Surround Filter');
xlabel('Pixels'); ylabel('Pixels');
hold on;
plot([1, size(inhCenterSurround, 2)], [round(centerPixel), round(centerPixel)], 'r--');

subplot(2,4,4);
plot(inhCenterSurround(round(centerPixel), :), 'r-', 'LineWidth', 2);
grid on;
title('Inhibitory Filter Cross Section');
xlabel('Position (pixels)'); ylabel('Filter Value');

% --- Bottom Row: Temporal Filters and Subunit Grid Patterns ---
subplot(2,4,5);
timeVectorExc = modelParams.dt:modelParams.dt:(length(modelParams.excTemporalFilter) * modelParams.dt);
timeVectorInh = modelParams.dt:modelParams.dt:(length(modelParams.inhTemporalFilter) * modelParams.dt);
plot(timeVectorExc, modelParams.excTemporalFilter, 'b-', 'LineWidth', 2); hold on;
plot(timeVectorInh, modelParams.inhTemporalFilter, 'r-', 'LineWidth', 2);
xlabel('Time (s)'); ylabel('Amplitude');
title('Temporal Filters');
legend('Excitatory','Inhibitory'); grid on;

% Excitatory subunit grid visualization
subplot(2,4,6);
excGrid = zeros(rfPixelSize);
excGrid(modelParams.excSubunitIndices) = 1;
for idx = 1:length(modelParams.excSubunitIndices)
    [y, x] = ind2sub([rfPixelSize, rfPixelSize], modelParams.excSubunitIndices(idx));
    [XX, YY] = meshgrid(1:rfPixelSize, 1:rfPixelSize);
    gaussianOverlay = exp(-((XX - x).^2 + (YY - y).^2) / (2 * (excSubunitSigma_px/2)^2));
    excGrid = excGrid + gaussianOverlay;
end
imagesc(excGrid); axis image; colorbar;
title('Excitatory Subunit Grid Pattern');
xlabel('Pixels'); ylabel('Pixels');

% Inhibitory subunit grid visualization
subplot(2,4,7);
inhGrid = zeros(rfPixelSize);
inhGrid(modelParams.inhSubunitIndices) = 1;
for idx = 1:length(modelParams.inhSubunitIndices)
    [y, x] = ind2sub([rfPixelSize, rfPixelSize], modelParams.inhSubunitIndices(idx));
    [XX, YY] = meshgrid(1:rfPixelSize, 1:rfPixelSize);
    gaussianOverlay = exp(-((XX - x).^2 + (YY - y).^2) / (2 * (inhSubunitSigma_px/2)^2));
    inhGrid = inhGrid + gaussianOverlay;
end
imagesc(inhGrid); axis image; colorbar;
title('Inhibitory Subunit Grid Pattern');
xlabel('Pixels'); ylabel('Pixels');

sgtitle('Spatiotemporal Filter Components and Subunit Grid Patterns','FontSize',14);
set(gcf, 'Color', 'w');

%% NATURAL IMAGE PATCH SIMULATION SETUP
% Define stimulus parameters for natural image patch simulation
stimulusParams = struct();
stimulusParams.preTime   = 0.2;  % Pre-stimulus duration (s)
stimulusParams.stimTime  = 0.3;  % Stimulus duration (s)
stimulusParams.tailTime  = 0.3;  % Post-stimulus duration (s)
stimulusParams.totalTime = stimulusParams.preTime + stimulusParams.stimTime + stimulusParams.tailTime;
stimulusParams.dt        = modelParams.dt;
stimulusParams.t         = 0:stimulusParams.dt:stimulusParams.totalTime - stimulusParams.dt;
stimulusParams.T         = length(stimulusParams.t);
stimulusParams.stimStart = round(stimulusParams.preTime / stimulusParams.dt);
stimulusParams.stimEnd   = round((stimulusParams.preTime + stimulusParams.stimTime) / stimulusParams.dt);
stimulusParams.rfSize    = rfPixelSize;
[stimulusParams.X, stimulusParams.Y] = meshgrid(1:stimulusParams.rfSize, 1:stimulusParams.rfSize);
stimulusParams.center    = stimulusParams.rfSize / 2;

% Generate a flashed grating stimulus as an example (uncomment to choose stimulus type)
spatialFreq       = 1/100;  % cycles per micron
orientationDeg    = 0;      % orientation in degrees
spatialContrast   = 0.8;
stimulus = generateFlashedGrating(modelParams, stimulusParams, spatialFreq, orientationDeg, spatialContrast);

% Process the stimulus through the model (excitatory and inhibitory pathways)
debug = processStimulus(stimulus, modelParams);

%% NATURAL IMAGE PATCH EXTRACTION AND ANALYSIS
% Configure analysis settings for natural image patches
analysisConfig = struct();
analysisConfig.noPatches       = 10;
analysisConfig.patchContrast   = 'negative';  % Options: 'all', 'positive', 'negative'
analysisConfig.patchSampling   = 'ranked';    % Options: 'random', 'ranked', 'biasedSpatialContrast'
analysisConfig.seed            = 1;
analysisConfig.preTime         = stimulusParams.preTime;
analysisConfig.stimTime        = stimulusParams.stimTime;
analysisConfig.tailTime        = stimulusParams.tailTime;
analysisConfig.dt              = modelParams.dt;
analysisConfig.totalTime       = analysisConfig.preTime + analysisConfig.stimTime + analysisConfig.tailTime;
analysisConfig.t               = 0:analysisConfig.dt:analysisConfig.totalTime-analysisConfig.dt;
analysisConfig.zoomFactor      = 6.6;
analysisConfig.stimStart       = stimulusParams.stimStart;
analysisConfig.stimEnd         = stimulusParams.stimEnd;
analysisConfig.filterSize      = rfPixelSize;

% Extract natural image patches from a specified resource directory
patches = extractNaturalPatches(analysisConfig);

% Process each patch and compute model responses
fprintf('Computing responses for scaled patches...\n');
responses = struct();
responses.image = struct('exc', zeros(size(patches,3), length(analysisConfig.t)), ...
                         'inh', zeros(size(patches,3), length(analysisConfig.t)), ...
                         'spike', zeros(size(patches,3), length(analysisConfig.t)));
responses.disc = struct('exc', zeros(size(patches,3), length(analysisConfig.t)), ...
                        'inh', zeros(size(patches,3), length(analysisConfig.t)), ...
                        'spike', zeros(size(patches,3), length(analysisConfig.t)));
patchMeans = zeros(size(patches,3), length(analysisConfig.t));

for i = 1:size(patches,3)
    patchStim = zeros(analysisConfig.filterSize, analysisConfig.filterSize, length(analysisConfig.t));
    patchStim(:, :, analysisConfig.stimStart:analysisConfig.stimEnd) = repmat(patches(:, :, i), [1, 1, analysisConfig.stimEnd - analysisConfig.stimStart + 1]);
    imageResp = processStimulus(patchStim, modelParams);
    patchMeans(i, :) = squeeze(mean(patchStim, [1, 2]));
    
    % Generate equivalent disc stimulus and process
    eqContrast = computeEquivalentContrast(patches(:, :, i), modelParams);
    discStim = generateDiscStimulus(eqContrast, analysisConfig.filterSize, analysisConfig.t, analysisConfig.stimStart, analysisConfig.stimEnd);
    discResp = processStimulus(discStim, modelParams);
    
    responses.image.exc(i, :) = imageResp.totalExc;
    responses.image.inh(i, :) = imageResp.totalInh;
    responses.disc.exc(i, :)  = discResp.totalExc;
    responses.disc.inh(i, :)  = discResp.totalInh;
    
    % Compute spike responses as 3×excitation minus inhibition
    responses.image.spike(i, :) = 3 * imageResp.totalExc - imageResp.totalInh;
    responses.disc.spike(i, :)  = 3 * discResp.totalExc - discResp.totalInh;
    fprintf('Processed %d/%d responses\n', i, size(patches,3));
end

% Run additional analysis (e.g., cell selection, metrics, and plotting)
responses = analyzeNeuralResponses(responses, patches, stimulusParams);

%% HELPER FUNCTION DEFINITIONS
%==========================================================================
function [subunitIndices, subunitLocations] = createJitteredSubunitMosaic(rfSize, spacing, jitterFraction)
    % Create a jittered mosaic of subunit positions.
    baseLocations = spacing:spacing:rfSize-spacing;
    [Xgrid, Ygrid] = meshgrid(baseLocations, baseLocations);
    
    jitterPixels = spacing * jitterFraction;
    Xgrid = Xgrid + jitterPixels * (rand(size(Xgrid)) - 0.5);
    Ygrid = Ygrid + jitterPixels * (rand(size(Ygrid)) - 0.5);
    
    % Constrain positions within valid boundaries and round
    Xgrid = round(min(max(Xgrid, spacing), rfSize - spacing));
    Ygrid = round(min(max(Ygrid, spacing), rfSize - spacing));
    
    subunitIndices = sub2ind([rfSize, rfSize], Ygrid(:), Xgrid(:));
    subunitLocations = [Xgrid(:) Ygrid(:)];
    
    [subunitIndices, uniqueIdx] = unique(subunitIndices);
    subunitLocations = subunitLocations(uniqueIdx, :);
end

%--------------------------------------------------------------------------
function stimulus = generateFlashedGrating(modelParams, stimulusParams, spatialFreq, orientationDeg, contrast)
    % Generate a flashed grating stimulus.
    pixelFreq = spatialFreq * modelParams.MicronsPerPixel;
    theta = orientationDeg * pi / 180;
    Xrot = stimulusParams.X * cos(theta) - stimulusParams.Y * sin(theta);
    grating = contrast * sin(2 * pi * pixelFreq * Xrot);
    
    stimulus = zeros(stimulusParams.rfSize, stimulusParams.rfSize, stimulusParams.T);
    stimulus(:, :, stimulusParams.stimStart:stimulusParams.stimEnd) = ...
        repmat(grating, [1, 1, stimulusParams.stimEnd - stimulusParams.stimStart + 1]);
end

%--------------------------------------------------------------------------
function [inhPooledMatrix, inhSubunitOutputs, spatialResponses, inhSTFilteredStimulus] = processInhibitorySignals(stimulus, modelParams)
    [height, width, T] = size(stimulus);
    numExcSubunits = length(modelParams.excSubunitIndices);
    filterLen = length(modelParams.inhTemporalFilter);
    
    centerSurroundFilter = modelParams.inhSubunitFilter - modelParams.inhSurroundStrength * modelParams.inhSubunitSurround;
    centerSurroundFilter = centerSurroundFilter / sum(centerSurroundFilter(:));
    
    spatialResponses = zeros(length(modelParams.inhSubunitIndices), T);
    fprintf('Processing inhibitory spatial filtering...\n');
    for t = 1:T
        frame = stimulus(:, :, t);
        spatialResponse = conv2(frame, centerSurroundFilter, 'same');
        spatialResponses(:, t) = spatialResponse(modelParams.inhSubunitIndices);
        if mod(t, round(T/2)) == 0
            fprintf('Progress: %.1f%%\n', 100*t/T);
        end
    end
    
    normFactor = sum(abs(modelParams.inhTemporalFilter));
    normalizedFilter = modelParams.inhTemporalFilter / normFactor;
    temporalOutput = zeros(size(spatialResponses));
    for i = 1:size(spatialResponses, 1)
        paddedResponse = [zeros(1, filterLen), spatialResponses(i, :), zeros(1, filterLen)];
        convResult = zeros(1, length(paddedResponse));
        for t = 1:length(paddedResponse)-filterLen+1
            convResult(t+filterLen-1) = sum(paddedResponse(t:t+filterLen-1) .* flip(normalizedFilter));
        end
        temporalOutput(i, :) = convResult(filterLen+1:end-filterLen);
    end
    
    inhSTFilteredStimulus = temporalOutput;
    inhSubunitOutputs = applyPiecewiseNonlinearity(temporalOutput, modelParams.inhRectRatio);
    
    inhPooledMatrix = zeros(numExcSubunits, T);
    debug.inhWeights = cell(numExcSubunits, 1);
    debug.contributingInhIndices = cell(numExcSubunits, 1);
    
    for exc_idx = 1:numExcSubunits
        [exc_y, exc_x] = ind2sub([height, width], modelParams.excSubunitIndices(exc_idx));
        inhWeights = zeros(length(modelParams.inhSubunitIndices), 1);
        for inh_idx = 1:length(modelParams.inhSubunitIndices)
            [inh_y, inh_x] = ind2sub([height, width], modelParams.inhSubunitIndices(inh_idx));
            y_offset = inh_y - exc_y + ceil(size(modelParams.inhPoolingFilter, 1)/2);
            x_offset = inh_x - exc_x + ceil(size(modelParams.inhPoolingFilter, 2)/2);
            if (y_offset >= 1 && y_offset <= size(modelParams.inhPoolingFilter, 1) && ...
                x_offset >= 1 && x_offset <= size(modelParams.inhPoolingFilter, 2))
                inhWeights(inh_idx) = modelParams.inhPoolingFilter(y_offset, x_offset);
            end
        end
        
        inhWeights = inhWeights / sum(inhWeights);
        debug.inhWeights{exc_idx} = inhWeights;
        debug.contributingInhIndices{exc_idx} = find(inhWeights > 0);
        for t = 1:T
            inhPooledMatrix(exc_idx, t) = sum(inhWeights .* inhSubunitOutputs(:, t));
        end
    end
end

%--------------------------------------------------------------------------
function debug = processStimulus(stimulus, modelParams)
    [height, width, T] = size(stimulus);
    numExcSubunits = length(modelParams.excSubunitIndices);
    filterLen = length(modelParams.excTemporalFilter);
    
    [inhPooledMatrix, inhSubunitOutputs, spatialResponse_inh, inhSTFilteredStimulus] = processInhibitorySignals(stimulus, modelParams);
    
    spatialResponses = zeros(numExcSubunits, T);
    fprintf('Processing excitatory spatial filtering...\n');
    for t = 1:T
        frame = stimulus(:, :, t);
        excResponse_spatial = conv2(frame, modelParams.excSubunitFilter, 'same');
        spatialResponses(:, t) = excResponse_spatial(modelParams.excSubunitIndices);
        if mod(t, round(T/2)) == 0
            fprintf('Progress: %.1f%%\n', 100*t/T);
        end
    end
    spatialResponse_exc = spatialResponses;
    
    normFactor = sum(abs(modelParams.excTemporalFilter));
    normalizedFilter = modelParams.excTemporalFilter / normFactor;
    temporalOutput = zeros(size(spatialResponses));
    for i = 1:numExcSubunits
        paddedResponse = [zeros(1, filterLen), spatialResponses(i, :), zeros(1, filterLen)];
        convResult = zeros(1, length(paddedResponse));
        for t = 1:length(paddedResponse)-filterLen+1
            convResult(t+filterLen-1) = sum(paddedResponse(t:t+filterLen-1) .* flip(normalizedFilter));
        end
        temporalOutput(i, :) = convResult(filterLen+1:end-filterLen);
    end
    STFilteredStimulus = temporalOutput;
    STFilteredStimulus = STFilteredStimulus * (1 - modelParams.exc_0) + modelParams.exc_0;
    
    n = zeros(numExcSubunits, T);
    n(:, 1) = modelParams.n_0;
    excResponse = zeros(numExcSubunits, T);
    excResponse(:, 1) = modelParams.exc_0;
    totalExc = zeros(1, T);
    totalInh = zeros(1, T);
    synapseMod_matrix = zeros(numExcSubunits, T);
    
    excWeightsAtSubunits = modelParams.excPoolingFilter(modelParams.excSubunitIndices);
    excWeightsAtSubunits = excWeightsAtSubunits / sum(excWeightsAtSubunits);
    
    inhWeightsAtSubunits = modelParams.excPoolingFilter(modelParams.inhSubunitIndices);
    inhWeightsAtSubunits = inhWeightsAtSubunits / sum(inhWeightsAtSubunits);
    
    for t = 2:T
        excMap = zeros(height, width);
        inhMap = zeros(height, width);
        for i = 1:numExcSubunits
            synapseMod = modelParams.inhPoolingStrength * inhPooledMatrix(i, t);
            synapseMod_matrix(i, t) = synapseMod;
            
            exc_input = STFilteredStimulus(i, t) - synapseMod;
            resp = exc_input - modelParams.exc_0;
            pieceResp = applyPiecewiseNonlinearity(resp, modelParams.excRectRatio);
            excResponse(i, t) = pieceResp * n(i, t-1) + modelParams.exc_0;
            
            dndt = compute_dndt(n(i, t-1), excResponse(i, t), modelParams.K_rec, modelParams.K_rel, modelParams.b, synapseMod_matrix(i, t), modelParams.beta);
            n(i, t) = max(0, min(1, n(i, t-1) + dndt * modelParams.dt));
            excMap(modelParams.excSubunitIndices(i)) = excResponse(i, t);
        end
        
        numInhSubunits = length(modelParams.inhSubunitIndices);
        for i = 1:numInhSubunits
            inhMap(modelParams.inhSubunitIndices(i)) = inhSubunitOutputs(i, t);
        end
        
        totalExc(t) = sum(excMap(modelParams.excSubunitIndices) .* excWeightsAtSubunits);
        totalInh(t) = sum(inhMap(modelParams.inhSubunitIndices) .* inhWeightsAtSubunits);
    end
    
    debug = struct('n_all', n, ...
                   'synapseMod', synapseMod_matrix, ...
                   'spatialResponse_exc', spatialResponse_exc, ...
                   'spatialResponse_inh', spatialResponse_inh, ...
                   'inhSTFilteredStimulus', inhSTFilteredStimulus, ...
                   'excSTFilteredStimulus', STFilteredStimulus, ...
                   'inhSubunitOutputs', inhSubunitOutputs, ...
                   'inhPooledMatrix', inhPooledMatrix, ...
                   'excResponse', excResponse, ...
                   'totalExc', totalExc, ...
                   'totalInh', totalInh);
end

%--------------------------------------------------------------------------
function y = applyPiecewiseNonlinearity(x, ratio)
    y = x;
    neg_idx = x < 0;
    y(neg_idx) = x(neg_idx) * ratio;
end

%--------------------------------------------------------------------------
function dndt = compute_dndt(n, Exc, K_rec, K_rel, b, inhAmp, beta)
    % compute_dndt: Computes the rate of change of vesicle occupancy n(t) 
    % in the dynamic synapse model for OFF-transient (OffT) retinal ganglion cells.
    %
    % MODEL OVERVIEW:
    %   Vesicle occupancy n(t) in the bipolar-to-RGC synapse can range from 0 to 1.
    %   - (1 - n) * K_rec : Recovers vesicles at rate K_rec, replenishing the pool.
    %   - b * K_rel * n * Exc : Depletes vesicles (release) at rate K_rel, scaled by b.
    %   - exp(-beta * inhAmp): Multiplies the release term to represent inhibition-
    %         induced suppression of exocytosis. Higher inhAmp → stronger suppression.
    %
    % INPUTS:
    %   n       : current vesicle occupancy in [0,1]
    %   Exc     : excitatory drive from the bipolar subunit
    %   K_rec   : rate of vesicle recovery (restocking)
    %   K_rel   : baseline release rate
    %   b       : dimensionless scaling factor for release
    %   inhAmp  : presynaptic inhibition amplitude
    %   beta    : sensitivity of release to inhibition
    %
    % OUTPUT:
    %   dndt    : instantaneous rate of change of n(t) based on these terms
    %
    % REFERENCE:
    %   For more details on the derivation and biological rationale, see Chen, et al.

    % Recovery term: (1 - n) * K_rec 
    % Release term: b * K_rel * n * Exc * exp(-beta * inhAmp)
    dndt = (1 - n) * K_rec - b * K_rel * n * Exc * exp(-beta * inhAmp);
end


%--------------------------------------------------------------------------
function scaledPatch = extractAndScalePatch(image, patchX, patchY, filterSize, zoomFactor)
    [XX, YY] = meshgrid(1:filterSize, 1:filterSize);
    centerPatch = filterSize / 2;
    scaledX = (XX - centerPatch) / zoomFactor + centerPatch;
    scaledY = (YY - centerPatch) / zoomFactor + centerPatch;
    radius = floor(filterSize / 2);
    patch = image(round(patchX - radius + 1):round(patchX + radius), ...
                  round(patchY - radius + 1):round(patchY + radius));
    patch = patch';
    scaledPatch = interp2(XX, YY, patch, scaledX, scaledY, 'cubic', 0);
end


%--------------------------------------------------------------------------
function patches = extractNaturalPatches(analysisConfig)
% Define paths (update these paths as needed)
    VH_PATH = fullfile(pwd, 'resources', 'vanhateren_iml', filesep);
    RESOURCE_PATH = fullfile(pwd, 'resources', filesep);
    imageName = '00152';
    fieldName = ['imk', imageName];
    
    rng(analysisConfig.seed);
    
    fileId = fopen([VH_PATH, fieldName, '.iml'], 'rb', 'ieee-be');
    img = fread(fileId, [1536, 1024], 'uint16');
    fclose(fileId);
    img = double(img);
    img = img ./ max(img(:));
    img_nomean = (img - mean(img(:))) ./ mean(img(:));
    
    load([RESOURCE_PATH, 'NaturalImageFlashLibrary_120117.mat']);
    
    LnResp = imageData.(fieldName).LnModelResponse;
    subunitResp = imageData.(fieldName).SubunitModelResponse;
    patchVariance = imageData.(fieldName).PatchVariance;
    xLoc = imageData.(fieldName).location(:, 1);
    yLoc = imageData.(fieldName).location(:, 2);
    
    if strcmp(analysisConfig.patchContrast, 'all')
        inds = 1:length(LnResp);
    elseif strcmp(analysisConfig.patchContrast, 'positive')
        inds = find(LnResp > 0);
    elseif strcmp(analysisConfig.patchContrast, 'negative')
        inds = find(LnResp <= 0);
    end
    
    xLoc = xLoc(inds);
    yLoc = yLoc(inds);
    subunitResp = subunitResp(inds);
    LnResp = LnResp(inds);
    patchVariance = patchVariance(inds);
    
    responseDifferences = subunitResp - LnResp;
    if strcmp(analysisConfig.patchSampling, 'random')
        pullInds = randsample(1:length(xLoc), analysisConfig.noPatches);
    elseif strcmp(analysisConfig.patchSampling, 'ranked')
        [~, ~, bin] = histcounts(responseDifferences, 2 * analysisConfig.noPatches);
        populatedBins = unique(bin);
        pullInds = arrayfun(@(b) find(b == bin, 1), populatedBins);
        pullInds = randsample(pullInds, analysisConfig.noPatches);
    elseif strcmp(analysisConfig.patchSampling, 'biasedSpatialContrast')
        [~, ~, bin] = histcounts(patchVariance, 2 * analysisConfig.noPatches);
        populatedBins = unique(bin);
        pullInds = arrayfun(@(b) find(b == bin, 1), populatedBins);
        pullInds = randsample(pullInds, analysisConfig.noPatches);
    end
    
    patches = zeros(analysisConfig.filterSize, analysisConfig.filterSize, analysisConfig.noPatches);
    fprintf('Extracting patches...\n');
    for pp = 1:analysisConfig.noPatches
        if mod(pp, 4) == 0
            fprintf('Processed %d/%d patches\n', pp, analysisConfig.noPatches);
        end
        patches(:, :, pp) = extractAndScalePatch(img_nomean, ...
            xLoc(pullInds(pp)), yLoc(pullInds(pp)), analysisConfig.filterSize, analysisConfig.zoomFactor);
    end
    fprintf('Patch extraction complete.\n');
end

%--------------------------------------------------------------------------
function responses = analyzeNeuralResponses(responses, patches, stimParams)
    analysisConfig = setupAnalysisConfig(stimParams);
    if ~isfield(analysisConfig, 't')
        timeLength = size(responses.image.exc, 2);
        analysisConfig.t = (0:(timeLength-1)) * analysisConfig.dt;
    end
    
    onsetWindow = round(analysisConfig.preTime/analysisConfig.dt) : round((analysisConfig.preTime + analysisConfig.onsetWindowSize)/analysisConfig.dt);
    offsetWindow = round((analysisConfig.preTime + analysisConfig.stimTime)/analysisConfig.dt) : round((analysisConfig.preTime + analysisConfig.stimTime + analysisConfig.offsetWindowSize)/analysisConfig.dt);
    
    responses = computeBaselineMetrics(responses, onsetWindow, offsetWindow, analysisConfig);
    example_indices = selectExampleCells(responses, 2);
    example_colors = createColorMap(length(example_indices));
    visualizePatchExamples(patches, example_indices, example_colors, analysisConfig);
    createComparisonFigure(responses, example_indices, example_colors, analysisConfig);
end

%--------------------------------------------------------------------------
function analysisConfig = setupAnalysisConfig(stimParams)
    analysisConfig = stimParams;
    analysisConfig.excOffset = 0.05;  % 50 ms offset for excitation and spike
    analysisConfig.inhOffset = 0.05;  % 50 ms offset for inhibition
    analysisConfig.offsetSamples.exc = round(analysisConfig.excOffset/analysisConfig.dt);
    analysisConfig.offsetSamples.inh = round(analysisConfig.inhOffset/analysisConfig.dt);
    analysisConfig.offsetSamples.spike = analysisConfig.offsetSamples.exc;
    analysisConfig.onsetWindowSize = 0.2;
    analysisConfig.offsetWindowSize = 0.2;
    if ~isfield(analysisConfig, 'patchSampling')
        analysisConfig.patchSampling = 'default';
    end
end

%--------------------------------------------------------------------------
function example_indices = selectExampleCells(responses, nExamples)
    [~, sortIdx] = sort(abs(responses.image.spike_onset_peak - responses.disc.spike_onset_peak), 'descend');
    percentiles = linspace(0, 1, nExamples + 4);
    percentiles = percentiles([2, end-1]);
    example_indices = zeros(1, nExamples);
    for i = 1:nExamples
        idx = round(percentiles(i) * length(sortIdx));
        idx = max(1, min(idx, length(sortIdx)));
        example_indices(i) = sortIdx(idx);
    end
end

%--------------------------------------------------------------------------
function colors = createColorMap(numColors)
    colors = jet(numColors);
end

%--------------------------------------------------------------------------
function visualizePatchExamples(patches, example_indices, example_colors, analysisConfig)
    patchFig = figure('Position', [100 100 1200 300], 'Name', 'Example Natural Image Patches');
    nPatches = length(example_indices);
    nCols = nPatches;
    nRows = 1;
    
    for i = 1:nPatches
        subplot(nRows, nCols, i);
        patch_idx = example_indices(i);
        current_patch = patches(:, :, patch_idx);
        imagesc(current_patch); axis image; colormap(gca, gray); colorbar;
        title(sprintf('Patch #%d', i), 'Color', example_colors(i, :), 'FontWeight', 'bold', 'FontSize', 12);
        set(gca, 'XTick', [], 'YTick', []);
        box on;
        set(gca, 'LineWidth', 2, 'XColor', example_colors(i, :), 'YColor', example_colors(i, :));
    end
    
    if isfield(analysisConfig, 'patchSampling')
        sgtitle(sprintf('Example Patches (%s sampling)', analysisConfig.patchSampling), 'FontSize', 14, 'FontWeight', 'bold');
    else
        sgtitle('Example Patches', 'FontSize', 14, 'FontWeight', 'bold');
    end
    set(patchFig, 'DefaultAxesPosition', [0.1, 0.1, 0.8, 0.8]);
end

%--------------------------------------------------------------------------
function createComparisonFigure(responses, example_indices, example_colors, analysisConfig)
    figure('Position', [50 50 1800 1000]);
    
    subplot(3,5,1)
    plotPeakScatterWithLabels(responses.disc.exc_onset_peak, responses.image.exc_onset_peak, 'Exc On Peak Comparison', example_indices, example_colors);
    
    subplot(3,5,2)
    plotPeakScatterWithLabels(responses.disc.inh_onset_peak, responses.image.inh_onset_peak, 'Inh On Peak Comparison', example_indices, example_colors);
    
    subplot(3,5,3)
    plotPeakScatterWithLabels(responses.disc.spike_onset_peak, responses.image.spike_onset_peak, 'Spike On Peak Comparison', example_indices, example_colors);
    
    subplot(3,5,4)
    plotPeakScatterWithLabels(responses.disc.exc_offset_peak, responses.image.exc_offset_peak, 'Exc Off Peak Comparison', example_indices, example_colors);
    
    subplot(3,5,5)
    plotPeakScatterWithLabels(responses.disc.inh_offset_peak, responses.image.inh_offset_peak, 'Inh Off Peak Comparison', example_indices, example_colors);
    
    subplot(3,5,6)
    plotPeakScatterWithLabels(responses.disc.spike_offset_peak, responses.image.spike_offset_peak, 'Spike Off Peak Comparison', example_indices, example_colors);
    
    subplot(3,5,7)
    plotPeakScatterWithLabels(responses.disc.exc_onset_area, responses.image.exc_onset_area, 'Exc On Area Comparison', example_indices, example_colors);
    
    subplot(3,5,8)
    plotPeakScatterWithLabels(responses.disc.inh_onset_area, responses.image.inh_onset_area, 'Inh On Area Comparison', example_indices, example_colors);
    
    subplot(3,5,9)
    plotPeakScatterWithLabels(responses.disc.spike_onset_area, responses.image.spike_onset_area, 'Spike On Area Comparison', example_indices, example_colors);
    
    subplot(3,5,10)
    plotPeakScatterWithLabels(responses.disc.exc_offset_area, responses.image.exc_offset_area, 'Exc Off Area Comparison', example_indices, example_colors);
    
    subplot(3,5,11)
    plotPeakScatterWithLabels(responses.disc.inh_offset_area, responses.image.inh_offset_area, 'Inh Off Area Comparison', example_indices, example_colors);
    
    subplot(3,5,12)
    plotPeakScatterWithLabels(responses.disc.spike_offset_area, responses.image.spike_offset_area, 'Spike Off Area Comparison', example_indices, example_colors);
    
    subplot(3,5,13);
    plotExampleTraces(responses, example_indices, example_colors, 'exc', 'Example Excitation Response');
    
    subplot(3,5,14);
    plotExampleTraces(responses, example_indices, example_colors, 'inh', 'Example Inhibition Response');
    
    subplot(3,5,15);
    plotExampleTraces(responses, example_indices, example_colors, 'spike', 'Example Spike Response');
    
    for plotNum = 13:15
        subplot(3,5,plotNum);
        xline(analysisConfig.t(analysisConfig.stimStart), '--k');
        xline(analysisConfig.t(analysisConfig.stimEnd), '--k');
    end
end

%--------------------------------------------------------------------------
function eqContrast = computeEquivalentContrast(patch, modelParams)
    [height, width] = size(patch);
    [X, Y] = meshgrid(1:width, 1:height);
    center = [width/2, height/2];
    sigma = modelParams.excPoolingSigma_um / modelParams.MicronsPerPixel;
    weights = exp(-((X - center(1)).^2 + (Y - center(2)).^2) / (2 * sigma^2));
    weights = weights / sum(weights(:));
    eqContrast = sum(patch(:) .* weights(:));
end

%--------------------------------------------------------------------------
function discStim = generateDiscStimulus(eqContrast, filterSize, t, stimStart, stimEnd)
    [X, Y] = meshgrid(1:filterSize, 1:filterSize);
    center = filterSize/2;
    radius = filterSize/4;
    disc = double(sqrt((X - center).^2 + (Y - center).^2) <= radius);
    disc = disc * eqContrast;
    discStim = zeros(filterSize, filterSize, length(t));
    discStim(:, :, stimStart:stimEnd) = repmat(disc, [1, 1, (stimEnd - stimStart + 1)]);
end

%--------------------------------------------------------------------------
function responses = computeBaselineMetrics(responses, onsetWindow, offsetWindow, analysisConfig)
    fields = {'image', 'disc'};
    onsetWindows.exc = onsetWindow + analysisConfig.offsetSamples.exc;
    onsetWindows.inh = onsetWindow + analysisConfig.offsetSamples.inh;
    onsetWindows.spike = onsetWindow + analysisConfig.offsetSamples.exc;
    
    offsetWindows.exc = offsetWindow + analysisConfig.offsetSamples.exc;
    offsetWindows.inh = offsetWindow + analysisConfig.offsetSamples.inh;
    offsetWindows.spike = offsetWindow + analysisConfig.offsetSamples.exc;
    
    nBaselinePoints = round(0.1 * (analysisConfig.stimEnd - analysisConfig.stimStart));
    nBaselinePoints = max(nBaselinePoints, 5);
    preStimBaselineWindow = (analysisConfig.stimStart - nBaselinePoints):(analysisConfig.stimStart - 1);
    
    for f = 1:length(fields)
        field = fields{f};
        baseline_exc = mean(responses.(field).exc(:, preStimBaselineWindow), 2);
        offset_baseline_exc = mean(responses.(field).exc(:, onsetWindows.exc(end-nBaselinePoints+1:end)), 2);
        exc_onset = responses.(field).exc(:, onsetWindows.exc) - baseline_exc;
        exc_offset = responses.(field).exc(:, offsetWindows.exc) - offset_baseline_exc;
        [responses.(field).exc_onset_peak, responses.(field).exc_onset_time] = findResponsePeak(exc_onset, 'absolute');
        [responses.(field).exc_offset_peak, responses.(field).exc_offset_time] = findResponsePeak(exc_offset);
        responses.(field).exc_onset_area = sum(exc_onset, 2);
        responses.(field).exc_offset_area = sum(exc_offset, 2);
        
        baseline_inh = mean(responses.(field).inh(:, preStimBaselineWindow), 2);
        offset_baseline_inh = baseline_inh;
        inh_onset = responses.(field).inh(:, onsetWindows.inh) - baseline_inh;
        inh_offset = responses.(field).inh(:, offsetWindows.inh) - offset_baseline_inh;
        [responses.(field).inh_onset_peak, responses.(field).inh_onset_time] = findResponsePeak(inh_onset);
        [responses.(field).inh_offset_peak, responses.(field).inh_offset_time] = findResponsePeak(inh_offset);
        responses.(field).inh_onset_area = sum(inh_onset, 2);
        responses.(field).inh_offset_area = sum(inh_offset, 2);
        
        baseline_spike = mean(responses.(field).spike(:, preStimBaselineWindow), 2);
        offset_baseline_spike = mean(responses.(field).spike(:, onsetWindows.spike(end-nBaselinePoints+1:end)), 2);
        spike_onset = responses.(field).spike(:, onsetWindows.spike) - baseline_spike;
        spike_offset = responses.(field).spike(:, offsetWindows.spike) - offset_baseline_spike;
        spike_onset_rect = spike_onset;
        spike_offset_rect = spike_offset;
        [responses.(field).spike_onset_peak, responses.(field).spike_onset_time] = findResponsePeak(spike_onset_rect, 'absolute');
        [responses.(field).spike_offset_peak, responses.(field).spike_offset_time] = findResponsePeak(spike_offset_rect);
        responses.(field).spike_onset_area = sum(spike_onset_rect, 2);
        responses.(field).spike_offset_area = sum(spike_offset_rect, 2);
        
        responses.(field).exc_onset_time = responses.(field).exc_onset_time + onsetWindows.exc(1);
        responses.(field).exc_offset_time = responses.(field).exc_offset_time + offsetWindows.exc(1);
        responses.(field).inh_onset_time = responses.(field).inh_onset_time + onsetWindows.inh(1);
        responses.(field).inh_offset_time = responses.(field).inh_offset_time + offsetWindows.inh(1);
        responses.(field).spike_onset_time = responses.(field).spike_onset_time + onsetWindows.spike(1);
        responses.(field).spike_offset_time = responses.(field).spike_offset_time + offsetWindows.spike(1);
        
        responses.(field).baselines = struct('exc_initial', baseline_exc, ...
                                             'exc_offset', offset_baseline_exc, ...
                                             'inh_initial', baseline_inh, ...
                                             'inh_offset', offset_baseline_inh, ...
                                             'spike_initial', baseline_spike, ...
                                             'spike_offset', offset_baseline_spike);
    end
end

%--------------------------------------------------------------------------
function [peak_vals, peak_times] = findResponsePeak(response_window, mode)
    if ~exist('mode', 'var')
        mean_val = mean(response_window);
        if mean_val >= 0
            mode = 'positive';
        else
            mode = 'negative';
        end
    end
    switch mode
        case 'positive'
            [peak_vals, peak_times] = max(response_window, [], 2);
        case 'negative'
            [min_vals, min_times] = min(response_window, [], 2);
            peak_vals = min_vals;
            peak_times = min_times;
        case 'absolute'
            [max_abs_vals, max_abs_times] = max(abs(response_window), [], 2);
            peak_times = max_abs_times;
            peak_vals = zeros(size(max_abs_vals));
            for i = 1:size(response_window, 1)
                peak_vals(i) = response_window(i, peak_times(i));
            end
    end
end

%--------------------------------------------------------------------------
function plotExampleTraces(currentTraces, example_indices, example_colors, traceType, titleText)
    if isfield(currentTraces, 't')
        t = currentTraces.t;
    elseif isfield(currentTraces, 'image') && isfield(currentTraces.image, 'exc')
        t = (1:size(currentTraces.image.exc, 2)) / 1e3;
    else
        warning('No time vector found. Using default.');
        t = 1:100;
    end
    
    hold on;
    if isfield(currentTraces, 'stim_start') && isfield(currentTraces, 'stim_end')
        xline(currentTraces.stim_start, 'k--');
        xline(currentTraces.stim_end, 'k--');
    end
    
    for i = 1:length(example_indices)
        idx = example_indices(i);
        if idx > 0
            if isfield(currentTraces, 'responses')
                resp = currentTraces.responses;
            else 
                resp = currentTraces;
            end
            switch traceType
                case 'exc'
                    if idx <= size(resp.image.exc, 1) && idx <= size(resp.disc.exc, 1)
                        plot(t, resp.image.exc(idx, :), '-', 'Color', example_colors(i, :), 'LineWidth', 2);
                        plot(t, resp.disc.exc(idx, :), '--', 'Color', example_colors(i, :), 'LineWidth', 1.5);
                    end
                case 'inh'
                    if idx <= size(resp.image.inh, 1) && idx <= size(resp.disc.inh, 1)
                        plot(t, resp.image.inh(idx, :), '-', 'Color', example_colors(i, :), 'LineWidth', 2);
                        plot(t, resp.disc.inh(idx, :), '--', 'Color', example_colors(i, :), 'LineWidth', 1.5);
                    end
                case 'spike'
                    if idx <= size(resp.image.exc, 1) && idx <= size(resp.disc.exc, 1) && ...
                       idx <= size(resp.image.inh, 1) && idx <= size(resp.disc.inh, 1)
                        combined_image = resp.image.exc(idx, :) - resp.image.inh(idx, :);
                        combined_disc = resp.disc.exc(idx, :) - resp.disc.inh(idx, :);
                        plot(t, combined_image, '-', 'Color', example_colors(i, :), 'LineWidth', 2);
                        plot(t, combined_disc, '--', 'Color', example_colors(i, :), 'LineWidth', 1.5);
                    end
            end
        end
    end
    xlabel('Time (s)'); ylabel('Response');
    title(titleText); grid on;
    hold off;
end

%--------------------------------------------------------------------------
function plotPeakScatterWithLabels(disc_data, image_data, titleStr, example_indices, example_colors)
    hold on;
    maxVal = max(max(disc_data), max(image_data));
    minVal = min(min(disc_data), min(image_data));
    range = maxVal - minVal;
    maxVal = maxVal + 0.1 * range;
    minVal = minVal - 0.1 * range;
    amp = max(abs(minVal), abs(maxVal));
    
    scatter(disc_data/amp, image_data/amp, 50, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha', 0.3);
    for i = 1:length(example_indices)
        idx = example_indices(i);
        scatter(disc_data(idx)/amp, image_data(idx)/amp, 100, example_colors(i, :), 'filled', 'MarkerFaceAlpha', 0.7);
        text(disc_data(idx)/amp + 0.02, image_data(idx)/amp, sprintf('%d', i), 'Color', example_colors(i, :), 'FontWeight', 'bold');
    end
    plot([minVal/amp, 1], [minVal/amp, 1], 'k--');
    xlabel('Disc Response'); ylabel('Image Response');
    title(titleStr); axis square; grid on;
    xlim([minVal/amp, maxVal/amp]); ylim([minVal/amp, maxVal/amp]);
    hold off;
end
