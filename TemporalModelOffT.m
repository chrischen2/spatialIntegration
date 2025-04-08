% OffTCRGPiecewise.m - Updated version with preStrength and beta comparisons
%   Updated by incorporating simulation loop and function naming conventions
%   from STRFOffTSimplified.m (see :contentReference[oaicite:1]{index=1})
clc; clearvars; close all;

%% Model parameters
modelParams = struct(...
    'K_rec', 10, ...           % Recovery/restocking rate for synaptic vesicles (Hz)
    'K_rel', 5, ...            % Release rate
    'b', 5, ...                % Scaling factor 
    'exc_0', 0.3, ...          % Baseline excitation
    'inhStrength', 0.9, ...    % Maximum release rate reduction
    'beta', 1, ...             % Sensitivity of release rate to inhibition 
    'excRectRatio', 0.4, ...      % Rectification ratio of excitation
    'inhRectRatio', 0); 
modelParams.n_0 = 1/(1 + modelParams.b * modelParams.exc_0 * modelParams.K_rel / modelParams.K_rec);

%% Time parameters
stimParams = struct(...
    'sampleRate', 1000, ...    % Sampling rate in Hz
    'preDuration', 0.3, ...    % Pre-stimulus period in seconds
    'stimDuration', 2, ...   % Stimulus duration in seconds
    'postDuration', 0.3 ...    % Post-stimulus period in seconds
);
stimParams.dt = 1/stimParams.sampleRate;
stimParams.t = (0:stimParams.dt:(stimParams.preDuration + stimParams.stimDuration + stimParams.postDuration - stimParams.dt))';

%% Load and prepare filters
% Load and prepare temporal filters
downsampleFactor = 10;  % Downsample factor: from 10 kHz to 1 kHz
nl = load('OffTNonlinearity.mat', 'exc', 'inh');

% Smooth the temporal filters using a window scaled by the downsample factor
excFilter = smooth(-nl.exc.temporalFilter(:), 5 * downsampleFactor);  % filter flipped to push polarity in the temporal filter
inhFilter = smooth(nl.inh.temporalFilter(:), 5 * downsampleFactor);

% Downsample the filters
excFilter_ds = excFilter(1:downsampleFactor:end)';
excFilter_ds = excFilter_ds / abs(sum(excFilter_ds));
inhFilter_ds = inhFilter(1:downsampleFactor:end)';
inhFilter_ds = inhFilter_ds / abs(sum(inhFilter_ds));

figure; plot(excFilter_ds); hold all;  plot(inhFilter_ds);
%% Stimulus definition -- 
% % 1. using split-field grating
% stim1 = [zeros(stimParams.preDuration * stimParams.sampleRate, 1);
%          0.9 * ones(stimParams.stimDuration * stimParams.sampleRate, 1);
%          zeros(stimParams.postDuration * stimParams.sampleRate, 1)];
% stim2 = [zeros(stimParams.preDuration * stimParams.sampleRate, 1);
%          -0.9 * ones(stimParams.stimDuration * stimParams.sampleRate, 1);
%          zeros(stimParams.postDuration * stimParams.sampleRate, 1)];
% 

% 2. Sinusoidal stimulus
tf = 2; % Hz
stim_t = stimParams.t(stimParams.t >= stimParams.preDuration & ...
    stimParams.t < stimParams.preDuration + stimParams.stimDuration) - stimParams.preDuration;
stim1 = [zeros(stimParams.preDuration * stimParams.sampleRate, 1);
    sin(2*pi*tf*stim_t);
    zeros(stimParams.postDuration * stimParams.sampleRate, 1)];
stim2 = [zeros(stimParams.preDuration * stimParams.sampleRate, 1);
    -sin(2*pi*tf*stim_t);
    zeros(stimParams.postDuration * stimParams.sampleRate, 1)];

% Ensure correct dimensions for filters and stimuli
if ~isequal(size(excFilter_ds), size(stim1))
    stim1 = stim1(:)';
    stim2 = stim2(:)';
end

%% Create figures for analyses
figHandles.mainFig = figure('Position', [100 100 1200 500]);

% Top subplot: Response
subplot(2,1,1);
ax1 = gca;
hold(ax1, 'on');
title(ax1, 'Response');
xlabel(ax1, 'Time (s)');
ylabel(ax1, 'Response');

% Bottom subplot: Vesicle Occupancy
subplot(2,1,2);
ax2 = gca;
hold(ax2, 'on');
title(ax2, 'Vesicle Occupancy');
xlabel(ax2, 'Time (s)');
ylabel(ax2, 'n(t)');

%% Process signals through temporal filters
exc = temporal_filter_signal(stim1(:)', excFilter_ds);
exc2 = temporal_filter_signal(stim2(:)', excFilter_ds);
exc = modelParams.exc_0 + exc;
exc2 = modelParams.exc_0 + exc2;

inh = temporal_filter_signal(stim1(:)', inhFilter_ds);
inh2 = temporal_filter_signal(stim2(:)', inhFilter_ds);
inh = applyPiecewiseNonlinearity(inh, modelParams.inhRectRatio);
inh2 = applyPiecewiseNonlinearity(inh2, modelParams.inhRectRatio);

%% First simulation: Varying preStrength
% Compute total inhibition from both channels
total_inh = inh + inh2;

% Here the synaptic modulation is computed directly (as in STRFOffTSimplified.m)
% rather than passing through a slow inhibition dynamics
synapseMod = @(preStrength) preStrength * total_inh; 

preStrengthValues = 0.1:0.2:1;
colors_preStrength = pmkmp(length(preStrengthValues),'IsoL'); 

% Pre-allocate graphic handles and legend entries
resp_lines = gobjects(length(preStrengthValues), 1);
n1_lines = gobjects(length(preStrengthValues), 1);
n2_lines = gobjects(length(preStrengthValues), 1);
legend_entries_resp = cell(length(preStrengthValues) + 1, 1);
legend_entries_n = cell(2 * length(preStrengthValues), 1);

for i = 1:length(preStrengthValues)
    preStrength = preStrengthValues(i);
    % Run simulation using the updated runSimulation (note: synapseMod used directly)
    [total_exc, n1, n2, total_inh_sim] = runSimulation(exc, exc2, preStrength, modelParams, stimParams, synapseMod);
    
    % Plot responses
    subplot(2,1,1);
    resp_lines(i) = plot(stimParams.t, total_exc, 'Color', colors_preStrength(i,:), 'LineWidth', 2);
    legend_entries_resp{i} = sprintf('preStr=%.1f', preStrength);
    
    % Plot vesicle occupancy for both channels
    subplot(2,1,2);
    n1_lines(i) = plot(stimParams.t, n1, 'Color', colors_preStrength(i,:), 'LineWidth', 2, 'LineStyle', '-');
    n2_lines(i) = plot(stimParams.t, n2, 'Color', colors_preStrength(i,:), 'LineWidth', 2, 'LineStyle', '--');
    
    legend_entries_n{i} = sprintf('n1 (preStr=%.1f)', preStrength);
    legend_entries_n{i + length(preStrengthValues)} = sprintf('n2 (preStr=%.1f)', preStrength);
end

% Add inhibition trace to response plot for context
subplot(2,1,1);
inh_line = plot(stimParams.t, total_inh/3, '--k', 'LineWidth', 2);
legend_entries_resp{end} = 'Inh';

% Add legends
subplot(2,1,1);
legend([resp_lines; inh_line], legend_entries_resp, 'Location', 'eastoutside');
subplot(2,1,2);
legend([n1_lines; n2_lines], legend_entries_n, 'Location', 'eastoutside');
sgtitle('Presyn Strength Analysis', 'FontSize', 14);

%% Second simulation: Inhibition sensitivity (beta analysis)
figHandles.betaFig = figure('Position', [100 600 1200 500]);
betaValues = [1 2 4 8 16];
colors_beta = pmkmp(length(betaValues),'IsoL');

subplot(2,1,1);
ax3 = gca;
hold(ax3, 'on');
title(ax3, 'Response');
xlabel(ax3, 'Time (s)');
ylabel(ax3, 'Response');

subplot(2,1,2);
ax4 = gca;
hold(ax4, 'on');
title(ax4, 'Vesicle Occupancy');
xlabel(ax4, 'Time (s)');
ylabel(ax4, 'n(t)');

% Use fixed preStrength for beta analysis
preStrength = 0.9;
original_beta = modelParams.beta;

% Plot excitation for different beta values
subplot(2,1,1);
total_lines = [];
for i = 1:length(betaValues)
    modelParams.beta = betaValues(i);
    [total_exc, ~, ~, total_inh_sim] = runSimulation(exc, exc2, preStrength, modelParams, stimParams, synapseMod);
    total_lines(i) = plot(stimParams.t, total_exc, 'Color', colors_beta(i,:), 'LineWidth', 2);
end
inh_line = plot(stimParams.t, total_inh/3, '--k', 'LineWidth', 2);
legend_entries = cell(length(betaValues) + 1, 1);
for i = 1:length(betaValues)
    legend_entries{i} = sprintf('\\beta=%.0f', betaValues(i));
end
legend_entries{end} = 'Inh';
legend([total_lines, inh_line], legend_entries, 'Location', 'eastoutside');

% Plot vesicle occupancy for different beta values
subplot(2,1,2);
all_lines = [];
all_labels = {};
for i = 1:length(betaValues)
    modelParams.beta = betaValues(i);
    [~, n1, n2, ~] = runSimulation(exc, exc2, preStrength, modelParams, stimParams, synapseMod);
    
    line1 = plot(stimParams.t, n1, '-', 'Color', colors_beta(i,:), 'LineWidth', 2);
    line2 = plot(stimParams.t, n2, '--', 'Color', colors_beta(i,:), 'LineWidth', 2);
    
    all_lines = [all_lines; line1; line2];
    all_labels = [all_labels; sprintf('n1 (\\beta=%.0f)', betaValues(i)); ...
                            sprintf('n2 (\\beta=%.0f)', betaValues(i))];
end
legend(all_lines, all_labels, 'Location', 'eastoutside');
sgtitle('Inh Sensitivity (\beta) Analysis', 'FontSize', 14);
modelParams.beta = original_beta;  % Restore original beta value

%% Helper Functions

function [total_exc, n1, n2, total_inh_sim] = runSimulation(exc, exc2, preStrength, params, stimParams, synapseModFcn)
    % In the updated simulation, synaptic modulation is applied instantaneously.
    total_inh_sim = []; % Unused in simulation loop (can be used for extended analysis)
    
    % Compute the synaptic modulation directly from the total inhibition
    synMod = synapseModFcn(preStrength); 
    
    % Initialize state variables
    n1 = zeros(size(stimParams.t));
    n2 = zeros(size(stimParams.t));
    n1(1) = params.n_0;
    n2(1) = params.n_0;
    exc_response = zeros(size(stimParams.t));
    exc_response2 = zeros(size(stimParams.t));
    exc_response(1) = params.exc_0;
    exc_response2(1) = params.exc_0;
    
    % Simulation loop: use instantaneous synaptic modulation (synMod)
    for k = 2:length(stimParams.t)
        % Subtract instantaneous inhibition from excitation inputs
        exc_input = exc(k) - synMod(k);
        exc_input2 = exc2(k) - synMod(k);
        
        % Compute the effective response (including a leak term if any)
        resp1 = exc_input - params.exc_0;
        resp2 = exc_input2 - params.exc_0;
        
        % Apply piecewise nonlinearity (updated function name)
        exc_response(k) = applyPiecewiseNonlinearity(resp1, params.excRectRatio) * n1(k-1) + params.exc_0;
        exc_response2(k) = applyPiecewiseNonlinearity(resp2, params.excRectRatio) * n2(k-1) + params.exc_0;
        
        % Compute change in vesicle occupancy using instantaneous synaptic modulation
        dn1dt = compute_dndt(n1(k-1), exc_response(k), params.K_rec, params.K_rel, params.b, synMod(k), params.beta);
        dn2dt = compute_dndt(n2(k-1), exc_response2(k), params.K_rec, params.K_rel, params.b, synMod(k), params.beta);
        
        n1(k) = max(0, min(1, n1(k-1) + dn1dt * stimParams.dt));
        n2(k) = max(0, min(1, n2(k-1) + dn2dt * stimParams.dt));
    end
    
    total_exc = (exc_response + exc_response2)/2;
end

% Updated compute_dndt function; identical in operation to STRFOffTSimplified.m
function dndt = compute_dndt(n, Exc, K_rec, K_rel, b, inhAmp, beta)
    dndt = (1 - n) * K_rec - b * K_rel * n * Exc * exp(-beta * inhAmp);
end

% Updated function name to match STRFOffTSimplified.m convention
function y = applyPiecewiseNonlinearity(x, ratio)
    y = x;
    neg_idx = x < 0;
    y(neg_idx) = x(neg_idx) * ratio;
end

% Convolution-based temporal filtering (remains similar to original)
function conv_result = temporal_filter_signal(signal, filter)
    filterLen = length(filter);
    normFactor = sum(abs(filter));
    normalizedFilter = filter / normFactor;
    
    paddedSignal = [zeros(1, filterLen) signal zeros(1, filterLen)];
    convResult = zeros(1, length(paddedSignal));
    for t = 1:length(paddedSignal)-filterLen+1
        convResult(t+filterLen-1) = sum(paddedSignal(t:t+filterLen-1) .* flip(normalizedFilter));
    end
    conv_result = convResult(filterLen+1:end-filterLen);
end
