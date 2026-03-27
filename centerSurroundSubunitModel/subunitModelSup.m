function [err] = subunitModelSup(params)
% subunitModelSup  Objective function for parameter landscape exploration.
%   Computes mean fitting error across multiple cells for given surround
%   parameters, with center SD fixed at 15 um.
%   Paper reference: Figure 4F, Supplementary Figure 4 (Chen & Rieke, 2026)
%
%   params = [SurroundSD, surroundWeight]

SurroundSD     = params(1);
surroundWeight = params(2);
CenterSD       = 15;

positions = 0:10:500;
xloc = 0:0.1:600;
barWidth = 10:5:160;
numShuffles = 5;

% Scale surround weight by center/surround ratio
scaledSurroundWeight = surroundWeight * CenterSD / SurroundSD;

% Compute model response using shared core function
barResp = computeSubunitResponse(CenterSD, SurroundSD, scaledSurroundWeight, ...
    barWidth, positions, xloc, numShuffles);
barResp = barResp ./ max(barResp);

% Load population tuning data (relative path)
load('AllTuningReversingGrating.mat', 'inhBarAll', 'inhBarRespAll');

% Compute fitting error across selected cells
cellToFit = [1 6 7 8 13 15];
err = zeros(1, numel(cellToFit));
for c = 1:numel(cellToFit)
    resampledResp = interp1(inhBarAll{cellToFit(c)}, inhBarRespAll{cellToFit(c)}, barWidth);
    err(c) = sum((resampledResp - barResp).^2) / sum(resampledResp.^2);
end
err = mean(err);
end
