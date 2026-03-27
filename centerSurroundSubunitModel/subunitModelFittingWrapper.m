function [err] = subunitModelFittingWrapper(params, targetBarSize, targetBarResp)
% subunitModelFittingWrapper  Objective function for fitting the center-surround
%   subunit model to contrast-reversing grating data.
%   Paper reference: Figure 4F, Supplementary Figure 4 (Chen & Rieke, 2026)
%
%   params = [SurroundSD, surroundWeight, CenterSD, baseline]

SurroundSD     = params(1);
surroundWeight = params(2);
CenterSD       = params(3);
baseline       = params(4);

positions = 0:10:500;
xloc = 0:0.1:500;
numShuffles = 10;

% Scale surround weight by center/surround ratio
scaledSurroundWeight = surroundWeight * CenterSD / SurroundSD;

% Resample target data to denser bar widths for fitting
denseBar = [min(targetBarSize):5:51, 60:10:max(targetBarSize)];
targetBarResp = interp1(targetBarSize, targetBarResp, denseBar);

% Compute model response using shared core function
barResp = computeSubunitResponse(CenterSD, SurroundSD, scaledSurroundWeight, ...
    denseBar, positions, xloc, numShuffles);

% Normalize and add baseline
barResp = barResp ./ max(barResp);
barResp = barResp + baseline;
barResp = barResp ./ max(barResp);

% Compute normalized sum-of-squares error
err = sum((targetBarResp - barResp).^2) / sum(targetBarResp.^2);
end
