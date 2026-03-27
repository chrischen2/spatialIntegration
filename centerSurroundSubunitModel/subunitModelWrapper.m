function [barResp] = subunitModelWrapper(params, barWidth)
% subunitModelWrapper  Compute normalized subunit model response for given parameters.
%   Paper reference: Figure 4F, Supplementary Figure 4 (Chen & Rieke, 2026)
%
%   params = [SurroundSD, surroundWeight, CenterSD, baseline]

SurroundSD     = params(1);
surroundWeight = params(2);
CenterSD       = params(3);
baseline       = params(4);

positions = 0:10:500;
xloc = 0:0.1:500;
numShuffles = 5;

% Scale surround weight by center/surround ratio
scaledSurroundWeight = surroundWeight * CenterSD / SurroundSD;

% Compute model response using shared core function
barResp = computeSubunitResponse(CenterSD, SurroundSD, scaledSurroundWeight, ...
    barWidth, positions, xloc, numShuffles);

% Normalize and add baseline
barResp = barResp ./ max(barResp);
barResp = barResp + baseline;
barResp = barResp ./ max(barResp);
end
