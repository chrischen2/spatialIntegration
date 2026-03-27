function barResp = computeSubunitResponse(CenterSD, SurroundSD, surroundWeight, barWidth, positions, xloc, numShuffles)
% computeSubunitResponse  Compute rectified subunit responses to bar stimuli.
%   barResp = computeSubunitResponse(CenterSD, SurroundSD, surroundWeight,
%       barWidth, positions, xloc, numShuffles)
%
%   Builds a DoG (Difference of Gaussians) receptive field mosaic and
%   computes half-wave-rectified responses to square-wave bar stimuli of
%   varying widths, averaged over random spatial phase shifts.
%
%   This implements the center-surround subunit model described in the
%   Methods section of Chen & Rieke (2026), Figure 4F / Supp. Figure 4.
%   Each subunit RF is:
%       RF(x) = exp(-x^2/(2*sigmaC^2)) - delta*(sigmaC/sigmaS)*exp(-x^2/(2*sigmaS^2))
%   where delta is the relative surround strength (surroundWeight before
%   scaling by CenterSD/SurroundSD).
%
%   Inputs:
%       CenterSD       - center Gaussian sigma (um)
%       SurroundSD     - surround Gaussian sigma (um)
%       surroundWeight - surround strength (already scaled by CenterSD/SurroundSD)
%       barWidth       - vector of bar widths to test
%       positions      - subunit center positions
%       xloc           - spatial sampling positions
%       numShuffles    - number of random phase shifts per bar width
%
%   Output:
%       barResp - unnormalized response for each bar width (1 x nWidths)

% Build DoG receptive field for each subunit position (vectorized)
nSub = length(positions);
posMat = positions(:);            % nSub x 1
xMat   = xloc(:)';               % 1 x nX
dx     = posMat - xMat;          % nSub x nX (broadcast subtraction)
GaussRF = exp(-dx.^2 / (2*CenterSD^2)) ...
        - surroundWeight * exp(-dx.^2 / (2*SurroundSD^2));  % nSub x nX

% Compute responses to each bar width
barResp = zeros(1, length(barWidth));
for w = 1:length(barWidth)
    barStim = sign(sin(2*pi .* xloc ./ (barWidth(w)*2)));  % square-wave grating
    for s = 1:numShuffles
        tempBarStim = circshift(barStim, randi(barWidth(w)));
        % Vectorized: matrix multiply gives all subunit responses at once
        subResp = GaussRF * tempBarStim(:);  % nSub x 1
        % Half-wave rectification and sum
        barResp(w) = barResp(w) + sum(max(subResp, 0));
    end
end
end
