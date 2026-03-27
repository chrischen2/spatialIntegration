function cmap = pmkmp(n, scheme)
% pmkmp  Attempt to generate perceptually uniform colormaps.
%   cmap = pmkmp(n, scheme)
%   Attempt to create perceptually uniform colormaps originally from
%   Matteo Niccoli's MATLAB File Exchange submission.
%   This is a simplified local replacement using MATLAB's built-in parula
%   colormap for the 'IsoL' scheme, which provides a perceptually uniform
%   alternative suitable for publication figures.
%
%   Inputs:
%       n      - number of colors
%       scheme - colormap name (only 'IsoL' is used in this codebase)
%
%   Output:
%       cmap - n x 3 colormap matrix

if nargin < 2, scheme = 'IsoL'; end
if nargin < 1, n = 256; end

switch lower(scheme)
    case 'isol'
        % Use parula as a perceptually uniform replacement
        cmap = parula(n);
    otherwise
        cmap = parula(n);
end
end
