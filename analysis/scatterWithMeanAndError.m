function [ax] = scatterWithMeanAndError(g, y, my, ey, gName, showScatter, jitterAmount)
% SCATTERWITHMEANANDERROR Creates a scatter plot with mean and error bars
%   g:           Group indices
%   y:           Individual data points
%   my:          Mean values for each group
%   ey:          Error values for each group
%   gName:       Names for the groups (for x-tick labels)
%   showScatter: Flag to show individual data points (true/false)
%   jitterAmount: Standard deviation of the Gaussian jitter (default: 0.05)
%
%   Returns: ax - The axis handle

    % Set default jitter amount if not provided
    if nargin < 7 || isempty(jitterAmount)
        jitterAmount = 0.1;
    end

    if showScatter
        % Apply Gaussian jitter to x-coordinates
        jitter = randn(size(g)) * jitterAmount;
        scatter(g + jitter, y, 100, 'MarkerEdgeColor', [0.5 0.5 0.5], 'MarkerFaceColor', [0.5 0.5 0.5]); 
        hold all;
        offset = 0.02;
    else
        offset = 0;
    end
    
    eb = errorbar(unique(g) + offset, my, ey, 'vertical', 'LineStyle', 'none');
    scatter(unique(g) + offset, my);
    set(eb, 'color', 'r', 'LineWidth', 2)
    set(gca, 'xtick', unique(g), 'xtickLabel', gName);
    ax = gca;
    xlim([0.4 max(unique(g)) + 0.4]);
end