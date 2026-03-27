function f = linedScatterplot(data, groups, groupName, figTitle, legendNames)
% linedScatterplot - Create a line-scatter plot with error bars.
%
% USAGE:
%   f = linedScatterplot(data, groups, groupName, figTitle)
%   f = linedScatterplot(data, groups, groupName, figTitle, legendNames)
%
% INPUTS:
%   data: Either a numeric matrix (m x n) or a cell array of m x n matrices.
%         For numeric data: m = number of groups (e.g., contrasts) and n = observations (cells).
%         For a cell array: each cell is plotted separately.
%   groups: Vector of x positions (e.g., contrast values in %) for each group.
%   groupName: Cell array of strings for the x-axis tick labels.
%   figTitle: Title for the figure.
%   legendNames (optional): Cell array of names to label each dataset.
%
% This function plots error bars (mean ± SEM) and overlays the individual data
% points (with a slight horizontal jitter). It uses nanmean and nanstd so that NaN values are ignored.
    
    f = figure('Color','w');
    ax = axes('NextPlot','add','FontSize',16,'TickDir','out');
    
    if iscell(data)
        numSets = length(data);
        colors = lines(numSets);
        legendHandles = gobjects(numSets,1);
        for k = 1:numSets
            currData = data{k};
            % Use nanmean and nanstd to ignore NaN values.
            meanData = nanmean(currData, 2);
            semData = nanstd(currData, 0, 2) ./ sqrt(sum(~isnan(currData),2));
            h = errorbar(groups, meanData, semData, '-o', 'Color', colors(k,:),...
                'LineWidth',2, 'MarkerSize',8);
            legendHandles(k) = h;
            [numGroups, numPoints] = size(currData);
            for i = 1:numGroups
                jitterAmount = 0.05 * groups(i);
                xJitter = groups(i) + (rand(1,numPoints)-0.5) * jitterAmount;
                scatter(xJitter, currData(i,:), 64, 'MarkerEdgeColor', colors(k,:),...
                    'MarkerFaceColor','w','LineWidth',1.5);
            end
        end
        if nargin >= 5 && ~isempty(legendNames)
            legend(legendHandles, legendNames, 'Location','Best');
        else
            legend(legendHandles, arrayfun(@(x) sprintf('Set %d', x), (1:numSets)',...
                'UniformOutput', false), 'Location','Best');
        end
    else
        meanData = nanmean(data, 2);
        semData = nanstd(data, 0, 2) ./ sqrt(sum(~isnan(data),2));
        errorbar(groups, meanData, semData, '-o', 'LineWidth',2, 'MarkerSize',8, 'Color','b');
        [numGroups, numPoints] = size(data);
        for i = 1:numGroups
            jitterAmount = 0.05 * groups(i);
            xJitter = groups(i) + (rand(1,numPoints)-0.5) * jitterAmount;
            scatter(xJitter, data(i,:), 64, 'MarkerEdgeColor','k',...
                    'MarkerFaceColor','w','LineWidth',1.5);
        end
    end
    
    set(ax, 'XTick', groups, 'XTickLabel', groupName);
    xlabel('Contrast (%)');
    ylabel('Pulse Ratio (amp2/amp1)');
    title(figTitle);
    box on;
end
