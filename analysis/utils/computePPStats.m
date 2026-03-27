function [means, sems, counts, allIntervals] = computePPStats(ppSpotsInterval, cellIdxs, valueFcn)
% computePPStats  Compute mean, SEM, and count across intervals for paired-pulse data.
%   [means, sems, counts, allIntervals] = computePPStats(ppSpotsInterval, cellIdxs, valueFcn)
%   Paper reference: Figure 7A-H (Chen & Rieke, 2026)

    allIntervals = unique(cell2mat(arrayfun(@(x) x.intervalArray, ppSpotsInterval(cellIdxs), 'UniformOutput', false)));
    allIntervals = sort(allIntervals);
    means = zeros(size(allIntervals));
    sems = zeros(size(allIntervals));
    counts = zeros(size(allIntervals));
    for intIdx = 1:length(allIntervals)
        currentInterval = allIntervals(intIdx);
        values = [];
        for cellIdx = cellIdxs
            intervalIdx = find(ppSpotsInterval(cellIdx).intervalArray == currentInterval);
            if ~isempty(intervalIdx)
                values = [values, valueFcn(ppSpotsInterval(cellIdx), intervalIdx)];
            end
        end
        if ~isempty(values)
            means(intIdx) = mean(values);
            sems(intIdx) = std(values) / sqrt(length(values));
            counts(intIdx) = length(values);
        end
    end
end
