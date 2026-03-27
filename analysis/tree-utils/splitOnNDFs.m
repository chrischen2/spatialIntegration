function [v] = splitOnNDFs(epoch)
    % splitOnNDFs Extracts and combines NDF information while handling FW and non-FW NDFs separately
    % Handles cell arrays of NDFs and filter wheel values
    
    % Initialize empty cell arrays for collecting different types of NDFs
    nonFwNdfs = {};
    fwNdfs = {};
    
    % Helper function to parse NDF values and filter out FW entries
    function ndfs = parseNDFValue(ndfVal)
        if isempty(ndfVal) || strcmp(ndfVal, '[]')
            ndfs = {};
            return;
        end
        
        % If it's already a cell array
        if iscell(ndfVal)
            rawNdfs = ndfVal;
        % If it's a JSON-style string array
        elseif ischar(ndfVal) && contains(ndfVal, '[')
            % Remove brackets and quotes, split on commas
            ndfVal = strrep(ndfVal, '[', '');
            ndfVal = strrep(ndfVal, ']', '');
            ndfVal = strrep(ndfVal, '"', '');
            % Split and trim
            parts = strtrim(split(ndfVal, ','));
            rawNdfs = parts(~cellfun(@isempty, parts));
        else
            rawNdfs = {ndfVal};
        end
        
        % Filter out any FW entries (case insensitive)
        ndfs = {};
        for i = 1:length(rawNdfs)
            if ~startsWith(lower(rawNdfs{i}), 'fw')
                ndfs{end+1} = rawNdfs{i};
            end
        end
    end
    
    % Check various LED and device NDFs
    ndfSources = {
        'stimulus:UV_LED:ndfs'
        'stimulus:Blue_LED:ndfs'
        'stimulus:Red_LED:ndfs'
        'background:UV_LED:ndfs'
        'background:Blue_LED:ndfs'
        'background:Red_LED:ndfs'
        'background:Microdisplay_Stage@localhost:ndfs'
        'background:LightCrafter_Stage@localhost:ndfs'
    };
    
    % Process each NDF source for non-FW NDFs
    for i = 1:length(ndfSources)
        try
            if epoch.protocolSettings.keySet.contains(ndfSources{i})
                currentNdfs = parseNDFValue(epoch.protocolSettings(ndfSources{i}));
                if ~isempty(currentNdfs)
                    nonFwNdfs = [nonFwNdfs; currentNdfs(:)]; % Ensure column cell array
                end
            end
        catch
            continue;
        end
    end
    
    % Check filter wheel NDF separately
    if epoch.protocolSettings.keySet.contains('background:FilterWheel:NDF')
        filterWheelNDF = epoch.protocolSettings('background:FilterWheel:NDF');
        if ~isempty(filterWheelNDF)
            fwNdfs{end+1} = ['FW' num2str(filterWheelNDF)];
        end
    end
    
    % Combine all NDFs
    allNdfs = [nonFwNdfs; fwNdfs];
    
    % If no valid NDFs found
    if isempty(allNdfs)
        v = 'wrong device info';
        return;
    end
    
    % Remove duplicates while preserving order
    [~, uniqueIdx] = unique(allNdfs, 'stable');
    allNdfs = allNdfs(sort(uniqueIdx));
    
    % Convert to string with proper format
    v = strjoin(allNdfs, '+');
end