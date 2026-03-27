function [v] = sortOnStimulusIndex(epoch)
if ~epoch.protocolSettings.keySet.contains('zeroMeanStep')
    index=mod(epoch.protocolSettings('stimulusIndex')-1, 3 + ...,
        2*length(epoch.protocolSettings('variableFlashTime'))) + 1;
    if  index<3        
        v='response to grates only';
    elseif index==3
        v='response to background steps';
    else
        v='now the real stuff';
    end
else
     index=mod(epoch.protocolSettings('stimulusIndex')-1, 1 + ...,
        2*length(epoch.protocolSettings('variableFlashTime'))) + 1;
    if index<2
        v='response to background steps';
    else
        v='now the real stuff';    
    end
end

