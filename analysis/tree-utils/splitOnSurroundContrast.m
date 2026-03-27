function [v] = splitOnSurroundContrast(epoch)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
if epoch.protocolSettings.keySet.contains('currentSurroundContrast')
    
    v=epoch.protocolSettings('currentSurroundContrast');
elseif  epoch.protocolSettings.keySet.contains('currentSurroundStepContrast')
    v=epoch.protocolSettings('currentSurroundStepContrast');
    
else
    v='not the right contrast';
end
end

