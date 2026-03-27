function [v] = splitOnZeroMean(epoch)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
if epoch.protocolSettings.keySet.contains('zeroMeanStep')
    if  epoch.protocolSettings('zeroMeanStep')
        v='new protocol: zero mean';
    else
        v='new protocol: bar';
    end
elseif  epoch.protocolSettings.keySet.contains('zeroMean')
    if  epoch.protocolSettings('zeroMean')
        v='chris protocol: zero mean';
    else
        v='chris protocol: bar';
    end
elseif epoch.protocolSettings.keySet.contains('centerZeroMean')
    if  epoch.protocolSettings('centerZeroMean')
        v='surround adapt protocol: zero mean';
    else
        v='surround adapt protocol: bar';
    end
else
    v='old protocol/no zero mean option';
end
end

