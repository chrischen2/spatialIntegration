function V = splitOnStimTypeOrBarWidth(epoch)
    if epoch.protocolSettings.keySet.contains('stimulus type')
        V = epoch.protocolSettings('stimulus type');
    elseif epoch.protocolSettings.keySet.contains('currentBarWidth')
        V = epoch.protocolSettings('currentBarWidth');
    else 
        V=1;
    end
end
