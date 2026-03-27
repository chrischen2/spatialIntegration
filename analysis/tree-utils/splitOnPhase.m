function V = splitOnPhase(epoch)
    if epoch.protocolSettings.keySet.contains('currentPhase')
        V = epoch.protocolSettings('currentPhase');
    else 
        V='no phase info';
    end
end
