function V = splitOnRecordingType(epoch)
if epoch.protocolSettings.keySet.contains('epochGroup:recordingTechnique')
    V = epoch.protocolSettings.get('epochGroup:recordingTechnique');
elseif epoch.protocolSettings.keySet.contains('psth')
    if epoch.protocolSettings('psth')
        V='cell-attached';
    else
        V='whole-cell';
    end
end
end

