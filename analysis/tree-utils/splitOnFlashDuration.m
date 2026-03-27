function V = splitOnFlashDuration(epoch)
if epoch.protocolSettings.keySet.contains('flashDuration')
    V = strcat('flashDuration ', epoch.protocolSettings.get('flashDuration'));
else 
    V='no flash duration info';
end
    
