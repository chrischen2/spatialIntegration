function V = splitOnDeviceBrightNess(epoch)
if epoch.protocolSettings.keySet.contains('background:Microdisplay_Stage@localhost:microdisplayBrightness')
    V = epoch.protocolSettings.get('background:Microdisplay_Stage@localhost:microdisplayBrightness');
else
    V='Lightcrafter High';
end
    
