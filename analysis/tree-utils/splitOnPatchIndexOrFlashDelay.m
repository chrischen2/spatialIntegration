function V = splitOnPatchIndexOrFlashDelay(epoch)
    if epoch.protocolSettings.keySet.contains('currentPatchIndex')
        V = strcat('patchIndex--',num2str(epoch.protocolSettings('currentPatchIndex')));
    elseif epoch.protocolSettings.keySet.contains('currentFlashDelay')
        V = strcat('flashDelay--',num2str(epoch.protocolSettings('currentFlashDelay')));
    else 
        V='something is wrong';
    end
end
