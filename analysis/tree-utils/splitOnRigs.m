function [v] = splitOnRigs(epoch)

if epoch.protocolSettings.keySet.contains('experiment:rig')
   v=epoch.protocolSettings('experiment:rig');
else
    v='unknown rig';
end