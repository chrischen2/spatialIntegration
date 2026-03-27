function V = splitOnBarWidthOrImgID(epoch)
    if epoch.protocolSettings.keySet.contains('imgID')
        V = epoch.protocolSettings('imgID');
    elseif epoch.protocolSettings.keySet.contains('currentBarWidth')
        V = strcat('barWidth--',num2str(epoch.protocolSettings('currentBarWidth')));
    else 
        V='no imgID or barWidth';
    end
end
