function V = splitOnSplitField( epoch )
% online analysis extracellular and none will be set as cell attach
%   Detailed explanation goes here
     V = epoch.protocolSettings.get('splitField');
     if V==1
         V='split-field';
     else 
         V='full-field';
     end
end
