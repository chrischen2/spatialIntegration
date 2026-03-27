function [err] = subunitModelFittingWrapper(params,targetBarSize, targetBarResp)
SurroundSD=params(1); surroundWeight=params(2);   CenterSD=params(3);   baseline=params(4);
 

positions = 0:10:500; 
xloc = 0:0.1:500;

surroundWeight =surroundWeight* CenterSD/SurroundSD;

for sub = 1:length(positions)
    GaussRF(sub, :) = exp(-(xloc - positions(sub)).^2/(2*CenterSD^2)) - surroundWeight * exp(-(xloc - positions(sub)).^2/(2*SurroundSD^2));
end


% Bar stimulus

numShuffles = 10;
% resample the bar sizes 
%%%%%%%%%%%%%%%%
denseBar=[min(targetBarSize):5:51 60:10:max(targetBarSize)];
targetBarResp=interp1(targetBarSize, targetBarResp, denseBar); 
barWidth=denseBar;  
%%%%%%%%%%%%%%%%
barResp = zeros(1, length(barWidth));
for width = 1:length(barWidth)
    barStim = sign(sin(2*pi.*xloc./(barWidth(width)*2)));
    for shuffle = 1:numShuffles
        tempBarStim = circshift(barStim, randi(barWidth(width)));
        for sub = 1:length(positions)
            subResp = sum(GaussRF(sub, :) .* tempBarStim);
            if (subResp > 0)
                barResp(width) = barResp(width) + subResp;
            end
        end
    end
end
barResp=barResp./max(barResp);
barResp=barResp+baseline;
barResp=barResp./max(barResp);
err= sum((targetBarResp - barResp).^2) / sum(targetBarResp.^2);
end

