function [err] = subunitModelSup(params)
SurroundSD=params(1); surroundWeight=params(2); CenterSD = 15;
positions = 0:10:500;
xloc = 0:0.1:600;
barWidth=10:5:160;
surroundWeight =surroundWeight* CenterSD/SurroundSD;

for sub = 1:length(positions)
    GaussRF(sub, :) = exp(-(xloc - positions(sub)).^2/(2*CenterSD^2)) - surroundWeight * exp(-(xloc - positions(sub)).^2/(2*SurroundSD^2));
end

% Bar stimulus
numShuffles =5;
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

%sup=barResp(end);
load('/Users/chrischen/Dropbox/research/projects/spatialIntegration/AllTuningReversingGrating.mat');
% compute the difference to the tuning curves of cell 1, 7, 16 
cellToFit=[1 6 7 8 13 15];
for c=1:numel(cellToFit)
    resampledResp=interp1(inhBarAll{cellToFit(c)}, inhBarRespAll{cellToFit(c)}, barWidth);
    err(c)=sum((resampledResp - barResp).^2) / sum(resampledResp.^2);
end

err=mean(err);