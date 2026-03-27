function NewEpochData = DeglitchEpochs(EpochData, params)

% deglitch
GlitchThresh = 0.5;
WindowSize = 20;

for epoch = 1:size(EpochData, 1)
    DiffEpochData = diff(EpochData(epoch, :));
    Indices = find(abs(DiffEpochData) > params.GlitchThresh);

    if (length(Indices) > 0)
        clear MaxLoc;
        for pnt = 1:length(Indices)
            if (Indices(pnt) > params.WindowSize)
                StartPnt = Indices(pnt) - params.WindowSize;
            else
                StartPnt = 1;
            end
            if ((Indices(pnt) + params.WindowSize) > size(EpochData,2))
                EndPnt = size(EpochData,2);
            else
                EndPnt = Indices(pnt) + params.WindowSize;
            end
            [MaxVal, MaxLoc(pnt)] = max(abs(EpochData(epoch, StartPnt:EndPnt)));
            MaxLoc(pnt) = MaxLoc(pnt) + Indices(pnt) - params.WindowSize;
        end

        GlitchIndices = unique(MaxLoc);

        for pnt = 1:length(GlitchIndices)
            if (GlitchIndices(pnt) > params.WindowSize)
                StartPnt = GlitchIndices(pnt) - params.WindowSize;
            else
                StartPnt = 1;
            end
            if ((GlitchIndices(pnt) + params.WindowSize) > size(EpochData,2))
                EndPnt = size(EpochData,2);
            else
                EndPnt = GlitchIndices(pnt) + params.WindowSize;
            end
            StartVal = EpochData(epoch, StartPnt);
            Slope = (EpochData(epoch, EndPnt) - EpochData(epoch, StartPnt)) / (2*params.WindowSize+1);
            Patch = StartVal + Slope * (1:2*params.WindowSize+1);
            EpochData(epoch, StartPnt:EndPnt) = Patch(1:EndPnt-StartPnt+1);
        end
    end
    
end

NewEpochData = EpochData;