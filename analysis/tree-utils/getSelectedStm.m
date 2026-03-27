function epochStm = getSelectedStm(epochList, streamName, ResampleFactor)

if (strcmp(epochList.elements(1).protocolSettings.get('notes:protocolName'), 'Spatial stimulus'))
    StimulusString = strcat('stimuli:', streamName);    
    if (epochList.valueByIndex(1).protocolSettings.(strcat(StimulusString,':stimClass')) == 'SplitFieldNoiseStimulus')
        TotPts = epochList.valueByIndex(1).protocolSettings.('acquirino:dataPoints');
        StimFrames = epochList.valueByIndex(1).protocolSettings.(strcat(StimulusString,':spatial_stimpts'));
        PreFrames = epochList.valueByIndex(1).protocolSettings.(strcat(StimulusString,':spatial_prepts'));
        PostFrames = epochList.valueByIndex(1).protocolSettings.(strcat(StimulusString,':spatial_postpts'));
        RanSeed = epochList.valueByIndex(1).protocolSettings.(strcat(StimulusString,':randSeed'));
        StimSD = epochList.valueByIndex(1).protocolSettings.(strcat(StimulusString,':stimStd'));
        FrameDwell = epochList.valueByIndex(1).protocolSettings.(strcat(StimulusString,':frameDwell'));
        SamplingInterval = epochList.valueByIndex(1).protocolSettings.('acquirino:samplingInterval') * 1.25e-6;
        FlipInterval = epochList.valueByIndex(1).protocolSettings.(strcat(StimulusString,':flipInterval'));
        for epoch = 1:epochList.length
            temp = getSplitFieldStimulus(RanSeed, StimSD, PreFrames, StimFrames, FrameDwell, PostFrames);
            tempStm(epoch, :) = resample(temp, round(ResampleFactor*100), 100);
        end
    else
        fprintf(1, 'Stimulus type not supported\n');
    end
else
    tempStm = epochList.stimuliByStreamName(streamName);
end

for epoch = 1:epochList.length
    isSelected(epoch) = epochList.valueByIndex(epoch).isSelected;
end

sum(isSelected)

selectedEpochs = find(isSelected == 1);
epochStm = tempStm(selectedEpochs, :);

end