function epochStm = getSelectedStmOvation(epochList, streamName)


tempStm = riekesuite.getStimulusMatrix(epochList, streamName);

for epoch = 1:epochList.length
    isSelected(epoch) = epochList.valueByIndex(epoch).isSelected;
end

selectedEpochs = find(isSelected == 1);
epochStm = tempStm(selectedEpochs, :);

end