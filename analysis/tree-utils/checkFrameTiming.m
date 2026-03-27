function badFrames = checkFrameTiming(selectedNode)
    % checking the frame timing over the super long epoch 
    numElements = selectedNode.epochList.elements.length;
    badFrames = zeros(1, numElements);
    for i = 1:numElements
        if ~isempty(strfind(selectedNode.epochList.elements(i).keywords, 'badFrameTiming'))
            badFrames(i) = 1;
        end
    end
    if sum(badFrames) ~= 0
        fprintf('%s %s\n', 'bad frames as: epoch ', mat2str(find(badFrames == 1)));
    end
    badFrames=find(badFrames==1);
end
