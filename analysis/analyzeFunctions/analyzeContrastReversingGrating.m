function [output ] = analyzeContrastReversingGrating(selectedNodes, paras)
output=cell(1,numel(selectedNodes));

% Define time interval and amplitude
dt = 1/paras.sampleRate; % 0.1 ms
amplitude = 100;

% Calculate the total number of points
totalPts = paras.prePts + paras.stimPts + paras.tailPts;

% Create time vector
time = (0:totalPts-1) * dt;

% Create the sinusoid signal
sinusoid = zeros(1, totalPts);
stimTime = (paras.prePts+1):(paras.prePts+paras.stimPts);
sinusoid(stimTime) = amplitude * sin(2 * pi * paras.tempFreq * time(stimTime));


for node=1:numel(selectedNodes)
    resMat=riekesuite.getResponseMatrix(selectedNodes{node}.epochList,'Amp1');
    paras.epochRange=1:size(resMat,1);
    % find only exc trials or inh trials. ( for those that exc/inh is not
    % properly labelled
    % Check for blended EPSC and IPSC

    %         if strcmp(selectedNodes{node}.splitValue, 'exc')
    %                 paras.epochRange=find(mean(resMat,2)<150);
    %         elseif strcmp(selectedNodes{node}.splitValue, 'inh')
    % paras.epochRange=find(mean(resMat,2)<50);
    % resMat=resMat(paras.epochRange,:);
    %         end

    % if node==2
    %     resMat=resMat+300;
    % end
    barWidths=zeros(size(resMat,1),1);
    for i=1:size(resMat,1)
        if strcmp(selectedNodes{node}.splitValue, 'extracellular') || paras.spikeTag
            resMat(i,:)=resMat(i,:)-movmedian(resMat(i,:),100);
        else
            resMat(i,:)=smooth(resMat(i,:),200);
        end
        barWidths(i)=selectedNodes{node}.epochList.elements(paras.epochRange(i)).protocolSettings('currentBarWidth');
    end

    isBlended = false;
    if numel(find(mean(resMat, 2) < 0)) > 0 && numel(find(mean(resMat, 2) > 0)) > 0 && ~strcmp(selectedNodes{node}.splitValue, 'extracellular')
        fprintf('%s  \n', 'both EPSC and IPSC are blended in the node');
        isBlended = true;
    end


    if strcmp(selectedNodes{node}.splitValue, 'extracellular') || paras.spikeTag
        [resMat,~,emptyTrial,~]=smoothPSTH(resMat,paras.psthSigma, paras.sampleRate,paras.spikeTh);
        paras.epochRange(emptyTrial==1)=[];
        output{node}.onlineAnalysis='extracellular';
    else
        if mean2(resMat)<0
            output{node}.onlineAnalysis='exc';  % some recordings did not set the right onlineAnalysis
        else
            output{node}.onlineAnalysis='inh';
        end
        resMat=resMat-repmat(mean(resMat(:,1:paras.prePts),2),1,size(resMat,2));
    end

    if strcmp(selectedNodes{node}.splitValue, 'extracellular') || paras.spikeTag
        plotOffset=0;
    elseif strcmp(selectedNodes{node}.splitValue, 'exc')
        plotOffset=0;
    else
        plotOffset=0;
    end
    cycleOffset=0;
    barList=unique(barWidths);
    F1=zeros(numel(barList),1); F2=zeros(numel(barList),1);
    output{node}.meanRes=zeros(numel(barList),size(resMat,2));
    output{node}.isBlended = isBlended;

    noCycles=paras.tempFreq*paras.stimPts/(1e4);
    cycleLen=paras.stimPts/noCycles;
    cycleMean{node}=zeros(numel(barList), cycleLen);
    % cycleErr{node}=zeros(numel(barList), cycleLen);

    for i=1:numel(barList)
        barInd=find(barWidths==barList(i));
        tp=sum(resMat(barInd,:),1)/numel(barInd);
        output{node}.meanRes(i,:)=tp;

        if isBlended
            excInd = find(barWidths == barList(i) & mean(resMat, 2) < 0);
            inhInd = find(barWidths == barList(i) & mean(resMat, 2) > 0);
            if ~isempty(excInd)
                tpExc = sum(resMat(excInd, :), 1) / numel(excInd);
                output{node}.meanResExc(i, :) = tpExc;
            else
                fprintf('Error: No exc trials found for bar width %d\n', barList(i));
            end
            if ~isempty(inhInd)
                tpInh = sum(resMat(inhInd, :), 1) / numel(inhInd);
                output{node}.meanResInh(i, :) = tpInh;
            else
                fprintf('Error: No inh trials found for bar width %d\n', barList(i));
            end

            % Compute cycle average for blended data
            if ~isempty(excInd)
                cycleClipsExc = reshape(output{node}.meanResExc(i,paras.prePts+1+cycleOffset:paras.prePts+paras.stimPts+cycleOffset), cycleLen, noCycles)';
                cycleClipsExc(1, :) = [];
                output{node}.cycleMeanExc(i, :) = mean(cycleClipsExc);
            end
            if ~isempty(inhInd)
                cycleClipsInh = reshape(output{node}.meanResInh(i,paras.prePts+1+cycleOffset:paras.prePts+paras.stimPts+cycleOffset), cycleLen, noCycles)';
                cycleClipsInh(1, :) = [];
                output{node}.cycleMeanInh(i, :) = mean(cycleClipsInh);
            end

        end


        %     F2(i)=sum(tp(timeToPts(preTime)+1:timeToPts(preTime+stimTime)))/sampleRate;
        [F1(i), F2(i)]=computeF1F2(tp,paras.sampleRate,paras.tempFreq);
        % compute cycle average
        cycleClips=reshape(output{node}.meanRes(i,paras.prePts+1+cycleOffset:paras.prePts+paras.stimPts+cycleOffset),cycleLen,noCycles)'; cycleClips(1,:)=[];
        cycleMean{node}(i,:)=mean(cycleClips);
        % cycleErr{node}(i,:)=std(cycleClips)/sqrt(size(cycleClips,1));
    end

    % create examplary plot
    colors=pmkmp(numel(barList),'Isol');
    figure('position',[50 100 800 1200],'color','w');
    hold all;
    for i=1:numel(barList)
        plot(output{node}.meanRes(i,:)+plotOffset*(i-1),'color',colors(i,:),'DisplayName', ['Bar ', num2str(barList(i))]);
    end
    title([selectedNodes{node}.parent.parent.parent.splitValue ' ' selectedNodes{node}.splitValue]);
    legend('show');
    legend boxoff;
    F2=F2./max(F2);



    condStr{node}=[selectedNodes{node}.parent.parent.parent.splitValue ' ' selectedNodes{node}.splitValue];
    %     st=sgtitle(condStr{node}); set(st,'fontsize',24);
    denseF2=interp1(barList, F2, 1: max(barList));
    subUnitSize=find(denseF2 >= 0.5, 1, 'first');
    output{node}.barList=barList';
    %     output{node}.F2=F2'; output{node}.subUnitSize=subUnitSize;
    [~,supIndex]=min(abs((barList-120)));
    %     output{node}.suppress= F2(supIndex);
    drugAdded{node}=selectedNodes{node}.epochList.firstValue.protocolSettings('epochGroup:externalSolutionAdditions');
    fprintf('%s , %s \n', 'drugs added are ::', drugAdded{node});


    %%%fit with sinusoid and plot the tuning of fitting results
    %%% Fit with sinusoid and plot the tuning of fitting results
    f3 = figure('position', [850 300 1000 600], 'color', 'w');
    subplot(1, 2, 1);
    hold all;
    sinoF2Exc = zeros(1, numel(barList));
    sinoF2Inh = zeros(1, numel(barList));
    sinoF2 = zeros(1, numel(barList));

    for i = 1:numel(barList)
        if isBlended
            if isfield(output{node}, 'cycleMeanExc') && ~isempty(output{node}.cycleMeanExc)
                [sExc, xpExc, ypExc] = fitSinusoidFixedFreq((1:cycleLen) / 1e4, output{node}.cycleMeanExc(i, :), 2 * paras.tempFreq);
                sinoF2Exc(i) = abs(sExc(1));
                plot((1:cycleLen) / 1e4, output{node}.cycleMeanExc(i, :), 'color', colors(i, :),'linewidth', 3, 'DisplayName', ['Excitatory Bar ', num2str(barList(i))]);
            end
            if isfield(output{node}, 'cycleMeanInh') && ~isempty(output{node}.cycleMeanInh)
                [sInh, xpInh, ypInh] = fitSinusoidFixedFreq((1:cycleLen) / 1e4, output{node}.cycleMeanInh(i, :), 2 * paras.tempFreq);
                sinoF2Inh(i) = abs(sInh(1));
                plot((1:cycleLen) / 1e4, output{node}.cycleMeanInh(i, :), 'color', colors(i, :), 'linewidth', 3, 'DisplayName', ['Inhibitory Bar ', num2str(barList(i))]);
            end
        else
            [s, xp, yp] = fitSinusoidFixedFreq((1:cycleLen) / 1e4, cycleMean{node}(i, :), 2 * paras.tempFreq); % double freq, F2
            sinoF2(i) = abs(s(1)); % amplitude of sine curve
            plot((1:cycleLen) / 1e4, cycleMean{node}(i, :), 'color', colors(i, :), 'linewidth', 3, 'DisplayName', ['Bar ', num2str(barList(i))]);
        
        end
    end

    % Normalizing and plotting the sinusoid fitting results
    if isBlended
        if any(sinoF2Exc)
            % sinoF2Exc = sinoF2Exc / max(sinoF2Exc);
            output{1}.sinoF2Exc=sinoF2Exc; 
        end
        if any(sinoF2Inh)
            % sinoF2Inh = sinoF2Inh / max(sinoF2Inh);
            output{1}.sinoF2Inh=sinoF2Inh;
        end
    else
        % sinoF2 = sinoF2 / max(sinoF2);
        output{node}.sinoF2=sinoF2;
    end

    legend('show');
    legend boxoff;

    subplot(1, 2, 2);
    hold all;
    if isBlended
        if any(sinoF2Exc)
            h3Exc = line(barList, sinoF2Exc, 'Color', 'm', 'LineWidth', 2, 'Marker', 'o', 'markersize', 10, 'DisplayName', 'Excitatory');
        end
        if any(sinoF2Inh)
            h3Inh = line(barList, sinoF2Inh, 'Color', 'g', 'LineWidth', 2, 'Marker', 'o', 'markersize', 10, 'DisplayName', 'Inhibitory');
        end
    else
        h3 = line(barList, sinoF2, 'Color', 'k', 'LineWidth', 2, 'Marker', 'o', 'markersize', 10, 'DisplayName', 'Overall');
    end

    % ylim([0 1]);
    setAxes(f3);
    initFig(gca(f3), 'Bar width (um)', 'Norm Sine amp');

    if isBlended
        legend('show');
    else
        legend(h3, 'show');
    end
    legend boxoff;

    output{node}.sinoF2 = sinoF2;
    lgds{node} = output{node}.onlineAnalysis;


end



meanIntensity=selectedNodes{node}.epochList.firstValue.protocolSettings ...
    ('epoch:Microdisplay_Stage@localhost:white:rodConversionFactor')*paras.meanIntensity;
fprintf('%s , %f \n', 'mean Luminance::',meanIntensity');

timeStamps=(1:size(output{1}.meanRes,2))/paras.sampleRate;

% f1 = figure('position', [50 50 600 1200], 'color', 'w');
% hold all;
% f2 = figure('position', [50 50 900 900], 'color', 'w');
% hold all;
% f3 = figure('position', [50 50 1600 1200], 'color', 'w');
% hold all;

% for node = 1:numel(selectedNodes)
%     plot(gca(f1), barList, allF2(node, :), 'linewidth', 2);
%     legend(gca(f1), lgds);
%     legend boxoff;
%     for j = 1:numel(output{1}.exampleBar)
%         h2(node) = plot(gca(f2), timeStamps, output{node}.meanRes(output{1}.exampleBar(j), :) + 700 * (j - 1), 'linewidth', 2, 'color', colors(node));
%         if node == 1
%             tx = text(0.1, (j - 1) * 700 + 50, ['barWidth::', num2str(barList(output{1}.exampleBar(j)))]);
%             set(tx, 'fontsize', 15, 'color', 'k');
%         end
%         h3(node) = plot(gca(f3), (1:cycleLen) / 1e4, cycleMean{node}(output{1}.exampleBar(j), :), 'linewidth', 2, 'color', colors(node));
%     end
% end
% legend(gca(f2), h2, lgds);
% legend boxoff;
% legend(gca(f3), h3, lgds);
% legend boxoff;

if numel(selectedNodes) == 2
    numBars = numel(barList);
    figure('position', [50 50 800 600 * numBars], 'color', 'w');
    for i = 1:numBars
        subplot(numBars, 1, i);
        hold all;

        % Plot spike from the node with extracellular onlineAnalysis
        for node = 1:2
            if strcmp(output{node}.onlineAnalysis, 'extracellular')
                plot(timeStamps, output{node}.meanRes(i, :), 'linewidth', 1, 'color', 'b', 'DisplayName', 'extracellular');
            else
                if any(cellfun(@(x) x.isBlended, output))
                    if ~isempty(output{node}.meanResExc)
                        plot(timeStamps, output{node}.meanResExc(i, :), 'linewidth', 1, 'color', 'm', 'DisplayName', ['exc Node ', num2str(node)]);
                        plot(timeStamps, output{node}.meanResInh(i, :), 'linewidth', 1, 'color', 'g', 'DisplayName', ['inh Node ', num2str(node)]);
                    end
                else
                    plot(timeStamps, output{node}.meanRes(i, :), 'linewidth', 1, 'color', colors(node,:), 'DisplayName', [output{node}.onlineAnalysis num2str(node)]);
                end
            end
        end
        plot(timeStamps, sinusoid, 'k--','linewidth',2); % Add sinusoid to plot

        legend('show');
        xlabel('Time (s)');
        ylabel('Response');
        title(['Bar Width: ', num2str(barList(i))]);
        hold off;
    end
elseif numel(selectedNodes) == 3
    numBars = numel(barList);
    figure('position', [50 50 800 600 * numBars], 'color', 'w');
    for i = 1:numBars
        subplot(numBars, 1, i);
        hold all;

        % Plot the average trace for each bar width for all three nodes
        avgRes1 = output{1}.meanRes(i, :);
        avgRes2 = output{2}.meanRes(i, :);
        avgRes3 = output{3}.meanRes(i, :);

        plot(timeStamps, avgRes1, 'linewidth', 1, 'color', 'r', 'DisplayName', output{1}.onlineAnalysis);
        plot(timeStamps, avgRes2, 'linewidth', 1, 'color', 'b', 'DisplayName', output{2}.onlineAnalysis);
        plot(timeStamps, avgRes3, 'linewidth', 1, 'color', 'g', 'DisplayName', output{3}.onlineAnalysis);
        plot(timeStamps, sinusoid, 'k--','linewidth',2); % Add sinusoid to plot

        legend('show');
        xlabel('Time (s)');
        ylabel('Response');
        title(['Bar Width: ', num2str(barList(i))]);
        hold off;
    end

elseif numel(selectedNodes) == 1 && output{1}.isBlended
    numBars = numel(barList);
    figure('position', [50 50 800 600 * numBars], 'color', 'w');
    for i = 1:numBars
        subplot(numBars, 1, i);
        hold all;

        if ~isempty(output{1}.meanResExc)
            plot(timeStamps, output{1}.meanResExc(i, :), 'linewidth', 1, 'color', 'm', 'DisplayName', 'exc');
            plot(timeStamps, output{1}.meanResInh(i, :), 'linewidth', 1, 'color', 'g', 'DisplayName', 'inh');
        end
        plot(timeStamps, sinusoid, 'k--','linewidth',2); % Add sinusoid to plot

        legend('show');
        xlabel('Time (s)');
        ylabel('Response');
        title(['Bar Width: ', num2str(barList(i))]);
        hold off;
    end
end

% Comparing Excitatory and Inhibitory Traces
if any(cellfun(@(x) x.isBlended, output)) || (numel(selectedNodes) == 2 && any(strcmp({output{1}.onlineAnalysis, output{2}.onlineAnalysis}, 'exc')) ...
        && any(strcmp({output{1}.onlineAnalysis, output{2}.onlineAnalysis}, 'inh'))) ...
        || (numel(selectedNodes) == 3 && any(strcmp({output{1}.onlineAnalysis, output{2}.onlineAnalysis, output{3}.onlineAnalysis}, 'extracellular')))
    % Prepare subplots for temporal offset, amplitude ratio, and center of mass
    f1=figure('position', [850 300 1000 1200], 'color', 'w');
    tpEIRatio=zeros(numel(barList),2 * paras.tempFreq * paras.stimPts / paras.sampleRate);
    tpEIOffset=zeros(numel(barList),2 * paras.tempFreq * paras.stimPts / paras.sampleRate);
    maxLag = round(paras.sampleRate / (2*paras.tempFreq)); % Maximum lag in samples
    tpEICorr=zeros(numel(barList), 2*maxLag+1);
    for i = 1:numel(barList)
        if any(cellfun(@(x) x.isBlended, output))
            % Find the node with blended data
            blendedNode = find(cellfun(@(x) x.isBlended, output));
            excTrace = -output{blendedNode}.meanResExc(i, :); % Flip the excitatory trace
            inhTrace = output{blendedNode}.meanResInh(i, :);
        else
            % Find the nodes for exc and inh
            if numel(selectedNodes)==3
                excNode = find(strcmp({output{1}.onlineAnalysis, output{2}.onlineAnalysis, output{3}.onlineAnalysis}, 'exc'));
                inhNode = find(strcmp({output{1}.onlineAnalysis, output{2}.onlineAnalysis, output{3}.onlineAnalysis}, 'inh'));
            else
                excNode = find(strcmp({output{1}.onlineAnalysis, output{2}.onlineAnalysis}, 'exc'));
                inhNode = find(strcmp({output{1}.onlineAnalysis, output{2}.onlineAnalysis}, 'inh'));
            end
            if ~isempty(excNode) && ~isempty(inhNode)
                excTrace = -output{excNode}.meanRes(i, :); % Flip the excitatory trace
                inhTrace = output{inhNode}.meanRes(i, :);
            end
        end

        [inh_base, inh_high_freq]=low_pass(paras.tempFreq/2,inhTrace);
        [exc_base, exc_high_freq]=low_pass(paras.tempFreq/2,excTrace);


        % Find peaks
        [excPeaks, excLocs] = findpeaks(excTrace(paras.prePts + 1:end), ...
            'MinPeakDistance', paras.sampleRate / (3 * paras.tempFreq), ...
            'NPeaks', 2 * paras.tempFreq * paras.stimPts / paras.sampleRate, ...
            'MinPeakHeight', std(exc_high_freq)/2);
        [inhPeaks, inhLocs] = findpeaks(inhTrace(paras.prePts + 1:end), ...
            'MinPeakDistance', paras.sampleRate / (3 * paras.tempFreq), ...
            'NPeaks', 2 * paras.tempFreq * paras.stimPts / paras.sampleRate, ...
            'MinPeakHeight', std(inh_high_freq)/2);

        % Adjust locations to account for prePts offset
        excLocs = excLocs + paras.prePts;
        inhLocs = inhLocs + paras.prePts;

        % Initialize arrays to store temporal offsets, amplitude ratios, and center of mass differences
        temporalOffsets = zeros(min(length(excLocs), length(inhLocs)), 1);
        amplitudeRatios = zeros(min(length(excLocs), length(inhLocs)), 1);
        centerOfMassDifferences = zeros(min(length(excLocs), length(inhLocs)), 1);

        % Create a new figure for raw traces with subplots
        subplot(numel(barList), 2, 2 * i - 1, 'parent', f1);

        hold all;
        plot(timeStamps, -excTrace, 'r', 'DisplayName', 'Excitatory');
        plot(timeStamps, inhTrace, 'b', 'DisplayName', 'Inhibitory');
        if numel(selectedNodes) == 2 && any(strcmp({output{1}.onlineAnalysis, output{2}.onlineAnalysis}, 'extracellular'))
            for node = 1:2
                if strcmp(output{node}.onlineAnalysis, 'extracellular')
                    plot(timeStamps, output{node}.meanRes(i, :), 'k', 'DisplayName', 'Spikes');
                end
            end
        elseif numel(selectedNodes) == 3 && any(strcmp({output{1}.onlineAnalysis, output{2}.onlineAnalysis, output{3}.onlineAnalysis}, 'extracellular'))
            for node = 1:3
                if strcmp(output{node}.onlineAnalysis, 'extracellular')
                    plot(timeStamps, output{node}.meanRes(i, :), 'k', 'DisplayName', 'Spikes');
                end
            end
        end
        % Process excitation peaks
        for j = 1:length(excLocs)
            % Define lobe area around each peak (20% threshold with padding)
            padding = paras.sampleRate / (2 * paras.tempFreq);
            excLobeStart = max(excLocs(j) - padding, 1);
            excLobeEnd = min(excLocs(j) + padding, numel(excTrace));

            excMean = mean(exc_base(excLobeStart:excLobeEnd));
            excThrh = excPeaks(j) - 2 * (excPeaks(j) - excMean) * (1 - paras.CoMTh);

            % Confine the lobe within the 20% rise and fall threshold
            tempStart = find(excTrace(excLobeStart:excLocs(j)) <= excThrh, 1, 'last');
            tempEnd = find(excTrace(excLocs(j):excLobeEnd) <= excThrh, 1, 'first');

            if isempty(tempStart)
                [~, tempStart] = min(excTrace(excLobeStart:excLocs(j)));
                tempStart = tempStart + excLobeStart - 1;
            else
                tempStart = tempStart + excLobeStart - 1;
            end
            if isempty(tempEnd)
                [~, tempEnd] = min(excTrace(excLocs(j):excLobeEnd));
                tempEnd = tempEnd + excLocs(j) - 1;
            else
                tempEnd = tempEnd + excLocs(j) - 1;
            end

            excLobeStart = tempStart;
            excLobeEnd = tempEnd;
            % Compute center of mass for the lobe
            excLobe = excTrace(excLobeStart:excLobeEnd);
            excTime = (excLobeStart:excLobeEnd) / paras.sampleRate;
            excCenterOfMass = sum(excTime .* excLobe) / sum(excLobe);

            % Mark peaks and plot vertical lines for centers of mass
            % plot(excLocs(j) / paras.sampleRate, excPeaks(j), 'ro');
            % Get the current y-axis limits
            currentYLim = ylim;

            % Plot the constant line as a regular line
            plot([excCenterOfMass excCenterOfMass], currentYLim, 'color',[0.9 0.1 0.1], 'LineWidth', 1);

        end

        % Process inhibition peaks
        for j = 1:length(inhLocs)
            % Define lobe area around each peak (20% threshold with padding)
            padding = paras.sampleRate / (2 * paras.tempFreq);
            inhLobeStart = max(inhLocs(j) - padding, 1);
            inhLobeEnd = min(inhLocs(j) + padding, numel(inhTrace));

            inhMean = mean(inh_base(inhLobeStart:inhLobeEnd));
            inhThrh = inhPeaks(j) - 2 * (inhPeaks(j) - inhMean) * (1 - paras.CoMTh);

            % Confine the lobe within the 20% rise and fall threshold
            tempStart = find(inhTrace(inhLobeStart:inhLocs(j)) <= inhThrh, 1, 'last');
            tempEnd = find(inhTrace(inhLocs(j):inhLobeEnd) <= inhThrh, 1, 'first');

            if isempty(tempStart)
                [~, tempStart] = min(inhTrace(inhLobeStart:inhLocs(j)));
                tempStart = tempStart + inhLobeStart - 1;
            else
                tempStart = tempStart + inhLobeStart - 1;
            end

            if isempty(tempEnd)
                [~, tempEnd] = min(inhTrace(inhLocs(j):inhLobeEnd));
                tempEnd = tempEnd + inhLocs(j) - 1;
            else
                tempEnd = tempEnd + inhLocs(j) - 1;
            end

            inhLobeStart = tempStart;
            inhLobeEnd = tempEnd;

            % Compute center of mass for the lobe
            inhLobe = inhTrace(inhLobeStart:inhLobeEnd);
            inhTime = (inhLobeStart:inhLobeEnd) / paras.sampleRate;
            inhCenterOfMass = sum(inhTime .* inhLobe) / sum(inhLobe);

            % Mark peaks and plot vertical lines for centers of mass
            % plot(inhLocs(j) / paras.sampleRate, inhPeaks(j), 'bo');
            % Get the current y-axis limits
            currentYLim = ylim;

            % Plot the constant line as a regular line
            plot([inhCenterOfMass inhCenterOfMass], currentYLim, 'color',[0.1 0.1 0.9],'linewidth',1);

        end

        % Calculate temporal offset, amplitude ratio, and center of mass difference if needed
        for j = 1:min(length(excLocs), length(inhLocs))
            temporalOffsets(j) = inhCenterOfMass - excCenterOfMass;
            amplitudeRatios(j) = excPeaks(j) / inhPeaks(j);
            centerOfMassDifferences(j) = excCenterOfMass - inhCenterOfMass;
        end
        % Ensure the plot is displayed
        hold off;
        % legend('show');
        xlabel('Time (s)');
        ylabel('Response');
        title(['Bar Width: ', num2str(barList(i))]);


        % Display mean temporal offset, mean amplitude ratio, and mean center of mass difference

        % meanAmplitudeRatio = mean(amplitudeRatios);
        meanCenterOfMassDifference = mean(centerOfMassDifferences);
        output{1}.meanEIOffset(i)=mean(centerOfMassDifferences)*1e3;
        % disp(['Mean temporal offset for Bar Width ', num2str(barList(i)), ': ', num2str(meanOffset), ' s']);
        % disp(['Mean amplitude ratio for Bar Width ', num2str(barList(i)), ': ', num2str(meanAmplitudeRatio)]);
        disp(['Mean center of mass difference for Bar Width ', num2str(barList(i)), ': ', num2str(meanCenterOfMassDifference*1000), ' ms']);


        % subplot(3, 1, 1);
        % legend('show');
        % subplot(3, 1, 2);
        % legend('show');
        % subplot(3, 1, 3);
        % legend('show');
        % Compute and plot the cross-correlation between exc and inh traces
        [xcorrVals, lags] = xcorr(excTrace, inhTrace, maxLag, 'coeff');
        lags = lags / paras.sampleRate; % Convert lags to seconds

        % Slice out values within the range of -maxLag to maxLag
        validRange = (lags >= -maxLag) & (lags <= maxLag);
        xcorrVals = xcorrVals(validRange);
        lags = lags(validRange)*1e3;

        subplot(numel(barList), 2, 2 * i, 'parent', f1);
        plot(lags, xcorrVals, 'k', 'LineWidth', 2);
        % set(gca,'xtick',[-2*pi, -pi,0, pi, 2*pi],'xticklabel',{'-2pi','-pi','0','pi','2pi'})
        xlabel('Lag (ms)');
        ylabel('cc');
        title('correlation between Exc and Inh traces');
        tpEICorr(i,:)=xcorrVals;
     end
        output{1}.eiOffset=tpEIOffset;
        output{1}.eiRatio=tpEIRatio;
        output{1}.lags=lags;
        output{1}.eiCorr=tpEICorr;

        % overlay the EI F2 tuning 
        f=figure;  hold all;
        if any(cellfun(@(x) x.isBlended, output))
            plot(barList, output{1}.sinoF2Exc, 'Color', 'r', 'LineWidth', 2, 'Marker', 'o', 'markersize', 10, 'DisplayName', 'Overall');
            plot(barList, output{1}.sinoF2Inh, 'Color', 'k', 'LineWidth', 2, 'Marker', 'o', 'markersize', 10, 'DisplayName', 'Overall');
        else
            plot(barList, output{1}.sinoF2, 'Color', 'r', 'LineWidth', 2, 'Marker', 'o', 'markersize', 10, 'DisplayName', 'Overall');
            plot(barList, output{2}.sinoF2, 'Color', 'k', 'LineWidth', 2, 'Marker', 'o', 'markersize', 10, 'DisplayName', 'Overall');

        end
        % ylim([0 1]);
        setAxes(f);
        initFig(gca(f), 'Bar width (um)', 'Norm Sine amp');

        % Calculate and plot the I/E ratio
        f2 = figure;
        hold all;

        if any(cellfun(@(x) x.isBlended, output))
            % Calculate the I/E ratio for blended output
            ieRatio = output{1}.sinoF2Inh ./ output{1}.sinoF2Exc;
        else
            % Calculate the I/E ratio for separate output
            ieRatio = output{2}.sinoF2 ./ output{1}.sinoF2;
        end

        % Plot the I/E ratio
        plot(barList(3:end), ieRatio(3:end), 'Color', 'b', 'LineWidth', 2, 'Marker', 'o', 'markersize', 10, 'DisplayName', 'I/E Ratio');

        % Set plot axes and labels for I/E ratio
        setAxes(f2);
        initFig(gca(f2), 'Bar width (um)', 'I/E Ratio');


end

    function [est_baseline, high_freq_component]=low_pass(cutoff_freq,data)
        fs = 10000; % sampling frequency (in Hz)
        fc = cutoff_freq; % cutoff frequency (in Hz)
        [b, a] = butter(2, fc / (fs / 2), 'low'); % 2nd order Butterworth low-pass filter

        % Apply the low-pass filter to the data to extract the baseline
        est_baseline = filtfilt(b, a, data);

        % Subtract the estimated baseline from the original data to get the high-frequency component
        high_freq_component = data - est_baseline;

    end
end

