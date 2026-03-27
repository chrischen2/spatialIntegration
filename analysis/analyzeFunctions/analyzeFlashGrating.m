function [f, stats] = analyzeFlashGrating(selectedNodes,paras)
for node=1:numel(selectedNodes)
    resMat=riekesuite.getResponseMatrix(selectedNodes{node}.epochList,'Amp1');
    paras.epochRange=1:size(resMat,1);

    % find only exc trials or inh trials, or detect mixed EXC/INH responses
    % Calculate mean for each trial
    trialMeans = mean(resMat,2);
    
    % Check for strong excitatory and inhibitory responses
    hasStrongExc = any(trialMeans < -50);
    hasStrongInh = any(trialMeans > 50);
    
    if ~strcmp(selectedNodes{node}.splitValue,'extracellular') && hasStrongExc && hasStrongInh
        fprintf('%s  \n', 'WARNING: Both strong excitatory (<-50) and inhibitory (>50) responses detected in the node!');
    end
%     
%     if strcmp(selectedNodes{node}.splitValue, 'exc')
        % paras.epochRange=find(mean(resMat,2)<50); 
%     elseif strcmp(selectedNodes{node}.splitValue, 'inh')
        % paras.epochRange=find(mean(resMat,2)>50);
%     end

%     paras.epochRange=25:48;

    resMat=resMat(paras.epochRange,:);
    % outlier detection 
    outliers=find(isoutlier(mean(resMat(:,1:paras.prePts),2)),1);
    if ~isempty(outliers)
        paras.epochRange(outliers)=[];
        resMat(outliers,:)=[];
        fprintf('%s %d \n', 'outlier trial removed', outliers);
    end
    barWidths=zeros(size(resMat,1),1);
    for i=1:size(resMat,1)
        if strcmp(selectedNodes{node}.splitValue, 'extracellular') || paras.spikeTag
            resMat(i,:)=resMat(i,:)-movmedian(resMat(i,:),100);
        else
            resMat(i,:)=smooth(resMat(i,:),200);
        end
        barWidths(i)=selectedNodes{node}.epochList.elements(paras.epochRange(i)).protocolSettings('currentBarWidth');
    end
    if strcmp(selectedNodes{node}.splitValue, 'extracellular') || paras.spikeTag
        [spikeTimes,~,~,~]=SpikeDetectorNew(resMat, 'thresholdSpikeFactor', paras.spikeTh);
        psth=spikeTimeToPSTH(resMat, spikeTimes, paras.psthSigma, paras.sampleRate);
        onlineAnalysis='extracellular';
    else
        if mean2(resMat)<0
            onlineAnalysis='exc';  % some recordings did not set the right onlineAnalysis
            % For excitatory responses, we need to flip sign for conductance representation
            isExc = true;
        else
            onlineAnalysis='inh';
            isExc = false;
        end
        resMat=resMat-repmat(mean(resMat(:,1:paras.prePts),2),1,size(resMat,2));
    end
    % analyze the onset and offset response
    response.onset=zeros(size(resMat,1),1);
    response.offset=zeros(size(resMat,1),1);
    peak.onset=zeros(size(resMat,1),1);
    peak.offset=zeros(size(resMat,1),1);
    response.offset_baseline=zeros(size(resMat,1),1);
    
    % Define the offset baseline period (from 3/4 of stimulus to end of stimulus)
    offset_baseline_start = round(paras.prePts + paras.stimPts*3/4);
    offset_baseline_end = paras.prePts + paras.stimPts;
    
    for i=1:size(resMat,1)
        if strcmp(selectedNodes{node}.splitValue, 'extracellular') || paras.spikeTag
            % For extracellular/spike recording
            response.baseline(i)=length(spikeTimes{i}(spikeTimes{i}<paras.prePts));
            response.onset(i)=length(spikeTimes{i}(spikeTimes{i}>paras.prePts+paras.spikeoffset & spikeTimes{i}<paras.prePts+paras.stimPts+paras.spikeoffset))-response.baseline(i)*paras.stimPts/paras.prePts;
            
            % Calculate offset baseline (# of spikes during the specified period)
            offset_baseline_count = length(spikeTimes{i}(spikeTimes{i}>offset_baseline_start+paras.spikeoffset & spikeTimes{i}<offset_baseline_end+paras.spikeoffset));
            response.offset_baseline(i) = offset_baseline_count / (offset_baseline_end - offset_baseline_start) * (round(paras.tailPts/2)-paras.spikeoffset);
            
            % Recalculate offset response using new baseline and half tail period
            response.offset(i)=length(spikeTimes{i}(spikeTimes{i}>paras.prePts+paras.stimPts+paras.spikeoffset & spikeTimes{i}<paras.prePts+paras.stimPts+round(paras.tailPts/2)+paras.spikeoffset))-response.offset_baseline(i);
            
            % Clip calculation for onset (unchanged)
            onClip=psth(i,paras.prePts+paras.spikeoffset:paras.prePts+paras.stimPts+paras.spikeoffset)-mean(psth(i,1:paras.prePts));
            
            % New clip calculation for offset using the new baseline period
            offset_baseline_psth = mean(psth(i,offset_baseline_start+paras.spikeoffset:offset_baseline_end+paras.spikeoffset));
            % Use only first half of tail period for offset analysis
            offClip=psth(i,paras.prePts+paras.stimPts+paras.spikeoffset:paras.prePts+paras.stimPts+round(paras.tailPts/2)+paras.spikeoffset)-offset_baseline_psth;
            
            % Analyze the amplitudes
            % For onset: determine peak polarity based on area sum sign
            if mean(onClip) > 0 % If area sum is positive
                [peak.onset(i), ind] = max(onClip); % Use positive peak
            else
                [peak.onset(i), ind] = min(onClip); % Use negative peak
            end
            
            % For offset: determine peak polarity based on area sum sign
            if mean(offClip) > 0 % If area sum is positive
                [peak.offset(i), ind] = max(offClip); % Use positive peak
            else
                [peak.offset(i), ind] = min(offClip); % Use negative peak
            end
        else
            % For non-spike recording
            onClip=resMat(i,paras.prePts+paras.wcoffset:paras.prePts+paras.stimPts+paras.wcoffset);
            
            % Calculate the new baseline for offset
            offset_baseline_val = mean(resMat(i,offset_baseline_start+paras.wcoffset:offset_baseline_end+paras.wcoffset));
            response.offset_baseline(i) = offset_baseline_val;
            
            % Calculate offClip with the new baseline, using only first half of tail period
            offClip=resMat(i,paras.prePts+paras.stimPts+paras.wcoffset:paras.prePts+paras.stimPts+round(paras.tailPts/2)+paras.wcoffset) - offset_baseline_val;
            
            % Calculate responses
            response.onset(i)=mean(onClip)*paras.stimPts/1e4; % pA*s
            response.offset(i)=mean(offClip)*(paras.tailPts-paras.wcoffset)/1e4;
            response.baseline(i)=0;
            
            % Analyze the peak amplitudes
            [~, ind]=max(abs(onClip)); peak.onset(i)=onClip(ind);
            [~, ind]=max(abs(offClip)); peak.offset(i)=offClip(ind);
            
            % For excitatory responses, flip the sign to represent in conductance units
            if isExc
                response.onset(i) = -response.onset(i);
                response.offset(i) = -response.offset(i);
                peak.onset(i) = -peak.onset(i);
                peak.offset(i) = -peak.offset(i);
            end
        end
    end
    
    drugAdded{node}=selectedNodes{node}.epochList.firstValue.protocolSettings('epochGroup:externalSolutionAdditions');
    fprintf('%s , %s \n', 'drugs added are ::', drugAdded{node});
    condStr{node}=[selectedNodes{node}.parent.parent.parent.splitValue ' ' selectedNodes{node}.splitValue];
    
    barList=unique(barWidths);
    colors=pmkmp(numel(barList),'IsoL');
    f(node)=figure('position',[150 350 600 600]);
    meanRes{node}=zeros(numel(barList),size(resMat,2)); hold all;
    for i=1:numel(barList)
        % Get indices of trials with current bar width
        barIndices = find(barList(i)==barWidths);
        numTrials = length(barIndices);
        
        % Display number of trials for this bar width
        fprintf('Bar width %d: %d trial(s)\n', barList(i), numTrials);
        
        if numTrials == 0
            warning('No trials found for bar width %d', barList(i));
            % Fill with NaNs if no trials found
            meanRes{node}(i,:) = NaN(1, size(resMat,2));
            stats.onset(i) = NaN;
            stats.offset(i) = NaN;
            stats.baseline(i) = NaN;
            stats.offset_baseline(i) = NaN;
            stats.peakOnset(i) = NaN;
            stats.peakOffset(i) = NaN;
            continue;
        end
        
        % Calculate mean response (works for single trials too)
        if strcmp(selectedNodes{node}.splitValue, 'extracellular') || paras.spikeTag
            if numTrials == 1
                meanRes{node}(i,:) = psth(barIndices,:);
            else
                % Use explicit dimension for averaging across trials
                meanRes{node}(i,:) = sum(psth(barIndices,:), 1) / numTrials;
            end
        else
            if numTrials == 1
                meanRes{node}(i,:) = resMat(barIndices,:);
            else
                % Use explicit dimension for averaging across trials
                meanRes{node}(i,:) = sum(resMat(barIndices,:), 1) / numTrials;
            end
        end
        
        % Plot mean response
        plot(meanRes{node}(i,:),'color',colors(i,:),'linewidth',3);
        
        % Calculate statistics (works for single trials too)
        if numTrials == 1
            stats.onset(i) = response.onset(barIndices);
            stats.offset(i) = response.offset(barIndices);
            stats.baseline(i) = response.baseline(barIndices);
            stats.offset_baseline(i) = response.offset_baseline(barIndices);
            stats.peakOnset(i) = peak.onset(barIndices);
            stats.peakOffset(i) = peak.offset(barIndices);
        else
            stats.onset(i) = sum(response.onset(barIndices)) / numTrials;
            stats.offset(i) = sum(response.offset(barIndices)) / numTrials;
            stats.baseline(i) = sum(response.baseline(barIndices)) / numTrials;
            stats.offset_baseline(i) = sum(response.offset_baseline(barIndices)) / numTrials;
            stats.peakOnset(i) = sum(peak.onset(barIndices)) / numTrials;
            stats.peakOffset(i) = sum(peak.offset(barIndices)) / numTrials;
        end
    end
    legend(cellstr(num2str(barList, 'bar size %-d')),'fontsize',15); legend boxoff; 
    title(condStr{node});
    stats.barList=barList';  
    
    % Create a figure showing both onset and offset responses
    figure('position',[750 350 1000 800],'color','w');
    subplot(2,2,1);
    plot(stats.barList, stats.peakOnset,'linewidth',3); 
    title('Onset Amplitude'); box off;
    xlabel('Bar width'); 
    if strcmp(onlineAnalysis, 'exc')
        ylabel('Response (nS)');
    else
        ylabel('Response (pA)');
    end
    
    subplot(2,2,2);
    plot(stats.barList, stats.onset,'linewidth',3); 
    title('Onset Charge Transfer'); box off;
    xlabel('Bar width'); 
    if strcmp(onlineAnalysis, 'exc')
        ylabel('Response (nS·s)');
    else
        ylabel('Response (pC)');
    end
    
    subplot(2,2,3);
    plot(stats.barList, stats.peakOffset,'linewidth',3); 
    title('Offset Amplitude'); box off;
    xlabel('Bar width'); 
    if strcmp(onlineAnalysis, 'exc')
        ylabel('Response (nS)');
    else
        ylabel('Response (pA)');
    end
    
    subplot(2,2,4);
    plot(stats.barList, stats.offset,'linewidth',3); 
    title('Offset Charge Transfer'); box off;
    xlabel('Bar width'); 
    if strcmp(onlineAnalysis, 'exc')
        ylabel('Response (nS·s)');
    else
        ylabel('Response (pC)');
    end
    
    sgtitle(['Response analysis: ' condStr{node}]);
end

paras.meanLum=selectedNodes{1}.epochList.firstValue.protocolSettings ...
    ('epoch:Microdisplay_Stage@localhost:white:rodConversionFactor')*selectedNodes{node}.epochList.firstValue.protocolSettings('backgroundIntensity');
fprintf('%s , %f \n', 'mean Luminance::', paras.meanLum');

if numel(selectedNodes)>1
    figure('position',[100 50 700 900],'color','w');
    exampleBar=1:numel(barList);
    for bar=1:numel(exampleBar)
        subplot(numel(exampleBar),1,bar); hold all;
        for node=1:numel(selectedNodes)          
            plot(meanRes{node}(exampleBar(bar),:),'color',colors(node,:),'linewidth',2);
        end  
        legend(condStr);  legend boxoff; title(['bar size', ' ', num2str(barList(exampleBar(bar)))]);
    end
end
stats.cellType=shortCellType(selectedNodes{1}.epochList.firstValue.protocolSettings('source:type'));
stats.expDate=datestr(selectedNodes{1}.epochList.elements(1).startDate','yyyy/mm/dd'); 
stats.cellLabel=selectedNodes{1}.epochList.firstValue.protocolSettings('source:label');
stats.onlineAnalysis=onlineAnalysis;
end