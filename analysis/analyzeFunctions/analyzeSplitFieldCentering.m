function [ax1,output] = analyzeSplitFieldCentering(selectedNodes, paras)
figure('position',[350 400 1000 600],'color','w'); ax1=axes;  hold all; 
figure('position',[350 400 1000 600],'color','w'); ax2=axes;  hold all;
colors=pmkmp(numel(selectedNodes,'IsoL'));  
for node=1:numel(selectedNodes)
    resMat=riekesuite.getResponseMatrix(selectedNodes{node}.epochList,'Amp1');
    epochRange=1:size(resMat,1);
    %     epochRange= find(mean(resMat,2)<50);
%     epochRange=1:2;
    resMat=resMat(epochRange,:);
    
    if strcmp(selectedNodes{node}.splitValue, 'extracellular') || paras.spikeTag
        for i=1:size(resMat,1)
            resMat(i,:)=resMat(i,:)-movmedian(resMat(i,:),100);
        end
        recType='extracellular';
    elseif strcmp(selectedNodes{node}.epochList.firstValue.protocolSettings('epochGroup:pipetteSolution'),'potassium')
         recType='currentClamp';   
         for i=1:size(resMat,1)
             resMat(i,:)=smooth(resMat(i,:),100);
         end
    else 
        for i=1:size(resMat,1)
            resMat(i,:)=smooth(resMat(i,:),100);
        end
        if mean(resMat(:,1:paras.prePts))<0
            recType='exc';
        else
            recType='inh';
        end
    end
    
    
    if strcmp(selectedNodes{node}.splitValue, 'extracellular') || paras.spikeTag
        resMat=smoothPSTH(resMat,paras.psthSigma, paras.sampleRate,paras.spikeTh);
    else
        resMat=resMat-repmat(mean(resMat(:,1:paras.prePts),2),1,size(resMat,2));
    end
    
    noCycles=paras.tempFreq*paras.stimPts/(1e4);
    cycleLen=paras.stimPts/noCycles;
    meanRes{node}=mean(resMat,1);
    cycleClips=reshape(meanRes{node}(paras.prePts+1+paras.wcOffset:paras.prePts+paras.stimPts+paras.wcOffset),cycleLen,noCycles)';
    cycleMean{node}=mean(cycleClips);
%     cycleErr=std(cycleClips)/sqrt(size(cycleClips,1));
%     plot(ax1,(1:numel(meanRes{node}))/1e4, resMat', 'color', 'k','linewidth',0.5);
%     l(node)=plot(ax1,(1:numel(meanRes{node}))/1e4, meanRes{node}, 'color', colors(node,:),'linewidth',3);
    plot(ax1,(1:numel(meanRes{node}))/1e4, meanRes{node});
    plot(ax2,(1:cycleLen)/1e4, cycleMean{node}, 'color', colors(node,:),'linewidth',3);
    lgStr{node}=[selectedNodes{node}.parent.parent.parent.parent.splitValue ' ' selectedNodes{node}.splitValue] ;
    drugAdded{node}=selectedNodes{node}.epochList.firstValue.protocolSettings('epochGroup:externalSolutionAdditions');
    fprintf('%s , %s \n', 'drugs added are ::', drugAdded{node}); 
    % analyze stats of the average trace 
    if strcmp(recType,'exc') 
        modulation.positive(node)= -min(cycleMean{node});  modulation.negative(node)= max(cycleMean{node});
    elseif strcmp(recType,'inh') || strcmp(recType,'currentClamp')
        modulation.positive(node)= max(cycleMean{node});  modulation.negative(node)= min(cycleMean{node});
    end
    rectIndex(node)=(modulation.positive(node)- modulation.negative(node))/(modulation.positive(node)+modulation.negative(node));
    if strcmp(selectedNodes{node}.parent.parent.parent.splitValue, 'full-field')
        [ s, ~,~ ] = fitSinusoidFixedFreq((1:cycleLen)/1e4,cycleMean{node},paras.tempFreq); % double freq, F2
    else
        [ s, ~,~ ] = fitSinusoidFixedFreq((1:cycleLen)/1e4,cycleMean{node},2*paras.tempFreq); % double freq, F2
    end
    sinoF2(node)=abs(s(1));  % amplitude of sine curve
end
% legend(ax1,l,lgStr); legend boxoff;
legend(ax2,lgStr); legend boxoff;
meanIntensity=selectedNodes{node}.epochList.firstValue.protocolSettings ...
    ('epoch:Microdisplay_Stage@localhost:white:rodConversionFactor')*paras.meanIntensity;
fprintf('%s , %f \n', 'mean Luminance::',meanIntensity');
output.rectIndex=rectIndex; output.modulation=modulation; output.recType=recType;
output.sinoF2=sinoF2;
end
