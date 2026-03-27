function [stats] = analyzeLongSingleEpochNoise(epochNode,paras)
baseRange=paras.baseRange(1)*paras.sampleRate:paras.baseRange(end)*paras.sampleRate;
resMat=riekesuite.getResponseMatrix(epochNode.epochList,'Amp1');
if paras.spikeTag
    resMat=resMat-movmedian(resMat,100);
    [resMat,~,~]=smoothPSTH(resMat, paras.psthSigma,paras.sampleRate,paras.spikeTh);
    stats.recType='extracellular';
else
    resMat=smoothMatrix(resMat,50); % gentl smooth
    if mean(resMat)>0
        stats.recType='inh';
    else
        stats.recType='exc';
    end
    resMat=resMat-mean(resMat(:,baseRange),2);  % baseline adjust
    resMat=highPassFilter(resMat,0.5,1/paras.sampleRate);
end
% setup the LN model parameters
SETTINGS.filterLen=800;   % ms length of linear filter
if strcmp(stats.recType,'extracellular')
    SETTINGS.frequencyCutoff  =50;
else
    SETTINGS.frequencyCutoff  =30;
end

% length of ONE SIDE of filter (causal or anti-causal side)
SETTINGS.correctStimPower = true;
SETTINGS.useAnticausal    = false;
SETTINGS.evalIteration=3;
% Nonlinearity settings:
SETTINGS.numBins   = 100;
SETTINGS.binningType = 'equalN';  % 'equalLN'
SETTINGS.fittingMode='sigmoid';  % 'sigmoid' or 'logistic'
SETTINGS.baseShift=0;
stats.fitMode=SETTINGS.fittingMode;

noiseSeeds=epochNode.epochList.firstValue.protocolSettings('seed');
tpStim= createGaussianNoiseStimulus(paras,paras.lMean,paras.stdv,noiseSeeds);
%now downsample stimulus and response
stimulus=arrayfun(@(x) mean(tpStim(x:x+paras.downsample-1)),1:paras.downsample:length(tpStim)-paras.downsample+1);
response=arrayfun(@(x) mean(resMat(x:x+paras.downsample-1)),1:paras.downsample:length(resMat)-paras.downsample+1);
SETTINGS.samplingInterval=paras.downsample/paras.sampleRate;
SETTINGS.filterPts = (SETTINGS.filterLen/1e3)/SETTINGS.samplingInterval;
paras.sampleRate=paras.sampleRate/paras.downsample;
if paras.plotTrace
    figure('color','w','position',[100 100 1200 600]);
    plot(response,'k','linewidth',1);
end
% reshape resonse, and stimulus into matrix 
nClips=(paras.stimTime/1e3)/10; % into clips of 10s
stimulus=(reshape(stimulus, numel(stimulus)/nClips,nClips))';
response=(reshape(response, numel(response)/nClips,nClips))';
stats=computeSingleEpochLNmodel(paras,stimulus,response,stats,SETTINGS);
end



