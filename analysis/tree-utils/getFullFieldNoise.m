function StimMatrix = getFullFieldNoise(node, params)

%GETTEMPLFILTER Summary of this function goes here
% input: type: 'exc' current uses cumGaussian function; 'inh' current uses
%   cubic function
%   Detailed explanation goes here
%  1. result.filter -> temporal linear filter after fitting with whitenoise
%  2. result.fitNL -> nonlinear scalar function to account for the static
%  NL
%  3. result.sFactor -> a scale factor used to obtain stable fit to NL
%  scalar function.
   tempRespStream = node.epochList.responseStreamNames;
   respStream = char(tempRespStream(2));
   SampleEpoch = node.epochList.firstValue;
   sampleRate=double(SampleEpoch.protocolSettings.get('sampleRate')); %10K Hz
   SampleInterval = 1./sampleRate; % in ms
   prepts = SampleEpoch.protocolSettings.get('preTime')*sampleRate*1e-3;
   stimpts = SampleEpoch.protocolSettings.get('stimTime')*sampleRate*1e-3;
   tailpts = SampleEpoch.protocolSettings.get('tailTime')*sampleRate*1e-3;
 
   noiseStdv = SampleEpoch.protocolSettings.get('noiseStdv');
   background = SampleEpoch.protocolSettings.get('backgroundIntensity');
   frameDwell = SampleEpoch.protocolSettings.get('frameDwell');
   % assumed parameter values
   freqcutoff = 60;% stimulus
   frameRate = params.FrameRate / frameDwell;
   
   % assumed value ends
   preFrames = round(frameRate * (SampleEpoch.protocolSettings.get('preTime')/1e3));
   stimFrames = round(frameRate * (SampleEpoch.protocolSettings.get('stimTime')/1e3));
   tailFrames = round(frameRate * (SampleEpoch.protocolSettings.get('tailTime')/1e3));
   frameChunks = floor((1000/frameRate)*sampleRate*1e-3);
   StimMatrix = [];

    for i = 1:node.epochList.length
        noise = ones(1,stimpts+prepts+tailpts) * background;
        % get response Matrix
        currentNoiseSeed = node.epochList.elements(i).protocolSettings.get('noiseSeed');
        %display(currentNoiseSeed);
        noiseStream = RandStream('mt19937ar', 'Seed', currentNoiseSeed);
        for j = 1:stimFrames
            noise(1+frameChunks*(j-1+preFrames):frameChunks*(j+preFrames)) = noiseStdv*noiseStream.randn*background + background;
        end
        if (frameChunks*stimFrames < stimpts)
            noise(frameChunks*(stimFrames+preFrames):stimpts) = noiseStdv*noiseStream.randn*background + background; 
        end
        StimMatrix(i,:) = RotateVector(params.RotationPts, noise);
   end
   
end