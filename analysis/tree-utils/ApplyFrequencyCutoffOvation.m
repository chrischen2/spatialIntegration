function FilteredEpochData = ApplyFrequencyCutoffOvation(EpochData, FreqCutoff, SamplingInterval)
% ApplyFrequencyCutoffOvation.m
%
%  FilteredEpochData = ApplyFrequencyCutoffOvation(EpochData, FreqCutoff,
%  SamplingInterval)
%
% This function takes as input a matrix of data, cutoff frequency and sampling interval of
% the original data.  It eliminates all temporal frequencies beyond the cutoff and 
% returns the resulting matrix.
%  Created: FMR  11/08

EpochPts = size(EpochData, 2);
NumEpochs = size(EpochData, 1);

FreqStepSize = 1/(SamplingInterval * EpochPts);
FreqCutoffPts = round(FreqCutoff / FreqStepSize);

% eliminate frequencies beyond cutoff (middle of matrix given fft
% representation)
FFTData = fft(EpochData, [], 2);
FFTData(:,FreqCutoffPts:length(FFTData(1,:))-FreqCutoffPts) = 0;
FilteredEpochData = real(ifft(FFTData, [], 2));


	
