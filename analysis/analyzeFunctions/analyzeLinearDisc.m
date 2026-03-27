function [output] = analyzeLinearDisc(selectedNodes, paras)
    % Initialize output structure once
    output = struct();
    output.backgroundIntensity = selectedNodes{1}.epochList.firstValue.protocolSettings('backgroundIntensity');
    paras.maxIntensity = selectedNodes{1}.epochList.firstValue.protocolSettings ...
        ('epoch:Microdisplay_Stage@localhost:white:rodConversionFactor');
    fprintf('%s , %f \n', 'mean Luminance::', paras.maxIntensity * output.backgroundIntensity);
    noPatches = selectedNodes{1}.epochList.firstValue.protocolSettings('noPatches');
    
    % Setup cell type and experimental info early
    output.cellType = shortCellType(selectedNodes{1}.epochList.firstValue.protocolSettings('source:type'));
    output.expDate = datestr(selectedNodes{1}.epochList.elements(1).startDate', 'yyyy/mm/dd');
    output.cellLabel = selectedNodes{1}.epochList.firstValue.protocolSettings('source:label');
    output.imageName = selectedNodes{1}.epochList.firstValue.protocolSettings('imageName');
    
    % Process each node (exc, inh, or extracellular)
    for node = 1:numel(selectedNodes)
        nodeName = selectedNodes{node}.splitValue;
        resMat = riekesuite.getResponseMatrix(selectedNodes{node}.epochList, 'Amp1');
        paras.epochRange = 1:size(resMat, 1);
        paras.epochRange(paras.rmreps) = [];
        
        % Choose epochs based on response characteristics
        % paras.epochRange = find(mean(resMat, 2) < 100);
        
        resMat = resMat(paras.epochRange, :);
        intensities = zeros(size(resMat, 1), 1);
        
        % Preprocess based on recording type
        for i = 1:size(resMat, 1)
            if strcmp(nodeName, 'extracellular') || paras.spikeTag
                resMat(i, :) = resMat(i, :) - movmedian(resMat(i, :), 100);
                output.onlineAnalysis = 'extracellular'; 
                titleStr = 'spike (#)';
                resSign = 1;
            else
                if mean2(resMat) < 0
                    output.onlineAnalysis = 'exc'; 
                    titleStr = 'Excitation (pA*s)'; 
                    resSign = -1;
                else
                    output.onlineAnalysis = 'inh'; 
                    titleStr = 'Inhibition (pA*s)'; 
                    resSign = 1;
                end
                resMat(i, :) = smooth(resMat(i, :), 100);
            end
            intensities(i) = selectedNodes{node}.epochList.elements(paras.epochRange(i)).protocolSettings('equivalentIntensity') - output.backgroundIntensity;
        end
        
        % Process spike or current recordings
        if strcmp(nodeName, 'extracellular') || paras.spikeTag
            [spikeTimes, ~, ~, ~] = SpikeDetectorNew(resMat, 'thresholdSpikeFactor', paras.spikeTh, 'CheckDetection', false);
            psth = spikeTimeToPSTH(resMat, spikeTimes, paras.psthSigma, paras.sampleRate);
        else
            resMat = resMat - repmat(mean(resMat(:, 1:paras.prePts), 2), 1, size(resMat, 2));
        end
        
        % Extract stimulus information and calculate responses
        imageIndex = zeros(size(resMat, 1), 1);
        stimTag = zeros(size(resMat, 1), 1);
        response = struct('onset', zeros(size(resMat, 1), 1), ...
                          'offset', zeros(size(resMat, 1), 1), ...
                          'baseline', zeros(size(resMat, 1), 1));
        
        patchCount = 0; 
        traj = zeros(noPatches, 2); 
        intArray = zeros(noPatches, 1);
        
        for i = 1:size(resMat, 1)
            imageIndex(i) = selectedNodes{node}.epochList.elements(paras.epochRange(i)).protocolSettings('imagePatchIndex');
            tpTag = selectedNodes{node}.epochList.elements(paras.epochRange(i)).protocolSettings('stimulusTag');
            
            % Process by stimulus type
            switch tpTag
                case 'image'
                    stimTag(i) = 1;
                case 'intensity'
                    stimTag(i) = 2;
                    % Record intensity and location data for equivalent disc
                    if patchCount < noPatches
                        patchCount = patchCount + 1;
                        traj(patchCount, :) = convertJavaArrayList(selectedNodes{node}.epochList.elements(paras.epochRange(i)).protocolSettings('currentPatchLocation'));
                        intArray(patchCount) = selectedNodes{node}.epochList.elements(paras.epochRange(i)).protocolSettings('equivalentIntensity');
                    end
            end
            
            % Calculate response metrics based on recording type
            if strcmp(nodeName, 'extracellular') || paras.spikeTag
                % For spike recordings
                response.onset(i) = length(spikeTimes{i}(spikeTimes{i} > paras.prePts + paras.spikeoffset & ...
                                                          spikeTimes{i} < paras.prePts + paras.stimPts + paras.spikeoffset));
                
                offsetSpikes = spikeTimes{i}(spikeTimes{i} > paras.prePts + paras.stimPts + paras.spikeoffset);
                response.offset(i) = length(offsetSpikes);
                response.baseline(i) = length(spikeTimes{i}(spikeTimes{i} < paras.prePts));
                response.offsetBaseline(i) = 0;
            else
                % For exc/inh whole-cell recordings
                response.onset(i) = mean(resMat(i, paras.prePts + paras.wcoffset:paras.prePts + paras.stimPts + paras.wcoffset)) * paras.stimPts / 1e4;
                response.baseline(i) = mean(resMat(i, 1:paras.prePts)) * paras.stimPts / 1e4;
                
                offsetWindow = paras.prePts + paras.stimPts + paras.wcoffset:size(resMat,2);
                offsetBaselineWindow = paras.prePts + paras.stimPts + paras.wcoffset:(paras.prePts + paras.stimPts + paras.wcoffset + 1000);
                offsetBaselineWindow = min(offsetBaselineWindow, size(resMat, 2));
                response.offsetBaseline(i) = min(abs(resMat(i, offsetBaselineWindow))) * sign(response.baseline(i)) * paras.stimPts / 1e4;
                
                response.offset(i) = mean(resMat(i, offsetWindow)) * paras.stimPts / 1e4 - response.offsetBaseline(i);
            end
        end
        
        % Analyze responses by unique patch
        uniquePatches = unique(imageIndex);
        repCount = 0; 
        rmInd = [];
        
        for i = 1:numel(uniquePatches)
            if numel(find(imageIndex == uniquePatches(i))) >= 2
                repCount = repCount + 1;
                
                % Store statistics for this patch
                output.stats.image.onset.mean(repCount) = resSign * mean(response.onset((imageIndex == uniquePatches(i) & stimTag == 1)));
                output.stats.disc.onset.mean(repCount) = resSign * mean(response.onset((imageIndex == uniquePatches(i) & stimTag == 2)));
                output.stats.image.onset.ste(repCount) = ste(response.onset((imageIndex == uniquePatches(i) & stimTag == 1)));
                output.stats.disc.onset.ste(repCount) =  ste(response.onset((imageIndex == uniquePatches(i) & stimTag == 2)));

                output.stats.image.offset.mean(repCount) = resSign * mean(response.offset((imageIndex == uniquePatches(i) & stimTag == 1)));
                output.stats.disc.offset.mean(repCount) = resSign * mean(response.offset((imageIndex == uniquePatches(i) & stimTag == 2)));
                output.stats.image.offset.ste(repCount) =  ste(response.offset((imageIndex == uniquePatches(i) & stimTag == 1)));
                output.stats.disc.offset.ste(repCount) = ste(response.offset((imageIndex == uniquePatches(i) & stimTag == 2)));

                output.stats.image.baseline.mean(repCount) = resSign * mean(response.baseline((imageIndex == uniquePatches(i) & stimTag == 1)));
                output.stats.disc.baseline.mean(repCount) = resSign * mean(response.baseline((imageIndex == uniquePatches(i) & stimTag == 2)));
                output.stats.image.baseline.ste(repCount) =  ste(response.baseline((imageIndex == uniquePatches(i) & stimTag == 1)));
                output.stats.disc.baseline.ste(repCount) =  ste(response.baseline((imageIndex == uniquePatches(i) & stimTag == 2)));

                output.stats.image.offsetBaseline.mean(repCount) = resSign * mean(response.offsetBaseline((imageIndex == uniquePatches(i) & stimTag == 1)));
                output.stats.disc.offsetBaseline.mean(repCount) = resSign * mean(response.offsetBaseline((imageIndex == uniquePatches(i) & stimTag == 2)));
                output.stats.image.offsetBaseline.ste(repCount) =  ste(response.offsetBaseline((imageIndex == uniquePatches(i) & stimTag == 1)));
                output.stats.disc.offsetBaseline.ste(repCount) =  ste(response.offsetBaseline((imageIndex == uniquePatches(i) & stimTag == 2)));
                
                try 
                    output.deltaEqvInt(repCount) = unique(intensities(imageIndex == uniquePatches(i) & stimTag == 1));
                catch 
                    fprintf('Error processing patch %d\n', repCount);
                    disp(unique(intensities(imageIndex == uniquePatches(i) & stimTag == 1)));
                end
            else
                rmInd = [rmInd i];
            end
        end
        
        uniquePatches(rmInd) = [];
        output.patchRec = uniquePatches;
        
        % Calculate point-to-line distances for example selection
        distances = pointToLineDistance([output.stats.image.onset.mean' output.stats.disc.onset.mean'], [0 0], [1 1]);
        exampleInd = 1:numel(distances);
        
        % Calculate percentiles for sample selection
        dist_25th = prctile(distances, 30);
        dist_75th = prctile(distances, 70);
        
        furthestInd = exampleInd(distances(exampleInd) >= dist_75th);
        closestInd = exampleInd(distances <= dist_25th);

        % Sample patches based on paras.nSamples
        numSamples = min(paras.nSamples, min(numel(furthestInd), numel(closestInd)));
        if numSamples > 0
            sampledFurthestInd = furthestInd(randperm(length(furthestInd), numSamples));
            sampledClosestInd = closestInd(randperm(length(closestInd), numSamples));
        else
            sampledFurthestInd = [];
            sampledClosestInd = [];
        end

        % Get protocol parameters for patch extraction
        RFsigma = selectedNodes{node}.epochList.firstValue.protocolSettings('rfSigmaCenter');
        apertureDiameter = selectedNodes{node}.epochList.firstValue.protocolSettings('apertureDiameter');
        
        % Load image analysis once for all patches
        fprintf('Loading image analysis for image: %s\n', output.imageName);
        analyzeNaturalImagePatches('imageName', output.imageName, ...
                                  'apertureDiameter', apertureDiameter, ...
                                  'rfSigmaCenter', RFsigma, ...
                                  'noPatches', 1);
        
        % Get the output from workspace
        imgAnalysis = evalin('base', 'imageAnalysisOutput');
        
        
        % Create figures for visualization
        for k = 1:numSamples
            maxInd = sampledFurthestInd(k);
            minInd = sampledClosestInd(k);
            
            % Get patch indices for extracting patch images
            patchInd.max = find(uniquePatches(maxInd) == imageIndex, 1, 'first');
            patchInd.min = find(uniquePatches(minInd) == imageIndex, 1, 'first');
            
            % Extract patch information
            patchLoc.max = convertJavaArrayList(selectedNodes{node}.epochList.elements(paras.epochRange(patchInd.max)).protocolSettings('currentPatchLocation'));
            patchLoc.min = convertJavaArrayList(selectedNodes{node}.epochList.elements(paras.epochRange(patchInd.min)).protocolSettings('currentPatchLocation'));
            
            % Extract patches for both max and min
            [patchImg.max, patchStats.max] = extractSinglePatch(imgAnalysis, patchLoc.max);
            [patchImg.min, patchStats.min] = extractSinglePatch(imgAnalysis, patchLoc.min);
            
            % Create comprehensive figure
            f1 = figure('position', [50 10 1600 900], 'color', 'w', 'Name', ...
                sprintf('Patch_%d_vs_%d', uniquePatches(maxInd), uniquePatches(minInd)));
            
            % Row 1: Scatter plots
            ax(1) = subplot(3, 4, 1); hold all;
            scatterWithError(output.stats.image.onset.mean, output.stats.disc.onset.mean, ...
                           output.stats.image.onset.ste, output.stats.disc.onset.ste, 1);
            scatter(output.stats.image.onset.mean(maxInd), output.stats.disc.onset.mean(maxInd), 100, 'r', 'filled');
            scatter(output.stats.image.onset.mean(minInd), output.stats.disc.onset.mean(minInd), 100, 'g', 'filled');
            initFig(gca(f1), 'response to image', 'response to disc'); setAxes(f1); 
            title(sprintf('Onset | Cell: %s', output.cellLabel));
            
            ax(2) = subplot(3, 4, 2); hold all;
            scatterWithError(output.stats.image.offset.mean, output.stats.disc.offset.mean, ...
                           output.stats.image.offset.ste, output.stats.disc.offset.ste, 1);
            scatter(output.stats.image.offset.mean(maxInd), output.stats.disc.offset.mean(maxInd), 100, 'r', 'filled');
            scatter(output.stats.image.offset.mean(minInd), output.stats.disc.offset.mean(minInd), 100, 'g', 'filled');
            initFig(gca(f1), 'response to image', 'response to disc'); setAxes(f1); 
            title('Offset');
            
            % Row 2: Example traces for furthest patch (red)
            ax(3) = subplot(3, 4, 5); hold all;
            if strcmp(nodeName, 'extracellular') || paras.spikeTag
                meanTrace.image.max = mean(psth(imageIndex == uniquePatches(maxInd) & stimTag == 1, :), 1);
                meanTrace.disc.max = mean(psth(imageIndex == uniquePatches(maxInd) & stimTag == 2, :), 1);
            else
                meanTrace.image.max = mean(resMat(imageIndex == uniquePatches(maxInd) & stimTag == 1, :), 1);
                meanTrace.disc.max = mean(resMat(imageIndex == uniquePatches(maxInd) & stimTag == 2, :), 1);
            end
            plot(meanTrace.image.max, 'k', 'linewidth', 2);
            plot(meanTrace.disc.max, 'r', 'linewidth', 2);
            legend({'image', 'disc'}); legend boxoff;
            title(sprintf('Patch %d (High Nonlinearity)', uniquePatches(maxInd)));
            ylabel(titleStr);
            
            % Row 3: Example traces for closest patch (green)
            ax(4) = subplot(3, 4, 9); hold all;
            if strcmp(nodeName, 'extracellular') || paras.spikeTag
                meanTrace.image.min = mean(psth(imageIndex == uniquePatches(minInd) & stimTag == 1, :), 1);
                meanTrace.disc.min = mean(psth(imageIndex == uniquePatches(minInd) & stimTag == 2, :), 1);
            else
                meanTrace.image.min = mean(resMat(imageIndex == uniquePatches(minInd) & stimTag == 1, :), 1);
                meanTrace.disc.min = mean(resMat(imageIndex == uniquePatches(minInd) & stimTag == 2, :), 1);
            end
            plot(meanTrace.image.min, 'k', 'linewidth', 2);
            plot(meanTrace.disc.min, 'r', 'linewidth', 2);
            legend({'image', 'disc'}); legend boxoff;
            ylim(get(ax(3), 'ylim'));
            title(sprintf('Patch %d (Low Nonlinearity)', uniquePatches(minInd)));
            ylabel(titleStr);
            xlabel('Time (ms)');
            
            % Patch visualizations - High nonlinearity patch
            % Patch image
            subplot(3, 4, 3);
            displayPatchImage(patchImg.max, imgAnalysis.apertureMatrix, imgAnalysis.radX, imgAnalysis.radY);
            title(sprintf('Patch %d | EquivI=%.3f', uniquePatches(maxInd), patchStats.max.equivalentIntensity));
            
            % Contrast histogram
            subplot(3, 4, 7);
            plotContrastHistogram(patchImg.max.contrast, imgAnalysis.apertureMatrix, patchStats.max);
            title(sprintf('Mean=%.3f, Std=%.3f', patchStats.max.patchContrast, patchStats.max.patchStd));
            
            % RF-weighted patch
            subplot(3, 4, 11);
            displayWeightedPatch(patchImg.max.contrast, imgAnalysis.weightingFxn, imgAnalysis.apertureMatrix);
            title(sprintf('RF-weighted | EquivContrast=%.3f', patchStats.max.equivalentContrast));
            
            % Patch visualizations - Low nonlinearity patch  
            % Patch image
            subplot(3, 4, 4);
            displayPatchImage(patchImg.min, imgAnalysis.apertureMatrix, imgAnalysis.radX, imgAnalysis.radY);
            title(sprintf('Patch %d | EquivI=%.3f', uniquePatches(minInd), patchStats.min.equivalentIntensity));
            
            % Contrast histogram
            subplot(3, 4, 8);
            plotContrastHistogram(patchImg.min.contrast, imgAnalysis.apertureMatrix, patchStats.min);
            title(sprintf('Mean=%.3f, Std=%.3f', patchStats.min.patchContrast, patchStats.min.patchStd));
            
            % RF-weighted patch
            subplot(3, 4, 12);
            displayWeightedPatch(patchImg.min.contrast, imgAnalysis.weightingFxn, imgAnalysis.apertureMatrix);
            title(sprintf('RF-weighted | EquivContrast=%.3f', patchStats.min.equivalentContrast));
            
            % Main title
            sgtitle(sprintf('%s | %s | Image: %s | Date: %s | Sample %d/%d', ...
                output.cellLabel, output.onlineAnalysis, output.imageName, output.expDate, k, numSamples), ...
                'FontSize', 14, 'FontWeight', 'bold');
        end
        
        % Store all traces for later analysis
        for patch = 1:numel(uniquePatches)
            output.allTraces.image(patch, :) = mean(resMat(imageIndex == uniquePatches(patch) & stimTag == 1, :), 1);
            output.allTraces.disc(patch, :) = mean(resMat(imageIndex == uniquePatches(patch) & stimTag == 2, :), 1);
        end
        
        % Calculate and store nonlinearity indices
        if strcmp(nodeName, 'exc')
            thresh = 10;
        elseif strcmp(nodeName, 'inh')
            thresh = 5;
        else
            thresh = 3;
        end
        
        output.NLI.onset = (output.stats.image.onset.mean - output.stats.disc.onset.mean) ./ ...
            (abs(output.stats.image.onset.mean) + abs(output.stats.disc.onset.mean));
        output.NLI.offset = (output.stats.image.offset.mean - output.stats.disc.offset.mean) ./ ...
            (abs(output.stats.image.offset.mean) + abs(output.stats.disc.offset.mean));
        
        % Apply threshold filter
        output.NLI.onset(max(abs(output.stats.image.onset.mean), abs(output.stats.disc.onset.mean)) < thresh) = 0;
        output.NLI.offset(max(abs(output.stats.image.offset.mean), abs(output.stats.disc.offset.mean)) < thresh) = 0;
        
        % Clean NLI values
        output.NLI.onset(isnan(output.NLI.onset)) = [];
        output.NLI.offset(isnan(output.NLI.offset)) = [];
        output.NLI.onset(isinf(output.NLI.onset)) = [];
        output.NLI.offset(isinf(output.NLI.offset)) = [];
        
        % Print summary statistics
        fprintf('%s %1.2f, %s %1.2f \n', 'MeanNLI Onset::', mean(output.NLI.onset), '::OffSet::', mean(output.NLI.offset));
        fprintf('%s %1.2f, %s %1.2f \n', 'Median NLI Onset::', median(output.NLI.onset), '::OffSet::', median(output.NLI.offset));
        
        % Report drug information
        drugAdded = selectedNodes{node}.epochList.firstValue.protocolSettings('epochGroup:externalSolutionAdditions');
        fprintf('%s , %s \n', 'drugs added are ::', drugAdded);
    end
    
    % Handle comparison between excitation and inhibition if both are present
    if numel(selectedNodes) == 2 
        processMultipleNodes(output, selectedNodes, RFsigma, apertureDiameter);
    end
end

%% Helper function to extract single patch from image analysis
function [patchImg, patchStats] = extractSinglePatch(imgAnalysis, patchLoc)
    xCenter = round(patchLoc(1));
    yCenter = round(patchLoc(2));
    
    radX = imgAnalysis.radX;
    radY = imgAnalysis.radY;
    
    xStart = xCenter - radX + 1;
    xEnd = xCenter + radX;
    yStart = yCenter - radY + 1;
    yEnd = yCenter + radY;
    
    % Extract patches
    patchImg.intensity = imgAnalysis.wholeImageMatrix(xStart:xEnd, yStart:yEnd) / 255;
    patchImg.contrast = imgAnalysis.contrastImage(xStart:xEnd, yStart:yEnd);
    
    % Calculate statistics
    apertureMatrix = imgAnalysis.apertureMatrix;
    weightingFxn = imgAnalysis.weightingFxn;
    patchInAperture = patchImg.contrast(apertureMatrix');
    
    patchStats.patchMean = mean(patchImg.intensity(apertureMatrix'));
    patchStats.patchStd = std(patchInAperture);
    patchStats.patchContrast = mean(patchInAperture);
    patchStats.equivalentContrast = sum(sum(weightingFxn .* patchImg.contrast));
    patchStats.equivalentIntensity = imgAnalysis.backgroundIntensity + patchStats.equivalentContrast * imgAnalysis.backgroundIntensity;
end

%% Helper function to display patch image
function displayPatchImage(patchImg, apertureMatrix, radX, radY)
    % Get aperture radius
    apertureRadius_pix = sum(apertureMatrix(:, round(size(apertureMatrix, 2)/2)));
    displayRadius_pix = round(apertureRadius_pix * 2);
    
    cropXStart = max(1, radX - displayRadius_pix + 1);
    cropXEnd = min(size(patchImg.intensity, 1), radX + displayRadius_pix);
    cropYStart = max(1, radY - displayRadius_pix + 1);
    cropYEnd = min(size(patchImg.intensity, 2), radY + displayRadius_pix);
    
    croppedPatch = patchImg.intensity(cropXStart:cropXEnd, cropYStart:cropYEnd);
    croppedAperture = apertureMatrix(cropXStart:cropXEnd, cropYStart:cropYEnd);
    
    maskedPatch = ones(size(croppedPatch)) * 0.5;
    maskedPatch(croppedAperture) = croppedPatch(croppedAperture);
    
    imagesc(maskedPatch');
    colormap(gca, gray);
    caxis([0 1]);
    axis image;
    axis off;
    hold on;
    
    % Draw aperture circle
    theta = linspace(0, 2*pi, 100);
    circCenterX = displayRadius_pix;
    circCenterY = displayRadius_pix;
    circX = circCenterX + apertureRadius_pix * cos(theta);
    circY = circCenterY + apertureRadius_pix * sin(theta);
    plot(circX, circY, 'g-', 'LineWidth', 1.5);
    hold off;
end

%% Helper function to plot contrast histogram
function plotContrastHistogram(patchContrast, apertureMatrix, patchStats)
    patchInAperture = patchContrast(apertureMatrix');
    histogram(patchInAperture, 30, 'FaceColor', [0.3, 0.5, 0.8], 'EdgeColor', 'none');
    hold on;
    xline(mean(patchInAperture), 'r-', 'LineWidth', 2);
    xline(patchStats.equivalentContrast, 'g--', 'LineWidth', 2);
    hold off;
    xlabel('Weber Contrast');
    ylabel('Count');
    legend('', 'Mean', 'Equiv.', 'Location', 'best');
    legend boxoff;
end

%% Helper function to display RF-weighted patch
function displayWeightedPatch(patchContrast, weightingFxn, apertureMatrix)
    apertureRadius_pix = sum(apertureMatrix(:, round(size(apertureMatrix, 2)/2)));
    displayRadius_pix = round(apertureRadius_pix * 2);
    
    radX = round(size(patchContrast, 1) / 2);
    radY = round(size(patchContrast, 2) / 2);
    
    cropXStart = max(1, radX - displayRadius_pix + 1);
    cropXEnd = min(size(patchContrast, 1), radX + displayRadius_pix);
    cropYStart = max(1, radY - displayRadius_pix + 1);
    cropYEnd = min(size(patchContrast, 2), radY + displayRadius_pix);
    
    croppedContrast = patchContrast(cropXStart:cropXEnd, cropYStart:cropYEnd);
    croppedWeighting = weightingFxn(cropXStart:cropXEnd, cropYStart:cropYEnd);
    croppedAperture = apertureMatrix(cropXStart:cropXEnd, cropYStart:cropYEnd);
    
    weightedPatch = croppedContrast .* (croppedWeighting ./ max(croppedWeighting(:)));
    maskedWeightedPatch = zeros(size(weightedPatch));
    maskedWeightedPatch(croppedAperture) = weightedPatch(croppedAperture);
    
    imagesc(maskedWeightedPatch');
    colormap(gca, redblue(256));
    caxis([-0.5, 0.5]);
    axis image;
    axis off;
    colorbar;
end

%% Red-blue colormap
function cmap = redblue(n)
    if nargin < 1, n = 256; end
    half = floor(n/2);
    r1 = linspace(0, 1, half)';
    g1 = linspace(0, 1, half)';
    b1 = ones(half, 1);
    r2 = ones(n - half, 1);
    g2 = linspace(1, 0, n - half)';
    b2 = linspace(1, 0, n - half)';
    cmap = [r1, g1, b1; r2, g2, b2];
end

function processMultipleNodes(output, selectedNodes, RFsigma, apertureDiameter)
    % Process shared patches between exc and inh recordings
    sharedPatch = intersect(output.patchRec.exc, output.patchRec.inh);
    excInd = find(ismember(output.patchRec.exc, sharedPatch));
    inhInd = find(ismember(output.patchRec.inh, sharedPatch));
    
    % Filter the data to only include shared patches
    output.deltaEqvInt.exc = output.deltaEqvInt.exc(excInd);
    output.deltaEqvInt.inh = output.deltaEqvInt.inh(inhInd);
    output.allTraces.inh.image = output.allTraces.inh.image(inhInd, :);
    output.allTraces.inh.disc = output.allTraces.inh.disc(inhInd, :);
    output.allTraces.exc.image = output.allTraces.exc.image(excInd, :);
    output.allTraces.exc.disc = output.allTraces.exc.disc(excInd, :);
    
    output.stats.exc.image.onset.mean = output.stats.exc.image.onset.mean(excInd);
    output.stats.exc.disc.onset.mean = output.stats.exc.disc.onset.mean(excInd);
    output.stats.inh.image.onset.mean = output.stats.inh.image.onset.mean(inhInd);
    output.stats.inh.disc.onset.mean = output.stats.inh.disc.onset.mean(inhInd);
    
    output.NLI.exc.onset = output.NLI.exc.onset(excInd);
    output.NLI.inh.onset = output.NLI.inh.onset(inhInd);
    output.NLI.exc.offset = output.NLI.exc.offset(excInd);
    output.NLI.inh.offset = output.NLI.inh.offset(inhInd);
    
    % Cluster on inhibitory traces
    nClusters = 3;
    colors = pmkmp(nClusters, 'IsoL');
    [idx] = pcaClustering(output.allTraces.inh.image, 3, nClusters, 1);
    
    % Create visualization figures for clusters
    createClusterFigures(output, idx, nClusters, colors);
    
    % Store cluster information
    output.clusterIndex = idx';
    
    % Calculate and store cluster summary statistics
    output.clusterSummary.excMean = splitapply(@mean, output.NLI.exc.onset, idx');
    output.clusterSummary.excErr = splitapply(@ste, output.NLI.exc.onset, idx');
    output.clusterSummary.inhMean = splitapply(@mean, output.NLI.inh.onset, idx');
    output.clusterSummary.inhErr = splitapply(@ste, output.NLI.inh.onset, idx');
end

function createClusterFigures(output, idx, nClusters, colors)
    % Create figures comparing image vs disc responses for each cluster
    figure('position', [900 250 800 600]);
    for i = 1:nClusters
        subplot(2, 2, i); hold all;
        plot(mean(output.allTraces.inh.image(idx == i, :)), 'color', 'k');
        plot(mean(output.allTraces.inh.disc(idx == i, :)), 'color', 'r');
        legend({'inh image', 'inh disc'}); legend boxoff;
        title(['cluster--', num2str(i)]);
    end
    s = sgtitle('image vs. disc inhibition'); set(s, 'fontsize', 20);
    
    % Excitation comparison figure
    figure('position', [900 250 800 600]);
    for i = 1:nClusters
        subplot(2, 2, i); hold all;
        plot(mean(output.allTraces.exc.image(idx == i, :)), 'color', 'k');
        plot(mean(output.allTraces.exc.disc(idx == i, :)), 'color', 'r');
        legend({'exc image', 'exc disc'}); legend boxoff;
        title(['cluster--', num2str(i)]);
    end
    s = sgtitle('image vs. disc excitation'); set(s, 'fontsize', 20);
    
    % Inh vs exc comparison
    figure('position', [900 250 800 600]);
    for i = 1:nClusters
        subplot(2, 2, i); hold all;
        plot(mean(output.allTraces.inh.image(idx == i, :)) / 4, 'color', 'k');
        plot(mean(output.allTraces.exc.image(idx == i, :)), 'color', 'r');
        legend({'inh', 'exc'}); legend boxoff;
        title(['cluster--', num2str(i)]);
    end
    s = sgtitle('inh vs. exc image'); set(s, 'fontsize', 20);
    
    % Cluster scatter plots
    figure('position', [50 50 600 600]); hold all;
    lgds = cell(nClusters, 1);
    for i = 1:nClusters
        scatter(output.stats.inh.image.onset.mean(idx == i), output.stats.exc.image.onset.mean(idx == i), 200, colors(i, :), 'filled');
        lgds{i} = ['cluster--', num2str(i)];
    end
    xlabel('image, inh'); ylabel('image, exc');
    legend(lgds); legend boxoff;
    
    % Delta mean vs inh difference
    figure('position', [50 50 600 600]); hold all;
    for i = 1:nClusters
        scatter(output.deltaEqvInt.inh(idx == i), ...
            output.stats.inh.image.onset.mean(idx == i) - output.stats.inh.disc.onset.mean(idx == i), 200, colors(i, :), 'filled');
    end
    xlabel('delta Mean'); ylabel('image-disc inh');
    legend(lgds); legend boxoff;
    
    % Delta mean vs exc difference
    figure('position', [50 50 600 600]); hold all;
    for i = 1:nClusters
        scatter(output.deltaEqvInt.exc(idx == i), ...
            output.stats.exc.image.onset.mean(idx == i) - output.stats.exc.disc.onset.mean(idx == i), 200, colors(i, :), 'filled');
    end
    xlabel('delta Mean'); ylabel('image-disc exc');
    legend(lgds); legend boxoff;
    
    % NLI visualization
    figure('position', [50 50 1000 600]); hold all;
    subplot(1, 2, 1);
    scatterWithMeanAndError(idx', output.NLI.exc.onset, output.clusterSummary.excMean, output.clusterSummary.excErr, ...
                         {'Cluster 1', 'Cluster 2', 'Cluster 3'}, 1);
    title('excitation NLI');
    
    subplot(1, 2, 2);
    scatterWithMeanAndError(idx', output.NLI.inh.onset, output.clusterSummary.inhMean, output.clusterSummary.inhErr, ...
                         {'Cluster 1', 'Cluster 2', 'Cluster 3'}, 1);
    title('inhibition NLI');
end