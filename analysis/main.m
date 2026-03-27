% main.m - Entry point for spatial integration analysis
%
% Paper: "Spatially-local inhibition and synaptic plasticity together enable
%         dynamic, context-dependent integration of parallel sensory pathways"
% Authors: Qiang Chen, Fred Rieke
% Contact: rieke@uw.edu
% Code:    https://github.com/chrischen2/spatialIntegration
% DOI:     https://zenodo.org/records/18491916
%
% This script sets up paths and data, then provides GUI creation sections
% for each analysis group. Run each %% section to create the appropriate
% GUI, then open the corresponding analysis script in analyses/ and run
% its sections interactively.
%
% External dependencies:
%   - Rieke Lab Framework (riekesuite): data loading, epoch tree GUI
%     riekesuite.getResponseMatrix, riekesuite.analysis.buildTree,
%     riekesuite.util.SplitValueFunctionAdapter.buildMap,
%     epochTreeGUI, edu.washington.rieke.Analysis.*
%   - LNNodeModelWrapper / SigmoidNlNode: LN model fitting
%     (see https://github.com/chrischen2/cascadeGraph)
%   - fminsearchbnd: bounded Nelder-Mead optimizer (MATLAB File Exchange)
%
% Analysis scripts (in analyses/ folder):
% -------------------------------------------------------
%   analysisLinearDisc.m          - Fig 1-2   (NLI, natural image patches vs discs)
%   analysisFlashedGrating.m     - Fig 3, 5  (flashed grating, drug population)
%   analysisCRG.m                - Fig 4, 6C (contrast reversing grating, E/I)
%   analysisRFCharacterization.m - Fig 6A-B, Supp Fig 3A (expanding spots, contrast)
%   analysisPairedPulse.m        - Fig 7     (paired-pulse spots & gratings)
%   analysisSupplementary.m      - Methods, Supp Fig 5 (centering, noise/LN model)

clearvars; close all; clc;

%% Set up paths
addpath(fullfile(fileparts(mfilename('fullpath')), 'analyses'));
addpath(fullfile(fileparts(mfilename('fullpath')), 'analyzeFunctions'));
addpath(fullfile(fileparts(mfilename('fullpath')), 'utils'));
addpath(fullfile(fileparts(mfilename('fullpath')), 'tree-utils'));

% Summary data folder (adjust if needed)
summaryFolder = fullfile(fileparts(mfilename('fullpath')), 'summary');

%% Load data and initialize Rieke Lab framework
import auimodel.*
import vuidocument.*
loader = edu.washington.rieke.Analysis.getEntityLoader();
treeFactory = edu.washington.rieke.Analysis.getEpochTreeFactory();
listFactory = edu.washington.rieke.Analysis.getEpochListFactory();

% Configure data paths (adjust to your local data location)
dataFolder = fullfile(fileparts(mfilename('fullpath')), 'data');
ovaExportFolder = dataFolder;

%% Load epoch list and sort chronologically
list = loader.loadEpochList([ovaExportFolder filesep 'LinearEqvDisc.mat'], dataFolder);
for i = 1:list.length
    try
        list.elements(i).setProtocolSetting('user:startDate', ...
            datestr((list.elements(i).startDate)'));
    catch
        fprintf('fail to format  %i\n', i);
    end
end
listSorted = list.sortedBy('protocolSettings(user:startDate)');
fprintf('Data loaded. Run GUI sections below, then open analysis scripts.\n');

%% ===== Fig 1-2: Linear Equivalent Disc (NLI) =====
% Creates standard GUI used by analysisLinearDisc.m and analysisFlashedGrating.m
cellTypeSplit = @(listSorted)splitOnCellType(listSorted);
cellTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, cellTypeSplit);

dateSplit = @(listSorted)splitOnExperimentDate(listSorted);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, dateSplit);

ndfSplit = @(listSorted)splitOnNDFs(listSorted);
ndfSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, ndfSplit);

brightnessSplit = @(listSorted)splitOnDeviceBrightNess(listSorted);
brightnessSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, brightnessSplit);

tree = riekesuite.analysis.buildTree(listSorted,{cellTypeSplit_java,dateSplit_java, 'cell.label', 'protocolSettings(epochGroup:label)',...
    brightnessSplit_java, ndfSplit_java,'protocolSettings(onlineAnalysis)'});
gui = epochTreeGUI(tree);
%   >> run('analysisLinearDisc')

%% ===== Fig 3, 5: Flashed Grating & Pharmacology =====
% Uses same standard GUI as above (no re-creation needed)
%   >> run('analysisFlashedGrating')

%% ===== Fig 4, 6C: Contrast Reversing Grating =====
% Re-creates standard GUI for fresh tree
cellTypeSplit = @(listSorted)splitOnCellType(listSorted);
cellTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, cellTypeSplit);

dateSplit = @(listSorted)splitOnExperimentDate(listSorted);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, dateSplit);

ndfSplit = @(listSorted)splitOnNDFs(listSorted);
ndfSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, ndfSplit);

brightnessSplit = @(listSorted)splitOnDeviceBrightNess(listSorted);
brightnessSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, brightnessSplit);

tree = riekesuite.analysis.buildTree(listSorted,{cellTypeSplit_java,dateSplit_java, 'cell.label', 'protocolSettings(epochGroup:label)',...
    brightnessSplit_java, ndfSplit_java,'protocolSettings(onlineAnalysis)'});
gui = epochTreeGUI(tree);
%   >> run('analysisCRG')

%% ===== Supp Fig 3A, Fig 6A-B: RF Characterization =====
% Standard GUI (expanding spots + contrast spots share same splits)
cellTypeSplit = @(listSorted)splitOnCellType(listSorted);
cellTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, cellTypeSplit);

dateSplit = @(listSorted)splitOnExperimentDate(listSorted);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, dateSplit);

ndfSplit = @(listSorted)splitOnNDFs(listSorted);
ndfSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, ndfSplit);

brightnessSplit = @(listSorted)splitOnDeviceBrightNess(listSorted);
brightnessSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, brightnessSplit);

tree = riekesuite.analysis.buildTree(listSorted,{cellTypeSplit_java, dateSplit_java, 'cell.label','protocolSettings(epochGroup:label)',...
    brightnessSplit_java, ndfSplit_java,'protocolSettings(onlineAnalysis)'});
gui = epochTreeGUI(tree);
%   >> run('analysisRFCharacterization')

%% ===== Fig 7A-H: Paired-Pulse Spots =====
rigSplit = @(listSorted)splitOnRigs(listSorted);
rigSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, rigSplit);

cellTypeSplit = @(listSorted)splitOnCellType(listSorted);
cellTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, cellTypeSplit);

dateSplit = @(listSorted)splitOnExperimentDate(listSorted);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, dateSplit);

tree = riekesuite.analysis.buildTree(listSorted,{cellTypeSplit_java, dateSplit_java,rigSplit_java,'protocolSettings(psth)', ...
    'cell.label','protocolSettings(epochGroup:label)'});
gui = epochTreeGUI(tree);
%   >> run('analysisPairedPulse')  % run PP spots sections

%% ===== Fig 7I-N: Paired-Pulse Gratings =====
cellTypeSplit = @(listSorted)splitOnCellType(listSorted);
cellTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, cellTypeSplit);

dateSplit = @(listSorted)splitOnExperimentDate(listSorted);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, dateSplit);

tree = riekesuite.analysis.buildTree(listSorted,{cellTypeSplit_java, dateSplit_java, 'cell.label','protocolSettings(grateContrast)', ...
    'protocolSettings(psth)'});
gui = epochTreeGUI(tree);
%   >> run('analysisPairedPulse')  % run PP gratings sections

%% ===== Methods: Split-Field Centering =====
cellTypeSplit = @(listSorted)splitOnCellType(listSorted);
cellTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, cellTypeSplit);

dateSplit = @(listSorted)splitOnExperimentDate(listSorted);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, dateSplit);

splitSplit = @(listSorted)splitOnSplitField(listSorted);
splitSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, splitSplit);

ndfSplit = @(listSorted)splitOnNDFs(listSorted);
ndfSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, ndfSplit);

brightnessSplit = @(listSorted)splitOnDeviceBrightNess(listSorted);
brightnessSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, brightnessSplit);

tree = riekesuite.analysis.buildTree(listSorted,{cellTypeSplit_java,dateSplit_java, 'cell.label', 'protocolSettings(epochGroup:label)',...
    splitSplit_java,'protocolSettings(onlineAnalysis)',brightnessSplit_java, ndfSplit_java});
gui = epochTreeGUI(tree);
%   >> run('analysisSupplementary')  % run centering sections

%% ===== Supp Fig 5: Noise / LN Model =====
cellTypeSplit = @(listSorted)splitOnCellType(listSorted);
cellTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, cellTypeSplit);

dateSplit = @(listSorted)splitOnExperimentDate(listSorted);
dateSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, dateSplit);

protocolSplit = @(listSorted)splitOnShortProtocolID(listSorted);
ProtocolSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, protocolSplit);

ndfSplit = @(listSorted)splitOnNDFs(listSorted);
ndfSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, ndfSplit);

brightnessSplit = @(listSorted)splitOnDeviceBrightNess(listSorted);
brightnessSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, brightnessSplit);

recordingTypeSplit = @(listSorted)splitOnRecordingType(listSorted);
recordingTypeSplit_java = riekesuite.util.SplitValueFunctionAdapter.buildMap(listSorted, recordingTypeSplit);

tree = riekesuite.analysis.buildTree(listSorted,{cellTypeSplit_java,dateSplit_java, 'cell.label','protocolSettings(epochGroup:label)',...
    ProtocolSplit_java, brightnessSplit_java,ndfSplit_java,recordingTypeSplit_java});
gui = epochTreeGUI(tree);
%   >> run('analysisSupplementary')  % run noise sections
