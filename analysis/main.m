% main.m - Entry point for spatial integration analysis
%
% Paper: "Spatially-local inhibition and synaptic plasticity together enable
%         dynamic, context-dependent integration of parallel sensory pathways"
% Authors: Qiang Chen, Fred Rieke
% Contact: rieke@uw.edu
% Code:    https://github.com/chrischen2/spatialIntegration
% DOI:     https://zenodo.org/records/18491916
%
% This script sets up paths and data for analyzing OffT alpha RGC spatial
% integration. Run individual section scripts below for each analysis type.
% Each section script corresponds to specific paper figures as noted.
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
% Table of Contents (run each section script separately):
% -------------------------------------------------------
%   runExpandingSpots.m            - Supp. Fig 3A  (RF center size, DoG model)
%   runContrastSpots.m             - Fig 6A-B      (contrast response functions)
%   runContrastReversingGrating.m  - Fig 4         (subunit spatial tuning, F2)
%   runCRGPopulationEI.m           - Fig 4F, 6C    (E/I ratio across light levels)
%   runLinearDisc.m                - Fig 1-2       (NLI, natural image patches vs discs)
%   runFlashedGrating.m            - Fig 3, 5      (flashed grating, pharmacology)
%   runDrugPopulation.m            - Fig 5         (APB/LY, strychnine population)
%   runSplitFieldCentering.m       - Methods       (RF centering procedure)
%   runNoiseAnalysis.m             - Supp. Fig 5   (LN model identification)
%   runPPSpots.m                   - Fig 7A-H      (paired-pulse facilitation)
%   runPPGratings.m                - Fig 7I-N      (paired-pulse gratings, drug)

clearvars; close all; clc;

%% Set up paths
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
dataFolder = '/Users/chrischen/Library/CloudStorage/Dropbox/research/projects/spatialIntegration/fromFred/';
ovaExportFolder = dataFolder;

%% Load epoch list and sort chronologically
list = loader.loadEpochList([ovaExportFolder 'LinearEqvDisc.mat'], dataFolder);
for i = 1:list.length
    try
        list.elements(i).setProtocolSetting('user:startDate', ...
            datestr((list.elements(i).startDate)'));
    catch
        fprintf('fail to format  %i\n', i);
    end
end
listSorted = list.sortedBy('protocolSettings(user:startDate)');

fprintf('Data loaded. Run individual section scripts (e.g., runExpandingSpots) to proceed.\n');
