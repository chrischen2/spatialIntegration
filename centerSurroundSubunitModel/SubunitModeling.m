% SubunitModeling.m - Center-surround subunit model
%   Fits and visualizes subunit spatial tuning for contrast-reversing gratings.
%   Paper reference: Figure 4F, Supplementary Figure 4 (Chen & Rieke, 2026)
%   See Methods: "Center-Surround Subunit Model"
%
%   The model uses a Difference-of-Gaussians (DoG) receptive field for each
%   subunit, with half-wave rectification to capture nonlinear spatial
%   integration. The response to square-wave gratings of varying bar width
%   is computed by summing rectified subunit outputs across random phases.
%
%   External dependency: fminsearchbnd (MATLAB File Exchange, bounded Nelder-Mead)

clc; close all;
load AllTuningReversingGrating

%% Visualize single-cell RF and model fit
cellNo = 7;
positions = 0:5:400;
xloc = 0:0.1:400;
CenterSD = 20;
SurroundSD = 27;
surroundWeight = 1.02 * CenterSD / SurroundSD;

% Build DoG receptive field
nSub = length(positions);
GaussRF = zeros(nSub, length(xloc));
for sub = 1:nSub
    GaussRF(sub, :) = exp(-(xloc - positions(sub)).^2 / (2*CenterSD^2)) ...
        - surroundWeight * exp(-(xloc - positions(sub)).^2 / (2*SurroundSD^2));
end

figure(3); clf;
ax(1) = subplot(2, 1, 1);
plot(xloc, GaussRF(20, :), 'r', 'LineWidth', 3); hold on;
plot(xloc, GaussRF, 'k');
xlabel('position');
ylabel('weight');

% Compute bar responses using shared core function
barWidth = inhBarAll{cellNo};
denseBar = [5:5:51, 60:10:160];
inhBarRespAll{cellNo} = interp1(barWidth, inhBarRespAll{cellNo}, denseBar);
barWidth = denseBar;

barResp = computeSubunitResponse(CenterSD, SurroundSD, surroundWeight, ...
    barWidth, positions, xloc, 5);

ax(2) = subplot(2, 1, 2);
hold all;
plot(barWidth, inhBarRespAll{cellNo}, 'r');
plot(barWidth, barResp / max(barResp), 'k', 'LineWidth', 2);
xlabel('barwidth');
ylabel('response');
legend('data', 'model');

%% Fit subunit model to contrast-reversing grating data
clc; close all;
load AllTuningReversingGrating

options = optimset('PlotFcns', 'optimplotfval', 'TolX', 1e-2, 'MaxIter', 200, 'TolFun', 1e-3);
for cellNo = 7
    [x, fval] = fminsearchbnd(@(x) subunitModelFittingWrapper(x, inhBarAll{cellNo}, inhBarRespAll{cellNo}), ...
        [15 0.9 15 0.3], [12 0.6 10 0], [50 1.05 20 0.6], options);
    figure('position', [100 100 1000 500]);
    hold all;
    plot(inhBarAll{cellNo}, inhBarRespAll{cellNo}, 'k');
    barResp = subunitModelWrapper(x, inhBarAll{cellNo});
    plot(inhBarAll{cellNo}, barResp, 'r');
    legend('data', 'model');
    fprintf('cell ID %d  surround size %f  surround strength %f\n', cellNo, x(1), x(2));
    fitOutput(cellNo, :) = [x fval];
    title(sprintf('cell ID %d  surSize %f  Sstrength %f', cellNo, x(1), x(2)));
end

%% Plot summary of subunit fitting
figure;
scatter(fitOutput(:,1), fitOutput(:,2));
xlim([0 50]); ylim([0 1.2]); hold all;
eb(1) = errorbar(mean(fitOutput(:,1)), mean(fitOutput(:,2)), ste(fitOutput(:,1)), 'horizontal', 'LineStyle', 'none');
eb(2) = errorbar(mean(fitOutput(:,1)), mean(fitOutput(:,2)), ste(fitOutput(:,2)), 'vertical', 'LineStyle', 'none');

%% Parameter landscape: surround size vs. surround strength
[xx, yy] = meshgrid(10:1:60, 0.5:0.01:1);
F = zeros(size(xx));
for i = 1:size(F, 1)
    for j = 1:size(F, 2)
        F(i, j) = subunitModelSup([xx(i,j) yy(i,j)]);
    end
end
figure; surf(xx, yy, F); colorbar;
figure; contourf(F, 'showtext', 'on'); colorbar;
figure; surfc(xx, yy, F, 'FaceAlpha', 1); colorbar;
