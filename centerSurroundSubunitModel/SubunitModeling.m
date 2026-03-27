clc; close all;
load AllTuningReversingGrating
cellNo=7;
% grating stimuli re OffT
positions = 0:5:400;
xloc = 0:0.1:400;
CenterSD = 20;
SurroundSD =27;
surroundWeight = 01.02 * CenterSD/SurroundSD;
clear GaussRF;

for sub = 1:length(positions)
    GaussRF(sub, :) = exp(-(xloc - positions(sub)).^2/(2*CenterSD^2)) - surroundWeight * exp(-(xloc - positions(sub)).^2/(2*SurroundSD^2));
end

figure(3); clf
ax(1)=subplot(2, 1, 1);
plot(xloc, GaussRF(20, :), 'r', 'LineWidth', 3); hold on
plot(xloc, GaussRF, 'k');
xlabel('position');
ylabel('weight');

% Bar stimulus

barWidth=inhBarAll{cellNo};
denseBar=[5:5:51 60:10:160];
inhBarRespAll{cellNo}=interp1(barWidth, inhBarRespAll{cellNo}, denseBar);
barWidth=denseBar;
numShuffles =5;
barResp = zeros(1, length(barWidth));
for width = 1:length(barWidth)
    barStim = sign(sin(2*pi.*xloc./(barWidth(width)*2)));
    for shuffle = 1:numShuffles
        tempBarStim = circshift(barStim, randi(barWidth(width)));
        for sub = 1:length(positions)
            subResp = sum(GaussRF(sub, :) .* tempBarStim);
            if (subResp > 0)
                barResp(width) = barResp(width) + subResp;
            end
        end
    end
end
ax(2)=subplot(2, 1, 2);
hold all;
plot(barWidth, inhBarRespAll{cellNo},'r');
plot(barWidth, barResp / max(barResp),'k','LineWidth', 2);
xlabel('barwidth')
ylabel('response');
legend('data','model');


%% fit the subunit model for OffT traces to contrast reversing gratings.
clc; close all;
load AllTuningReversingGrating
% fitOutput=zeros(numel(inhBarAll),5);

options = optimset('PlotFcns','optimplotfval','TolX',1e-2,'MaxIter',200,'TolFun',1e-3);
for  cellNo=7
    [x fval] = fminsearchbnd(@(x) subunitModelFittingWrapper(x,inhBarAll{cellNo}, inhBarRespAll{cellNo}),[15 0.9 15 0.3],[12 0.6 10 0], [50, 1.05 20 0.6],options);
    figure('position',[100 100 1000 500]);
    hold all;
    plot(inhBarAll{cellNo}, inhBarRespAll{cellNo},'k');
    barResp=subunitModelWrapper(x, inhBarAll{cellNo});
    plot(inhBarAll{cellNo},barResp,'r');
    legend('data','model');
    fprintf('%s  %f %s %f %s %f \n','cell ID ', cellNo, 'surround size ', x(1),' surround strength ', x(2));
    fitOutput(cellNo,:)=[x fval];
    title(sprintf('%s %d  %s %f %s %f','cell ID ', cellNo, 'surSize ', x(1),' Sstrength ', x(2)));
end

%%  plot the summary of subunit fitting
figure; scatter(fitOutput(:,1), fitOutput(:,2)); xlim([0 50]); ylim([0 1.2]); hold all; 
 eb(1) = errorbar(mean(fitOutput(:,1)),mean(fitOutput(:,2)),ste(fitOutput(:,1)), 'horizontal', 'LineStyle', 'none');
    eb(2) = errorbar(mean(fitOutput(:,1)),mean(fitOutput(:,2)),ste(fitOutput(:,2)), 'vertical', 'LineStyle', 'none');
%% look at theoretical landscape of different surround size and strength 
[xx,yy]=meshgrid(10:1:60, 0.5:0.01:1);
F=zeros(size(xx));
for i=1:size(F,1)
    for j=1:size(F,2)
        tic
        F(i,j) = subunitModelSup([xx(i,j) yy(i,j)]);
        toc
    end
end
figure; surf(xx,yy,F); colorbar; 
figure;  contourf(F,'showtext','on'); colorbar;
figure;surfc(xx,yy,F,'FaceAlpha',1 ); colorbar; 