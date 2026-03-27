function [stats] = computeSingleEpochLNmodel(paras,stimulus,response,stats,SETTINGS)

stats.lnModel= LNNodeModelWrapper(SETTINGS,stimulus,response,'');

figure('color','w','position',[800 50 600 1200]); subplot(2,1,1);
hold all;
plot(stats.lnModel.filterTimeStamps,  stats.lnModel.filter,'linewidth',2);
xlabel('time (s)');  title('filter'); legend boxoff;

subplot(2,1,2); hold all;
normFit.nlX=stats.lnModel.nlX/max(abs(stats.lnModel.nlX));
normFit.nlY=stats.lnModel.nlY;
plot(normFit.nlX, normFit.nlY-SETTINGS.baseShift, 'o');
scatter(normFit.nlX, normFit.nlY-SETTINGS.baseShift,10,'filled');

normFit.node=SigmoidNlNode();
if   paras.spikeTag
    normFit.node.fitToSample(normFit.nlX, normFit.nlY);
    %                 normFit.node.fitToSample(normFit.nlX, normFit.nlY,[200, 1 -0.5 5], ...
    %                     [1.1*min(normFit.nlY), 0, -1, 0],[1.1*max(normFit.nlY), 2, 1, 20]);
else
    normFit.node.fitToSample(normFit.nlX, normFit.nlY,[500, 0.1 0 -200]);
end
plot(normFit.nlX,sigmoid(normFit.node, normFit.nlX),'linewidth',2);

xlabel('Normalized filtered stimulus value');
ylabel('stats.lnModel (pA)');
title('nonlinearity');
stats.normLNModel=normFit;
end

