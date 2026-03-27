function [Kc,sigmaC,Ks,sigmaS,baseF] = fitDoG(spotSizes,responses,params0)
% [Kc,sigmaC,Ks,sigmaS] = fitDoGAreaSummation(spotSizes,responses,params0)
% MHT 05/2016
LB = [0, 2, 0, 10,0]; UB = [3 200 3 1000,1];
fitOptions = optimset('MaxIter',5000,'MaxFunEvals',1000*length(LB),'Display','off');

[params, ~, ~]=lsqcurvefit(@DoG,params0,spotSizes,responses,LB,UB,fitOptions);
Kc = params(1); sigmaC = params(2); Ks = params(3); sigmaS = params(4); baseF=params(5);
end