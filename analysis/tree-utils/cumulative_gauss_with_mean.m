function [fit] = cumulative_gauss_with_mean(beta, x)

fit = beta(1) * normcdf(x, beta(2), abs(beta(3))) + beta(4);
