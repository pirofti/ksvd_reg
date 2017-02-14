%% Test Regularized K-SVD
clear; clc; close all; fclose all; format compact;
%%-------------------------------------------------------------------------
p = 16;         % problem dimension
n = 32;         % number of atoms in the dictionary
m = 100;        % number of training signals
s = 4;          % sparsity constraint
reg = 0.1;      % regularization
vanish = 1;     % regularization vanishing factor
regstop = 31;   % cancel regularization term starting with this iteration
iters = 50;     % DL iterations
%%-------------------------------------------------------------------------
% Path to OMP implementation by Ron Rubinstein
% (http://www.cs.technion.ac.il/~ronrubin/Software/ompbox10.zip)
addpath('ompbox10');
%%-------------------------------------------------------------------------
Y = randn(p,m);
D0 = normc(randn(p,n));

params = {'reg', reg, 'vanish', vanish, 'regstop', regstop};
[Df, Xf, errs] = DL(Y, D0, s, iters, params{:});
plot(1:iters, errs);