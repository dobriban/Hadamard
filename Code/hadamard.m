function [had] = hadamard(X,res)
%Compute hadamard estimator of variances of OLS
%See the paper "Robust Inference Under Heteroskedasticity via the Hadamard
%Estimator" by Dobriban and Su for reference.

% Inputs
%X  - data matrix
%each row is a sample
%each column is a feature

%res = y - X*b_ols; %OLS residuals

% Outputs
%had - Hadamard estimator of variances of OLS

[n,~] = size(X);
A = (X'*X)^(-1); 
Q = eye(n) - X*A*X';
T = Q.*Q;
T = (T+T')/2;
R = T^(-1); 
S = A*(X');
U = S.*S;
D = U*R; 
had = D*res.^2;