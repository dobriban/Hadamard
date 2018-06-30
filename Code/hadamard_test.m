function [D,had,R] = hadamard_test(X,Q,A,res)
%Test version used in simulations
%For applications use hadamard.m
%Compute hadamard estimator of variances of OLS

% Inputs
%X  - data matrix
%each row is a sample
%each column is a feature

%In addition: the following must be provided as inputs:
%Q - I - X(X' * X)^(-1)*X'
%A -  (X'*X)^(-1)
%res = y - X*b_ols; %OLS residuals

% Outputs
%D  - the matrix mapping squared residuals to hadamard estimator of variances of OLS
%had - Hadamard estimator
%R -  (Q \odot Q)^(-1)

T = Q.*Q;
T = (T+T')/2;
R = T^(-1); 
S = A*(X');
U = S.*S;
D = U*R; 
had = D*res.^2;