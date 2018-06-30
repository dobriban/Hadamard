%Experiment with White's and MacKinnon-White's
%covariance estimators in high dimensions
cd('C:\Dropbox\Projects\Sandwich\Experiments\Experiment 7 - Approx normality')
addpath('C:\Dropbox\Projects\Sandwich\Code')
%% Set parameters
n_mc = 1e3;
n = 1000;
%p = 100;
p = 50;

design = 1;
switch design
    case 1
        %Design 1: Toeplitz
        rho = 0.7;
        top = rho.^(0:1:n-1);
        Sigma = toeplitz(top);
        Sigma= diag(eig(Sigma));
        T_r = Sigma^(1/2);
    case 2
        %Design 2: Two variances
        Sigma = eye(n);
        c = 1e3;
        Sigma(1,1) = Sigma(1,1)+c*n/p;
        T_r = Sigma^(1/2);
end

%% MC simu of bias
rng(2);
X = randn(n,p); %could change it to correlateds
hSig = X'*X; %p^2n
A = (X'*X)^(-1); %p^3
S = A*(X');
T = (S.*S);
Q = eye(n) - X*A*(X'); %I-P, cost: np(n+p)
Lev = diag(Q);
[Z,~] = hadamard_test(X,Q,A,zeros(n,1));

ti = tic;
Vars = zeros(n_mc,1);
for i=1:n_mc
    ep  = T_r*randn(n,1); %heteroskedastic noise
    y =ep; %beta = 0
    b_ols = hSig\(X'*y); %p^3 
    res = y - X*b_ols; %resid
    unb_san = Z*res.^2;
    Vars(i) = unb_san(1);
end
time_el = toc(ti);

er = zscore(Vars);
[h,pv] = kstest(er);

%% Normality in a coordinate
rng(2);
savefigs=1;
k = 3*floor(sqrt(n_mc));
figure, hold on
histogram(er, 'NumBins', k);

xlabel('Z-score')
ylabel('Frequency')
set(gca,'fontsize',20)
str = sprintf( 'p/n=%.2f',p/n);
title(str);

if savefigs==1
    filename = ...
        sprintf( './zscore-unb-min-n=%d-p=%d-n-mc=%d-design=%d.png',...
        n,p,n_mc,design);
    saveas(gcf, filename,'png');
    fprintf(['Saved Results to ' filename '\n']);
    %close(gcf)
end

