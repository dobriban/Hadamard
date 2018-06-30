%Experiment with White's and MacKinnon-White's
%covariance estimators in high dimensions
cd('C:\Dropbox\Projects\Sandwich\Experiments\Experiment 6 - MSE')
addpath('C:\Dropbox\Projects\Sandwich\Code')
%% Set parameters
n_mc = 1e3;
n = 200;
p = 100;
%p = 150;

design = 1;
switch design
    case 1
        %Design 1: Toeplitz
        rho = 0.9;
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
n_met=3;

%% MC simu of bias
rng(2);
X = randn(n,p); %could change it to correlateds
hSig = X'*X; %p^2n
A = (X'*X)^(-1); %p^3
S = A*(X');
T = (S.*S);
Q = eye(n) - X*A*(X'); %I-P, cost: np(n+p)
Lev = diag(Q);
[Z,unb_san] = hadamard_test(X,Q,A,zeros(n,1));

ti = tic;
Var = zeros(p,n_met,n_mc);
NormBeta = zeros(n_met,n_mc);
for i=1:n_mc
    ep  = T_r*randn(n,1); %correlated noise
    y =ep; %beta = 0
    b_ols = hSig\(X'*y); %p^3
    
    res = y - X*b_ols; %resid
    D = res.^2;
    
    %White
    W = T*D;
    Var(:,1,i) = W;
    NormBeta(1,i) = norm(b_ols)^2-sum(W);
    
    %MacKinnon and White [extra cost np(n+p), and maybe less]
    D = diag(Lev)^(-1)*res.^2;
    MW = T*D;
    Var(:,2,i) = MW;
    NormBeta(2,i) = norm(b_ols)^2-sum(MW);
    
    %Unbiased
    unb_san = Z*res.^2;
    Var(:,3,i) = unb_san;
    NormBeta(3,i) = norm(b_ols)^2-sum(unb_san);
end
time_el = toc(ti);


V = (S.*S)*diag(Sigma); %this is the true vector of variances.
MSE = sum(V);
SNR= 0;
%find bias in estim MSE
bias_MSE = zeros(n_met,n_mc);
for i=1:n_met
    for j=1:n_mc
        bias_MSE(i,j) = sum(Var(:,i,j)) - MSE;
    end
end
%bias_MSE  = p^(1/2)*bias_MSE;
mb1 = mean(bias_MSE,2);


%% Bias overall
%mb = mean(bias_MSE,2); %mean over the MC
%mba = mean(abs(mb),1);
mb = bias_MSE';
mba = mean(mb,1);
rng(2);
savefigs=1;
k = 3*floor(sqrt(n_mc));
figure, hold on
histogram(mb(:,1), 'NumBins', k);
histogram(mb(:,2), 'NumBins', k);
histogram(mb(:,3), 'NumBins', k);

ws = sprintf( 'White %.5f', mba(1));
mws = sprintf( 'MW %.5f', mba(2));
mbs = sprintf( 'Hadamard %.5f', mba(3));

legend({ws,mws,mbs},'location','Best')
xlabel('Bias for MSE')
ylabel('Frequency')
set(gca,'fontsize',20)
str = sprintf( 'p/n=%.2f',p/n);
title(str);

if savefigs==1
    filename = ...
        sprintf( './mean-bias-MSE-White-MW-min-n=%d-p=%d-n-mc=%d-design=%d.png',...
        n,p,n_mc,design);
    saveas(gcf, filename,'png');
    fprintf(['Saved Results to ' filename '\n']);
    %close(gcf)
end

