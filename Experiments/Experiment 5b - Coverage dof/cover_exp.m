%Experiment with White's and MacKinnon-White's
%covariance estimators in high dimensions
cd('C:\Dropbox\Projects\Sandwich\Experiments\Experiment 5b - Coverage dof')
addpath('C:\Dropbox\Projects\Sandwich\Code')
%% Set parameters
n_mc = 1e3;
n = 1000;
p_arr = [20,50,100, 200, 500, 750,900];
%p_arr = [20];
L = length(p_arr);
for l=1:length(p_arr)
    p = p_arr(l);
    
    design = 1;
    switch design
        case 1
            %Design 1: Toeplitz
            %rho = 0.9;
            rho = 0;
            top = rho.^(0:1:n-1);
            Sigma = toeplitz(top);
            Sigma= diag(eig(Sigma));
            T_r = Sigma^(1/2);
            param = rho;
        case 2
            %Design 2: Two variances
            Sigma = eye(n);
            c = 1e3;
            Sigma(1,1) = Sigma(1,1)+c*n/p;
            T_r = Sigma^(1/2);
            param = c;
    end
    n_met=4;
    
    %% MC simu of coverage
    rng(2);
    X = randn(n,p); %could change it to correlateds
    hSig = X'*X; %p^2n
    A = (X'*X)^(-1); %p^3
    S = A*(X');
    T = (S.*S);
    Q = eye(n) - X*A*(X'); %I-P, cost: np(n+p)
    Lev = diag(Q);
    [Z,~,R] = hadamard_test(X,Q,A,zeros(n,1));
    %Zr = ridge_sandwich(T,Q,n,p);
        
    ti = tic;
    prop_covered = zeros(n_met,n_mc);
    first_coord_covered = zeros(n_met,n_mc);
    
    for i=1:n_mc
        ep  = T_r*randn(n,1); %heteroskedastic noise
        y =ep; %beta = 0
        b_ols = hSig\(X'*y); %p^3
        res = y - X*b_ols; %resid
        D = res.^2;
        
        %White
        W = T*D;
        [prop_covered(1,i),  first_coord_covered(1,i)] = cover(W,b_ols);
        
        %MacKinnon and White [extra cost np(n+p), and maybe less]
        D = diag(Lev)^(-1)*res.^2;
        MW = T*D;
        [prop_covered(2,i),  first_coord_covered(2,i)] = cover(MW,b_ols);
        
        %Hadamard
        hada = Z*res.^2;
        [prop_covered(3,i),  first_coord_covered(3,i)] = cover(hada,b_ols);
        
        %Hadamard dof
        dof = dof_hadamard_test(A,T,R,n);
        beta = zeros(p,1);
        [prop_covered(4,i),  first_coord_covered(4,i)] = cover(hada,b_ols,beta,'t',dof);
        
%         %Ridge
%         ri = Zr*res.^2;
%         [prop_covered(5,i),  first_coord_covered(5,i)] = cover(ri,b_ols);
    end
    time_el = toc(ti);
    
    %%
    % m = 1-mean(prop_covered,2);
    % sd_p = var(prop_covered).^(1/2);
%% Overall coverage
    %mb = 1- first_coord_covered; %mean over the MC
    mb = 1- prop_covered; %mean over the MC
    mba = mean(mb,2);
    rng(2);
    savefigs=1;    closefigs=1;
    k = floor(3*sqrt(p));
    figure, hold on
    histogram(mb(1,:), 'NumBins', k);
    histogram(mb(2,:), 'NumBins', k);
    histogram(mb(3,:), 'NumBins', k);
    histogram(mb(4,:), 'NumBins', k);
    %histogram(mb(5,:), 'NumBins', k);
    
    
   
    ws = sprintf( 'White %.5f', mba(1));
    mws = sprintf( 'MW %.5f', mba(2));
    mbs = sprintf( 'Hadamard %.5f', mba(3));
    tmbs = sprintf( 'Hadamard-t %.5f', mba(4));
   % r = sprintf( 'Ridge %.5f', mba(5));
    
    
    %legend({ws,mws,mbs,tmbs,r},'location','Best')
    legend({ws,mws,mbs,tmbs},'location','Best')
    xlabel('Mean type I error')
    ylabel('Frequency')
    set(gca,'fontsize',20)
    str = sprintf( 'p/n=%.2f',p/n);
    title(str);
    
    if savefigs==1
        filename = ...
            sprintf( './mean-cover-n=%d-p=%d-n-mc=%d-design=%d-param=%.2f.png',...
            n,p,n_mc,design,param);
        saveas(gcf, filename,'png');
        fprintf(['Saved Results to ' filename '\n']);
        if closefigs==1
            close(gcf)
        end
    end
 
%% Coverage in one coordinate
    mb = 1- first_coord_covered; %mean over the MC
    %mb = 1- prop_covered; %mean over the MC
    mba = mean(mb,2);
    rng(2);
    savefigs=1;    closefigs=1;
    k = floor(3*sqrt(p));
    figure, hold on
    histogram(mb(1,:), 'NumBins', k);
    histogram(mb(2,:), 'NumBins', k);
    histogram(mb(3,:), 'NumBins', k);
    histogram(mb(4,:), 'NumBins', k);
    %histogram(mb(5,:), 'NumBins', k);
    
    
   
    ws = sprintf( 'White %.5f', mba(1));
    mws = sprintf( 'MW %.5f', mba(2));
    mbs = sprintf( 'Hadamard %.5f', mba(3));
    tmbs = sprintf( 'Hadamard-t %.5f', mba(4));
   % r = sprintf( 'Ridge %.5f', mba(5));
    
    
    %legend({ws,mws,mbs,tmbs,r},'location','Best')
    legend({ws,mws,mbs,tmbs},'location','Best')
    xlabel('Type I error in first coordinate')
    ylabel('Frequency')
    set(gca,'fontsize',20)
    str = sprintf( 'p/n=%.2f',p/n);
    title(str);
    
    if savefigs==1
        filename = ...
            sprintf( './cover-first-n=%d-p=%d-n-mc=%d-design=%d-param=%.2f.png',...
            n,p,n_mc,design,param);
        saveas(gcf, filename,'png');
        fprintf(['Saved Results to ' filename '\n']);
        if closefigs==1
            close(gcf)
        end
    end
end
