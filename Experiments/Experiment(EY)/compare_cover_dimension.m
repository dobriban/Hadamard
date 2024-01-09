% cd('/Users/elsayang/Dropbox (Penn)/sandwich/Code/new/')
n_mc = 1e3;
n = 1000;
%p_arr = [20,52,100, 200, 500, 750,900];
%p_arr=[100,500,750];
%p_arr=600:100:800;
%p_arr=800
p_arr = 100:100:800
L = length(p_arr);
n_met=6;
first_coord_covered_mean_diffp = zeros(n_met,L);
second_coord_covered_mean_diffp = zeros(n_met,L);
neg1=zeros(n_met,L);  neg2=neg1; pos1=neg1; pos2=neg1;
for l=1:length(p_arr)
    p = p_arr(l);
    
    design = 3;
    switch design
        case 1
            %Design 1: Toeplitz
            rho = 0.9;
            %rho = 0;
            X = randn(n,p);
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

        case 3
            X = trnd(10, n, p);
            %X1 = trnd(10, n, p);
            Sigma =abs(X(:,1));
            T_r = (diag(Sigma))^(1/2);

        case 4
            X = trnd(10, n, p);
            X1 = trnd(10, n, p);
            Sigma =abs(X1(:,1));
            %Sigma = diag(sum(X.^2,2));
            T_r = (diag(Sigma))^(1/2);

        case 5
            X = randn(n,p);
            %Sigma =abs(X1(:,1));
            Sigma = diag(sum(X.^2,2));
            T_r = (diag(Sigma))^(1/2);
            
        case 6
            X = randn(n,p);
            Sigma = [ones(1,n/2),3*ones(1,n/2)];
            T_r = diag(Sigma);
            
    end
    

rng(2);


hSig = X'*X; %p^2n
    A = (X'*X)^(-1); %p^3
    S = A*(X');
    T = (S.*S);
    Q = eye(n) - X*A*(X'); %I-P, cost: np(n+p)
    Lev = diag(Q);
    [Z,~,R] = hadamard_test(X,Q,A,zeros(n,1));
    %Zr = ridge_sandwich(T,Q,n,p);
   
%check condition of Theorem 5
J = A*X'*T_r;
J2 = J*J';
for i = 1:p
    for j = 1:p
      B(i,j)=(J(i,j))^2/J2(i,i);
    end
end
plot(max(B'))  % diaplay the maximum of each row of B

    ti = tic;
    prop_covered = zeros(n_met,n_mc);
    first_coord_covered = zeros(n_met,n_mc);
    %covered = zeros(p,n_mc);
    all_coord_covered = cell(1, n_met);
    for i = 1:n_met
      all_coord_covered{i} = zeros(p, n_mc);
    end
    
      for i=1:n_mc
            ep  = T_r*randn(n,1); %heteroskedastic noise
            %ep0 = rand(1,n)-0.5;
            %ep  = T_r* reshape(ep0,[],1);
            y =ep; %beta = 0
            b_ols = hSig\(X'*y); %p^3
            res = y - X*b_ols; %resid
            D = res.^2;

            %White
            W = T*D;

            %W_covered(,i) = cover(W,b_ols);

            [prop_covered(1,i),  first_coord_covered(1,i), all_coord_covered{1}(:,i)] = cover(W,b_ols);

            
            %[prop_covered1, first_coord_covered1, covered{1}(:,i)] = cover(W,b_ols);
            %covered{1}(:,i) = covered1;

            %MacKinnon and White [extra cost np(n+p), and maybe less]
            D = diag(Lev)^(-1)*res.^2;
            MW = T*D;
            [prop_covered(2,i),  first_coord_covered(2,i),all_coord_covered{2}(:,i)] = cover(MW,b_ols);

            %Hadamard
            hada = Z*res.^2;
            [prop_covered(3,i),  first_coord_covered(3,i),all_coord_covered{3}(:,i)] = cover(hada,b_ols);

            %Hadamard dof
            dof = dof_hadamard(A,T,R,n);
            beta = zeros(p,1);
            [prop_covered(4,i),  first_coord_covered(4,i),all_coord_covered{4}(:,i)] = cover(hada,b_ols,beta,'t',dof);
    
            n_bootstrap = 100;
            %5th method
            b_pair = zeros(p,n_bootstrap);
            ZZ = [X,y];
            for j=1:n_bootstrap
                Z_tmp = datasample(ZZ, n, 1, 'Replace', true);
                X_tmp = Z_tmp(:,1:p);
                y_tmp = Z_tmp(:,p+1);
                hSig_tmp = X_tmp'*X_tmp;
                if p>500
                    b_pair(:,j) = pinv(hSig_tmp)*(X_tmp'*y_tmp); %p^3
                else
                    b_pair(:,j) = hSig_tmp\(X_tmp'*y_tmp); %p^3
                end
            end
            q_lo  = quantile(b_pair',0.025)';
            q_hi  = quantile(b_pair',1-0.025)';
            [prop_covered(5,i),  first_coord_covered(5,i),all_coord_covered{5}(:,i)] = cover_bootstrap(b_ols, q_lo,q_hi);

            %6th method residual bootstrap
            b_resboot = zeros(p,n_bootstrap);
            for j=1:n_bootstrap
                ep_tmp = datasample(ep, n);
                y_tmp =ep_tmp; %beta = 0
                b_resboot(:,j) = hSig\(X'*y_tmp); %p^3
            end
            q_lo  = quantile(b_resboot',0.025)';
            q_hi  = quantile(b_resboot',1-0.025)';
            [prop_covered(6,i),  first_coord_covered(6,i),all_coord_covered{6}(:,i)] = cover_bootstrap(b_ols, q_lo,q_hi);

           end
        time_el = toc(ti);
    
    
    figure, hold on
    
    %plot(MW);
    plot(hada);
    plot(diag(J2));
    close(gcf);

    m_W =1- mean(all_coord_covered{1},2);
    m_MW= 1-mean(all_coord_covered{2},2);
    m_hada=1- mean(all_coord_covered{3},2);
    m_hada_t=1- mean(all_coord_covered{4},2);
    m_pairboot = 1- mean(all_coord_covered{5},2);
    m_resboot = 1- mean(all_coord_covered{6},2);
    
    MAD = zeros(n_met);
    MAD(1)=sum(abs(m_W-0.05))/p;
    MAD(2)=sum(abs(m_MW-0.05))/p;
    MAD(3) = sum(abs(m_hada-0.05))/p;
    MAD(4) = sum(abs(m_hada_t-0.05))/p;
    MAD(5) = sum(abs(m_pairboot-0.05))/p;
    MAD(6) = sum(abs(m_resboot-0.05))/p;

    savefigs=1;    closefigs=1;

    figure, hold on
    k = 3*floor(sqrt(p));
    color1=[0, 0.4470, 0.7410];
    color2=[0.8500, 0.3250, 0.0980];
    color3=[0.9290, 0.6940, 0.1250];
    color4=[0.4940, 0.1840, 0.5560];
    color5=[0.4660 0.6740 0.1880];
    color6=[0.3010 0.7450 0.9330];

    histogram(m_W, 'NumBins',k,'FaceColor', color1);
    histogram(m_MW, 'NumBins',k, 'FaceColor', color2);
    histogram(m_hada, 'NumBins', k, 'FaceColor', color3);
    histogram(m_hada_t, 'NumBins', k, 'FaceColor', color4);
    histogram(m_pairboot, 'NumBins', k, 'FaceColor', color5);
    histogram(m_resboot, 'NumBins', k, 'FaceColor', color6);
   
    ws = sprintf( 'White MAD %.5f', MAD(1));
    mws = sprintf( 'MW MAD %.5f', MAD(2));
    mbs = sprintf( 'Hadamard MAD %.5f', MAD(3));
    tmbs = sprintf( 'Hadamard-t MAD %.5f', MAD(4));
    pairboot = sprintf( 'Pair bootstrap %.5f', MAD(5));
    resboot = sprintf( 'Residual bootstrap %.5f', MAD(6));


    legend({ws,mws,mbs,tmbs,pairboot,resboot},'Location','northeast','FontSize',14)
    xlabel('type I error for each coordinate')
    ylabel('Frequency')
    set(gca,'fontsize',20)
    str = sprintf( 'p/n=%.2f',p/n);
    title(str);
    if savefigs==1
        filename = ...
            sprintf( './figs/all-coordinateds-cover-n=%d-p=%d-n-mc=%d-design=%d.png',...
            n,p,n_mc,design);
        saveas(gcf, filename,'png');
        fprintf(['Saved Results to ' filename '\n']);
        if closefigs==1
            close(gcf)
        end
    end
    

    
    %Overall coverage
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
    histogram(mb(5,:), 'NumBins', k);
    histogram(mb(6,:), 'NumBins', k);
    
    
   
    ws = sprintf( 'White %.5f', mba(1));
    mws = sprintf( 'MW %.5f', mba(2));
    mbs = sprintf( 'Hadamard %.5f', mba(3));
    tmbs = sprintf( 'Hadamard-t %.5f', mba(4));
    pairboot = sprintf( 'Pair bootstrap %.5f', mba(5));
    resboot = sprintf( 'Residual bootstrap %.5f', mba(6));
    
    
    %legend({ws,mws,mbs,tmbs,r},'location','Best')
    legend({ws,mws,mbs,tmbs, pairboot,resboot},'location','Best')
    xlabel('Mean type I error')
    ylabel('Frequency')
    set(gca,'fontsize',20)
    str = sprintf( 'p/n=%.2f',p/n);
    title(str);
    
    if savefigs==1
        filename = ...
            sprintf( './figs/mean-cover-n=%d-p=%d-n-mc=%d-design=%d.png',...
            n,p,n_mc,design);
        saveas(gcf, filename,'png');
        fprintf(['Saved Results to ' filename '\n']);
        if closefigs==1
            close(gcf)
        end
    end
    

 
%% Coverage in one coordinate
%  mb = 1- first_coord_covered; %mean over the MC
% mba = mean(mb,2);
     mba1 = [m_W(1);m_MW(1);m_hada(1);m_hada_t(1);m_pairboot(1);m_resboot(1)];
     pci1 = zeros(numel(mba1), 2);
    
    % Calculate the confidence interval for each mba value
    for i = 1:numel(mba1)
        [~, ci] = binofit(round(mba1(i)*n_mc), n_mc, 0.05);
        pci1(i, :) = ci;
    end
    neg1(:,l) = mba1-pci1(:,1);
    pos1(:,l) = pci1(:,2)-mba1;
        
     
    first_coord_covered_mean_diffp(:,l)=mba1;

     mba2 = [m_W(2);m_MW(2);m_hada(2);m_hada_t(2);m_pairboot(2);m_resboot(2)];
     pci2 = zeros(numel(mba2), 2);

     for i = 1:numel(mba1)
        [~, ci] = binofit(round(mba2(i)*n_mc), n_mc, 0.05);
        pci2(i, :) = ci;
     end
     neg2(:,l) = mba2-pci2(:,1);
     pos2(:,l) = pci2(:,2)-mba2;

     second_coord_covered_mean_diffp(:,l)=mba2;

end


%Plot the Clopper-Pearson interval for type I error for the first coordinate 
% as a function of dimension p

savefigs=1;    closefigs=1;

color1=[0, 0.4470, 0.7410];
color2=[0.8500, 0.3250, 0.0980];
color3=[0.9290, 0.6940, 0.1250];
color4= [0.4940, 0.1840, 0.5560];
color5=[0.4660 0.6740 0.1880];
color6=[0.3010 0.7450 0.9330];

figure, hold on
plot(p_arr, first_coord_covered_mean_diffp(1,:),'o-', 'LineWidth', 2,'Color', color1);
plot(p_arr, first_coord_covered_mean_diffp(2,:),'s--', 'LineWidth', 2,'Color', color2);
plot(p_arr, first_coord_covered_mean_diffp(3,:),'+-.', 'LineWidth', 2,'Color', color3);
plot(p_arr, first_coord_covered_mean_diffp(4,:),'*:', 'LineWidth', 2,'Color', color4);
plot(p_arr, first_coord_covered_mean_diffp(5,:),'*:', 'LineWidth', 2,'Color', color5);
plot(p_arr, first_coord_covered_mean_diffp(6,:),'*:', 'LineWidth', 2,'Color', color6);
set(gca, 'FontSize', 14);

a = {'-','--','-.',':', '-', ':'};
h1=errorbar(p_arr, first_coord_covered_mean_diffp(1,:), neg1(1,:),pos1(1,:), a{1},  'LineWidth', 2, 'Color', color1); % Method 1
h2=errorbar(p_arr, first_coord_covered_mean_diffp(2,:), neg1(2,:),pos1(2,:), a{2}, 'LineWidth', 2, 'Color', color2); % Method 2
h3=errorbar(p_arr, first_coord_covered_mean_diffp(3,:), neg1(3,:),pos1(3,:), a{3}, 'LineWidth', 2, 'Color', color3); % Method 2
h4=errorbar(p_arr, first_coord_covered_mean_diffp(4,:), neg1(4,:),pos1(4,:), a{4}, 'LineWidth', 2, 'Color', color4); % Method 2
h5=errorbar(p_arr, first_coord_covered_mean_diffp(5,:), neg1(5,:),pos1(5,:), a{5}, 'LineWidth', 2, 'Color', color5); % Method 2
h6=errorbar(p_arr, first_coord_covered_mean_diffp(6,:), neg1(6,:),pos1(6,:), a{6}, 'LineWidth', 2, 'Color', color6); % Method 2

yline(0.05,'--','LineWidth',4,'Color','black');
xlabel('dimension p','FontSize',14);
ylabel('coverage ratio','FontSize',14);
legend('White', 'MW', 'Hadamard','Hadamard-t','Pair bootstrap', 'Residual bootstrap', 'TextColor', 'black', 'Color', 'none','Location', 'northwest','FontSize',14); % for transparent background

savefigs=1;    closefigs=1;
if savefigs==1
        filename = ...
            sprintf( './figs/compare6-cover-first-n=%d-n-mc=%d-design=%d-plot.png',...
            n,n_mc,design);
        saveas(gcf, filename,'png');
        fprintf(['Saved Results to ' filename '\n']);
        if closefigs==1
            close(gcf)
        end
end


figure, hold on
%plot(p_arr, first_coord_covered_mean_diffp(1,:),'o-', 'LineWidth', 2,'Color', color1);
plot(p_arr, first_coord_covered_mean_diffp(2,:),'s--', 'LineWidth', 2,'Color', color2);
plot(p_arr, first_coord_covered_mean_diffp(3,:),'+-.', 'LineWidth', 2,'Color', color3);
plot(p_arr, first_coord_covered_mean_diffp(4,:),'*:', 'LineWidth', 2,'Color', color4);
plot(p_arr, first_coord_covered_mean_diffp(5,:),'o-', 'LineWidth', 2,'Color', color5);

set(gca, 'FontSize', 12);

ylim([0.02, max(first_coord_covered_mean_diffp(2,:))+0.03]);
ylim([0, max(first_coord_covered_mean_diffp(2,:))+0.03]);

a = {'-','--','-.',':', '-'};
%h1=errorbar(p_arr, first_coord_covered_mean_diffp(1,:), neg1(1,:),pos1(1,:), a{1},  'LineWidth', 2, 'Color', color1); % Method 1
h2=errorbar(p_arr, first_coord_covered_mean_diffp(2,:), neg1(2,:),pos1(2,:), a{2}, 'LineWidth', 2, 'Color', color2); % Method 2
h3=errorbar(p_arr, first_coord_covered_mean_diffp(3,:), neg1(3,:),pos1(3,:), a{3}, 'LineWidth', 2, 'Color', color3); % Method 2
h4=errorbar(p_arr, first_coord_covered_mean_diffp(4,:), neg1(4,:),pos1(4,:), a{4}, 'LineWidth', 2, 'Color', color4); % Method 2
h5=errorbar(p_arr, first_coord_covered_mean_diffp(5,:), neg1(5,:),pos1(5,:), a{5}, 'LineWidth', 2, 'Color', color5); % Method 2

yline(0.05,'--','LineWidth',4,'Color','black');
xlabel('Dimension p','FontSize',12);
ylabel('Type I error in the first coordinate','FontSize',12);
%legend('White', 'MW', 'Hadamard','Hadamard-t','Pair boostrap', 'TextColor', 'black', 'Color', 'none','Location', 'northwest','FontSize',14); % for transparent background
legend('MW', 'Hadamard','Hadamard-t','Pair boostrap','TextColor', 'black', 'Color', 'none','Location', 'northwest','FontSize',12); % for transparent background


if savefigs==1
        filename = ...
            sprintf( './figs/compare4-cover-first-n=%d-n-mc=%d-design=%d-plot.png',...
            n,n_mc,design);
        saveas(gcf, filename,'png');
        fprintf(['Saved Results to ' filename '\n']);
        if closefigs==1
            close(gcf)
        end
end

hold off


% plot the results for the second coordinate
figure, hold on
%plot(p_arr, second_coord_covered_mean_diffp(1,:),'o-', 'LineWidth', 2,'Color', color1);
plot(p_arr, second_coord_covered_mean_diffp(2,:),'s--', 'LineWidth', 2,'Color', color2);
plot(p_arr, second_coord_covered_mean_diffp(3,:),'+-.', 'LineWidth', 2,'Color', color3);
plot(p_arr, second_coord_covered_mean_diffp(4,:),'*:', 'LineWidth', 2,'Color', color4);
plot(p_arr, second_coord_covered_mean_diffp(5,:),'o-', 'LineWidth', 2,'Color', color5);

set(gca, 'FontSize', 12);

ylim([0.02, max(second_coord_covered_mean_diffp(2,:))+0.03]);
ylim([0, max(second_coord_covered_mean_diffp(2,:))+0.03]);
a = {'-','--','-.',':','-'};
%h1=errorbar(p_arr, second_coord_covered_mean_diffp(1,:), neg2(1,:),pos2(1,:), a{1},  'LineWidth', 2, 'Color', color1); % Method 1
h2=errorbar(p_arr, second_coord_covered_mean_diffp(2,:), neg2(2,:),pos2(2,:), a{2}, 'LineWidth', 2, 'Color', color2); % Method 2
h3=errorbar(p_arr, second_coord_covered_mean_diffp(3,:), neg2(3,:),pos2(3,:), a{3}, 'LineWidth', 2, 'Color', color3); % Method 2
h4=errorbar(p_arr, second_coord_covered_mean_diffp(4,:), neg2(4,:),pos2(4,:), a{4}, 'LineWidth', 2, 'Color', color4); % Method 2
h5=errorbar(p_arr, second_coord_covered_mean_diffp(5,:), neg2(5,:),pos2(5,:), a{5}, 'LineWidth', 2, 'Color', color5); % Method 2

yline(0.05,'--','LineWidth',4,'Color','black');
xlabel('Dimension p','FontSize',12);
ylabel('Type I error in the second coordinate','FontSize',12);
%legend('White', 'MW', 'Hadamard','Hadamard-t','TextColor', 'black', 'Color', 'none','Location', 'northwest','FontSize',14); % for transparent background
legend('MW', 'Hadamard','Hadamard-t','Pair bootstrap','TextColor', 'black', 'Color', 'none','Location', 'northwest','FontSize',12); % for transparent background


if savefigs==1
        filename = ...
            sprintf( './figs/compare4-cover-second-n=%d-n-mc=%d-design=%d-plot.png',...
            n,n_mc,design);
        saveas(gcf, filename,'png');
        fprintf(['Saved Results to ' filename '\n']);
        if closefigs==1
            close(gcf)
        end
end

hold off

    
   