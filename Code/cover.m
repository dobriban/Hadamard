function  [prop_covered,  first_coord_covered,standardized_resid] = cover(V,b_ols,beta,dist,dof)
%evaluate coverage of the confidence intervals with variances given by V

p = size(b_ols,1);

if ~exist('beta','var')
    beta = zeros(p,1);
end

if ~exist('dist','var')
    dist = 'normal';
end

if ~exist('dof','var')
    dof = 0;
end

std_err = V.^(1/2);

alpha = 0.05;

switch dist
    case 'normal'
    crit = 1.96;
    %norminv(1- alpha/2)
    
    case 't'
    crit = tinv(1- alpha/2, dof); 
    %crit = 1.96;
end

u = b_ols+crit.*std_err; 
l = b_ols-crit.*std_err;

covered = min((beta<u),(beta>l));

prop_covered = mean(covered);
first_coord_covered = covered(1);

standardized_resid = (b_ols-beta)./std_err; 