function dof = dof_hadamard(A,T,R,n)
%Degrees of freedom calculation
%for Hadamard estimator

% A = (X'*X)^(-1); %p^3
% S = A*(X');
% T = (S.*S);
% Q = eye(n) - X*A*(X'); %I-P, cost: np(n+p)
% Lev = diag(Q);
%U = Q.*Q;
%U = (U+U)/2;
%R = U^(-1); 

d1 = diag(A);
d1 = d1.^2;
B = T*ones(n,1);
d2 = B.*B;
C = T*R*(T');
d3 = diag(C); 

denom = d2+2*d3-d1; 

dof = 2*d1./denom;