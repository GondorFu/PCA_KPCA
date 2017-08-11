function [U, S, Km] = kpca(X, type, para)
%PCA Run principal component analysis on the dataset X
%   [U, S, Km] = kpca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S and
%   Normalize Kernel matrix
%

% Useful values
[m, n] = size(X);

% ====================== YOUR CODE HERE ======================
% Instructions: You should first compute the covariance matrix. Then, you
%               should use the "svd" function to compute the eigenvectors
%               and eigenvalues of the covariance matrix. 
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).
%
K = kernel(X, type, para);

% Normalize Kernel matrix
I = ones(m,m)/m;
Km = K - I*K - K*I + I*K*I;

[U, S] = svd(Km);

% Normalize eigenvector
U = bsxfun(@rdivide, U, sqrt(diag(S)'./m));




% =========================================================================

end
