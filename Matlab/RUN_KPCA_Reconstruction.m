
%% Initialization
clear ; close all; clc

%% =============== Part 4: Loading and Visualizing Face Data =============
%  We start the exercise by first loading and visualizing the dataset.
%  The following code will load the dataset into your environment
%
fprintf('\nLoading face dataset.\n\n');

%  Load Face dataset
load ('ex7faces.mat')

X = X(1:100,:);
m = size(X, 1);
K = 100; % reduction dim

%  Display the first 100 faces in the dataset
figure;
displayData(X(1:100, :));

%% =================== PCA ===============================
[X_norm, mu, sigma] = featureNormalize(X);
[U, S] = pca(X_norm);
Y = projectData(X_norm, U, K);
X_PCA_rec  = recoverData(Y, U, K);

%% =========== Part 5: KPCA on Face Data: Eigenfaces  ===================
%  Run PCA and visualize the eigenvectors which are in this case eigenfaces
%  We display the first 36 eigenfaces.
%
fprintf(['\nRunning KPCA on face dataset.\n' ...
         '(this might take a minute or two ...)\n\n']);

%  Run PCA
DIST = distanceMatrix(X);
DIST(DIST==0) = inf;
DIST = min(DIST);
para = 5 * mean(DIST);
[U, S, Km] = kpca(X, 'gaussian', para);

% %  Visualize the top 36 eigenvectors found
% figure
% displayData(U(:, 1:36)');


%% ============= Part 6: Dimension Reduction for Faces =================
%  Project images to the eigen space using the top k eigenvectors 
%  If you are applying a machine learning algorithm 
fprintf('\nDimension reduction for face dataset.\n\n');

Z = projectData(Km, U, K);

fprintf('The projected data Z has a size of: ')
fprintf('%d ', size(Z));

%% ==== Part 7: Visualization of Faces after KPCA Dimension Reduction ====
%  Project images to the eigen space using the top K eigen vectors and 
%  visualize only using those K dimensions
%  Compare to the original input, which is also displayed

fprintf('\nVisualizing the projected (reduced dimension) faces.\n\n');

X_KPCA_rec = zeros(size(X));
for i = 1:m
    z = Z(i,:)';
    X_KPCA_rec(i,:) = kPCA_PreImage(z, U, X, para)';
end

% Display PCA reconstructed data
figure
subplot(1, 2, 1);
displayData(X_PCA_rec(1:100,:));
title('PCA Recovered faces');
axis square;

% Display KPCA reconstructed data
subplot(1, 2, 2);
displayData(X_KPCA_rec(1:100,:));
title('KPCA Recovered faces');
axis square;

