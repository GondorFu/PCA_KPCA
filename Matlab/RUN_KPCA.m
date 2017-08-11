
%% Initialization
clear ; close all; clc

load ('SyntheticData.mat')
K = 2;
%% original data
figure;
hold on;
plot3(data(1:2:end,1),data(1:2:end,2),data(1:2:end,3),'b*');
plot3(data(2:2:end,1),data(2:2:end,2),data(2:2:end,3),'ro');
legend('class 1','class 2');
axis equal;
xlabel('x');
ylabel('y');
zlabel('z');
axis([-110 110 -110 110 -110 110]);
title('original data');
drawnow;
hold off;

% ================ PCA ==========================
[X_norm, mu, sigma] = featureNormalize(data);
[U, S] = pca(X_norm);
Y1 = projectData(X_norm, U, K);

%% standard PCA
figure;hold on;
plot(Y1(1:2:end,1),Y1(1:2:end,2),'b*');
plot(Y1(2:2:end,1),Y1(2:2:end,2),'ro');
legend('class 1','class 2');
title('standard PCA');
drawnow;

% ================ KPCA ==========================
para = 5;
[U, S, Km] = kpca(data, 'poly', para);
Y2 = projectData(Km, U, K);

%% Polynomial KPCA
figure;
hold on;
plot(Y2(1:2:end,1),Y2(1:2:end,2),'b*');
plot(Y2(2:2:end,1),Y2(2:2:end,2),'ro');
legend('class 1','class 2');
title('Polynomial kernel PCA');
drawnow;
hold off


DIST = distanceMatrix(data);
DIST(DIST==0) = inf;
DIST = min(DIST);
para = 5 * mean(DIST);
[U, S, Km] = kpca(data, 'gaussian', para);
Y3 = projectData(Km, U, K);

%% Gaussian KPCA
figure;
hold on;
plot(Y3(1:2:end,1),Y3(1:2:end,2),'b*');
plot(Y3(2:2:end,1),Y3(2:2:end,2),'ro');
legend('class 1','class 2');
title('Gaussian kernel PCA');
drawnow;
hold off


