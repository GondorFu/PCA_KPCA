clc; clear; close all


load ('ex7faces.mat')
X = X(1:100, :)';
Y = kernelpca_tutorial(X,100);