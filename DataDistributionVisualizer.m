% This script is used to visualized the distribution of malignant and
% benign spectral data through a line plot and a pca bi-plot (two of which
% were used for the research).

clear; clc;

%% Plotting the median FTIR spectrum of samples

load("data/median absorbance data/malignant_absorbance.mat"); % loading the median FTIR absorbance of malignant spectral data.
load("data/median absorbance data/benign_absorbance.mat"); % loading the median FTIR absorbance of benign spectral data.
load("data/median absorbance data/wavenumbers.mat"); % loading the wavenumbers considered for axis labeling purposes.

figure();
hold on;
m = plot(w_lab, b_num,'r','LineWidth',3); % ploting malignant spectrum
b = plot(w_lab, m_num,'b','LineWidth',3); % plotting benign spectrum
legend([m,b],'$Malignant$','$Benign$','Interpreter','latex','FontSize',24);

% setting graph details accordingly...
xlim([min(w_lab) max(w_lab)]);
xl = xlabel("$Wavenumber$ $(cm^{-1})$",'Interpreter','latex');
yl = ylabel("$Absorbance$ $(\%)$",'Interpreter','latex');
xl.FontSize = 36;
yl.FontSize = 36;
ax = gca;
ax.XDir = 'reverse';

% NOTE: Through this plot, the peaks were identified through visual
% inspection.

%% Plotting the sample distribution via a PCA biplot

% NOTE: All loaded parameters were obtained by using MATLAB's built-in
% "pca" function ([COEFF, SCORE, LATENT] = pca(X)) where each column in the
% COEFF variable is a vector corresponding to the direction of each
% wavenumber from the first to the last principal axis; SCORE is the
% projection of the spectral data towards the principal axes, and LATENT is
% the eigenvalue of each principal component vector which is used for
% purpose of determining the %variation accounted by each PC axis.

load("data/pca data/pca_points.mat"); % loading the coordinates of spectral data accross the two most dominant principal components.
load("data/pca data/pca_points_label.mat"); % loading the corresponding numerical label of spectral data
load("data/pca data/wavenumber_vectors.mat"); % loading the vector projection of each wavenumber accross the two most dominant principal components.
load("data/pca data/variation_profile.mat"); % loading the eigenvalues associated to each PC axis/component

ax1 = figure();

peak_variables = {'P1632/P1636 - Amide I protien','P1539/P1540 - Amide II protein','P1452/P1452 - Lipids',...
    'P1399/P1401 - Lipids','P1337/P1337','P1279/P1279','P1236/P1236 - DNA, RNA, phospholipids',...
    'P1160/P1160 - Carbohydrates','P1032/P1030 - Glycogen','P880/P878 - Phosphorylated protein'}; % identified peaks via visual analysis
hbi = biplot(coefs([83,128,171,196,226,255,275,312,375,448],1:2),'Scores',scores(:,1:2),'VarLabels',peak_variables,'markersize',32); % plotting the pca biplot.
n_sep = find(f_lab(:,1) == 2,1);
set(hbi((3*size(peak_variables,2)+1):(n_sep + 3*size(peak_variables,2))),'Color','b');
set(hbi((n_sep + 3*size(peak_variables,2)):end-1),'Color','r');
set(hbi([1:(3*size(peak_variables,2)+1)]),'Color',[0.5,0.5,0.5]);

% setting graph details accordingly...
xl = xlabel("$F_1$ $component$ $("+ round(latent(1)*100/sum(latent),2) +"\%)$",'Interpreter','latex');
yl = ylabel("$F_2$ $component$ $(" + round(latent(2)*100/sum(latent),2) + "\%)$",'Interpreter','latex');
xl.FontSize = 36;
yl.FontSize = 36;
