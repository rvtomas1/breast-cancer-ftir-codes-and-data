% This script corresponds to the fifth out of five script sequences that
% was performed for the research.
%
% This script is used to analyze and plot the perturbation response of each 
% FNN type trained for sensitivity analysis.

clc; clear;
% loading matlab file for labeling purposes(x - axis).
load("data/dataset/wavenumbers.mat") % a vector containing the wavenumber values considered in the spectrum.

load('data/sensitivity analysis/MSE2.mat');
MSE = MSE - MSE(11,:,:,:); % deducting the base perturbation response (@ 0% perturbation; which is indexed at 11)
MSE = abs(MSE);
a = mean(MSE,3); % obtaining the mean perturbation of all FNN models considered
a = mean(a,4); % obtaining the mean perturbation of all the spectral data considered.
mixed2 = mean(a,1); % obtaining the mean perturbation for each instance of perturbation considered
mixed2 = mixed2*100/sum(mixed2); % normalizing values so that the total area under the curve is ~100%

% NOTE: one can directly access the perturbation response of the networks
% through loading the files under 'data/sensitivity analysis/fnn2_perturbation_response.mat'

% implementation for FNN4
load('data/sensitivity analysis/MSE4.mat');
MSE = MSE - MSE(11,:,:,:);
MSE = abs(MSE);
a = mean(MSE,3);
a = mean(a,4);
mixed4 = mean(a,1);
mixed4 = mixed4*100/sum(mixed4);

% NOTE: one can directly access the perturbation response of the networks
% through loading the files under 'data/sensitivity analysis/fnn4_perturbation_response.mat'

% implementation for FNN8
load('data/sensitivity analysis/MSE8.mat');
MSE = MSE - MSE(11,:,:,:);
MSE = abs(MSE);
a = mean(MSE,3);
a = mean(a,4);
mixed8 = mean(a,1);
mixed8 = mixed8*100/sum(mixed8);

% NOTE: one can directly access the perturbation response of the networks
% through loading the files under 'data/sensitivity analysis/fnn8_perturbation_response.mat'

figure();
hold on;

% setting graph details accordingly...
xlim([min(w_lab) max(w_lab)]);
ylim([1.05*min(mixed8) 1.05*max(mixed8)]);
xl = xlabel('Wavenumber ($cm^{-1}$)','Interpreter','latex');
yl = ylabel('Percent contribution (\%)','Interpreter','latex');
xl.FontSize = 36;
yl.FontSize = 36;

% plotting the perturbation response lines for each FNN
p1 = plot(w_lab(1:end-1),flip(mixed2),'LineWidth',3);
p2 = plot(w_lab(1:end-1),flip(mixed4),'LineWidth',3);
p3 = plot(w_lab(1:end-1),flip(mixed8),'LineWidth',3);

[~, hobj, ~, ~] = legend([p1, p2, p3],{'$FNN2$','$FNN4$','$FNN8$'},'Location','northeast','FontSize',25,'interpreter', 'latex');
hl = findobj(hobj,'type','line');
set(hl,'LineWidth',3);

ax = gca;
ax.XDir = 'reverse';
