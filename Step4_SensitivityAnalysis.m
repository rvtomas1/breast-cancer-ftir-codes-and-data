% This script corresponds to the fourth out of five script sequences that
% was performed for the research.
%
% This script is used to carry out the sensitivity analysis. The
% sensitivity analysis utilizes the pre-trained NN models from script 3
% ('Step3_TrainingOptimizedNN'). The sensitivity analysis carries out a
% committee-based NN perturbation analysis which utilizes the whole data
% set to carry out the analysis.

clc; clear all;
addpath('functions/fnn/');

% loading Excel file containing the whole data set.
load("data/dataset/ftir_spectra.mat");
load("data/dataset/ftir_labels.mat");


NN_TYPE = 2; % for FNN 2; edit this***

% initialization of some constant variables
N = size(S,1); % total number of spectral vectors considered.
nn_trials = 20; % total number of nn considered.

% formatting all spectral data variables (for validation function)
C = max(S_label); % total number of classes considered (here is 2 classes - benign  and malignant)
S = S(:,2:end);
for i = 1:N
    val_num = S(i,:);
    val_label = zeros(1,C);
    val_label(S_label(i)) = 1;

    B_DATA_ALL{i}{1} = val_num; 
    B_DATA_ALL{i}{2} = val_label; 
end

% loading pre-trained optimized neural networks
for i = 1:nn_trials % iteration for loading each trained NN
    load("data/optimized model parameters/fnn" + NN_TYPE + "_model_"+ i +".mat");
    for j = 1:size(nn_best,2) % iteration for loading each nn trained per fold
       nn{(i-1)*10 + j} = nn_best{j};
    end
end

% total number of spectral data variables  considered; 462 wavenumbers/variables for each spectral data
L = size(S,1); 

% base perturbation (0% added perturbation)
S_med = median(S);

for j = 1:L % iteration for all spectral vector variables/wavenumbers
    count = 1;
    tic % for recording time spent for simulation
    for k = -0.5:0.05:0.5 % iteration for each perturbation (from -50% to 50% at 5% intervals)
        for i = 1:N % iteration for all spectral vector within the data set
            for m = 1:size(nn,2) % iteration for all nn models trained considered
                input = B_DATA_ALL{i};
                input{1}(1,j) = input{1}(1,j) + k*S_med(1,j); % adding the perturbation
                [~,o] = evalNN_v(nn{m},input); % evaluating neural network performance
                MSE(count,j,i,m) = real(0.5*sqrt(sum(o.^2-input{2}.^2,2))); % computation of mean squared error
            end
        end
    count = count + 1;
    end
    t = toc;
    
    % saving data per wavenumber iteration
    save("data/sensitivity analysis/MSE" + NN_TYPE,'MSE');
    
    % display for monitoring simulation progress
    display("finished performing perturbation response for the " + j + "th wavenumber. It took " + t + " seconds to perform the iteration.");
end

