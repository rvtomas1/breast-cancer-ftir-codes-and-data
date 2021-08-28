% This script corresponds to the third out of five script sequences that
% was performed for the research.
%
% This script is used to train and evaluate (using the test set) optimized
% neural networks (as a result from the grid search). Here, three FNNs were
% trained using the learning rate and number of neurons per layer 
% hyperparameter combinations determined using the grid search in script 2 
% ('Step2_GridPointSelector'). The FNNs were trained using a 10-fold cross
% validation proceedure for 20 trials for stability, over 1000 epochs.

clc; clear;
addpath('functions/fnn/');

FOLD = 10; % n value for the n-fold crossvalidation
trials = 10; % number of trials

%% learning rate selections
LR(1) = 1;
for i = 2:10
    if (mod(i,2) == 0)
        LR(i) = LR(i-1)/2;
    else
        LR(i) = LR(i-1)/5;
    end
end % 10 selections

%%  layer size selection
NL = [10,15,20,25,30,40,50,60,80,100,120,140,160,180,200,250,300,350,400,462]; % 20 selections.

%% epoch length selection
 E = 1000; 

%% epsilon selection
eps = LR;

% NOTE: Using the implemented grid search, the following indexes for the
% variables: learning rate (LR) and number of neurons per layer (NL) was
% identified respectively:
%
% FNN2: LR = index 5 (LR = 0.01); NL = index 18 (NL = 350);
% FNN4: LR = index 5 (LR = 0.01); NL = index 19 (NL = 400);
% FNN8: LR = index 5 (LR = 0.01); NL = index 17 (NL = 300);


    for i = 1:trials % iteration for each trial
        [input_t, input_v, input_ts] = createDataSet(FOLD); % creation of data set
        
        [nn_best, perf] = trainNN3(2, E, NL(18), LR(5), eps(1),input_t, input_ts, 20, FOLD); % training of and evaluation of neural network
        save("data/optimized model parameters/fnn2_model_"+i,'nn_best'); % saving the parameters of the neural network.
        save(strcat('data/optimized model performance/perf_nn2_t',num2str(i)),'perf'); % saving for the performance metric of the neural network.
        [nn_best, perf] = trainNN3(4, E, NL(19), LR(5), eps(1),input_t, input_ts, 20, FOLD);
        save("data/optimized model parameters/fnn4_model_"+i,'nn_best');
        save(strcat('data/optimized model performance/perf_nn4_t',num2str(i)),'perf');
        [nn_best, perf] = trainNN3(8, E, NL(17), LR(5), eps(1),input_t, input_ts, 20, FOLD);
        save("data/optimized model parameters/fnn8_model_"+i,'nn_best');
        save(strcat('data/optimized model performance/perf_nn8_t',num2str(i)),'perf');
    end
