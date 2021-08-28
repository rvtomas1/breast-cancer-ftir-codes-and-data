function [Perf] = testNN(data_set, nn)
% This function obtains the performance metrices (accuracy, area under the
% ROC curve, positive predictive value, negative predictive value,
% specificity rate, and recall rate) of a neural network "nn" given a data 
% set "data_set".
%
% INPUTS:
% data_set - data set which may be that of the training set, validation set
%           or the test set. This parameter is a {1, FOLD} dimensional
%           cell, where FOLD is the number of folds considered for the
%           n-fold cross validation set. Each cell under a fold is a {1,N}
%           dimensional cell which contains the data set, where N is the
%           number of spectral vectors with their corresponding true
%           labels.
% nn -  a neural network to be evaluated.
%
% OUTPUTS:
% Perf - performance metric of a neural network. This is a 6x2-element vector
%        which arranges the metrics: accuracy, area under the
%        ROC curve, positive predictive value, negative predictive value,
%        specificity rate, and recall rate respectively from element 1 to
%        6. The first row denotes the mean of the perf. metrices while the
%        second row denotes the standard deviations

%% loading of vlidation data
FOLD = size(data_set,2);
loss = 0;

% initializing data storage variables for variables needed to determine performance metrices
ac = zeros(1,FOLD); % accuracy
auc = zeros(1,FOLD); % auc
TP = zeros(1,FOLD); % true positive
FP = zeros(1,FOLD); % false positive
TN = zeros(1,FOLD); % true negative
FN = zeros(1,FOLD); % false negative

for j = 1:FOLD
    miss_count = 0; % number of miss counts/ falsely predicted data per fold
    I_v = size(data_set{j},2); % total number of spectral vectors per fold
    O = zeros(I_v,4); % denotes the probability of each prediction
    for i = 1:I_v

        [~, output ] = evalNN_v( nn{j}, data_set{j}{i}); % evaluation of a single spectral vector 
        [~, loss] = computeLoss( output, data_set{j}{i}{2}, loss);
        [~,out_n] = max(output); % determines the diagnosis as the probability having the greater value (p> 0.5)
        [~,in_n] = max(data_set{j}{i}{2});
        O(i,:) = [output, out_n, in_n];
        
        % determining number of miss counts
        if(out_n - in_n ~= 0 )
            miss_count = miss_count + 1;
        end
        
        % determining true positives, false positives, true negatives, and
        % false negatives.
        if(in_n == 2)
            if(out_n == 2)
                TP(j) = TP(j) + 1;
            else
                FN(j) = FN(j) + 1;
            end
        else
            if(out_n == 1)
                TN(j) = TN(j) + 1;
            else
                FP(j) = FP(j) + 1;
            end
        end
        
    end
    
    [X,Y,T,AUC] = perfcurve(O(:,4),O(:,2),2); % determining AUC
    ac(j) = (size(data_set{j},2)-miss_count)/size(data_set{j},2); % determining accuracy
    auc(j) = AUC;
end

% Determining the mean performance metrices.
Accuracy = mean(ac);
AUC = mean(auc);
PPV = mean(TP./(TP + FP));
NPV = mean(TN./(TN + FN));
SR = mean(TN./(TN + FP));
RR = mean(TP./(TP + FN));

Perf(1,:) = [Accuracy, AUC, PPV, NPV, SR, RR];

% Determining the standard deviation performance metrices.
Accuracy = std(ac);
AUC = std(auc);
PPV = std(TP./(TP + FP));
NPV = std(TN./(TN + FN));
SR = std(TN./(TN + FP));
RR = std(TP./(TP + FN));
 
Perf(2,:) =  [Accuracy, AUC, PPV, NPV, SR, RR];


end

