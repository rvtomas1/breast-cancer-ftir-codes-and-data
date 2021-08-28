% This script is used to train a Naive Baye's classifier via a ten-fold cross
% validation proceedure. The model is trained and validated for a specified
% number of trials for stability.

clear; clc;
FOLD = 10; % number of folds considered
[input_t, ~, input_ts] = createDataSet(FOLD); % creation of training and test set

model_trial = 50; % number of model training-testing trials
trial_metic = zeros(model_trial,6,2); % initializing storage variable for recording n-fold model performance

for l = 1:model_trial % iterations per model trial
    tic
    n_fold_metric = zeros(FOLD,6); % initializing storage variable for recording performance within the n-fold validation proceedure
    for k = 1:FOLD

    % Training
    for i = 1:size(input_t{k},2)
        num(i,:) = input_t{k}{i}{1};

        [a,b] = max(input_t{k}{i}{2});
        label(i) = b;
    end
    model = fitcnb(num,label); 

    % Validation/Testing
    clear num;
    clear label;
    for i = 1:size(input_ts{k},2)
        num(i,:) = input_ts{k}{i}{1};

        [a,b] = max(input_ts{k}{i}{2});
        label(i) = b;
    end

    [labels,score] = predict(model,num);
    
    % Computation of performance metrics
    TP = 0;
    TN = 0;
    FP = 0;
    FN = 0;

        for j = 1:size(label,2)
            if(label(j) == 2)
               if(score(j,1)<score(j,2))
                   TP = TP + 1;
               else
                   FP = FP + 1;
               end
            elseif(label(j) == 1)
               if(score(j,2)<score(j,1))
                   TN = TN + 1;
               else
                   FN = FN + 1;
               end
            end
        end

        [X,Y,T,AUC] = perfcurve(label,score(:,1),1);

        Accuracy = (TP+TN)/(TP+TN+FP+FN);
        AUC = AUC;
        PPV = TP/(TP + FP);
        NPV = TN/(TN + FN);
        SR = TN/(TN + FP);
        RR = TP/(TP + FN);

        n_fold_metric(k,:) = [AUC,Accuracy,PPV,NPV,SR,RR];

    end
    % computing for the average n-fold validation performance
    n_fold_metric(isnan(n_fold_metric)) = 0;
    mean_perf = mean(n_fold_metric);
    std_perf = std(n_fold_metric);
    
    % storing the average n-fold validation performance per model trial
    trial_metic(l,:,1) = mean_perf;
    trial_metric(l,:,2) = std_perf;
    
    t = toc;
    
    % display for user update purposes.
    display("Successfully trained and validated model in " + t + " seconds for model trial " + l);
end

% computing for the average mean and std for all trials considered
avg_trial_metric = mean(trial_metric,1);

% displaying performance metrics
NB_mean_perf = {'AUC','accuracy','PPV','NPV','SR','RR';mean_perf(1),mean_perf(2),mean_perf(3),mean_perf(4),mean_perf(5),mean_perf(6)}
NB_std_perf = {'AUC','accuracy','PPV','NPV','SR','RR';std_perf(1),std_perf(2),std_perf(3),std_perf(4),std_perf(5),std_perf(6)}

