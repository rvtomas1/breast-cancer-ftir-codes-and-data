% This script is used to train an LDA classifier via a ten-fold cross
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
    for k = 1:10
         
    % Training
    for i = 1:size(input_t{k},2)
        num(i,:) = input_t{k}{i}{1};

        [a,b] = max(input_t{k}{i}{2});
        label(i) = b;
    end
    
    % the LDA algorithm from this script was based from the LDA implemented
    % on the paper of Santillan et. al. (DOI: 10.1007/s00216-021-03183-0)
    
    % PCA part
    m = mean(num); % computation of training data mean
    a = num - m;
    S = a'*a; % computation of covarriance matrix
    [V_PC,D] = eig(S); % computation of eigenvalues
    diag_D = diag(D);
    diag_S = diag(S);
    n = 1;
    p_vector = (size(V_PC,2)-n):size(V_PC,2); 
    vectors = (a)*V_PC(:,p_vector); % projection of spectral data to the 2 most dominant eigenvectors (principal axes vectors)

    ben_label = find(label == 1);
    mal_label = find(label == 2);
    
    % computing for the std and mean among and between classes
     class_X = {vectors(mal_label,:),vectors(ben_label,:)};
     class_mean = {};
     class_std = {};

     interclass_std = {};
     for i = 1:2
        class_mean{i} = mean(class_X{i});
        class_std{i} = cov(class_X{i});    
     end

     Sw = class_std{1} + class_std{2};
     Mu = (class_mean{1} + class_mean{2})/2;

    for i = 1:2
        interclass_std{i} = size(class_X{i},1)*(class_mean{i}'-Mu)*(class_mean{i}' - Mu)';
    end

     SB = interclass_std{1} + interclass_std{2};
    
     invSw = inv(Sw);
     invSw_by_SB = invSw*SB;
    
     % determining the best projection vector across the PCA axis
     [V,D] = eig(invSw_by_SB);

     W1 = V(:,1)';
     W2 = V(:,2)';
    
    % determining parameters of the gaussian probability density functions
    y1_w2 = sort((class_X{1})*W2');
    y2_w2 = sort((class_X{2})*W2');
    mean1 = mean(y1_w2);
    std1 = std(y1_w2);
    mean2 = mean(y2_w2);
    std2 = std(y2_w2);

    % Validation/Testing
    clear num;
    clear label;
    for i = 1:size(input_ts{k},2)
        num(i,:) = input_ts{k}{i}{1};

        [a,b] = max(input_ts{k}{i}{2});
        label(i) = b;
    end

    TP = 0;
    TN = 0;
    FP = 0;
    FN = 0;
    prob_class1 = [];

    m = mean(num);
    a = num - m;
    vectors = (a)*V_PC(:,p_vector);
    ben_label = find(label == 1);
    mal_label = find(label == 2);
    class_X = {vectors(mal_label,:),vectors(ben_label,:)};

    for i = 1:2
        p = (class_X{i}*W2')';
        m = normpdf(p,mean1,std1);
        b = normpdf(p,mean2,std2);
        prob = softmax([m;b]);

        for j = 1:size(p,2)
            if(i == 1)
               if(prob(1,j)>prob(2,j))
                   TP = TP + 1;
               else
                   FP = FP + 1;
               end
            elseif(i == 2)
               if(prob(2,j)>prob(1,j))
                   TN = TN + 1;
               else
                   FN = FN + 1;
               end
            end
        end
        prob_class1 = [prob_class1, prob(1,:)];
    end

    [X,Y,T,AUC] = perfcurve(label,prob_class1,1);

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
LDA_mean_perf = {'AUC','accuracy','PPV','NPV','SR','RR';mean_perf(1),mean_perf(2),mean_perf(3),mean_perf(4),mean_perf(5),mean_perf(6)}
LDA_std_perf = {'AUC','accuracy','PPV','NPV','SR','RR';std_perf(1),std_perf(2),std_perf(3),std_perf(4),std_perf(5),std_perf(6)}

