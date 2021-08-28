% This script is used to determine a sub-optimal value for the number of
% trees to be considered in training the random forest classifier. Here, a
% sweep from 1 to 100 trees was considered. The performance metric (AUC,
% accuracy, PPV, NPV, RR, and SR) was recorded for each sweep iteration. To
% determine the best hyperparameter (N - number of trees) value, the sweep's
% performance response plot was visually inspected. The N-value along the
% "knee" of the curve was determined as the sub-optimal value for the
% hyperparameter.

clear; clc;

% Plots the performance response plot if present
if isfile("treeBag_perf_sweep.mat")
    load("treeBag_perf_sweep.mat");
    figure();
    plot(perf(:,2),'LineWidth',3);
    xl = xlabel('$Number$ $of$ $trees(N)$','Interpreter','latex');
    yl = ylabel('$Accuracy$ $(\%)$','Interpreter','latex');
    xl.FontSize = 36;
    yl.FontSize = 36;
    
else % else performs the optimization sweep
    
    perf = zeros(100,6);
    for n = 1:100 % iteration for each considered number of trees
        trial_mean_perf = zeros(1,6);
        for m = 1:50 % iteration for each n-fold validation trial

            FOLD = 10;
            [input_t, input_v, input_ts] = createDataSet(FOLD); % creation of data set

            Results = zeros(10,6);

            for k = 1:FOLD % iteration for each fold

            % training
            num = zeros(size(input_t{k},2),size(input_t{k}{1}{1},2));
            label = zeros(size(input_t{k},2),1)';
            for i = 1:size(input_t{k},2)
                num(i,:) = input_t{k}{i}{1};

                [a,b] = max(input_t{k}{i}{2});
                label(i) = b;
            end

            model = TreeBagger(n,num,label); 


            % validation/testing
            num = zeros(size(input_v{k},2),size(input_v{k}{1}{1},2));
            label = zeros(size(input_v{k},2),1)';
            for i = 1:size(input_v{k},2)
                num(i,:) = input_v{k}{i}{1};

                [a,b] = max(input_v{k}{i}{2});
                label(i) = b;
            end

            [labels,score] = predict(model,num);

            % determination of performance metrics
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

                Results(k,:) = [AUC,Accuracy,PPV,NPV,SR,RR];

            end
            Results(isnan(Results)) = 0;
            mean_perf = mean(Results);
            std_perf = std(Results);

            trial_mean_perf(m,:) = mean_perf;
        end

        perf(n,:) = mean(trial_mean_perf,1);

        clc;
        display("Finished simulating model considering " + n + " number of trees");
    end
    save('treeBag_perf_sweep','perf');

end
