% This script corresponds to the second out of five script sequences that
% was performed for the research.
%
% This script is used to analyze and plot the grid search data aquired
% using the first script ('Step1_GridSearch'). In this script, the accuracy
% is plotted for each point within the considered search space for each
% considered NN design (FFN with L = 2, L = 4, and L = 8). In order to
% identify the best parameter combination, the maxima is visually
% identified manually.

clc; clear all;

L = [2,4,8]; % number of hidden layers considered.
for l = L % plotting the grid of an NN model one at a time
    
    % loading data for the grid search performed (at data/grid search/ 
    % folder with the file format "perf_data_*layers.mat"
    load("data/grid search/perf_data_"+ l +"layers.mat");

    % This variable may be changed into any integer values within the bounds
    % [1,6], which corresponds to the six performance metrices: accuracy, auc,
    % ppv, npv, sr, and rr respectively.
    k = 1; % plot the grid search using the accuracy metric.

    a = mean(perf_data,1); % obtaining the mean metric for for each point within the search space

    % conversion of the 4D data to a 2D data matrix
    b = zeros(size(perf_data,3),size(perf_data,4));
    for i = 1:size(b,1)
       for j = 1:size(b,2)
           b(i,j) = a(k,2,i,j);
       end
    end

    % plotting the grid using a surface plot
    figure();
    surf(1:size(b,2), 1:size(b,1), b);

    % learning rate selections
    LR(1) = 1;
    for i = 2:10
        if (mod(i,2) == 0)
            LR(i) = LR(i-1)/2;
        else
            LR(i) = LR(i-1)/5;
        end
    end % 10 selections

    %  layer size selection
    NL = [10,15,20,25,30,40,50,60,80,100,120,140,160,180,200,250,300,350,400,462]; % 20 selections.

    % setting graph details accordingly...
    title("FNN" + l + " Optimization Plot",'FontSize',30,'Interpreter','latex')
    xl = xlabel('Learning rate','Interpreter','latex');
    yl = ylabel('Number of neurons per layer','Interpreter','latex');
    zl = zlabel('Validation accuracy','Interpreter','latex');
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',18)
    xl.FontSize = 36;
    yl.FontSize = 36;
    zl.FontSize = 36;
    xticks([1:1:10]);
    xticklabels(LR([1:1:10]));
    yticks([1:2:20]);
    yticklabels(NL([1:2:20]));
end

% NOTE: Using the implemented grid search, the following indexes for the
% variables: learning rate (LR) and number of neurons per layer (NL) was
% identified respectively:
%
% FNN2: LR = index 5 (LR = 0.01); NL = index 18 (NL = 350);
% FNN4: LR = index 5 (LR = 0.01); NL = index 19 (NL = 400);
% FNN8: LR = index 5 (LR = 0.01); NL = index 17 (NL = 300);