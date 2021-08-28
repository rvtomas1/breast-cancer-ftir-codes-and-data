function [ nn_best, perf_t ] = trainNN3(Layer_num, E, NL, LR, eps, input_t, input_ts, E_intervals ,FOLD)

% This function trains and evaluates a feed forward neural network model
% using the training set and the test set respectively.
%
% INPUTS:
% Layer_num - number of hidden layers of created neural network.
% E - maximum number of epochs needed for the neural network to train.
% NL - number of neurons per hidden layer (same for all hidden layers).
% LR - learning rate.
% eps - adaGrad constant for numerical stability 
% input_t - spectral data set for the training set. This parameter is a 
%           {1, FOLD} dimensional cell, where FOLD is the number of folds 
%           considered for the n-fold cross validation set. Each cell under
%           a fold is a {1,N} dimensional cell which contains the data set, 
%           where N is the number of spectral vectors with their 
%           corresponding true labels.
% input_ts - spectral data set for the test set. This parameter is a 
%           {1, FOLD} dimensional cell, where FOLD is the number of folds 
%           considered for the n-fold cross validation set. Each cell under 
%           a fold is a {1,N} dimensional cell which contains the data set, 
%           where N is the number of spectral vectors with their 
%           corresponding true labels.
% E_intervals - number of epoch intervals for each display updates. This
%           parameter is only for troubleshooting and displaying purposes.
% FOLD - number of folds considered
%
% OUTPUTS:
% nn_best - a cell containing the nn parameters of a dimension {2,L}
%           where L is the number of FNN layers. Index {1} contains
%           the weights, while the index {2} contains the biases.
% perf - performance metric of a neural network. This is a 6x2-element vector
%        which arranges the metrics: accuracy, area under the
%        ROC curve, positive predictive value, negative predictive value,
%        specificity rate, and recall rate respectively from element 1 to
%        6. The first row denotes the mean of the perf. metrices while the
%        second row denotes the standard deviations

% creation of neural network models
fnn = genNN(Layer_num, size(input_t{1}{1}{1},2),size(input_t{1}{1}{2},2),NL,'SELU',1); 
for i = 1:FOLD
    nn{i} = fnn;
    nn_best{i} = {};
    perf_t = zeros(size(E_intervals,2),6,2);
end
t_loss = zeros(1,FOLD);

        % dropout rate internally initialized as 90%.
        DP = 0.90;

        for ii = 1:(E/E_intervals) % evaluations per display updates.
        tic % for recording time spent for simulation
        parfor m = 1:FOLD % simultaneous evaluation per fold
        
        I = size(input_t{m},2);
        loss = 0;
        train_data_counter = 0;
        for i = 1:E_intervals % evaluation per epoch
            
            % initialization of cache variables
            nn{m}{3} = 0;
            delta_acc_nn = genNN(size(nn{m}{1},2)-1,size(input_t{m}{1}{1},2),size(input_t{m}{1}{2},2),NL,'SELU',0);
            delta_nn = genNN(size(nn{m}{1},2)-1,size(input_t{m}{1}{1},2),size(input_t{m}{1}{2},2),NL,'SELU',0);
            
            for j = 1:I % evaluation per individual (stochastic gradient descent - SGD)
                %% neural dropout
                [dp_nn, dp_index] = dropN(nn{m},DP);

                 %% forward sweep.
                [ ~, nn{m}, h_out, fh_out ] = evalNN_tr( dp_nn, dp_index, nn{m}, input_t{m}{j});       

                %% backward sweep     
                [delta_nn] = computeBPP_gradients( dp_nn, h_out, fh_out, input_t{m}{j});

                 %% update of gradients and parameters
                 for k = 1:size(delta_acc_nn{1},2)
                     
                    % updte of gradients
                    delta_acc_nn{1}{k} = delta_acc_nn{1}{k} + delta_nn{1}{k}.^2;
                    delta_acc_nn{2}{k} = delta_acc_nn{2}{k} + delta_nn{2}{k}.^2;

                    % update nn parameters
                     nn{m}{1}{k} = nn{m}{1}{k} - LR*delta_nn{1}{k}./(sqrt(delta_acc_nn{1}{k}) + eps);
                     nn{m}{2}{k} = nn{m}{2}{k} - LR*delta_nn{2}{k}./(sqrt(delta_acc_nn{2}{k}) + eps);

                 end
            end
            
            % Premature ending training due to divergence.
            if(isnan(nn{m}{3}) == 1)
               break; 
            end

            if(nn{m}{3} == 0)
                break;
            end

            % Records the best-trained neural network parameter per epoch.
            % This conditional statement compares the performance of the
            % recently trained (after one epoch), to that of the previously
            % recorded best. If the recently trained is better than the
            % recently best, the trained nn is recorded to be the best.
            if(i == 1)
                nn_best{m} = nn{m};
            elseif(i > 1)
                if((nn_best{m}{3} > nn{m}{3}) & nn_best{m}{3} ~= 0)
                    nn_best{m} = nn{m};
                end
            end
            
            double([i nn{m}{3}]);
            t_loss(m) = nn_best{m}{3};
            end       
        end
        
        % obtaining for the performance metrics of the trained neural
        % network for the test set.
        perf = testNN(input_ts, nn_best);
        perf_t(ii,:,1) = perf(1,:); % mean perf. metrics
        perf_t(ii,:,2) = perf(2,:); % standard deviation perf. metrics
        
        % displaying training details
        t = toc; % time spent training for E_intervals epochs
        display("It took " + t + " seconds to train from epoch " + (ii-1)*E_intervals + " to " + (ii)*E_intervals); % epochs already spent training 
        perf(isnan(perf)) = 0;
        fprintf(strcat('AUC: ' , num2str(perf(1,1)) , '\nACC: ' , num2str(perf(1,2)) , '\nPPV: ' , num2str(perf(1,3)) , '\nNPV: ' , num2str(perf(1,4)), '\nSR: ',num2str(perf(1,5)),'\nRR: ',num2str(perf(1,6)),'\n')); % displaying test performance metrics.
        
    end

end

