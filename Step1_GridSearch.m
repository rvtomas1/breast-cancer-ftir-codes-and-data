% This script corresponds to the first out of five script sequences that
% was performed for the research.
%
% This script was used to implement the grid search. The grid search
% involved finding the sub-optimal learning rate and layer size for 3 feed
% forward neural networks (of varrying hidden layer sizes L = 2; L = 4; and
% L = 8). While the search was automated for finding the sub-optimal values
% for the learning rate and that of the layer size, the number of hidden
% layers was manually adjusted per instance of the grid search. The search
% space is a 200 (10 points for learning rate x 20 points for layer size) 
% point search space.

clc; clear;
addpath('functions/fnn/');

FOLD = 10; % number of folds for the n-fold cross validation proceedure
N = 10; % number of trials for evaluating each point within the grid.

%% learning rate selections
LR(1) = 1;
for k = 2:10
    if (mod(k,2) == 0)
        LR(k) = LR(k-1)/2;
    else
        LR(k) = LR(k-1)/5;
    end
end % 20 selections

%%  layer size selection
NL = [10,15,20,25,30,40,50,60,80,100,120,140,160,180,200,250,300,350,400,462]; % 20 selections.

%% epoch length selection
E = 1000;

%% epsilon selection
eps = LR;

%% number of layers considered
L = 2; % edit here***

    % Initializing storage variables for performance metrics.
    perf_t = zeros(N,6,length(NL),length(LR)); % storage for perf metric data 
    
     for i = 1:length(NL) % loop for number of neurons per layer sweep
         for j = 1:length(LR) % loop for learning rate sweep
            
            for k = 1:N % loop for number of trials sweep
                tic % for recording time spent for simulation
                
                % For each point in the grid, a 10-fold crossvalidation
                % proceedure was performed N times --- hence training and
                % evaluating a neural net N number of times! The whole
                % process is typically very time-consuming.
                
                [input_t, input_v, ~] = createDataSet(FOLD); % creating a data set.
                [nn,~,~] = trainNN2(L, E, NL(i), LR(j), eps(1), input_t, input_v, FOLD); % training of NN.
                perf = testNN(input_v, nn); % evaluating NN.
                
                % storage of  validation data.
                perf_t(k,:,i,j) = perf(1,:);
                
                t = toc;
                perf = perf(1,:);
                perf(isnan(perf)) = 0;
                
                % displaying test performance metrics.
                fprintf(strcat('Successfully evaluated tiral: ',num2str(k),' of grid search, for learning rate point: ',num2str(j),' ,and number of neurons per layer point: ',num2str(i),'. Simulations took: ', num2str(t),' seconds to complete.\n'));
                fprintf(strcat('AUC: ' , num2str(perf(1,1)) , '\nACC: ' , num2str(perf(1,2)) , '\nPPV: ' , num2str(perf(1,3)) , '\nNPV: ' , num2str(perf(1,4)), '\nSR: ',num2str(perf(1,5)),'\nRR: ',num2str(perf(1,6)),'\n\n')); 
        
            end
            
            % saving training and validation data
             perf_data = perf_t;
             save("data/grid search/perf_data_" + L + "layers",'perf_data');
             
         end
     end
