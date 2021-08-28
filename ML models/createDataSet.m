function [input_t, input_v, input_ts] = createDataSet(FOLD)

% This function creats  three data sets: a training set, a validation set,
% and a test set using the n-fold cross validation proceedure. 
%
% INPUTS: 
% FOLD - number of folds considered
%
% OUTPUTS:
% input_t - spectral data set for the training set. This parameter is a 
%           {1, FOLD} dimensional cell, where FOLD is the number of folds 
%           considered for the n-fold cross validation set. Each cell under 
%           a fold is a {1,N} dimensional cell which contains the data set, 
%           where N is the number of spectral vectors with their 
%           corresponding true labels.
% input_v - spectral data set for the validation set. This parameter is a 
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

% loading Excel file containing the whole data set.
load("ftir_spectra.mat");
load("ftir_labels.mat");

% initialization of some constant variables
N = size(S,1); % total number of spectral vectors considered.
C = max(S_label); % total number of classes considered (here is 2 classes - benign  and malignant)

% shuffling data set
rng('shuffle');
for i = 1:size(S,1)
    for j = 1:size(S,1)
        r = randi(size(S,1));
        [S(i,:),S(r,:)] = swap(S(i,:),S(r,:));
        [S_label(i),S_label(r)] = swap(S_label(i),S_label(r));
    end
end

% encoding all data
for i = 1:N
    val_num = single(S(i,:));
    val_label = zeros(1,C);
    val_label(S_label(i)) = 1;

    B_DATA_ALL{i}{1} = val_num; 
    B_DATA_ALL{i}{2} = val_label; 
end

% initialization of label parameters
label = zeros(1,C);
count = num2cell(zeros(1,C)); % quantity of samples perr class: 2 classes.
C_DATA = [];

% labeling of malignant or non-malignant.
for(i = 1:N)

    count{S_label(i)} = count{S_label(i)} + 1;
    C_DATA{S_label(i)}{count{S_label(i)}}{1} =  B_DATA_ALL{i}{1};
    C_DATA{S_label(i)}{count{S_label(i)}}{2} =  B_DATA_ALL{i}{2};

end

% grouping and division of data by folds.
T_DATA = [];
V_DATA = [];
TS_DATA = [];
 for i = 1:(FOLD)
    t_DATA = {};
    v_DATA = {};
    ts_DATA = {};
    
    for j = 1:size(C_DATA,2) % grouping malignant and non-malignant data per fold.
         C_size = ceil(size(C_DATA{j},2)/(FOLD + 1));
         if(i < (FOLD))
             t = (1+(C_size*(i-1))):(C_size*i);
         else
             t = (1+(C_size*(i-1))):size(C_DATA{j},2);
         end
         
         nmt = 1:size(C_DATA{j},2); 
         nmt(t) = [];
         
         dummy_DATA = C_DATA{j};
         dummy_DATA = dummy_DATA(t(1:ceil(size(t,2)/2)));
         ts_DATA{j} = dummy_DATA;
         
         dummy_DATA = C_DATA{j};
         dummy_DATA = dummy_DATA(t((ceil(size(t,2)/2) + 1):end));
         v_DATA{j} = dummy_DATA;

         dummy_DATA = C_DATA{j};
         dummy_DATA = dummy_DATA(nmt);
         t_DATA{j} = dummy_DATA;
         
    end

         T_DATA{i} = {[t_DATA{1}, t_DATA{2}]};
         V_DATA{i} = {[v_DATA{1}, v_DATA{2}]};
         TS_DATA{i} = {[ts_DATA{1}, ts_DATA{2}]};
 end

 for i = 1:FOLD % concatenating fold data sets to a single set (for training, validation and testing)
    for j = 1:size(T_DATA{i}{1},2)    
           m = ceil(size(T_DATA{i}{1},2)*rand(1,1));
           [T_DATA{i}{1}{j}, T_DATA{i}{1}{m}] = swap(T_DATA{i}{1}{j}, T_DATA{i}{1}{m});
    end
    
   input_t{i} = T_DATA{i}{1};
   input_v{i} = V_DATA{i}{1};
   input_ts{i} = TS_DATA{i}{1};
   
 end

end

