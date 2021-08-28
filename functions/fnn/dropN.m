function [ nn, dp_index ] = dropN( nn, dp )

% This function implements NN dropout for the hidden layers.
%
% INPUTS:
% nn - a cell containing the nn parameters of a dimension {2,L}
%           where L is the number of FNN layers. Index {1} contains
%           the weights, while the index {2} contains the biases.
% dp - drop-out rate (usually initialize as 90% for SELU)
%
% OUTPUTS:
% nn - a cell containing dropped-out nn parameters of a dimension {2,L}
%           where L is the number of FNN layers. Index {1} contains
%           the weights, while the index {2} contains the biases.
% dp_index - a cell of size {L} containing the indexes of neurons droped
%           out per layer.

    % quantifying the number of layers considered.
    L = size(nn{1},2);
     
    for i = 1:L-1
        % indexes of neurons to be dropped.
        n = randperm(size(nn{1}{i},2),ceil((1-dp)*size(nn{1}{i},2))); 
        
        % "nullification" of parameter values
        if i == 1
            nn{1}{i} = my_dropout(nn{1}{i},n,dp,'c');
        end
        nn{2}{i} = my_dropout(nn{2}{i},n,dp,'c');
        nn{1}{i+1} = my_dropout(nn{1}{i+1},n,dp,'r');
        
        % value assignment for dp_indexes.
        dp_index{i} = n;
    end
    
end

