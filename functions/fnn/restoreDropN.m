function [ nn ] = restoreDropN( nn, dp )

% This function multiplies the value of the dropout rate to the weights and
% bias as  consequence of dropout.
%
% INPUTS:
% dp - drop-out rate (usually initialize as 90% for SELU)
% nn - a cell containing the nn parameters of a dimension {2,L}
%           where L is the number of FNN layers. Index {1} contains
%           the weights, while the index {2} contains the biases.
%
% OUTPUTS:
% nn (restored) - a cell containing the nn parameters of a dimension {2,L}
%           where L is the number of FNN layers. Index {1} contains
%           the weights, while the index {2} contains the biases.

    % quantifying the number of layers considered.
    L = size(nn{1},2);


    for i = 2:L
        nn{1}{i} = dp*nn{1}{i};
        if(i < L) % bias of NN is not aaffected.
            nn{2}{i} = dp*nn{2}{i};
        end
    end


end

