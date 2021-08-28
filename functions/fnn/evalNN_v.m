function [ delta_loss, p_label ] = evalNN_v( nn, S )

% This function implements the forward pass for a feed forward neural
% network. The function receives a single input, which is a single spectral
% vector, S. This function is used for the neural network validation/testing.
%
% INPUTS:
% nn - a cell containing the nn parameters of a dimension {2,L}
%           where L is the number of FNN layers. Index {1} contains
%           the weights, while the index {2} contains the biases.
% S - a cell containing a spectral data set with its corresponding
%           labels. The cell has a dimension of {2,N}, where N is the
%           number of spectral data samples contained. Index {1} contains 
%           the spectral data which is a 450+ element vector, while index
%           {2} contains the corresponding label.
%
% OUTPUTS:
% delta_loss - error value due to the sample S.
% p_label - probability prediction for S being benign.

% quantifying the number of layers considered.
L = size(nn{1},2);

% The forward pass
a = S{1}*nn{1}{1} + nn{2}{1};
f_a = arrayfun(@my_selu, a);

    for i = 2:L-1
        a = f_a*nn{1}{i} + nn{2}{i};
        f_a = arrayfun(@my_selu, a);
    end
    a = f_a*nn{1}{L} + nn{2}{L};
    f_a = my_softmax(a);
    
    % cancer probabilities
    p_label = f_a;
    
    % computation of nn set loss/error.
    [delta_loss, nn{3}] = computeLoss(p_label,S{2},nn{3});
end

