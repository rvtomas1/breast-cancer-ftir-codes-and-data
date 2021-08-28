function [ delta_loss, nn, h_out, fh_out, p_label ] = evalNN_tr( dp_nn, dp_index, nn, S )

% This function implements the forward pass for a feed forward neural
% network. The function receives a single input, which is a single spectral
% vector, S. This function is used for the neural network training.
%
% INPUTS:
% dp_nn - a cell containing dropped-out nn parameters of a dimension {2,L}
%           where L is the number of FNN layers. Index {1} contains
%           the weights, while the index {2} contains the biases.
% dp_index - a cell of size {L} containing the indexes of neurons droped
%           out per layer.
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
% nn - a cell containing the nn parameters of a dimension {2,L}
%           where L is the number of FNN layers. Index {1} contains
%           the weights, while the index {2} contains the biases.
% h_out - a cell containing the neuron non-activated products h = w*f(h) + b.
%           The cell has a dimension of {1,L} where L is the number of FNN
%           layers.
% fh_out - a cell containing the neuron function-activated values f(h).
%           This cell has a dimension similar to h_out.
% p_label - probability prediction for S being benign.

% initializing SELU-based parameters
lambda = 1.0507009873554804934193349852946;
alpha = 1.6732632423543772848170429916717;

% quantifying the number of layers considered.
L = size(dp_nn{1},2);

% The forward pass
a = S{1}*dp_nn{1}{1} + dp_nn{2}{1};
f_a = arrayfun(@my_selu, a);
h_out{1} = a;
fh_out{1} = f_a;

    for i = 2:L-1
        a = f_a*dp_nn{1}{i} + dp_nn{2}{i};
        f_a = arrayfun(@my_selu, a);
        
        % SELU-based dropout specific
        a(dp_index{i}) = -inf;
        f_a(dp_index{i}) = -lambda*alpha;
        
        h_out{i} = a;
        fh_out{i} = f_a;
    end
    a = f_a*dp_nn{1}{L} + nn{2}{L};
    f_a = my_softmax(a);
    h_out{L} = a;
    fh_out{L} = f_a;
    
    % cancer probabilities
    p_label = f_a;
    
    % computation of loss function. NN third index {3} is a variable
    % storing its training error.
    [delta_loss, nn{3}] = computeLoss(p_label,S{2},nn{3}); 
    
end

