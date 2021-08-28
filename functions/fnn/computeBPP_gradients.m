function [ delta_nn ] = computeBPP_gradients( dp_nn, h_out, fh_out, S )

% This function computes for the gradients via back propagation.
%
% INPUTS:
% dp_nn - a cell containing dropped-out nn parameters of a dimension {2,L}
%           where L is the number of FNN layers. Index {1} contains
%           the weights, while the index {2} contains the biases.
% h_out - a cell containing the neuron non-activated products h = w*f(h) + b.
%           The cell has a dimension of {1,L} where L is the number of FNN
%           layers.
% fh_out - a cell containing the neuron function-activated values f(h).
%           This cell has a dimension similar to h_out.
% S - a cell containing a spectral data set with its corresponding
%           labels. The cell has a dimension of {2,N}, where N is the
%           number of spectral data samples contained. Index {1} contains 
%           the spectral data which is a 450+ element vector, while index
%           {2} contains the corresponding label.
%
% OUTPUTS:
% delta_nn - a cell containing the computed parameter gradients of a dimension {2,L}
%           where L is the number of FNN layers. Index {1} contains
%           the weights, while the index {2} contains the biases.

    % quantifying the number of layers considered.
    L = size(h_out,2); 
    
    % computing for the binary cross-entrophy cost gradient.
    l_m0 = 1./abs(my_softmax(h_out{L}));
    l_m0(isinf(l_m0)) = 0; % this line was implemented for numerical stability.
    l_m0 = single(S{2}).*l_m0*(1/log(10));
    l_m1 = 1./(abs(1-my_softmax(h_out{L})));
    l_m1(isinf(l_m1)) = 0; % this line was implemented for numerical stability.
    l_m1 = (1 - single(S{2})).*l_m1*(1/log(10));
    E = (l_m1 - l_m0)/size(S{2},2);
    
    % computation of gradients accross each parameter from the Lth layer to
    % the first layer.
    delta_nn{2}{L} = E.*my_softmax_d1(h_out{L});
    delta_nn{1}{L} = E.*my_softmax_d1(h_out{L}).*fh_out{L-1}';
    
    for i = (L-1):-1:2
        delta_nn{2}{i} = sum((delta_nn{2}{i+1}.*dp_nn{1}{i+1})').*arrayfun(@my_selu_d1, h_out{i});
        delta_nn{1}{i} = delta_nn{2}{i}.*fh_out{i-1}';
    end
    
    delta_nn{2}{1} = sum((delta_nn{2}{2}.*dp_nn{1}{2})').*arrayfun(@my_selu_d1, h_out{1});
    delta_nn{1}{1} = delta_nn{2}{1}.*S{1}';

end

