function [ nn ] = genNN( L, input_num, output_num, layer_width, name, activation )

% Creates a L-layered neural network in which each hidden layer has the
% same quantity of neurons. Initializes the weights using gaussian random 
% initialization with a mean of 0 and a standard deviation of 1/(# of 
% neurons from previous layer) for the weights of a considered layer, while 
% biases are initialized as 0.
%
% INPUTS:
% L - number of layers.
% input_num - number of neurons for the input.
% output_num - number of neurons for the output.
% name - type of activation function. Serves only as a indictor, does not
%           realy do much as a programable variable.
% activation - if weights to be created were to be set 0 or not (if needed
%                for initialization purposes).
%
% OUTPUTS:
% nn - a cell containing the nn parameters of a dimension {2,L}
%           where L is the number of FNN layers. Index {1} contains
%           the weights, while the index {2} contains the biases.

h = [];
b = [];
rng('shuffle');

if(activation == 1) % gaussian weight initialization
    h{1} = single(normrnd(0,sqrt(1/input_num),input_num, layer_width));
    b{1} = single(zeros(1, layer_width)); 
    for i = 2:L
        h{size(h,2)+1} = single(normrnd(0,sqrt(1/layer_width) ,layer_width, layer_width));
        b{size(b,2)+1} = single(zeros(1, layer_width));  
        end
    h{L+1} = single(normrnd(0,sqrt(1/layer_width) ,layer_width, output_num));
    b{L+1} = single(zeros(1, output_num));
    
elseif(activation == 0) % zero-weight initialization
    h{1} = single(zeros(input_num, layer_width));
    b{1} = single(zeros(1, layer_width));  
    for i = 2:L
        h{size(h,2)+1} = single(zeros(layer_width, layer_width));
        b{size(b,2)+1} = single(zeros(1, layer_width));
        end
    h{L+1} = single(zeros(layer_width, output_num));
    b{L+1} = single(zeros(1, output_num));
    
    end
        nn = {h,b,0};
end

