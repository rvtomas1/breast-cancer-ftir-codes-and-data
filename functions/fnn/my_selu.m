function [ a ] = my_selu( a )
    
% This function evaluates a single value using the SELU activation function
%
% INPUT:
% a - a value from a neuron upon evaluating: h = w*f(h) + b.
%
% OUTPUT:
% a - the evaluated value of h, f(h).

    % initializing SELU-based parameters
    lambda = 1.0507009873554804934193349852946;
    alpha = 1.6732632423543772848170429916717;
    
    % SELU evaluation
    if (a>=0)
        a = single(a*lambda);
    else
        a = single(lambda*(alpha*exp(a)-alpha));
    end
end

