function b = my_softmax( a )

% This function implements the softmax function. This function achieves
% similar results to MATLAB's softmax() function.
%
% INPUT:
% a - a vector of numerical values
%
% OUTPUT:
% b - a vector of length equal to the length of the input "a". The vector
%       elements are that of the softmax-evaluated elements of input "a".
     
    % implementation was done using exponential addition/subtraction rather
    % than exp(a)/sum(exp(a)) for numerical stability.
     max_a = max(a);
     log_b = a - (max_a + log(sum(exp(a-max_a))));
     b = single(exp(log_b));
end

