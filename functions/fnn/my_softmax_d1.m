function b = my_softmax_d1( a )
    
% This function implements the first derivative of the softmax function. 
%
% INPUT:
% a - a vector of numerical values
%
% OUTPUT:
% b - a vector of length equal to the length of the input "a". The vector
%       elements are that of the first derivative softmax-evaluated 
%       elements of input "a".

    for i = 1:size(a,2)
        rep_a = a;
        rep_a(i) = [];
        
        % implementation was done using exponential addition/subtraction rather
        % than directly computing for exp(a) for numerical stability.
        log_b = a(i) + (max(rep_a) + log(sum(exp(rep_a-max(rep_a)))))- 2*(max(a) + log(sum(exp(a-max(a)))));
        b(i) = single(exp(log_b));
    end

end

