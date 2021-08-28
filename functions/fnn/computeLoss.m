function [delta_loss, total_loss] = computeLoss(p_label, true_label, total_loss)

% This function computes for the cost/error value credited by one spectral 
% data, S, using the binary cross-entrophy cost function.
%
% INPUTS:
% p_label - probability prediction for S being benign.
% true_label - true label of the sample S.
% total_loss - accumulated cost (if any).
%
% OUTPUTS:
% delta_loss - error value due to the sample S.
% total_loss - accumulated cost (if any).
    
    % computing for the individual error
    l_m0 = log10(abs(p_label));
    l_m0(isinf(l_m0)) = 0; % this line was implemented for numerical stability.
    l_m0 = single(true_label).*l_m0;
    l_m1 = log10(abs(1-p_label));
    l_m1(isinf(l_m1)) = 0; % this line was implemented for numerical stability.
    l_m1 = single((1 - true_label)).*l_m1;
    delta_loss = -1*sum(l_m0 + l_m1)/size(true_label,2);
    
    % computing for the total error
    total_loss = total_loss + delta_loss;
end

