function [ nn_layer ] = my_dropout( nn_layer ,dp_index , dp, r_or_c )

% This function implements a SELU-based dropout for a single layer.
%
% INPUTS:
% nn_layer - a neural network layer of a specified number of neurons
% dp_index - - a vector containing the indexes of neurons droped
% dp - drop-out rate (usually initialize as 90% for SELU)
% r_or_c - an indicator which denotes whether to "nullify" a row or a
%           column.
%
% OUTPUTS: 
% nn_layer - a neural network layer of a specified number of neurons
%           (with dropped neurons)


% initializing SELU-based parameters
lambda = 1.0507009873554804934193349852946;
alpha = 1.6732632423543772848170429916717;

% initializing SELU-based dropout constant
beta1 = sqrt(dp + ((lambda*alpha)^2)*dp*(1-dp));
beta2 = -sqrt(dp + ((lambda*alpha)^2)*dp*(1-dp))*((1-dp)*(-lambda*alpha));

% SELU-based dropout
if(r_or_c == 'r')
    nn_layer(dp_index,:) = beta1*(nn_layer(dp_index,:)*dp -lambda*alpha*(1-dp) ) + beta2;
    elseif(r_or_c == 'c')
    nn_layer(:,dp_index) = beta1*(nn_layer(:,dp_index)*dp -lambda*alpha*(1-dp) ) + beta2;
    end


end

