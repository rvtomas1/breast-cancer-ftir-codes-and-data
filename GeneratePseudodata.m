% This script is used to generate a data set of malignant and benign ftir
% absorbance plots using the eigenvectors of the true (utilized) data set.
% The generation of a pseudodata set has been presented since the true data
% set can not be given due to patenting concerns.

clc; clear all;

% loading eigenvectors and data set information
load("data/set generator/ben_eigenvectors.mat");
load("data/set generator/mal_eigenvectors.mat");
load("data/set generator/class_stat.mat");

% number of spectral vectors to be generated per class.
N_mal = 88;
N_ben = 78;

% generation of malignant samples.
mean_mal = stat{1}{1};
std_mal = stat{2}{1};

for i = 1:size(mean_mal,2)
    generated_ftir_mal(:,i) = normrnd(mean_mal(i),std_mal(i),N_mal,1);
end
 generated_ftir_mal = generated_ftir_mal*mal_eigenvectors';
 generated_ftir_mal = normalize(generated_ftir_mal,2,'zscore');
 
% generation of benign samples.
mean_ben = stat{1}{2};
std_ben = stat{2}{2};

for i = 1:size(mean_ben,2)
    generated_ftir_ben(:,i) = normrnd(mean_ben(i),std_ben(i),N_ben,1);
end
 generated_ftir_ben = generated_ftir_ben*ben_eigenvectors';
 generated_ftir_ben = normalize(generated_ftir_ben,2,'zscore');
 
 figure();
 hold on;
 plot(generated_ftir_mal','r');
 plot(generated_ftir_ben','b');
 
 % saving of relevant information
 S = [generated_ftir_ben;generated_ftir_mal];
 S_label = [ones(N_mal,1);2*ones(N_mal,1)]';
 
 save("data/dataset/ftir_spectra",'S');
 save("data/dataset/ftir_labels",'S_label');
 
 save("ML models/ftir_spectra",'S');
 save("ML models/ftir_labels",'S_label');