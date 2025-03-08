close all; clear; clc;

% Set the arguements for VQ codebook using the LGB algorithm
M = 16;
epsilon = 0.01;
threshold = 1e-5;
max_iter = 100;

filename_1 = sprintf('C:/Users/dc365/Desktop/Study/PHD/25winter/201/final project/Final_Project_export/GivenSpeech_Data/Training_Data/s1.wav');
filename_2 = sprintf('C:/Users/dc365/Desktop/Study/PHD/25winter/201/final project/Final_Project_export/GivenSpeech_Data/Training_Data/s2.wav');
filename_3 = sprintf('C:/Users/dc365/Desktop/Study/PHD/25winter/201/final project/Final_Project_export/GivenSpeech_Data/Training_Data/s3.wav');

% Obtain MFCC vectors
mfcc_1 = mfcc(filename_1, 256, 100);
mfcc_2 = mfcc(filename_2, 256, 100);
mfcc_3 = mfcc(filename_3, 256, 100);

% VQ codewords
codebook_1 = vq_lbg(mfcc_1, M, epsilon, threshold, max_iter);
codebook_2 = vq_lbg(mfcc_2, M, epsilon, threshold, max_iter);
codebook_3 = vq_lbg(mfcc_3, M, epsilon, threshold, max_iter);

% Plot the resulting VQ codewords
scatter(codebook_1(:, 1), codebook_1(:, 2),100, 'filled','r','c');
hold on
scatter(codebook_2(:, 1), codebook_2(:, 2),100, 'filled','b','^');
scatter(codebook_3(:, 1), codebook_3(:, 2),100, 'filled','g','d');
scatter(mfcc_1(:, 1), mfcc_1(:, 2),'r','c');
scatter(mfcc_2(:, 1), mfcc_2(:, 2),'b','^');
scatter(mfcc_3(:, 1), mfcc_3(:, 2),'g','d');
xlabel("'MFCC Dimension 1")
ylabel("'MFCC Dimension 2")
legend('Speaker 1', 'Speaker 2', 'Speaker 3');