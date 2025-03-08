close all; clear; clc;

figure;
for i = 1:3
    % Input sounds files
    filename = sprintf('C:/Users/dc365/Desktop/Study/PHD/25winter/201/final project/Final_Project_export/GivenSpeech_Data/Training_Data/s%d.wav', i);
    
    % Obtain MFCC vectors
    mfcc_test = mfcc(filename, 256, 100);

    % Plot two dimensions in a 2D plane
    scatter(mfcc_test(:, 1), mfcc_test(:, 2), 100);
    hold on;
end