close all; clear; clc;

train_path = "C:\Users\dc365\Desktop\Study\PHD\25winter\201\final project\Final_Project_export\GivenSpeech_Data\Training_Data";
test_path = "C:\Users\dc365\Desktop\Study\PHD\25winter\201\final project\Final_Project_export\GivenSpeech_Data\Test_Data";

% Parameters
K = 17;
epsilon = 0.01;
threshold = 1e-5;
max_iter = 100;
N = 256;
M = 100;
number_train = 11;
number_test = 8;

% Initialize the codebook
codebooks = cell(number_train, 1);

% Training step
for i = 1:number_train
    train_file = fullfile(train_path, sprintf('s%d.wav', i));
    % Obtain MFCC vectors
    MFCC_train = mfcc(train_file, N, M);
    % VQ codewords
    codebooks{i} = vq_lbg(MFCC_train, K, epsilon, threshold, max_iter);
end

% Testing step
accuracy = 0;
for i = 1:number_test
    test_file = fullfile(test_path, sprintf('s%d.wav', i));
    % Obtain MFCC vectors
    MFCC_test = mfcc(test_file, N, M);
    number_frames = size(MFCC_test,1);
    
    % Initialize
    distortions = zeros(number_train, 1);
    
    % Match the testing and training data
    for j = 1:number_train
        % For each codebook of each training signal
        codebook = codebooks{j};
        distortion = 0;
        % For each frame of testing signal
        for k = 1:number_frames
            % Calculate the distance between training and testing
            d = disteu(MFCC_test(k, :)', codebook');
            distortion = distortion + min(d);
        end
        distortions(j) = distortion / number_frames;
    end

    % Obtain the minimal difference and save the prediction result
    [~, prediction] = min(distortions);

    % Compare with the real label
    if prediction == i
        accuracy = accuracy + 1;
    end
end

%Calculate the accuracy rate
accuracy = (accuracy / number_test) * 100;