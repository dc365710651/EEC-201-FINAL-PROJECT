close all; clear; clc;

% Input sounds files
[y,Fs] = audioread('C:/Users/dc365/Desktop/Study/PHD/25winter/201/final project/Final_Project_export/GivenSpeech_Data/Training_Data/s1.wav');
sound(y, Fs);

% If the audio is dual channel, convert it to mono channel
if size(y,2) == 2
    y = mean(y, 2);
end

% Normalize
y_max = y / max(abs(y));
% Set different frame size
frame_sizes = [128, 256, 512];

for i = 1:length(frame_sizes)
    N = frame_sizes(i);
    % Set the frame increment M to be about N/3.
    M = floor(N/3);

    % Windowing
    hamming_window = hamming(N);

    % Using STFT
    [s,f,t] = spectrogram(y,hamming_window,N-M,N,Fs);

    % Periodogram
    Periodogram = abs(s).^2;
    
    % Locate the region in the plot that contains most of the energy
    figure;
    imagesc(t*1000, f, 10*log10(Periodogram));
    axis xy;
    xlabel("time/milliseconds")
    ylabel("Hz")
    title("Periodogram of audio 1")
end
