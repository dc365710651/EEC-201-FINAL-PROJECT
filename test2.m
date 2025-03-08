close all; clear; clc;

milliseconds = zeros(1,11);
sample_rate = zeros(1,11);

for i = 1:11
    % Input sounds files
    filename = sprintf('C:/Users/dc365/Desktop/Study/PHD/25winter/201/final project/Final_Project_export/GivenSpeech_Data/Training_Data/s%d.wav', i);
    [y,Fs] = audioread(filename);
    sample_rate(i) = Fs;
    sound(y, Fs);
    
    % If the audio is dual channel, convert it to mono channel
    if size(y,2) == 2
        y = mean(y, 2);
    end
    % Normalize
    y = y / max(abs(y));
    
    % compute how many milliseconds of speech are contained in a block of 256 samples
    milliseconds(i) = (256/Fs)*1000;

    % Plot the signal
    figure;
    plot((0:length(y)-1)/Fs, y)
    xlabel("time/second")
    title("Signal it in the time domain")
end