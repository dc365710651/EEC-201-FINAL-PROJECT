close all; clear; clc;

% Input sounds file
[y,Fs] = audioread('C:/Users/dc365/Desktop/Study/PHD/25winter/201/final project/Final_Project_export/GivenSpeech_Data/Training_Data/s1.wav');
sound(y, Fs);

% If the audio is dual channel, convert it to mono channel
if size(y,2) == 2
    y = mean(y, 2);
end

% Normalize
y = y / max(abs(y));

N_fft = 256;

% Number of the mel
p = 20;

% Plot the mel-spaced filter bank responses
m = melfb_own(p, N_fft, Fs);

positive_n = 1 + floor(N_fft/2);

freq_axis = linspace(0, Fs/2, positive_n);

% Theoretical responses (e.g. triangle shape)
figure;
hold on;
for i = 1:p
    plot(freq_axis, m(i,:), 'LineWidth', 1.5);
end
hold off;

% Choose one frame
frame = y(1:N_fft);

% Windowing
hamming_window = hamming(N_fft);
frame_window = frame .* hamming_window;

% FFT
FFT_frame = fft(frame_window, N_fft);

% Power Spectrum
Power_Spectrum = (abs(FFT_frame(1:positive_n)).^2) / N_fft;

% Before the melfrequency wrapping step
figure;
plot(freq_axis, 10*log10(Power_Spectrum), 'LineWidth', 1.5);
xlabel("Hz")
ylabel("dB")
title("The spectrum before the melfrequency wrapping step")

% Mel-frequency
mel_frequency = m * Power_Spectrum;

% log transform
log_mel = log(mel_frequency);

% After the melfrequency wrapping step
figure;
plot(1:p, log_mel);
xlabel("Hz")
ylabel("dB")
title("The spectrum after the melfrequency wrapping step")