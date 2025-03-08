function MFCCseq = mfcc(filename, N, M)
% N: The first frame consists of the first N samples
% M: The second frame begins M samples after the first frame, and overlaps it by N - M samples

    [y,Fs] = audioread(filename);

    % If the audio is dual channel, convert it to mono channel
    if size(y,2) == 2
        y = mean(y, 2);
    end

    % Normalize
    y = y / max(abs(y));
    
    % Frame Blocking
    % Compute the number of frames
    number_frames = floor((length(y)-N)/M) + 1;
    
    % p = number of filters in filterbank
    % n = length of fft
    % fs = sample rate in Hz
    p = 20;
    number_components = 12;
    n = N;
    
    % Outputs: x = a (sparse) matrix containing the filterbank amplitudes
    % size(x) = [p, 1+floor(n/2)]
    m = melfb_own(p, n, Fs);
    
    % Initial the sequency
    MFCCseq = zeros(number_frames, number_components);

    for i = 1:number_frames

        % Set indices of each frame
        begin_point = 1 + (i-1)*M;
        end_point = begin_point + N - 1;
        frame = y(begin_point:end_point);

        % Windowing
        hamming_window = hamming(N);
        frame_window = frame .* hamming_window;
        
        % FFT
        FFT_frame = fft(frame_window, N);
        positive_n = 1 + floor(N/2);

        % Power Spectrum
        Power_Spectrum = (abs(FFT_frame(1:positive_n)).^2) / N;

        % Mel-frequency
        mel_frequency = m * Power_Spectrum;

        % log transform
        log_mel = log(mel_frequency);
        
        % Test 4
        % Cepstrum using Discrete Cosine Transform
        cepstrum = dct(log_mel);

        % Exclude the first component
        mfcc = cepstrum(2:(number_components+1));

        % Output the sequency
        MFCCseq(i,:) = mfcc;
    end
end