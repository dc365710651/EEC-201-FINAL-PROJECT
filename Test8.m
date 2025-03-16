function Test8()
    % Test8:
    % 1) Loads or trains VQ codebooks for each known speaker in the training set.
    % 2) Applies a notch filter (or multiple filters) to each test file to suppress 
    %    certain frequency components.
    % 3) Extracts MFCC from the filtered signal, computes distortion to each speaker's 
    %    codebook, finds the best match, and records recognition accuracy.
    % 4) Reports accuracy under notch filtering to indicate system robustness.

    trainDataPath = '/Users/solemnity/Documents/EEC 201/Final_Project/GivenSpeech_Data/Training_Data';
    testDataPath  = '/Users/solemnity/Documents/EEC 201/Final_Project/GivenSpeech_Data/Test_Data';

    % Train codebooks from the training set
    numSpeakers = 11;
    codebookSize = 128;
    codebooks = cell(numSpeakers,1);
    for spk = 1:numSpeakers
        fname = fullfile(trainDataPath, ['s', num2str(spk), '.wav']);
        [mfccData, ~] = computeMFCC(fname);
        codebooks{spk} = lbgTrainCodebook(mfccData, codebookSize);
    end

    numTestFiles = 8;

    % For example, a notch filter is designed at 60 Hz
    notchFrequencies = [60]; 

    % The accuracy at different notch center frequencies was recorded
    for nf = 1:length(notchFrequencies)
        freqToRemove = notchFrequencies(nf);
        correctCount = 0;

        fprintf('\n=== Now testing with a notch filter at %d Hz ===\n', freqToRemove);
        for t = 1:numTestFiles
            fnameTest = fullfile(testDataPath, ['s', num2str(t), '.wav']);
            [signal, fs] = audioread(fnameTest);

            filteredSignal = applyNotchFilter(signal, fs, freqToRemove);

            [testMFCC, ~] = computeMFCC_fromSignal(filteredSignal, fs);

            distList = zeros(numSpeakers,1);
            for spk = 1:numSpeakers
                distList(spk) = computeVQDistortion(testMFCC, codebooks{spk});
            end

            [~, predictedSpeaker] = min(distList);

            if predictedSpeaker == t
                correctCount = correctCount + 1;
            end
            fprintf('Test file s%d.wav => recognized as Speaker %d\n', t, predictedSpeaker);
        end

        accuracy = correctCount / numTestFiles * 100;
        fprintf('Test8 (Notch %d Hz): Accuracy on these %d test files = %.2f%%\n', ...
            freqToRemove, numTestFiles, accuracy);
    end

    fprintf('Test8 completed. Compare results under different notch filters.\n');
end


function filteredSignal = applyNotchFilter(signal, fs, freqToRemove)

    wo = freqToRemove/(fs/2);
    if wo >= 1
        % If the frequency is too high or the sampling rate is too low, wo> = 1, then no processing is performed
        filteredSignal = signal;
        return;
    end
    Q = 35; 
    [b, a] = iirnotch(wo, wo/Q);
    filteredSignal = filter(b, a, signal);
end


function [mfccMatrix, fs] = computeMFCC_fromSignal(signal, fs)

    signal = signal(:);

    N = 256;              
    M = 100;              
    numMelFilters = 26;   
    numCepstra = 12;      

    frames = blockFrames(signal, N, M);
    numFrames = size(frames, 2);
    mfccMatrix = zeros(numFrames, numCepstra);

    melFB = melfb_own(numMelFilters, N, fs);

    for i = 1:numFrames
        frameData = frames(:, i);
        w = hamming(N);
        windowed = frameData .* w;
        spectrum = abs(fft(windowed, N)).^2;
        halfSpec = spectrum(1:floor(N/2)+1);
        melSpec = melFB * halfSpec;
        logMelSpec = log(melSpec + eps);
        c = dct(logMelSpec);
        mfccMatrix(i, :) = c(2:numCepstra+1);
    end
end


function [mfccMatrix, fs] = computeMFCC(wavFile)
    [signal, fs] = audioread(wavFile);
    [mfccMatrix, fs] = computeMFCC_fromSignal(signal, fs);
end


function frames = blockFrames(sig, N, M)
    L = length(sig);
    numFrames = ceil((L - N)/M) + 1;
    frames = zeros(N, numFrames);
    idx = 1;
    for n = 1:numFrames
        if idx+N-1 <= L
            frames(:, n) = sig(idx:idx+N-1);
        else
            leftover = L - idx + 1;
            frames(1:leftover, n) = sig(idx:end);
        end
        idx = idx + M;
    end
end

function m = melfb_own(p, n, fs)
    f0 = 700 / fs;
    fn2 = floor(n/2);
    Lr = log(1 + 0.5 / f0) / (p + 1);

    Bv = n * (f0 * (exp([0 1 p p+1] * Lr) - 1));
    b1 = floor(Bv(1)) + 1;
    b2 = ceil(Bv(2));
    b3 = floor(Bv(3));
    b4 = min(fn2, ceil(Bv(4))) - 1;

    if b2 > b4 || b1 > b3
        m = sparse(p, 1 + fn2);
        return;
    end

    pf = log(1 + (b1:b4) / n / f0) / Lr;
    fp = floor(pf);
    pm = pf - fp;

    idx_hi = (b2 - b1 + 1):(b4 - b1 + 1);
    idx_lo = 1:(b3 - b1 + 1);

    r = [fp(idx_hi) + 1, fp(idx_lo)];
    c = [b2:b4, b1:b3] + 1;
    v = 2 * [1 - pm(idx_hi), pm(idx_lo)];

    r(r < 1) = 1;
    r(r > p) = p;
    m = sparse(r, c, v, p, 1 + fn2);
end


function codebook = lbgTrainCodebook(data, M)
    epsilon = 0.01;
    codebook = mean(data, 1);  
    codebook = codebook(:)';  

    currentSize = 1;
    while currentSize < M
        codebookPlus  = codebook .* (1 + epsilon);
        codebookMinus = codebook .* (1 - epsilon);
        codebook = [codebookPlus; codebookMinus];
        currentSize = size(codebook, 1);

        distortionOld = 1e9;
        while true
            assignments = zeros(size(data,1), 1);
            for i = 1:size(data,1)
                dists = sum((data(i,:) - codebook).^2, 2);
                [~, idx] = min(dists);
                assignments(i) = idx;
            end
            for c = 1:currentSize
                clusterPoints = data(assignments == c, :);
                if ~isempty(clusterPoints)
                    codebook(c, :) = mean(clusterPoints, 1);
                end
            end
            distortion = 0;
            for i = 1:size(data,1)
                distortion = distortion + sum((data(i,:) - codebook(assignments(i),:)).^2);
            end
            distortion = distortion / size(data,1);
            if abs(distortionOld - distortion) < 1e-5
                break
            else
                distortionOld = distortion;
            end
        end
    end
end

function d = computeVQDistortion(mfccData, codebook)
    d = 0;
    for i = 1:size(mfccData,1)
        distVec = sum((mfccData(i,:) - codebook).^2, 2);
        d = d + min(distVec);
    end
    d = d / size(mfccData,1);
end
