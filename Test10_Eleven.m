function Test10_Eleven()
    % Test9_Eleven:
    % Speaker recognition for word "Eleven."
    
    baseTrainPath = '/Users/solemnity/Documents/EEC 201/Final_Project/GivenSpeech_Data/Eleven_Training_Data';
    baseTestPath  = '/Users/solemnity/Documents/EEC 201/Final_Project/GivenSpeech_Data/Eleven_Test_Data';

    numSpeakers   = 23;
    codebookSize  = 128;  % example
    codebooks     = cell(numSpeakers, 1);

    % Train codebooks
    for spk = 1:numSpeakers
        trainFile = fullfile(baseTrainPath, sprintf('s%d.wav', spk));
        [mfccData, ~] = computeMFCC(trainFile);
        codebooks{spk} = lbgTrainCodebook(mfccData, codebookSize);
    end

    % Test codebooks
    correctCount = 0;
    for spk = 1:numSpeakers
        testFile = fullfile(baseTestPath, sprintf('s%d.wav', spk));
        [testMFCC, ~] = computeMFCC(testFile);

        distVals = zeros(numSpeakers,1);
        for k = 1:numSpeakers
            distVals(k) = computeVQDistortion(testMFCC, codebooks{k});
        end
        [~, predictedSpk] = min(distVals);

        fprintf('Eleven-Word: s%d.wav => recognized as Speaker %d\n', spk, predictedSpk);
        if predictedSpk == spk
            correctCount = correctCount + 1;
        end
    end

    accuracy = correctCount / numSpeakers * 100;
    fprintf('Test10_Eleven: Accuracy for "Eleven" across 23 speakers = %.2f%%\n', accuracy);
end

%%%%%%%%%%%%%%%%%%%% (Same subfunctions) %%%%%%%%%%%%%%%%%%%

function [mfccMatrix, fs] = computeMFCC(wavFile)
    [signal, fs] = audioread(wavFile);
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
    % Compute the total VQ distortion between 
    % all frames in mfccData and the given codebook.
    % codebook is [M x D], mfccData is [N x D].
    d = 0;
    for i = 1:size(mfccData,1)
        distVec = sum((mfccData(i,:) - codebook).^2, 2);
        d = d + min(distVec);
    end
    d = d / size(mfccData,1);
end
