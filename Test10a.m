function Test10a()
    % This code trains separate codebooks for each speaker's "Zero" data and each speaker's "Twelve" data, then tests all samples to identify both the speaker and the spoken word.

    % Adjust paths as needed:
    baseTrainPathZero   = '/Users/solemnity/Documents/EEC 201/Final_Project/GivenSpeech_Data/Zero_Training_Data';
    baseTestPathZero    = '/Users/solemnity/Documents/EEC 201/Final_Project/GivenSpeech_Data/Zero_Test_Data';
    baseTrainPathTwelve = '/Users/solemnity/Documents/EEC 201/Final_Project/GivenSpeech_Data/Twelve_Training_Data';
    baseTestPathTwelve  = '/Users/solemnity/Documents/EEC 201/Final_Project/GivenSpeech_Data/Twelve_Test_Data';

    numSpeakers   = 18;
    codebookSize  = 32;  % Example size for VQ
    codebooksZero = cell(numSpeakers, 1);
    codebooksTwelve = cell(numSpeakers, 1);

    % ------------------ Training -------------------
    for spk = 1:numSpeakers
        % Train on "Zero"
        zeroTrainFile = fullfile(baseTrainPathZero, sprintf('Zero_train%d.wav', spk+1));
        [mfccDataZ, ~] = computeMFCC(zeroTrainFile);
        codebooksZero{spk} = lbgTrainCodebook(mfccDataZ, codebookSize);

        % Train on "Twelve"
        twelveTrainFile = fullfile(baseTrainPathTwelve, sprintf('Twelve_train%d.wav', spk+1));
        [mfccDataT, ~] = computeMFCC(twelveTrainFile);
        codebooksTwelve{spk} = lbgTrainCodebook(mfccDataT, codebookSize);
    end

    % ------------------ Testing -------------------
    % We will test both "Zero" and "Twelve" files. For each test file, we check distances against all codebooks (both Zero and Twelve). Whichever codebook is closest, we take that as the predicted speaker and word.

    totalSpeakerCorrect = 0;
    totalWordCorrect    = 0;
    totalSamples        = numSpeakers * 2; % Each speaker says both words

    % --- Test "Zero" samples ---
    for spk = 1:numSpeakers
        zeroTestFile = fullfile(baseTestPathZero, sprintf('Zero_test%d.wav', spk+1));
        [testMFCC, ~] = computeMFCC(zeroTestFile);

        % Distances to Zero codebooks
        distZero = zeros(numSpeakers, 1);
        for k = 1:numSpeakers
            distZero(k) = computeVQDistortion(testMFCC, codebooksZero{k});
        end

        % Distances to Twelve codebooks
        distTwelve = zeros(numSpeakers, 1);
        for k = 1:numSpeakers
            distTwelve(k) = computeVQDistortion(testMFCC, codebooksTwelve{k});
        end

        % Find the smallest distance among all 36 codebooks
        [minValZero, idxZero] = min(distZero);
        [minValTwelve, idxTwelve] = min(distTwelve);

        if minValZero < minValTwelve
            predictedWord = 'Zero';
            predictedSpk  = idxZero;
        else
            predictedWord = 'Twelve';
            predictedSpk  = idxTwelve;
        end

        % Check correctness
        if predictedSpk == spk
            totalSpeakerCorrect = totalSpeakerCorrect + 1;
        end
        if strcmp(predictedWord, 'Zero')
            totalWordCorrect = totalWordCorrect + 1;
        end

        fprintf('Test file: Zero_test%d.wav => Predicted: Speaker %d, Word "%s"\n', ...
                spk, predictedSpk, predictedWord);
    end

    % --- Test "Twelve" samples ---
    for spk = 1:numSpeakers
        twelveTestFile = fullfile(baseTestPathTwelve, sprintf('Twelve_test%d.wav', spk+1));
        [testMFCC, ~] = computeMFCC(twelveTestFile);

        distZero = zeros(numSpeakers, 1);
        for k = 1:numSpeakers
            distZero(k) = computeVQDistortion(testMFCC, codebooksZero{k});
        end

        distTwelve = zeros(numSpeakers, 1);
        for k = 1:numSpeakers
            distTwelve(k) = computeVQDistortion(testMFCC, codebooksTwelve{k});
        end

        [minValZero, idxZero] = min(distZero);
        [minValTwelve, idxTwelve] = min(distTwelve);

        if minValZero < minValTwelve
            predictedWord = 'Zero';
            predictedSpk  = idxZero;
        else
            predictedWord = 'Twelve';
            predictedSpk  = idxTwelve;
        end

        % Check correctness
        if predictedSpk == spk
            totalSpeakerCorrect = totalSpeakerCorrect + 1;
        end
        if strcmp(predictedWord, 'Twelve')
            totalWordCorrect = totalWordCorrect + 1;
        end

        fprintf('Test file: Twelve_test%d.wav => Predicted: Speaker %d, Word "%s"\n', ...
                spk, predictedSpk, predictedWord);
    end

    % ------------------ Results -------------------
    speakerAccuracy = totalSpeakerCorrect / totalSamples * 100;
    wordAccuracy    = totalWordCorrect / totalSamples * 100;

    fprintf('\nOverall speaker identification accuracy: %.2f%%\n', speakerAccuracy);
    fprintf('Overall word identification accuracy: %.2f%%\n', wordAccuracy);
end

% ------------------ Subfunctions -------------------
function [mfccMatrix, fs] = computeMFCC(wavFile)
    [signal, fs] = audioread(wavFile);
    signal = signal(:);

    N = 256;
    M = 100;
    numMelFilters = 40;
    numCepstra = 25;

    frames = blockFrames(signal, N, M);
    numFrames = size(frames, 2);
    mfccMatrix = zeros(numFrames, numCepstra);

    melFB = melfb_own(numMelFilters, N, fs);

    for i = 1:numFrames
        frameData = frames(:, i);
        w = hamming(N);
        windowed = frameData .* w;
        powerSpec = abs(fft(windowed, N)).^2;
        halfSpec = powerSpec(1:floor(N/2)+1);
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

    Bv = n * (f0*(exp([0 1 p p+1]*Lr) - 1));
    b1 = floor(Bv(1)) + 1;
    b2 = ceil(Bv(2));
    b3 = floor(Bv(3));
    b4 = min(fn2, ceil(Bv(4))) - 1;

    if b2 > b4 || b1 > b3
        m = sparse(p, 1 + fn2);
        return;
    end

    pf = log(1 + (b1:b4)/n/f0) / Lr;
    fp = floor(pf);
    pm = pf - fp;

    idx_hi = (b2-b1+1):(b4-b1+1);
    idx_lo = 1:(b3-b1+1);

    r = [fp(idx_hi)+1, fp(idx_lo)];
    c = [b2:b4, b1:b3] + 1;
    v = 2*[1 - pm(idx_hi), pm(idx_lo)];

    r(r<1) = 1;
    r(r>p) = p;
    m = sparse(r, c, v, p, 1+fn2);
end

function codebook = lbgTrainCodebook(data, M)
    epsilon = 0.01;
    codebook = mean(data,1);
    codebook = codebook(:)';  
    currentSize = 1;
    while currentSize < M
        codebookPlus  = codebook .* (1 + epsilon);
        codebookMinus = codebook .* (1 - epsilon);
        codebook = [codebookPlus; codebookMinus];
        currentSize = size(codebook,1);

        distortionOld = 1e9;
        while true
            assignments = zeros(size(data,1),1);
            for i = 1:size(data,1)
                dists = sum((data(i,:) - codebook).^2, 2);
                [~, idx] = min(dists);
                assignments(i) = idx;
            end

            for c = 1:currentSize
                clusterPoints = data(assignments == c,:);
                if ~isempty(clusterPoints)
                    codebook(c,:) = mean(clusterPoints,1);
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
