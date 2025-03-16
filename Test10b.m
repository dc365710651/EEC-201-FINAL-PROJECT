function Test10b()
    % This code trains separate codebooks for each speaker's "Five" data
    % and each speaker's "Eleven" data, then tests all samples to identify
    % both the speaker and the spoken word.

    % Adjust paths as needed:
    baseTrainPathFive   = '/Users/solemnity/Documents/EEC 201/Final_Project/GivenSpeech_Data/Five_Training_Data';
    baseTestPathFive    = '/Users/solemnity/Documents/EEC 201/Final_Project/GivenSpeech_Data/Five_Test_Data';
    baseTrainPathEleven = '/Users/solemnity/Documents/EEC 201/Final_Project/GivenSpeech_Data/Eleven_Training_Data';
    baseTestPathEleven  = '/Users/solemnity/Documents/EEC 201/Final_Project/GivenSpeech_Data/Eleven_Test_Data';

    numSpeakers  = 23; 
    codebookSize = 32;  % Example size for VQ

    codebooksFive   = cell(numSpeakers, 1);
    codebooksEleven = cell(numSpeakers, 1);

    % ------------------ Training -------------------
    for spk = 1:numSpeakers
        % Train on "Five"
        fiveTrainFile = fullfile(baseTrainPathFive, sprintf('s%d.wav', spk));
        [mfccDataF, ~] = computeMFCC(fiveTrainFile);
        codebooksFive{spk} = lbgTrainCodebook(mfccDataF, codebookSize);

        % Train on "Eleven"
        elevenTrainFile = fullfile(baseTrainPathEleven, sprintf('s%d.wav', spk));
        [mfccDataE, ~] = computeMFCC(elevenTrainFile);
        codebooksEleven{spk} = lbgTrainCodebook(mfccDataE, codebookSize);
    end

    % ------------------ Testing -------------------
    % We will test both "Five" and "Eleven" files. For each test file,
    % we check distances against both sets of codebooks (Five and Eleven).
    % Whichever set is closest, we take that as the predicted word and speaker.

    totalSpeakerCorrect = 0;
    totalWordCorrect    = 0;
    totalSamples        = numSpeakers * 2; % Each speaker says both words

    % --- Test "Five" samples ---
    for spk = 1:numSpeakers
        fiveTestFile = fullfile(baseTestPathFive, sprintf('s%d.wav', spk));
        [testMFCC, ~] = computeMFCC(fiveTestFile);

        % Distances to "Five" codebooks
        distFive = zeros(numSpeakers, 1);
        for k = 1:numSpeakers
            distFive(k) = computeVQDistortion(testMFCC, codebooksFive{k});
        end

        % Distances to "Eleven" codebooks
        distEleven = zeros(numSpeakers, 1);
        for k = 1:numSpeakers
            distEleven(k) = computeVQDistortion(testMFCC, codebooksEleven{k});
        end

        [minValFive, idxFive] = min(distFive);
        [minValEleven, idxEleven] = min(distEleven);

        if minValFive < minValEleven
            predictedWord = 'Five';
            predictedSpk  = idxFive;
        else
            predictedWord = 'Eleven';
            predictedSpk  = idxEleven;
        end

        % Check correctness
        if predictedSpk == spk
            totalSpeakerCorrect = totalSpeakerCorrect + 1;
        end
        if strcmp(predictedWord, 'Five')
            totalWordCorrect = totalWordCorrect + 1;
        end

        fprintf('Test file: Five s%d.wav => Predicted: Speaker %d, Word "%s"\n', ...
                spk, predictedSpk, predictedWord);
    end

    % --- Test "Eleven" samples ---
    for spk = 1:numSpeakers
        elevenTestFile = fullfile(baseTestPathEleven, sprintf('s%d.wav', spk));
        [testMFCC, ~] = computeMFCC(elevenTestFile);

        distFive = zeros(numSpeakers, 1);
        for k = 1:numSpeakers
            distFive(k) = computeVQDistortion(testMFCC, codebooksFive{k});
        end

        distEleven = zeros(numSpeakers, 1);
        for k = 1:numSpeakers
            distEleven(k) = computeVQDistortion(testMFCC, codebooksEleven{k});
        end

        [minValFive, idxFive] = min(distFive);
        [minValEleven, idxEleven] = min(distEleven);

        if minValFive < minValEleven
            predictedWord = 'Five';
            predictedSpk  = idxFive;
        else
            predictedWord = 'Eleven';
            predictedSpk  = idxEleven;
        end

        % Check correctness
        if predictedSpk == spk
            totalSpeakerCorrect = totalSpeakerCorrect + 1;
        end
        if strcmp(predictedWord, 'Eleven')
            totalWordCorrect = totalWordCorrect + 1;
        end

        fprintf('Test file: Eleven s%d.wav => Predicted: Speaker %d, Word "%s"\n', ...
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
        if idx + N - 1 <= L
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
                dists = sum((data(i,:)-codebook).^2, 2);
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
