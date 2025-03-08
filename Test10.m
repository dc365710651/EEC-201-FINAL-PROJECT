function Test10()
    % Test10_FiveEleven:
    %  We have 23 speakers. Each speaker has 2 training wav files:
    %    1) sX_five.wav in "Five_Training_Data"
    %    2) sX_eleven.wav in "Eleven_Training_Data"
    %  and 2 test wav files:
    %    1) sX_five.wav in "Five_Test_Data"
    %    2) sX_eleven.wav in "Eleven_Test_Data"
    %
    % We treat each (speaker, word) as a separate class. That is 46 classes total.

    numSpeakers  = 23;
    codebookSize = 256;

    % Folder paths
    trainFivePath   = '/Users/solemnity/Documents/EEC 201/Final_Project/GivenSpeech_Data/Five_Training_Data';
    trainElevenPath = '/Users/solemnity/Documents/EEC 201/Final_Project/GivenSpeech_Data/Eleven_Training_Data';

    testFivePath    = '/Users/solemnity/Documents/EEC 201/Final_Project/GivenSpeech_Data/Five_Test_Data';
    testElevenPath  = '/Users/solemnity/Documents/EEC 201/Final_Project/GivenSpeech_Data/Eleven_Test_Data';

    totalClasses = 2 * numSpeakers;
    codebooks = cell(totalClasses, 1);


    % TRAIN codebooks

    classIndex = 1;
    for spk = 1:numSpeakers
        % 1) The "five" class for speaker spk
        fileFive = fullfile(trainFivePath, sprintf('s%d.wav', spk));
        [mfccFive, ~] = computeMFCC(fileFive);
        codebooks{classIndex} = lbgTrainCodebook(mfccFive, codebookSize);
        classIndex = classIndex + 1;

        % 2) The "eleven" class for speaker spk
        fileEleven = fullfile(trainElevenPath, sprintf('s%d.wav', spk));
        [mfccEleven, ~] = computeMFCC(fileEleven);
        codebooks{classIndex} = lbgTrainCodebook(mfccEleven, codebookSize);
        classIndex = classIndex + 1;
    end


    % TEST codebooks

    correctCount = 0;
    totalTests   = 2 * numSpeakers;  % each speaker => "five" test + "eleven" test

    for spk = 1:numSpeakers
        % (A) Test "five"
        testFiveFile = fullfile(testFivePath, sprintf('s%d.wav', spk));
        [testFiveMFCC, ~] = computeMFCC(testFiveFile);
        predictedClassFive = findMinDistClass(testFiveMFCC, codebooks);

        % The true class for (spk, five) is 2*(spk-1) + 1 (assuming each speaker's "five" is the odd index, "eleven" is the even index)
        trueClassFive = 2*(spk-1) + 1;
        if predictedClassFive == trueClassFive
            correctCount = correctCount + 1;
        end
        fprintf('Speaker %d "five" => recognized class %d (true %d)\n', ...
                spk, predictedClassFive, trueClassFive);

        % (B) Test "eleven"
        testElevenFile = fullfile(testElevenPath, sprintf('s%d.wav', spk));
        [testElevenMFCC, ~] = computeMFCC(testElevenFile);
        predictedClassEleven = findMinDistClass(testElevenMFCC, codebooks);

        % The true class for (spk, eleven) is 2*(spk-1) + 2
        trueClassEleven = 2*(spk-1) + 2;
        if predictedClassEleven == trueClassEleven
            correctCount = correctCount + 1;
        end
        fprintf('Speaker %d "eleven" => recognized class %d (true %d)\n', ...
                spk, predictedClassEleven, trueClassEleven);
    end

    finalAccuracy = correctCount / totalTests * 100;
    fprintf('Test10_FiveEleven: Overall accuracy (speaker + word) = %.2f%%\n', finalAccuracy);
end


%%%%%%%%%%%%%% Helper function to pick best class %%%%%%%%%%%%%%
function classID = findMinDistClass(mfccData, codebookList)
    C = length(codebookList);
    dists = zeros(C,1);
    for c = 1:C
        dists(c) = computeVQDistortion(mfccData, codebookList{c});
    end
    [~, classID] = min(dists);
end


%%%%%%%%%%%%%%% (Same subfunctions: computeMFCC, etc.) %%%%%%%%%%%%%%

function [mfccMatrix, fs] = computeMFCC(wavFile)
    [signal, fs] = audioread(wavFile);
    signal = signal(:);

    N = 256;              
    M = 100;              
    numMelFilters = 36;   
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
    Lr = log(1 + 0.5/f0) / (p + 1);

    Bv = n * (f0*(exp([0 1 p p+1]*Lr) - 1));
    b1 = floor(Bv(1))+1;
    b2 = ceil(Bv(2));
    b3 = floor(Bv(3));
    b4 = min(fn2, ceil(Bv(4))) - 1;
    if b2>b4 || b1>b3
        m = sparse(p,1+fn2);
        return;
    end

    pf = log(1 + (b1:b4)/n/f0)/Lr;
    fp = floor(pf);
    pm = pf - fp;

    idx_hi = (b2-b1+1):(b4-b1+1);
    idx_lo = 1:(b3-b1+1);

    r = [fp(idx_hi)+1, fp(idx_lo)];
    c = [b2:b4, b1:b3]+1;
    v = 2*[1 - pm(idx_hi), pm(idx_lo)];
    r(r<1)=1; r(r>p)=p;
    m = sparse(r, c, v, p, 1+fn2);
end

function codebook = lbgTrainCodebook(data, M)
    epsilon = 0.01;
    codebook = mean(data,1);
    codebook = codebook(:)';

    currentSize = 1;
    while currentSize < M
        codebookPlus  = codebook.*(1+epsilon);
        codebookMinus = codebook.*(1-epsilon);
        codebook      = [codebookPlus; codebookMinus];
        currentSize   = size(codebook,1);

        distortionOld = 1e9;
        while true
            assignments = zeros(size(data,1),1);
            for i=1:size(data,1)
                dists = sum((data(i,:) - codebook).^2,2);
                [~, idx] = min(dists);
                assignments(i) = idx;
            end
            for c = 1:currentSize
                clusterPoints = data(assignments==c,:);
                if ~isempty(clusterPoints)
                    codebook(c,:) = mean(clusterPoints,1);
                end
            end
            distortion=0;
            for i=1:size(data,1)
                distortion = distortion + sum((data(i,:)-codebook(assignments(i),:)).^2);
            end
            distortion = distortion / size(data,1);
            if abs(distortionOld - distortion)<1e-5
                break
            else
                distortionOld=distortion;
            end
        end
    end
end

function d = computeVQDistortion(mfccData, codebook)
    d=0;
    for i=1:size(mfccData,1)
        distVec = sum((mfccData(i,:) - codebook).^2,2);
        d = d + min(distVec);
    end
    d = d / size(mfccData,1);
end
