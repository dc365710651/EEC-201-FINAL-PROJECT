function codebook = vq_lbg(y, M, epsilon, threshold, max_iter)
% y: data vector
% M: codebook vectors

% Initialize with a single-vector codebook
codebook = mean(y, 1);

% Iteration 2: repeat steps 2, 3 and 4 until a codebook size of M is designed.
while size(codebook, 1) < M
    new_codebook = [];
    % Double the size of the codebook by splitting each current codebook yn according to the rule
    for i = 1:size(codebook, 1)
        y_n = codebook(i,:);
        new_codebook = [new_codebook; y_n*(1+epsilon); y_n*(1-epsilon)];
    end
    % Iteration 1: repeat steps 3 and 4 until the average distance falls below a preset threshold
    prev_distortion = [];
    for iter = 1:max_iter
        % Nearest-Neighbor Search
        % Euclidean distance
        distance = disteu(new_codebook', y')';
        [min_distance, index] = min(distance, [], 2);
        distortion = mean(min_distance.^2);

        % Convergence condition
        if abs(prev_distortion - distortion) / distortion < threshold
            break;
        end
        prev_distortion = distortion;

        % Centroid update
        for j = 1:size(new_codebook, 1)
            if any(index == j)
                new_codebook(j, :) = mean(y(index == j, :), 1);
            end
        end
    end
% Update the codebook
codebook = new_codebook;
end
