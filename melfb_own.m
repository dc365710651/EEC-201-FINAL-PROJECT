function m = melfb_own(p, n, fs)
% MELFB_own Determine matrix for a mel-spaced filterbank
%
% Inputs: p = number of filters in filterbank
% n = length of fft
% fs = sample rate in Hz
%
% Outputs: x = a (sparse) matrix containing the filterbank amplitudes
% size(x) = [p, 1+floor(n/2)]
%
% Usage: Compute the mel-scale spectrum of a colum-vector s, with length n and sample rate fs:
% f = fft(s);
% m = melfb(p, n, fs);
% n2 = 1 + floor(n/2);
% z = m * abs(f(1:n2)).^2;
%
% z would contain p samples of the desired mel-scale spectrum
% To plot filterbank responses:
% plot(linspace(0, (12500/2), 129), melfb(20, 256, 12500)'),
% title('Mel-spaced filterbank'), xlabel('Frequency (Hz)');
f0 = 700 / fs; fn2 = floor(n/2);
Lr = log(1 + 0.5/f0) / (p+1);
% convert to fft bin numbers with 0 for DC term
Bv = n*(f0*(exp([0 1 p p+1]*Lr) - 1));
b1 = floor(Bv(1)) + 1; b2 = ceil(Bv(2));
b3 = floor(Bv(3)); b4 = min(fn2, ceil(Bv(4))) - 1;
pf = log(1 + (b1:b4)/n/f0) / Lr; fp = floor(pf); pm = pf - fp;
r = [fp(b2:b4) 1+fp(1:b3)];
c = [b2:b4 1:b3] + 1;
v = 2 * [1-pm(b2:b4) pm(1:b3)];
m = sparse(r, c, v, p, 1+fn2);
% To compute Euclidean distance D between two vectors or matrices X and Y:
% D = norm(X-Y,2)