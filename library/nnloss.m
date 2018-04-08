function Y = nnloss(X, N, C, ~)
%NNLOSS: calculate the square error of restored image and original image
%   X: estimated noise
%   N: noisy image
%   C: clean image
%
if nargin ==3
    S = (X - N + C).^2;
    Y = sum(S(:));
elseif nargin == 4
    Y = 2 * (X - N + C);
else
    error('Input arguments number not proper.');
end;

end