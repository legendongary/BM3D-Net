function [o1, o2] = nnrbfc(x, w, m, p, d)
%NNRBF: Evaluate the variable using a linear combination of radial basis functions.
% input:
%   x: variable to be evaluated size of MxNxK
%   w: weights of rbf size of PxK
%   m: means of rbf size of Px1
%   p: precision of rbf size of 1x1
%   d: grdient of loss w.r.t. rbf output size of MxNxK (only backward mode)
% output:
%   -- forward --
%   o1: evaluated variable size of MxNxK
%   -- backward --
%   o1: gradient of loss w.r.t. variable x size of MxNxK
%   o2: gradient of loss w.r.t. weights w size of PxK

[sx, sy, sz] = size(x);

if nargout == 1
    x = reshape(x, sx*sy, sz);
    y = x;
    for n=1:sz
        y(:, n) = nngkrcu(x(:, n), m, p) * w(:, n);
    end;
    o1 = reshape(y, sx, sy, sz);
    
elseif nargout == 2
    x = reshape(x, sx*sy, sz);
    d = reshape(d, sx*sy, sz);
    z = x;
    g = w;
    for n=1:sz
        y = nngkrcu(x(:, n), m, p);
        z(:, n) = nngkrcu(x(:, n), y, m, p) * w(:, n) .* d(:, n);
        g(:, n) = y' * d(:, n);
    end;
    o1 = reshape(z, sx, sy, sz);
    o2 = g;
    
else
    error('Invalid input numbers.');
end;
end

