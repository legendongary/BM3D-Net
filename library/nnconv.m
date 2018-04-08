function [o1, o2, o3] = nnconv(x, w, b, d)

[sx, sy, sz] = size(x);
[wx, wy] = size(w);
x = reshape(x, sx*sy, sz);

if nargout == 1
    y = x * w;
    if ~isempty(b) && nargin > 2
        y = bsxfun(@plus, y, b);
    end;
    o1 = reshape(y, sx, sy, wy);
    
elseif nargout == 2
    dy = reshape(d, sx*sy, wy);
    dx = dy * w';
    dw = x' * dy;
    o1 = reshape(dx, sx, sy, sz);
    o2 = reshape(dw, wx, wy);
    
elseif nargout == 3
    dy = reshape(d, sx*sy, wy);
    dx = dy * w';
    dw = x' * dy;
    o1 = reshape(dx, sx, sy, sz);
    o2 = reshape(dw, wx, wy);
    o3 = sum(dy);
    
else
    error('Invalid input numbers.');
    
end;


end