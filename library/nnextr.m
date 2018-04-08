function Y = nnextr(X, index, DzDy)

PS = 7;
WS = 25;

FS = (PS - 1) / 2;
if nargin == 2
    X = padarray(X, [FS FS], 'symmetric');
    Y = nnextrcu(X, index, [PS PS]);
elseif nargin == 3
    [IM, IN] = nnaggrcu(DzDy, index, [PS WS]);
    Y = IM .* IN;
else
    error('Invalid input');
end;