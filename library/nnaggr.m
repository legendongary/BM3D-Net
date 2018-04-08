function [Y, I] = nnaggr(patch, index, DzDy, in)
%NNAGGR: aggragate image patches into one image
% input:
%   patch: size ISxISx(PS^2*GN)
%   index: size ISxISxGNx2
%   DzDy: backward mode, gradients transfered in, size ISxISx(PS^2*GN)
%   in: I forget what it is T^T
% output:
%   Y: forward mode, aggragated image; backward mode, gradients w.r.t. input x
%   I: forward mode, guess what it is ^_^

PS = 7;
WS = 25;
FS = (PS - 1)/2;
if nargin == 2
    [Y, I] = nnaggrcu(patch, index, [PS WS]);
elseif nargin == 4
    
    weiGrad = DzDy./in;
    weiGrad = padarray(weiGrad, [FS FS], 'symmetric');
    gradPatch = nnextrcu(weiGrad, index, [PS PS]);
    Y = gradPatch;
    I = [];
else
    error('Input arguments number not proper.');
end;
end