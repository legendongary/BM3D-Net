function [loss, dw1, dw2, dw3] = lgsingle(data, w1, w2, w3)

clean = data.clean;
noisy = data.noisy;
index = data.index;

means = gpuArray((-310:10:310));
pre   = gpuArray(10);

x0 = noisy;
x1 = nnextr(x0, index);
x2 = nnconv(x1, w1, []);
x3 = nnrbfc(x2, w2, means, pre);
x4 = nnconv(x3, w3, []);
[x5, in] = nnaggr(x4, index);
x6 = nnloss(x5, noisy, clean);
if nargout == 1
    loss = x6;
else
    loss = x6;
    dx6 = 1;
    dx5 = nnloss(x5, noisy, clean, dx6);
    dx4 = nnaggr(x4, index, dx5, in);
    [dx3, dw3] = nnconv(x3, w3, [], dx4);
    [dx2, dw2] = nnrbfc(x2, w2, means, pre, dx3);
    [ ~ , dw1] = nnconv(x1, w1, [], dx2);
end

end

