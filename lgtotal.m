function [LOSS, GRAD] = lgtotal(weights, data)
global filtN;
basis = load('basis.mat');
basis = basis.basis;
basis = basis(:,2:end);
basis = gpuArray(basis);
n0 = 0;
n1 = n0 + filtN^2;
n2 = n1 + filtN*63;
weights = gpuArray(weights);
c1 = weights(n0+1:n1);
c2 = weights(n1+1:n2);
c1 = reshape(c1, [], filtN);
c2 = reshape(c2, [], filtN);
[nc, fs] = nnnorm(c1);
w1 = basis * nc;
w2 = c2;
w3 = w1';

TN = 400;
lam = 1;
if nargout == 1
    LOSS = 0;
    for n=1:TN
        loss = lgsingle(data{n}, w1, w2, w3);
        LOSS = LOSS + loss;
        
    end;
    REGU = lam/2 * sum(weights.^2);
    LOSS = LOSS / TN;
    LOSS = gather(LOSS) + REGU / 12;
else
    LOSS = 0;
    Dw1 = 0;
    Dw2 = 0;
    Dw3 = 0;
    for n=1:TN
        [loss, dw1, dw2, dw3] = lgsingle(data{n}, w1, w2, w3);
        LOSS = LOSS + loss;
        Dw1 = Dw1 + dw1;
        Dw2 = Dw2 + dw2;
        Dw3 = Dw3 + dw3;
    end;
    LOSS = LOSS / TN;
    Dw2 = Dw2 / TN;
    Dw1 = (Dw1 + Dw3') / TN;
    Dw1 = basis' * Dw1;
    Dw1 = nncoff(nc, Dw1, fs);
    GRAD = [Dw1(:); Dw2(:)];
    REGU = lam/2 * sum(weights.^2);
    DREG = lam * weights;
    LOSS = gather(LOSS) + gather(REGU) / 12;
    GRAD = gather(GRAD) + gather(DREG) / 12;
end;

end