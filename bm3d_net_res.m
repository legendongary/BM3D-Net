function resim = bm3d_net_res(noisy, sigma)

noisy = double(noisy);
image = noisy * 15 / sigma;
resss = BM3D(1, image, 15);
imsiz = size(image);
mmind = mIndex(25, imsiz(1), imsiz(2));
index = fast_nl_patches(resss, 3, 12, 8, mmind);
image = gpuArray(image);
index = gpuArray(index);

model = load('weight.mat');
weights = model.weight;

basis = load('basis.mat');
basis = basis.basis;
basis = basis(:,2:end);
basis = gpuArray(basis);
filtN = 391;
n0 = 0;
n1 = n0 + filtN^2;
n2 = n1 + filtN*63;
weights = gpuArray(weights);
c1 = weights(n0+1:n1);
c2 = weights(n1+1:n2);
c1 = reshape(c1, [], filtN);
c2 = reshape(c2, [], filtN);
nc = nnnorm(c1);
w1 = basis * nc;
w2 = c2;
w3 = w1';

means = gpuArray((-310:10:310));
pre = gpuArray(10);

x0 = image;
x0 = nnextr(x0, index);
x0 = nnconv(x0, w1, []);
x0 = nnrbfc(x0, w2, means, pre);
x0 = nnconv(x0, w3, []);
x0 = nnaggr(x0, index);

resim = image - x0;
resim = max(min(resim, 255), 0);
resim = gather(resim);
resim = resim * sigma / 15;

end