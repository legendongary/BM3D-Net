function gen_train(sigma)

data = cell(400, 1);
Mindex = mIndex(25, 180, 180);
for n=1:400
    data{n}.clean = double(imread(sprintf('./train-images/test_%03d.png', n)));
    data{n}.noisy = data{n}.clean + sigma * randn(size(data{n}.clean));
    resim = BM3D(1, data{n}.noisy, sigma);
    data{n}.index = FAST_NL_Patches(resim, 3, 12, 8, Mindex);
end
save(sprintf('./train-sigma-%02d.mat', sigma), 'data');
end