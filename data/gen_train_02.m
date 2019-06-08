clear;clc;

sigma = 100;

n = 1;
data = cell(400, 1);
Mindex = mIndex(25, 180, 180);
while n <= 400
    data{n}.clean = double(imread(sprintf('./train-images/test_%03d.png', n)));
    try
        data{n}.noisy = data{n}.clean + sigma * randn(size(data{n}.clean));
        resim = BM3D(1, data{n}.noisy, sigma);
        data{n}.index = fast_nl_patches(resim, 3, 12, 8, Mindex);
    catch
        fprintf('image %03d, error and retry\n', n);
    end
    fprintf('image %03d\n', n);
    n = n + 1;
end

save(sprintf('./train-sigma-%02d.mat', sigma), 'data');