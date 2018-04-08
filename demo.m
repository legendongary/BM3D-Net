clear;clc;close all;
sigma = 15;
image = imread('im-01.jpg');
image = imresize(image, 0.5);
image = rgb2gray(image);
figure, imshow(image);
noisy = double(image) + sigma*randn(size(image));
figure, imshow(uint8(noisy));
resim = bm3d_net_res(noisy, sigma);
figure, imshow(uint8(resim))